
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from functools import partial

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpt0.model import GPT0
from gpt0.checkpoints import load_checkpoint
from gpt0.dataset import (
    MathSequenceDataset,
    create_train_loader,
    collate_math_sequences,
)


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# Learning rate: warmup + cosine decay
# --------------------------------------------------

def get_lr_lambda(warmup_iters, max_iters, min_ratio):

    def lr_lambda(step):
        if step < warmup_iters:
            return (step + 1) / max(1, warmup_iters)

        progress = (step - warmup_iters) / max(
            1, max_iters - warmup_iters
        )
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return min_ratio + (1.0 - min_ratio) * cosine

    return lr_lambda


# --------------------------------------------------
# Validation
# --------------------------------------------------

@torch.inference_mode()
def evaluate(model, loader, device, use_amp):

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            _, loss = model(x, y)

        valid_tokens = (y != -100).sum().item()

        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

    if total_tokens == 0:
        raise RuntimeError("Validation contains no valid targets")

    return total_loss / total_tokens


# --------------------------------------------------
# Checkpoints
# --------------------------------------------------

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    step,
    best_val_loss,
    cfg,
    sampler,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "config": cfg,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "sampler_rng": sampler.generator.get_state(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng"] = torch.cuda.get_rng_state_all()

    torch.save(checkpoint, path)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/small.yaml",
    )

    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint to resume",
    )

    parser.add_argument(
        "--max-iters",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        default="checkpoints",
        help="Directory for best.pt and latest.pt (default: checkpoints)",
    )

    args = parser.parse_args()

    cfg = yaml.safe_load(
        (ROOT / args.config).read_text(encoding="utf-8")
    )

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    seed = train_cfg["seed"]
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_amp = device.type == "cuda"

    # --------------------------------------------------
    # Load tokenizer and verify vocabulary
    # --------------------------------------------------

    tokenizer = json.loads(
        (ROOT / data_cfg["tokenizer_path"]).read_text(
            encoding="utf-8"
        )
    )

    vocab_size = len(tokenizer["model"]["vocab"])

    if vocab_size != model_cfg["vocab_size"]:
        raise ValueError(
            f"Config vocabulary {model_cfg['vocab_size']} "
            f"does not match tokenizer vocabulary {vocab_size}"
        )

    # --------------------------------------------------
    # Load datasets
    # --------------------------------------------------

    train_path = ROOT / data_cfg["train_sequences_path"]
    val_path = ROOT / data_cfg["val_sequences_path"]

    train_loader = create_train_loader(
        train_path,
        batch_size=train_cfg["batch_size"],
        theory_probability=data_cfg["theory_probability"],
        pad_token_id=0,
        seed=seed,
    )

    val_dataset = MathSequenceDataset(val_path)

    def make_eval_loader(indices):
        subset = Subset(val_dataset, indices)

        return DataLoader(
            subset,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=use_amp,
            collate_fn=partial(
                collate_math_sequences,
                pad_token_id=0,
            ),
        )

    all_indices = list(range(len(val_dataset)))

    theory_indices = [
        i for i, record in enumerate(val_dataset.records)
        if record["type"] == "theory"
    ]

    gsm8k_indices = [
        i for i, record in enumerate(val_dataset.records)
        if record["type"] == "gsm8k"
    ]

    val_all = make_eval_loader(all_indices)
    val_theory = make_eval_loader(theory_indices)
    val_gsm8k = make_eval_loader(gsm8k_indices)

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------

    model = GPT0(
        vocab_size=vocab_size,
        block_size=model_cfg["block_size"],
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        n_embd=model_cfg["n_embd"],
        dropout=model_cfg["dropout"],
        pad_token_id=0,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    max_iters = (
        args.max_iters
        if args.max_iters is not None
        else train_cfg["max_iters"]
    )

    warmup_iters = train_cfg["warmup_iters"]

    warmup_iters = min(
    warmup_iters,
    max_iters - 1,
)

    if max_iters <= 0:
        raise ValueError("max_iters must be positive")

    if not 0 <= warmup_iters < max_iters:
        raise ValueError("warmup_iters must be smaller than max_iters")

    min_lr_ratio = (
        train_cfg["min_lr"] / train_cfg["learning_rate"]
    )

    if not 0 < min_lr_ratio <= 1:
        raise ValueError("min_lr must be between 0 and learning_rate")

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        get_lr_lambda(
            warmup_iters,
            max_iters,
            min_lr_ratio,
        ),
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    # --------------------------------------------------
    # Resume
    # --------------------------------------------------

    start_step = 0
    best_val_loss = float("inf")

    if args.resume:
        checkpoint = load_checkpoint(
            ROOT / args.resume,
            map_location=device,
        )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])

        start_step = checkpoint["step"]
        best_val_loss = checkpoint["best_val_loss"]

        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        train_loader.sampler.generator.set_state(
            checkpoint["sampler_rng"]
        )

        if "cuda_rng" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])

        print(f"Resuming from step {start_step}")

    # --------------------------------------------------
    # Information
    # --------------------------------------------------

    parameters = sum(
        p.numel() for p in model.parameters()
    )

    print("\nGPT-0 Math Training")
    print("-------------------")
    print(f"Device: {device}")
    print(f"AMP enabled: {use_amp}")
    print(f"Parameters: {parameters:,}")
    print(f"Vocabulary: {vocab_size:,}")
    print(f"Context: {model_cfg['block_size']}")
    print(f"Batch size: {train_cfg['batch_size']}")
    print(f"Training examples: {len(train_loader.dataset):,}")
    print(f"Validation examples: {len(val_dataset):,}")
    print(f"Maximum iterations: {max_iters}")

    if use_amp:
        torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    train_iterator = iter(train_loader)

    checkpoint_dir = ROOT / args.output_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()

    for step in range(start_step, max_iters):

        model.train()

        try:
            x, y = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            x, y = next(train_iterator)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            _, loss = model(x, y)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at step {step + 1}"
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        completed_step = step + 1

        # --------------------------------------------------
        # Evaluation
        # --------------------------------------------------

        should_evaluate = (
            completed_step == 1
            or completed_step % train_cfg["eval_interval"] == 0
            or completed_step == max_iters
        )

        if should_evaluate:

            val_loss = evaluate(
                model, val_all, device, use_amp
            )

            theory_loss = evaluate(
                model, val_theory, device, use_amp
            )

            gsm8k_loss = evaluate(
                model, val_gsm8k, device, use_amp
            )

            current_lr = optimizer.param_groups[0]["lr"]

            elapsed = time.perf_counter() - start_time

            print(
                f"\nStep {completed_step}/{max_iters}"
                f"\n  Train loss:   {loss.item():.4f}"
                f"\n  Val loss:     {val_loss:.4f}"
                f"\n  Theory loss:  {theory_loss:.4f}"
                f"\n  GSM8K loss:   {gsm8k_loss:.4f}"
                f"\n  Learning rate:{current_lr:.7f}"
                f"\n  Elapsed:      {elapsed:.1f}s"
            )

            if use_amp:
                memory = (
                    torch.cuda.max_memory_allocated()
                    / 1024**3
                )

                print(f"  Peak VRAM:    {memory:.2f} GB")

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    completed_step,
                    best_val_loss,
                    cfg,
                    train_loader.sampler,
                )

                print("  Best checkpoint updated.")

            save_checkpoint(
                checkpoint_dir / "latest.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                completed_step,
                best_val_loss,
                cfg,
                train_loader.sampler,
            )

            print("  Latest checkpoint saved.")

    print("\nTraining completed.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {checkpoint_dir}")


if __name__ == "__main__":
    main()
