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

from gpt0.checkpoints import load_checkpoint
from gpt0.model import GPT0
from gpt0.dataset import (
    MathSequenceDataset,
    collate_math_sequences,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr_lambda(warmup, max_iters, min_ratio):
    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)

        progress = (step - warmup) / max(1, max_iters - warmup)
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return min_ratio + (1.0 - min_ratio) * cosine

    return lr_lambda


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

        valid = (y != -100).sum().item()

        total_loss += loss.item() * valid
        total_tokens += valid

    if total_tokens == 0:
        raise RuntimeError("Validation has no valid targets")

    return total_loss / total_tokens


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    step,
    best_val_loss,
    cfg,
    sampler_indices,
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
        "sampler_indices": sampler_indices,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng"] = torch.cuda.get_rng_state_all()

    torch.save(checkpoint, path)


def build_coverage_indices(records, seed):
    theory = [
        i for i, r in enumerate(records)
        if r["type"] == "theory"
    ]

    gsm8k = [
        i for i, r in enumerate(records)
        if r["type"] == "gsm8k"
    ]

    if not theory or not gsm8k:
        raise ValueError("Both theory and GSM8K examples are required")

    theory_draws = round(len(gsm8k) * 0.10 / 0.90)

    rng = random.Random(seed)

    gsm_order = gsm8k.copy()
    rng.shuffle(gsm_order)

    theory_order = [
        rng.choice(theory)
        for _ in range(theory_draws)
    ]

    combined = gsm_order + theory_order
    rng.shuffle(combined)

    return combined, gsm8k, theory, theory_draws


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/small.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best.pt",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and sampling without training",
    )

    parser.add_argument(
        "--run",
        action="store_true",
        help="Explicitly authorize training",
    )

    args = parser.parse_args()

    if args.dry_run == args.run:
        parser.error("Specify exactly one of --dry-run or --run")

    cfg = yaml.safe_load(
        (ROOT / args.config).read_text(encoding="utf-8")
    )

    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    seed = train_cfg["seed"]
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_amp = device.type == "cuda"

    tokenizer = json.loads(
        (ROOT / data_cfg["tokenizer_path"]).read_text(
            encoding="utf-8"
        )
    )

    vocab_size = len(tokenizer["model"]["vocab"])

    if vocab_size != model_cfg["vocab_size"]:
        raise ValueError("Tokenizer vocabulary mismatch")

    train_path = ROOT / data_cfg["train_sequences_path"]
    val_path = ROOT / data_cfg["val_sequences_path"]

    train_dataset = MathSequenceDataset(train_path)

    sampler_indices, gsm_indices, theory_indices, theory_draws = (
        build_coverage_indices(train_dataset.records, seed)
    )

    batch_size = train_cfg["batch_size"]

    if len(sampler_indices) % batch_size:
        raise ValueError("Sampler length must be divisible by batch size")

    max_iters = len(sampler_indices) // batch_size

    if max_iters != 3736:
        raise ValueError(
            f"Expected 3736 iterations, found {max_iters}"
        )

    val_dataset = MathSequenceDataset(val_path)

    def make_eval_loader(indices):
        return DataLoader(
            Subset(val_dataset, indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=use_amp,
            collate_fn=partial(
                collate_math_sequences,
                pad_token_id=0,
            ),
        )

    val_all = make_eval_loader(list(range(len(val_dataset))))

    val_theory_indices = [
        i for i, r in enumerate(val_dataset.records)
        if r["type"] == "theory"
    ]

    val_gsm_indices = [
        i for i, r in enumerate(val_dataset.records)
        if r["type"] == "gsm8k"
    ]

    val_theory = make_eval_loader(val_theory_indices)
    val_gsm8k = make_eval_loader(val_gsm_indices)

    checkpoint_path = ROOT / args.checkpoint

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = load_checkpoint(
        checkpoint_path,
        map_location=device,
    )

    if checkpoint["config"]["model"] != model_cfg:
        raise ValueError("Checkpoint model configuration mismatch")

    if checkpoint["step"] != 2000:
        raise ValueError(
            f"Expected original checkpoint at step 2000, "
            f"found {checkpoint['step']}"
        )

    print("\nGPT-0 Math — Coverage Training")
    print("--------------------------------")
    print("Device:", device)
    print("AMP:", use_amp)
    print("Initial checkpoint:", checkpoint_path)
    print("Initial checkpoint step:", checkpoint["step"])
    print("GSM8K:", len(gsm_indices))
    print("Theory documents:", len(theory_indices))
    print("Theory exposures:", theory_draws)
    print("Total samples:", len(sampler_indices))
    print("Batch size:", batch_size)
    print("Training updates:", max_iters)
    print("GSM8K coverage: 100%")
    print("GSM8K repetitions: 0")
    print("Validation examples:", len(val_dataset))

    if args.dry_run:
        print("\nDRY RUN PASSED.")
        print("No training performed.")
        print("No checkpoint written.")
        return

    output_dir = ROOT / "checkpoints" / "coverage"

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"{output_dir} is not empty. "
            "Refusing to overwrite existing experiment."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    model = GPT0(
        vocab_size=vocab_size,
        block_size=model_cfg["block_size"],
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        n_embd=model_cfg["n_embd"],
        dropout=model_cfg["dropout"],
        pad_token_id=0,
    ).to(device)

    model.load_state_dict(checkpoint["model"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-5,
        weight_decay=0.01,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        get_lr_lambda(
            warmup=100,
            max_iters=max_iters,
            min_ratio=0.1,
        ),
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler_indices,
        collate_fn=partial(
            collate_math_sequences,
            pad_token_id=0,
        ),
        num_workers=0,
        pin_memory=use_amp,
    )

    best_val_loss = float("inf")
    start_time = time.perf_counter()

    if use_amp:
        torch.cuda.reset_peak_memory_stats()

    for step, (x, y) in enumerate(train_loader):
        model.train()

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
            raise RuntimeError(f"Non-finite loss at step {step + 1}")

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

        should_evaluate = (
            completed_step == 1
            or completed_step % 250 == 0
            or completed_step == max_iters
        )

        if not should_evaluate:
            continue

        val_loss = evaluate(model, val_all, device, use_amp)
        theory_loss = evaluate(model, val_theory, device, use_amp)
        gsm_loss = evaluate(model, val_gsm8k, device, use_amp)

        elapsed = time.perf_counter() - start_time

        print(
            f"\nStep {completed_step}/{max_iters}"
            f"\n  Train loss:  {loss.item():.4f}"
            f"\n  Val loss:    {val_loss:.4f}"
            f"\n  Theory loss: {theory_loss:.4f}"
            f"\n  GSM8K loss:  {gsm_loss:.4f}"
            f"\n  LR:          {optimizer.param_groups[0]['lr']:.8f}"
            f"\n  Elapsed:     {elapsed:.1f}s"
        )

        if use_amp:
            memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  Peak VRAM:   {memory:.2f} GB")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                completed_step,
                best_val_loss,
                cfg,
                sampler_indices,
            )

            print("  New best checkpoint saved.")

        save_checkpoint(
            output_dir / "latest.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            completed_step,
            best_val_loss,
            cfg,
            sampler_indices,
        )

        print("  Latest checkpoint saved.")

    print("\nCoverage training completed.")
    print("Best validation loss:", best_val_loss)
    print("Output:", output_dir)


if __name__ == "__main__":
    main()
