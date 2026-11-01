
from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from gpt0.checkpoints import load_checkpoint
from gpt0.arithmetic_dataset import (
    ArithmeticDataset,
    arithmetic_collate_fn,
)
from gpt0.arithmetic_model import ArithmeticGPT


ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr_lambda(
    warmup_iters: int,
    max_iters: int,
    min_lr_ratio: float,
):
    def schedule(step: int) -> float:
        if warmup_iters > 0 and step < warmup_iters:
            return max(step, 1) / warmup_iters

        progress = (
            (step - warmup_iters)
            / max(1, max_iters - warmup_iters)
        )
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return schedule


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        x = batch["input_ids"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)

        valid_tokens = (y != -100).sum().item()

        if valid_tokens == 0:
            continue

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            _, loss = model(x, y)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss.")

        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

    if total_tokens == 0:
        raise RuntimeError("Validation contains no supervised tokens.")

    model.train()
    return total_loss / total_tokens


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    step,
    best_val_loss,
    config,
    train_generator,
):
    checkpoint = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "config": config,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "sampler_rng": train_generator.get_state(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng"] = torch.cuda.get_rng_state_all()

    temporary_path = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def main():
    parser = argparse.ArgumentParser(
        description="Train GPT-0 Math on arithmetic."
    )

    parser.add_argument(
        "--config",
        default="configs/arithmetic.yaml",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/arithmetic_v1",
    )
    parser.add_argument(
        "--resume",
        default=None,
    )

    args = parser.parse_args()

    config_path = ROOT / args.config

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]

    seed = train_cfg["seed"]
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_amp = device.type == "cuda"

    dataset_dir = ROOT / data_cfg["dataset_dir"]

    train_dataset = ArithmeticDataset(
        data_cfg["train_split"],
        data_dir=dataset_dir,
    )

    val_dataset = ArithmeticDataset(
        data_cfg["validation_split"],
        data_dir=dataset_dir,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=use_amp,
        collate_fn=arithmetic_collate_fn,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=use_amp,
        collate_fn=arithmetic_collate_fn,
    )

    model = ArithmeticGPT(
        block_size=model_cfg["block_size"],
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        n_embd=model_cfg["n_embd"],
        dropout=model_cfg["dropout"],
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

    warmup_iters = min(
        train_cfg["warmup_iters"],
        max_iters - 1,
    )

    if max_iters <= 0:
        raise ValueError("max_iters must be positive.")

    min_lr_ratio = (
        train_cfg["min_lr"] / train_cfg["learning_rate"]
    )

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

    start_step = 0
    best_val_loss = float("inf")

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        checkpoint = load_checkpoint(
            ROOT / args.resume,
            map_location=device,
            )

        if checkpoint["config"]["model"] != model_cfg:
            raise ValueError("Model configuration mismatch.")

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])

        start_step = checkpoint["step"]
        best_val_loss = checkpoint["best_val_loss"]

        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        generator.set_state(checkpoint["sampler_rng"])

        if "cuda_rng" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])

        print(f"Resuming from step {start_step}")

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = (
        ROOT / "results" / "arithmetic"
        / f"training_{timestamp}.jsonl"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nGPT-0 Math — Arithmetic Training")
    print("--------------------------------")
    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"Parameters: {parameters:,}")
    print(f"Train examples: {len(train_dataset):,}")
    print(f"Validation examples: {len(val_dataset):,}")
    print(f"Batch size: {train_cfg['batch_size']}")
    print(f"Iterations: {max_iters}")
    print(f"Output: {output_dir}")

    if use_amp:
        torch.cuda.reset_peak_memory_stats()

    train_iterator = iter(train_loader)
    start_time = time.perf_counter()

    for step in range(start_step, max_iters):
        model.train()

        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        x = batch["input_ids"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)

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

        should_evaluate = (
            completed_step == 1
            or completed_step % train_cfg["eval_interval"] == 0
            or completed_step == max_iters
        )

        if not should_evaluate:
            continue

        val_loss = evaluate(
            model,
            val_loader,
            device,
            use_amp,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - start_time

        metrics = {
            "step": completed_step,
            "train_loss": loss.item(),
            "validation_loss": val_loss,
            "learning_rate": current_lr,
            "elapsed_seconds": elapsed,
        }

        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(metrics) + "\n")

        print(
            f"\nStep {completed_step}/{max_iters}"
            f"\n  Train loss: {loss.item():.5f}"
            f"\n  Val loss:   {val_loss:.5f}"
            f"\n  LR:         {current_lr:.7f}"
            f"\n  Elapsed:    {elapsed:.1f}s"
        )

        if use_amp:
            memory = (
                torch.cuda.max_memory_allocated() / 1024**3
            )
            print(f"  Peak VRAM:  {memory:.2f} GB")

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
                config,
                generator,
            )

            print("  Best checkpoint saved.")

        save_checkpoint(
            output_dir / "latest.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            completed_step,
            best_val_loss,
            config,
            generator,
        )

        print("  Latest checkpoint saved.")

    print("\nTraining completed.")
    print(f"Best validation loss: {best_val_loss:.5f}")
    print(f"Metrics: {metrics_path}")
    print(f"Checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
