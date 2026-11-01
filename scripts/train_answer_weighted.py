
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpt0.model import GPT0
from gpt0.dataset import MathSequenceDataset


# ============================================================
# SPECIAL TOKENS
# ============================================================

PAD = 0
BOS = 2
EOS = 3

PROBLEM = 4
SOLUTION = 5
ANSWER = 6


# ============================================================
# REGIONS
# ============================================================

OTHER = 0
PROMPT = 1
REASONING = 2
ANSWER_REGION = 3
EOS_REGION = 4

REGION_NAMES = {
    OTHER: "other",
    PROMPT: "prompt",
    REASONING: "reasoning",
    ANSWER_REGION: "answer",
    EOS_REGION: "eos",
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# COLLATOR WITH REGION INFORMATION
# ============================================================

def collate_with_regions(batch, pad_token_id=PAD):

    inputs = []
    targets = []
    region_sequences = []

    for record in batch:

        sequence = record["tokens"]

        if len(sequence) < 2:
            raise ValueError("Sequence too short")

        input_ids = sequence[:-1]
        target_ids = sequence[1:]

        inputs.append(
            torch.tensor(input_ids, dtype=torch.long)
        )

        targets.append(
            torch.tensor(target_ids, dtype=torch.long)
        )

        # Classify each target according to the active region.
        state = OTHER
        regions = []

        for token_id in target_ids:

            # The target is classified using the region
            # active immediately before this token.
            if token_id == EOS:
                region = EOS_REGION

            elif state == PROMPT:
                region = PROMPT

            elif state == REASONING:
                region = REASONING

            elif state == ANSWER_REGION:
                region = ANSWER_REGION

            else:
                region = OTHER

            regions.append(region)

            # Special markers change the state for
            # the following target token.
            if token_id == PROBLEM:
                state = PROMPT

            elif token_id == SOLUTION:
                state = REASONING

            elif token_id == ANSWER:
                state = ANSWER_REGION

            elif token_id == EOS:
                state = OTHER

        region_sequences.append(
            torch.tensor(regions, dtype=torch.long)
        )

    max_length = max(len(x) for x in inputs)
    batch_size = len(batch)

    x = torch.full(
        (batch_size, max_length),
        pad_token_id,
        dtype=torch.long,
    )

    y = torch.full(
        (batch_size, max_length),
        -100,
        dtype=torch.long,
    )

    regions = torch.full(
        (batch_size, max_length),
        -1,
        dtype=torch.long,
    )

    for i, (input_ids, target_ids, region_ids) in enumerate(
        zip(inputs, targets, region_sequences)
    ):

        length = len(input_ids)

        x[i, :length] = input_ids
        y[i, :length] = target_ids
        regions[i, :length] = region_ids

    return x, y, regions


# ============================================================
# DATA LOADERS
# ============================================================

def make_train_loader(
    dataset,
    batch_size,
    theory_probability,
    seed,
):

    if not 0.0 <= theory_probability <= 1.0:
        raise ValueError("Invalid theory_probability")

    theory_indices = [
        i
        for i, record in enumerate(dataset.records)
        if record["type"] == "theory"
    ]

    gsm8k_indices = [
        i
        for i, record in enumerate(dataset.records)
        if record["type"] == "gsm8k"
    ]

    if not theory_indices or not gsm8k_indices:
        raise ValueError(
            "Both theory and GSM8K records are required"
        )

    weights = []

    for record in dataset.records:

        if record["type"] == "theory":

            weight = (
                theory_probability / len(theory_indices)
            )

        elif record["type"] == "gsm8k":

            weight = (
                (1.0 - theory_probability)
                / len(gsm8k_indices)
            )

        else:

            raise ValueError(
                f"Unknown record type: {record['type']}"
            )

        weights.append(weight)

    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_with_regions,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return loader, sampler


def make_validation_loader(dataset, batch_size):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_with_regions,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# ANSWER-WEIGHTED CROSS ENTROPY
# ============================================================

def compute_weighted_loss(
    logits,
    targets,
    regions,
    answer_weight,
):

    per_token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(targets)

    weights = torch.ones_like(per_token_loss)

    # Only numerical answer tokens receive additional weight.
    weights[regions == ANSWER_REGION] = answer_weight

    # Padding is never included.
    weights[targets == -100] = 0.0

    denominator = weights.sum()

    if denominator.item() == 0:
        raise RuntimeError(
            "Batch contains no valid targets"
        )

    return (
        (per_token_loss * weights).sum()
        / denominator
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.inference_mode()
def evaluate(model, loader, device, use_amp):

    model.eval()

    totals = {
        "loss_sum": 0.0,
        "count": 0,
        "correct": {r: 0 for r in range(5)},
        "loss": {r: 0.0 for r in range(5)},
        "count_region": {r: 0 for r in range(5)},
    }

    for x, y, regions in loader:

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        regions = regions.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):

            logits, _ = model(x)

        per_token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(y)

        predictions = logits.argmax(dim=-1)

        valid = y != -100

        totals["loss_sum"] += (
            per_token_loss[valid].sum().item()
        )

        totals["count"] += valid.sum().item()

        for region in range(5):

            mask = valid & (regions == region)

            count = mask.sum().item()

            if count == 0:
                continue

            totals["loss"][region] += (
                per_token_loss[mask].sum().item()
            )

            totals["correct"][region] += (
                (predictions[mask] == y[mask]).sum().item()
            )

            totals["count_region"][region] += count

    if totals["count"] == 0:
        raise RuntimeError(
            "Validation contains no valid targets"
        )

    result = {
        "loss": totals["loss_sum"] / totals["count"]
    }

    for region, name in REGION_NAMES.items():

        count = totals["count_region"][region]

        if count == 0:
            continue

        result[name + "_loss"] = (
            totals["loss"][region] / count
        )

        result[name + "_accuracy"] = (
            totals["correct"][region] / count
        )

        result[name + "_tokens"] = count

    return result


# ============================================================
# CHECKPOINTS
# ============================================================

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

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "config": cfg,
        "answer_weight": cfg["experiment"]["answer_weight"],
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "sampler_rng": sampler.generator.get_state(),
    }

    if torch.cuda.is_available():

        checkpoint["cuda_rng"] = (
            torch.cuda.get_rng_state_all()
        )

    torch.save(checkpoint, path)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

def get_lr_lambda(
    warmup_iters,
    max_iters,
    min_ratio,
):

    def lr_lambda(step):

        if step < warmup_iters:

            return (
                (step + 1)
                / max(1, warmup_iters)
            )

        progress = (
            (step - warmup_iters)
            / max(1, max_iters - warmup_iters)
        )

        progress = min(
            max(progress, 0.0),
            1.0,
        )

        cosine = 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        return (
            min_ratio
            + (1.0 - min_ratio) * cosine
        )

    return lr_lambda


# ============================================================
# TOKENIZER VALIDATION
# ============================================================

def validate_tokenizer(tokenizer_path, expected_vocab_size):

    tokenizer = json.loads(
        tokenizer_path.read_text(encoding="utf-8")
    )

    vocabulary = tokenizer["model"]["vocab"]

    if len(vocabulary) != expected_vocab_size:

        raise ValueError(
            "Tokenizer vocabulary size does not match config"
        )

    expected_special_tokens = {
        "[PAD]": PAD,
        "[BOS]": BOS,
        "[EOS]": EOS,
        "[PROBLEM]": PROBLEM,
        "[SOLUTION]": SOLUTION,
        "[ANSWER]": ANSWER,
    }

    for token, expected_id in expected_special_tokens.items():

        actual_id = vocabulary.get(token)

        if actual_id != expected_id:

            raise ValueError(
                f"Invalid ID for {token}: "
                f"expected {expected_id}, got {actual_id}"
            )

    return len(vocabulary)


# ============================================================
# DRY RUN
# ============================================================

def run_dry_run(
    model,
    train_dataset,
    device,
    use_amp,
    answer_weight,
):

    # Select real GSM8K records explicitly.
    # Never depend on the random sampler for this test.

    gsm8k_records = [
        record
        for record in train_dataset.records
        if record["type"] == "gsm8k"
    ]

    if len(gsm8k_records) < 2:

        raise RuntimeError(
            "Not enough GSM8K records for dry-run"
        )

    batch = gsm8k_records[:2]

    x, y, regions = collate_with_regions(batch)

    x = x.to(device)
    y = y.to(device)
    regions = regions.to(device)

    model.train()

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=use_amp,
    ):

        logits, _ = model(x)

        standard_loss = compute_weighted_loss(
            logits,
            y,
            regions,
            1.0,
        )

        weighted_loss = compute_weighted_loss(
            logits,
            y,
            regions,
            answer_weight,
        )

    if not torch.isfinite(standard_loss):
        raise RuntimeError("Standard loss is not finite")

    if not torch.isfinite(weighted_loss):
        raise RuntimeError("Weighted loss is not finite")

    answer_count = (
        (regions == ANSWER_REGION).sum().item()
    )

    prompt_count = (
        (regions == PROMPT).sum().item()
    )

    reasoning_count = (
        (regions == REASONING).sum().item()
    )

    if answer_count == 0:
        raise RuntimeError(
            "ANSWER region was not detected"
        )

    if prompt_count == 0:
        raise RuntimeError(
            "PROMPT region was not detected"
        )

    if reasoning_count == 0:
        raise RuntimeError(
            "REASONING region was not detected"
        )

    # Check the actual gradient computation.
    weighted_loss.backward()

    gradient_count = 0

    for parameter in model.parameters():

        if parameter.grad is not None:

            if not torch.isfinite(parameter.grad).all():

                raise RuntimeError(
                    "Non-finite gradients detected"
                )

            gradient_count += 1

    if gradient_count == 0:
        raise RuntimeError(
            "No gradients were produced"
        )

    model.zero_grad(set_to_none=True)

    print("\n" + "=" * 60)
    print("DRY-RUN PASSED")
    print("=" * 60)

    print(f"Standard loss:          {standard_loss.item():.6f}")
    print(f"Weighted loss:          {weighted_loss.item():.6f}")

    print(
        f"Loss difference:        "
        f"{weighted_loss.item() - standard_loss.item():.6f}"
    )

    print(f"Answer weight:          {answer_weight}x")

    print(f"Prompt target tokens:   {prompt_count}")
    print(f"Reasoning target tokens:{reasoning_count:>7}")
    print(f"Answer target tokens:   {answer_count}")

    print(f"Parameters with grads:  {gradient_count}")

    print("Backward pass:          OK")
    print("Optimizer step:         NOT EXECUTED")
    print("Checkpoint saved:       NO")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="GPT-0 Math Answer-Weighted Training"
    )

    parser.add_argument(
        "--config",
        default="configs/small.yaml",
    )

    parser.add_argument(
        "--answer-weight",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--max-iters",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    if (
        not math.isfinite(args.answer_weight)
        or args.answer_weight < 1.0
    ):

        raise ValueError(
            "--answer-weight must be finite and >= 1"
        )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    cfg = yaml.safe_load(
        (ROOT / args.config).read_text(
            encoding="utf-8"
        )
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

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    vocab_size = validate_tokenizer(
        ROOT / data_cfg["tokenizer_path"],
        model_cfg["vocab_size"],
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = MathSequenceDataset(
        ROOT / data_cfg["train_sequences_path"]
    )

    val_dataset = MathSequenceDataset(
        ROOT / data_cfg["val_sequences_path"]
    )

    train_loader, sampler = make_train_loader(
        train_dataset,
        train_cfg["batch_size"],
        data_cfg["theory_probability"],
        seed,
    )

    val_loader = make_validation_loader(
        val_dataset,
        train_cfg["batch_size"],
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = GPT0(
        vocab_size=vocab_size,
        block_size=model_cfg["block_size"],
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        n_embd=model_cfg["n_embd"],
        dropout=model_cfg["dropout"],
        pad_token_id=PAD,
    ).to(device)

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    max_iters = (
        args.max_iters
        if args.max_iters is not None
        else train_cfg["max_iters"]
    )

    if max_iters <= 0:
        raise ValueError("max-iters must be positive")

    warmup_iters = min(
        train_cfg["warmup_iters"],
        max_iters - 1,
    )

    min_lr_ratio = (
        train_cfg["min_lr"]
        / train_cfg["learning_rate"]
    )

    if not 0 < min_lr_ratio <= 1:
        raise ValueError("Invalid minimum learning rate ratio")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
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

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    print("\nGPT-0 Math — Answer-Weighted Experiment")
    print("----------------------------------------")

    print(f"Device:              {device}")
    print(f"AMP enabled:         {use_amp}")
    print(f"Parameters:          {parameters:,}")
    print(f"Vocabulary:          {vocab_size:,}")
    print(f"Batch size:          {train_cfg['batch_size']}")
    print(f"Answer weight:       {args.answer_weight}x")
    print(f"Training records:    {len(train_dataset)}")
    print(f"Validation records:  {len(val_dataset)}")
    print(f"Maximum iterations:  {max_iters}")

    # --------------------------------------------------------
    # Dry run
    # --------------------------------------------------------

    if args.dry_run:

        run_dry_run(
            model,
            train_dataset,
            device,
            use_amp,
            args.answer_weight,
        )

        return

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = (
        ROOT
        / "checkpoints"
        / "answer_weighted"
        / f"weight_{args.answer_weight:g}x"
    )

    if (
        (output_dir / "best.pt").exists()
        or (output_dir / "latest.pt").exists()
    ):

        raise FileExistsError(
            f"Checkpoints already exist in {output_dir}. "
            "Use a separate experiment directory before rerunning."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg["experiment"] = {
        "name": "answer_weighted",
        "answer_weight": args.answer_weight,
        "seed": seed,
    }

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_iterator = iter(train_loader)

    best_val_loss = float("inf")

    start_time = time.perf_counter()

    if use_amp:
        torch.cuda.reset_peak_memory_stats()

    for step in range(max_iters):

        model.train()

        try:

            x, y, regions = next(train_iterator)

        except StopIteration:

            train_iterator = iter(train_loader)

            x, y, regions = next(train_iterator)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        regions = regions.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):

            logits, _ = model(x)

            loss = compute_weighted_loss(
                logits,
                y,
                regions,
                args.answer_weight,
            )

        if not torch.isfinite(loss):

            raise RuntimeError(
                f"Non-finite loss at step {step + 1}"
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

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        should_evaluate = (
            completed_step == 1
            or completed_step % train_cfg["eval_interval"] == 0
            or completed_step == max_iters
        )

        if not should_evaluate:
            continue

        metrics = evaluate(
            model,
            val_loader,
            device,
            use_amp,
        )

        elapsed = time.perf_counter() - start_time

        print(
            f"\nStep {completed_step}/{max_iters}"
            f"\n  Train weighted loss: {loss.item():.4f}"
            f"\n  Val standard loss:   {metrics['loss']:.4f}"
            f"\n  Prompt loss:         {metrics.get('prompt_loss', float('nan')):.4f}"
            f"\n  Reasoning loss:      {metrics.get('reasoning_loss', float('nan')):.4f}"
            f"\n  Answer loss:         {metrics.get('answer_loss', float('nan')):.4f}"
            f"\n  Answer accuracy:     {metrics.get('answer_accuracy', 0.0) * 100:.2f}%"
            f"\n  Elapsed:             {elapsed:.1f}s"
        )

        if use_amp:

            peak_memory = (
                torch.cuda.max_memory_allocated()
                / 1024**3
            )

            print(
                f"  Peak VRAM:           {peak_memory:.2f} GB"
            )

        # Save best checkpoint using STANDARD validation loss.
        if metrics["loss"] < best_val_loss:

            best_val_loss = metrics["loss"]

            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                completed_step,
                best_val_loss,
                cfg,
                sampler,
            )

            print("  Best checkpoint updated.")

        save_checkpoint(
            output_dir / "latest.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            completed_step,
            best_val_loss,
            cfg,
            sampler,
        )

        print("  Latest checkpoint saved.")

    # --------------------------------------------------------
    # End
    # --------------------------------------------------------

    print("\nExperiment completed.")

    print(
        f"Best standard validation loss: {best_val_loss:.4f}"
    )

    print(f"Checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
