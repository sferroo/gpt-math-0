
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from functools import partial

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpt0.checkpoints import load_checkpoint
from gpt0.model import GPT0
from gpt0.dataset import MathSequenceDataset, collate_math_sequences


# --------------------------------------------------
# Experiment configuration
# --------------------------------------------------

SOURCE_CHECKPOINT = ROOT / "checkpoints/overfit_32.pt"
OUTPUT_CHECKPOINT = ROOT / "checkpoints/overfit_32_extended.pt"
DATA_PATH = ROOT / "data/processed/train_sequences.jsonl"

STEPS = 1500
BATCH_SIZE = 2
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.0
EVAL_INTERVAL = 100
SEED = 42

PAD_ID = 0
SOLUTION_ID = 5
ANSWER_ID = 6
EOS_ID = 3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# Sequence boundaries
# --------------------------------------------------

def get_boundaries(tokens):
    """
    Returns positions of SOLUTION, ANSWER and EOS
    in the original, unshifted token sequence.
    """
    try:
        solution_pos = tokens.index(SOLUTION_ID)
        answer_pos = tokens.index(ANSWER_ID)
        eos_pos = tokens.index(EOS_ID, answer_pos + 1)
    except ValueError as exc:
        raise ValueError("Invalid math sequence markers") from exc

    if not solution_pos < answer_pos < eos_pos:
        raise ValueError("Invalid marker ordering")

    return solution_pos, answer_pos, eos_pos


# --------------------------------------------------
# Correct teacher-forced metrics
# --------------------------------------------------

@torch.inference_mode()
def evaluate_subset(model, records, device, use_amp):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    reasoning_loss = 0.0
    reasoning_correct = 0
    reasoning_tokens = 0
    exact_reasoning = 0

    answer_loss = 0.0
    answer_correct = 0
    answer_tokens = 0

    for record in records:
        tokens = record["tokens"]
        solution_pos, answer_pos, eos_pos = get_boundaries(tokens)

        x = torch.tensor(
            tokens[:-1],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        y = torch.tensor(
            tokens[1:],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits, _ = model(x)

        predictions = logits.argmax(dim=-1)

        # All sequence targets.
        valid = y != -100
        total_correct += (
            (predictions == y) & valid
        ).sum().item()
        total_tokens += valid.sum().item()

        # Reasoning targets:
        # original token positions solution_pos+1 ... answer_pos-1
        # corresponding shifted prediction positions:
        # solution_pos ... answer_pos-2
        r_start = solution_pos
        r_end = answer_pos - 1

        r_logits = logits[0, r_start:r_end].float()
        r_targets = y[0, r_start:r_end]

        if r_targets.numel() == 0:
            raise ValueError(f"Empty reasoning: {record['id']}")

        r_predictions = r_logits.argmax(dim=-1)
        r_correct = (r_predictions == r_targets).sum().item()
        r_count = r_targets.numel()

        r_losses = torch.nn.functional.cross_entropy(
            r_logits,
            r_targets,
            reduction="sum",
        )

        reasoning_loss += r_losses.item()
        reasoning_correct += r_correct
        reasoning_tokens += r_count

        exact_reasoning += int(
            torch.equal(r_predictions, r_targets)
        )

        # Final answer targets, including EOS.
        a_start = answer_pos
        a_end = eos_pos

        a_logits = logits[0, a_start:a_end].float()
        a_targets = y[0, a_start:a_end]

        if a_targets.numel() > 0:
            a_predictions = a_logits.argmax(dim=-1)
            a_correct = (a_predictions == a_targets).sum().item()
            a_count = a_targets.numel()

            a_losses = torch.nn.functional.cross_entropy(
                a_logits,
                a_targets,
                reduction="sum",
            )

            answer_loss += a_losses.item()
            answer_correct += a_correct
            answer_tokens += a_count

    return {
        "token_accuracy": total_correct / max(1, total_tokens),
        "reasoning_loss": reasoning_loss / max(1, reasoning_tokens),
        "reasoning_accuracy": reasoning_correct / max(1, reasoning_tokens),
        "reasoning_exact": exact_reasoning,
        "answer_loss": answer_loss / max(1, answer_tokens),
        "answer_accuracy": answer_correct / max(1, answer_tokens),
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "answer_tokens": answer_tokens,
        "exact_total": len(records),
    }


def print_metrics(title, metrics):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Overall token accuracy: {metrics['token_accuracy']:.2%}")
    print(f"Reasoning loss:          {metrics['reasoning_loss']:.4f}")
    print(f"Reasoning accuracy:      {metrics['reasoning_accuracy']:.2%}")
    print(
        f"Exact reasoning:         "
        f"{metrics['reasoning_exact']}/{metrics['exact_total']}"
    )
    print(f"Answer loss:             {metrics['answer_loss']:.4f}")
    print(f"Answer accuracy:         {metrics['answer_accuracy']:.2%}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    if not SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError(SOURCE_CHECKPOINT)

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    if OUTPUT_CHECKPOINT.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {OUTPUT_CHECKPOINT}"
        )

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_amp = device.type == "cuda"

    print("GPT-0 EXTENDED OVERFIT EXPERIMENT")
    print("--------------------------------")
    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"Steps: {STEPS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")

    # Load source checkpoint.
    checkpoint = load_checkpoint(
        SOURCE_CHECKPOINT,
        map_location=device,
    )

    if "model" not in checkpoint or "selected_ids" not in checkpoint:
        raise KeyError(
            "Source checkpoint must contain model and selected_ids"
        )

    selected_ids = checkpoint["selected_ids"]

    if len(selected_ids) != 32 or len(set(selected_ids)) != 32:
        raise ValueError("Expected exactly 32 unique selected IDs")

    # Load original dataset and recover exactly the same records.
    dataset = MathSequenceDataset(DATA_PATH)

    records_by_id = {
        record["id"]: record
        for record in dataset.records
    }

    missing = [
        item_id for item_id in selected_ids
        if item_id not in records_by_id
    ]

    if missing:
        raise ValueError(f"Missing selected examples: {missing}")

    records = [records_by_id[item_id] for item_id in selected_ids]

    if len(records) != 32:
        raise RuntimeError("Selected dataset does not contain 32 records")

    if any(record["type"] != "gsm8k" for record in records):
        raise ValueError("Unexpected non-GSM8K record in subset")

    print(f"\nVerified selected examples: {len(records)}")
    print(f"First ID: {selected_ids[0]}")
    print(f"Last ID:  {selected_ids[-1]}")

    # Read model configuration from the source checkpoint.
    cfg = checkpoint["config"]
    model_cfg = cfg["model"]

    model = GPT0(
        vocab_size=model_cfg["vocab_size"],
        block_size=model_cfg["block_size"],
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        n_embd=model_cfg["n_embd"],
        dropout=model_cfg["dropout"],
        pad_token_id=PAD_ID,
    ).to(device)

    model.load_state_dict(checkpoint["model"])

    # Fresh optimizer: the previous experimental checkpoint
    # did not preserve optimizer state.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    loader = DataLoader(
        records,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
        pin_memory=use_amp,
        collate_fn=partial(
            collate_math_sequences,
            pad_token_id=PAD_ID,
        ),
    )

    # Baseline metrics before additional training.
    initial_metrics = evaluate_subset(
        model, records, device, use_amp
    )
    print_metrics("BEFORE EXTENDED TRAINING", initial_metrics)

    # Training.
    model.train()
    iterator = iter(loader)
    start_time = time.perf_counter()

    for step in range(1, STEPS + 1):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)

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
                f"Non-finite loss at step {step}"
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        scaler.step(optimizer)
        scaler.update()

        if step % EVAL_INTERVAL == 0 or step == STEPS:
            metrics = evaluate_subset(
                model, records, device, use_amp
            )

            elapsed = time.perf_counter() - start_time

            print(
                f"\nStep {step}/{STEPS}"
                f"\n  Training loss: {loss.item():.4f}"
                f"\n  Reasoning accuracy: "
                f"{metrics['reasoning_accuracy']:.2%}"
                f"\n  Exact reasoning: "
                f"{metrics['reasoning_exact']}/32"
                f"\n  Answer accuracy: "
                f"{metrics['answer_accuracy']:.2%}"
                f"\n  Elapsed: {elapsed:.1f}s"
            )

            model.train()

    # Final metrics.
    final_metrics = evaluate_subset(
        model, records, device, use_amp
    )
    print_metrics("AFTER EXTENDED TRAINING", final_metrics)

    # Save only to a new checkpoint.
    result = {
        "step": checkpoint.get("step", 500) + STEPS,
        "additional_steps": STEPS,
        "source_checkpoint": str(SOURCE_CHECKPOINT.name),
        "model": model.state_dict(),
        "selected_ids": selected_ids,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "config": cfg,
        "experiment": {
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
        },
    }

    torch.save(result, OUTPUT_CHECKPOINT)

    print("\nExperiment completed.")
    print(f"Checkpoint saved: {OUTPUT_CHECKPOINT}")
    print("Original checkpoints were not modified.")


if __name__ == "__main__":
    main()
