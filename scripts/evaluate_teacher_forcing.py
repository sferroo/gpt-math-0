
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import yaml

from gpt0.checkpoints import load_checkpoint
from gpt0.model import GPT0


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "small.yaml"
VALIDATION_PATH = ROOT / "data" / "processed" / "val_sequences.jsonl"

CHECKPOINTS = [
    ROOT / "checkpoints" / "best.pt",
    ROOT / "checkpoints" / "coverage" / "best.pt",
]

PAD_ID = 0
BOS_ID = 2
EOS_ID = 3
PROBLEM_ID = 4
SOLUTION_ID = 5
ANSWER_ID = 6

BATCH_SIZE = 2


def load_records(path: Path):
    print(f"[INFO] Loading validation data: {path}", flush=True)

    if not path.exists():
        raise FileNotFoundError(f"Validation file not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            item = json.loads(line)

            tokens = item.get("tokens", item.get("input_ids"))

            if tokens is None:
                raise KeyError(
                    f"Missing tokens at line {line_number}. "
                    f"Available fields: {list(item.keys())}"
                )

            tokens = [int(x) for x in tokens]

            if (
                PROBLEM_ID in tokens
                and SOLUTION_ID in tokens
                and ANSWER_ID in tokens
            ):
                records.append(tokens)

    if not records:
        raise RuntimeError(
            "No GSM8K records found. "
            "Expected PROBLEM, SOLUTION and ANSWER markers."
        )

    print(f"[INFO] GSM8K validation examples: {len(records)}", flush=True)

    return records


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Invalid checkpoint format.")

    for key in (
        "model_state_dict",
        "model",
        "state_dict",
        "model_state",
    ):
        value = checkpoint.get(key)

        if isinstance(value, dict) and value:
            if all(isinstance(v, torch.Tensor) for v in value.values()):
                return value

    if checkpoint and all(
        isinstance(v, torch.Tensor)
        for v in checkpoint.values()
    ):
        return checkpoint

    raise KeyError(
        "Could not find model weights in checkpoint."
    )


def load_model(checkpoint_path, model_config, device):
    print(f"[INFO] Loading checkpoint: {checkpoint_path}", flush=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = load_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )

    model = GPT0(**model_config)

    state_dict = extract_state_dict(checkpoint)

    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    print("[INFO] Model loaded successfully.", flush=True)

    return model


def new_stats():
    return {
        "loss_sum": 0.0,
        "correct": 0,
        "count": 0,
    }


def update_region(stats, logits, targets):
    if targets.numel() == 0:
        return

    logits = logits.float()

    loss = F.cross_entropy(
        logits,
        targets,
        reduction="sum",
    )

    predictions = logits.argmax(dim=-1)

    stats["loss_sum"] += loss.item()
    stats["correct"] += (
        predictions == targets
    ).sum().item()

    stats["count"] += targets.numel()


def find_marker_positions(tokens):
    if tokens[0] != BOS_ID:
        raise ValueError("Sequence does not start with BOS.")

    if tokens[-1] != EOS_ID:
        raise ValueError("Sequence does not end with EOS.")

    solution_pos = tokens.index(SOLUTION_ID)
    answer_pos = tokens.index(ANSWER_ID)

    if not (
        0 < solution_pos < answer_pos < len(tokens) - 1
    ):
        raise ValueError("Invalid marker positions.")

    return solution_pos, answer_pos


@torch.inference_mode()
def evaluate(model, records, device):
    stats = defaultdict(new_stats)
    total_stats = new_stats()

    use_amp = device.type == "cuda"

    for start in range(0, len(records), BATCH_SIZE):
        batch_records = records[start:start + BATCH_SIZE]

        sequences = [
            torch.tensor(seq[:-1], dtype=torch.long)
            for seq in batch_records
        ]

        max_len = max(len(seq) for seq in sequences)

        x = torch.full(
            (len(sequences), max_len),
            PAD_ID,
            dtype=torch.long,
            device=device,
        )

        for row, seq in enumerate(sequences):
            x[row, :len(seq)] = seq.to(device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            output = model(x)

        logits = (
            output[0]
            if isinstance(output, (tuple, list))
            else output
        )

        for row, tokens in enumerate(batch_records):
            solution_pos, answer_pos = find_marker_positions(tokens)

            sequence_length = len(tokens)

            regions = {
                # Includes the SOLUTION marker.
                "prompt": range(1, solution_pos + 1),

                # Includes the ANSWER marker.
                "reasoning": range(
                    solution_pos + 1,
                    answer_pos + 1,
                ),

                # Excludes EOS.
                "answer": range(
                    answer_pos + 1,
                    sequence_length - 1,
                ),

                "eos": [sequence_length - 1],
            }

            for region_name, positions in regions.items():
                positions = list(positions)

                if not positions:
                    continue

                prediction_positions = torch.tensor(
                    [p - 1 for p in positions],
                    dtype=torch.long,
                    device=device,
                )

                targets = torch.tensor(
                    [tokens[p] for p in positions],
                    dtype=torch.long,
                    device=device,
                )

                selected_logits = logits[
                    row,
                    prediction_positions,
                    :,
                ]

                update_region(
                    stats[region_name],
                    selected_logits,
                    targets,
                )

                update_region(
                    total_stats,
                    selected_logits,
                    targets,
                )

        processed = min(
            start + len(batch_records),
            len(records),
        )

        if processed % 100 == 0 or processed == len(records):
            print(
                f"[PROGRESS] {processed}/{len(records)}",
                flush=True,
            )

    return stats, total_stats


def print_stats(label, stats):
    count = stats["count"]

    if count == 0:
        print(f"{label:<12} No tokens")
        return

    mean_loss = stats["loss_sum"] / count

    accuracy = (
        100.0 * stats["correct"] / count
    )

    print(
        f"{label:<12} "
        f"tokens={count:>8,}  "
        f"loss={mean_loss:.4f}  "
        f"accuracy={accuracy:.3f}%"
    )


def main():
    print("=" * 65, flush=True)
    print("GPT-0 MATH - TEACHER FORCING AUDIT", flush=True)
    print("=" * 65, flush=True)

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(CONFIG_PATH)

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(VALIDATION_PATH)

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_config = config["model"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"[INFO] Device: {device}", flush=True)

    if device.type == "cuda":
        print(
            f"[INFO] GPU: {torch.cuda.get_device_name(0)}",
            flush=True,
        )

    records = load_records(VALIDATION_PATH)

    print(
        "[INFO] Starting evaluation. No training will be performed.",
        flush=True,
    )

    for checkpoint_path in CHECKPOINTS:
        print("\n" + "=" * 65)
        print(
            f"CHECKPOINT: {checkpoint_path.relative_to(ROOT)}",
            flush=True,
        )
        print("=" * 65)

        model = load_model(
            checkpoint_path,
            model_config,
            device,
        )

        region_stats, total_stats = evaluate(
            model,
            records,
            device,
        )

        print("\nRESULTS")
        print("-" * 65)

        print_stats("PROMPT", region_stats["prompt"])
        print_stats("REASONING", region_stats["reasoning"])
        print_stats("ANSWER", region_stats["answer"])
        print_stats("EOS", region_stats["eos"])
        print("-" * 65)
        print_stats("TOTAL", total_stats)

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n[COMPLETE] Teacher-forcing audit finished.", flush=True)


if __name__ == "__main__":
    main()
