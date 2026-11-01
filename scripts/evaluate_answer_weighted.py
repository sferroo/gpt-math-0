
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import yaml

from gpt0.checkpoints import load_checkpoint
from gpt0.model import GPT0


CONFIG_PATH = ROOT / "configs" / "small.yaml"
VALIDATION_PATH = ROOT / "data" / "processed" / "val_sequences.jsonl"
OUTPUT_PATH = ROOT / "reports" / "answer_weighted_comparison.csv"

CHECKPOINTS = {
    "coverage": ROOT / "checkpoints" / "coverage" / "best.pt",
    "answer_1x": ROOT / "checkpoints" / "answer_weighted" / "weight_1x" / "best.pt",
    "answer_2x": ROOT / "checkpoints" / "answer_weighted" / "weight_2x" / "best.pt",
    "answer_5x": ROOT / "checkpoints" / "answer_weighted" / "weight_5x" / "best.pt",
}

PAD_ID = 0
BOS_ID = 2
EOS_ID = 3
PROBLEM_ID = 4
SOLUTION_ID = 5
ANSWER_ID = 6

BATCH_SIZE = 2

REGIONS = ("prompt", "reasoning", "answer", "eos")


def load_records(path):
    print(f"[INFO] Loading validation data: {path}", flush=True)

    if not path.exists():
        raise FileNotFoundError(path)

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            item = json.loads(line)
            tokens = item.get("tokens", item.get("input_ids"))

            if tokens is None:
                raise KeyError(f"Missing tokens at line {line_number}")

            tokens = [int(token) for token in tokens]

            if all(marker in tokens for marker in (
                PROBLEM_ID, SOLUTION_ID, ANSWER_ID
            )):
                records.append(tokens)

    if not records:
        raise RuntimeError("No GSM8K validation records found.")

    print(f"[INFO] GSM8K validation examples: {len(records)}", flush=True)
    return records


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Invalid checkpoint format.")

    for key in ("model_state_dict", "model", "state_dict", "model_state"):
        value = checkpoint.get(key)

        if isinstance(value, dict) and value:
            if all(isinstance(v, torch.Tensor) for v in value.values()):
                return value

    if checkpoint and all(
        isinstance(value, torch.Tensor)
        for value in checkpoint.values()
    ):
        return checkpoint

    raise KeyError("Could not find model weights in checkpoint.")


def load_model(path, model_config, device):
    print(f"[INFO] Loading checkpoint: {path}", flush=True)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = load_checkpoint(
        path,
        map_location="cpu",
    )

    model = GPT0(**model_config)
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
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
    stats["correct"] += (predictions == targets).sum().item()
    stats["count"] += targets.numel()


def find_marker_positions(tokens):
    if tokens[0] != BOS_ID:
        raise ValueError("Sequence does not start with BOS.")

    if tokens[-1] != EOS_ID:
        raise ValueError("Sequence does not end with EOS.")

    solution_pos = tokens.index(SOLUTION_ID)
    answer_pos = tokens.index(ANSWER_ID)

    if not (0 < solution_pos < answer_pos < len(tokens) - 1):
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
            torch.tensor(sequence[:-1], dtype=torch.long)
            for sequence in batch_records
        ]

        max_len = max(len(sequence) for sequence in sequences)

        x = torch.full(
            (len(sequences), max_len),
            PAD_ID,
            dtype=torch.long,
            device=device,
        )

        for row, sequence in enumerate(sequences):
            x[row, :len(sequence)] = sequence.to(device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            output = model(x)

        logits = output[0] if isinstance(output, (tuple, list)) else output

        for row, tokens in enumerate(batch_records):
            solution_pos, answer_pos = find_marker_positions(tokens)
            sequence_length = len(tokens)

            regions = {
                "prompt": range(1, solution_pos + 1),
                "reasoning": range(solution_pos + 1, answer_pos + 1),
                "answer": range(answer_pos + 1, sequence_length - 1),
                "eos": [sequence_length - 1],
            }

            for region_name, positions in regions.items():
                positions = list(positions)

                if not positions:
                    continue

                prediction_positions = torch.tensor(
                    [position - 1 for position in positions],
                    dtype=torch.long,
                    device=device,
                )

                targets = torch.tensor(
                    [tokens[position] for position in positions],
                    dtype=torch.long,
                    device=device,
                )

                selected_logits = logits[
                    row,
                    prediction_positions,
                    :,
                ]

                update_region(stats[region_name], selected_logits, targets)
                update_region(total_stats, selected_logits, targets)

        processed = min(start + len(batch_records), len(records))

        if processed % 100 == 0 or processed == len(records):
            print(f"[PROGRESS] {processed}/{len(records)}", flush=True)

    return stats, total_stats


def metric_values(stats):
    count = stats["count"]

    return {
        "tokens": count,
        "loss": stats["loss_sum"] / count if count else None,
        "accuracy_percent": (
            100.0 * stats["correct"] / count if count else None
        ),
    }


def main():
    print("=" * 70)
    print("GPT-0 MATH — ANSWER-WEIGHTED COMPARISON")
    print("=" * 70)

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(CONFIG_PATH)

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}", flush=True)

    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    records = load_records(VALIDATION_PATH)

    print("[INFO] Evaluation only. No training will be performed.", flush=True)

    results = []

    for name, checkpoint_path in CHECKPOINTS.items():
        print("\n" + "=" * 70)
        print(f"MODEL: {name}")
        print("=" * 70)

        model = load_model(checkpoint_path, config["model"], device)
        region_stats, total_stats = evaluate(model, records, device)

        row = {"model": name}

        for region in REGIONS:
            metrics = metric_values(region_stats[region])

            row[f"{region}_tokens"] = metrics["tokens"]
            row[f"{region}_loss"] = metrics["loss"]
            row[f"{region}_accuracy_percent"] = metrics["accuracy_percent"]

            print(
                f"{region.upper():<12}"
                f"tokens={metrics['tokens']:>8,}  "
                f"loss={metrics['loss']:.4f}  "
                f"accuracy={metrics['accuracy_percent']:.3f}%",
                flush=True,
            )

        total = metric_values(total_stats)
        row["total_tokens"] = total["tokens"]
        row["total_loss"] = total["loss"]
        row["total_accuracy_percent"] = total["accuracy_percent"]

        print(
            f"TOTAL        tokens={total['tokens']:>8,}  "
            f"loss={total['loss']:.4f}  "
            f"accuracy={total['accuracy_percent']:.3f}%",
            flush=True,
        )

        results.append(row)

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 70)
    print("[COMPLETE] Comparison finished.")
    print(f"[OUTPUT] {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
