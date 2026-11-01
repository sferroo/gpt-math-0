
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt0.arithmetic_model import ArithmeticGPT
from gpt0.numeric_tokenizer import NumericTokenizer


DATASET_DIR = PROJECT_ROOT / "data" / "arithmetic" / "v1"
RESULTS_DIR = PROJECT_ROOT / "results" / "arithmetic"

MODEL_CONFIG = {
    "block_size": 32,
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "dropout": 0.0,
    "vocab_size": NumericTokenizer.VOCAB_SIZE,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

                if limit is not None and len(records) >= limit:
                    break

    return records


def load_model(
    checkpoint: Path | None,
    device: torch.device,
):
    model = ArithmeticGPT(**{
        key: value
        for key, value in MODEL_CONFIG.items()
        if key != "vocab_size"
    })

    checkpoint_hash = None
    checkpoint_name = "random_initialization"

    if checkpoint is not None:
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}"
            )

        saved = torch.load(
            checkpoint,
            map_location=device,
            weights_only=True,
        )

        if isinstance(saved, dict) and "model_state_dict" in saved:
            state_dict = saved["model_state_dict"]
        elif isinstance(saved, dict) and "model" in saved:
            state_dict = saved["model"]
        else:
            state_dict = saved

        model.load_state_dict(state_dict)

        checkpoint_hash = sha256_file(checkpoint)
        checkpoint_name = str(checkpoint.resolve())

    model.to(device)
    model.eval()

    return model, checkpoint_name, checkpoint_hash


def decode_prediction(
    output: torch.Tensor,
    prompt_length: int,
) -> dict:
    generated = output[0, prompt_length:].tolist()

    eos_found = NumericTokenizer.EOS in generated

    if eos_found:
        eos_index = generated.index(NumericTokenizer.EOS)
        digit_tokens = generated[:eos_index]
    else:
        digit_tokens = generated

    if not digit_tokens:
        return {
            "prediction": None,
            "numeric_valid": False,
            "eos_found": eos_found,
            "status": "empty_output",
        }

    try:
        prediction = NumericTokenizer.decode_digits(digit_tokens)

        return {
            "prediction": str(prediction),
            "numeric_valid": True,
            "eos_found": eos_found,
            "status": "valid" if eos_found else "missing_eos",
        }

    except (ValueError, TypeError, IndexError):
        return {
            "prediction": None,
            "numeric_valid": False,
            "eos_found": eos_found,
            "status": "invalid_numeric_output",
        }


def summarize_group(group: dict) -> dict:
    total = group["total"]
    correct = group["correct"]

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
    }


def evaluate_split(
    model,
    records: list[dict],
    split_name: str,
    device: torch.device,
    max_new_tokens: int,
):
    errors = []

    groups = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
        }
    )

    diagnostics = defaultdict(int)

    correct = 0
    numeric_valid = 0
    eos_terminated = 0

    start_time = time.perf_counter()

    for index, record in enumerate(records):
        a = int(record["a"])
        b = int(record["b"])
        expected = int(record["result"])
        has_carry = bool(record["has_carry"])

        prompt = torch.tensor(
            [NumericTokenizer.encode_input(a, b)],
            dtype=torch.long,
            device=device,
        )

        with torch.inference_mode():
            output = model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
            )

        decoded = decode_prediction(
            output,
            prompt.shape[1],
        )

        prediction = decoded["prediction"]
        status = decoded["status"]

        if decoded["numeric_valid"]:
            numeric_valid += 1

        if decoded["eos_found"]:
            eos_terminated += 1

        is_correct = (
            decoded["numeric_valid"]
            and prediction == str(expected)
        )

        if is_correct:
            correct += 1
            diagnostics["correct"] += 1
        else:
            diagnostics[status] += 1

            if decoded["numeric_valid"]:
                diagnostics["incorrect_answer"] += 1

            errors.append({
                "index": index,
                "a": a,
                "b": b,
                "expected": str(expected),
                "predicted": prediction,
                "status": status,
                "numeric_valid": decoded["numeric_valid"],
                "eos_found": decoded["eos_found"],
                "has_carry": has_carry,
            })

        carry_group = "carry" if has_carry else "no_carry"

        # Stratification by operand digit lengths.
        operand_group = f"{len(str(a))}_digit_operands"

        # Stratification by result length.
        result_group = f"{len(str(expected))}_digit_result"

        group_keys = {
            "carry": carry_group,
            "operand_digits": operand_group,
            "result_digits": result_group,
        }

        for category, key in group_keys.items():
            group = groups[f"{category}:{key}"]
            group["total"] += 1
            group["correct"] += int(is_correct)

        if (index + 1) % 100 == 0:
            print(
                f"[{split_name}] "
                f"{index + 1}/{len(records)} examples evaluated"
            )

    elapsed = time.perf_counter() - start_time
    total = len(records)

    stratified = {
        "carry": {},
        "operand_digits": {},
        "result_digits": {},
    }

    for combined_key, values in groups.items():
        category, key = combined_key.split(":", 1)
        stratified[category][key] = summarize_group(values)

    report = {
        "split": split_name,
        "total": total,
        "correct": correct,
        "exact_match_accuracy": correct / total if total else 0.0,
        "numeric_valid_outputs": numeric_valid,
        "numeric_valid_rate": numeric_valid / total if total else 0.0,
        "eos_terminated_outputs": eos_terminated,
        "eos_termination_rate": eos_terminated / total if total else 0.0,
        "diagnostics": dict(diagnostics),
        "stratified_metrics": stratified,
        "elapsed_seconds": elapsed,
        "examples_per_second": total / elapsed if elapsed else 0.0,
        "errors": errors,
    }

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate arithmetic generalization in mathGPT."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional trained model checkpoint.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum examples per split. Use 0 for all examples.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit cannot be negative.")

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 60)
    print("mathGPT - Arithmetic Evaluation v2")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Seed: {args.seed}")

    model, checkpoint_name, checkpoint_hash = load_model(
        args.checkpoint,
        device,
    )

    print(f"Checkpoint: {checkpoint_name}")

    limit = args.limit if args.limit > 0 else None

    split_files = {
        "test_id": DATASET_DIR / "test_id.jsonl",
        "test_ood": DATASET_DIR / "test_ood.jsonl",
    }

    reports = {}
    dataset_hashes = {}

    for split_name, path in split_files.items():
        if not path.exists():
            raise FileNotFoundError(path)

        dataset_hashes[split_name] = sha256_file(path)
        records = load_jsonl(path, limit)

        if not records:
            raise ValueError(
                f"Empty dataset split: {split_name}"
            )

        print(
            f"\nEvaluating {split_name}: "
            f"{len(records)} examples"
        )

        report = evaluate_split(
            model=model,
            records=records,
            split_name=split_name,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )

        reports[split_name] = report

        print(
            f"\nExact Match: "
            f"{report['correct']}/{report['total']} "
            f"({report['exact_match_accuracy']:.2%})"
        )

        print(
            f"Numeric valid: "
            f"{report['numeric_valid_outputs']}/{report['total']} "
            f"({report['numeric_valid_rate']:.2%})"
        )

        print(
            f"EOS terminated: "
            f"{report['eos_terminated_outputs']}/{report['total']} "
            f"({report['eos_termination_rate']:.2%})"
        )

        print("\nDiagnostics:")
        for name, count in sorted(report["diagnostics"].items()):
            print(f"  {name}: {count}")

        print("\nAccuracy by carry:")
        for name, values in report["stratified_metrics"]["carry"].items():
            print(
                f"  {name}: "
                f"{values['correct']}/{values['total']} "
                f"({values['accuracy']:.2%})"
            )

        print("\nAccuracy by operand digits:")
        for name, values in sorted(
            report["stratified_metrics"]["operand_digits"].items()
        ):
            print(
                f"  {name}: "
                f"{values['correct']}/{values['total']} "
                f"({values['accuracy']:.2%})"
            )

        print("\nAccuracy by result digits:")
        for name, values in sorted(
            report["stratified_metrics"]["result_digits"].items()
        ):
            print(
                f"  {name}: "
                f"{values['correct']}/{values['total']} "
                f"({values['accuracy']:.2%})"
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    output_path = RESULTS_DIR / f"evaluation_v2_{timestamp}.json"

    final_report = {
        "experiment": "arithmetic_generalization_v1",
        "evaluator_version": 2,
        "checkpoint": checkpoint_name,
        "checkpoint_sha256": checkpoint_hash,
        "device": str(device),
        "torch_version": torch.__version__,
        "seed": args.seed,
        "model_config": MODEL_CONFIG,
        "max_new_tokens": args.max_new_tokens,
        "limit_per_split": args.limit,
        "selection": "first N records in each split" if limit else "full split",
        "dataset_sha256": dataset_hashes,
        "results": reports,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            final_report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print("Evaluation v2 completed.")
    print(f"Report saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
