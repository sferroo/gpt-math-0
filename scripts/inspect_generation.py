
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt0.arithmetic_model import ArithmeticGPT
from gpt0.numeric_tokenizer import NumericTokenizer


RESULTS_DIR = PROJECT_ROOT / "results" / "arithmetic"

MODEL_CONFIG = {
    "block_size": 32,
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "dropout": 0.0,
}

EXAMPLES = [
    (12, 25),
    (78, 45),
    (99, 99),
    (1234, 5678),
    (9999, 9999),
]


def inspect_output(output, prompt_length):
    generated = output[0, prompt_length:].tolist()

    eos_found = NumericTokenizer.EOS in generated

    if eos_found:
        eos_index = generated.index(NumericTokenizer.EOS)
        digit_tokens = generated[:eos_index]
    else:
        digit_tokens = generated

    try:
        prediction = (
            NumericTokenizer.decode_digits(digit_tokens)
            if digit_tokens
            else None
        )
    except (ValueError, TypeError, IndexError):
        prediction = None

    return {
        "generated_token_ids": generated,
        "generated_digit_tokens": digit_tokens,
        "prediction": str(prediction) if prediction is not None else None,
        "eos_found": eos_found,
        "generated_token_count": len(generated),
        "digit_count": len(digit_tokens),
    }


def main():
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = ArithmeticGPT(**MODEL_CONFIG)
    model.to(device)
    model.eval()

    print("=" * 60)
    print("mathGPT - Generation Diagnostics v1")
    print("=" * 60)
    print(f"Device: {device}")
    print("Checkpoint: Random initialization")
    print("Seed: 42")

    results = []

    for a, b in EXAMPLES:
        expected = a + b

        prompt_ids = NumericTokenizer.encode_input(a, b)

        prompt = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=device,
        )

        with torch.inference_mode():
            output = model.generate(
                prompt,
                max_new_tokens=6,
            )

        decoded = inspect_output(
            output,
            prompt.shape[1],
        )

        item = {
            "a": a,
            "b": b,
            "expected": str(expected),
            **decoded,
            "exact_match": decoded["prediction"] == str(expected),
        }

        results.append(item)

        print("\n" + "-" * 60)
        print(f"Input: {a} + {b}")
        print(f"Expected: {expected}")
        print(f"Generated token IDs: {decoded['generated_token_ids']}")
        print(f"Prediction: {decoded['prediction']}")
        print(f"EOS found: {decoded['eos_found']}")
        print(f"Exact match: {item['exact_match']}")

    predictions = [
        item["prediction"]
        for item in results
        if item["prediction"] is not None
    ]

    summary = {
        "experiment": "generation_diagnostics_v1",
        "seed": 42,
        "device": str(device),
        "checkpoint": "random_initialization",
        "max_new_tokens": 6,
        "examples": results,
        "summary": {
            "total_examples": len(results),
            "unique_predictions": len(set(predictions)),
            "eos_terminated": sum(
                item["eos_found"] for item in results
            ),
            "exact_matches": sum(
                item["exact_match"] for item in results
            ),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "generation_diagnostics_v1.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print("Diagnostic completed.")
    print(f"Unique predictions: {summary['summary']['unique_predictions']}")
    print(f"EOS terminated: {summary['summary']['eos_terminated']}/5")
    print(f"Exact matches: {summary['summary']['exact_matches']}/5")
    print(f"Report: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
