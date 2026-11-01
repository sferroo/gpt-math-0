
from __future__ import annotations

import json
import math
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


def token_name(token_id: int) -> str:
    if token_id == NumericTokenizer.PAD:
        return "PAD"
    if token_id == NumericTokenizer.BOS:
        return "BOS"
    if token_id == NumericTokenizer.EOS:
        return "EOS"
    if token_id == NumericTokenizer.ADD:
        return "ADD"
    if token_id == NumericTokenizer.EQ:
        return "EQ"

    if (
        NumericTokenizer.DIGIT_OFFSET
        <= token_id
        < NumericTokenizer.VOCAB_SIZE
    ):
        return str(token_id - NumericTokenizer.DIGIT_OFFSET)

    return f"UNKNOWN_{token_id}"


def analyze_distribution(logits: torch.Tensor) -> dict:
    probabilities = torch.softmax(logits.float(), dim=-1)

    # Entropy in bits.
    entropy = -torch.sum(
        probabilities * torch.log2(
            probabilities.clamp_min(1e-12)
        )
    ).item()

    top_count = min(5, probabilities.numel())

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=top_count,
    )

    top_tokens = []

    for token_id, probability in zip(
        top_indices.tolist(),
        top_probabilities.tolist(),
    ):
        top_tokens.append({
            "token_id": token_id,
            "token": token_name(token_id),
            "probability": probability,
        })

    return {
        "logits": logits.float().cpu().tolist(),
        "probabilities": probabilities.cpu().tolist(),
        "entropy_bits": entropy,
        "top_tokens": top_tokens,
        "max_probability": probabilities.max().item(),
    }


def compare_distributions(
    first: list[float],
    second: list[float],
) -> dict:
    p = torch.tensor(first, dtype=torch.float64)
    q = torch.tensor(second, dtype=torch.float64)

    # Total variation distance: range [0, 1].
    total_variation = (
        0.5 * torch.sum(torch.abs(p - q))
    ).item()

    cosine_similarity = torch.nn.functional.cosine_similarity(
        p.unsqueeze(0),
        q.unsqueeze(0),
    ).item()

    # Jensen-Shannon divergence, in bits.
    midpoint = (p + q) / 2

    def kl_divergence(x, y):
        return torch.sum(
            x * torch.log2(
                x.clamp_min(1e-12)
                / y.clamp_min(1e-12)
            )
        )

    js_divergence = (
        0.5 * kl_divergence(p, midpoint)
        + 0.5 * kl_divergence(q, midpoint)
    ).item()

    return {
        "total_variation_distance": total_variation,
        "cosine_similarity": cosine_similarity,
        "jensen_shannon_divergence_bits": js_divergence,
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

    print("=" * 65)
    print("mathGPT - Logits Analysis v1")
    print("=" * 65)
    print(f"Device: {device}")
    print("Checkpoint: Random initialization")
    print("Seed: 42")

    results = []
    distributions = []

    for a, b in EXAMPLES:
        expected = a + b

        prompt_ids = NumericTokenizer.encode_input(a, b)

        prompt = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=device,
        )

        with torch.inference_mode():
            logits, _ = model(prompt)

        # Logits at the last prompt position predict
        # the first generated answer token.
        next_token_logits = logits[0, -1, :]

        analysis = analyze_distribution(
            next_token_logits
        )

        item = {
            "a": a,
            "b": b,
            "expected": str(expected),
            "prompt_token_ids": prompt_ids,
            **analysis,
        }

        results.append(item)
        distributions.append(analysis["probabilities"])

        print("\n" + "-" * 65)
        print(f"Input: {a} + {b} = {expected}")
        print(f"Entropy: {analysis['entropy_bits']:.5f} bits")
        print(
            f"Maximum probability: "
            f"{analysis['max_probability']:.6f}"
        )

        print("Top-5 next tokens:")

        for rank, token in enumerate(
            analysis["top_tokens"],
            start=1,
        ):
            print(
                f"  {rank}. Token {token['token_id']} "
                f"({token['token']}): "
                f"{token['probability']:.6f}"
            )

    comparisons = []

    print("\n" + "=" * 65)
    print("PAIRWISE DISTRIBUTION COMPARISONS")
    print("=" * 65)

    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            comparison = compare_distributions(
                distributions[i],
                distributions[j],
            )

            pair = {
                "first_input": (
                    results[i]["a"],
                    results[i]["b"],
                ),
                "second_input": (
                    results[j]["a"],
                    results[j]["b"],
                ),
                **comparison,
            }

            comparisons.append(pair)

            print(
                f"\n{results[i]['a']}+{results[i]['b']} "
                f"vs "
                f"{results[j]['a']}+{results[j]['b']}"
            )

            print(
                f"  Total variation: "
                f"{comparison['total_variation_distance']:.6f}"
            )

            print(
                f"  Cosine similarity: "
                f"{comparison['cosine_similarity']:.6f}"
            )

            print(
                f"  JS divergence: "
                f"{comparison['jensen_shannon_divergence_bits']:.6f} bits"
            )

    summary = {
        "experiment": "logits_analysis_v1",
        "seed": 42,
        "device": str(device),
        "checkpoint": "random_initialization",
        "model_config": MODEL_CONFIG,
        "analysis_position": "last_prompt_token",
        "predicted_token": "first_answer_token",
        "examples": results,
        "pairwise_comparisons": comparisons,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "logits_analysis_v1.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 65)
    print("Logits analysis completed.")
    print(f"Report: {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
