
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

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


def token_label(token_id: int) -> str:
    special = {
        NumericTokenizer.PAD: "PAD",
        NumericTokenizer.BOS: "BOS",
        NumericTokenizer.EOS: "EOS",
        NumericTokenizer.ADD: "ADD",
        NumericTokenizer.EQ: "EQ",
    }

    if token_id in special:
        return special[token_id]

    if (
        NumericTokenizer.DIGIT_OFFSET
        <= token_id
        < NumericTokenizer.VOCAB_SIZE
    ):
        return str(token_id - NumericTokenizer.DIGIT_OFFSET)

    return f"UNKNOWN_{token_id}"


def make_token_labels(a: int, b: int) -> list[str]:
    ids = NumericTokenizer.encode_input(a, b)
    return [token_label(token_id) for token_id in ids]


def attention_entropy(weights: torch.Tensor) -> torch.Tensor:
    probabilities = weights.clamp_min(1e-12)

    return -torch.sum(
        probabilities * torch.log2(probabilities),
        dim=-1,
    )


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

    captured = {}
    handles = []

    # Temporarily replace each MHA forward call with an equivalent
    # call that requests per-head attention weights.
    for layer_index, block in enumerate(model.blocks):
        attention_module = block.attn.attn

        original_forward = attention_module.forward

        def make_forward(original, layer):
            def diagnostic_forward(*args, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False

                output, weights = original(*args, **kwargs)

                captured[layer] = weights.detach().float().cpu()

                return output, weights

            return diagnostic_forward

        attention_module.forward = make_forward(
            original_forward,
            layer_index,
        )

        handles.append(
            (attention_module, original_forward)
        )

    results = {}

    try:
        for a, b in EXAMPLES:
            prompt_ids = NumericTokenizer.encode_input(a, b)

            prompt = torch.tensor(
                [prompt_ids],
                dtype=torch.long,
                device=device,
            )

            labels = make_token_labels(a, b)

            captured.clear()

            with torch.inference_mode():
                model(prompt)

            example_name = f"{a}+{b}"

            example_result = {
                "a": a,
                "b": b,
                "expected": str(a + b),
                "tokens": labels,
                "layers": {},
            }

            print("\n" + "=" * 65)
            print(f"Input: {example_name}")
            print(f"Tokens: {labels}")

            for layer_index in range(len(model.blocks)):
                weights = captured[layer_index][0]
                # Shape: [heads, target_length, source_length]

                heads_result = []

                for head_index in range(weights.shape[0]):
                    matrix = weights[head_index]

                    entropy = attention_entropy(
                        matrix
                    )

                    heads_result.append({
                        "head": head_index,
                        "mean_entropy_bits": entropy.mean().item(),
                        "mean_max_attention": matrix.max(
                            dim=-1
                        ).values.mean().item(),
                        "attention_matrix": matrix.tolist(),
                    })

                    print(
                        f"Layer {layer_index}, "
                        f"Head {head_index}: "
                        f"entropy={entropy.mean().item():.4f} bits, "
                        f"max_attention="
                        f"{matrix.max(dim=-1).values.mean().item():.4f}"
                    )

                example_result["layers"][
                    f"layer_{layer_index}"
                ] = {
                    "heads": heads_result,
                }

            results[example_name] = example_result

    finally:
        # Restore every original PyTorch method.
        for module, original_forward in handles:
            module.forward = original_forward

    report = {
        "experiment": "attention_analysis_v1",
        "seed": 42,
        "device": str(device),
        "checkpoint": "random_initialization",
        "model_config": MODEL_CONFIG,
        "attention_type": "causal_multihead_self_attention",
        "examples": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "attention_analysis_v1.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 65)
    print("Attention analysis completed.")
    print(f"Report: {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
