
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
    {"a": 12, "b": 25, "label": "12+25"},
    {"a": 78, "b": 45, "label": "78+45"},
    {"a": 99, "b": 99, "label": "99+99"},
    {"a": 1234, "b": 5678, "label": "1234+5678"},
    {"a": 9999, "b": 9999, "label": "9999+9999"},
]


def vector_metrics(x: torch.Tensor, y: torch.Tensor) -> dict:
    x = x.float().flatten()
    y = y.float().flatten()

    return {
        "cosine_similarity": F.cosine_similarity(
            x.unsqueeze(0),
            y.unsqueeze(0),
        ).item(),
        "euclidean_distance": torch.linalg.vector_norm(x - y).item(),
        "l2_norm_x": torch.linalg.vector_norm(x).item(),
        "l2_norm_y": torch.linalg.vector_norm(y).item(),
    }


def extract_activations(model, prompt, device):
    activations = {}
    handles = []

    def save_activation(name):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output

            activations[name] = tensor.detach().float().cpu()

        return hook

    handles.append(
        model.token_emb.register_forward_hook(
            save_activation("embedding")
        )
    )

    for index, block in enumerate(model.blocks):
        handles.append(
            block.register_forward_hook(
                save_activation(f"block_{index}")
            )
        )

    handles.append(
        model.ln_f.register_forward_hook(
            save_activation("ln_f")
        )
    )

    try:
        with torch.inference_mode():
            logits, _ = model(prompt)
    finally:
        for handle in handles:
            handle.remove()

    return activations, logits.detach().float().cpu()


def get_semantic_positions(a: int, b: int) -> dict:
    """
    Prompt format:
    BOS, digits(a), ADD, digits(b), EQ

    Returns positions of:
    - final digit of operand A
    - final digit of operand B
    - EQ token (last prompt position)
    """

    digits_a = len(str(a))
    digits_b = len(str(b))

    return {
        "operand_a_last_digit": digits_a,
        "operand_b_last_digit": digits_a + digits_b + 1,
        "eq_position": digits_a + digits_b + 2,
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
    print("mathGPT - Representation Analysis v1")
    print("=" * 65)
    print(f"Device: {device}")
    print("Checkpoint: Random initialization")
    print("Seed: 42")

    all_results = {}

    for example in EXAMPLES:
        a = example["a"]
        b = example["b"]
        label = example["label"]

        prompt_ids = NumericTokenizer.encode_input(a, b)

        prompt = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=device,
        )

        activations, logits = extract_activations(
            model,
            prompt,
            device,
        )

        positions = get_semantic_positions(a, b)

        example_result = {
            "a": a,
            "b": b,
            "expected": str(a + b),
            "prompt_token_ids": prompt_ids,
            "semantic_positions": positions,
            "layers": {},
        }

        print("\n" + "-" * 65)
        print(f"Input: {label}")

        for layer_name, tensor in activations.items():
            # Shape: [batch, sequence, embedding_dim]
            sequence = tensor[0]

            layer_metrics = {
                "sequence_length": sequence.shape[0],
                "embedding_dim": sequence.shape[1],
                "sequence_mean_l2_norm": torch.linalg.vector_norm(
                    sequence.mean(dim=0)
                ).item(),
                "sequence_mean_abs": sequence.abs().mean().item(),
                "positions": {},
            }

            for position_name, position in positions.items():
                vector = sequence[position]

                layer_metrics["positions"][position_name] = {
                    "position": position,
                    "l2_norm": torch.linalg.vector_norm(
                        vector
                    ).item(),
                    "mean": vector.mean().item(),
                    "std": vector.std(unbiased=False).item(),
                    "values": vector.tolist(),
                }

            example_result["layers"][layer_name] = layer_metrics

            eq_norm = layer_metrics["positions"][
                "eq_position"
            ]["l2_norm"]

            print(
                f"{layer_name:12s} "
                f"EQ L2 norm: {eq_norm:.6f}"
            )

        example_result["next_token_logits"] = logits[
            0, -1
        ].tolist()

        all_results[label] = example_result

    # Compare semantically aligned positions across examples.
    comparisons = []

    labels = list(all_results.keys())

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            first_label = labels[i]
            second_label = labels[j]

            first = all_results[first_label]
            second = all_results[second_label]

            for layer_name in first["layers"]:
                for position_name in (
                    "operand_a_last_digit",
                    "operand_b_last_digit",
                    "eq_position",
                ):
                    vector_a = torch.tensor(
                        first["layers"][layer_name][
                            "positions"
                        ][position_name]["values"]
                    )

                    vector_b = torch.tensor(
                        second["layers"][layer_name][
                            "positions"
                        ][position_name]["values"]
                    )

                    metrics = vector_metrics(
                        vector_a,
                        vector_b,
                    )

                    comparisons.append({
                        "first_input": first_label,
                        "second_input": second_label,
                        "layer": layer_name,
                        "position": position_name,
                        **metrics,
                    })

    summary = {
        "experiment": "representation_analysis_v1",
        "seed": 42,
        "device": str(device),
        "checkpoint": "random_initialization",
        "model_config": MODEL_CONFIG,
        "examples": all_results,
        "comparisons": comparisons,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        RESULTS_DIR / "representations_v1.json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 65)
    print("Representation analysis completed.")
    print(f"Report: {output_path}")
    print(f"Comparisons: {len(comparisons)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
