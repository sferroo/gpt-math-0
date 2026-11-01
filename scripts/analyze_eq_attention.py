
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/arithmetic/attention_analysis_v1.json"
OUTPUT = ROOT / "results/arithmetic/eq_attention_analysis_v1.json"

LAYERS = [0, 5]


def get_token_groups(tokens):
    """Identify operand and operator positions."""
    add_index = tokens.index("ADD")
    eq_index = tokens.index("EQ")

    return {
        "BOS": [0],
        "operand_a": list(range(1, add_index)),
        "ADD": [add_index],
        "operand_b": list(range(add_index + 1, eq_index)),
        "EQ": [eq_index],
    }


def analyze_head(matrix, groups):
    """Analyze the attention row belonging to EQ."""
    eq_index = len(matrix) - 1
    eq_row = matrix[eq_index]

    result = {}

    for group_name, positions in groups.items():
        values = [eq_row[i] for i in positions]

        result[group_name] = {
            "total_attention": sum(values),
            "mean_per_token": mean(values) if values else 0.0,
            "token_count": len(values),
        }

    return result


def analyze_example(example_name, example):
    tokens = example["tokens"]
    groups = get_token_groups(tokens)

    result = {
        "tokens": tokens,
        "groups": groups,
        "layers": {},
    }

    for layer_index in LAYERS:
        layer_name = f"layer_{layer_index}"
        heads = example["layers"][layer_name]["heads"]

        head_results = []

        for head in heads:
            metrics = analyze_head(
                head["attention_matrix"],
                groups
            )

            head_results.append({
                "head": head["head"],
                "groups": metrics,
            })

        group_summary = {}

        for group_name in groups:
            group_summary[group_name] = {
                "mean_total_attention": mean(
                    h["groups"][group_name]["total_attention"]
                    for h in head_results
                ),
                "mean_per_token": mean(
                    h["groups"][group_name]["mean_per_token"]
                    for h in head_results
                ),
            }

        result["layers"][layer_name] = {
            "heads": head_results,
            "summary": group_summary,
        }

    # Difference between the last and first layer.
    first = result["layers"]["layer_0"]["summary"]
    last = result["layers"]["layer_5"]["summary"]

    result["layer_5_minus_layer_0"] = {}

    for group_name in groups:
        result["layer_5_minus_layer_0"][group_name] = {
            "total_attention": (
                last[group_name]["mean_total_attention"]
                - first[group_name]["mean_total_attention"]
            ),
            "mean_per_token": (
                last[group_name]["mean_per_token"]
                - first[group_name]["mean_per_token"]
            ),
        }

    return result


def print_summary(example_name, result):
    print("\n" + "=" * 72)
    print(f"EXAMPLE: {example_name}")
    print("EQ attention analysis")
    print("=" * 72)

    for layer_name in ["layer_0", "layer_5"]:
        print(f"\n{layer_name.upper()}")

        summary = result["layers"][layer_name]["summary"]

        print(
            f"{'Group':<14}"
            f"{'Total attention':>18}"
            f"{'Mean/token':>16}"
        )

        for group_name, values in summary.items():
            print(
                f"{group_name:<14}"
                f"{values['mean_total_attention']:>18.6f}"
                f"{values['mean_per_token']:>16.6f}"
            )

    print("\nCHANGE: layer_5 - layer_0")

    for group_name, values in result["layer_5_minus_layer_0"].items():
        print(
            f"{group_name:<14}"
            f"total={values['total_attention']:+.6f}  "
            f"per_token={values['mean_per_token']:+.6f}"
        )


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Report not found: {INPUT}")

    with INPUT.open(encoding="utf-8") as f:
        data = json.load(f)

    results = {
        "experiment": "eq_attention_analysis_v1",
        "source": str(INPUT),
        "layers_compared": LAYERS,
        "examples": {},
    }

    for example_name, example in data["examples"].items():
        result = analyze_example(example_name, example)
        results["examples"][example_name] = result

        print_summary(example_name, result)

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("EQ ATTENTION ANALYSIS COMPLETED")
    print(f"Report: {OUTPUT}")
    print("=" * 72)


if __name__ == "__main__":
    main()
