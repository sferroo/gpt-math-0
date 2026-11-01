
import json
import math
from pathlib import Path
from statistics import mean, stdev
from itertools import combinations


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results/arithmetic/eq_attention_dataset_v1.json"
OUTPUT = ROOT / "results/arithmetic/head_specialization_v1.json"

LAYERS = ["layer_0", "layer_5"]

GROUPS = ["BOS", "operand_a", "ADD", "operand_b", "EQ"]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def stratum_name(observation):
    length = observation["length_group"]
    carry = "carry" if observation["has_carry"] else "no_carry"
    return f"{length}_{carry}"


def get_head_vector(head):
    return [
        head["groups"][group]["total_attention"]
        for group in GROUPS
    ]


def analyze_head_means(observations, layer_name):
    result = {}

    for head_index in range(6):
        result[f"head_{head_index}"] = {}

        for group in GROUPS:
            values = [
                observation["layers"][layer_name][head_index]
                ["groups"][group]["total_attention"]
                for observation in observations
            ]

            result[f"head_{head_index}"][group] = {
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }

    return result


def analyze_similarity(observations, layer_name):
    pair_values = {
        (a, b): []
        for a, b in combinations(range(6), 2)
    }

    for observation in observations:
        heads = observation["layers"][layer_name]

        vectors = [
            get_head_vector(head)
            for head in heads
        ]

        for pair in pair_values:
            a, b = pair
            similarity = cosine_similarity(
                vectors[a],
                vectors[b]
            )
            pair_values[pair].append(similarity)

    result = []

    for (a, b), values in pair_values.items():
        result.append({
            "head_a": a,
            "head_b": b,
            "mean_cosine_similarity": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        })

    return result


def analyze_stratum(observations):
    result = {}

    for layer_name in LAYERS:
        result[layer_name] = {
            "head_means": analyze_head_means(
                observations,
                layer_name
            ),
            "pairwise_similarity": analyze_similarity(
                observations,
                layer_name
            ),
        }

    return result


def print_results(stratum, result):
    print("\n" + "=" * 72)
    print(f"STRATUM: {stratum}")
    print("=" * 72)

    for layer_name in LAYERS:
        print(f"\n{layer_name.upper()} — MEAN ATTENTION BY HEAD")

        heads = result[layer_name]["head_means"]

        print(
            f"{'Head':<10}" +
            "".join(f"{group:>12}" for group in GROUPS)
        )

        for head_name, metrics in heads.items():
            values = "".join(
                f"{metrics[group]['mean']:>12.4f}"
                for group in GROUPS
            )
            print(f"{head_name:<10}{values}")

        similarities = result[layer_name]["pairwise_similarity"]

        avg_similarity = mean(
            item["mean_cosine_similarity"]
            for item in similarities
        )

        print(
            f"\nMean pairwise cosine similarity: "
            f"{avg_similarity:.5f}"
        )


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    with INPUT.open(encoding="utf-8") as f:
        data = json.load(f)

    strata = {}

    for observation in data["observations"]:
        name = stratum_name(observation)
        strata.setdefault(name, []).append(observation)

    results = {
        "experiment": "head_specialization_v1",
        "source": str(INPUT),
        "metric": "EQ-row grouped attention",
        "groups": GROUPS,
        "layers": LAYERS,
        "total_observations": len(data["observations"]),
        "strata": {},
    }

    for name, observations in sorted(strata.items()):
        analysis = analyze_stratum(observations)

        results["strata"][name] = {
            "n": len(observations),
            **analysis,
        }

        print_results(name, analysis)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("HEAD SPECIALIZATION ANALYSIS COMPLETED")
    print("Observations:", len(data["observations"]))
    print("Report:", OUTPUT)
    print("=" * 72)


if __name__ == "__main__":
    main()
