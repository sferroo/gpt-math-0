
import json
import random
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results/arithmetic/eq_attention_dataset_v1.json"
OUTPUT = ROOT / "results/arithmetic/attention_head_robustness_v1.json"

GROUPS = ["BOS", "operand_a", "ADD", "operand_b", "EQ"]
LAYERS = ["layer_0", "layer_5"]
METRICS = ["total_attention", "mean_per_token"]

BOOTSTRAP_ITERATIONS = 2000
SEED = 20261101


def percentile(sorted_values, p):
    index = round((len(sorted_values) - 1) * p)
    return sorted_values[index]


def bootstrap_ci(values, rng):
    if not values:
        raise ValueError("Bootstrap ricevuto un campione vuoto.")

    n = len(values)
    estimates = []

    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(mean(sample))

    estimates.sort()

    return {
        "lower_95": percentile(estimates, 0.025),
        "upper_95": percentile(estimates, 0.975),
        "iterations": BOOTSTRAP_ITERATIONS,
    }


def describe(values, rng):
    return {
        "n": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "bootstrap_ci": bootstrap_ci(values, rng),
    }


def get_group_value(observation, layer, head_index, group, metric):
    head = observation["layers"][layer][head_index]
    return head["groups"][group][metric]


def get_stratum(observation):
    length = observation["length_group"]
    carry = "carry" if observation["has_carry"] else "no_carry"
    return f"{length}_{carry}"


def analyze_stratum(observations, rng):
    result = {
        "n": len(observations),
        "layers": {},
        "layer_delta": {},
    }

    for layer in LAYERS:
        result["layers"][layer] = {}

        for metric in METRICS:
            result["layers"][layer][metric] = {}

            for head_index in range(6):
                head_name = f"head_{head_index}"
                result["layers"][layer][metric][head_name] = {}

                for group in GROUPS:
                    values = [
                        get_group_value(
                            obs, layer, head_index, group, metric
                        )
                        for obs in observations
                    ]

                    result["layers"][layer][metric][head_name][group] = (
                        describe(values, rng)
                    )

    # Confronto appaiato: stessa operazione, Layer 5 meno Layer 0.
    for metric in METRICS:
        result["layer_delta"][metric] = {}

        for head_index in range(6):
            head_name = f"head_{head_index}"
            result["layer_delta"][metric][head_name] = {}

            for group in GROUPS:
                deltas = []

                for obs in observations:
                    before = get_group_value(
                        obs, "layer_0", head_index, group, metric
                    )
                    after = get_group_value(
                        obs, "layer_5", head_index, group, metric
                    )
                    deltas.append(after - before)

                result["layer_delta"][metric][head_name][group] = (
                    describe(deltas, rng)
                )

    return result


def compare_carry_groups(strata, rng):
    comparisons = {}

    for length in ["short", "long"]:
        carry_name = f"{length}_carry"
        no_carry_name = f"{length}_no_carry"

        if carry_name not in strata or no_carry_name not in strata:
            continue

        carry = strata[carry_name]
        no_carry = strata[no_carry_name]

        comparison_key = f"{length}_carry_minus_no_carry"
        comparisons[comparison_key] = {
            "n_carry": len(carry),
            "n_no_carry": len(no_carry),
            "layers": {},
        }

        for layer in LAYERS:
            comparisons[comparison_key]["layers"][layer] = {}

            for metric in METRICS:
                comparisons[comparison_key]["layers"][layer][metric] = {}

                for head_index in range(6):
                    head_name = f"head_{head_index}"
                    comparisons[comparison_key]["layers"][layer][metric][
                        head_name
                    ] = {}

                    for group in GROUPS:
                        carry_values = [
                            get_group_value(
                                obs, layer, head_index, group, metric
                            )
                            for obs in carry
                        ]

                        no_carry_values = [
                            get_group_value(
                                obs, layer, head_index, group, metric
                            )
                            for obs in no_carry
                        ]

                        observed_difference = (
                            mean(carry_values) - mean(no_carry_values)
                        )

                        bootstrap_differences = []

                        for _ in range(BOOTSTRAP_ITERATIONS):
                            sample_carry = [
                                carry_values[rng.randrange(len(carry_values))]
                                for _ in range(len(carry_values))
                            ]

                            sample_no_carry = [
                                no_carry_values[
                                    rng.randrange(len(no_carry_values))
                                ]
                                for _ in range(len(no_carry_values))
                            ]

                            bootstrap_differences.append(
                                mean(sample_carry) - mean(sample_no_carry)
                            )

                        bootstrap_differences.sort()

                        comparisons[comparison_key]["layers"][layer][
                            metric
                        ][head_name][group] = {
                            "mean_difference": observed_difference,
                            "bootstrap_ci": {
                                "lower_95": percentile(
                                    bootstrap_differences, 0.025
                                ),
                                "upper_95": percentile(
                                    bootstrap_differences, 0.975
                                ),
                            },
                        }

    return comparisons


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    with INPUT.open(encoding="utf-8") as f:
        data = json.load(f)

    observations = data["observations"]

    if len(observations) != 100:
        raise ValueError(
            f"Attese 100 osservazioni, trovate {len(observations)}"
        )

    strata = {}

    for observation in observations:
        name = get_stratum(observation)
        strata.setdefault(name, []).append(observation)

    rng = random.Random(SEED)

    results = {
        "experiment": "attention_head_robustness_v1",
        "source": str(INPUT),
        "total_observations": len(observations),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": SEED,
        "confidence_level": 0.95,
        "metrics": METRICS,
        "groups": GROUPS,
        "layers": LAYERS,
        "methodology": {
            "layer_delta": "paired layer_5 minus layer_0",
            "carry_comparison": "independent bootstrap means",
            "length_comparisons": (
                "Not interpreted as causal because source splits differ."
            ),
            "limitations": [
                "Analysis uses the existing 100 observations.",
                "Short and long groups come from different source splits.",
                "Bootstrap intervals are descriptive, not proof of causality.",
                "No multiple-comparison correction is applied."
            ],
        },
        "strata": {},
    }

    print("\nATTENTION HEAD ROBUSTNESS V1")
    print("=" * 72)

    for name, group_observations in sorted(strata.items()):
        print(f"\n{name}: n={len(group_observations)}")

        analysis = analyze_stratum(group_observations, rng)
        results["strata"][name] = analysis

        for head_index in range(6):
            head_name = f"head_{head_index}"
            eq_delta = analysis["layer_delta"]["total_attention"][
                head_name
            ]["EQ"]

            print(
                f"  {head_name}: "
                f"EQ delta={eq_delta['mean']:+.5f} "
                f"CI=[{eq_delta['bootstrap_ci']['lower_95']:+.5f}, "
                f"{eq_delta['bootstrap_ci']['upper_95']:+.5f}]"
            )

    results["carry_comparisons"] = compare_carry_groups(strata, rng)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("ROBUSTNESS ANALYSIS COMPLETED")
    print("Observations:", len(observations))
    print("Bootstrap iterations:", BOOTSTRAP_ITERATIONS)
    print("Report:", OUTPUT)
    print("=" * 72)


if __name__ == "__main__":
    main()
