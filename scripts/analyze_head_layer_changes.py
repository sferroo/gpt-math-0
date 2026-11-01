
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results/arithmetic/head_specialization_v1.json"
OUTPUT = ROOT / "results/arithmetic/head_layer_changes_v1.json"

GROUPS = ["BOS", "operand_a", "ADD", "operand_b", "EQ"]


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Report non trovato: {INPUT}")

    with INPUT.open(encoding="utf-8") as f:
        data = json.load(f)

    results = {
        "experiment": "head_layer_changes_v1",
        "source": str(INPUT),
        "comparison": "layer_5_minus_layer_0",
        "strata": {},
    }

    for stratum, stratum_data in data["strata"].items():
        layer0 = stratum_data["layer_0"]["head_means"]
        layer5 = stratum_data["layer_5"]["head_means"]

        changes = {}

        print("\n" + "=" * 76)
        print(f"STRATUM: {stratum}")
        print("=" * 76)

        print(
            f"{'Head':<10}" +
            "".join(f"{group:>12}" for group in GROUPS)
        )

        for head_index in range(6):
            head_name = f"head_{head_index}"
            changes[head_name] = {}

            values = []

            for group in GROUPS:
                before = layer0[head_name][group]["mean"]
                after = layer5[head_name][group]["mean"]
                delta = after - before

                changes[head_name][group] = {
                    "layer_0": before,
                    "layer_5": after,
                    "delta": delta,
                }

                values.append(delta)

            print(
                f"{head_name:<10}" +
                "".join(f"{value:>12.4f}" for value in values)
            )

        all_changes = [
            {
                "head": head,
                "group": group,
                "delta": values["delta"],
            }
            for head, groups in changes.items()
            for group, values in groups.items()
        ]

        largest_increase = max(
            all_changes, key=lambda item: item["delta"]
        )
        largest_decrease = min(
            all_changes, key=lambda item: item["delta"]
        )

        print("\nLargest increase:")
        print(largest_increase)

        print("\nLargest decrease:")
        print(largest_decrease)

        results["strata"][stratum] = {
            "n": stratum_data["n"],
            "head_changes": changes,
            "largest_increase": largest_increase,
            "largest_decrease": largest_decrease,
        }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 76)
    print("HEAD LAYER TRANSITION COMPLETED")
    print("Report:", OUTPUT)
    print("=" * 76)


if __name__ == "__main__":
    main()
