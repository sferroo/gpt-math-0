
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt0.arithmetic_model import ArithmeticGPT
from gpt0.numeric_tokenizer import NumericTokenizer


DATA_DIR = PROJECT_ROOT / "data" / "arithmetic" / "v1"
RESULTS_DIR = PROJECT_ROOT / "results" / "arithmetic"

OUTPUT = RESULTS_DIR / "eq_attention_dataset_v1.json"

SEED = 42
SAMPLE_SEED = 2026
SAMPLES_PER_STRATUM = 25
LAYERS = [0, 5]

MODEL_CONFIG = {
    "block_size": 32,
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "dropout": 0.0,
}

STRATA = [
    ("short", True, "test_id.jsonl"),
    ("short", False, "test_id.jsonl"),
    ("long", True, "test_ood.jsonl"),
    ("long", False, "test_ood.jsonl"),
]


def load_jsonl(path):
    records = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def select_samples():
    rng = random.Random(SAMPLE_SEED)
    selected = []

    for length_group, carry, filename in STRATA:
        path = DATA_DIR / filename
        records = load_jsonl(path)

        candidates = [
            record
            for record in records
            if bool(record["has_carry"]) == carry
        ]

        if len(candidates) < SAMPLES_PER_STRATUM:
            raise ValueError(
                f"Not enough samples for {length_group}, "
                f"carry={carry}: {len(candidates)}"
            )

        chosen = rng.sample(candidates, SAMPLES_PER_STRATUM)

        for record in chosen:
            a = int(record["a"])
            b = int(record["b"])

            actual_length = max(
                len(str(a)),
                len(str(b))
            )

            if length_group == "short" and actual_length > 2:
                raise ValueError("Unexpected short operand")

            if length_group == "long" and actual_length < 3:
                raise ValueError("Unexpected long operand")

            selected.append({
                "a": a,
                "b": b,
                "result": int(record["result"]),
                "has_carry": carry,
                "length_group": length_group,
                "source": filename,
            })

    return selected


def get_token_groups(tokens):
    add_index = tokens.index(NumericTokenizer.ADD)
    eq_index = tokens.index(NumericTokenizer.EQ)

    return {
        "BOS": [0],
        "operand_a": list(range(1, add_index)),
        "ADD": [add_index],
        "operand_b": list(range(add_index + 1, eq_index)),
        "EQ": [eq_index],
    }


def capture_attention(model, prompt, device):
    captured = {}
    originals = []

    for layer_index, block in enumerate(model.blocks):
        module = block.attn.attn
        original = module.forward

        def make_forward(original_forward, index):
            def diagnostic_forward(*args, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False

                output, weights = original_forward(*args, **kwargs)

                captured[index] = (
                    weights.detach().float().cpu()
                )

                return output, weights

            return diagnostic_forward

        module.forward = make_forward(original, layer_index)
        originals.append((module, original))

    try:
        with torch.inference_mode():
            model(prompt)
    finally:
        for module, original in originals:
            module.forward = original

    return captured


def analyze_eq_row(matrix, groups):
    # Last prompt token is EQ.
    row = matrix[-1]

    metrics = {}

    for name, positions in groups.items():
        values = [float(row[position]) for position in positions]

        metrics[name] = {
            "total_attention": sum(values),
            "mean_per_token": (
                mean(values) if values else 0.0
            ),
            "token_count": len(values),
        }

    return metrics


def calculate_statistics(values):
    n = len(values)
    avg = mean(values)

    if n > 1:
        sd = stdev(values)
        margin = 1.96 * sd / (n ** 0.5)
    else:
        sd = 0.0
        margin = 0.0

    return {
        "n": n,
        "mean": avg,
        "std": sd,
        "ci95_approx": [
            avg - margin,
            avg + margin,
        ],
    }


def main():
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    samples = select_samples()

    print("Selected samples:", len(samples))
    print("Device:", device)

    model = ArithmeticGPT(**MODEL_CONFIG)
    model.to(device)
    model.eval()

    observations = []

    for index, sample in enumerate(samples, start=1):
        a = sample["a"]
        b = sample["b"]

        token_ids = NumericTokenizer.encode_input(a, b)

        prompt = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=device,
        )

        captured = capture_attention(model, prompt, device)

        tokens = token_ids
        groups = get_token_groups(tokens)

        observation = {
            "a": a,
            "b": b,
            "result": sample["result"],
            "has_carry": sample["has_carry"],
            "length_group": sample["length_group"],
            "source": sample["source"],
            "layers": {},
        }

        for layer_index in LAYERS:
            weights = captured[layer_index][0]

            head_metrics = []

            for head_index in range(weights.shape[0]):
                matrix = weights[head_index].numpy()

                metrics = analyze_eq_row(matrix, groups)

                head_metrics.append({
                    "head": head_index,
                    "groups": metrics,
                })

            observation["layers"][f"layer_{layer_index}"] = (
                head_metrics
            )

        observations.append(observation)

        if index % 10 == 0:
            print(f"Processed {index}/{len(samples)}")

    # Aggregate each example first across heads.
    # Then calculate statistics across examples.
    aggregate = defaultdict(lambda: defaultdict(list))

    for observation in observations:
        stratum = (
            f"{observation['length_group']}_"
            f"{'carry' if observation['has_carry'] else 'no_carry'}"
        )

        for layer_name, heads in observation["layers"].items():
            group_names = heads[0]["groups"].keys()

            for group_name in group_names:
                total = mean(
                    h["groups"][group_name]["total_attention"]
                    for h in heads
                )

                per_token = mean(
                    h["groups"][group_name]["mean_per_token"]
                    for h in heads
                )

                aggregate[
                    (stratum, layer_name, group_name)
                ]["total"].append(total)

                aggregate[
                    (stratum, layer_name, group_name)
                ]["per_token"].append(per_token)

    summary = {}

    for (stratum, layer_name, group_name), values in aggregate.items():
        summary.setdefault(stratum, {})
        summary[stratum].setdefault(layer_name, {})

        summary[stratum][layer_name][group_name] = {
            "total_attention": calculate_statistics(
                values["total"]
            ),
            "mean_per_token": calculate_statistics(
                values["per_token"]
            ),
        }

    # Compute layer 5 minus layer 0 for each stratum.
    changes = {}

    for stratum, layer_data in summary.items():
        changes[stratum] = {}

        for group_name in layer_data["layer_0"]:
            first = layer_data["layer_0"][group_name]
            last = layer_data["layer_5"][group_name]

            changes[stratum][group_name] = {
                "total_attention_change": (
                    last["total_attention"]["mean"]
                    - first["total_attention"]["mean"]
                ),
                "per_token_change": (
                    last["mean_per_token"]["mean"]
                    - first["mean_per_token"]["mean"]
                ),
            }

    report = {
        "experiment": "eq_attention_dataset_v1",
        "model_initialization": "random",
        "checkpoint": None,
        "model_seed": SEED,
        "sample_seed": SAMPLE_SEED,
        "device": str(device),
        "model_config": MODEL_CONFIG,
        "layers_compared": LAYERS,
        "samples_per_stratum": SAMPLES_PER_STRATUM,
        "total_samples": len(observations),
        "strata": {
            "short_carry": 25,
            "short_no_carry": 25,
            "long_carry": 25,
            "long_no_carry": 25,
        },
        "summary": summary,
        "layer_5_minus_layer_0": changes,
        "observations": observations,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("EQ ATTENTION DATASET ANALYSIS COMPLETED")
    print("Samples:", len(observations))
    print("Report:", OUTPUT)
    print("=" * 65)

    for stratum, layer_data in summary.items():
        print(f"\n{stratum.upper()}")

        for layer_name in ["layer_0", "layer_5"]:
            print(f"\n{layer_name}")

            for group_name, metrics in layer_data[layer_name].items():
                value = metrics["total_attention"]["mean"]
                sd = metrics["total_attention"]["std"]

                print(
                    f"{group_name:<12} "
                    f"mean={value:.5f} std={sd:.5f}"
                )


if __name__ == "__main__":
    main()
