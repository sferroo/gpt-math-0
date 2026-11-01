
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/arithmetic/attention_analysis_v1.json"
OUTPUT = ROOT / "results/arithmetic/attention_heatmaps"

OUTPUT.mkdir(parents=True, exist_ok=True)


def plot_layer(example_name, example, layer_index):
    layer_name = f"layer_{layer_index}"
    layer = example["layers"][layer_name]
    tokens = example["tokens"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Attention: {example_name} | {layer_name}",
        fontsize=16
    )

    for head_data, ax in zip(layer["heads"], axes.flat):
        matrix = np.array(head_data["attention_matrix"])

        image = ax.imshow(
            matrix,
            vmin=0,
            vmax=1,
            interpolation="nearest",
            aspect="auto"
        )

        ax.set_title(
            f"Head {head_data['head']} | "
            f"H={head_data['mean_entropy_bits']:.3f} bits"
        )

        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens)
        ax.set_yticklabels(tokens)

        ax.set_xlabel("Key: token osservato")
        ax.set_ylabel("Query: token che presta attenzione")

    fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label="Attention weight",
        shrink=0.8
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_file = OUTPUT / f"{example_name}_layer_{layer_index}.png"
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_file}")


def main():
    with INPUT.open(encoding="utf-8") as f:
        data = json.load(f)

    for example_name, example in data["examples"].items():
        # Confrontiamo primo e ultimo layer.
        plot_layer(example_name, example, 0)
        plot_layer(example_name, example, 5)

    print("\nAttention heatmaps completed.")
    print(f"Output directory: {OUTPUT}")


if __name__ == "__main__":
    main()
