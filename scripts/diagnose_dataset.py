
import json
import sys
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpt0.checkpoints import load_checkpoint
from gpt0.model import GPT0


CHECKPOINT_PATH = ROOT / "checkpoints" / "best.pt"
TOKENIZER_PATH = ROOT / "data" / "tokenizer.json"

DATASETS = {
    "TRAIN": ROOT / "data" / "processed" / "train_sequences.jsonl",
    "VALIDATION": ROOT / "data" / "processed" / "val_sequences.jsonl",
}

SAMPLE_SIZE = 100
SEED = 42

SOLUTION_ID = 5
ANSWER_ID = 6
EOS_ID = 3


def load_examples(path):
    examples = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            if record["type"] == "gsm8k":
                examples.append(record)

    return examples


def select_examples(examples):
    # Deterministic selection without changing the dataset.
    import random

    rng = random.Random(SEED)
    selected = examples.copy()
    rng.shuffle(selected)

    return selected[:min(SAMPLE_SIZE, len(selected))]


def evaluate_example(model, tokens, device):
    solution_pos = tokens.index(SOLUTION_ID)
    answer_pos = tokens.index(ANSWER_ID)

    x = torch.tensor(
        [tokens[:-1]],
        dtype=torch.long,
        device=device,
    )

    targets = torch.tensor(
        [tokens[1:]],
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        logits, _ = model(x)

        losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)

        predictions = logits.argmax(dim=-1)

        solution_losses = losses[0, solution_pos:answer_pos]
        answer_losses = losses[0, answer_pos:len(tokens) - 1]

        solution_targets = targets[0, solution_pos:answer_pos]
        answer_targets = targets[0, answer_pos:len(tokens) - 1]

        solution_predictions = predictions[0, solution_pos:answer_pos]
        answer_predictions = predictions[0, answer_pos:len(tokens) - 1]

        return {
            "solution_loss": solution_losses.mean().item(),
            "answer_loss": answer_losses.mean().item(),
            "solution_accuracy": (
                solution_predictions == solution_targets
            ).float().mean().item(),
            "answer_accuracy": (
                answer_predictions == answer_targets
            ).float().mean().item(),
            "solution_tokens": solution_targets.numel(),
            "answer_tokens": answer_targets.numel(),
        }


def summarize(name, results):
    print(f"\n========== {name} ==========")
    print(f"Examples evaluated: {len(results)}")

    for metric in (
        "solution_loss",
        "answer_loss",
        "solution_accuracy",
        "answer_accuracy",
    ):
        values = [result[metric] for result in results]
        value = mean(values)

        if "accuracy" in metric:
            print(f"{metric:22}: {value * 100:.2f}%")
        else:
            print(f"{metric:22}: {value:.4f}")

    solution_tokens = sum(r["solution_tokens"] for r in results)
    answer_tokens = sum(r["answer_tokens"] for r in results)

    print(f"Total solution tokens: {solution_tokens}")
    print(f"Total answer tokens:   {answer_tokens}")


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    checkpoint = load_checkpoint(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model_config = checkpoint["config"]["model"]

    model = GPT0(**model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"Checkpoint step: {checkpoint['step']}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")

    # Confirm tokenizer is available and compatible.
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    print(f"Tokenizer vocabulary: {tokenizer.get_vocab_size()}")

    all_results = {}

    for name, path in DATASETS.items():
        examples = load_examples(path)
        selected = select_examples(examples)

        if not selected:
            print(f"No GSM8K examples found in {path}")
            continue

        print(f"\nEvaluating {name}: {len(selected)} examples...")

        results = []

        for index, example in enumerate(selected, start=1):
            result = evaluate_example(
                model,
                example["tokens"],
                device,
            )

            results.append(result)

            if index % 25 == 0:
                print(f"  Processed {index}/{len(selected)}")

        all_results[name] = results
        summarize(name, results)

    if "TRAIN" in all_results and "VALIDATION" in all_results:
        train_loss = mean(
            r["solution_loss"] for r in all_results["TRAIN"]
        )
        val_loss = mean(
            r["solution_loss"] for r in all_results["VALIDATION"]
        )

        print("\n========== GENERALIZATION GAP ==========")
        print(f"Train solution loss:      {train_loss:.4f}")
        print(f"Validation solution loss: {val_loss:.4f}")
        print(f"Difference:                {val_loss - train_loss:+.4f}")

    print("\nDiagnostic completed.")


if __name__ == "__main__":
    main()
