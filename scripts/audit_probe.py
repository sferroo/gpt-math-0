
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpt0.checkpoints import load_checkpoint
from gpt0.model import GPT0

CHECKPOINT = ROOT / "checkpoints" / "best.pt"
TOKENIZER_PATH = ROOT / "data" / "tokenizer.json"

DATASETS = {
    "train": ROOT / "data" / "processed" / "train_sequences.jsonl",
    "validation": ROOT / "data" / "processed" / "val_sequences.jsonl",
}

REPORT_PATH = ROOT / "reports" / "audit_probe.json"

PAD, BOS, EOS = 0, 2, 3
PROBLEM, SOLUTION, ANSWER = 4, 5, 6

SAMPLE_SIZE = 100
SEED = 42


def load_records(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def repeated_ngram_onset(ids, n=4, repetitions=3):
    seen = Counter()

    for i in range(len(ids) - n + 1):
        gram = tuple(ids[i:i + n])
        seen[gram] += 1

        if seen[gram] >= repetitions:
            return i

    return None


def token_probability(logits, target):
    probabilities = torch.softmax(logits.float(), dim=-1)
    return probabilities.gather(
        -1, target.unsqueeze(-1)
    ).squeeze(-1)


def inspect_example(model, tokenizer, record, device):
    tokens = record["tokens"]

    solution_pos = tokens.index(SOLUTION)
    answer_pos = tokens.index(ANSWER)
    eos_pos = len(tokens) - 1

    x = torch.tensor(
        [tokens[:-1]], dtype=torch.long, device=device
    )
    y = torch.tensor(
        [tokens[1:]], dtype=torch.long, device=device
    )

    with torch.inference_mode():
        logits, _ = model(x)

        losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="none",
        ).reshape_as(y)

        predictions = logits.argmax(dim=-1)

        # Numeric answer tokens: exclude ANSWER and EOS.
        numeric_start = answer_pos
        numeric_end = eos_pos - 1

        numeric_targets = y[0, numeric_start:numeric_end]
        numeric_predictions = predictions[
            0, numeric_start:numeric_end
        ]

        numeric_losses = losses[
            0, numeric_start:numeric_end
        ]

        numeric_probabilities = token_probability(
            logits[0, numeric_start:numeric_end],
            numeric_targets,
        )

        eos_target_index = eos_pos - 1
        eos_probability = torch.softmax(
            logits[0, eos_target_index].float(),
            dim=-1,
        )[EOS].item()

        solution_losses = losses[
            0, solution_pos:answer_pos
        ]

        answer_marker_probability = torch.softmax(
            logits[0, answer_pos - 1].float(),
            dim=-1,
        )[ANSWER].item()

        answer_marker_loss = losses[
            0, answer_pos - 1
        ].item()

    solution_ids = tokens[solution_pos + 1:answer_pos]
    answer_ids = tokens[answer_pos + 1:eos_pos]

    solution_text = tokenizer.decode(solution_ids)
    answer_text = tokenizer.decode(answer_ids)

    numbers = re.findall(
        r"-?\d[\d,]*(?:\.\d+)?",
        solution_text,
    )

    last_solution_number = (
        numbers[-1].replace(",", "") if numbers else None
    )

    answer_number = answer_text.strip().replace(",", "")

    # Reconstruct the evaluator-style prompt.
    question_ids = tokens[2:solution_pos]
    question_text = tokenizer.decode(question_ids)

    evaluation_prompt = (
        "[BOS][PROBLEM] "
        + question_text
        + " [SOLUTION]"
    )

    evaluation_ids = tokenizer.encode(evaluation_prompt).ids
    training_prefix = tokens[:solution_pos + 1]

    common_length = min(
        len(evaluation_ids),
        len(training_prefix),
    )

    first_difference = next(
        (
            i for i in range(common_length)
            if evaluation_ids[i] != training_prefix[i]
        ),
        None,
    )

    if first_difference is None and (
        len(evaluation_ids) != len(training_prefix)
    ):
        first_difference = common_length

    return {
        "id": record["id"],
        "total_tokens": len(tokens),
        "question_tokens": solution_pos - 2,
        "solution_tokens": len(solution_ids),
        "answer_tokens": len(answer_ids),
        "solution_loss": solution_losses.mean().item(),
        "numeric_loss": (
            numeric_losses.mean().item()
            if len(numeric_losses) else None
        ),
        "numeric_token_accuracy": (
            (numeric_predictions == numeric_targets)
            .float().mean().item()
            if len(numeric_targets) else None
        ),
        "numeric_target_probability_mean": (
            numeric_probabilities.mean().item()
            if len(numeric_probabilities) else None
        ),
        "numeric_target_probability_min": (
            numeric_probabilities.min().item()
            if len(numeric_probabilities) else None
        ),
        "eos_probability_at_correct_position": eos_probability,
        "answer_marker_probability_at_correct_position":
            answer_marker_probability,
        "answer_marker_loss": answer_marker_loss,
        "answer_text": answer_text,
        "last_number_in_solution": last_solution_number,
        "answer_matches_last_solution_number": (
            answer_number == last_solution_number
            if last_solution_number is not None else None
        ),
        "training_prefix_length": len(training_prefix),
        "evaluation_prefix_length": len(evaluation_ids),
        "prompt_prefix_matches": evaluation_ids == training_prefix,
        "first_prompt_difference": first_difference,
    }


def test_causal_mask(model, record, device):
    tokens = record["tokens"][:min(80, len(record["tokens"]))]

    if len(tokens) < 8:
        return {"tested": False}

    original = torch.tensor(
        [tokens], dtype=torch.long, device=device
    )

    modified_tokens = tokens.copy()
    position = len(tokens) // 2

    modified_tokens[position] = (
        modified_tokens[position] + 1
    ) % model.token_emb.num_embeddings

    modified = torch.tensor(
        [modified_tokens], dtype=torch.long, device=device
    )

    with torch.inference_mode():
        original_logits, _ = model(original)
        modified_logits, _ = model(modified)

    # Earlier positions must not depend on a future token.
    difference = (
        original_logits[:, :position, :]
        - modified_logits[:, :position, :]
    ).abs().max().item()

    return {
        "tested": True,
        "modified_position": position,
        "max_difference_before_modified_position": difference,
        "causal_mask_passed": difference < 1e-5,
    }


def dataset_statistics(records):
    gsm = [r for r in records if r["type"] == "gsm8k"]
    theory = [r for r in records if r["type"] != "gsm8k"]

    gsm_lengths = [len(r["tokens"]) for r in gsm]
    theory_lengths = [len(r["tokens"]) for r in theory]

    def average(values):
        return sum(values) / len(values) if values else 0

    gsm_mean = average(gsm_lengths)
    theory_mean = average(theory_lengths)

    # Expected token share under 90/10 sequence sampling.
    theory_token_share = (
        0.10 * theory_mean /
        (0.10 * theory_mean + 0.90 * gsm_mean)
        if gsm_mean and theory_mean else None
    )

    gsm_draws = 3600
    unique_expected = (
        len(gsm) * (
            1 - (1 - 1 / len(gsm)) ** gsm_draws
        )
        if gsm else 0
    )

    return {
        "gsm8k_sequences": len(gsm),
        "theory_sequences": len(theory),
        "gsm8k_mean_length": gsm_mean,
        "theory_mean_length": theory_mean,
        "expected_theory_token_share": theory_token_share,
        "expected_unique_gsm8k_examples_after_3600_draws":
            unique_expected,
        "expected_gsm8k_coverage_percent": (
            unique_expected / len(gsm) * 100 if gsm else 0
        ),
        "theory_weight_per_sequence": (
            0.10 / len(theory) if theory else None
        ),
        "gsm8k_weight_per_sequence": (
            0.90 / len(gsm) if gsm else None
        ),
    }


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("GPT-0 READ-ONLY AUDIT PROBE")
    print("Device:", device)

    checkpoint = load_checkpoint(
        CHECKPOINT,
        map_location=device,
    )

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    model_config = checkpoint["config"]["model"]

    model = GPT0(**model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print("Checkpoint step:", checkpoint["step"])

    report = {
        "checkpoint_step": checkpoint["step"],
        "datasets": {},
        "causal_mask_test": None,
    }

    all_records = {}

    for name, path in DATASETS.items():
        records = load_records(path)
        all_records[name] = records

        stats = dataset_statistics(records)
        report["datasets"][name] = {
            "statistics": stats,
        }

        print(f"\n{name.upper()} statistics:")
        print(json.dumps(stats, indent=2))

        examples = [
            r for r in records
            if r["type"] == "gsm8k"
        ]

        rng = random.Random(SEED)
        rng.shuffle(examples)
        examples = examples[:SAMPLE_SIZE]

        results = [
            inspect_example(model, tokenizer, r, device)
            for r in examples
        ]

        report["datasets"][name]["examples"] = results

        numeric_accuracies = [
            r["numeric_token_accuracy"]
            for r in results
            if r["numeric_token_accuracy"] is not None
        ]

        numeric_probabilities = [
            r["numeric_target_probability_mean"]
            for r in results
            if r["numeric_target_probability_mean"] is not None
        ]

        report["datasets"][name]["summary"] = {
            "examples_evaluated": len(results),
            "numeric_token_accuracy_mean": (
                sum(numeric_accuracies) / len(numeric_accuracies)
                if numeric_accuracies else None
            ),
            "numeric_target_probability_mean": (
                sum(numeric_probabilities) / len(numeric_probabilities)
                if numeric_probabilities else None
            ),
            "prompt_exact_match_count": sum(
                r["prompt_prefix_matches"] for r in results
            ),
            "answer_matches_last_solution_number_count": sum(
                r["answer_matches_last_solution_number"] is True
                for r in results
            ),
            "mean_eos_probability": sum(
                r["eos_probability_at_correct_position"]
                for r in results
            ) / len(results) if results else None,
            "mean_answer_marker_probability": sum(
                r["answer_marker_probability_at_correct_position"]
                for r in results
            ) / len(results) if results else None,
        }

        print(f"\n{name.upper()} summary:")
        print(json.dumps(
            report["datasets"][name]["summary"],
            indent=2,
        ))

    first_train = next(
        r for r in all_records["train"]
        if r["type"] == "gsm8k"
    )

    report["causal_mask_test"] = test_causal_mask(
        model, first_train, device
    )

    print("\nCausal mask test:")
    print(json.dumps(report["causal_mask_test"], indent=2))

    # Inspect the existing generation report without changing it.
    generation_report = (
        ROOT / "reports" / "gsm8k_validation_results.json"
    )

    if generation_report.exists():
        data = json.loads(
            generation_report.read_text(encoding="utf-8")
        )

        tokenizer_results = []

        for result in data["results"]:
            ids = tokenizer.encode(
                result.get("generated", "")
            ).ids

            tokenizer_results.append(
                repeated_ngram_onset(ids)
            )

        valid_onsets = [
            x for x in tokenizer_results if x is not None
        ]

        report["generation_loop_analysis"] = {
            "examples": len(tokenizer_results),
            "examples_with_detected_loop": len(valid_onsets),
            "mean_loop_start_token": (
                sum(valid_onsets) / len(valid_onsets)
                if valid_onsets else None
            ),
            "loop_start_positions": tokenizer_results,
        }

        print("\nGeneration loop analysis:")
        print(json.dumps(
            report["generation_loop_analysis"],
            indent=2,
        ))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nAudit completed.")
    print("Report:", REPORT_PATH)


if __name__ == "__main__":
    main()
