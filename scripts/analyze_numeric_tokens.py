
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpt0.checkpoints import load_checkpoint
from gpt0.model import GPT0


DATASET_PATH = ROOT / "data" / "processed" / "val_sequences.jsonl"
TOKENIZER_PATH = ROOT / "data" / "tokenizer.json"

CHECKPOINTS = [
    ROOT / "checkpoints" / "best.pt",
    ROOT / "checkpoints" / "coverage" / "best.pt",
]

REPORT_DIR = ROOT / "reports"

PAD_ID = 0
BOS_ID = 2
EOS_ID = 3
PROBLEM_ID = 4
SOLUTION_ID = 5
ANSWER_ID = 6

EXPECTED_VALIDATION = 748
TOP_K = 10


def load_examples():
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        records = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    examples = [
        record for record in records
        if record.get("type") == "gsm8k"
    ]

    if len(examples) != EXPECTED_VALIDATION:
        raise RuntimeError(
            f"Expected {EXPECTED_VALIDATION} examples, "
            f"found {len(examples)}."
        )

    return examples


def split_answer(tokens):
    if tokens[0] != BOS_ID or tokens[-1] != EOS_ID:
        raise ValueError("Invalid BOS/EOS sequence.")

    answer_pos = tokens.index(ANSWER_ID)
    return tokens[answer_pos + 1:-1]


def decode_ids(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=False)


def load_model(checkpoint_path, device):
    checkpoint = load_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )

    model = GPT0(
        **checkpoint["config"]["model"],
        pad_token_id=PAD_ID,
    ).to(device)

    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    step = checkpoint.get("step")
    del checkpoint

    return model, step


def summarize_answer_tokenization(examples, tokenizer):
    answer_lengths = Counter()
    token_frequency = Counter()
    digit_character_counts = Counter()

    for record in examples:
        answer_ids = split_answer(record["tokens"])
        answer_text = decode_ids(tokenizer, answer_ids)

        answer_lengths[len(answer_ids)] += 1
        token_frequency.update(answer_ids)

        for char in answer_text:
            if char.isdigit():
                digit_character_counts[char] += 1

    return answer_lengths, token_frequency, digit_character_counts


@torch.inference_mode()
def analyze_checkpoint(checkpoint_path, examples, tokenizer, device):
    print("\n" + "=" * 72, flush=True)
    print(f"CHECKPOINT: {checkpoint_path.relative_to(ROOT)}", flush=True)
    print("=" * 72, flush=True)

    model, training_step = load_model(checkpoint_path, device)

    position_count = 0
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0

    token_rank_counter = Counter()
    per_length = defaultdict(lambda: {
        "positions": 0,
        "top1": 0,
        "top5": 0,
        "top10": 0,
    })

    sample_details = []

    for index, record in enumerate(examples, start=1):
        tokens = record["tokens"]
        answer_pos = tokens.index(ANSWER_ID)
        answer_ids = tokens[answer_pos + 1:-1]

        if not answer_ids:
            continue

        x = torch.tensor(
            [tokens[:-1]],
            dtype=torch.long,
            device=device,
        )

        logits, _ = model(x)

        example_all_top1 = True

        for offset, target_id in enumerate(answer_ids):
            target_position = answer_pos + 1 + offset
            prediction_position = target_position - 1

            next_logits = logits[0, prediction_position].float().clone()

            for forbidden_id in (
                PAD_ID,
                BOS_ID,
                PROBLEM_ID,
                SOLUTION_ID,
                ANSWER_ID,
            ):
                next_logits[forbidden_id] = float("-inf")

            top_ids = torch.topk(
                next_logits,
                k=TOP_K,
            ).indices.tolist()

            hit1 = target_id == top_ids[0]
            hit5 = target_id in top_ids[:5]
            hit10 = target_id in top_ids[:10]

            position_count += 1
            top1_correct += int(hit1)
            top5_correct += int(hit5)
            top10_correct += int(hit10)

            rank = (
                top_ids.index(target_id) + 1
                if target_id in top_ids else None
            )

            token_rank_counter[
                rank if rank is not None else "outside_top10"
            ] += 1

            length = len(answer_ids)
            per_length[length]["positions"] += 1
            per_length[length]["top1"] += int(hit1)
            per_length[length]["top5"] += int(hit5)
            per_length[length]["top10"] += int(hit10)

            example_all_top1 = example_all_top1 and hit1

            if len(sample_details) < 30:
                sample_details.append({
                    "example_id": record.get("id", index),
                    "answer_position": offset,
                    "gold_token_id": target_id,
                    "gold_piece": tokenizer.id_to_token(target_id),
                    "gold_rank": rank,
                    "top_predictions": [
                        {
                            "id": token_id,
                            "piece": tokenizer.id_to_token(token_id),
                            "is_gold": token_id == target_id,
                        }
                        for token_id in top_ids
                    ],
                })

        if index % 100 == 0 or index == len(examples):
            print(f"[PROGRESS] {index}/{len(examples)}", flush=True)

    results = {
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "training_step": training_step,
        "dataset": "GSM8K validation",
        "examples": len(examples),
        "answer_token_positions": position_count,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "top10_correct": top10_correct,
        "top1_accuracy": top1_correct / max(position_count, 1),
        "top5_accuracy": top5_correct / max(position_count, 1),
        "top10_accuracy": top10_correct / max(position_count, 1),
        "accuracy_by_answer_token_length": {},
        "token_rank_counts": {
            str(key): value
            for key, value in token_rank_counter.items()
        },
        "sample_topk_predictions": sample_details,
    }

    for length, values in sorted(per_length.items()):
        results["accuracy_by_answer_token_length"][str(length)] = {
            **values,
            "top1_accuracy": values["top1"] / values["positions"],
            "top5_accuracy": values["top5"] / values["positions"],
            "top10_accuracy": values["top10"] / values["positions"],
        }

    print("\nTEACHER-FORCED ANSWER PREDICTION", flush=True)
    print(f"Positions: {position_count}", flush=True)
    print(f"Top-1: {results['top1_accuracy']:.2%}", flush=True)
    print(f"Top-5: {results['top5_accuracy']:.2%}", flush=True)
    print(f"Top-10: {results['top10_accuracy']:.2%}", flush=True)

    print("\nACCURACY BY ANSWER TOKEN LENGTH", flush=True)

    for length, values in sorted(per_length.items()):
        print(
            f"length={length:>2} | "
            f"positions={values['positions']:>4} | "
            f"top1={values['top1'] / values['positions']:.2%} | "
            f"top5={values['top5'] / values['positions']:.2%} | "
            f"top10={values['top10'] / values['positions']:.2%}",
            flush=True,
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORT_DIR / (
        f"numeric_prediction_{checkpoint_path.parent.name}_"
        f"{checkpoint_path.stem}.json"
    )

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] {report_path.relative_to(ROOT)}", flush=True)

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


def main():
    print("=" * 72, flush=True)
    print("GPT-0 MATH - NUMERIC TOKENIZATION AUDIT", flush=True)
    print("=" * 72, flush=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"[INFO] Device: {device}", flush=True)

    examples = load_examples()
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    print(f"[INFO] Validation examples: {len(examples)}", flush=True)
    print(
        f"[INFO] Vocabulary size: {tokenizer.get_vocab_size()}",
        flush=True,
    )

    lengths, frequencies, digits = summarize_answer_tokenization(
        examples,
        tokenizer,
    )

    print("\nANSWER TOKENIZATION DISTRIBUTION", flush=True)

    for length, count in sorted(lengths.items()):
        print(f"{length} token(s): {count} answers", flush=True)

    print("\nMOST FREQUENT ANSWER TOKENS", flush=True)

    for token_id, count in frequencies.most_common(25):
        print(
            f"id={token_id:>5} | "
            f"piece={tokenizer.id_to_token(token_id)!r} | "
            f"decoded={decode_ids(tokenizer, [token_id])!r} | "
            f"count={count}",
            flush=True,
        )

    print("\nDIGIT CHARACTER COUNTS", flush=True)
    print(dict(sorted(digits.items())), flush=True)

    all_results = []

    for checkpoint_path in CHECKPOINTS:
        all_results.append(
            analyze_checkpoint(
                checkpoint_path,
                examples,
                tokenizer,
                device,
            )
        )

    print("\n" + "=" * 72, flush=True)
    print("FINAL COMPARISON", flush=True)

    for result in all_results:
        print(f"\n{result['checkpoint']}", flush=True)
        print(f"Top-1: {result['top1_accuracy']:.2%}", flush=True)
        print(f"Top-5: {result['top5_accuracy']:.2%}", flush=True)
        print(f"Top-10: {result['top10_accuracy']:.2%}", flush=True)

    print("\n[COMPLETE] No training performed.", flush=True)
    print("[INFO] No checkpoint or dataset modified.", flush=True)


if __name__ == "__main__":
    main()
