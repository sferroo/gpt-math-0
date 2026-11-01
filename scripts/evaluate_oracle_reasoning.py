
from __future__ import annotations

import json
import re
import sys
from decimal import Decimal, InvalidOperation
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
    ROOT / "checkpoints" / "coverage" / "best.pt",
    ROOT / "checkpoints" / "answer_weighted" / "weight_1x" / "best.pt",
    ROOT / "checkpoints" / "answer_weighted" / "weight_2x" / "best.pt",
    ROOT / "checkpoints" / "answer_weighted" / "weight_5x" / "best.pt",
]

REPORT_DIR = ROOT / "reports"

PAD_ID = 0
BOS_ID = 2
EOS_ID = 3
PROBLEM_ID = 4
SOLUTION_ID = 5
ANSWER_ID = 6

EXPECTED_VALIDATION = 748
MAX_NEW_TOKENS = 128


def load_validation():
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


def split_sequence(tokens):
    if tokens[0] != BOS_ID or tokens[-1] != EOS_ID:
        raise ValueError("Invalid BOS/EOS sequence.")

    solution_pos = tokens.index(SOLUTION_ID)
    answer_pos = tokens.index(ANSWER_ID)

    if not (0 < solution_pos < answer_pos < len(tokens) - 1):
        raise ValueError("Invalid marker ordering.")

    # Problem + correct reasoning + ANSWER marker
    prompt = tokens[:answer_pos + 1]

    # Correct final answer
    gold_answer = tokens[answer_pos + 1:-1]

    return prompt, gold_answer


def normalize_text(text):
    text = text.strip().casefold()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = re.sub(r"\s+", "", text)

    return text.rstrip(".")


def extract_last_number(text):
    cleaned = text.replace(",", "")

    matches = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)",
        cleaned,
    )

    if not matches:
        return None

    try:
        return Decimal(matches[-1]).normalize()
    except InvalidOperation:
        return None


@torch.inference_mode()
def generate_answer(model, prompt, device):
    context = torch.tensor(
        [prompt],
        dtype=torch.long,
        device=device,
    )

    generated = []

    available_context = model.block_size - len(prompt)
    max_tokens = min(MAX_NEW_TOKENS, available_context)

    stopped_by_eos = False

    for _ in range(max_tokens):
        logits, _ = model(context)

        next_logits = logits[:, -1, :].float().clone()

        # Structural tokens that must not appear in the answer.
        for token_id in (
            PAD_ID,
            BOS_ID,
            PROBLEM_ID,
            SOLUTION_ID,
            ANSWER_ID,
        ):
            next_logits[:, token_id] = float("-inf")

        next_token = next_logits.argmax(
            dim=-1,
            keepdim=True,
        )

        token_id = next_token.item()

        if token_id == EOS_ID:
            stopped_by_eos = True
            break

        generated.append(token_id)

        context = torch.cat(
            (context, next_token),
            dim=1,
        )

    return generated, stopped_by_eos, max_tokens


def load_model(checkpoint_path, device):
    checkpoint = load_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )

    model = GPT0(
        **checkpoint["config"]["model"],
        pad_token_id=PAD_ID,
    ).to(device)

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    model.eval()

    step = checkpoint.get("step")

    del checkpoint

    return model, step


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint_path,
    examples,
    tokenizer,
    device,
):
    print("\n" + "=" * 65, flush=True)
    print(f"CHECKPOINT: {checkpoint_path.relative_to(ROOT)}", flush=True)
    print("=" * 65, flush=True)

    model, training_step = load_model(
        checkpoint_path,
        device,
    )

    exact_matches = 0
    numeric_matches = 0
    numeric_comparable = 0
    generated_numbers = 0

    eos_count = 0
    limit_count = 0
    empty_count = 0

    total_generated_tokens = 0
    total_gold_tokens = 0

    example_results = []

    for index, record in enumerate(examples, start=1):
        tokens = record["tokens"]

        prompt, gold_answer_tokens = split_sequence(tokens)

        if len(prompt) >= model.block_size:
            raise RuntimeError(
                f"Prompt exceeds block size: {record.get('id')}"
            )

        generated, stopped_by_eos, max_tokens = generate_answer(
            model,
            prompt,
            device,
        )

        generated_text = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        gold_text = tokenizer.decode(
            gold_answer_tokens,
            skip_special_tokens=True,
        ).strip()

        exact_match = (
            bool(generated_text)
            and normalize_text(generated_text)
            == normalize_text(gold_text)
        )

        generated_number = extract_last_number(generated_text)
        gold_number = extract_last_number(gold_text)

        comparable = (
            generated_number is not None
            and gold_number is not None
        )

        numeric_match = (
            comparable
            and generated_number == gold_number
        )

        exact_matches += int(exact_match)
        numeric_comparable += int(comparable)
        numeric_matches += int(numeric_match)

        generated_numbers += int(
            generated_number is not None
        )

        eos_count += int(stopped_by_eos)
        limit_count += int(not stopped_by_eos)
        empty_count += int(not generated_text)

        total_generated_tokens += len(generated)
        total_gold_tokens += len(gold_answer_tokens)

        example_results.append({
            "id": record.get("id", index),
            "gold_answer": gold_text,
            "generated_answer": generated_text,
            "exact_match": exact_match,
            "gold_numeric": (
                str(gold_number)
                if gold_number is not None else None
            ),
            "generated_numeric": (
                str(generated_number)
                if generated_number is not None else None
            ),
            "numeric_match": (
                bool(numeric_match)
                if comparable else None
            ),
            "stopped_by_eos": stopped_by_eos,
            "generated_tokens": len(generated),
        })

        if index <= 5:
            print(f"\nExample {index}", flush=True)
            print(f"Gold:      {gold_text!r}", flush=True)
            print(f"Generated: {generated_text!r}", flush=True)
            print(f"Exact:     {exact_match}", flush=True)
            print(f"Numeric:   {numeric_match}", flush=True)

        if index % 50 == 0 or index == len(examples):
            print(
                f"[PROGRESS] {index}/{len(examples)}",
                flush=True,
            )

    count = len(examples)

    results = {
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "training_step": training_step,
        "evaluation": "oracle_reasoning_greedy",
        "dataset": "GSM8K validation",
        "examples": count,
        "exact_match": exact_matches,
        "exact_match_rate": exact_matches / count,
        "numeric_match": numeric_matches,
        "numeric_comparable": numeric_comparable,
        "numeric_accuracy_all": numeric_matches / count,
        "numeric_accuracy_comparable": (
            numeric_matches / numeric_comparable
            if numeric_comparable else None
        ),
        "generated_number_count": generated_numbers,
        "stopped_by_eos": eos_count,
        "reached_generation_limit": limit_count,
        "empty_generations": empty_count,
        "average_generated_tokens": (
            total_generated_tokens / count
        ),
        "average_gold_answer_tokens": (
            total_gold_tokens / count
        ),
        "max_new_tokens": MAX_NEW_TOKENS,
        "examples_detail": example_results,
    }

    print("\n---------- RESULTS ----------", flush=True)

    print(
        f"Exact match: {exact_matches}/{count} "
        f"({results['exact_match_rate']:.2%})",
        flush=True,
    )

    print(
        f"Numeric match: {numeric_matches}/{count} "
        f"({results['numeric_accuracy_all']:.2%})",
        flush=True,
    )

    print(
        f"Numeric comparable: {numeric_comparable}/{count}",
        flush=True,
    )

    if numeric_comparable:
        print(
            "Numeric accuracy among comparable outputs: "
            f"{results['numeric_accuracy_comparable']:.2%}",
            flush=True,
        )

    print(f"Generated a number: {generated_numbers}/{count}", flush=True)
    print(f"Stopped at EOS: {eos_count}/{count}", flush=True)
    print(f"Reached generation limit: {limit_count}/{count}", flush=True)
    print(f"Empty generations: {empty_count}/{count}", flush=True)

    print(
        f"Average generated tokens: "
        f"{results['average_generated_tokens']:.2f}",
        flush=True,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_name = (
        f"oracle_reasoning_{checkpoint_path.parent.name}_"
        f"{checkpoint_path.stem}.json"
    )

    report_path = REPORT_DIR / report_name

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[SAVED] {report_path.relative_to(ROOT)}", flush=True)

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


def main():
    print("=" * 65, flush=True)
    print("GPT-0 MATH - ORACLE REASONING EVALUATION", flush=True)
    print("=" * 65, flush=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"[INFO] Device: {device}", flush=True)

    if device.type == "cuda":
        print(
            f"[INFO] GPU: {torch.cuda.get_device_name(0)}",
            flush=True,
        )

    examples = load_validation()

    print(
        f"[INFO] GSM8K validation examples: {len(examples)}",
        flush=True,
    )

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    print(
        "[INFO] Context = problem + gold reasoning + ANSWER marker.",
        flush=True,
    )

    print(
        "[INFO] Greedy generation. No training will be performed.",
        flush=True,
    )

    all_results = []

    for checkpoint_path in CHECKPOINTS:
        result = evaluate_checkpoint(
            checkpoint_path,
            examples,
            tokenizer,
            device,
        )

        all_results.append(result)

    print("\n" + "=" * 65, flush=True)
    print("FINAL COMPARISON", flush=True)
    print("=" * 65, flush=True)

    for result in all_results:
        print(f"\n{result['checkpoint']}", flush=True)

        print(
            f"Exact match: {result['exact_match']}/"
            f"{result['examples']} "
            f"({result['exact_match_rate']:.2%})",
            flush=True,
        )

        print(
            f"Numeric match: {result['numeric_match']}/"
            f"{result['examples']} "
            f"({result['numeric_accuracy_all']:.2%})",
            flush=True,
        )

    print("\n[COMPLETE] Evaluation finished.", flush=True)
    print("[INFO] Official GSM8K test set was not accessed.", flush=True)
    print("[INFO] No training or checkpoint modifications.", flush=True)


if __name__ == "__main__":
    main()
