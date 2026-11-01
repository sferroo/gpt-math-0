
import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpt0.checkpoints import load_checkpoint
from gpt0.model import GPT0


def normalize_number(value):
    if value is None:
        return None
    return value.replace(",", "").strip()


def extract_gold_answer(answer):
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer)
    return normalize_number(match.group(1)) if match else None


def extract_prediction(text):
    # Non accettiamo numeri generati senza il marker ANSWER.
    marker = "[ANSWER]"

    if marker not in text:
        return None

    answer_text = text.rsplit(marker, 1)[1]

    numbers = re.findall(
        r"-?\d[\d,]*(?:\.\d+)?",
        answer_text,
    )

    return normalize_number(numbers[-1]) if numbers else None


def has_repeated_ngram(token_ids, n=4, minimum_repetitions=3):
    if len(token_ids) < n * minimum_repetitions:
        return False

    counts = {}

    for i in range(len(token_ids) - n + 1):
        gram = tuple(token_ids[i:i + n])
        counts[gram] = counts.get(gram, 0) + 1

        if counts[gram] >= minimum_repetitions:
            return True

    return False


@torch.inference_mode()
def generate(model, tokenizer, question, device, max_new_tokens):
    prompt = "[BOS][PROBLEM] " + question + " [SOLUTION]"
    input_ids = tokenizer.encode(prompt).ids

    if len(input_ids) >= model.block_size:
        raise ValueError("Prompt is too long for model context.")

    idx = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device,
    )

    eos_id = tokenizer.token_to_id("[EOS]")

    forbidden_ids = [
        tokenizer.token_to_id(token)
        for token in ("[PAD]", "[BOS]", "[PROBLEM]")
    ]

    forbidden_ids = [
        token_id for token_id in forbidden_ids
        if token_id is not None
    ]

    stopped_on_eos = False

    for _ in range(max_new_tokens):
        context = idx[:, -model.block_size:]
        logits, _ = model(context)

        next_logits = logits[:, -1, :].clone()

        for token_id in forbidden_ids:
            next_logits[:, token_id] = float("-inf")

        next_token = torch.argmax(
            next_logits,
            dim=-1,
            keepdim=True,
        )

        idx = torch.cat((idx, next_token), dim=1)

        if eos_id is not None and next_token.item() == eos_id:
            stopped_on_eos = True
            break

    generated_ids = idx[0, len(input_ids):].tolist()

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
    )

    return generated_text, generated_ids, stopped_on_eos


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="reports/gsm8k_validation_results.json",
    )

    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be positive")

    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    checkpoint_path = ROOT / "checkpoints" / "best.pt"
    dataset_path = ROOT / "data" / "processed" / "gsm8k_val.jsonl"
    tokenizer_path = ROOT / "data" / "tokenizer.json"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Loading GPT-0 Math...")
    print(f"Device: {device}")

    checkpoint = load_checkpoint(
        checkpoint_path,
        map_location=device,
    )

    model_config = checkpoint["config"]["model"]

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    model = GPT0(
        vocab_size=model_config["vocab_size"],
        block_size=model_config["block_size"],
        n_layer=model_config["n_layer"],
        n_head=model_config["n_head"],
        n_embd=model_config["n_embd"],
        dropout=model_config["dropout"],
        pad_token_id=0,
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    with dataset_path.open(encoding="utf-8") as file:
        examples = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    examples = examples[:min(args.limit, len(examples))]

    print(f"Checkpoint step: {checkpoint['step']}")
    print(f"Examples: {len(examples)}")
    print(f"Maximum generated tokens: {args.max_new_tokens}")

    results = []

    for index, example in enumerate(examples, start=1):
        generated, generated_ids, stopped_on_eos = generate(
            model,
            tokenizer,
            example["question"],
            device,
            args.max_new_tokens,
        )

        gold = extract_gold_answer(example["answer"])
        prediction = extract_prediction(generated)

        correct = (
            prediction is not None
            and gold is not None
            and prediction == gold
        )

        has_answer_marker = "[ANSWER]" in generated
        has_solution_marker = "[SOLUTION]" in generated
        repeated = has_repeated_ngram(generated_ids)

        result = {
            "id": index,
            "question": example["question"],
            "generated": generated,
            "prediction": prediction,
            "expected": gold,
            "correct": correct,
            "has_answer_marker": has_answer_marker,
            "has_solution_marker": has_solution_marker,
            "stopped_on_eos": stopped_on_eos,
            "repeated_ngram": repeated,
            "generated_tokens": len(generated_ids),
        }

        results.append(result)

        status = "CORRECT" if correct else "WRONG"

        print(
            f"[{index}/{len(examples)}] {status} | "
            f"expected={gold} | predicted={prediction} | "
            f"tokens={len(generated_ids)}"
        )

    total = len(results)
    correct = sum(r["correct"] for r in results)

    answer_markers = sum(r["has_answer_marker"] for r in results)
    solution_markers = sum(r["has_solution_marker"] for r in results)
    eos_count = sum(r["stopped_on_eos"] for r in results)
    repetitions = sum(r["repeated_ngram"] for r in results)

    accuracy = correct / total * 100 if total else 0.0

    report = {
        "model": "GPT-0 Math",
        "checkpoint": "checkpoints/best.pt",
        "checkpoint_step": checkpoint["step"],
        "dataset": "GSM8K validation",
        "seed": args.seed,
        "evaluated_examples": total,
        "correct": correct,
        "incorrect": total - correct,
        "exact_match_percent": accuracy,
        "answer_marker_count": answer_markers,
        "solution_marker_count": solution_markers,
        "eos_count": eos_count,
        "repeated_ngram_count": repetitions,
        "average_generated_tokens": (
            sum(r["generated_tokens"] for r in results) / total
            if total else 0
        ),
        "results": results,
    }

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== VALIDATION RESULTS ==========")
    print(f"Evaluated:              {total}")
    print(f"Correct:                {correct}")
    print(f"Exact Match:            {accuracy:.2f}%")
    print(f"[ANSWER] marker:        {answer_markers}/{total}")
    print(f"[SOLUTION] marker:      {solution_markers}/{total}")
    print(f"EOS generated:          {eos_count}/{total}")
    print(f"Repeated n-grams:       {repetitions}/{total}")
    print(f"Average generated size: {report['average_generated_tokens']:.1f} tokens")
    print(f"Report:                 {output_path}")


if __name__ == "__main__":
    main()
