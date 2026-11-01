
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

CHECKPOINT_PATH = ROOT / "checkpoints" / "best.pt"
DATASET_PATH = ROOT / "data" / "processed" / "gsm8k_val.jsonl"
TOKENIZER_PATH = ROOT / "data" / "tokenizer.json"
OUTPUT_PATH = ROOT / "reports" / "prompt_alignment_results.json"

LIMIT = 100
SEED = 42
MAX_NEW_TOKENS = 512

BOS_ID = 2
EOS_ID = 3
PROBLEM_ID = 4
SOLUTION_ID = 5
ANSWER_ID = 6
PAD_ID = 0


def extract_gold(answer):
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer)
    return match.group(1).replace(",", "") if match else None


def extract_prediction(text):
    if "[ANSWER]" not in text:
        return None

    answer_text = text.rsplit("[ANSWER]", 1)[1]

    numbers = re.findall(
        r"-?\d[\d,]*(?:\.\d+)?",
        answer_text,
    )

    return numbers[-1].replace(",", "") if numbers else None


def make_exact_prompt(tokenizer, question):
    # Identico alla costruzione delle sequenze in prepare_data.py:
    # token speciali + codifica separata della domanda.
    question_ids = tokenizer.encode(question).ids

    return (
        [BOS_ID, PROBLEM_ID]
        + question_ids
        + [SOLUTION_ID]
    )


def make_legacy_prompt(tokenizer, question):
    # Metodo utilizzato precedentemente da evaluate.py.
    prompt = (
        "[BOS][PROBLEM] "
        + question
        + " [SOLUTION]"
    )

    return tokenizer.encode(prompt).ids


@torch.inference_mode()
def generate(model, prompt_ids, tokenizer, device):
    if len(prompt_ids) >= model.block_size:
        raise ValueError("Prompt exceeds model context length.")

    idx = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    forbidden_ids = [
        PAD_ID,
        BOS_ID,
        PROBLEM_ID,
    ]

    for _ in range(MAX_NEW_TOKENS):
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

        if next_token.item() == EOS_ID:
            break

    generated_ids = idx[0, len(prompt_ids):].tolist()

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
    )

    return generated_ids, generated_text


def repeated_ngram(ids, n=4, minimum=3):
    counts = {}

    for i in range(len(ids) - n + 1):
        gram = tuple(ids[i:i + n])
        counts[gram] = counts.get(gram, 0) + 1

        if counts[gram] >= minimum:
            return True

    return False


def evaluate_generation(ids, text, gold):
    prediction = extract_prediction(text)

    return {
        "prediction": prediction,
        "expected": gold,
        "correct": (
            prediction is not None
            and gold is not None
            and prediction == gold
        ),
        "has_answer_marker": ANSWER_ID in ids,
        "has_eos": EOS_ID in ids,
        "has_solution_marker": SOLUTION_ID in ids,
        "repeated_ngram": repeated_ngram(ids),
        "generated_tokens": len(ids),
        "generated_text": text,
    }


def summarize(results):
    total = len(results)

    return {
        "examples": total,
        "exact_match": sum(r["correct"] for r in results),
        "exact_match_percent": (
            sum(r["correct"] for r in results) / total * 100
            if total else 0
        ),
        "answer_marker_count": sum(
            r["has_answer_marker"] for r in results
        ),
        "eos_count": sum(r["has_eos"] for r in results),
        "solution_marker_count": sum(
            r["has_solution_marker"] for r in results
        ),
        "repeated_ngram_count": sum(
            r["repeated_ngram"] for r in results
        ),
        "average_generated_tokens": (
            sum(r["generated_tokens"] for r in results) / total
            if total else 0
        ),
    }


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("GPT-0 PROMPT ALIGNMENT EXPERIMENT")
    print("Device:", device)

    checkpoint = load_checkpoint(
        CHECKPOINT_PATH,
        map_location=device,
    )

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    model_config = checkpoint["config"]["model"]

    model = GPT0(**model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with DATASET_PATH.open(encoding="utf-8") as file:
        examples = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    rng = random.Random(SEED)
    rng.shuffle(examples)
    examples = examples[:min(LIMIT, len(examples))]

    legacy_results = []
    exact_results = []
    prompt_comparisons = []

    print("Checkpoint step:", checkpoint["step"])
    print("Examples:", len(examples))
    print("Max generated tokens:", MAX_NEW_TOKENS)

    for index, example in enumerate(examples, start=1):
        question = example["question"]
        gold = extract_gold(example["answer"])

        legacy_ids = make_legacy_prompt(tokenizer, question)
        exact_ids = make_exact_prompt(tokenizer, question)

        matches = legacy_ids == exact_ids

        first_difference = None

        for position, (a, b) in enumerate(
            zip(legacy_ids, exact_ids)
        ):
            if a != b:
                first_difference = position
                break

        if first_difference is None and len(legacy_ids) != len(exact_ids):
            first_difference = min(len(legacy_ids), len(exact_ids))

        prompt_comparisons.append({
            "example": index,
            "exact_match": matches,
            "legacy_length": len(legacy_ids),
            "exact_length": len(exact_ids),
            "first_difference": first_difference,
        })

        legacy_ids_generated, legacy_text = generate(
            model, legacy_ids, tokenizer, device
        )

        exact_ids_generated, exact_text = generate(
            model, exact_ids, tokenizer, device
        )

        legacy_result = evaluate_generation(
            legacy_ids_generated,
            legacy_text,
            gold,
        )

        exact_result = evaluate_generation(
            exact_ids_generated,
            exact_text,
            gold,
        )

        legacy_results.append(legacy_result)
        exact_results.append(exact_result)

        print(
            f"[{index}/{len(examples)}] "
            f"Prompt match: {matches} | "
            f"Legacy: {legacy_result['prediction']} | "
            f"Exact: {exact_result['prediction']} | "
            f"Expected: {gold}"
        )

    report = {
        "checkpoint_step": checkpoint["step"],
        "dataset": "GSM8K validation",
        "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompt_comparison": {
            "examples": len(prompt_comparisons),
            "exact_matches": sum(
                p["exact_match"] for p in prompt_comparisons
            ),
            "mismatches": sum(
                not p["exact_match"] for p in prompt_comparisons
            ),
            "details": prompt_comparisons,
        },
        "legacy_generation": {
            "summary": summarize(legacy_results),
            "results": legacy_results,
        },
        "exact_training_prefix_generation": {
            "summary": summarize(exact_results),
            "results": exact_results,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== PROMPT COMPARISON ==========")
    print(json.dumps(report["prompt_comparison"], indent=2)[:1500])

    print("\n========== LEGACY GENERATION ==========")
    print(json.dumps(report["legacy_generation"]["summary"], indent=2))

    print("\n========== EXACT TRAINING PREFIX ==========")
    print(json.dumps(
        report["exact_training_prefix_generation"]["summary"],
        indent=2,
    ))

    print("\nReport:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
