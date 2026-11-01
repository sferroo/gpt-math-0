
import argparse
import json
import re
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpt0.checkpoints import load_checkpoint
from gpt0.model import GPT0


def extract_gold_answer(answer):
    """Extract the official GSM8K answer after ####."""
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer)

    if not match:
        return None

    return normalize_number(match.group(1))


def normalize_number(value):
    if value is None:
        return None

    return value.replace(",", "").strip()


def extract_prediction(text):
    """Extract the final number generated after [ANSWER]."""
    marker = "[ANSWER]"

    if marker in text:
        text = text.rsplit(marker, 1)[1]

    numbers = re.findall(
        r"-?\d[\d,]*(?:\.\d+)?",
        text,
    )

    if not numbers:
        return None

    return normalize_number(numbers[-1])


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    question,
    device,
    max_new_tokens=512,
):
    model.eval()

    prompt = (
        "[BOS][PROBLEM] "
        + question
        + " [SOLUTION]"
    )

    input_ids = tokenizer.encode(prompt).ids

    if len(input_ids) >= model.block_size:
        raise ValueError(
            "Prompt is too long for the model context."
        )

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
        token_id
        for token_id in forbidden_ids
        if token_id is not None
    ]

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

        idx = torch.cat(
            (idx, next_token),
            dim=1,
        )

        if eos_id is not None and next_token.item() == eos_id:
            break

    generated_ids = idx[0, len(input_ids):].tolist()

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best.pt",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--output",
        default="reports/gsm8k_results.json",
    )

    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be positive")

    checkpoint_path = ROOT / args.checkpoint
    dataset_path = ROOT / "data/benchmarks/gsm8k/test.jsonl"
    tokenizer_path = ROOT / "data/tokenizer.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Loading GPT-0 Math...")

    checkpoint = load_checkpoint(
        checkpoint_path,
        map_location=device,
    )

    cfg = checkpoint["config"]
    model_cfg = cfg["model"]

    tokenizer = Tokenizer.from_file(
        str(tokenizer_path)
    )

    model = GPT0(
        vocab_size=len(tokenizer.get_vocab()),
        block_size=model_cfg["block_size"],
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        n_embd=model_cfg["n_embd"],
        dropout=model_cfg["dropout"],
        pad_token_id=0,
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"Device: {device}")
    print(f"Checkpoint step: {checkpoint['step']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Evaluation limit: {args.limit}")

    with dataset_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        examples = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    examples = examples[:args.limit]

    results = []

    correct = 0
    attempted = 0

    for index, example in enumerate(examples, start=1):
        question = example["question"]
        gold = extract_gold_answer(example["answer"])

        generated = generate(
            model,
            tokenizer,
            question,
            device,
            max_new_tokens=args.max_new_tokens,
        )

        prediction = extract_prediction(generated)

        is_correct = (
            prediction is not None
            and gold is not None
            and prediction == gold
        )

        attempted += 1

        if is_correct:
            correct += 1

        result = {
            "id": index,
            "question": question,
            "generated": generated,
            "prediction": prediction,
            "expected": gold,
            "correct": is_correct,
        }

        results.append(result)

        status = "CORRECT" if is_correct else "WRONG"

        print(f"\n[{index}/{len(examples)}] {status}")
        print(f"Expected:   {gold}")
        print(f"Predicted:  {prediction}")
        print(f"Generated:  {generated[:300]!r}")

    accuracy = (
        correct / attempted * 100
        if attempted
        else 0.0
    )

    report = {
        "model": "GPT-0 Math",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint["step"],
        "dataset": "GSM8K official test",
        "evaluated_examples": attempted,
        "correct": correct,
        "incorrect": attempted - correct,
        "accuracy_percent": accuracy,
        "results": results,
    }

    output_path = ROOT / args.output
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 40)
    print("GPT-0 MATH — GSM8K RESULTS")
    print("=" * 40)
    print(f"Evaluated: {attempted}")
    print(f"Correct:   {correct}")
    print(f"Incorrect: {attempted - correct}")
    print(f"Accuracy:  {accuracy:.2f}%")
    print(f"Report:    {output_path}")


if __name__ == "__main__":
    main()
