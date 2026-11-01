import json
import sys
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

PAD_ID = 0
BOS_ID = 2
EOS_ID = 3
PROBLEM_ID = 4
SOLUTION_ID = 5
ANSWER_ID = 6

MAX_NEW_TOKENS = 512
EXPECTED_VALIDATION = 748


def load_validation():
    with DATASET_PATH.open(encoding="utf-8") as f:
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
            f"Expected {EXPECTED_VALIDATION} GSM8K validation examples, "
            f"found {len(examples)}. Check the dataset metadata."
        )

    return examples


def split_sequence(tokens):
    if tokens[0] != BOS_ID or tokens[-1] != EOS_ID:
        raise ValueError("Invalid BOS/EOS sequence")

    solution_pos = tokens.index(SOLUTION_ID)
    answer_pos = tokens.index(ANSWER_ID)

    if answer_pos <= solution_pos:
        raise ValueError("Invalid SOLUTION/ANSWER ordering")

    prompt = tokens[:solution_pos + 1]
    gold_reasoning = tokens[solution_pos + 1:answer_pos]
    gold_answer = tokens[answer_pos + 1:-1]
    gold_completion = tokens[solution_pos + 1:]

    return prompt, gold_reasoning, gold_answer, gold_completion


@torch.inference_mode()
def generate(model, prompt, device):
    context = torch.tensor(
        [prompt],
        dtype=torch.long,
        device=device,
    )

    generated = []

    max_tokens = min(
        MAX_NEW_TOKENS,
        model.block_size - len(prompt),
    )

    stopped_by_eos = False

    for _ in range(max_tokens):
        logits, _ = model(context)

        next_logits = logits[:, -1, :].float().clone()

        for token_id in (PAD_ID, BOS_ID, PROBLEM_ID, SOLUTION_ID):
            next_logits[:, token_id] = float("-inf")

        next_token = next_logits.argmax(
            dim=-1,
            keepdim=True,
        )

        token_id = next_token.item()
        generated.append(token_id)

        context = torch.cat(
            (context, next_token),
            dim=1,
        )

        if token_id == EOS_ID:
            stopped_by_eos = True
            break

    return generated, stopped_by_eos


def split_generated(generated):
    if ANSWER_ID not in generated:
        return None, None

    answer_pos = generated.index(ANSWER_ID)

    reasoning = generated[:answer_pos]

    answer_end = (
        generated.index(EOS_ID, answer_pos)
        if EOS_ID in generated[answer_pos:]
        else len(generated)
    )

    answer = generated[answer_pos + 1:answer_end]

    return reasoning, answer


@torch.inference_mode()
def evaluate_checkpoint(checkpoint_path, examples, tokenizer, device):
    print("\n" + "=" * 65)
    print("CHECKPOINT:", checkpoint_path.name)
    print("=" * 65)

    checkpoint = load_checkpoint(
        checkpoint_path,
        map_location=device,
    )

    model = GPT0(
        **checkpoint["config"]["model"],
        pad_token_id=PAD_ID,
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    print("Training step:", checkpoint["step"])
    print("Examples:", len(examples))

    exact_reasoning = 0
    exact_answer = 0
    exact_completion = 0

    reasoning_matches = 0
    reasoning_total = 0

    answer_matches = 0
    answer_total = 0

    answer_marker_count = 0
    eos_count = 0
    limit_count = 0

    malformed_count = 0
    total_generated = 0
    total_gold = 0

    for index, record in enumerate(examples, start=1):
        tokens = record["tokens"]

        prompt, gold_reasoning, gold_answer, gold_completion = (
            split_sequence(tokens)
        )

        if len(prompt) >= model.block_size:
            raise RuntimeError(
                f"Prompt exceeds context window: {record.get('id')}"
            )

        generated, stopped_by_eos = generate(
            model,
            prompt,
            device,
        )

        generated_reasoning, generated_answer = split_generated(
            generated
        )

        has_answer = generated_reasoning is not None

        if has_answer:
            answer_marker_count += 1

            reasoning_common = min(
                len(generated_reasoning),
                len(gold_reasoning),
            )

            answer_common = min(
                len(generated_answer),
                len(gold_answer),
            )

            reasoning_matches += sum(
                generated_reasoning[i] == gold_reasoning[i]
                for i in range(reasoning_common)
            )

            answer_matches += sum(
                generated_answer[i] == gold_answer[i]
                for i in range(answer_common)
            )

            reasoning_total += len(gold_reasoning)
            answer_total += len(gold_answer)

            exact_reasoning += int(
                generated_reasoning == gold_reasoning
            )

            exact_answer += int(
                generated_answer == gold_answer
            )

        else:
            malformed_count += 1
            reasoning_total += len(gold_reasoning)
            answer_total += len(gold_answer)

        exact_completion += int(
            generated == gold_completion
        )

        eos_count += int(stopped_by_eos)
        limit_count += int(not stopped_by_eos)

        total_generated += len(generated)
        total_gold += len(gold_completion)

        if index <= 3:
            print(f"\nExample {index} | ID={record.get('id')}")

            print(
                "Gold:",
                repr(tokenizer.decode(
                    gold_completion,
                    skip_special_tokens=False,
                )[:250]),
            )

            print(
                "Generated:",
                repr(tokenizer.decode(
                    generated,
                    skip_special_tokens=False,
                )[:250]),
            )

            print(
                "Exact reasoning:",
                has_answer and generated_reasoning == gold_reasoning,
            )

            print(
                "Exact answer:",
                has_answer and generated_answer == gold_answer,
            )

        if index % 50 == 0:
            print(f"Progress: {index}/{len(examples)}")

    count = len(examples)

    results = {
        "checkpoint": checkpoint_path.name,
        "reasoning_exact": exact_reasoning,
        "answer_exact": exact_answer,
        "completion_exact": exact_completion,
        "reasoning_accuracy": reasoning_matches / max(reasoning_total, 1),
        "answer_accuracy": answer_matches / max(answer_total, 1),
        "answer_marker": answer_marker_count,
        "eos": eos_count,
        "limit": limit_count,
        "malformed": malformed_count,
        "avg_generated": total_generated / count,
        "avg_gold": total_gold / count,
    }

    print("\n---------- RESULTS ----------")
    print(f"Reasoning exact: {exact_reasoning}/{count}")
    print(f"Answer exact: {exact_answer}/{count}")
    print(f"Full completion exact: {exact_completion}/{count}")
    print(f"Reasoning token accuracy: {results['reasoning_accuracy']:.2%}")
    print(f"Answer token accuracy: {results['answer_accuracy']:.2%}")
    print(f"Generated ANSWER marker: {answer_marker_count}/{count}")
    print(f"Generated EOS: {eos_count}/{count}")
    print(f"Reached generation limit: {limit_count}/{count}")
    print(f"Missing ANSWER marker: {malformed_count}/{count}")
    print(f"Average generated tokens: {results['avg_generated']:.2f}")
    print(f"Average gold tokens: {results['avg_gold']:.2f}")

    del model
    del checkpoint

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("GPT-0 GENERALIZATION EVALUATION")
    print("--------------------------------")
    print("Device:", device)
    print("Validation set: GSM8K")
    print("Expected examples:", EXPECTED_VALIDATION)

    examples = load_validation()
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    print("Verified validation examples:", len(examples))

    results = []

    for checkpoint_path in CHECKPOINTS:
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        result = evaluate_checkpoint(
            checkpoint_path,
            examples,
            tokenizer,
            device,
        )

        results.append(result)

    print("\n" + "=" * 65)
    print("FINAL CHECKPOINT COMPARISON")
    print("=" * 65)

    for result in results:
        print(f"\n{result['checkpoint']}")
        print(
            f"Reasoning exact: {result['reasoning_exact']}/{EXPECTED_VALIDATION}"
        )
        print(
            f"Answer exact: {result['answer_exact']}/{EXPECTED_VALIDATION}"
        )
        print(
            f"Completion exact: {result['completion_exact']}/{EXPECTED_VALIDATION}"
        )
        print(
            f"Reasoning token accuracy: {result['reasoning_accuracy']:.2%}"
        )
        print(
            f"Answer token accuracy: {result['answer_accuracy']:.2%}"
        )

    print("\nEvaluation completed.")
    print("No training performed.")
    print("No checkpoint modified.")
    print("No dataset modified.")
    print("Official GSM8K test set was not accessed.")


if __name__ == "__main__":
    main()