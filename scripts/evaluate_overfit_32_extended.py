import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpt0.checkpoints import load_checkpoint
from gpt0.model import GPT0

CHECKPOINT_PATH = ROOT / "checkpoints" / "overfit_32_extended.pt"
DATASET_PATH = ROOT / "data" / "processed" / "train_sequences.jsonl"
TOKENIZER_PATH = ROOT / "data" / "tokenizer.json"

PAD_ID = 0
BOS_ID = 2
EOS_ID = 3
PROBLEM_ID = 4
SOLUTION_ID = 5
ANSWER_ID = 6

MAX_NEW_TOKENS = 512


def split_sequence(tokens):
    solution_pos = tokens.index(SOLUTION_ID)
    answer_pos = tokens.index(ANSWER_ID)

    if tokens[0] != BOS_ID or tokens[-1] != EOS_ID:
        raise ValueError("Invalid sequence")

    if answer_pos <= solution_pos:
        raise ValueError("Invalid marker ordering")

    return (
        tokens[:solution_pos + 1],
        tokens[solution_pos + 1:answer_pos],
    )


def load_examples(checkpoint):
    with DATASET_PATH.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    records_by_id = {
        str(record.get("id", i)): record
        for i, record in enumerate(records)
    }

    examples = []

    for selected_id in checkpoint["selected_ids"]:
        key = str(selected_id)

        if key not in records_by_id:
            raise RuntimeError(f"Missing example: {key}")

        record = records_by_id[key]

        if record.get("type") != "gsm8k":
            raise RuntimeError(f"Unexpected example type: {key}")

        examples.append(record)

    if len(examples) != 32:
        raise RuntimeError(f"Expected 32 examples, found {len(examples)}")

    return examples


@torch.inference_mode()
def generate(model, prompt, device):
    context = torch.tensor([prompt], dtype=torch.long, device=device)
    generated = []

    max_tokens = min(
        MAX_NEW_TOKENS,
        model.block_size - len(prompt),
    )

    stopped_answer = False
    stopped_eos = False

    for _ in range(max_tokens):
        logits, _ = model(context)

        next_logits = logits[:, -1, :].float().clone()

        for token_id in (PAD_ID, BOS_ID, PROBLEM_ID, SOLUTION_ID):
            next_logits[:, token_id] = float("-inf")

        next_token = next_logits.argmax(dim=-1, keepdim=True)
        token_id = next_token.item()

        if token_id == ANSWER_ID:
            stopped_answer = True
            break

        if token_id == EOS_ID:
            stopped_eos = True
            break

        generated.append(token_id)
        context = torch.cat((context, next_token), dim=1)

    reached_limit = not stopped_answer and not stopped_eos

    return generated, stopped_answer, stopped_eos, reached_limit


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("GPT-0 EXTENDED OVERFIT EVALUATION")
    print("---------------------------------")
    print("Device:", device)

    checkpoint = load_checkpoint(
        CHECKPOINT_PATH,
        map_location=device,
    )

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    model = GPT0(
        **checkpoint["config"]["model"],
        pad_token_id=PAD_ID,
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    examples = load_examples(checkpoint)

    print("Checkpoint:", CHECKPOINT_PATH.name)
    print("Training steps:", checkpoint["step"])
    print("Selected examples:", len(examples))

    exact_matches = 0
    token_matches = 0
    total_gold_tokens = 0

    answer_stops = 0
    eos_stops = 0
    limit_stops = 0

    total_generated = 0
    total_gold = 0

    print("\n========== AUTOREGRESSIVE GENERATION ==========")

    for index, record in enumerate(examples, start=1):
        prompt, gold = split_sequence(record["tokens"])

        if len(prompt) >= model.block_size:
            raise RuntimeError(f"Prompt too long: {record.get('id')}")

        generated, stopped_answer, stopped_eos, reached_limit = generate(
            model, prompt, device
        )

        common = min(len(generated), len(gold))
        matching = sum(
            generated[i] == gold[i]
            for i in range(common)
        )

        exact = generated == gold

        exact_matches += int(exact)
        token_matches += matching
        total_gold_tokens += len(gold)

        total_generated += len(generated)
        total_gold += len(gold)

        answer_stops += int(stopped_answer)
        eos_stops += int(stopped_eos)
        limit_stops += int(reached_limit)

        if index <= 5 or not exact:
            print(f"\nExample {index:02d} | ID={record.get('id')}")
            print(
                "Gold:",
                repr(tokenizer.decode(gold, skip_special_tokens=False)[:200]),
            )
            print(
                "Generated:",
                repr(tokenizer.decode(generated, skip_special_tokens=False)[:200]),
            )
            print(f"Generated tokens: {len(generated)}")
            print(f"Gold tokens: {len(gold)}")
            print("Exact match:", exact)

    print("\n========== AUTOREGRESSIVE SUMMARY ==========")
    print(f"Exact reasoning: {exact_matches}/32")
    print(
        "Position-aligned token accuracy:",
        f"{token_matches / max(total_gold_tokens, 1):.2%}",
    )
    print(f"Average generated length: {total_generated / 32:.2f}")
    print(f"Average gold length: {total_gold / 32:.2f}")
    print(f"Stopped at ANSWER: {answer_stops}/32")
    print(f"Stopped at EOS: {eos_stops}/32")
    print(f"Reached generation limit: {limit_stops}/32")

    print("\nEvaluation completed.")
    print("No training performed.")
    print("No checkpoint modified.")
    print("No dataset modified.")


if __name__ == "__main__":
    main()
