
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
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
EXPECTED_EXAMPLES = 32


def load_selected_examples(checkpoint):
    with DATASET_PATH.open(encoding="utf-8") as f:
        records = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    records_by_id = {
        str(record.get("id", i)): record
        for i, record in enumerate(records)
    }

    selected_ids = checkpoint["selected_ids"]

    examples = []

    for selected_id in selected_ids:
        key = str(selected_id)

        if key not in records_by_id:
            raise RuntimeError(f"Missing dataset example: {key}")

        record = records_by_id[key]

        if record.get("type") != "gsm8k":
            raise RuntimeError(f"Non-GSM8K example: {key}")

        examples.append(record)

    if len(examples) != EXPECTED_EXAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_EXAMPLES} examples, "
            f"found {len(examples)}"
        )

    if len(set(map(str, selected_ids))) != EXPECTED_EXAMPLES:
        raise RuntimeError("Duplicate selected IDs detected")

    return examples


def split_sequence(tokens):
    if tokens[0] != BOS_ID or tokens[-1] != EOS_ID:
        raise ValueError("Invalid BOS/EOS sequence")

    if SOLUTION_ID not in tokens or ANSWER_ID not in tokens:
        raise ValueError("Missing SOLUTION/ANSWER marker")

    solution_pos = tokens.index(SOLUTION_ID)
    answer_pos = tokens.index(ANSWER_ID)

    if answer_pos <= solution_pos:
        raise ValueError("Invalid marker ordering")

    prompt = tokens[:solution_pos + 1]
    gold_solution = tokens[solution_pos + 1:answer_pos]

    return prompt, gold_solution


@torch.inference_mode()
def teacher_forcing_evaluation(model, examples, device):
    model.eval()

    total_tokens = 0
    correct_tokens = 0
    total_loss = 0.0
    exact_matches = 0

    print("\n========== TEACHER FORCING ==========")

    for index, record in enumerate(examples, start=1):
        tokens = record["tokens"]
        prompt, gold_solution = split_sequence(tokens)

        input_ids = torch.tensor(
            [tokens[:-1]],
            dtype=torch.long,
            device=device,
        )

        logits, _ = model(input_ids)

        solution_start = len(prompt) - 1
        solution_end = solution_start + len(gold_solution)

        solution_logits = logits[
            0, solution_start:solution_end, :
        ].float()

        targets = torch.tensor(
            gold_solution,
            dtype=torch.long,
            device=device,
        )

        log_probs = F.log_softmax(solution_logits, dim=-1)

        target_log_probs = log_probs.gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)

        predictions = solution_logits.argmax(dim=-1)
        correct = predictions.eq(targets)

        total_tokens += targets.numel()
        correct_tokens += correct.sum().item()
        total_loss += -target_log_probs.sum().item()

        exact = bool(correct.all().item())

        if exact:
            exact_matches += 1

    accuracy = correct_tokens / max(total_tokens, 1)
    mean_loss = total_loss / max(total_tokens, 1)

    print(f"Reasoning tokens: {total_tokens}")
    print(f"Token accuracy: {accuracy:.2%}")
    print(f"Mean reasoning loss: {mean_loss:.6f}")
    print(f"Exact reasoning: {exact_matches}/{EXPECTED_EXAMPLES}")

    return {
        "token_accuracy": accuracy,
        "mean_loss": mean_loss,
        "exact_matches": exact_matches,
    }


@torch.inference_mode()
def generate_solution(model, prompt, device):
    model.eval()

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

    stopped_by_answer = False
    stopped_by_eos = False

    for _ in range(max_tokens):
        logits, _ = model(context)

        next_logits = logits[:, -1, :].float().clone()

        # Prevent generation of invalid structural tokens.
        next_logits[:, PAD_ID] = float("-inf")
        next_logits[:, BOS_ID] = float("-inf")
        next_logits[:, PROBLEM_ID] = float("-inf")
        next_logits[:, SOLUTION_ID] = float("-inf")

        next_token = next_logits.argmax(
            dim=-1,
            keepdim=True,
        )

        token_id = next_token.item()

        if token_id == ANSWER_ID:
            stopped_by_answer = True
            break

        if token_id == EOS_ID:
            stopped_by_eos = True
            break

        generated.append(token_id)

        context = torch.cat(
            (context, next_token),
            dim=1,
        )

    reached_limit = (
        not stopped_by_answer and not stopped_by_eos
    )

    return (
        generated,
        stopped_by_answer,
        stopped_by_eos,
        reached_limit,
    )


@torch.inference_mode()
def autoregressive_evaluation(
    model,
    examples,
    tokenizer,
    device,
):
    print("\n========== AUTOREGRESSIVE GENERATION ==========")

    exact_matches = 0
    token_matches = 0
    total_gold_tokens = 0

    answer_terminations = 0
    eos_terminations = 0
    limit_terminations = 0

    total_generated_tokens = 0
    total_gold_lengths = 0

    for index, record in enumerate(examples, start=1):
        tokens = record["tokens"]
        prompt, gold_solution = split_sequence(tokens)

        generated, stopped_answer, stopped_eos, reached_limit = (
            generate_solution(model, prompt, device)
        )

        common_length = min(
            len(generated),
            len(gold_solution),
        )

        matching = sum(
            generated[i] == gold_solution[i]
            for i in range(common_length)
        )

        token_matches += matching
        total_gold_tokens += len(gold_solution)

        total_generated_tokens += len(generated)
        total_gold_lengths += len(gold_solution)

        exact = generated == gold_solution

        if exact:
            exact_matches += 1

        answer_terminations += int(stopped_answer)
        eos_terminations += int(stopped_eos)
        limit_terminations += int(reached_limit)

        if index <= 5 or not exact:
            generated_text = tokenizer.decode(
                generated,
                skip_special_tokens=False,
            )

            gold_text = tokenizer.decode(
                gold_solution,
                skip_special_tokens=False,
            )

            print(f"\nExample {index:02d} | ID={record.get('id')}")
            print("Gold:", repr(gold_text[:200]))
            print("Generated:", repr(generated_text[:200]))
            print(
                f"Generated length: {len(generated)} | "
                f"Gold length: {len(gold_solution)}"
            )
            print("Exact match:", exact)

    token_accuracy = token_matches / max(total_gold_tokens, 1)

    avg_generated = total_generated_tokens / EXPECTED_EXAMPLES
    avg_gold = total_gold_lengths / EXPECTED_EXAMPLES

    print("\n========== AUTOREGRESSIVE SUMMARY ==========")
    print(f"Exact reasoning: {exact_matches}/{EXPECTED_EXAMPLES}")
    print(f"Position-aligned token accuracy: {token_accuracy:.2%}")
    print(f"Average generated length: {avg_generated:.2f}")
    print(f"Average gold length: {avg_gold:.2f}")
    print(f"Stopped at ANSWER: {answer_terminations}/{EXPECTED_EXAMPLES}")
    print(f"Stopped at EOS: {eos_terminations}/{EXPECTED_EXAMPLES}")
    print(f"Reached generation limit: {limit_terminations}/{EXPECTED_EXAMPLES}")

    return {
        "exact_matches": exact_matches,
        "token_accuracy": token_accuracy,
        "answer_terminations": answer_terminations,
        "eos_terminations": eos_terminations,
        "limit_terminations": limit_terminations,
        "average_generated_length": avg_generated,
        "average_gold_length": avg_gold,
    }


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("GPT-0 EXTENDED OVERFIT EVALUATION")
    print("---------------------------------")
    print("Device:", device)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

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

    examples = load_selected_examples(checkpoint)

    print("Checkpoint:", CHECKPOINT_PATH.name)
    print("Training steps:", checkpoint["step"])
    print("Selected examples:", len(examples))

    # Verify that every prompt fits inside the model context.
    for record in examples:
        prompt, _ = split_sequence(record["tokens"])

        if len(prompt) >= model.block_size:
            raise RuntimeError(
                f"Prompt exceeds context window: {record.get('id')}"
            )

    teacher_results = teacher_forcing_evaluation(
        model, examples, device
    )

    generation_results = autoregressive_evaluation(
        model, examples, tokenizer, device
    )

    print("\n========== FINAL COMPARISON ==========")
    print(
        "Teacher-forced exact reasoning:",
        f"{teacher_results['exact_matches']}/{EXPECTED_EXAMPLES}",
    )
    print(
        "Autoregressive exact reasoning:",
        f"{generation_results['exact_matches']}/{EXPECTED_EXAMPLES}",
    )
    print(
        "Teacher-forced token accuracy:",
        f"{teacher_results['token_accuracy']:.2%}",
    )
    print(
        "Autoregressive token accuracy:",
        f"{generation_results['token_accuracy']:.2%}",
    )

    print("\nEvaluation completed.")
    print("No training performed.")
    print("No checkpoint modified.")
    print("No dataset modified.")


if __name__ == "__main__":
    main()
