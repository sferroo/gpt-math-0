
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

CHECKPOINT_PATH = ROOT / "checkpoints" / "best.pt"
DATASET_PATH = ROOT / "data" / "processed" / "gsm8k_val.jsonl"
TOKENIZER_PATH = ROOT / "data" / "tokenizer.json"

BOS_ID = 2
PROBLEM_ID = 4
SOLUTION_ID = 5
ANSWER_ID = 6

MAX_TOKENS = 20


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("GPT-0 TEACHER FORCING SEQUENCE DIAGNOSTICS")
    print("Device:", device)

    # Load checkpoint
    checkpoint = load_checkpoint(
        CHECKPOINT_PATH,
        map_location=device,
    )

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    model = GPT0(
        **checkpoint["config"]["model"]
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Load first validation example
    with DATASET_PATH.open(encoding="utf-8") as f:
        example = next(
            json.loads(line)
            for line in f
            if line.strip()
        )

    question = example["question"]
    answer_text = example["answer"]

    if "####" not in answer_text:
        raise ValueError(
            "GSM8K answer does not contain ####"
        )

    solution_text, final_answer = answer_text.split(
        "####", 1
    )

    # Reconstruct the exact training sequence
    prompt_ids = (
        [BOS_ID, PROBLEM_ID]
        + tokenizer.encode(question).ids
        + [SOLUTION_ID]
    )

    solution_ids = tokenizer.encode(
        solution_text.strip()
    ).ids

    if not solution_ids:
        raise ValueError("Empty solution")

    # Select first 20 solution tokens
    target_ids = solution_ids[:MAX_TOKENS]

    full_ids = prompt_ids + target_ids

    idx = torch.tensor(
        [full_ids],
        dtype=torch.long,
        device=device,
    )

    print("\nCheckpoint step:", checkpoint["step"])
    print("Prompt length:", len(prompt_ids))
    print("Solution tokens analyzed:", len(target_ids))

    print("\nQuestion:")
    print(question)

    print("\nOriginal solution:")
    print(solution_text.strip())

    with torch.inference_mode():

        logits, _ = model(idx)

        # Position i predicts token i+1.
        start = len(prompt_ids) - 1
        end = start + len(target_ids)

        prediction_logits = logits[
            0, start:end, :
        ].float()

        targets = torch.tensor(
            target_ids,
            dtype=torch.long,
            device=device,
        )

        log_probs = F.log_softmax(
            prediction_logits,
            dim=-1,
        )

        probabilities = log_probs.exp()

        target_log_probs = log_probs.gather(
            1,
            targets.unsqueeze(1),
        ).squeeze(1)

        target_probs = target_log_probs.exp()

        predicted_ids = prediction_logits.argmax(
            dim=-1
        )

        token_losses = -target_log_probs

        correct = predicted_ids.eq(targets)

                # Token ranks: 1 means highest probability.
        # Calculate each rank independently to avoid
        # allocating a huge [20, vocab_size, vocab_size] tensor.

        target_ranks = torch.stack([
            (
                prediction_logits[i]
                > prediction_logits[i, targets[i]]
            ).sum() + 1
            for i in range(len(targets))
        ])

        print("\n========== TOKEN-BY-TOKEN ANALYSIS ==========")

        for i in range(len(target_ids)):
            expected_id = targets[i].item()
            predicted_id = predicted_ids[i].item()

            expected_text = tokenizer.decode(
                [expected_id],
                skip_special_tokens=False,
            )

            predicted_text = tokenizer.decode(
                [predicted_id],
                skip_special_tokens=False,
            )

            probability = target_probs[i].item()
            loss = token_losses[i].item()
            rank = target_ranks[i].item()

            status = "CORRECT" if correct[i].item() else "WRONG"

            print(
                f"\nPosition {i + 1:02d} | {status}"
            )
            print(
                f"  Expected:  {repr(expected_text)} "
                f"(ID={expected_id})"
            )
            print(
                f"  Predicted: {repr(predicted_text)} "
                f"(ID={predicted_id})"
            )
            print(f"  Probability: {probability:.8f}")
            print(f"  Rank: {rank}")
            print(f"  Loss: {loss:.6f}")

        # Summary
        accuracy = correct.float().mean().item()
        mean_loss = token_losses.mean().item()
        mean_probability = target_probs.mean().item()

        print("\n========== SUMMARY ==========")
        print(
            f"Correct tokens: "
            f"{correct.sum().item()}/{len(target_ids)}"
        )
        print(f"Token accuracy: {accuracy:.4%}")
        print(f"Mean token loss: {mean_loss:.6f}")
        print(
            f"Mean target probability: "
            f"{mean_probability:.8f}"
        )

        print(
            f"Median target rank: "
            f"{target_ranks.float().median().item():.1f}"
        )


if __name__ == "__main__":
    main()
