
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
DATA_PATH = ROOT / "data" / "processed" / "train_sequences.jsonl"
TOKENIZER_PATH = ROOT / "data" / "tokenizer.json"

TARGET_ID = "gsm8k-02438"

BOS_ID = 2
SOLUTION_ID = 5
ANSWER_ID = 6
EOS_ID = 3


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    checkpoint = load_checkpoint(
        CHECKPOINT_PATH,
        map_location=device,
    )

    config = checkpoint["config"]["model"]

    model = GPT0(**config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    record = None

    with DATA_PATH.open(encoding="utf-8") as file:
        for line in file:
            candidate = json.loads(line)

            if candidate["id"] == TARGET_ID:
                record = candidate
                break

    if record is None:
        raise ValueError(f"Example not found: {TARGET_ID}")

    tokens = record["tokens"]

    if tokens[0] != BOS_ID:
        raise ValueError("Sequence does not start with BOS")

    if tokens[-1] != EOS_ID:
        raise ValueError("Sequence does not end with EOS")

    solution_pos = tokens.index(SOLUTION_ID)
    answer_pos = tokens.index(ANSWER_ID)

    x = torch.tensor(
        [tokens[:-1]],
        dtype=torch.long,
        device=device,
    )

    y = torch.tensor(
        [tokens[1:]],
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        logits, _ = model(x)

        # Predict the next token at every position.
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="none",
        ).reshape_as(y)

        predictions = logits.argmax(dim=-1)

        overall_loss = token_losses.mean().item()

        # The target at position i corresponds to tokens[i + 1].
        solution_start = solution_pos
        answer_start = answer_pos

        solution_end = answer_start
        answer_end = len(tokens) - 1

        solution_losses = token_losses[
            0, solution_start:solution_end
        ]

        answer_losses = token_losses[
            0, answer_start:answer_end
        ]

        solution_predictions = predictions[
            0, solution_start:solution_end
        ]

        answer_predictions = predictions[
            0, answer_start:answer_end
        ]

        solution_targets = y[
            0, solution_start:solution_end
        ]

        answer_targets = y[
            0, answer_start:answer_end
        ]

        solution_accuracy = (
            solution_predictions == solution_targets
        ).float().mean().item()

        answer_accuracy = (
            answer_predictions == answer_targets
        ).float().mean().item()

        print("\n========== TEACHER-FORCED DIAGNOSTIC ==========")
        print(f"Checkpoint step: {checkpoint.get('step', 'unknown')}")
        print(f"Example ID: {TARGET_ID}")
        print(f"Total sequence tokens: {len(tokens)}")

        print("\n--- LOSS ---")
        print(f"Overall loss:  {overall_loss:.4f}")
        print(f"Solution loss: {solution_losses.mean().item():.4f}")
        print(f"Answer loss:   {answer_losses.mean().item():.4f}")

        print("\n--- TOP-1 TOKEN ACCURACY ---")
        print(f"Solution accuracy: {solution_accuracy * 100:.2f}%")
        print(f"Answer accuracy:   {answer_accuracy * 100:.2f}%")

        print("\n--- EXPECTED ANSWER ---")
        decoded = tokenizer.decode(tokens[answer_pos + 1:-1])
        print(decoded)

        print("\n===============================================")


if __name__ == "__main__":
    main()
