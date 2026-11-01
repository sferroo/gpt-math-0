
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
EOS_ID = 3

MAX_NEW_TOKENS = 20
TOP_K = 10


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("GPT-0 FIRST TOKEN DIAGNOSTICS")
    print("Device:", device)

    # Load checkpoint
    checkpoint = load_checkpoint(
        CHECKPOINT_PATH,
        map_location=device,
    )

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

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
            "GSM8K answer does not contain the #### separator"
        )

    solution_text, final_answer = answer_text.split(
        "####", 1
    )

    # Reconstruct exact training prompt
    question_ids = tokenizer.encode(question).ids

    prompt_ids = (
        [BOS_ID, PROBLEM_ID]
        + question_ids
        + [SOLUTION_ID]
    )

    idx = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    print("\nCheckpoint step:", checkpoint["step"])
    print("Prompt tokens:", len(prompt_ids))

    print("\nQuestion:")
    print(question)

    with torch.inference_mode():

        # Initial forward pass
        logits, _ = model(idx)

        first_logits = logits[:, -1, :]
        probabilities = F.softmax(
            first_logits.float(),
            dim=-1,
        )

        # Probability distribution entropy
        entropy = -(
            probabilities
            * torch.log(
                probabilities.clamp_min(1e-12)
            )
        ).sum(dim=-1)

        effective_vocab_size = torch.exp(entropy)

        # Top-K predictions
        top_probs, top_ids = torch.topk(
            probabilities[0],
            TOP_K,
        )

        print("\n========== FIRST TOKEN DISTRIBUTION ==========")
        print(f"Entropy: {entropy.item():.6f} nat")
        print(
            f"Effective vocabulary size: "
            f"{effective_vocab_size.item():.2f}"
        )

        print("\nTop tokens:")

        for token_id, probability in zip(
            top_ids.tolist(),
            top_probs.tolist(),
        ):
            token_text = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
            )

            print(
                f"ID={token_id:5d} | "
                f"P={probability:.8f} | "
                f"Token={repr(token_text)}"
            )

        # Teacher forcing: probability of the real first solution token
        print("\n========== TEACHER FORCING ==========")

        solution_ids = tokenizer.encode(
            solution_text.strip()
        ).ids

        if not solution_ids:
            raise ValueError(
                "Empty solution token sequence"
            )

        expected_token_id = solution_ids[0]

        expected_probability = probabilities[
            0, expected_token_id
        ].item()

        expected_text = tokenizer.decode(
            [expected_token_id],
            skip_special_tokens=False,
        )

        rank = (
            probabilities[0] > probabilities[
                0, expected_token_id
            ]
        ).sum().item() + 1

        print("Expected first solution token:")
        print("ID:", expected_token_id)
        print("Token:", repr(expected_text))
        print(
            f"Probability: {expected_probability:.8f}"
        )
        print("Rank:", rank)

        # Autoregressive generation
        print("\n========== AUTOREGRESSIVE GENERATION ==========")

        generated = []
        generation_idx = idx.clone()

        for step in range(MAX_NEW_TOKENS):

            context = generation_idx[
                :, -model.block_size:
            ]

            step_logits, _ = model(context)

            next_logits = step_logits[
                :, -1, :
            ].clone()

            # Same restrictions as evaluate_prompt_alignment.py
            next_logits[:, 0] = float("-inf")
            next_logits[:, BOS_ID] = float("-inf")
            next_logits[:, PROBLEM_ID] = float("-inf")

            next_token = torch.argmax(
                next_logits,
                dim=-1,
                keepdim=True,
            )

            token_id = next_token.item()
            generated.append(token_id)

            token_text = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
            )

            print(
                f"Step {step + 1:02d} | "
                f"ID={token_id:5d} | "
                f"Token={repr(token_text)}"
            )

            generation_idx = torch.cat(
                (generation_idx, next_token),
                dim=1,
            )

            if token_id == EOS_ID:
                break

        print("\nGenerated text:")
        print(
            tokenizer.decode(
                generated,
                skip_special_tokens=False,
            )
        )


if __name__ == "__main__":
    main()
