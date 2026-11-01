
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gpt0.checkpoints import load_checkpoint
from gpt0.model import GPT0
from evaluate import generate


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = ROOT / "checkpoints/best.pt"
    tokenizer_path = ROOT / "data/tokenizer.json"
    dataset_path = (
        ROOT / "data/raw/english/problem_solving/gsm8k/train.jsonl"
    )

    checkpoint = load_checkpoint(
        checkpoint_path,
        map_location=device,
    )

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    cfg = checkpoint["config"]["model"]

    model = GPT0(
        vocab_size=len(tokenizer.get_vocab()),
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=cfg["dropout"],
        pad_token_id=0,
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    with dataset_path.open(encoding="utf-8") as file:
        example = json.loads(next(file))

    question = example["question"]
    expected = example["answer"].split("####")[-1].strip()

    print("GPT-0 MATH — TRAINING MEMORY TEST")
    print("=" * 45)
    print("Device:", device)
    print("Checkpoint step:", checkpoint["step"])
    print("\nQUESTION:")
    print(question)
    print("\nEXPECTED ANSWER:")
    print(expected)

    generated = generate(
        model,
        tokenizer,
        question,
        device,
        max_new_tokens=256,
    )

    print("\nGENERATED RESPONSE:")
    print(generated)


if __name__ == "__main__":
    main()
