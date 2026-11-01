
import json
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]

tokenizer = Tokenizer.from_file(
    str(ROOT / "data/tokenizer.json")
)

raw_path = (
    ROOT
    / "data/raw/english/problem_solving/gsm8k/train.jsonl"
)

train_path = ROOT / "data/processed/train_sequences.jsonl"
val_path = ROOT / "data/processed/val_sequences.jsonl"

with raw_path.open(encoding="utf-8") as file:
    original = json.loads(next(file))

question_tokens = tokenizer.encode(
    original["question"]
).ids

print("Checking original GSM8K example...")
print("Question:", original["question"][:100])

for path in (train_path, val_path):
    found = False

    with path.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            if record["type"] != "gsm8k":
                continue

            tokens = record["tokens"]

            for start in range(
                len(tokens) - len(question_tokens) + 1
            ):
                if tokens[start:start + len(question_tokens)] == question_tokens:
                    print("\nMATCH FOUND")
                    print("File:", path.name)
                    print("ID:", record["id"])
                    found = True
                    break

            if found:
                break

    if not found:
        print(f"\nNo match in {path.name}")
