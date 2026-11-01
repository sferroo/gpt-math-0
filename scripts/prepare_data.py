
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpt0.data_pipeline import (
    load_documents,
    split_documents,
    write_split_manifest,
)

SEED = 42
BLOCK_SIZE = 1024
OUT = ROOT / "data/processed"


# 1. Load tokenizer
tok = Tokenizer.from_file(str(ROOT / "data/tokenizer.json"))

special = {
    name: tok.token_to_id(name)
    for name in (
        "[BOS]", "[EOS]", "[PROBLEM]", "[SOLUTION]", "[ANSWER]"
    )
}

if any(value is None for value in special.values()):
    raise RuntimeError("Missing special tokens")

bos = special["[BOS]"]
eos = special["[EOS]"]


# 2. Load and split theory documents
docs = load_documents(ROOT / "data/raw/manifest.jsonl")

train_docs, val_docs = split_documents(
    docs,
    val_fraction=0.2,
    seed=SEED,
)


# 3. Load GSM8K splits
def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


gsm_train = load_jsonl(OUT / "gsm8k_train.jsonl")
gsm_val = load_jsonl(OUT / "gsm8k_val.jsonl")


# 4. Encode GSM8K
def encode_gsm8k(example):
    question = example["question"].strip()
    raw_answer = example["answer"].strip()

    if "####" not in raw_answer:
        raise ValueError("Missing GSM8K answer separator ####")

    solution, final_answer = raw_answer.rsplit("####", 1)

    if not solution.strip() or not final_answer.strip():
        raise ValueError("Empty GSM8K solution or final answer")

    return (
        [bos, special["[PROBLEM]"]]
        + tok.encode(question).ids
        + [special["[SOLUTION]"]]
        + tok.encode(solution.strip()).ids
        + [special["[ANSWER]"]]
        + tok.encode(final_answer.strip()).ids
        + [eos]
    )


# 5. Build individual sequences
def build_sequences(theory_docs, math_examples):
    records = []

    for doc in theory_docs:
        tokens = [bos] + tok.encode(doc.text).ids + [eos]

        records.append({
            "type": "theory",
            "id": doc.doc_id,
            "tokens": tokens,
        })

    for index, example in enumerate(math_examples):
        tokens = encode_gsm8k(example)

        records.append({
            "type": "gsm8k",
            "id": f"gsm8k-{index:05d}",
            "source": example.get("source"),
            "license": example.get("license"),
            "tokens": tokens,
        })

    return records


train_records = build_sequences(train_docs, gsm_train)
val_records = build_sequences(val_docs, gsm_val)


# 6. Validate sequence lengths and token IDs
def validate_records(name, records):
    lengths = []

    for record in records:
        tokens = record["tokens"]

        if not tokens:
            raise ValueError(f"Empty sequence in {name}")

        if len(tokens) > BLOCK_SIZE:
            raise ValueError(
                f"Sequence {record['id']} has {len(tokens)} tokens; "
                f"maximum is {BLOCK_SIZE}"
            )

        if min(tokens) < 0 or max(tokens) >= tok.get_vocab_size():
            raise ValueError(f"Invalid token ID in {record['id']}")

        lengths.append(len(tokens))

    print(f"\n{name}:")
    print(f"  Examples: {len(records):,}")
    print(f"  Mean length: {np.mean(lengths):.1f}")
    print(f"  Median length: {np.median(lengths):.1f}")
    print(f"  Maximum length: {max(lengths):,}")


validate_records("Training", train_records)
validate_records("Validation", val_records)


# 7. Save individual sequences
def save_sequences(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


OUT.mkdir(parents=True, exist_ok=True)

save_sequences(OUT / "train_sequences.jsonl", train_records)
save_sequences(OUT / "val_sequences.jsonl", val_records)


# 8. Keep the existing binary files for now
def pack_sequences(records):
    flat = [
        token
        for record in records
        for token in record["tokens"]
    ]
    return np.asarray(flat, dtype=np.uint16)


pack_sequences(train_records).tofile(OUT / "train.bin")
pack_sequences(val_records).tofile(OUT / "val.bin")

write_split_manifest(
    train_docs,
    val_docs,
    OUT / "split_manifest.jsonl",
)


# 9. Final report
print("\nDataset preparation completed.")
print(f"Vocabulary: {tok.get_vocab_size():,}")
print(f"Block size: {BLOCK_SIZE:,}")

print(f"Training theory documents: {len(train_docs)}")
print(f"Validation theory documents: {len(val_docs)}")
print(f"GSM8K training examples: {len(gsm_train):,}")
print(f"GSM8K validation examples: {len(gsm_val):,}")

print("\nSaved individual sequences:")
print(OUT / "train_sequences.jsonl")
print(OUT / "val_sequences.jsonl")

print("\nOfficial GSM8K test remains excluded.")
