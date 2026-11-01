import json
import random
import sys
from pathlib import Path

from tokenizers import (
    Tokenizer,
    models,
    trainers,
    pre_tokenizers,
    decoders,
    normalizers,
)

# Add project root to Python module search path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpt0.data_pipeline import load_documents, split_documents

SEED = 42

# --------------------------------------------------
# 1. Load mathematical theory documents
# --------------------------------------------------

docs = load_documents(ROOT / "data/raw/manifest.jsonl")

train_docs, val_docs = split_documents(
    docs,
    val_fraction=0.2,
    seed=SEED,
)

# --------------------------------------------------
# 2. Load and split GSM8K training dataset
# --------------------------------------------------

gsm8k_path = (
    ROOT / "data/raw/english/problem_solving/gsm8k/train.jsonl"
)

with gsm8k_path.open("r", encoding="utf-8") as f:
    gsm8k_examples = [json.loads(line) for line in f if line.strip()]

rng = random.Random(SEED)
rng.shuffle(gsm8k_examples)

split_index = int(len(gsm8k_examples) * 0.9)

gsm8k_train = gsm8k_examples[:split_index]
gsm8k_val = gsm8k_examples[split_index:]


def format_gsm8k(example):
    """Convert a GSM8K example into a structured training sequence."""

    question = example["question"].strip()
    answer = example["answer"].strip()

    if "####" not in answer:
        raise ValueError("GSM8K answer is missing the #### separator")

    solution, final_answer = answer.rsplit("####", 1)

    return (
        "[PROBLEM]\n"
        + question
        + "\n[SOLUTION]\n"
        + solution.strip()
        + "\n[ANSWER]\n"
        + final_answer.strip()
    )


# Save the deterministic split for the data preparation stage.
split_dir = ROOT / "data/processed"
split_dir.mkdir(parents=True, exist_ok=True)

for filename, examples in (
    ("gsm8k_train.jsonl", gsm8k_train),
    ("gsm8k_val.jsonl", gsm8k_val),
):
    with (split_dir / filename).open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


# --------------------------------------------------
# 3. Configure BPE tokenizer
# --------------------------------------------------

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFC()
])

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=False
)

tokenizer.decoder = decoders.ByteLevel()

special_tokens = [
    "[PAD]",
    "[UNK]",
    "[BOS]",
    "[EOS]",
    "[PROBLEM]",
    "[SOLUTION]",
    "[ANSWER]",
]

trainer = trainers.BpeTrainer(
    vocab_size=16000,
    min_frequency=2,
    special_tokens=special_tokens,
)


# --------------------------------------------------
# 4. Train tokenizer on training data only
# --------------------------------------------------

def training_texts():
    for doc in train_docs:
        yield doc.text

    for example in gsm8k_train:
        yield format_gsm8k(example)


tokenizer.train_from_iterator(
    training_texts(),
    trainer=trainer,
)


# --------------------------------------------------
# 5. Save tokenizer
# --------------------------------------------------

out = ROOT / "data/tokenizer.json"
out.parent.mkdir(parents=True, exist_ok=True)

tokenizer.save(str(out))


# --------------------------------------------------
# 6. Verify special tokens
# --------------------------------------------------

print("\nTokenizer training completed.")
print(f"Tokenizer saved: {out}")
print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

print("\nSpecial tokens:")
for token in special_tokens:
    print(f"  {token}: {tokenizer.token_to_id(token)}")

print("\nDataset split:")
print(f"Theory training documents: {len(train_docs)}")
print(f"Theory validation documents excluded: {len(val_docs)}")
print(f"GSM8K training examples: {len(gsm8k_train)}")
print(f"GSM8K validation examples: {len(gsm8k_val)}")
print("GSM8K official test examples: excluded")
