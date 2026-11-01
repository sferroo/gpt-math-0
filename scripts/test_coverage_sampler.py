import json
import random
from pathlib import Path
from collections import Counter

SEED = 42
THEORY_RATIO = 0.10
BATCH_SIZE = 2

path = Path("data/processed/train_sequences.jsonl")

records = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

theory = [
    i for i, r in enumerate(records)
    if r["type"] == "theory"
]

gsm8k = [
    i for i, r in enumerate(records)
    if r["type"] == "gsm8k"
]

if not theory or not gsm8k:
    raise ValueError("Dataset incompleto")

theory_draws = round(len(gsm8k) * THEORY_RATIO / (1 - THEORY_RATIO))

rng = random.Random(SEED)

gsm_order = gsm8k.copy()
rng.shuffle(gsm_order)

theory_order = [
    rng.choice(theory)
    for _ in range(theory_draws)
]

combined = gsm_order + theory_order
rng.shuffle(combined)

if len(combined) % BATCH_SIZE:
    raise ValueError("Numero di campioni non divisibile per batch size")

counts = Counter(combined)

assert len(gsm_order) == len(gsm8k)
assert all(counts[i] == 1 for i in gsm8k)
assert sum(counts[i] for i in theory) == theory_draws

print("========== COVERAGE SAMPLER ==========")
print("GSM8K examples:", len(gsm8k))
print("Theory examples:", len(theory))
print("Theory repetitions:", theory_draws)
print("Total samples:", len(combined))
print("Batch size:", BATCH_SIZE)
print("Training updates:", len(combined) // BATCH_SIZE)
print("Unique GSM8K covered:", sum(counts[i] > 0 for i in gsm8k))
print("GSM8K repeated:", sum(counts[i] > 1 for i in gsm8k))
print("GSM8K missing:", sum(counts[i] == 0 for i in gsm8k))
print("Theory unique covered:", sum(counts[i] > 0 for i in theory))
print("Theory total exposures:", sum(counts[i] for i in theory))
print("Theory fraction:", f"{theory_draws / len(combined):.2%}")
print("All assertions passed.")
