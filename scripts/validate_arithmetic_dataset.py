from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "arithmetic" / "v1"

EXPECTED = {
    "train": 3800,
    "validation": 500,
    "test_id": 750,
    "test_ood": 10000,
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def calculate_carry(a: int, b: int) -> bool:
    carry = 0

    while a > 0 or b > 0:
        total = a % 10 + b % 10 + carry
        carry = int(total >= 10)

        if carry:
            return True

        a //= 10
        b //= 10

    return False


def validate_record(record: dict, split: str) -> None:
    a = record["a"]
    b = record["b"]
    result = record["result"]

    assert isinstance(a, int) and a >= 0
    assert isinstance(b, int) and b >= 0
    assert isinstance(result, int)
    assert result == a + b

    expected_input = (
        ["<BOS>"]
        + list(str(a))
        + ["<ADD>"]
        + list(str(b))
        + ["<EQ>"]
    )

    expected_target = list(str(result)) + ["<EOS>"]

    assert record["input"].split() == expected_input
    assert record["target"].split() == expected_target

    assert record["has_carry"] == calculate_carry(a, b)

    if split in {"train", "validation", "test_id"}:
        assert 0 <= a <= 99
        assert 0 <= b <= 99

    if split == "test_ood":
        assert 100 <= a <= 9999
        assert 100 <= b <= 9999


def main() -> None:
    metadata_path = DATA_DIR / "metadata.json"
    assert metadata_path.exists(), "Missing metadata.json"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["version"] == "1.0"
    assert metadata["operation"] == "addition"

    datasets = {}

    for split, expected_count in EXPECTED.items():
        path = DATA_DIR / f"{split}.jsonl"
        assert path.exists(), f"Missing {path.name}"

        records = load_jsonl(path)
        assert len(records) == expected_count, (
            f"{split}: expected {expected_count}, got {len(records)}"
        )

        for record in records:
            validate_record(record, split)

        datasets[split] = records

    # Check uniqueness and disjointness of all ID pairs.
    id_pairs = {}

    for split in ("train", "validation", "test_id"):
        pairs = [(r["a"], r["b"]) for r in datasets[split]]

        assert len(pairs) == len(set(pairs)), (
            f"Duplicate pairs inside {split}"
        )

        id_pairs[split] = set(pairs)

    assert id_pairs["train"].isdisjoint(id_pairs["validation"])
    assert id_pairs["train"].isdisjoint(id_pairs["test_id"])
    assert id_pairs["validation"].isdisjoint(id_pairs["test_id"])

    # Check OOD pair uniqueness.
    ood_pairs = [
        (r["a"], r["b"])
        for r in datasets["test_ood"]
    ]
    assert len(ood_pairs) == len(set(ood_pairs))

    # Check that all 10,000 possible OOD operand values
    # are not required, but every sampled pair is outside ID.
    for a, b in ood_pairs:
        assert a > 99 and b > 99

    print("ARITHMETIC DATASET VALIDATION PASSED")
    print("-----------------------------------")

    for split, records in datasets.items():
        carries = sum(r["has_carry"] for r in records)
        percentage = 100 * carries / len(records)

        print(
            f"{split:12} "
            f"samples={len(records):5} "
            f"carry={carries:5} "
            f"({percentage:.2f}%)"
        )

    print("\nNo arithmetic errors detected.")
    print("No duplicate pairs inside splits.")
    print("No overlap between ID splits.")
    print("OOD operands are outside the training range.")


if __name__ == "__main__":
    main()