from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from gpt0.numeric_tokenizer import NumericTokenizer


class ArithmeticDataset(Dataset):
    """
    Dataset for autoregressive arithmetic training.

    The model receives:
        BOS + operand A + ADD + operand B + EQ + result + EOS

    Loss is calculated only on:
        result digits + EOS
    """

    VALID_SPLITS = {
        "train",
        "validation",
        "test_id",
        "test_ood",
    }

    def __init__(
        self,
        split: str,
        data_dir: str | Path | None = None,
    ) -> None:
        if split not in self.VALID_SPLITS:
            raise ValueError(f"Unsupported split: {split}")

        if data_dir is None:
            root = Path(__file__).resolve().parents[1]
            data_dir = root / "data" / "arithmetic" / "v1"

        self.data_dir = Path(data_dir)
        self.split = split
        self.path = self.data_dir / f"{split}.jsonl"

        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self.records = self._load_records()

    def _load_records(self) -> list[dict[str, Any]]:
        records = []

        with self.path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at line {line_number}"
                    ) from exc

                self._validate_record(record, line_number)
                records.append(record)

        if not records:
            raise ValueError(f"Dataset is empty: {self.path}")

        return records

    @staticmethod
    def _validate_record(
        record: dict[str, Any],
        line_number: int,
    ) -> None:
        required = {
            "a",
            "b",
            "result",
            "input",
            "target",
            "has_carry",
        }

        missing = required - record.keys()

        if missing:
            raise ValueError(
                f"Missing fields at line {line_number}: {missing}"
            )

        a = record["a"]
        b = record["b"]
        result = record["result"]

        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (a, b, result)
        ):
            raise TypeError(
                f"Operands and result must be integers at line {line_number}"
            )

        if a < 0 or b < 0:
            raise ValueError(
                f"Negative operands at line {line_number}"
            )

        if result != a + b:
            raise ValueError(
                f"Incorrect arithmetic result at line {line_number}"
            )

        expected_input = (
            ["<BOS>"]
            + list(str(a))
            + ["<ADD>"]
            + list(str(b))
            + ["<EQ>"]
        )

        expected_target = list(str(result)) + ["<EOS>"]

        if record["input"].split() != expected_input:
            raise ValueError(
                f"Incorrect input encoding at line {line_number}"
            )

        if record["target"].split() != expected_target:
            raise ValueError(
                f"Incorrect target encoding at line {line_number}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]

        prompt = NumericTokenizer.encode_input(
            record["a"],
            record["b"],
        )

        response = NumericTokenizer.encode_target(
            record["result"],
        )

        full_sequence = prompt + response

        # Next-token prediction:
        # input at position t predicts token at position t+1.
        input_ids = full_sequence[:-1]
        labels = full_sequence[1:]

        # The first response token is at full_sequence[len(prompt)].
        # Its corresponding label index is len(prompt) - 1.
        labels = labels.copy()
        labels[:len(prompt) - 1] = [-100] * (len(prompt) - 1)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def arithmetic_collate_fn(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """
    Pad variable-length arithmetic sequences.

    Input padding uses PAD_ID=0.
    Target padding uses -100, ignored by CrossEntropyLoss.
    """

    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    max_length = max(
        item["input_ids"].numel()
        for item in batch
    )

    batch_size = len(batch)

    input_ids = torch.full(
        (batch_size, max_length),
        NumericTokenizer.PAD,
        dtype=torch.long,
    )

    labels = torch.full(
        (batch_size, max_length),
        -100,
        dtype=torch.long,
    )

    for index, item in enumerate(batch):
        inputs = item["input_ids"]
        targets = item["labels"]

        if inputs.numel() != targets.numel():
            raise ValueError(
                "Input and label sequence lengths must match."
            )

        length = inputs.numel()

        input_ids[index, :length] = inputs
        labels[index, :length] = targets

    return {
        "input_ids": input_ids,
        "labels": labels,
    }