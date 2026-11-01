
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class MathSequenceDataset(Dataset):
    def __init__(self, path):
        self.path = Path(path)
        self.records = []

        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    self.records.append(record)

        if not self.records:
            raise ValueError(f"Dataset is empty: {self.path}")

        for record in self.records:
            tokens = record["tokens"]

            if len(tokens) < 2:
                raise ValueError(
                    f"Sequence too short: {record.get('id')}"
                )

            if tokens[0] != 2 or tokens[-1] != 3:
                raise ValueError(
                    f"Invalid BOS/EOS: {record.get('id')}"
                )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def collate_math_sequences(batch, pad_token_id=0):
    """
    Convert independent sequences into a dynamically padded batch.

    Inputs:
        tokens[:-1]

    Targets:
        tokens[1:]

    Padding targets receive -100 and are ignored by cross entropy.
    """

    inputs = [
        torch.tensor(record["tokens"][:-1], dtype=torch.long)
        for record in batch
    ]

    targets = [
        torch.tensor(record["tokens"][1:], dtype=torch.long)
        for record in batch
    ]

    max_length = max(len(x) for x in inputs)
    batch_size = len(batch)

    x = torch.full(
        (batch_size, max_length),
        pad_token_id,
        dtype=torch.long,
    )

    y = torch.full(
        (batch_size, max_length),
        -100,
        dtype=torch.long,
    )

    for i, (input_ids, target_ids) in enumerate(
        zip(inputs, targets)
    ):
        length = len(input_ids)
        x[i, :length] = input_ids
        y[i, :length] = target_ids

    return x, y


def create_train_loader(
    path,
    batch_size=2,
    theory_probability=0.10,
    pad_token_id=0,
    seed=42,
):
    if not 0.0 <= theory_probability <= 1.0:
        raise ValueError("theory_probability must be between 0 and 1")

    dataset = MathSequenceDataset(path)

    theory_indices = [
        i for i, record in enumerate(dataset.records)
        if record["type"] == "theory"
    ]

    gsm8k_indices = [
        i for i, record in enumerate(dataset.records)
        if record["type"] == "gsm8k"
    ]

    if not theory_indices or not gsm8k_indices:
        raise ValueError("Both theory and GSM8K examples are required")

    weights = []

    for record in dataset.records:
        if record["type"] == "theory":
            weight = theory_probability / len(theory_indices)
        elif record["type"] == "gsm8k":
            weight = (1.0 - theory_probability) / len(gsm8k_indices)
        else:
            raise ValueError(f"Unknown sequence type: {record['type']}")

        weights.append(weight)

    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=lambda batch: collate_math_sequences(
            batch,
            pad_token_id=pad_token_id,
        ),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return loader


def create_validation_loader(
    path,
    batch_size=2,
    pad_token_id=0,
):
    dataset = MathSequenceDataset(path)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_math_sequences(
            batch,
            pad_token_id=pad_token_id,
        ),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
