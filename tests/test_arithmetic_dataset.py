import json

import pytest
import torch
from torch.utils.data import DataLoader

from gpt0.arithmetic_dataset import (
    ArithmeticDataset,
    arithmetic_collate_fn,
)
from gpt0.numeric_tokenizer import NumericTokenizer


def write_record(path, a, b):
    result = a + b

    record = {
        "a": a,
        "b": b,
        "result": result,
        "input": " ".join(
            ["<BOS>"]
            + list(str(a))
            + ["<ADD>"]
            + list(str(b))
            + ["<EQ>"]
        ),
        "target": " ".join(list(str(result)) + ["<EOS>"]),
        "has_carry": False,
    }

    path.write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )


def make_dataset(tmp_path, records):
    path = tmp_path / "train.jsonl"

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    return ArithmeticDataset("train", data_dir=tmp_path)


def test_load_real_training_dataset():
    dataset = ArithmeticDataset("train")

    assert len(dataset) == 3800


def test_load_real_validation_dataset():
    dataset = ArithmeticDataset("validation")

    assert len(dataset) == 500


def test_load_real_test_id_dataset():
    dataset = ArithmeticDataset("test_id")

    assert len(dataset) == 750


def test_load_real_test_ood_dataset():
    dataset = ArithmeticDataset("test_ood")

    assert len(dataset) == 10000


def test_input_and_label_lengths_match(tmp_path):
    dataset = ArithmeticDataset("train")

    item = dataset[0]

    assert item["input_ids"].ndim == 1
    assert item["labels"].ndim == 1
    assert item["input_ids"].shape == item["labels"].shape


def test_loss_is_masked_on_prompt(tmp_path):
    dataset = ArithmeticDataset("train")

    item = dataset[0]
    prompt = NumericTokenizer.encode_input(
        dataset.records[0]["a"],
        dataset.records[0]["b"],
    )

    # All context targets must be ignored.
    assert torch.all(item["labels"][:len(prompt) - 1] == -100)

    # First result digit must be supervised.
    assert item["labels"][len(prompt) - 1] >= 5


def test_eos_is_supervised():
    dataset = ArithmeticDataset("train")

    for index in range(20):
        item = dataset[index]
        supervised = item["labels"][item["labels"] != -100]

        assert supervised[-1].item() == NumericTokenizer.EOS


def test_no_prompt_tokens_are_supervised():
    dataset = ArithmeticDataset("train")

    for index in range(20):
        item = dataset[index]
        labels = item["labels"]

        supervised = labels[labels != -100]

        assert all(
            token >= NumericTokenizer.EOS
            for token in supervised
        )


def test_collate_pads_inputs_and_masks_labels():
    batch = [
        {
            "input_ids": torch.tensor([1, 12, 3, 4]),
            "labels": torch.tensor([-100, -100, 6, 2]),
        },
        {
            "input_ids": torch.tensor([1, 6, 7, 3, 8, 4]),
            "labels": torch.tensor([-100, -100, -100, -100, 9, 2]),
        },
    ]

    result = arithmetic_collate_fn(batch)

    assert result["input_ids"].shape == (2, 6)
    assert result["labels"].shape == (2, 6)

    assert torch.equal(
        result["input_ids"][0],
        torch.tensor([1, 12, 3, 4, 0, 0]),
    )

    assert torch.equal(
        result["labels"][0],
        torch.tensor([-100, -100, 6, 2, -100, -100]),
    )


def test_collate_rejects_empty_batch():
    with pytest.raises(ValueError):
        arithmetic_collate_fn([])


def test_dataloader_batch():
    dataset = ArithmeticDataset("train")

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=arithmetic_collate_fn,
    )

    batch = next(iter(loader))

    assert batch["input_ids"].shape[0] == 8
    assert batch["labels"].shape == batch["input_ids"].shape


def test_dataset_rejects_incorrect_result(tmp_path):
    record = {
        "a": 10,
        "b": 20,
        "result": 999,
        "input": "<BOS> 1 0 <ADD> 2 0 <EQ>",
        "target": "9 9 9 <EOS>",
        "has_carry": False,
    }

    with pytest.raises(ValueError):
        make_dataset(tmp_path, [record])


def test_dataset_rejects_invalid_split(tmp_path):
    with pytest.raises(ValueError):
        ArithmeticDataset("unknown", data_dir=tmp_path)