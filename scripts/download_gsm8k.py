import json
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "data" / "raw" / "english" / "problem_solving" / "gsm8k"
TEST_DIR = ROOT / "data" / "benchmarks" / "gsm8k"

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)


def save_jsonl(dataset, path):
    count = 0

    with path.open("w", encoding="utf-8") as file:
        for item in dataset:
            record = {
                "question": item["question"],
                "answer": item["answer"],
                "source": "openai/gsm8k",
                "license": "MIT",
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def main():
    print("Downloading GSM8K from Hugging Face...")

    dataset = load_dataset("openai/gsm8k", "main")

    train_count = save_jsonl(
        dataset["train"],
        TRAIN_DIR / "train.jsonl",
    )

    test_count = save_jsonl(
        dataset["test"],
        TEST_DIR / "test.jsonl",
    )

    metadata = {
        "name": "GSM8K",
        "source": "https://huggingface.co/datasets/openai/gsm8k",
        "license": "MIT",
        "language": "en",
        "training_examples": train_count,
        "test_examples": test_count,
        "test_used_for_training": False,
    }

    with (TRAIN_DIR / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nGSM8K download completed.")
    print(f"Training examples: {train_count}")
    print(f"Test examples: {test_count}")
    print(f"Training: {TRAIN_DIR}")
    print(f"Benchmark: {TEST_DIR}")


if __name__ == "__main__":
    main()
