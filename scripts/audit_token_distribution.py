
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "TRAIN": ROOT / "data/processed/train_sequences.jsonl",
    "VALIDATION": ROOT / "data/processed/val_sequences.jsonl",
}

SPECIAL_TOKENS = {
    "[BOS]": "BOS",
    "[EOS]": "EOS",
    "[PROBLEM]": "PROMPT",
    "[SOLUTION]": "SOLUTION",
    "[ANSWER]": "ANSWER",
}

def read_records(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON non valido in {path}, riga {line_number}: {exc}"
                ) from exc
    return records


def get_token_ids(record):
    for key in ("input_ids", "tokens", "token_ids"):
        value = record.get(key)
        if isinstance(value, list):
            return value
    raise KeyError(
        f"Non trovo input_ids/tokens/token_ids. "
        f"Chiavi disponibili: {list(record.keys())}"
    )


def get_text(record):
    for key in ("text", "sequence", "content"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return None


def classify_sequence(ids, special_ids):
    counts = Counter()
    current_region = "OTHER"

    for token_id in ids:
        if token_id == special_ids.get("[PROBLEM]"):
            current_region = "PROMPT"
            continue
        if token_id == special_ids.get("[SOLUTION]"):
            current_region = "REASONING"
            continue
        if token_id == special_ids.get("[ANSWER]"):
            current_region = "ANSWER"
            continue
        if token_id == special_ids.get("[EOS]"):
            counts["EOS"] += 1
            current_region = "OTHER"
            continue
        if token_id == special_ids.get("[BOS]"):
            counts["BOS"] += 1
            continue

        counts[current_region] += 1

    return counts


def main():
    tokenizer_path = ROOT / "data/tokenizer.json"

    if not tokenizer_path.exists():
        raise FileNotFoundError(tokenizer_path)

    try:
        from tokenizers import Tokenizer
    except ImportError:
        raise RuntimeError(
            "Pacchetto tokenizers non installato nell'ambiente Python."
        )

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    special_ids = {}
    for token, _ in SPECIAL_TOKENS.items():
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Token speciale mancante: {token}")
        special_ids[token] = token_id

    print("=" * 72)
    print("GPT-0 MATH — TOKEN DISTRIBUTION AUDIT")
    print("=" * 72)

    for split_name, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(path)

        records = read_records(path)
        totals = Counter()
        token_frequency = Counter()
        answer_lengths = Counter()
        numeric_answers = 0
        answer_count = 0
        examples_without_answer = 0

        for record in records:
            ids = get_token_ids(record)
            region_counts = classify_sequence(ids, special_ids)
            totals.update(region_counts)

            # Conteggio dei token numerici nella risposta:
            # individua la sezione ANSWER nella sequenza originale.
            try:
                answer_start = ids.index(special_ids["[ANSWER]"]) + 1
                eos_id = special_ids["[EOS]"]
                answer_end = ids.index(eos_id, answer_start)
                answer_ids = ids[answer_start:answer_end]
            except ValueError:
                examples_without_answer += 1
                continue

            answer_count += 1
            token_frequency.update(answer_ids)
            answer_lengths[len(answer_ids)] += 1

            decoded = tokenizer.decode(answer_ids).strip()
            if re.search(r"\d", decoded):
                numeric_answers += 1

        total_region_tokens = sum(
            totals[k] for k in ("PROMPT", "REASONING", "ANSWER", "EOS")
        )

        print(f"\n{'-' * 72}")
        print(f"{split_name}: {path.relative_to(ROOT)}")
        print(f"Record: {len(records)}")
        print(f"Record senza ANSWER/EOS valido: {examples_without_answer}")

        print("\nDistribuzione per regione:")
        for region in ("PROMPT", "REASONING", "ANSWER", "EOS"):
            count = totals[region]
            percentage = (
                100 * count / total_region_tokens
                if total_region_tokens else 0
            )
            print(f"  {region:12s} {count:10,d} token  ({percentage:6.2f}%)")

        print("\nLunghezza delle risposte in token:")
        for length, count in sorted(answer_lengths.items()):
            print(f"  {length:3d} token: {count:6,d} risposte")

        print(
            f"\nRisposte contenenti cifre: {numeric_answers}/{answer_count}"
            f" ({100 * numeric_answers / answer_count if answer_count else 0:.2f}%)"
        )

        print("\nTop 20 token nelle risposte:")
        for token_id, count in token_frequency.most_common(20):
            token = tokenizer.id_to_token(token_id)
            print(f"  ID {token_id:5d} | {count:6,d} occorrenze | {token!r}")

    print("\n" + "=" * 72)
    print("AUDIT COMPLETATO — nessun file del progetto è stato modificato.")
    print("=" * 72)


if __name__ == "__main__":
    main()
