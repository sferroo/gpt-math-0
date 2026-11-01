from __future__ import annotations
import hashlib, json, random
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Document:
    doc_id: str
    path: str
    language: str
    source: str
    license: str
    text: str
    text_sha256: str

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()

def load_documents(manifest_path: str | Path) -> list[Document]:
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent.resolve()
    docs, ids, hashes = [], set(), set()
    for line_no, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
        required = ("id", "path", "language", "source", "license")
        missing = [k for k in required if not row.get(k)]
        if missing:
            raise ValueError(f"Missing fields on line {line_no}: {missing}")
        if row["id"] in ids:
            raise ValueError(f"Duplicate document id: {row['id']}")
        if row["language"] not in {"en", "it"}:
            raise ValueError(f"Unsupported language: {row['language']}")
        path = (root / row["path"]).resolve()
        if root not in path.parents:
            raise ValueError(f"Path escapes manifest directory: {row['path']}")
        if not path.is_file():
            raise FileNotFoundError(path)
        text = normalize_text(path.read_text(encoding="utf-8-sig"))
        if not text:
            raise ValueError(f"Empty document: {path}")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in hashes:
            continue
        ids.add(row["id"]); hashes.add(digest)
        docs.append(Document(row["id"], row["path"], row["language"],
                             row["source"], row["license"], text, digest))
    if len(docs) < 4:
        raise ValueError("At least four unique documents are required for a train/validation split.")
    return docs

def split_documents(docs: list[Document], val_fraction: float = 0.2, seed: int = 42):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    rng = random.Random(seed)
    train, val = [], []
    for lang in ("en", "it"):
        group = [d for d in docs if d.language == lang]
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_fraction)) if len(group) > 1 else 0
        val.extend(group[:n_val]); train.extend(group[n_val:])
    rng.shuffle(train); rng.shuffle(val)
    if not train or not val:
        raise ValueError("Split produced an empty train or validation set")
    return train, val

def write_split_manifest(train, val, path):
    with Path(path).open("w", encoding="utf-8") as f:
        for split, docs in (("train", train), ("validation", val)):
            for d in docs:
                f.write(json.dumps({"id": d.doc_id, "split": split, "language": d.language,
                                    "path": d.path, "sha256": d.text_sha256}, ensure_ascii=False)+"\n")
