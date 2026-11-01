# Public release candidate inventory

Inventory date: 2026-11-01. Classification applies recursively to listed directories; more-specific rows override their parent. This inventory names paths only and does not include private file contents. The owner must review the actual future index because this workspace currently has no commits and no tracked files.

| Exact path | Classification | Reason / unresolved concern |
|---|---|---|
| `.git/` | PRIVATE | Local repository metadata, not release content. |
| `.venv/` | PRIVATE | Local Python environment and installed dependencies; rebuild from requirements. |
| `.pytest_cache/` | PRIVATE | Local test cache. |
| `checkpoints/` | PRIVATE | All trained weights and checkpoints; GPT-0 distribution decision postponed. |
| `configs/` | PUBLIC | `small.yaml` and `arithmetic.yaml`; configuration only, paths require separately supplied data. |
| `data/` | REVIEW REQUIRED | Mixed public documentation/template and private data; only the specific public rows below may be included. |
| `data/README.md` | PUBLIC | Availability notice; no dataset examples. |
| `data/raw/` | REVIEW REQUIRED | Contains both public template and private corpus paths; include only the explicit template below. |
| `data/raw/manifest.example.jsonl` | PUBLIC | Placeholder template, not the private corpus manifest. |
| `data/raw/manifest.jsonl` | PRIVATE | Private corpus inventory and provenance metadata. |
| `data/raw/english/` | PRIVATE | Educational corpus and local GSM8K training files. |
| `data/raw/italian/` | PRIVATE | Educational corpus documents. |
| `data/arithmetic/` | PRIVATE | Synthetic arithmetic split files and metadata. |
| `data/benchmarks/` | PRIVATE | GSM8K official test data. |
| `data/processed/` | PRIVATE | Token sequences, split records, and binary arrays; may contain corpus/GSM8K content. |
| `data/tokenizer.json` | PRIVATE | Derived tokenizer artifact. |
| `docs/` | PUBLIC | Approved documentation, including sanitized aggregate results and availability/provenance caveats. |
| `gpt0/` | PUBLIC | Source code under Apache License 2.0. Dependencies are listed separately. |
| `.gitignore` | PUBLIC | Release exclusion rules. |
| `GPT0_BASELINE.md` | PUBLIC | Historical results and limitations; artifact paths are provenance references, not included files. |
| `LICENSE` | PUBLIC | Apache License 2.0 for source code only. It does not grant rights to excluded datasets or weights. |
| `README.md` | PUBLIC | Project, installation, results, and availability documentation. |
| `requirements.txt` | PUBLIC | Python dependencies. PyTorch is installed separately for hardware compatibility; third-party dependency licenses remain their own. |
| `reports/` | PRIVATE | Detailed reports and per-example GSM8K/question/generation artifacts. The sanitized aggregate has been reproduced in `docs/EXPERIMENTAL_RESULTS.md`; do not include the directory. |
| `results/` | PRIVATE | Local training/evaluation records, sample-level analyses, plots, and report artifacts. Only sanitized aggregate values are reproduced in documentation. |
| `runs/` | PRIVATE | Local TensorBoard event files. |
| `scripts/` | PUBLIC, with explicit exclusions | Reviewed source scripts. Scripts requiring private or external data are documented as such. |
| `scripts/organize_corpus.py` | PRIVATE | Manages the private educational corpus and can move corpus documents when invoked with `--apply`. Not required for the code-only release. |
| `scripts/generate_arithmetic.py` | PRIVATE | Excluded because its deterministic default can reconstruct the withheld arithmetic split. |
| `tests/` | PUBLIC | Source tests. The documented dataset-free selection collects 40 tests; 39 passed and one CUDA-dependent test was skipped in the verified CPU environment. Nine additional dataset tests require the private arithmetic split. |
| `wandb/` | PRIVATE | Local W&B run data includes machine-specific paths and experiment artifacts. |

## Release checks

- `.gitignore` excludes the private paths above, all common checkpoint extensions, reports, results, local runs, and the arithmetic generator.
- No negated ignore pattern is present. Keep these exclusions and review `git ls-files` before any future release preparation.
- `docs/EXPERIMENTAL_RESULTS.md` contains aggregate values only; its source records remain private.
- External GSM8K terms must be respected for any separate data release. The source-code Apache license does not cover GSM8K, the educational corpus, the synthetic dataset, or model weights.
