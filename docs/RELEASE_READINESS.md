# GPT-0 Math release readiness

Audit date: 2026-11-01

## Status

**Code-only candidate prepared for owner review; not published.** The owner's current decision is to release source code under Apache License 2.0 while keeping all datasets, derived data, model weights, reports, and experiment outputs private. Publication is not authorized. The existing repository has no commits and no tracked files, so a later owner-controlled release candidate still needs a curated source inventory and explicit approval.

## Checks completed

- Reviewed Python package modules, training, evaluation, data preparation and download scripts, configs, tests, reports, requirements, README, baseline record, data notes, license file, ignore rules, and artifact inventory.
- Confirmed the working tree has no commits and all project files are currently untracked on `master`. No files were staged, committed, tagged, pushed, or published.
- Expanded `.gitignore` for a code-only release: it excludes both private corpus directories and manifest, arithmetic dataset, GSM8K benchmark data, processed data, tokenizer, all checkpoints and common model-weight extensions, all reports/results, W&B and TensorBoard runs, and temporary/local files. The public example manifest template remains available.
- Calculated `checkpoints/best.pt` SHA-256 as `fa1c2449f33ad9b631edea3f418ea3d9005c41ca593f4bee8bc1d2fca3beb85d`, matching the supplied verified hash. The 206,312,814-byte checkpoint remains excluded.
- Counted current corpus and historical sequences without rewriting them: 18 current documents (11 English, 7 Italian); historical theory sequences contain 9 English train and 2 English validation documents, no Italian; GSM8K sequences contain 6,725 train and 748 validation examples.
- Confirmed the current pipeline would split all 18 corpus entries into 9 English + 6 Italian train and 2 English + 1 Italian validation documents. The scripts that create this split overwrite ignored tokenizer and processed files, so they were not run.
- Scanned project reports/configuration/logs for machine-specific paths. Five arithmetic JSON report metadata paths were changed to repository-relative paths; metrics and recorded hashes were preserved. A local W&B run also contains Windows paths but is excluded by `.gitignore`.
- Searched text files for common secret markers and private-key headers. No credentials were identified in the scan. This is a pattern scan, not proof that no secret exists.
- Found unrestricted pickle loading for PyTorch checkpoints in the training, evaluation, and diagnostic scripts. Replaced those loads with a shared restricted loader using `weights_only=True` and a narrow allowlist for the NumPy RNG state stored in project checkpoints. The historical checkpoint loads successfully through this restricted path; it was not modified.
- Ran the existing suite: `49 passed in 4.80s` using `.venv\Scripts\python.exe -m pytest -q --basetemp .pytest_cache\code-only-audit`.
- Ran the private-data-free subset: `40 passed in 4.31s`; it covers tokenizer/model behavior and arithmetic dataset validation/collation using temporary fixtures.
- Confirmed `scripts/train_tokenizer.py` requires `data/raw/english/problem_solving/gsm8k/train.jsonl`. `scripts/download_gsm8k.py` downloads both train and test splits and writes local files. `scripts/prepare_data.py` then rewrites `data/processed/` files and the tokenizer pipeline also overwrites `data/tokenizer.json`.
- Verified all 88 files currently present under private dataset, checkpoint, report, and result paths are ignored; zero were unignored. `git ls-files` and `git diff --cached --name-only` both return no paths, so no private files are tracked or staged. The local private directories remain present.
- Recomputed the main checkpoint, processed-sequence, tokenizer, arithmetic test-split, sanitized GSM8K summary, and arithmetic evaluation-report hashes after testing; all match the values recorded before this code-only preparation. The arithmetic report still records 744/750 ID and 0/10,000 OOD; the GSM8K summary still records 0/100 validation and 0/20 official test.

## Findings and remaining blockers

1. **No publication approval.** The owner explicitly prohibited automatic publication; no Git or publishing operations have been performed. Obtain approval before any publication-related operation.
2. **Initial Git history is absent.** The workspace has no commits and all source files are untracked. The owner must choose and review the initial source-only repository inventory when ready.
3. **Private data and model decisions remain separate.** Dataset licenses and checkpoint distribution are not resolved for any future data/model release. This does not block the code-only scope because those artifacts are excluded.
4. **Historical training provenance is incomplete.** No original GPT-0 language-model training log was located. The checkpoint lacks hashes for historical processed sequences and tokenizer, so exact training data cannot be cryptographically established from the checkpoint alone.
5. **GSM8K evaluation is limited.** Only 100 validation and 20 of 1,319 official test examples were evaluated. The result is not a full benchmark score.
6. **Tests are partly data-dependent.** Nine tests in `tests/test_arithmetic_dataset.py` require the private arithmetic dataset. The remaining 40 tests pass without private files; see `docs/DATA_AVAILABILITY.md`.
7. **Synthetic data generator excluded.** `.gitignore` excludes `scripts/generate_arithmetic.py` because it can recreate the private deterministic default split. It has no import/dependency callers in the other source scripts.
8. **PyTorch install is hardware-specific.** It is intentionally installed separately via the official selector; PyTorch 2.6 or newer is required by the restricted checkpoint loader. The `datasets` dependency required by the optional GSM8K download helper is declared in `requirements.txt`.

## Public repository structure recommendation

The owner's code-only candidate should contain source, tests, configs, and documentation only:

```text
README.md
GPT0_BASELINE.md
LICENSE
requirements.txt
configs/
data/README.md
docs/EXPERIMENTAL_PROVENANCE.md
docs/DATA_AVAILABILITY.md
docs/EXPERIMENTAL_RESULTS.md
docs/PUBLIC_RELEASE_INVENTORY.md
docs/RELEASE_READINESS.md
gpt0/
scripts/
tests/
```

## Keep excluded

- `data/raw/english/`, `data/raw/italian/`, and `data/raw/manifest.jsonl`.
- `data/arithmetic/`, `data/benchmarks/`, `data/processed/`, and `data/tokenizer.json`.
- `checkpoints/` and `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, and `*.bin` files.
- `reports/` and `results/` in full; the code-only release does not include historical result files.
- `.venv/`, `__pycache__/`, `.pytest_cache/`, `wandb/`, `runs/`, logs, temporary files, and local environment files.
- Any future local-machine paths, credentials, personal files, or unreviewed large artifacts.

The current ignore rules cover the categories above. Git ignore rules do not remove files already tracked; `git ls-files` must be reviewed in the eventual release candidate.

## CLI and reproduction notes

The language-model preparation order is: provide the private corpus and permitted GSM8K training data; optionally use `python scripts/download_gsm8k.py` (downloads train and test); run `python scripts/train_tokenizer.py`; run `python scripts/prepare_data.py`; then run `python scripts/train.py`. These inputs and outputs are not part of the code-only release. The preparation commands rewrite tokenizer/processed data. Do not run them in the owner's preserved research copy.

Training uses `--output-dir` to select the checkpoint directory. Always choose a new directory, for example `python scripts/train.py --max-iters 20 --output-dir checkpoints/smoke_test`; the default directory contains the historical `best.pt` and `latest.pt` names.

The arithmetic pipeline is separate: `scripts/train_arithmetic.py` and `scripts/evaluate_arithmetic.py`. Training/evaluation require data or checkpoints that are excluded from the code-only release. The generator is withheld because it can recreate the deterministic default dataset; byte-for-byte historical reproduction is not claimed. Do not use arithmetic results to describe GSM8K or language-model results.

## Suggested owner release procedure

1. Prepare a clean code-only release candidate without touching the preserved research copy.
2. Inspect `git ls-files`, `git check-ignore`, and the complete proposed source inventory. Confirm no private dataset, checkpoint, report, or result file is included.
3. Run the dataset-free test subset and documented CLI help commands; do not run training as a release check.
4. Review the complete diff and confirm historical checkpoint/data hashes and metrics remain unchanged.
5. Stop for the owner's explicit approval before any Git publication operation.

## Files changed by this audit

- `.gitignore`
- `GPT0_BASELINE.md`
- `README.md`
- `data/README.md`
- `docs/DATA_AVAILABILITY.md`
- `data/raw/manifest.jsonl` (clarified that corpus distribution terms are unresolved; text content and corpus hashes were not changed)
- `requirements.txt`
- `scripts/train.py`
- `gpt0/checkpoints.py`; restricted loading is used by `scripts/analyze_numeric_tokens.py`, `scripts/audit_probe.py`, `scripts/diagnose_dataset.py`, `scripts/diagnose_first_tokens.py`, `scripts/diagnose_generation.py`, `scripts/diagnose_teacher_forcing.py`, `scripts/diagnose_teacher_forcing_sequence.py`, `scripts/evaluate.py`, `scripts/evaluate_answer_weighted.py`, `scripts/evaluate_generalization.py`, `scripts/evaluate_generalization_weighted.py`, `scripts/evaluate_oracle_reasoning.py`, `scripts/evaluate_oracle_reasoning_weighted.py`, `scripts/evaluate_overfit_32.py`, `scripts/evaluate_overfit_32_extended.py`, `scripts/evaluate_prompt_alignment.py`, `scripts/evaluate_teacher_forcing.py`, `scripts/evaluate_validation.py`, `scripts/overfit_32.py`, `scripts/overfit_32_extended.py`, `scripts/train_arithmetic.py`, `scripts/train_coverage.py`, and `scripts/train.py`.
- `docs/EXPERIMENTAL_PROVENANCE.md`
- `docs/RELEASE_READINESS.md`
- Five arithmetic JSON reports: machine-specific path metadata only; report metrics and hashes were preserved.

The five reports are `results/arithmetic/attention_head_robustness_v1.json`, `results/arithmetic/eq_attention_analysis_v1.json`, `results/arithmetic/evaluation_v2_20261101_103132.json`, `results/arithmetic/head_layer_changes_v1.json`, and `results/arithmetic/head_specialization_v1.json`.
