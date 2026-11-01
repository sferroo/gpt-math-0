# Data and artifact availability

## First-release scope

The planned first public release is code-only. Source code is released under Apache License 2.0. The private research inputs, derived data, model weights, and local experiment records listed below are not part of that release. The GPT-0 checkpoint is explicitly **not included in the first release**; its distribution decision is postponed.

The deterministic arithmetic generator is excluded from the public candidate; see `EXPERIMENTAL_RESULTS.md` and `PUBLIC_RELEASE_INVENTORY.md` for release scope.

These exclusions are owner decisions. They do not indicate that the local working copy is being cleaned: its private files and historical artifacts remain in place. `.gitignore` prevents their accidental addition to a release candidate.

## Excluded local paths

| Material | Local paths protected by `.gitignore` | Reason |
|---|---|---|
| Educational corpus (18 documents) and its corpus manifest | `data/raw/english/`, `data/raw/italian/`, `data/raw/manifest.jsonl` | Owner is keeping corpus material private. The example manifest template remains source documentation. |
| Project-generated synthetic arithmetic dataset and metadata | `data/arithmetic/` | Owner is keeping the generated dataset private. |
| Original GSM8K training files | `data/raw/english/` (including the local GSM8K subdirectory) | External benchmark files are excluded from the code release. |
| GSM8K official test files | `data/benchmarks/` | External benchmark data is excluded. |
| Processed sequences, split files, and binary token arrays | `data/processed/` | These can contain private corpus text and GSM8K questions/answers. |
| Trained tokenizer | `data/tokenizer.json` | Derived from private corpus and benchmark training material. |
| Checkpoints and model weights | `checkpoints/`, plus `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, and `*.bin` anywhere in the repository | Checkpoint distribution is postponed; all local weights stay private. |
| Reports and experiment outputs | `reports/`, `results/`, plus already excluded `runs/` and `wandb/` | The planned release is code-only. Historical experiment records remain local and unchanged. |

The ignored directories and files remain available in the owner's local research copy. Git ignore rules do not untrack files that were previously committed, so a release candidate must be checked with `git ls-files` as well as `git check-ignore`.

## Commands and their prerequisites

The following language-model pipeline needs local data that is not included in the code-only release:

1. `python scripts/train_tokenizer.py` reads `data/raw/manifest.jsonl`, every corpus document referenced by that manifest, and `data/raw/english/problem_solving/gsm8k/train.jsonl`. It writes the tokenizer and GSM8K splits.
2. `python scripts/prepare_data.py` reads the corpus manifest/documents, `data/tokenizer.json`, and the processed GSM8K train/validation splits. It rewrites files under `data/processed/`.
3. `python scripts/train.py` reads the tokenizer and processed train/validation sequences, then writes checkpoints.
4. `python scripts/evaluate.py` needs a GPT-0 checkpoint, tokenizer, and `data/benchmarks/gsm8k/test.jsonl`.

`python scripts/download_gsm8k.py` requires network access and the `datasets` dependency. It downloads both GSM8K training and official test data to ignored local paths. Downloading the official test split is not required for tokenizer training or model training. Never use official test examples for training, fine-tuning, hyperparameter tuning, or model selection. Keep benchmark evaluation separate from the training workflow.

Downloading GSM8K does not grant permission to redistribute its files. Users must comply with the upstream dataset license and terms. The downloaded files are not part of this public release.

The separate arithmetic pipeline uses `data/arithmetic/v1/`:

- `scripts/train_arithmetic.py` needs the arithmetic training and validation splits. It writes checkpoints and a training log.
- `scripts/evaluate_arithmetic.py` needs the ID and OOD test splits; a model checkpoint is optional for a random-initialization baseline.
- `scripts/validate_arithmetic_dataset.py` needs the metadata and all four dataset splits.
- `scripts/analyze_eq_attention_dataset.py` also reads the arithmetic dataset. Other arithmetic analysis/plot scripts such as `scripts/analyze_eq_attention.py`, `scripts/analyze_head_layer_changes.py`, `scripts/analyze_head_specialization.py`, `scripts/analyze_attention_head_robustness.py`, and `scripts/plot_attention.py` consume local files under `results/arithmetic/` and produce local result files.

The GPT-0 language-model diagnostics and variant trainers/evaluators also refer to private processed sequences, tokenizer, and/or checkpoints. This includes `scripts/analyze_numeric_tokens.py`, `scripts/audit_probe.py`, `scripts/audit_token_distribution.py`, `scripts/check_split.py`, the `scripts/diagnose_*.py` scripts, `scripts/evaluate.py`, `scripts/evaluate_answer_weighted.py`, `scripts/evaluate_generalization*.py`, `scripts/evaluate_oracle_reasoning*.py`, `scripts/evaluate_overfit_32*.py`, `scripts/evaluate_prompt_alignment.py`, `scripts/evaluate_teacher_forcing.py`, `scripts/evaluate_validation.py`, `scripts/overfit_32.py`, `scripts/overfit_32_extended.py`, `scripts/test_coverage_sampler.py`, `scripts/train_answer_weighted.py`, and `scripts/train_coverage.py`. These commands are not runnable as configured in a code-only checkout. Inspect the paths in a script before adapting it to separately obtained data.

`scripts/generate_arithmetic.py` is excluded from the public candidate because it can recreate the deterministic default split. The local script remains in the research working copy and is ignored by Git. The historical dataset and its generated files remain private; byte-for-byte reproduction of the historical files is not claimed.

No other public script imports or depends on this generator.

`scripts/organize_corpus.py` reads the private corpus and manifest. Its default is a dry run; `--apply` moves corpus documents and rewrites the manifest. It is not needed for the code-only release and must not be run in the owner's working copy.

## Tests

The full test suite currently passes in the owner's local copy because the private arithmetic dataset is present: 49 tests pass. Nine tests in `tests/test_arithmetic_dataset.py` load that local dataset and will not run in a code-only clone without it.

The following test selection collects 40 tests and does not require any private dataset, tokenizer, or checkpoint. In the verified Python 3.13 CPU environment, 39 tests passed and one CUDA-dependent test was skipped because CUDA was unavailable.

```bash
python -m pytest -q \
  tests/test_numeric_tokenizer.py \
  tests/test_arithmetic_model.py \
  tests/test_model.py \
  tests/test_arithmetic_dataset.py::test_collate_pads_inputs_and_masks_labels \
  tests/test_arithmetic_dataset.py::test_collate_rejects_empty_batch \
  tests/test_arithmetic_dataset.py::test_dataset_rejects_incorrect_result \
  tests/test_arithmetic_dataset.py::test_dataset_rejects_invalid_split
```

To run the full suite, first verify that the four local arithmetic JSONL split files exist. The tests do not download data. No training or evaluation script is run by pytest.

## Verify prerequisites without reading dataset contents

PowerShell checks file presence only:

```powershell
@(
  'data/raw/manifest.jsonl',
  'data/raw/english/problem_solving/gsm8k/train.jsonl',
  'data/tokenizer.json',
  'data/processed/train_sequences.jsonl',
  'data/processed/val_sequences.jsonl',
  'data/benchmarks/gsm8k/test.jsonl',
  'data/arithmetic/v1/metadata.json',
  'data/arithmetic/v1/train.jsonl',
  'data/arithmetic/v1/validation.jsonl',
  'data/arithmetic/v1/test_id.jsonl',
  'data/arithmetic/v1/test_ood.jsonl',
  'checkpoints/best.pt'
) | ForEach-Object { '{0}: {1}' -f $_, (Test-Path -LiteralPath $_) }
```

Before preparing a code-only release, inspect the index and verify representative private paths are ignored:

```powershell
git ls-files
git check-ignore -v data/raw/english/example.txt data/raw/italian/example.txt data/raw/manifest.jsonl data/arithmetic/v1/train.jsonl data/processed/train_sequences.jsonl data/tokenizer.json data/benchmarks/gsm8k/test.jsonl checkpoints/best.pt reports/example.json results/example.json
```

The example paths in `git check-ignore` need not exist; it checks the matching rules without reading any private data. A real release audit must also review every path returned by `git ls-files`.
