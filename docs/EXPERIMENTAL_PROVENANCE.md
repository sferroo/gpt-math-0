# GPT-0 Math experimental provenance

This document separates the present-day repository contents from the historical experiments. Metrics are retained as recorded; this document does not infer missing training history or dataset identity.

## Current educational corpus

`data/raw/manifest.jsonl` currently lists 18 AI-assisted educational documents: 11 in English and 7 in Italian. The material was prepared for this project. Its distribution license has not been decided, so the corpus must not be publicly redistributed until the owner resolves the rights and license terms. The source code's Apache-2.0 license does not cover it by default.

The current data pipeline loads all 18 documents. With its seed 42 and language-stratified split, the expected split is 9 English + 6 Italian training documents and 2 English + 1 Italian validation document.

## Historical processed training data

The existing ignored processed sequence files contain:

| Component | Training | Validation |
|---|---:|---:|
| English theory documents | 9 | 2 |
| Italian theory documents | 0 | 0 |
| GSM8K examples | 6,725 | 748 |

The current `data/processed/split_manifest.jsonl` contains only the 11 English theory documents. None of the seven Italian documents appears in the existing processed sequences. The current 18-document corpus and the split the present pipeline would generate are not the historical GPT-0 training data.

The checkpoint records paths to the processed sequence files and tokenizer. It does not record SHA-256 hashes of those data files. Consequently, the precise historical training data cannot be cryptographically established from the checkpoint alone. No original GPT-0 language-model training log has been identified; the two JSONL training logs in `results/arithmetic/` belong to the separate arithmetic experiment.

Do not run `scripts/train_tokenizer.py` or `scripts/prepare_data.py` in a copy containing the historical processed files: they rewrite ignored tokenizer and processed-data artifacts.

## GPT-0 language-model checkpoint

The historical checkpoint is `checkpoints/best.pt`. Its verified SHA-256 is:

`fa1c2449f33ad9b631edea3f418ea3d9005c41ca593f4bee8bc1d2fca3beb85d`

Recorded facts: training step 2,000; best validation loss 4.554548854685838; decoder-only Transformer with 6 layers, 6 attention heads, embedding size 384, vocabulary size 16,000, context length 1,024, and dropout 0.1; batch size 2, learning rate 0.0003, weight decay 0.1, and seed 42. The checkpoint includes model, optimizer, scheduler, scaler, and available RNG state. The checkpoint is excluded from the code-only first release; its distribution decision is postponed.

The reported GSM8K validation result was 0/100. The reported official test sample result was 0/20. The complete official test set has 1,319 examples, of which only 20 were evaluated. Validation generation was repetitive and did not produce valid final answers in the reported evaluation. These results are not a complete GSM8K benchmark evaluation. See `reports/gsm8k_summary.json`, the sanitized aggregate report.

## Arithmetic experiment

This is a separate, project-generated synthetic addition task using digit-level tokens; it is not the GSM8K language-model experiment.

| Split | Examples |
|---|---:|
| Training | 3,800 |
| Validation | 500 |
| ID test | 750 |
| OOD test | 10,000 |

Seed: 42. Recorded exact match is 744/750 (99.20%) on ID and 0/10,000 (0%) on OOD. The OOD split changes operand magnitude, sequence length, positional patterns, result length, and carry distributions. These factors are confounded; the failure cannot be attributed exclusively to positional encoding. The synthetic dataset's distribution license also needs an explicit owner decision before public redistribution.

The historical arithmetic model weights, checkpoints, dataset, generator, and reports under `results/arithmetic/` remain local and are excluded from the code-only first release. The generator can reconstruct the deterministic default split, but byte-for-byte reproduction of historical files has not been established. Any machine-specific absolute paths found in five JSON reports were changed to repository-relative paths only in an earlier audit; recorded metrics and hashes were not changed by that path cleanup.

## GSM8K data and evaluation

GSM8K is an external benchmark. The local download metadata labels its license as MIT, but that local field is not sufficient release clearance. Verify the authoritative upstream terms, attribution, and permitted redistribution before including any benchmark material or derived data. Raw GSM8K files, processed question sequences, the official test file, and detailed question/generation outputs must remain excluded from public distribution.

The sanitized aggregate `reports/gsm8k_summary.json` remains in the local research copy and is excluded with all reports from the code-only first release. It records the limited 100-example validation and 20-example official-test evaluations noted above.

## Hashes and limits

The main checkpoint SHA-256 was independently calculated during this audit and matches the verified value above. Dataset hashes are not stored in the main checkpoint. Hashes recorded in arithmetic evaluation reports identify the arithmetic test split files used by that run; they do not establish the provenance of the GPT-0 language-model data.
