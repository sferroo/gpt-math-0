# GPT-0 Math — verified experimental results

These results summarize existing local experiment records. No dataset examples, prompts, generations, token IDs, individual errors, model weights, or private file paths are included here. The source records named below remain local and are not part of the code-only release.

## Arithmetic generalization experiment

This digit-level addition experiment is separate from the historical GPT-0 GSM8K language-model experiment. The stored evaluation record identifies evaluator version 2, seed 42, a 6-layer, 6-head, 384-dimensional model, context length 32, vocabulary size 15, and a maximum of six generated tokens. The recorded split sizes are 3,800 training, 500 validation, 750 ID test, and 10,000 OOD test examples.

| Split | Exact match | Numeric-valid outputs | EOS-terminated outputs |
|---|---:|---:|---:|
| ID test | 744/750 (99.2%) | 750/750 | 750/750 |
| OOD test | 0/10,000 (0%) | 10,000/10,000 | 9,053/10,000 (90.53%) |

The OOD split changes operand magnitude, sequence length, positional patterns, result length, and carry distributions. These factors are confounded; the result does not isolate positional encoding as the cause of the failure.

The stored arithmetic training log records validation loss `0.005076836962840596` at step 4,250. This is an arithmetic-experiment value, not a GPT-0 language-model training result.

## Historical GPT-0 language-model experiment

The local checkpoint metadata records step 2,000 and best validation loss `4.554548854685838`. The historical checkpoint is not included in the first public release. It contains no SHA-256 hashes for the processed training sequences or tokenizer, so the exact historical dataset cannot be cryptographically established from the checkpoint. No original GPT-0 language-model training log was identified.

## GSM8K evaluation

| Split | Correct / evaluated | Accuracy |
|---|---:|---:|
| Validation | 0/100 | 0% |
| Official test subset | 0/20 | 0% |

Only 20 of the 1,319 available official test examples were evaluated. This is not a complete GSM8K benchmark evaluation. The stored summary reports repetitive validation generation and does not include original questions, answers, or generated outputs.

## Evidence records

The figures above were checked against existing local records without recalculating metrics:

- GSM8K: `gsm8k_summary.json` (sanitized aggregate report).
- Arithmetic evaluation: `evaluation_v2_20261101_103132.json` (aggregate fields only; its per-example error records remain private).
- Arithmetic training: `training_20261101_102406.jsonl` (stored step-4,250 row).
- Arithmetic split/configuration: `metadata.json` and the recorded evaluation configuration.
- Historical GPT-0 step, validation loss, and configuration: local checkpoint metadata.

Those source records, datasets, checkpoints, and reports remain excluded from the code-only release. The arithmetic generator is excluded; byte-for-byte reproduction of the historical arithmetic files is not claimed.
