# GPT-0 Math — Scientific Baseline

**Status:** Historical experimental baseline
**Version:** GPT-0
**Date of documentation:** 2026-11-01

**Availability:** This is a record of locally retained experiments. The code-only first release excludes all datasets, derived data, tokenizer files, experiment reports, and model checkpoints. In particular, the GPT-0 checkpoint is not included; its distribution decision is postponed. File paths below identify local historical inputs and outputs, not files promised in the public release.

The synthetic arithmetic dataset and `scripts/generate_arithmetic.py` are also withheld. The generator can reconstruct the deterministic default split; byte-for-byte reproduction of historical files has not been established.

## 1. Project overview

GPT-0 Math is an experimental decoder-only Transformer developed to investigate mathematical language modeling and arithmetic generalization.

The project contains two distinct experimental tracks:

1. Language-based mathematical problem solving using GSM8K.
2. Specialized digit-level arithmetic using a dedicated addition dataset.

These experiments must be evaluated and reported independently.

GPT-0 is preserved as the historical baseline for subsequent GPT-1 Math experiments.

## 2. Architecture

| Parameter | Value |
|---|---:|
| Parameters | 23,328,768 |
| Transformer layers | 6 |
| Attention heads | 6 |
| Embedding dimension | 384 |
| Vocabulary size | 16,000 |
| Context length | 1,024 |
| Dropout | 0.1 |
| Batch size | 2 |
| Activation | GELU |
| Positional encoding | Learned absolute embeddings |
| Attention | Causal self-attention |
| Normalization | Pre-LayerNorm |
| Output head | Tied token embeddings |

The architecture is a custom experimental Transformer. It should not be described as an exact reproduction of the original OpenAI GPT-1.

## 3. Original training configuration

Historical local source: `checkpoints/best.pt` (excluded from the first release).

| Parameter | Value |
|---|---:|
| Training iterations | 2,000 |
| Evaluation interval | 250 |
| Learning rate | 0.0003 |
| Minimum learning rate | 0.00003 |
| Warmup iterations | 100 |
| Weight decay | 0.1 |
| Random seed | 42 |
| Best validation loss | 4.554548854685838 |

Training data paths recorded in the checkpoint:

- `data/processed/train_sequences.jsonl`
- `data/processed/val_sequences.jsonl`
- `data/tokenizer.json`

The training configuration records `theory_probability: 0.1`.

The original training metrics log has not been located in the inspected results directory. Therefore, a complete historical training curve cannot currently be reconstructed.

## 4. Dataset provenance

### GSM8K

Source: https://huggingface.co/datasets/openai/gsm8k

The local download metadata labels the dataset license as MIT. Verify the authoritative upstream dataset terms and attribution requirements before redistributing data or derived materials; this audit does not treat the local field alone as legal clearance.

| Split | Examples |
|---|---:|
| Original training set | 7,473 |
| Processed training set | 6,725 |
| Processed validation set | 748 |
| Official test set | 1,319 |

The current pipeline creates a 90/10 split of the downloaded training set. The checkpoint records data paths but no dataset hashes, so the exact historical input files and upstream revision cannot be cryptographically established from the checkpoint alone.

The current corpus manifest contains 11 English and 7 Italian educational documents. The historical processed sequences contain only 9 English theory training documents and 2 English theory validation documents; none of the Italian documents appears. The historical `split_manifest.jsonl` lists only those 11 English documents. The current pipeline loads all 18 documents and would split them into 9 English + 6 Italian training documents and 2 English + 1 Italian validation documents. Do not confuse that expected current split with the historical training inputs.

The tokenizer training script uses training examples and excludes validation examples.

In the local research copy, the official test set was stored separately at:

`data/benchmarks/gsm8k/test.jsonl`

SHA256:

`E33726D2A350B296631F1036027637C56FEC8FCE4BF97D01F491A247685D0795`

### Data integrity limitation

The current code supports separation between training, validation and official test evaluation.

However, the historical absence of all test-set exposure or influence on development decisions cannot be established retroactively.

## 5. GSM8K evaluation results

Historical checkpoint:

`checkpoints/best.pt`

Checkpoint step: 2,000

### Validation

| Metric | Result |
|---|---:|
| Evaluated examples | 100 |
| Correct | 0 |
| Incorrect | 100 |

### Official test

| Metric | Result |
|---|---:|
| Evaluated examples | 20 |
| Correct | 0 |
| Incorrect | 20 |
| Accuracy | 0% |

The official GSM8K test evaluation covered only 20 examples, not the complete 1,319-example test set.

The first inspected example showed repetitive generation and an incorrect extracted numerical prediction.

These results indicate that GPT-0 did not demonstrate successful GSM8K problem-solving performance in the inspected evaluations.

They must not be presented as a complete benchmark evaluation.

## 6. Arithmetic experiment

The arithmetic experiment is separate from GSM8K.

### Dataset

Local private dataset location: `data/arithmetic/v1/` (excluded from the first release).

| Split | Examples |
|---|---:|
| Train | 3,800 |
| Validation | 500 |
| Test ID | 750 |
| Test OOD | 10,000 |

The model uses digit-level tokenization:

- PAD: 0
- BOS: 1
- EOS: 2
- ADD: 3
- EQ: 4
- Digits 0–9: IDs 5–14

Training and ID operands are in the range [0, 99].
OOD operands are in the range [100, 9999].

### Arithmetic architecture

| Parameter | Value |
|---|---:|
| Transformer layers | 6 |
| Attention heads | 6 |
| Embedding dimension | 384 |
| Vocabulary size | 15 |
| Context length | 32 |
| Parameters | 10,665,600 |

### Training

- Optimizer: AdamW
- Learning rate: 0.0003
- Weight decay: 0.1
- Training iterations: 5,000
- Batch size: 32
- Seed: 42

Best validation loss: 0.00508 at step 4,250.

### Evaluation

| Metric | Test ID | Test OOD |
|---|---:|---:|
| Exact match | 99.20% | 0.00% |
| Numeric validity | 100% | 100% |
| EOS termination | 100% | 90.53% |
| Examples | 750 | 10,000 |

The OOD evaluation demonstrates a severe generalization failure.

The OOD split differs from training in operand length, prompt positions, result length and carry distribution. These factors are confounded. The observed failure cannot be attributed exclusively to positional encoding.

### Arithmetic artifacts

- Best full checkpoint: `checkpoints/arithmetic_v1/best.pt`
- Latest checkpoint: `checkpoints/arithmetic_v1/latest.pt`
- Model-only checkpoint: `checkpoints/arithmetic_v1/best_weights.pt`
- Training log: `results/arithmetic/training_20261101_102406.jsonl`
- Evaluation report: `results/arithmetic/evaluation_v2_20261101_103132.json`

Best model-only checkpoint SHA256:

`D70908299FDed5A16E34E60755AFCC695CD18F3137D383682B438B928922909B`

Dataset SHA256:

- Test ID: `992b64a00d6f8cae20d4d03c944c02da9ca07eb6480e4e6cbd2046d7857b097d`
- Test OOD: `033583f3c49102b10e3e55088d82b85de6b6e96b058cd1a99461825ec87394a0`

## 7. Historical checkpoint integrity

### Main language-model checkpoints

| File | SHA256 |
|---|---|
| `checkpoints/best.pt` | `FA1C2449F33AD9B631EDEA3F418EA3D9005C41CA593F4BEE8BC1D2FCA3BEB85D` |
| `checkpoints/latest.pt` | `0AE707BF3BD3C361A8D92EB495A3B9F5E35221B8A34A186923CE0DD04B2BA813` |

Both checkpoints:

- Have step 2,000.
- Have identical configuration.
- Have identical best validation loss.
- Contain 77 model tensors.
- Have exactly identical model tensor values.
- Have identical optimizer, scheduler, scaler and RNG states in the inspected comparison.

Their file-level SHA256 hashes differ despite the identical loaded contents.

The precise binary serialization difference has not been established.

### Checkpoint preservation

The original checkpoint files must remain unchanged.

GPT-0 is a historical baseline and must not be overwritten by GPT-1 experiments.

## 8. Scientific limitations

1. GSM8K evaluation used only 20 official test examples.
2. GSM8K validation evaluation used 100 examples.
3. Historical training curves for the original language-model experiment have not been located.
4. Historical test-set exposure cannot be ruled out completely.
5. Arithmetic OOD evaluation changes multiple factors simultaneously.
6. Results from GSM8K and digit-level addition measure different capabilities.
7. The experiments do not establish broad mathematical reasoning ability.

## 9. Lessons for GPT-1 Math

GPT-1 Math should address the following limitations:

- Better algorithmic generalization.
- Controlled OOD evaluation.
- Balanced operand-length distributions.
- Explicit separation of positional extrapolation and arithmetic generalization.
- Multiple random seeds.
- Repeated evaluations and uncertainty reporting.
- Reproducible dataset generation and hashes.
- Checkpoint selection using validation only.
- Frozen test sets and documented evaluation protocols.

The planned positional encoding comparison is:

1. Learned absolute positional embeddings.
2. Rotary Positional Embeddings (RoPE).

All other experimental conditions should remain controlled.

## 10. Baseline conclusion

GPT-0 Math established an experimental foundation for mathematical language modeling and specialized arithmetic generation.

It achieved 99.20% exact-match accuracy on its in-distribution arithmetic test set, but failed completely on its longer-operand OOD test set.

Its inspected GSM8K evaluations also showed 0% accuracy on the reported samples.

These limitations motivate GPT-1 Math, whose primary research objective is improved generalization under controlled and reproducible experiments.

GPT-0 remains preserved as the historical baseline.
