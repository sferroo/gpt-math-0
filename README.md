# GPT-0 Math

An experimental research project exploring mathematical language modeling, arithmetic generalization, and decoder-only Transformer architectures.

GPT-0 Math is an educational and experimental decoder-only Transformer project developed from scratch. It explores mathematical text processing, language modeling, autoregressive generation, arithmetic experiments, and generalization beyond the training distribution.

> **Research status:** Experimental. The results below are preliminary observations from historical local experiments. They are not evidence of general mathematical reasoning capabilities and have not been independently reproduced from this public release.

## First public release: source code only

The first public release contains source code and approved documentation under the Apache License 2.0.

The following materials are intentionally **not included**:

- The private Italian and English educational corpus and its manifest.
- The project-generated synthetic arithmetic dataset and its metadata.
- Original GSM8K data files.
- Processed datasets, token sequences, and split files.
- The trained tokenizer.
- Model checkpoints and weights, including the GPT-0 checkpoint.
- Raw reports, detailed experiment outputs, and local training records.
- Machine-specific artifacts.

The arithmetic dataset generator is also excluded because it can reconstruct the deterministic default split.

A public checkout is therefore **not a self-contained reproduction package**. Training and evaluation workflows that depend on private or external inputs require users to obtain their own data and confirm that their intended use is permitted.

Before running data preparation, training, or evaluation scripts, read:

- [Data and artifact availability](docs/DATA_AVAILABILITY.md)
- [Public release inventory](docs/PUBLIC_RELEASE_INVENTORY.md)
- [Experimental provenance](docs/EXPERIMENTAL_PROVENANCE.md)

The curated [experimental results](docs/EXPERIMENTAL_RESULTS.md) report aggregate observations. The underlying private evidence records are not included.

## Research objectives

GPT-0 Math investigates:

- Mathematical language modeling with a decoder-only Transformer.
- Mathematical text processing in Italian and English.
- Autoregressive generation and numerical prediction.
- In-distribution (ID) and out-of-distribution (OOD) arithmetic generalization.
- Structured training, evaluation, and checkpoint management.
- Architectural improvements for future experimental models.

GPT-0 is retained as a historical baseline for subsequent experiments, including the planned GPT-1 Math work.

## GPT-0 baseline

The original GSM8K language-model experiment used this architecture:

| Parameter | Value |
|---|---:|
| Architecture | Decoder-only Transformer |
| Layers | 6 |
| Attention heads | 6 |
| Embedding dimension | 384 |
| Vocabulary size | 16,000 |
| Context length | 1,024 |
| Dropout | 0.1 |
| Parameters | 23,328,768 |
| Positional encoding | Learned absolute embeddings |

The recorded training configuration was:

- Optimizer: AdamW
- Learning rate: `0.0003`
- Weight decay: `0.1`
- Batch size: `2`
- Training iterations: `2,000`
- Random seed: `42`

The historical checkpoint recorded model and optimizer state, scheduler state, configuration, and available random-number-generator states. The checkpoint is not part of this release.

See [GPT0_BASELINE.md](GPT0_BASELINE.md) for the baseline record and [experimental provenance](docs/EXPERIMENTAL_PROVENANCE.md) for the distinction between historical and current data.

## Experimental results

These values summarize existing local experiment records. They are presented as historical observations, **not as independently reproduced results from this public repository**.

### 1. GSM8K language-model experiment

| Evaluation split | Examples evaluated | Correct | Accuracy |
|---|---:|---:|---:|
| Validation | 100 | 0 | 0% |
| Official test subset | 20 | 0 | 0% |

The official test evaluation covered only 20 of the 1,319 available examples. It is not a complete benchmark evaluation.

The inspected validation generations were repetitive and did not show the expected answer markers, solution markers, or EOS termination. The baseline therefore did not demonstrate successful mathematical problem-solving performance on the evaluated samples.

### 2. Arithmetic generalization experiment

The separate arithmetic experiment investigated addition using a digit-level vocabulary.

| Split | Examples |
|---|---:|
| Training | 3,800 |
| Validation | 500 |
| ID test | 750 |
| OOD test | 10,000 |

| Evaluation | Exact match |
|---|---:|
| ID test | 99.20% (744/750) |
| OOD test | 0% (0/10,000) |

The model performed well on the in-distribution test set but failed on the evaluated OOD distribution.

The OOD split changes operand magnitude and also affects sequence lengths, positional patterns, result lengths, and carry distributions. This experiment does **not** isolate the causal contribution of positional encoding; the observed failure cannot be attributed exclusively to positional embeddings.

The synthetic dataset, model weights, and detailed experiment artifacts remain private. See [experimental results and limitations](docs/EXPERIMENTAL_RESULTS.md).

## Repository structure

The repository is organized into the following directories and files:

```text
gpt-0-math/
├── configs/
│   ├── arithmetic.yaml
│   └── small.yaml
├── data/
│   ├── README.md
│   └── raw/
│       └── manifest.example.jsonl
├── docs/
│   ├── DATA_AVAILABILITY.md
│   ├── EXPERIMENTAL_PROVENANCE.md
│   ├── EXPERIMENTAL_RESULTS.md
│   ├── PUBLIC_RELEASE_INVENTORY.md
│   └── RELEASE_READINESS.md
├── gpt0/
│   ├── __init__.py
│   ├── arithmetic_dataset.py
│   ├── arithmetic_model.py
│   ├── checkpoints.py
│   ├── data_pipeline.py
│   ├── dataset.py
│   ├── model.py
│   └── numeric_tokenizer.py
├── scripts/
│   └── ...
├── tests/
│   ├── test_arithmetic_dataset.py
│   ├── test_arithmetic_model.py
│   ├── test_model.py
│   └── test_numeric_tokenizer.py
├── .gitignore
├── GPT0_BASELINE.md
├── LICENSE
├── README.md
└── requirements.txt
```

This tree represents the public source release. Some directories contain additional files.

For the exact classification of public and excluded materials, see:

- [Public Release Inventory](docs/PUBLIC_RELEASE_INVENTORY.md)
- [Data Availability](docs/DATA_AVAILABILITY.md)
- [.gitignore](.gitignore)

## Installation
Python 3.11-3.13 is the intended supported range for the documented CI matrix. Use a compatible Python version.

### Windows / PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux / macOS

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a compatible PyTorch build for your operating system and hardware by following the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

Verify the installation:

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## Development and testing

The full local test suite depends on private arithmetic dataset files that are not included in this release. In the research workspace, 49 tests passed; 9 dataset-dependent tests require the private arithmetic dataset. Those figures describe the local workspace, not a clean public checkout.

For the exact dataset-free test commands and prerequisites, see [docs/DATA_AVAILABILITY.md](docs/DATA_AVAILABILITY.md). A public checkout should not be described as passing the full local suite unless the tests have actually been run in that checkout.

## Data, benchmark use, and licensing

### Private project data

The original research workspace contains AI-assisted educational material in Italian and English. The corpus and its manifest are private because separate distribution terms have not been established. The project-generated synthetic arithmetic dataset is also private.

The current bilingual corpus is not the same dataset as the one represented by the historical GPT-0 processed sequences. The historical sequences contained 9 English theory documents in training, 2 English theory documents in validation, 6,725 GSM8K training examples, 748 GSM8K validation examples, and no Italian documents.

The historical checkpoint records processed-data paths but does not contain SHA-256 hashes for those inputs. The exact historical training data therefore cannot be cryptographically established from the checkpoint alone. No original GPT-0 language-model training log has been identified.

### GSM8K separation

The official GSM8K test split is **not required** for tokenizer training or model training. It must remain separate from the training workflow. Never use official test examples for:

- Training or fine-tuning.
- Hyperparameter tuning.
- Model selection.
- Iterative prompt or architecture optimization.

Use benchmark test data only in a separately documented evaluation procedure. Downloading GSM8K does not grant permission to redistribute its files; users must comply with the dataset's upstream license and terms. The original GSM8K files are not included in this repository.

The optional download and preparation scripts have local-data prerequisites. Review [docs/DATA_AVAILABILITY.md](docs/DATA_AVAILABILITY.md) before using them. Do not run data preparation in a research workspace containing historical artifacts that must be preserved.

### License boundaries

The Apache License 2.0 applies to the source code covered by the repository license. It does not automatically relicense external datasets, third-party content, private educational material, or model weights. Each component has separate provenance and licensing considerations.

GSM8K source: [Hugging Face - openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k).

## Reproducibility and limitations

The codebase supports structured experiment workflows, but the historical experiments cannot be independently reproduced from this code-only release alone.

Important limitations include:

1. The original GSM8K language-model baseline did not solve the evaluated examples.
2. Arithmetic performance dropped to zero on the evaluated OOD split.
3. The initial educational corpus is too small for competitive mathematical language modeling.
4. The arithmetic OOD experiment changes several distributional properties simultaneously.
5. Historical training provenance is incomplete.
6. The public release does not contain all data and artifacts required to reproduce the original experiments.

These limitations are part of the research record and should not be hidden or overstated.

## Future research: GPT-1 Math

Planned research directions include:

- Improved mathematical tokenization and representations.
- Controlled comparisons of positional encoding strategies.
- Experiments with learned absolute positional embeddings and Rotary Position Embeddings (RoPE).
- More rigorous generalization benchmarks.
- Repeated evaluations and statistical analysis.
- Integration with TraceAI for experiment tracking and regression detection.

GPT-0 remains the historical baseline against which future improvements will be evaluated.

## License

The source code in this repository is distributed under the Apache License 2.0. See [LICENSE](LICENSE) for the full text.

Copyright 2026 Silvio Ferro.

Dataset, corpus, benchmark, and checkpoint rights are governed separately as described above.
