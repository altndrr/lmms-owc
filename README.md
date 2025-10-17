<div align="center">

<a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/-Python_3.12-blue?logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch_2-ee4c2c?logo=pytorch&logoColor=white"></a>

# On Large Multimodal Models as Open-World Image Classifiers

[Alessandro Conti](https://scholar.google.com/citations?user=EPImyCcAAAAJ), [Massimiliano Mancini](https://scholar.google.com/citations?user=bqTPA8kAAAAJ), [Enrico Fini](https://scholar.google.com/citations?user=OQMtSKIAAAAJ), [Yiming Wang](https://scholar.google.com/citations?user=KBZ3zrEAAAAJ), [Paolo Rota](https://scholar.google.com/citations?user=K1goGQ4AAAAJ), [Elisa Ricci](https://scholar.google.com/citations?user=xf1T870AAAAJ)

</div>

Traditional image classification requires a predefined list of semantic categories. In contrast, Large Multimodal Models (LMMs) can sidestep this requirement by classifying images directly using natural language (*e.g.*, answering the prompt *"What is the main object in the image?"*). Despite this remarkable capability, most existing studies on LMM classification performance are surprisingly limited in scope, often assuming a closed-world setting with a predefined set of categories. In this work, we address this gap by thoroughly evaluating LMM classification performance in a truly open-world setting. We first formalize the task and introduce an evaluation protocol, defining various metrics to assess the alignment between predicted and ground truth classes. We then evaluate 13 models across 10 benchmarks, encompassing prototypical, non-prototypical, fine-grained, and very fine-grained classes, demonstrating the challenges LMMs face in this task.
Further analyses based on the proposed metrics reveal the types of errors LMMs make, highlighting challenges related to granularity and fine-grained capabilities, showing how tailored prompting and reasoning can alleviate them.

## Setup

### Install dependencies

```bash
# clone project
git clone https://github.com/altndrr/lmms-owc
cd lmms-owc

# (recommended) use uv to set up the python version
# and to install the required dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen

# (alternative) use conda to set up the python version
# and pip to install the required dependencies
conda create --name py3.12 python=3.12
conda activate py3.12
python -m venv .venv
.venv/bin/python3 -m pip install -e .

# activate virtual environment
source .venv/bin/activate
```

### Setup environment variables

```bash
# copy .env.example to .env
cp .env.example .env

# edit .env file
vim .env
```

## Usage

Once the environment is set up, you can run evaluations and analyses using the provided scripts and entrypoints.

> **TL;DR**
>
> - Use `--help` on any script to explore its options.
> - Use `scripts/schedule_batch.sh` for **local sequential runs**.
> - Use `scripts/schedule_sbatch.sh` for **distributed Slurm runs**.
> - Compute metrics **offline** when possible, and use ranking to **compare model performance**.

### Overview

The repository provides three main entrypoints:

| Script            | Purpose                                                                           |
| ----------------- | --------------------------------------------------------------------------------- |
| `eval_model.py`   | Runs evaluations of large multimodal models on the tasks.                         |
| `eval_metrics.py` | Computes metrics offline for previously obtained predictions.                     |
| `eval_ranking.py` | Computes *Elo-style rankings* across models based on pairwise evaluation results. |

In addition, two helper scripts (`scripts/schedule_batch.sh` and `scripts/schedule_sbatch.sh`) simplify large-scale or distributed experiment scheduling.

### Running model evaluations

You can run model evaluations directly:

```bash
python eval_model.py --help
```

or use one of the wrapper scripts below for running larger sets of experiments.

#### `scripts/schedule_batch.sh`

Runs multiple model–task pairs **sequentially** on a single machine (e.g., local or private server).

```bash
bash scripts/schedule_batch.sh --models qwen2-vl-7b --tasks caltech101,dtd,food101
```

**Main options:**

| Option         | Description                                              |
| -------------- | -------------------------------------------------------- |
| `--models`     | Comma-separated list of models to evaluate.              |
| `--tasks`      | Comma-separated list of tasks to evaluate on.            |
| `--limit`      | Limit the number of samples per task.                    |
| `--model-args` | Extra comma-separated arguments for the models.          |
| `--no-samples` | Disable saving of sample predictions to disk.            |
| `--no-wandb`   | Disable Weights & Biases logging.                        |
| `--output`     | Output directory for results (default: `logs/schedule`). |

#### `scripts/schedule_sbatch.sh`

Submits **parallel jobs** to a Slurm cluster (one per model–task pair), enabling distributed evaluations across multiple GPUs.

```bash
bash scripts/schedule_sbatch.sh --partition gpu --gpu a100.40:8 --models qwen2-vl-2b,qwen2-vl-7b --tasks flowers102,ucf101
```

**Slurm options:**

| Option                              | Description                                                                   |
| ----------------------------------- | ----------------------------------------------------------------------------- |
| `--partition`, `--account`          | Slurm partition and account to use.                                           |
| `--cpu`, `--gpu`, `--mem`, `--time` | Resource allocation per job (default: 12 CPUs, 8×A100 GPUs, 128 GB RAM, 2 h). |
| `--nodes`, `--name`                 | Number of nodes and job name.                                                 |

**Evaluation options:** identical to `schedule_batch.sh`.

Each evaluation automatically downloads the required models and datasets (if not already cached) and stores logs and predictions in the `logs/` directory.

### Computing metrics

After evaluations are complete, you can compute the metrics for the generated predictions:

```bash
python eval_metrics.py -i logs/schedule/ -m concept_semantic_similarity,semantic_similarity,textual_inclusion,textual_inclusion_llama32
```

You can also use glob patterns to select specific experiments:

```bash
# Example: evaluate all experiments whose names end with "_cot"
python eval_metrics.py -i "logs/schedule/*_cot/" -m concept_semantic_similarity,semantic_similarity,textual_inclusion,textual_inclusion_llama32
```

Metrics can be evaluated **online** (during model evaluation) or **offline** (in a separate post-processing step).
While most lightweight metrics can be computed online, it is recommended to run **model-based metrics** offline, as they typically execute on a **single GPU**.
Running them separately avoids underutilizing resources when using multi-GPU setups (e.g., 8 GPUs allocated for `eval_model`).

By default, the repository **excludes `textual_inclusion_llama32` from online evaluation**, as it is a model-based metric and is evaluated offline.

### Ranking models

You can compare model performance using an **Elo-style ranking** computed from pairwise evaluation outcomes:

```bash
python eval_ranking.py -i logs/schedule/caltech101/ --criterion llama_score
```

This script computes Elo scores across models based on the selected criterion (`llama_score` or `semantic_similarity`), producing a ranking that summarizes their relative performance.

### Enabling FlashAttention

If your GPUs support **FlashAttention**, you can enable it by installing the corresponding extra dependencies:

```bash
uv sync --frozen --extra nvidia --no-build-isolation
```

> ⚠️ **Note:** FlashAttention should only be installed on compatible NVIDIA hardware.
> Attempting to install it on unsupported GPUs may result in build errors or degraded performance.

This setup enables GPU-specific optimizations that can significantly improve inference speed during model evaluation.

## Development

### Install dependencies

```bash
# (recommended) use uv to install the dependencies
uv sync --frozen --extra dev

# (alternative) if you have install dependencies with pip
# install the required dependencies for development
.venv/bin/python3 -m pip install -e .[dev]
```

### Install pre-commit hooks

```bash
# install pre-commit hooks
pre-commit install
```

### Run tests

```bash
# run fast tests
make test
```

Note: some tests can fail due to CUDA out-of-memory or network timeouts — simply re-running them usually resolves the issue.

### Format code

```bash
# run linters
make format
```

### Clean repository

```bash
# remove autogenerated files
make clean

# remove logs
make clean-logs
```

## Citation

```latext
@article{conti2025large,
  title={On large multimodal models as open-world image classifiers},
  author={Conti, Alessandro and Mancini, Massimiliano and Fini, Enrico and Wang, Yiming and Rota, Paolo and Ricci, Elisa},
  year={2025}
  journal={ICCV},
}
```

### Acknowledgements

We thank [EvolvingLMMs-Lab/lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for their repository on benchmarking large multi-modal language models, which was used as a starting point for our code repository.
