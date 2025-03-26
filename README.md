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

### Acknowledgements

We thank [EvolvingLMMs-Lab/lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for their repository on benchmarking large multi-modal language models, which was used as a starting point for our code repository.
