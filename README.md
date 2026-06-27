# SML RLAIF Alignment

Reproducible repository for running the emotional RLAIF alignment experiments. It includes a minimal vendored copy of LLaMA-Factory at the repository root to pin the execution version, plus the SML experiment project in `rlaif-model/rlaif-llama-factory-training`.

## Contents

- `src/`, `setup.py`, `pyproject.toml`, `requirements.txt`: LLaMA-Factory runtime used by the experiments.
- `rlaif-model/rlaif-llama-factory-training/run_sml.sh`: sequential training, prediction, and analysis pipeline.
- `rlaif-model/rlaif-llama-factory-training/examples/train_lora/`: model and stage-specific YAML configurations.
- `rlaif-model/rlaif-llama-factory-training/data/dataset_info.json`: LLaMA-Factory dataset registry.
- `rlaif-model/rlaif-llama-factory-training/emotional_results.py`: emotional metric aggregation script.

Datasets, checkpoints, adapters, predictions, and logs are not tracked in Git. The data files are published at:

https://huggingface.co/datasets/mario-rc/aif-emotional-generation

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[torch,metrics]"
```

Meta Llama models may require accepting the model license and authenticating with Hugging Face:

```bash
huggingface-cli login
```

## Data

Download the dataset JSON files from Hugging Face into the SML working directory:

```bash
cd rlaif-model/rlaif-llama-factory-training
huggingface-cli download mario-rc/aif-emotional-generation \
  --repo-type dataset \
  --local-dir data \
  --include "*.json"
```

The `data/dataset_info.json` file is tracked in Git. All other JSON files under `data/` are ignored and should come from the Hugging Face dataset.

## Execution

From `rlaif-model/rlaif-llama-factory-training`:

```bash
./run_sml.sh
```

Useful variables:

```bash
CUDA_VISIBLE_DEVICES=1 ./run_sml.sh
RUN_ID=experiment_001 ./run_sml.sh
LLAMAFACTORY_CLI=/path/to/llamafactory-cli ./run_sml.sh
```

You can also run a single configuration:

```bash
llamafactory-cli train examples/train_lora/gemma2-2b/gemma-2-2b-it_lora_sft_d_3ep.yaml
```

## Local Artifacts

These paths are generated locally and ignored by Git:

- `rlaif-model/rlaif-llama-factory-training/data/*.json`, except `dataset_info.json`
- `rlaif-model/rlaif-llama-factory-training/logs/`
- `rlaif-model/rlaif-llama-factory-training/saves/`
- `__pycache__/`, `.venv/`, `.env`
