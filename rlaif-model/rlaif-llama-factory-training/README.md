# SML Emotional RLAIF

Experiment directory for training, predicting, and analyzing conversational models with an emotional RLAIF workflow based on LLaMA-Factory.

Configured models:

- `google/gemma-2-2b-it`
- `meta-llama/Llama-3.2-1B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`

## Structure

```text
.
  data/dataset_info.json  # tracked dataset registry
  examples/train_lora/    # training and prediction YAML files by model
  emotional_results.py    # emotional result aggregation
  run_sml.sh              # sequential pipeline
  logs/                   # local logs ignored by Git
  saves/                  # adapters, predictions, and results ignored by Git
```

## Setup

First install the vendored runtime from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[torch,metrics]"
```

Then enter this experiment directory:

```bash
cd rlaif-model/rlaif-llama-factory-training
```

Optionally copy `.env.example` to `.env` to set local overrides. `.env` is ignored by Git.

## Data

Only `data/dataset_info.json` is tracked in Git. Download the remaining datasets from:

https://huggingface.co/datasets/mario-rc/aif-emotional-generation

With `huggingface-cli`:

```bash
huggingface-cli download mario-rc/aif-emotional-generation \
  --repo-type dataset \
  --local-dir data \
  --include "*.json"
```

Expected files under `data/`:

- `sft_demonstration_dataset.json`
- `sft_demonstration_dataset_foundation.json`
- `sft_demonstration_dataset_test.json`
- `sft_demonstration_dataset_test_history.json`
- `dpo_preference_dataset.json`
- `rm_preference_dataset.json`
- `rm_preference_dataset_test.json`
- `human_annotations_reward_model_test.json`
- `ppo_unlabeled_prompts_dataset.json`
- `ppo_unlabeled_prompts_dataset_test.json`

## Pipeline

Run the full pipeline:

```bash
./run_sml.sh
```

Select a GPU or run identifier:

```bash
CUDA_VISIBLE_DEVICES=1 ./run_sml.sh
RUN_ID=experiment_001 ./run_sml.sh
```

The pipeline runs the configurations in `examples/train_lora/` in this general order:

1. SFT on the demonstration dataset.
2. SFT prediction.
3. SFT on DPR prompts.
4. DPR prediction.
5. Reward model training.
6. Reward model prediction.
7. PPO.
8. PPO prediction.
9. DPO.
10. DPO prediction.
11. Emotional analysis with `emotional_results.py`.

Run a single configuration:

```bash
llamafactory-cli train examples/train_lora/gemma2-2b/gemma-2-2b-it_lora_sft_d_3ep.yaml
```

## Analysis

Regenerate metrics from existing predictions:

```bash
python emotional_results.py --verbose
```

Limit analysis to selected models:

```bash
python emotional_results.py --models gemma-2-2b-it Llama-3.2-1B-Instruct --verbose
```

Outputs are written under `saves/<model>/emotional_balanced/`, which is ignored by Git.
