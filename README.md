# SML Emotional RLAIF Alignment

Reproducible project for training, predicting, and evaluating compact
instruction-tuned language models with an emotional RLAIF workflow based on
LoRA fine-tuning in LLaMA-Factory.

Repository name:

```text
sml-emotional-rlaif
```

This repository keeps the SML experiment files at the project root alongside a
vendored LLaMA-Factory runtime. Datasets, Hugging Face caches, logs,
checkpoints, adapters, predictions, and metric outputs are generated locally
and are intentionally excluded from Git.

## Scope

Models:

- `google/gemma-2-2b-it`
- `meta-llama/Llama-3.2-1B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`

Training and evaluation stages:

- SFT LoRA on the emotional demonstration dataset, 3 epochs (`sft_d_3ep`).
- Test-set prediction after the demonstration SFT run.
- SFT LoRA on the DPR/unlabeled prompt dataset, 3 epochs (`sft_dpr_3ep`).
- Test-set prediction after the DPR SFT run.
- Reward model LoRA, 1 epoch, initialized from `sft_d_3ep`.
- Reward model prediction on the held-out preference set.
- PPO LoRA, 1 epoch, initialized from `sft_d_3ep` and paired with `rm_1ep`.
- PPO test-set prediction.
- DPO LoRA, 1 epoch, initialized from `sft_d_3ep`.
- DPO test-set prediction.
- Gemma-only DPO LoRA, 3 epochs, with corresponding test-set prediction.
- Emotional metric aggregation from generated prediction files.

## Data Source

Dialogue and DPO preference datasets are published in the Hugging Face dataset
[`mario-rc/aif-emotional-generation`](https://huggingface.co/datasets/mario-rc/aif-emotional-generation).
The tracked LLaMA-Factory dataset registry lives at `data/dataset_info.json`.

Download the source JSON datasets into the project data directory:

```bash
huggingface-cli download mario-rc/aif-emotional-generation \
  --repo-type dataset \
  --local-dir data \
  --include "*.json"
```

The download preserves the `dialogues/` and `aif_annotations/` subdirectories;
it does not create the filenames expected by the training YAMLs. Prepare them as follows:

- Copy `dialogues/train.json` and `dialogues/test.json` to `ppo_unlabeled_prompts_dataset.json` and `ppo_unlabeled_prompts_dataset_test.json` inside `data/`.
- Filter those dialogue files to rows with `set == "sft-demonstration"` to create `sft_demonstration_dataset.json` and `sft_demonstration_dataset_test.json`.
- Copy `aif_annotations/train.json` to `data/dpo_preference_dataset.json`.
- Supply `rm_preference_dataset.json` and `rm_preference_dataset_test.json` in `data/`. They are separate RM datasets, not replacements for the DPO preference files downloaded above.

Expected local dataset files (optional entries are not used by `run_sml.sh`):

| Local file | Used for |
| --- | --- |
| `data/sft_demonstration_dataset.json` | demonstration SFT training |
| `data/sft_demonstration_dataset_foundation.json` | optional registered foundation SFT split |
| `data/sft_demonstration_dataset_test.json` | demonstration SFT prediction and analysis |
| `data/sft_demonstration_dataset_test_history.json` | optional registered history-aware SFT test split |
| `data/rm_preference_dataset.json` | reward model training |
| `data/rm_preference_dataset_test.json` | reward model prediction |
| `data/dpo_preference_dataset.json` | DPO training |
| `data/ppo_unlabeled_prompts_dataset.json` | DPR SFT and PPO training |
| `data/ppo_unlabeled_prompts_dataset_test.json` | DPR, PPO, and DPO prediction analysis |

## Layout

```text
sml-emotional-rlaif/
  README.md
  LICENSE
  pyproject.toml
  setup.py
  requirements.txt
  .env.example                      # optional local run overrides
  run_sml.sh                        # sequential SML experiment pipeline
  emotional_results.py              # emotional metric aggregation
  src/                              # vendored LLaMA-Factory runtime
  data/
    dataset_info.json               # tracked LLaMA-Factory dataset registry
  examples/train_lora/              # model and stage-specific YAML configs
  logs/                             # local run logs, ignored by Git
  saves/                            # adapters, predictions, and results, ignored by Git
```

## Setup After Clone

Create and activate a project-local Python 3.10 environment from the repository
root. Pin the core training dependencies to the versions used for these models:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[torch,metrics]" "torch==2.5.1" "transformers==4.45.2" "peft==0.11.1" "trl==0.8.6" "accelerate==0.34.0" "datasets==2.16.0" "huggingface_hub==0.27.1"
```

If your node requires a specific CUDA-enabled PyTorch build, install that
PyTorch wheel first, then install the editable package:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[metrics]" "transformers==4.45.2" "peft==0.11.1" "trl==0.8.6" "accelerate==0.34.0" "datasets==2.16.0" "huggingface_hub==0.27.1"
```

The editable install exposes the `llamafactory-cli` console entrypoint used by
the SML scripts.

`run_sml.sh` uses this repository's source and prefers its local `.venv`.

Gemma and Meta Llama checkpoints require accepting the base-model terms and
authenticating with Hugging Face:

```bash
huggingface-cli login
```

Then prepare the dataset files as described in [Data Source](#data-source).

## Full Pipeline

Run the complete SML workflow from the repository root:

```bash
./run_sml.sh
```

The script runs the configured YAML files sequentially for Gemma 2B, Llama 3.2
1B, and Llama 3.2 3B, then launches `emotional_results.py`.

Useful run overrides:

```bash
CUDA_VISIBLE_DEVICES=1 ./run_sml.sh
RUN_ID=experiment_001 ./run_sml.sh
LLAMAFACTORY_CLI=/path/to/llamafactory-cli ./run_sml.sh
```

You can also copy `.env.example` to `.env` at the repository root and store
local overrides there. `.env` is ignored by Git.

## Rerun Behavior

The batch script is sequential and the training YAML files set
`overwrite_output_dir: true`. Re-running a stage can overwrite the corresponding
directory under `saves/<model>/`. Use a separate checkout, move existing
artifacts, or edit the YAML output paths when you need to preserve a previous
adapter or prediction run.

Log files are grouped by CUDA device and run identifier:

```text
logs/sml_retrain_cuda<device>_<run_id>/
```

Each LLaMA-Factory command writes its own log file, and the pipeline writes a
run-level `summary.log`.

## Running By Stage

Individual stages can be launched directly with LLaMA-Factory:

```bash
llamafactory-cli train examples/train_lora/gemma2-2b/gemma-2-2b-it_lora_sft_d_3ep.yaml
llamafactory-cli train examples/train_lora/gemma2-2b/gemma-2-2b-it_lora_rm_1ep.yaml
llamafactory-cli train examples/train_lora/gemma2-2b/gemma-2-2b-it_lora_ppo_1ep.yaml
llamafactory-cli train examples/train_lora/gemma2-2b/gemma-2-2b-it_lora_dpo_1ep.yaml
```

Prediction configs use the same command:

```bash
llamafactory-cli train examples/train_lora/gemma2-2b/gemma-2-2b-it_lora_sft_d_predict_3ep.yaml
llamafactory-cli train examples/train_lora/gemma2-2b/gemma-2-2b-it_lora_ppo_predict_1ep.yaml
llamafactory-cli train examples/train_lora/gemma2-2b/gemma-2-2b-it_lora_dpo_predict_1ep.yaml
```

Swap the model subdirectory and file prefix for `llama3.2-1B/` or
`llama3.2-3B/` when running Llama configurations.

## Analysis

Regenerate emotional metrics from existing prediction files:

```bash
python emotional_results.py --verbose
```

Limit analysis to selected models:

```bash
python emotional_results.py --models gemma-2-2b-it Llama-3.2-1B-Instruct --verbose
```

The script reads generated prediction files from `saves/<model>/predict/` and
writes per-model summaries under `saves/<model>/emotional_balanced/`.

The released DPO/PPO adapters were evaluated on the same 392 English dialogue
examples from `mario-rc/aif-emotional-generation/dialogues`, split `test`.
SFT and RM use their respective stage-specific evaluation sets.

## Outputs

Expected artifacts for each model are grouped by training and prediction stage
under `saves/`: adapters (`adapter_model.safetensors`), predictions
(`generated_predictions.jsonl`) and metric summaries. This also includes the
additional 3-epoch DPO run for Gemma 2B.

Per-model metric files and `emotional_results_summary.json` are written under
`saves/<model>/emotional_balanced/`, as described in [Analysis](#analysis).

## Environment

`run_sml.sh` loads `.env` when present and sets the following defaults before
launching LLaMA-Factory:

- `CUDA_VISIBLE_DEVICES=0`
- `DISABLE_VERSION_CHECK=1`
- `PYTHONUNBUFFERED=1`
- `LLAMAFACTORY_CLI=llamafactory-cli`
- `RUN_ID=<current timestamp>`

Use `LLAMAFACTORY_CLI` when the desired executable is not the one installed in
the active environment.

## Hugging Face Models

These six selected adapters are available on Hugging Face. Use the inference example and tokenizer from the corresponding adapter repository; each model card documents its training parameters and license.

| Model | Alignment | Hugging Face |
| --- | --- | --- |
| Gemma 2 2B IT | PPO | [`mario-rc/emotional-rlaif-ppo-gemma-2-2b-it`](https://huggingface.co/mario-rc/emotional-rlaif-ppo-gemma-2-2b-it) |
| Gemma 2 2B IT | DPO | [`mario-rc/emotional-rlaif-dpo-gemma-2-2b-it`](https://huggingface.co/mario-rc/emotional-rlaif-dpo-gemma-2-2b-it) |
| Llama 3.2 1B Instruct | PPO | [`mario-rc/emotional-rlaif-ppo-llama-3.2-1b-instruct`](https://huggingface.co/mario-rc/emotional-rlaif-ppo-llama-3.2-1b-instruct) |
| Llama 3.2 1B Instruct | DPO | [`mario-rc/emotional-rlaif-dpo-llama-3.2-1b-instruct`](https://huggingface.co/mario-rc/emotional-rlaif-dpo-llama-3.2-1b-instruct) |
| Llama 3.2 3B Instruct | PPO | [`mario-rc/emotional-rlaif-ppo-llama-3.2-3b-instruct`](https://huggingface.co/mario-rc/emotional-rlaif-ppo-llama-3.2-3b-instruct) |
| Llama 3.2 3B Instruct | DPO | [`mario-rc/emotional-rlaif-dpo-llama-3.2-3b-instruct`](https://huggingface.co/mario-rc/emotional-rlaif-dpo-llama-3.2-3b-instruct) |

## Project Origin

This project is based on the original [Mario-RC/aif-emotional-model](https://github.com/Mario-RC/aif-emotional-model) project.

## License

This project is released under the Apache License 2.0. See `LICENSE`.

The license applies to this repository's code, configs, and documentation.
Third-party model weights, datasets, upstream projects, and Hugging Face
artifacts remain under their own licenses and terms.
