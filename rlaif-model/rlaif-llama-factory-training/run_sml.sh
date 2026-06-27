#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DISABLE_VERSION_CHECK=1
export PYTHONUNBUFFERED=1

LLAMAFACTORY="${LLAMAFACTORY_CLI:-llamafactory-cli}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
CUDA_LABEL="${CUDA_VISIBLE_DEVICES//[^a-zA-Z0-9_-]/_}"
LOG_DIR="logs/sml_retrain_cuda${CUDA_LABEL}_${RUN_ID}"
SUMMARY_LOG="${LOG_DIR}/summary.log"

mkdir -p "${LOG_DIR}"

if ! command -v "${LLAMAFACTORY}" >/dev/null 2>&1; then
  echo "ERROR: llamafactory-cli not found. Set LLAMAFACTORY_CLI or install LLaMA-Factory." >&2
  exit 127
fi

run_train() {
  local yaml="$1"
  local step_name="${yaml#examples/train_lora/}"
  local step_log="${LOG_DIR}/${step_name//\//__}.log"

  {
    echo
    echo "============================================================"
    echo "$(date '+%F %T') START ${yaml}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "log=${step_log}"
    echo "============================================================"
  } | tee -a "${SUMMARY_LOG}"

  "${LLAMAFACTORY}" train "${yaml}" 2>&1 | tee "${step_log}"

  echo "$(date '+%F %T') DONE ${yaml}" | tee -a "${SUMMARY_LOG}"
}

run_train examples/train_lora/gemma2-2b/gemma-2-2b-it_lora_sft_d_3ep.yaml
run_train examples/train_lora/gemma2-2b/gemma-2-2b-it_lora_sft_d_predict_3ep.yaml
run_train examples/train_lora/gemma2-2b/gemma-2-2b-it_lora_sft_dpr_3ep.yaml
run_train examples/train_lora/gemma2-2b/gemma-2-2b-it_lora_sft_dpr_predict_3ep.yaml
run_train examples/train_lora/gemma2-2b/gemma-2-2b-it_lora_rm_1ep.yaml
run_train examples/train_lora/gemma2-2b/gemma-2-2b-it_lora_rm_predict_1ep.yaml
run_train examples/train_lora/gemma2-2b/gemma-2-2b-it_lora_ppo_1ep.yaml
run_train examples/train_lora/gemma2-2b/gemma-2-2b-it_lora_ppo_predict_1ep.yaml
run_train examples/train_lora/gemma2-2b/gemma-2-2b-it_lora_dpo_1ep.yaml
run_train examples/train_lora/gemma2-2b/gemma-2-2b-it_lora_dpo_predict_1ep.yaml
run_train examples/train_lora/gemma2-2b/gemma-2-2b-it_lora_dpo_3ep.yaml
run_train examples/train_lora/gemma2-2b/gemma-2-2b-it_lora_dpo_predict_3ep.yaml

run_train examples/train_lora/llama3.2-1B/Llama-3.2-1B-Instruct_lora_sft_d_3ep.yaml
run_train examples/train_lora/llama3.2-1B/Llama-3.2-1B-Instruct_lora_sft_d_predict_3ep.yaml
run_train examples/train_lora/llama3.2-1B/Llama-3.2-1B-Instruct_lora_sft_dpr_3ep.yaml
run_train examples/train_lora/llama3.2-1B/Llama-3.2-1B-Instruct_lora_sft_dpr_predict_3ep.yaml
run_train examples/train_lora/llama3.2-1B/Llama-3.2-1B-Instruct_lora_rm_1ep.yaml
run_train examples/train_lora/llama3.2-1B/Llama-3.2-1B-Instruct_lora_rm_predict_1ep.yaml
run_train examples/train_lora/llama3.2-1B/Llama-3.2-1B-Instruct_lora_ppo_1ep.yaml
run_train examples/train_lora/llama3.2-1B/Llama-3.2-1B-Instruct_lora_ppo_predict_1ep.yaml
run_train examples/train_lora/llama3.2-1B/Llama-3.2-1B-Instruct_lora_dpo_1ep.yaml
run_train examples/train_lora/llama3.2-1B/Llama-3.2-1B-Instruct_lora_dpo_predict_1ep.yaml

run_train examples/train_lora/llama3.2-3B/Llama-3.2-3B-Instruct_lora_sft_d_3ep.yaml
run_train examples/train_lora/llama3.2-3B/Llama-3.2-3B-Instruct_lora_sft_d_predict_3ep.yaml
run_train examples/train_lora/llama3.2-3B/Llama-3.2-3B-Instruct_lora_sft_dpr_3ep.yaml
run_train examples/train_lora/llama3.2-3B/Llama-3.2-3B-Instruct_lora_sft_dpr_predict_3ep.yaml
run_train examples/train_lora/llama3.2-3B/Llama-3.2-3B-Instruct_lora_rm_1ep.yaml
run_train examples/train_lora/llama3.2-3B/Llama-3.2-3B-Instruct_lora_rm_predict_1ep.yaml
run_train examples/train_lora/llama3.2-3B/Llama-3.2-3B-Instruct_lora_ppo_1ep.yaml
run_train examples/train_lora/llama3.2-3B/Llama-3.2-3B-Instruct_lora_ppo_predict_1ep.yaml
run_train examples/train_lora/llama3.2-3B/Llama-3.2-3B-Instruct_lora_dpo_1ep.yaml
run_train examples/train_lora/llama3.2-3B/Llama-3.2-3B-Instruct_lora_dpo_predict_1ep.yaml

python emotional_results.py 2>&1 | tee "${LOG_DIR}/emotional_results.log"
echo "$(date '+%F %T') ALL DONE" | tee -a "${SUMMARY_LOG}"
