#!/bin/bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
GPU_ID="${GPU_ID:-2}"
PIE_HOME="${PIE_HOME:-$HOME/.pie-eval}"
PIE_CONFIG_PATH="${PIE_CONFIG_PATH:-$PIE_HOME/config.toml}"

export PIE_HOME
export PIE_CLI_HOME="${PIE_CLI_HOME:-$PIE_HOME}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIE_PROJECT_DIR="${PIE_PROJECT_DIR:-${SCRIPT_DIR}/../pie}"

if [[ ! -f "${PIE_PROJECT_DIR}/pyproject.toml" ]]; then
  echo "PIE project not found at ${PIE_PROJECT_DIR} (missing pyproject.toml)." >&2
  exit 1
fi

if [[ ! -f "${PIE_CONFIG_PATH}" ]]; then
  echo "Config not found at ${PIE_CONFIG_PATH}, initializing..."
  uv run --project "${PIE_PROJECT_DIR}" pie config init --path "${PIE_CONFIG_PATH}"
fi

python3 "${SCRIPT_DIR}/set_pie_model.py" \
  --config "${PIE_CONFIG_PATH}" \
  --model "${MODEL_ID}" \
  --device "cuda:${GPU_ID}"

echo "Starting PIE with model: ${MODEL_ID} on cuda:${GPU_ID}"
exec uv run --project "${PIE_PROJECT_DIR}" pie serve --config "${PIE_CONFIG_PATH}" "$@"
