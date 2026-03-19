#!/bin/bash
set -euo pipefail

GPU_ID="${GPU_ID:-2}"
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
CONTAINER_NAME="${CONTAINER_NAME:-pie-eval-vllm-apc-gpu${GPU_ID}}"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run --rm --name "${CONTAINER_NAME}" --runtime nvidia --gpus "device=${GPU_ID}" \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:v0.6.0 \
    --model "${MODEL_ID}" \
    --max_model_len 4096 \
    --max-logprobs 64 \
    --disable-sliding-window \
    --enforce-eager \
    --block-size 16 \
    --dtype bfloat16 \
    --enable-prefix-caching
