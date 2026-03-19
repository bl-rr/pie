#!/bin/bash
set -euo pipefail

GPU_ID="${GPU_ID:-2}"
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
CONTAINER_NAME="${CONTAINER_NAME:-pie-eval-sglang-radix-cache-gpu${GPU_ID}}"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run --rm --name "${CONTAINER_NAME}" --gpus "device=${GPU_ID}" \
    -p 8000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    lmsysorg/sglang:v0.4.4-cu124 \
    python3 -m sglang.launch_server \
    --model-path "${MODEL_ID}" \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --page-size 16 \
    --stream-interval 1 \
    --grammar-backend "xgrammar" \
    --disable-cuda-graph-padding \
    --disable-cuda-graph
