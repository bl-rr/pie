#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[1/4] Building sdk/examples workspace..."
cd "${REPO_ROOT}/sdk/examples"
cargo build --target wasm32-wasip2 --release

echo "[2/4] Building std/text-completion..."
cd "${REPO_ROOT}/std/text-completion"
cargo build --target wasm32-wasip2 --release

echo "[3/4] Building std/beam-search..."
cd "${REPO_ROOT}/std/beam-search"
cargo build --target wasm32-wasip2 --release

echo "[4/4] Building benchmarks/inferlets workspace..."
cd "${REPO_ROOT}/benchmarks/inferlets"
cargo build --target wasm32-wasip2 --release

echo "Done."
