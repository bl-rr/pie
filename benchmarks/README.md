# Benchmark Instructions

All commands below assume your current directory is `benchmarks/`.

## Scope

This benchmark tree preserves the legacy SOSP'25 evaluation entrypoints and adds versioned baseline variants.

Legacy PIE entrypoints (unchanged):
- `test_1_agent_react_pie.py`
- `test_2_agent_codeact_pie.py`
- `test_3_agent_swarm_pie.py`
- `test_4_agent_case_study_pie.py`
- `test_5_text_completion_pie.py`
- `test_6_prefix_tree_pie.py`
- `test_7_tot_pie.py`
- `test_8_rot_pie.py`
- `test_9_got_pie.py`
- `test_10_skot_pie.py`
- `test_11_cache_pie.py`
- `test_12_ebnf_pie.py`
- `test_13_specdec_pie.py`
- `test_14_beamsearch_pie.py`
- `test_15_attnsink_pie.py`
- `microbench_spawn_time.py`
- `microbench_execution_latency.py`

New benchmarkable `sdk/examples` workload added:
- `test_16_parallel_generation_pie.py`

## Backend Version Freeze

Frozen on: `2026-03-02`.

| Backend | Pinned | Frozen Latest |
| --- | --- | --- |
| vLLM | `0.6.0` (`vllm/vllm-openai:v0.6.0`) | `0.16.0` (`vllm/vllm-openai:v0.16.0`) |
| SGLang | `0.4.4` (`lmsysorg/sglang:v0.4.4-cu124`) | `0.5.9` (`lmsysorg/sglang:v0.5.9`) |

Full matrix status is tracked in:
- `coverage_report.json`
- `coverage_report.md`

## Environment Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Build inferlets used by this benchmark suite:
```bash
# sdk/examples workloads
cd ../sdk/examples
cargo build --target wasm32-wasip2 --release

# std workloads used by benchmarks
echo "building std/text-completion"
cd ../std/text-completion
cargo build --target wasm32-wasip2 --release

echo "building std/beam-search"
cd ../beam-search
cargo build --target wasm32-wasip2 --release

# legacy eval-specific inferlets
cd ../../benchmarks/inferlets
cargo build --target wasm32-wasip2 --release

cd ..
```

3. Start PIE in another terminal, then run any `*_pie.py` script.
```bash
# Auto-updates ~/.pie-eval/config.toml model/device and starts PIE.
# Defaults: MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
#           GPU_ID=2  (sets device=cuda:2 for PIE model section)
./run_pie.sh
```

4. Start baseline servers as needed:
```bash
# compatibility aliases (pinned)
./run_vllm.sh
./run_sglang.sh

# explicit version scripts
./run_vllm_pinned.sh
./run_vllm_latest.sh
./run_sglang_pinned.sh
./run_sglang_latest.sh

# Optional overrides for all backend launchers:
#   GPU_ID=<gpu-index>   (default: 2)
#   MODEL_ID=<hf-model>  (default: meta-llama/Llama-3.1-8B-Instruct)

# Note: vLLM/SGLang launch scripts use Docker. Your user must have access to
# /var/run/docker.sock (or equivalent Docker daemon privileges).
```

## Batch Run Orchestration

Use one command to run full suites and auto-separate results by backend run:

```bash
./run_eval_matrix.sh
```

Default backends:
- `pie`
- `vllm_pinned`
- `vllm_latest`
- `sglang_pinned`
- `sglang_latest`

Execution behavior:
- Sequential only (one backend at a time).
- Explicit backend cleanup is performed before starting the next backend.
- A single `GPU_ID` value is applied to all backends in that run.
- PIE runs use `safe` load profile by default to reduce OOM risk on single-GPU 8B setups.
  This profile lowers per-script concurrency for PIE workloads only.
  Use `--pie-load-profile legacy` to run original script defaults.
- `--pie-load-profile safe-3.1-8b` uses a calibrated per-workload request map for
  Llama-3.1-8B and applies the same per-workload outer request/worker counts to
  vLLM/SGLang wrappers for comparability.
- With `safe-3.1-8b`, PIE backend restart-between-scripts is enabled by default
  (`--pie-restart-between-scripts`) to reduce cross-script OOM drift.
- If a Docker backend cannot start (e.g., no docker.sock permission), it is skipped by default
  and marked `unsupported` in `script_status.tsv`/summary.
  The runner also auto-tries `sudo docker` (non-interactive) when `DOCKER_CMD=docker`
  cannot access the daemon.
  Use `--strict-backends` to fail immediately instead.

Each backend run creates a directory under `benchmarks/results/` with this naming shape:
- `<timestamp>__<framework>[__v<version>]__gpu<id>__model-<model>__git-<sha>`

Examples:
- `20260301-233455__pie__gpu2__model-meta-llama_Llama-3.1-8B-Instruct__git-a1b2c3d`
- `20260301-235012__vllm__v0.6.0__gpu2__model-meta-llama_Llama-3.1-8B-Instruct__git-a1b2c3d`

Inside each run directory:
- `logs/` benchmark JSON logs (isolated for this run)
- `script_stdout/` stdout/stderr per benchmark script
- `backend_stdout.log` backend server output
- `script_status.tsv` per-script status (`ok`, `unsupported`, `failed`)
- `manifest.json` run metadata

After the matrix finishes, summary files are written at the results root:
- `run_summary_<batch-id>.md`
- `run_summary_latest.md` (updated pointer to the latest batch)
- includes per-script parsed benchmark output/metrics attributed to each backend run

Common options:
```bash
./run_eval_matrix.sh --backends "pie vllm_latest sglang_latest"
./run_eval_matrix.sh --gpu-id 2 --model-id meta-llama/Llama-3.1-8B-Instruct
./run_eval_matrix.sh --filter test_4
./run_eval_matrix.sh --no-microbench
./run_eval_matrix.sh --pie-load-profile legacy
./run_eval_matrix.sh --pie-load-profile safe-3.1-8b
./run_eval_matrix.sh --no-pie-restart-between-scripts
./run_eval_matrix.sh --script-timeout 1800
./run_eval_matrix.sh --strict-backends
./run_eval_matrix.sh --dry-run
```

Useful env override:
- `DOCKER_CMD="docker"` (default)
- `DOCKER_CMD="sudo docker"` (if your environment uses sudo for Docker)

Recalibrating `safe-3.1-8b` map (binary search from legacy defaults):
```bash
python3 -u calibrate_safe_31_8b.py --timeout 0 --poll-interval 10 \
  --output safe_31_8b_calibration_main16.json --scripts $(ls test_*_pie.py | sort -V)
```
Notes:
- The calibrator halves on failure, then binary-refines to the largest passing value.
- It polls backend logs every 10s for OOM signals and restarts PIE before the next probe.
- Per-attempt timeout defaults to `0` (disabled), so probes run until completion unless you pass `--timeout`.

## Legacy Commands (Unchanged)

Use the same commands as before:

```bash
python test_1_agent_react_pie.py
python test_2_agent_codeact_pie.py
python test_3_agent_swarm_pie.py
python test_4_agent_case_study_pie.py
python test_5_text_completion_pie.py
python test_6_prefix_tree_pie.py
python test_7_tot_pie.py
python test_8_rot_pie.py
python test_9_got_pie.py
python test_10_skot_pie.py
python test_11_cache_pie.py
python test_12_ebnf_pie.py
python test_13_specdec_pie.py
python test_14_beamsearch_pie.py
python test_15_attnsink_pie.py
python microbench_spawn_time.py
python microbench_execution_latency.py
```

Compatibility baseline entrypoints are preserved:
- `*_baseline.py` defaults to pinned vLLM behavior.
- existing `*_sglang.py` scripts remain runnable (`test_6_prefix_tree_sglang.py`, `test_7_tot_sglang.py`).

## Versioned Baseline Entrypoints

For each workload, wrappers are generated as exact versions only:
- `<workload>_vllm_<exactver>.py`
- `<workload>_sglang_<exactver>.py`

Examples:
```bash
python test_1_agent_react_vllm_0_6_0.py
python test_1_agent_react_vllm_0_16_0.py
python test_12_ebnf_sglang_0_5_9.py
python test_16_parallel_generation_sglang_0_4_4.py
```

## Omitted Cells

The following cells are intentionally omitted and return an explicit unsupported message:
- `test_13_specdec` on SGLang (pinned/latest): no equivalent n-gram speculative decoding endpoint contract.
- `test_14_beamsearch` on SGLang (pinned/latest): no validated parity for vLLM `use_beam_search` request contract.

See `coverage_report.md` for the complete matrix.

## Microbenchmark Helpers

Automation scripts remain available:
```bash
./microbench_spawn_time.sh
python microbench_spawn_time_viz.py

./microbench_execution_latency.sh
python microbench_execution_latency_viz.py
```
