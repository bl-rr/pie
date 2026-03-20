# Benchmark Instructions

All commands below assume your current directory is [`benchmarks/`](/home/leo/pie26-eval/pie-sosp-eval-update/benchmarks).

## Overview

This directory preserves the legacy SOSP'25 evaluation entrypoints and adds:

- exact-version vLLM and SGLang wrappers
- pinned and frozen-latest backend launchers
- a batch runner that starts backends automatically and writes isolated result directories
- coverage and summary artifacts for the benchmark matrix

There are two main ways to run benchmarks:

- direct/manual runs: you start the backend yourself, then run one benchmark script
- matrix runs: [`run_eval_matrix.sh`](/home/leo/pie26-eval/pie-sosp-eval-update/benchmarks/run_eval_matrix.sh) starts backends for you and runs a full suite

If you use [`run_eval_matrix.sh`](/home/leo/pie26-eval/pie-sosp-eval-update/benchmarks/run_eval_matrix.sh), you do not need to start PIE manually.

## Workloads

Legacy PIE entrypoints preserved as stable script names:

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

Additional benchmarkable workload added from current `sdk/examples`:

- `test_16_parallel_generation_pie.py`

Compatibility baseline entrypoints preserved:

- `*_baseline.py` defaults to pinned vLLM behavior
- legacy `*_sglang.py` compatibility wrappers remain where they previously existed

Exact-version baseline entrypoints exist per backend, for example:

- `test_1_agent_react_vllm_0_6_0.py`
- `test_1_agent_react_vllm_0_16_0.py`
- `test_12_ebnf_sglang_0_4_4.py`
- `test_12_ebnf_sglang_0_5_9.py`

Additional exact-version variants also exist for special cases that should be benchmarked separately. For example, prefix-tree includes extra vLLM variants such as `warmup` and `staged`. The matrix runner auto-discovers both:

- `test_*_<backend>_<version>.py`
- `test_*_<backend>_*_<version>.py`

## Backend Version Freeze

Frozen on: `2026-03-02`

| Backend | Pinned | Frozen Latest |
| --- | --- | --- |
| vLLM | `0.6.0` (`vllm/vllm-openai:v0.6.0`) | `0.16.0` (`vllm/vllm-openai:v0.16.0`) |
| SGLang | `0.4.4` (`lmsysorg/sglang:v0.4.4-cu124`) | `0.5.9` (`lmsysorg/sglang:v0.5.9`) |

Source of truth: [`backend_versions.toml`](/home/leo/pie26-eval/pie-sosp-eval-update/benchmarks/backend_versions.toml)

Matrix coverage status:

- [`coverage_report.json`](/home/leo/pie26-eval/pie-sosp-eval-update/benchmarks/coverage_report.json)
- [`coverage_report.md`](/home/leo/pie26-eval/pie-sosp-eval-update/benchmarks/coverage_report.md)

## Prerequisites

You need:

- Python with the dependencies in [`requirements.txt`](/home/leo/pie26-eval/pie-sosp-eval-update/benchmarks/requirements.txt)
- Rust + `wasm32-wasip2` target for inferlet builds
- `uv` for PIE local runs
- Docker access for vLLM and SGLang launchers
- model weights available in Hugging Face cache

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Build all inferlets used by this benchmark tree:

```bash
./build_inferlets.sh
```

That helper builds:

- `sdk/examples`
- `std/text-completion`
- `std/beam-search`
- `benchmarks/inferlets`

## Direct PIE Runs

Use this mode when you want to run PIE only, or when you want tight control over one workload at a time.

Start PIE:

```bash
./run_pie.sh
```

What [`run_pie.sh`](/home/leo/pie26-eval/pie-sosp-eval-update/benchmarks/run_pie.sh) does:

- uses `PIE_HOME=${HOME}/.pie-eval` by default
- initializes `~/.pie-eval/config.toml` if missing
- rewrites the configured model/device before launch
- runs `uv run --project ../pie pie serve --config ~/.pie-eval/config.toml`

Default launcher environment:

- `MODEL_ID=meta-llama/Llama-3.1-8B-Instruct`
- `GPU_ID=2`
- `PIE_HOME=$HOME/.pie-eval`
- `PIE_CONFIG_PATH=$PIE_HOME/config.toml`

Run a legacy PIE workload manually:

```bash
python test_1_agent_react_pie.py --server-uri ws://127.0.0.1:10009
python test_6_prefix_tree_pie.py --server-uri ws://127.0.0.1:10009
python test_16_parallel_generation_pie.py --server-uri ws://127.0.0.1:10009
```

Run a PIE workload with a smaller load:

```bash
python test_1_agent_react_pie.py --server-uri ws://127.0.0.1:10009 --num-instances 1
python test_3_agent_swarm_pie.py --server-uri ws://127.0.0.1:10009 --num-pipelines 1
```

## Direct Baseline Runs

Use this mode when you want one baseline backend running in another terminal and then invoke scripts by hand.

Start vLLM pinned:

```bash
./run_vllm_pinned.sh
```

Start vLLM latest:

```bash
./run_vllm_latest.sh
```

Start SGLang pinned:

```bash
./run_sglang_pinned.sh
```

Start SGLang latest:

```bash
./run_sglang_latest.sh
```

Compatibility aliases:

```bash
./run_vllm.sh
./run_sglang.sh
```

All baseline launchers accept:

- `GPU_ID=<gpu-index>`
- `MODEL_ID=<hf-model>`
- `DOCKER_CMD="docker"` or `DOCKER_CMD="sudo docker"`

Examples:

```bash
GPU_ID=2 MODEL_ID=meta-llama/Llama-3.1-8B-Instruct ./run_vllm_latest.sh
GPU_ID=2 MODEL_ID=meta-llama/Llama-3.1-8B-Instruct ./run_sglang_pinned.sh
```

Then run exact-version scripts directly:

```bash
python test_1_agent_react_vllm_0_6_0.py --host http://127.0.0.1 --port 8000
python test_6_prefix_tree_vllm_0_16_0.py --host http://127.0.0.1 --port 8000
python test_6_prefix_tree_vllm_warmup_0_16_0.py --host http://127.0.0.1 --port 8000
python test_12_ebnf_sglang_0_5_9.py --host http://127.0.0.1 --port 8000
```

## Matrix Runs

Use [`run_eval_matrix.sh`](/home/leo/pie26-eval/pie-sosp-eval-update/benchmarks/run_eval_matrix.sh) when you want the runner to manage backend startup, readiness checks, sequential execution, cleanup, and result summaries.

Default command:

```bash
./run_eval_matrix.sh
```

Default backends:

- `pie`
- `vllm_pinned`
- `vllm_latest`
- `sglang_pinned`
- `sglang_latest`

Important behavior:

- execution is sequential, one backend at a time
- the runner cleans up the previous backend before starting the next one
- PIE is started automatically if `pie` is included in `--backends`
- vLLM and SGLang are started in Docker
- all backends in the batch use the same `GPU_ID`
- the default PIE load profile is `safe`
- unavailable Docker backends are skipped and marked `unsupported` unless `--strict-backends` is used

The runner chooses backend launchers automatically:

- `pie` -> [`run_pie.sh`](/home/leo/pie26-eval/pie-sosp-eval-update/benchmarks/run_pie.sh)
- `vllm_pinned` -> [`run_vllm_pinned.sh`](/home/leo/pie26-eval/pie-sosp-eval-update/benchmarks/run_vllm_pinned.sh)
- `vllm_latest` -> [`run_vllm_latest.sh`](/home/leo/pie26-eval/pie-sosp-eval-update/benchmarks/run_vllm_latest.sh)
- `sglang_pinned` -> [`run_sglang_pinned.sh`](/home/leo/pie26-eval/pie-sosp-eval-update/benchmarks/run_sglang_pinned.sh)
- `sglang_latest` -> [`run_sglang_latest.sh`](/home/leo/pie26-eval/pie-sosp-eval-update/benchmarks/run_sglang_latest.sh)

## PIE-Only Runs

This is the recommended way to run the full PIE suite without any baseline backends.

Run all PIE workloads:

```bash
./run_eval_matrix.sh --backends "pie"
```

Run PIE only into a dedicated results tree:

```bash
./run_eval_matrix.sh --backends "pie" --results-root results/pie_only
```

Run one PIE workload only:

```bash
./run_eval_matrix.sh --backends "pie" --filter test_6_prefix_tree
```

Run PIE only with legacy script-default load:

```bash
./run_eval_matrix.sh --backends "pie" --pie-load-profile legacy
```

Run PIE only with the calibrated `safe-3.1-8b` map:

```bash
./run_eval_matrix.sh --backends "pie" --pie-load-profile safe-3.1-8b
```

Run PIE only with everything forced to one outer request/instance:

```bash
./run_eval_matrix.sh --backends "pie" --single-instance
```

Run PIE only with one total outer request/instance:

```bash
./run_eval_matrix.sh --backends "pie" --single-request
```

In PIE-only matrix mode, do not manually start PIE first. The runner launches and tears it down itself.
For most PIE scripts, `--single-instance` and `--single-request` are effectively the same because the script only exposes one outer count such as `--num-instances` or `--num-pipelines`.

## Common Matrix Options

Examples:

```bash
./run_eval_matrix.sh --backends "pie vllm_latest sglang_latest"
./run_eval_matrix.sh --gpu-id 2 --model-id meta-llama/Llama-3.1-8B-Instruct
./run_eval_matrix.sh --results-root ../result1
./run_eval_matrix.sh --filter test_4
./run_eval_matrix.sh --no-microbench
./run_eval_matrix.sh --single-instance
./run_eval_matrix.sh --single-request
./run_eval_matrix.sh --pie-load-profile legacy
./run_eval_matrix.sh --pie-load-profile safe-3.1-8b
./run_eval_matrix.sh --pie-restart-between-scripts
./run_eval_matrix.sh --no-pie-restart-between-scripts
./run_eval_matrix.sh --script-timeout 1800
./run_eval_matrix.sh --strict-backends
./run_eval_matrix.sh --dry-run
```

Most important knobs:

- `--backends "<list>"`: space-separated backend keys
- `--gpu-id <id>`: single GPU used by all backends in that run
- `--model-id <hf_repo>`: model passed to launchers and baseline scripts
- `--results-root <dir>`: output root; may be relative or absolute
- `--filter <substr>`: substring match on script filename
- `--no-microbench`: skip PIE microbench scripts
- `--single-instance`: force worker/concurrency knobs to `1`
- `--single-request`: force total request/instance knobs to `1`
- `--pie-load-profile <safe|safe-3.1-8b|legacy>`
- `--script-timeout <sec>`: per-script timeout; `0` disables
- `--dry-run`: print the planned backend/script commands without executing them

Useful environment variables:

- `DOCKER_CMD`
- `PIE_SCRIPT_ARGS`
- `BASELINE_SCRIPT_ARGS`
- `SAFE_31_8B_CONCURRENCY`
- `SINGLE_INSTANCE=1`
- `SINGLE_REQUEST=1`

Behavioral distinction:

- On baseline scripts, `--single-instance` limits worker concurrency, for example `--num-max-workers 1`, while leaving total request count unchanged.
- On baseline scripts, `--single-request` forces the total outer request count to `1`, for example `--num-requests 1` or `--num-pipelines 1`.
- On PIE scripts, both flags usually collapse to the same behavior because the legacy PIE entrypoints expose one outer count rather than separate total-request and worker knobs.

## Load Profiles

`safe`:

- default mode
- reduces PIE concurrency on riskier workloads
- leaves baseline scripts at their own defaults

`safe-3.1-8b`:

- calibrated for `meta-llama/Llama-3.1-8B-Instruct`
- uses a per-workload map for PIE concurrency
- applies comparable outer request/worker counts to baseline wrappers
- enables PIE restart-between-scripts by default when `PIE_RESTART_BETWEEN_SCRIPTS=auto`

`legacy`:

- preserves the benchmark scripts' original default arguments
- no runner-imposed PIE concurrency shaping

Recalibrate the `safe-3.1-8b` map:

```bash
python3 -u calibrate_safe_31_8b.py --timeout 0 --poll-interval 10 \
  --output safe_31_8b_calibration_main16.json --scripts $(ls test_*_pie.py | sort -V)
```

Calibration behavior:

- starts from legacy defaults
- halves on failure
- binary-refines to the largest passing value
- polls backend logs every 10s for OOM signals
- restarts PIE between attempts

## Results Layout

Each backend run creates one directory under the chosen results root:

- `<timestamp>__<framework>[__v<version>]__gpu<id>__model-<model>__git-<sha>`

Examples:

- `20260301-233455__pie__gpu2__model-meta-llama_Llama-3.1-8B-Instruct__git-a1b2c3d`
- `20260301-235012__vllm__v0.6.0__gpu2__model-meta-llama_Llama-3.1-8B-Instruct__git-a1b2c3d`

Inside each run directory:

- `backend_stdout.log`
- `script_stdout/`
- `logs/`
- `script_status.tsv`
- `scripts.txt`
- `manifest.json`

At the results root, the batch runner also writes:

- `last_runs.txt`
- `run_summary_<batch-id>.md`
- `run_summary_latest.md`

The generated summary parses and reports, when present in script stdout:

- total time
- throughput
- generated tokens
- per-token latency
- microbenchmark mean/median/stdev latency

## Legacy Manual Commands

These direct entrypoints remain runnable:

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
python test_16_parallel_generation_pie.py
python microbench_spawn_time.py
python microbench_execution_latency.py
```

When you run them manually, you are responsible for starting the backend yourself.

## Omitted Cells

The following backend/workload cells are intentionally omitted and return explicit unsupported output:

- `test_13_specdec` on SGLang pinned/latest
- `test_14_beamsearch` on SGLang pinned/latest

See [`coverage_report.md`](/home/leo/pie26-eval/pie-sosp-eval-update/benchmarks/coverage_report.md) for the full matrix and rationale.

## Microbenchmark Helpers

Helper scripts remain available:

```bash
./microbench_spawn_time.sh
python microbench_spawn_time_viz.py

./microbench_execution_latency.sh
python microbench_execution_latency_viz.py
```
