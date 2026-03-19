#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RESULTS_ROOT="${RESULTS_ROOT:-${SCRIPT_DIR}/results}"
GPU_ID="${GPU_ID:-2}"
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
BACKENDS="${BACKENDS:-pie vllm_pinned vllm_latest sglang_pinned sglang_latest}"
DOCKER_CMD="${DOCKER_CMD:-docker}"
PIE_SERVER_URI="${PIE_SERVER_URI:-ws://127.0.0.1:10009}"
OPENAI_HOST="${OPENAI_HOST:-http://127.0.0.1}"
OPENAI_PORT="${OPENAI_PORT:-8000}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
INCLUDE_MICROBENCH="${INCLUDE_MICROBENCH:-1}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
SKIP_UNAVAILABLE_BACKENDS="${SKIP_UNAVAILABLE_BACKENDS:-1}"
BENCH_FILTER="${BENCH_FILTER:-}"
PIE_SCRIPT_ARGS="${PIE_SCRIPT_ARGS:-}"
BASELINE_SCRIPT_ARGS="${BASELINE_SCRIPT_ARGS:-}"
PIE_LOAD_PROFILE="${PIE_LOAD_PROFILE:-safe}"
SAFE_31_8B_CONCURRENCY="${SAFE_31_8B_CONCURRENCY:-1}"
PIE_INSTANCE_RETRIES_SAFE="${PIE_INSTANCE_RETRIES_SAFE:-2}"
PIE_RESTART_BETWEEN_SCRIPTS="${PIE_RESTART_BETWEEN_SCRIPTS:-auto}"
SCRIPT_TIMEOUT="${SCRIPT_TIMEOUT:-0}"
DRY_RUN="${DRY_RUN:-0}"
SINGLE_INSTANCE="${SINGLE_INSTANCE:-0}"
BATCH_STARTED_AT="$(date -Iseconds)"

usage() {
    cat <<'EOF'
Usage: ./run_eval_matrix.sh [options]

Runs benchmark suites for PIE and baseline backends, creating one result directory
per backend run. Each directory name includes framework/version/gpu/timestamp/model/git.

Options:
  --backends "<list>"       Space-separated backend keys
                            Default: "pie vllm_pinned vllm_latest sglang_pinned sglang_latest"
                            Allowed: pie vllm_pinned vllm_latest sglang_pinned sglang_latest
  --gpu-id <id>             GPU index passed to backend launch scripts (default: 2)
  --model-id <hf_repo>      Model ID used by launchers and benchmark scripts
  --results-root <dir>      Output root directory (default: benchmarks/results)
  --pie-server-uri <uri>    PIE server URI for *_pie.py scripts (default: ws://127.0.0.1:10009)
  --openai-host <host>      Baseline host (default: http://127.0.0.1)
  --openai-port <port>      Baseline port (default: 8000)
  --startup-timeout <sec>   Seconds to wait for backend readiness (default: 600)
  --filter <substr>         Only run benchmark scripts containing this substring
  --no-microbench           Skip PIE microbench scripts
  --single-instance         Force all benchmark concurrency/request knobs to 1
  --pie-load-profile <mode> PIE load profile: safe|safe-3.1-8b|legacy (default: safe)
  --pie-restart-between-scripts
                            Restart PIE backend between PIE scripts
                            (default: auto, enabled for safe-3.1-8b)
  --no-pie-restart-between-scripts
                            Keep PIE backend running across PIE scripts
  --script-timeout <sec>    Per-script timeout in seconds; 0 disables timeout (default: 0)
  --stop-on-error           Stop the current backend run at first benchmark failure
  --strict-backends         Fail immediately if a backend cannot start (default: skip unavailable)
  --skip-unavailable-backends
                            Skip unavailable backends and mark scripts as unsupported
  --dry-run                 Print planned actions without running
  -h, --help                Show help

Environment variables with the same names are also supported.
Extra benchmark args:
  PIE_SCRIPT_ARGS="<args>"        appended to every *_pie.py call
  BASELINE_SCRIPT_ARGS="<args>"   appended to every baseline wrapper call
  SAFE_31_8B_CONCURRENCY=<n>      Fallback value for unmapped scripts in safe-3.1-8b profile
  SINGLE_INSTANCE=1               Equivalent to --single-instance
  DOCKER_CMD="<cmd>"              Docker command, e.g. "docker" or "sudo docker"

Examples:
  ./run_eval_matrix.sh
  ./run_eval_matrix.sh --backends "pie vllm_pinned vllm_latest sglang_pinned sglang_latest"
  ./run_eval_matrix.sh --backends "vllm_latest sglang_latest" --gpu-id 2
  ./run_eval_matrix.sh --pie-load-profile legacy
  BENCH_FILTER=test_4 ./run_eval_matrix.sh --backends "pie vllm_pinned"
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backends)
            BACKENDS="$2"
            shift 2
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --model-id)
            MODEL_ID="$2"
            shift 2
            ;;
        --results-root)
            RESULTS_ROOT="$2"
            shift 2
            ;;
        --pie-server-uri)
            PIE_SERVER_URI="$2"
            shift 2
            ;;
        --openai-host)
            OPENAI_HOST="$2"
            shift 2
            ;;
        --openai-port)
            OPENAI_PORT="$2"
            shift 2
            ;;
        --startup-timeout)
            STARTUP_TIMEOUT="$2"
            shift 2
            ;;
        --filter)
            BENCH_FILTER="$2"
            shift 2
            ;;
        --no-microbench)
            INCLUDE_MICROBENCH="0"
            shift
            ;;
        --single-instance)
            SINGLE_INSTANCE="1"
            shift
            ;;
        --pie-load-profile)
            PIE_LOAD_PROFILE="$2"
            shift 2
            ;;
        --pie-restart-between-scripts)
            PIE_RESTART_BETWEEN_SCRIPTS="1"
            shift
            ;;
        --no-pie-restart-between-scripts)
            PIE_RESTART_BETWEEN_SCRIPTS="0"
            shift
            ;;
        --script-timeout)
            SCRIPT_TIMEOUT="$2"
            shift 2
            ;;
        --stop-on-error)
            STOP_ON_ERROR="1"
            shift
            ;;
        --strict-backends)
            SKIP_UNAVAILABLE_BACKENDS="0"
            shift
            ;;
        --skip-unavailable-backends)
            SKIP_UNAVAILABLE_BACKENDS="1"
            shift
            ;;
        --dry-run)
            DRY_RUN="1"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 2
            ;;
    esac
done

case "${PIE_LOAD_PROFILE}" in
    safe|safe-3.1-8b|legacy) ;;
    *)
        echo "[run_eval_matrix] ERROR: Invalid --pie-load-profile: ${PIE_LOAD_PROFILE}. Use safe, safe-3.1-8b, or legacy." >&2
        exit 1
        ;;
esac

case "${PIE_RESTART_BETWEEN_SCRIPTS}" in
    auto|0|1) ;;
    *)
        echo "[run_eval_matrix] ERROR: Invalid PIE_RESTART_BETWEEN_SCRIPTS: ${PIE_RESTART_BETWEEN_SCRIPTS}. Use auto, 0, or 1." >&2
        exit 1
        ;;
esac

if [[ "${PIE_RESTART_BETWEEN_SCRIPTS}" == "auto" ]]; then
    if [[ "${PIE_LOAD_PROFILE}" == "safe-3.1-8b" ]]; then
        PIE_RESTART_BETWEEN_SCRIPTS="1"
    else
        PIE_RESTART_BETWEEN_SCRIPTS="0"
    fi
fi

if ! [[ "${SCRIPT_TIMEOUT}" =~ ^[0-9]+$ ]]; then
    echo "[run_eval_matrix] ERROR: Invalid --script-timeout: ${SCRIPT_TIMEOUT}. Use a non-negative integer." >&2
    exit 1
fi

if ! [[ "${PIE_INSTANCE_RETRIES_SAFE}" =~ ^[0-9]+$ ]]; then
    echo "[run_eval_matrix] ERROR: Invalid PIE_INSTANCE_RETRIES_SAFE: ${PIE_INSTANCE_RETRIES_SAFE}. Use a non-negative integer." >&2
    exit 1
fi

if ! [[ "${SAFE_31_8B_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[run_eval_matrix] ERROR: Invalid SAFE_31_8B_CONCURRENCY: ${SAFE_31_8B_CONCURRENCY}. Use a positive integer." >&2
    exit 1
fi

if ! [[ "${SKIP_UNAVAILABLE_BACKENDS}" =~ ^[01]$ ]]; then
    echo "[run_eval_matrix] ERROR: Invalid SKIP_UNAVAILABLE_BACKENDS: ${SKIP_UNAVAILABLE_BACKENDS}. Use 0 or 1." >&2
    exit 1
fi

if ! [[ "${SINGLE_INSTANCE}" =~ ^[01]$ ]]; then
    echo "[run_eval_matrix] ERROR: Invalid SINGLE_INSTANCE: ${SINGLE_INSTANCE}. Use 0 or 1." >&2
    exit 1
fi

mkdir -p "${RESULTS_ROOT}"

log() {
    echo "[run_eval_matrix] $*"
}

die() {
    echo "[run_eval_matrix] ERROR: $*" >&2
    exit 1
}

sanitize_component() {
    local value="$1"
    value="${value//\//_}"
    value="${value//:/_}"
    value="${value//@/_}"
    value="${value// /_}"
    value="$(printf '%s' "${value}" | sed -E 's/[^A-Za-z0-9._-]/_/g')"
    printf '%s' "${value}"
}

backend_family() {
    case "$1" in
        pie) echo "pie" ;;
        vllm_pinned|vllm_latest) echo "vllm" ;;
        sglang_pinned|sglang_latest) echo "sglang" ;;
        *) die "Unsupported backend key: $1" ;;
    esac
}

backend_mode() {
    case "$1" in
        pie) echo "local" ;;
        *_pinned) echo "pinned" ;;
        *_latest) echo "latest" ;;
        *) die "Unsupported backend key: $1" ;;
    esac
}

backend_launcher() {
    case "$1" in
        pie) echo "${SCRIPT_DIR}/run_pie.sh" ;;
        vllm_pinned) echo "${SCRIPT_DIR}/run_vllm_pinned.sh" ;;
        vllm_latest) echo "${SCRIPT_DIR}/run_vllm_latest.sh" ;;
        sglang_pinned) echo "${SCRIPT_DIR}/run_sglang_pinned.sh" ;;
        sglang_latest) echo "${SCRIPT_DIR}/run_sglang_latest.sh" ;;
        *) die "Unsupported backend key: $1" ;;
    esac
}

backend_container_name() {
    case "$1" in
        pie) echo "" ;;
        vllm_pinned) echo "pie-eval-vllm-pinned-gpu${GPU_ID}" ;;
        vllm_latest) echo "pie-eval-vllm-latest-gpu${GPU_ID}" ;;
        sglang_pinned) echo "pie-eval-sglang-pinned-gpu${GPU_ID}" ;;
        sglang_latest) echo "pie-eval-sglang-latest-gpu${GPU_ID}" ;;
        *) die "Unsupported backend key: $1" ;;
    esac
}

docker_is_available() {
    # shellcheck disable=SC2206
    local -a docker_cmd=( ${DOCKER_CMD} )
    local docker_bin="${docker_cmd[0]:-}"
    [[ -n "${docker_bin}" ]] && command -v "${docker_bin}" >/dev/null 2>&1
}

run_docker() {
    # shellcheck disable=SC2206
    local -a docker_cmd=( ${DOCKER_CMD} )
    "${docker_cmd[@]}" "$@"
}

docker_can_access_daemon() {
    run_docker info >/dev/null 2>&1
}

try_enable_sudo_docker() {
    if [[ "${DOCKER_CMD}" != "docker" ]]; then
        return 1
    fi
    if ! command -v sudo >/dev/null 2>&1; then
        return 1
    fi
    if sudo -n docker info >/dev/null 2>&1; then
        DOCKER_CMD="sudo docker"
        log "Docker daemon requires elevated access; using DOCKER_CMD='${DOCKER_CMD}'."
        return 0
    fi
    return 1
}

stop_named_container() {
    local container_name="${1:-}"
    if [[ -z "${container_name}" ]]; then
        return
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        return
    fi
    if docker_is_available; then
        run_docker rm -f "${container_name}" >/dev/null 2>&1 || true
    fi
}

generate_batch_summary() {
    local rows_file="$1"
    local summary_path="$2"
    local ended_at="$3"

    python3 - "${rows_file}" "${summary_path}" "${BATCH_STARTED_AT}" "${ended_at}" "${RESULTS_ROOT}" "${BACKENDS}" "${MODEL_ID}" "${GPU_ID}" "${PIE_SERVER_URI}" "${OPENAI_HOST}" "${OPENAI_PORT}" <<'PY'
import csv
import re
import sys
from collections import Counter
from pathlib import Path

(
    rows_file,
    summary_path,
    started_at,
    ended_at,
    results_root,
    backends,
    model_id,
    gpu_id,
    pie_server_uri,
    openai_host,
    openai_port,
) = sys.argv[1:]

rows: list[dict] = []
with open(rows_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        rows.append(row)

float_re = r"([0-9]+(?:\.[0-9]+)?)"
total_ms_re = re.compile(
    r"Total\s+Time\s+Taken:\s*" + float_re + r"\s*milliseconds", re.IGNORECASE
)
throughput_re = re.compile(
    r"Throughput:\s*" + float_re + r"\s*requests/second", re.IGNORECASE
)
mean_us_re = re.compile(r"mean\s+latency:\s*" + float_re + r"\s*[uμ]s", re.IGNORECASE)
median_us_re = re.compile(
    r"median\s+latency:\s*" + float_re + r"\s*[uμ]s", re.IGNORECASE
)
stdev_us_re = re.compile(
    r"(?:stdev|std(?:_dev)?|std\s+dev)\s+latency:\s*" + float_re + r"\s*[uμ]s",
    re.IGNORECASE,
)
generated_tokens_re = re.compile(
    r"Total\s+Generated\s+Tokens:\s*([0-9]+)", re.IGNORECASE
)
per_token_latency_ms_re = re.compile(
    r"Per-Token\s+Latency:\s*" + float_re + r"\s*ms/token", re.IGNORECASE
)
bench_block_re = re.compile(
    r"---\s*.*Benchmark Complete\s*---.*?(?:-{10,}|$)",
    re.IGNORECASE | re.DOTALL,
)


def _last_float(pattern: re.Pattern[str], text: str) -> float | None:
    match = None
    for candidate in pattern.finditer(text):
        match = candidate
    if match is None:
        return None
    return float(match.group(1))


def _extract_benchmark_snippet(text: str) -> str | None:
    matches = list(bench_block_re.finditer(text))
    if not matches:
        return None
    lines = [line.strip() for line in matches[-1].group(0).splitlines() if line.strip()]
    if not lines:
        return None
    return " | ".join(lines[:6])


def _last_int(pattern: re.Pattern[str], text: str) -> int | None:
    match = None
    for candidate in pattern.finditer(text):
        match = candidate
    if match is None:
        return None
    return int(match.group(1))


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "-"
    return str(value)


def _md_cell(value: str | None) -> str:
    if value is None or value == "":
        return "-"
    return value.replace("|", "\\|").replace("\n", " ")

total_scripts = 0
total_ok = 0
total_unsupported = 0
total_failed = 0

for row in rows:
    status_path = Path(row["run_dir"]) / "script_status.tsv"
    counts: Counter[str] = Counter()
    failed_scripts: list[str] = []
    unsupported_scripts: list[str] = []
    script_metrics: list[dict] = []

    if status_path.exists():
        with status_path.open("r", encoding="utf-8") as sf:
            status_reader = csv.DictReader(sf, delimiter="\t")
            for status_row in status_reader:
                status = (status_row.get("status") or "").strip()
                script = (status_row.get("script") or "").strip()
                counts[status] += 1
                if status == "failed":
                    failed_scripts.append(script)
                if status == "unsupported":
                    unsupported_scripts.append(script)

                log_path = Path(row["run_dir"]) / "script_stdout" / f"{Path(script).stem}.log"
                total_ms = None
                throughput_rps = None
                mean_us = None
                median_us = None
                stdev_us = None
                generated_tokens = None
                per_token_latency_ms = None
                snippet = None
                note = ""

                if log_path.exists():
                    text = log_path.read_text(encoding="utf-8", errors="replace")
                    total_ms = _last_float(total_ms_re, text)
                    throughput_rps = _last_float(throughput_re, text)
                    mean_us = _last_float(mean_us_re, text)
                    median_us = _last_float(median_us_re, text)
                    stdev_us = _last_float(stdev_us_re, text)
                    generated_tokens = _last_int(generated_tokens_re, text)
                    per_token_latency_ms = _last_float(per_token_latency_ms_re, text)
                    snippet = _extract_benchmark_snippet(text)
                    if status == "ok" and snippet is None:
                        note = "no benchmark completion block found"
                else:
                    note = "stdout log missing"

                script_metrics.append(
                    {
                        "script": script,
                        "status": status,
                        "log_path": str(log_path),
                        "total_ms": total_ms,
                        "throughput_rps": throughput_rps,
                        "mean_us": mean_us,
                        "median_us": median_us,
                        "stdev_us": stdev_us,
                        "generated_tokens": generated_tokens,
                        "per_token_latency_ms": per_token_latency_ms,
                        "snippet": snippet,
                        "note": note,
                    }
                )

    row["ok"] = str(counts.get("ok", 0))
    row["unsupported"] = str(counts.get("unsupported", 0))
    row["failed"] = str(counts.get("failed", 0))
    row["failed_scripts"] = failed_scripts
    row["unsupported_scripts"] = unsupported_scripts
    row["script_metrics"] = script_metrics

    total_scripts += int(row.get("total_scripts", 0))
    total_ok += int(row["ok"])
    total_unsupported += int(row["unsupported"])
    total_failed += int(row["failed"])

lines: list[str] = []
lines.append("# Eval Matrix Summary")
lines.append("")
lines.append(f"- Started: `{started_at}`")
lines.append(f"- Ended: `{ended_at}`")
lines.append(f"- Results Root: `{results_root}`")
lines.append(f"- Backends: `{backends}`")
lines.append(f"- Model: `{model_id}`")
lines.append(f"- GPU: `cuda:{gpu_id}`")
lines.append(f"- PIE URI: `{pie_server_uri}`")
lines.append(f"- Baseline Endpoint: `{openai_host}:{openai_port}`")
lines.append("")
lines.append("| Backend | Version | Mode | Scripts | Ok | Unsupported | Failed | Run Dir |")
lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- |")

for row in rows:
    version = row["version"] if row["version"] else "n/a"
    lines.append(
        f"| `{row['backend']}` | `{version}` | `{row['mode']}` | "
        f"{row['total_scripts']} | {row['ok']} | {row['unsupported']} | {row['failed']} | "
        f"`{row['run_dir']}` |"
    )

lines.append("")
lines.append("## Totals")
lines.append("")
lines.append(f"- Scripts: `{total_scripts}`")
lines.append(f"- Ok: `{total_ok}`")
lines.append(f"- Unsupported: `{total_unsupported}`")
lines.append(f"- Failed: `{total_failed}`")
lines.append("")

for row in rows:
    backend = row["backend"]
    failed = row["failed_scripts"]
    unsupported = row["unsupported_scripts"]
    if failed:
        lines.append(f"## Failed Scripts (`{backend}`)")
        lines.append("")
        for script in failed:
            lines.append(f"- `{script}`")
        lines.append("")
    if unsupported:
        lines.append(f"## Unsupported Scripts (`{backend}`)")
        lines.append("")
        for script in unsupported:
            lines.append(f"- `{script}`")
        lines.append("")

for row in rows:
    lines.append(f"## Script Metrics (`{row['backend']}`)")
    lines.append("")
    lines.append(
        "| Script | Status | Total ms | Throughput rps | Generated Tokens | Per-Token ms | Mean us | Median us | Stdev us | Parsed Output | Log |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    metrics = row.get("script_metrics", [])
    if not metrics:
        lines.append("| - | - | - | - | - | - | - | - | - | - | - |")
    else:
        for m in metrics:
            snippet_or_note = m["snippet"] or m["note"] or ""
            lines.append(
                f"| `{m['script']}` | `{m['status']}` | "
                f"{_fmt_float(m['total_ms'])} | {_fmt_float(m['throughput_rps'])} | "
                f"{_fmt_int(m['generated_tokens'])} | {_fmt_float(m['per_token_latency_ms'])} | "
                f"{_fmt_float(m['mean_us'])} | {_fmt_float(m['median_us'])} | {_fmt_float(m['stdev_us'])} | "
                f"{_md_cell(snippet_or_note)} | `{m['log_path']}` |"
            )
    lines.append("")

Path(summary_path).write_text("\n".join(lines), encoding="utf-8")
PY
}

backend_version() {
    local backend="$1"
    local family
    family="$(backend_family "${backend}")"
    local mode
    mode="$(backend_mode "${backend}")"

    if [[ "${family}" == "pie" ]]; then
        echo ""
        return
    fi

    python3 - "${SCRIPT_DIR}/backend_versions.toml" "${family}" "${mode}" <<'PY'
import sys
import tomllib

toml_path, family, mode = sys.argv[1:]
with open(toml_path, "rb") as f:
    data = tomllib.load(f)

print(data[family][mode]["version"])
PY
}

parse_host_port_from_uri() {
    python3 - "$1" <<'PY'
import sys
from urllib.parse import urlparse

uri = sys.argv[1]
parsed = urlparse(uri)
host = parsed.hostname or "127.0.0.1"
if parsed.port is not None:
    port = parsed.port
elif parsed.scheme in {"https", "wss"}:
    port = 443
else:
    port = 80
print(host)
print(port)
PY
}

check_tcp_once() {
    local host="$1"
    local port="$2"
    python3 - "${host}" "${port}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1.0)
try:
    sock.connect((host, port))
except OSError:
    sys.exit(1)
finally:
    sock.close()

sys.exit(0)
PY
}

check_openai_http_once() {
    local base_url="$1"
    python3 - "${base_url}" <<'PY'
import sys
import urllib.error
import urllib.request

base_url = sys.argv[1].rstrip("/")
paths = ("/v1/models", "/health")

for path in paths:
    url = base_url + path
    req = urllib.request.Request(url=url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            if resp.status in (200, 404):
                sys.exit(0)
    except urllib.error.HTTPError as e:
        if e.code in (200, 404):
            sys.exit(0)
    except Exception:
        pass

sys.exit(1)
PY
}

file_contains() {
    local pattern="$1"
    local path="$2"

    if command -v rg >/dev/null 2>&1; then
        rg -F -q -- "${pattern}" "${path}"
    else
        grep -F -q -- "${pattern}" "${path}"
    fi
}

check_pie_engine_ready_once() {
    local log_path="$1"
    if [[ ! -f "${log_path}" ]]; then
        return 1
    fi
    # Match a stable ASCII substring instead of relying on the leading checkmark glyph.
    file_contains "Engine running. Press Ctrl+C to stop" "${log_path}"
}

pie_profile_args_for_script() {
    local script="$1"

    if [[ "${SINGLE_INSTANCE}" == "1" ]]; then
        case "${script}" in
            test_3_agent_swarm_pie.py)
                printf '%s\n' "--num-pipelines" "1"
                ;;
            test_*_pie.py|microbench_spawn_time.py|microbench_execution_latency.py)
                printf '%s\n' "--num-instances" "1"
                ;;
        esac
        return 0
    fi

    if [[ "${PIE_LOAD_PROFILE}" == "legacy" ]]; then
        return 0
    fi

    if [[ "${PIE_LOAD_PROFILE}" == "safe-3.1-8b" ]]; then
        local target
        target="$(safe_31_8b_value_for_script "${script}")"
        case "${script}" in
            test_3_agent_swarm_pie.py)
                printf '%s\n' "--num-pipelines" "${target}"
                ;;
            test_*_pie.py|microbench_spawn_time.py|microbench_execution_latency.py)
                printf '%s\n' "--num-instances" "${target}"
                ;;
        esac
        return 0
    fi

    case "${script}" in
        test_11_cache_pie.py)
            printf '%s\n' "--num-instances" "1"
            ;;
        test_1_agent_react_pie.py|test_2_agent_codeact_pie.py|test_4_agent_case_study_pie.py|test_5_text_completion_pie.py|test_6_prefix_tree_pie.py|test_10_skot_pie.py|test_12_ebnf_pie.py|test_13_specdec_pie.py)
            printf '%s\n' "--num-instances" "2"
            ;;
        test_3_agent_swarm_pie.py)
            printf '%s\n' "--num-pipelines" "1"
            ;;
        test_7_tot_pie.py|test_8_rot_pie.py|test_9_got_pie.py|test_14_beamsearch_pie.py|test_15_attnsink_pie.py|test_16_parallel_generation_pie.py)
            printf '%s\n' "--num-instances" "1"
            ;;
        microbench_spawn_time.py|microbench_execution_latency.py)
            printf '%s\n' "--num-instances" "100"
            ;;
    esac
}

safe_31_8b_value_for_script() {
    local script="$1"
    case "${script}" in
        test_1_agent_react_*.py) echo "36" ;;
        test_2_agent_codeact_*.py) echo "128" ;;
        test_3_agent_swarm_*.py) echo "31" ;;
        test_4_agent_case_study_*.py) echo "60" ;;
        test_5_text_completion_*.py) echo "128" ;;
        test_6_prefix_tree_*.py) echo "58" ;;
        test_7_tot_*.py) echo "64" ;;
        test_8_rot_*.py) echo "32" ;;
        test_9_got_*.py) echo "48" ;;
        test_10_skot_*.py) echo "128" ;;
        test_11_cache_*.py) echo "1" ;;
        test_12_ebnf_*.py) echo "128" ;;
        test_13_specdec_*.py) echo "128" ;;
        test_14_beamsearch_*.py) echo "28" ;;
        test_15_attnsink_*.py) echo "128" ;;
        test_16_parallel_generation_*.py) echo "128" ;;
        microbench_spawn_time.py) echo "100" ;;
        microbench_execution_latency.py) echo "100" ;;
        *) echo "${SAFE_31_8B_CONCURRENCY}" ;;
    esac
}

baseline_profile_args_for_script() {
    local script="$1"

    if [[ "${SINGLE_INSTANCE}" == "1" ]]; then
        case "${script}" in
            test_3_agent_swarm_*.py)
                printf '%s\n' "--num-pipelines" "1"
                printf '%s\n' "--num-max-workers" "1"
                ;;
            test_*_vllm_*.py|test_*_sglang_*.py|test_*_baseline.py)
                printf '%s\n' "--num-requests" "1"
                printf '%s\n' "--num-max-workers" "1"
                ;;
        esac
        return 0
    fi

    if [[ "${PIE_LOAD_PROFILE}" != "safe-3.1-8b" ]]; then
        return 0
    fi

    local target
    target="$(safe_31_8b_value_for_script "${script}")"

    case "${script}" in
        test_3_agent_swarm_*.py)
            printf '%s\n' "--num-pipelines" "${target}"
            printf '%s\n' "--num-max-workers" "${target}"
            ;;
        test_7_tot_*.py|test_9_got_*.py)
            printf '%s\n' "--num-requests" "${target}"
            printf '%s\n' "--num-max-workers" "${target}"
            ;;
        test_*_vllm_*.py|test_*_sglang_*.py|test_*_baseline.py)
            printf '%s\n' "--num-requests" "${target}"
            printf '%s\n' "--num-max-workers" "${target}"
            ;;
    esac
}

collect_scripts_for_backend() {
    local backend="$1"
    local family
    family="$(backend_family "${backend}")"
    local version
    version="$(backend_version "${backend}")"
    local version_token="${version//./_}"

    local -a scripts=()
    if [[ "${family}" == "pie" ]]; then
        while IFS= read -r line; do
            scripts+=("${line}")
        done < <(cd "${SCRIPT_DIR}" && find . -maxdepth 1 -type f -name 'test_*_pie.py' -printf '%f\n' | sort -V)

        if [[ "${INCLUDE_MICROBENCH}" == "1" ]]; then
            scripts+=("microbench_spawn_time.py" "microbench_execution_latency.py")
        fi
    else
        while IFS= read -r line; do
            scripts+=("${line}")
        done < <(
            cd "${SCRIPT_DIR}" && {
                find . -maxdepth 1 -type f -name "test_*_${family}_${version_token}.py" -printf '%f\n'
                find . -maxdepth 1 -type f -name "test_*_${family}_*_${version_token}.py" -printf '%f\n'
            } | sort -u -V
        )
    fi

    if [[ -n "${BENCH_FILTER}" ]]; then
        local -a filtered=()
        local s
        for s in "${scripts[@]}"; do
            if [[ "${s}" == *"${BENCH_FILTER}"* ]]; then
                filtered+=("${s}")
            fi
        done
        scripts=("${filtered[@]}")
    fi

    printf '%s\n' "${scripts[@]}"
}

write_manifest() {
    local path="$1"
    local backend="$2"
    local framework="$3"
    local version="$4"
    local mode="$5"
    local run_name="$6"
    local started_at="$7"
    local git_sha="$8"
    local scripts_file="$9"
    local statuses_file="${10}"

    python3 - "${path}" "${backend}" "${framework}" "${version}" "${mode}" "${run_name}" "${started_at}" "${git_sha}" "${MODEL_ID}" "${GPU_ID}" "${PIE_SERVER_URI}" "${OPENAI_HOST}" "${OPENAI_PORT}" "${scripts_file}" "${statuses_file}" <<'PY'
import json
import sys
from pathlib import Path

(
    out_path,
    backend_key,
    framework,
    version,
    mode,
    run_name,
    started_at,
    git_sha,
    model_id,
    gpu_id,
    pie_server_uri,
    openai_host,
    openai_port,
    scripts_file,
    statuses_file,
) = sys.argv[1:]

scripts = []
statuses = []

scripts_path = Path(scripts_file)
if scripts_path.exists():
    scripts = [line.strip() for line in scripts_path.read_text(encoding="utf-8").splitlines() if line.strip()]

status_path = Path(statuses_file)
if status_path.exists():
    lines = [line for line in status_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    header = []
    for idx, line in enumerate(lines):
        cols = line.split("\t")
        if idx == 0:
            header = cols
            continue
        statuses.append(dict(zip(header, cols)))

manifest = {
    "backend_key": backend_key,
    "framework": framework,
    "version": version if version else None,
    "mode": mode,
    "run_name": run_name,
    "started_at": started_at,
    "git_sha": git_sha,
    "gpu_id": gpu_id,
    "model_id": model_id,
    "pie_server_uri": pie_server_uri,
    "openai_host": openai_host,
    "openai_port": int(openai_port),
    "scripts": scripts,
    "statuses": statuses,
}

Path(out_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY
}

BACKEND_PID=""
CURRENT_BACKEND=""
CURRENT_CONTAINER_NAME=""
CURRENT_BACKEND_LOG=""
BACKEND_START_ERROR=""
SUMMARY_ROWS_FILE=""
SUMMARY_PATH=""
LATEST_SUMMARY_PATH=""

cleanup_backend() {
    if [[ -n "${BACKEND_PID}" ]] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
        kill "${BACKEND_PID}" >/dev/null 2>&1 || true
        if command -v pkill >/dev/null 2>&1; then
            pkill -TERM -P "${BACKEND_PID}" >/dev/null 2>&1 || true
        fi
        local waited=0
        while kill -0 "${BACKEND_PID}" 2>/dev/null && [[ ${waited} -lt 15 ]]; do
            sleep 1
            waited=$((waited + 1))
        done
        if kill -0 "${BACKEND_PID}" 2>/dev/null; then
            kill -KILL "${BACKEND_PID}" >/dev/null 2>&1 || true
        fi
        wait "${BACKEND_PID}" >/dev/null 2>&1 || true
    fi
    stop_named_container "${CURRENT_CONTAINER_NAME}"
    BACKEND_PID=""
    CURRENT_BACKEND=""
    CURRENT_CONTAINER_NAME=""
    CURRENT_BACKEND_LOG=""
    BACKEND_START_ERROR=""
}
trap cleanup_backend EXIT

start_backend() {
    local backend="$1"
    local run_dir="$2"
    local launcher
    launcher="$(backend_launcher "${backend}")"
    local stdout_log="${run_dir}/backend_stdout.log"
    CURRENT_BACKEND="${backend}"
    CURRENT_CONTAINER_NAME="$(backend_container_name "${backend}")"
    CURRENT_BACKEND_LOG="${stdout_log}"
    BACKEND_START_ERROR=""

    # Ensure stale container from a previous interrupted run is gone.
    stop_named_container "${CURRENT_CONTAINER_NAME}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        log "[dry-run] would start ${backend} via ${launcher} (container=${CURRENT_CONTAINER_NAME:-n/a})"
        return
    fi

    if [[ -n "${CURRENT_CONTAINER_NAME}" ]]; then
        if ! docker_is_available; then
            BACKEND_START_ERROR="Docker is required for backend '${backend}' but command '${DOCKER_CMD}' is not available."
            if [[ "${SKIP_UNAVAILABLE_BACKENDS}" == "1" ]]; then
                log "WARNING: ${BACKEND_START_ERROR} Skipping backend '${backend}'."
                return 3
            fi
            die "${BACKEND_START_ERROR}"
        fi
        if ! docker_can_access_daemon; then
            if try_enable_sudo_docker && docker_is_available && docker_can_access_daemon; then
                :
            else
            BACKEND_START_ERROR="Docker daemon is not accessible for backend '${backend}' via '${DOCKER_CMD}'. Check /var/run/docker.sock permissions or set DOCKER_CMD."
            if [[ "${SKIP_UNAVAILABLE_BACKENDS}" == "1" ]]; then
                log "WARNING: ${BACKEND_START_ERROR} Skipping backend '${backend}'."
                return 3
            fi
            die "${BACKEND_START_ERROR}"
            fi
        fi
    fi

    (
        cd "${SCRIPT_DIR}"
        GPU_ID="${GPU_ID}" MODEL_ID="${MODEL_ID}" CONTAINER_NAME="${CURRENT_CONTAINER_NAME}" DOCKER_CMD="${DOCKER_CMD}" "${launcher}"
    ) >"${stdout_log}" 2>&1 &
    BACKEND_PID=$!

    log "Started ${backend} launcher with PID ${BACKEND_PID}"
}

wait_for_backend_readiness() {
    local backend="$1"

    if [[ "${DRY_RUN}" == "1" ]]; then
        log "[dry-run] would wait for readiness of ${backend}"
        return
    fi

    local family
    family="$(backend_family "${backend}")"
    local host port
    local deadline=$((SECONDS + STARTUP_TIMEOUT))
    if [[ "${family}" == "pie" ]]; then
        mapfile -t pie_host_port < <(parse_host_port_from_uri "${PIE_SERVER_URI}")
        host="${pie_host_port[0]}"
        port="${pie_host_port[1]}"
    else
        host="${OPENAI_HOST}"
        port="${OPENAI_PORT}"
    fi

    while (( SECONDS < deadline )); do
        if [[ -n "${BACKEND_PID}" ]] && ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
            echo "[run_eval_matrix] ERROR: backend '${backend}' exited before becoming ready." >&2
            if [[ -n "${CURRENT_BACKEND_LOG}" && -f "${CURRENT_BACKEND_LOG}" ]]; then
                echo "[run_eval_matrix] Last backend log lines:" >&2
                tail -n 80 "${CURRENT_BACKEND_LOG}" >&2 || true
            fi
            return 1
        fi

        if [[ "${family}" == "pie" ]]; then
            if check_tcp_once "${host}" "${port}" && check_pie_engine_ready_once "${CURRENT_BACKEND_LOG}"; then
                return 0
            fi
        else
            if check_openai_http_once "${OPENAI_HOST}:${OPENAI_PORT}"; then
                return 0
            fi
        fi
        sleep 1
    done

    echo "[run_eval_matrix] ERROR: timed out waiting for backend '${backend}' readiness." >&2
    if [[ -n "${CURRENT_BACKEND_LOG}" && -f "${CURRENT_BACKEND_LOG}" ]]; then
        echo "[run_eval_matrix] Last backend log lines:" >&2
        tail -n 80 "${CURRENT_BACKEND_LOG}" >&2 || true
    fi
    return 1
}

run_script() {
    local backend="$1"
    local script="$2"
    local run_dir="$3"
    local family
    family="$(backend_family "${backend}")"
    local pie_client_src="${REPO_ROOT}/client/python/src"
    local python_bin="python3"
    local pie_venv_python="${REPO_ROOT}/pie/.venv/bin/python"

    # PIE scripts require pie_client + runtime deps (e.g., msgpack), which are already
    # present in the repo's pie virtualenv.
    if [[ "${family}" == "pie" ]] && [[ -x "${pie_venv_python}" ]]; then
        python_bin="${pie_venv_python}"
    fi

    local -a cmd=("${python_bin}" "${script}")
    if [[ "${family}" == "pie" ]]; then
        cmd+=("--server-uri" "${PIE_SERVER_URI}")

        local -a profile_extra=()
        while IFS= read -r line; do
            [[ -n "${line}" ]] && profile_extra+=("${line}")
        done < <(pie_profile_args_for_script "${script}")
        cmd+=("${profile_extra[@]}")

        if [[ -n "${PIE_SCRIPT_ARGS}" ]]; then
            local -a extra=()
            # shellcheck disable=SC2206
            extra=(${PIE_SCRIPT_ARGS})
            cmd+=("${extra[@]}")
        fi
    else
        cmd+=("--host" "${OPENAI_HOST}" "--port" "${OPENAI_PORT}" "--model-path" "${MODEL_ID}")

        local -a profile_extra=()
        while IFS= read -r line; do
            [[ -n "${line}" ]] && profile_extra+=("${line}")
        done < <(baseline_profile_args_for_script "${script}")
        cmd+=("${profile_extra[@]}")

        if [[ -n "${BASELINE_SCRIPT_ARGS}" ]]; then
            local -a extra=()
            # shellcheck disable=SC2206
            extra=(${BASELINE_SCRIPT_ARGS})
            cmd+=("${extra[@]}")
        fi
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        log "[dry-run] ${cmd[*]}"
        return 0
    fi

    local stdout_log="${run_dir}/script_stdout/${script%.py}.log"
    mkdir -p "$(dirname "${stdout_log}")"

    (
        cd "${SCRIPT_DIR}"
        if [[ -d "${pie_client_src}" ]]; then
            if [[ -n "${PYTHONPATH:-}" ]]; then
                export PYTHONPATH="${pie_client_src}:${PYTHONPATH}"
            else
                export PYTHONPATH="${pie_client_src}"
            fi
        fi
        if [[ "${family}" == "pie" ]] && [[ "${PIE_LOAD_PROFILE}" != "legacy" ]]; then
            export PIE_BENCH_INSTANCE_RETRIES="${PIE_INSTANCE_RETRIES_SAFE}"
        fi
        if [[ "${SCRIPT_TIMEOUT}" != "0" ]] && command -v timeout >/dev/null 2>&1; then
            timeout --foreground "${SCRIPT_TIMEOUT}s" "${cmd[@]}"
        else
            "${cmd[@]}"
        fi
    ) >"${stdout_log}" 2>&1
}

run_one_backend() {
    local backend="$1"
    local framework
    framework="$(backend_family "${backend}")"
    local mode
    mode="$(backend_mode "${backend}")"
    local version
    version="$(backend_version "${backend}")"
    local now
    now="$(date '+%Y%m%d-%H%M%S')"
    local git_sha
    git_sha="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "nogit")"
    local model_slug
    model_slug="$(sanitize_component "${MODEL_ID}")"

    local run_name="${now}__${framework}"
    if [[ -n "${version}" ]]; then
        run_name="${run_name}__v${version}"
    fi
    run_name="${run_name}__gpu${GPU_ID}__model-${model_slug}__git-${git_sha}"

    local run_dir="${RESULTS_ROOT}/${run_name}"
    local scripts_file="${run_dir}/scripts.txt"
    local statuses_file="${run_dir}/script_status.tsv"
    if [[ "${DRY_RUN}" != "1" ]]; then
        mkdir -p "${run_dir}/logs" "${run_dir}/script_stdout"
        echo -e "script\tstatus\texit_code" > "${statuses_file}"
    fi

    local -a scripts=()
    while IFS= read -r line; do
        [[ -n "${line}" ]] && scripts+=("${line}")
    done < <(collect_scripts_for_backend "${backend}")

    if [[ "${#scripts[@]}" -eq 0 ]]; then
        die "No benchmark scripts selected for backend=${backend}. Check --filter/BACKENDS."
    fi

    if [[ "${DRY_RUN}" != "1" ]]; then
        printf '%s\n' "${scripts[@]}" > "${scripts_file}"
    fi

    log "=================================================="
    log "Run directory: ${run_dir}"
    log "Backend: ${backend} (framework=${framework}, mode=${mode}, version=${version:-n/a})"
    log "Scripts: ${#scripts[@]}"
    log "GPU assignment for this backend: cuda:${GPU_ID}"
    if [[ "${SINGLE_INSTANCE}" == "1" ]]; then
        log "Single-instance mode: enabled (all script concurrency forced to 1)"
    fi
    if [[ "${framework}" == "pie" ]]; then
        log "PIE load profile: ${PIE_LOAD_PROFILE}"
        if [[ "${PIE_LOAD_PROFILE}" == "safe" ]]; then
            log "PIE instance retries (safe): ${PIE_INSTANCE_RETRIES_SAFE}"
        elif [[ "${PIE_LOAD_PROFILE}" == "safe-3.1-8b" ]]; then
            log "safe-3.1-8b profile: calibrated per-script request map enabled"
        fi
        if [[ "${PIE_RESTART_BETWEEN_SCRIPTS}" == "1" ]]; then
            log "PIE backend restart between scripts: enabled"
        else
            log "PIE backend restart between scripts: disabled"
        fi
    elif [[ "${PIE_LOAD_PROFILE}" == "safe-3.1-8b" ]]; then
        log "safe-3.1-8b profile: calibrated per-script request map enabled"
    fi
    if [[ "${SCRIPT_TIMEOUT}" != "0" ]]; then
        log "Per-script timeout: ${SCRIPT_TIMEOUT}s"
    fi
    log "=================================================="

    local failed_count=0
    local skipped_count=0

    # Sequential execution with explicit teardown before starting each backend.
    cleanup_backend
    set +e
    start_backend "${backend}" "${run_dir}"
    local start_rc=$?
    set -e
    if [[ ${start_rc} -eq 3 ]]; then
        if [[ "${DRY_RUN}" != "1" ]]; then
            local skip_script
            for skip_script in "${scripts[@]}"; do
                local out_file="${run_dir}/script_stdout/${skip_script%.py}.log"
                printf 'unsupported baseline variant: %s\n' "${BACKEND_START_ERROR}" > "${out_file}"
                echo -e "${skip_script}\tunsupported\t125" >> "${statuses_file}"
            done
        fi
        skipped_count=${#scripts[@]}

        if [[ "${DRY_RUN}" != "1" ]]; then
            write_manifest \
                "${run_dir}/manifest.json" \
                "${backend}" \
                "${framework}" \
                "${version}" \
                "${mode}" \
                "${run_name}" \
                "$(date -Iseconds)" \
                "${git_sha}" \
                "${scripts_file}" \
                "${statuses_file}"
            echo "${run_dir}" >> "${RESULTS_ROOT}/last_runs.txt"
            echo -e "${backend}\t${framework}\t${version}\t${mode}\t${run_dir}\t${#scripts[@]}\t${failed_count}\t${skipped_count}" >> "${SUMMARY_ROWS_FILE}"
        fi
        log "Completed ${backend}: failed=${failed_count}, unsupported=${skipped_count} (backend unavailable)"
        return 0
    fi

    if [[ ${start_rc} -ne 0 ]]; then
        die "Failed to start backend '${backend}' (exit code ${start_rc})."
    fi

    wait_for_backend_readiness "${backend}"
    log "Backend ready: ${backend}"

    if [[ "${DRY_RUN}" != "1" ]]; then
        export PIE_BENCH_LOG_DIR="${run_dir}/logs"
    fi

    local script
    local idx=0
    for script in "${scripts[@]}"; do
        if [[ "${framework}" == "pie" ]] && [[ "${PIE_RESTART_BETWEEN_SCRIPTS}" == "1" ]] && [[ ${idx} -gt 0 ]]; then
            log "Restarting PIE backend before ${script}"
            cleanup_backend
            set +e
            start_backend "${backend}" "${run_dir}"
            local restart_rc=$?
            set -e
            if [[ ${restart_rc} -ne 0 ]]; then
                die "Failed to restart backend '${backend}' before script '${script}' (exit code ${restart_rc})."
            fi
            wait_for_backend_readiness "${backend}"
            log "Backend ready after restart: ${backend}"
        fi

        log "Running ${script}"
        set +e
        run_script "${backend}" "${script}" "${run_dir}"
        local rc=$?
        set -e

        local status="ok"
        if [[ ${rc} -ne 0 ]]; then
            local out_file="${run_dir}/script_stdout/${script%.py}.log"
            if [[ ${rc} -eq 2 ]] && file_contains "unsupported baseline variant" "${out_file}"; then
                status="unsupported"
                skipped_count=$((skipped_count + 1))
            else
                status="failed"
                failed_count=$((failed_count + 1))
            fi
        fi

        if [[ "${DRY_RUN}" != "1" ]]; then
            echo -e "${script}\t${status}\t${rc}" >> "${statuses_file}"
        fi

        if [[ "${status}" == "failed" && "${STOP_ON_ERROR}" == "1" ]]; then
            log "Stopping early due to failure in ${script}"
            break
        fi
        idx=$((idx + 1))
    done

    if [[ "${DRY_RUN}" != "1" ]]; then
        unset PIE_BENCH_LOG_DIR
    fi
    cleanup_backend

    if [[ "${DRY_RUN}" != "1" ]]; then
        write_manifest \
            "${run_dir}/manifest.json" \
            "${backend}" \
            "${framework}" \
            "${version}" \
            "${mode}" \
            "${run_name}" \
            "$(date -Iseconds)" \
            "${git_sha}" \
            "${scripts_file}" \
            "${statuses_file}"
    fi

    log "Completed ${backend}: failed=${failed_count}, unsupported=${skipped_count}"
    if [[ "${DRY_RUN}" != "1" ]]; then
        echo "${run_dir}" >> "${RESULTS_ROOT}/last_runs.txt"
        echo -e "${backend}\t${framework}\t${version}\t${mode}\t${run_dir}\t${#scripts[@]}\t${failed_count}\t${skipped_count}" >> "${SUMMARY_ROWS_FILE}"
    fi
}

log "Results root: ${RESULTS_ROOT}"
log "Backends: ${BACKENDS}"
log "Model: ${MODEL_ID}"
log "GPU: ${GPU_ID}"
log "Backend execution mode: sequential (cleanup between backends enabled)"
log "Docker command: ${DOCKER_CMD}"
if [[ "${SINGLE_INSTANCE}" == "1" ]]; then
    log "Single-instance mode: enabled (all script concurrency forced to 1)"
fi
if [[ "${SKIP_UNAVAILABLE_BACKENDS}" == "1" ]]; then
    log "Unavailable backends: skip and mark unsupported"
else
    log "Unavailable backends: strict fail"
fi
log "PIE load profile: ${PIE_LOAD_PROFILE}"
if [[ "${PIE_LOAD_PROFILE}" == "safe" ]]; then
    log "PIE instance retries (safe): ${PIE_INSTANCE_RETRIES_SAFE}"
elif [[ "${PIE_LOAD_PROFILE}" == "safe-3.1-8b" ]]; then
    log "safe-3.1-8b profile: calibrated per-script request map enabled"
fi
if [[ "${PIE_RESTART_BETWEEN_SCRIPTS}" == "1" ]]; then
    log "PIE backend restart between scripts: enabled"
else
    log "PIE backend restart between scripts: disabled"
fi
if [[ "${SCRIPT_TIMEOUT}" != "0" ]]; then
    log "Per-script timeout: ${SCRIPT_TIMEOUT}s"
fi
log "PIE URI: ${PIE_SERVER_URI}"
log "Baseline endpoint: ${OPENAI_HOST}:${OPENAI_PORT}"
[[ -n "${BENCH_FILTER}" ]] && log "Filter: ${BENCH_FILTER}"

if [[ "${DRY_RUN}" != "1" ]]; then
    batch_slug="$(date '+%Y%m%d-%H%M%S')__gpu${GPU_ID}__model-$(sanitize_component "${MODEL_ID}")"
    SUMMARY_ROWS_FILE="$(mktemp "${RESULTS_ROOT}/.run_eval_matrix_rows.${batch_slug}.XXXXXX.tsv")"
    SUMMARY_PATH="${RESULTS_ROOT}/run_summary_${batch_slug}.md"
    LATEST_SUMMARY_PATH="${RESULTS_ROOT}/run_summary_latest.md"
    echo -e "backend\tframework\tversion\tmode\trun_dir\ttotal_scripts\tfailed\tunsupported" > "${SUMMARY_ROWS_FILE}"
fi

IFS=' ' read -r -a BACKEND_KEYS <<< "${BACKENDS}"
for backend in "${BACKEND_KEYS[@]}"; do
    run_one_backend "${backend}"
done

if [[ "${DRY_RUN}" != "1" ]]; then
    BATCH_ENDED_AT="$(date -Iseconds)"
    generate_batch_summary "${SUMMARY_ROWS_FILE}" "${SUMMARY_PATH}" "${BATCH_ENDED_AT}"
    cp "${SUMMARY_PATH}" "${LATEST_SUMMARY_PATH}"
    rm -f "${SUMMARY_ROWS_FILE}"
    log "Wrote summary: ${SUMMARY_PATH}"
    log "Updated latest summary: ${LATEST_SUMMARY_PATH}"
else
    log "Dry-run mode: summary file is not generated."
fi

log "All requested backend runs completed."
