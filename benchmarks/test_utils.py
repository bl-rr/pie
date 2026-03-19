from functools import partial
import time
import requests
import json
import os
from pathlib import Path

PIE_BENCH_GENERATED_TOKENS_ENV = "PIE_BENCH_GENERATED_TOKENS"
_BASELINE_GENERATED_TOKENS = 0


def _count_text_tokens(value) -> int:
    if value is None:
        return 0
    text = str(value).strip()
    if not text:
        return 0
    return len(text.split())


def _record_generated_tokens(count: int | None) -> int:
    global _BASELINE_GENERATED_TOKENS
    if count is None:
        return 0
    try:
        parsed = int(count)
    except (TypeError, ValueError):
        return 0
    if parsed <= 0:
        return 0
    _BASELINE_GENERATED_TOKENS += parsed
    return parsed


def _read_int_env(name: str) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return 0
    try:
        parsed = int(raw)
    except ValueError:
        return 0
    return parsed if parsed > 0 else 0


def _resolve_log_path(log_file_path: str) -> Path:
    path = Path(log_file_path)
    override_dir = os.environ.get("PIE_BENCH_LOG_DIR")
    if not override_dir or path.is_absolute():
        return path

    # Keep legacy script paths unchanged while allowing per-run log isolation.
    # Typical input is "./logs/<name>.json" and will map to
    # "${PIE_BENCH_LOG_DIR}/<name>.json".
    parts = list(path.parts)
    if parts and parts[0] == ".":
        parts = parts[1:]
    if parts and parts[0] == "logs":
        parts = parts[1:]
    if not parts:
        parts = [path.name]

    return Path(override_dir).expanduser().resolve() / Path(*parts)


def get_call_generate(backend, host, port, model_path, call_overhead_ms=0):
    if backend == "vllm":
        return partial(call_generate_vllm, url=f"{host}:{port}/v1/completions", model_path=model_path, call_overhead_ms=call_overhead_ms)
    elif backend == "sglang":
        # return partial(call_generate_srt_raw, url=f"{host}:{port}/generate")
        return partial(call_generate_vllm, url=f"{host}:{port}/v1/completions", model_path=model_path)

    else:
        raise ValueError(f"Invalid backend: {backend}")


def call_generate_vllm(
        prompt,
        temperature,
        max_tokens,
        stop=None,
        n=1,
        url=None,
        model_path=None,
        call_overhead_ms=0,
        return_usage=False,
):
    assert url is not None

    data = {
        "model": model_path,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
        "n": n,
    }
    res = requests.post(url, json=data)
    assert res.status_code == 200
    res_json = res.json()

    # Add call overhead
    time.sleep(call_overhead_ms / 1000)

    # print(res.json()['choices'][0]['text'])

    if n == 1:
        pred = res_json["choices"][0]["text"]
    else:
        pred = [x["text"] for x in res_json["choices"]]

    usage = res_json.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        if n == 1:
            completion_tokens = _count_text_tokens(pred)
        else:
            completion_tokens = sum(_count_text_tokens(text) for text in pred)
    completion_tokens = _record_generated_tokens(completion_tokens)

    if return_usage:
        return pred, completion_tokens

    return pred


def call_generate_outlines(
        prompt, temperature, max_tokens, stop=None, regex=None, n=1, url=None
):
    assert url is not None

    data = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
        "regex": regex,
        "n": n,
    }
    res = requests.post(url, json=data)
    assert res.status_code == 200
    if n == 1:
        pred = res.json()["text"][0][len(prompt):]
    else:
        pred = [x[len(prompt):] for x in res.json()["text"]]
    if n == 1:
        _record_generated_tokens(_count_text_tokens(pred))
    else:
        _record_generated_tokens(sum(_count_text_tokens(text) for text in pred))
    return pred


def call_generate_srt_raw(prompt, temperature, max_tokens, stop=None, url=None):
    assert url is not None

    data = {
        "text": prompt,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "stop": stop,
        },
    }
    res = requests.post(url, json=data)
    assert res.status_code == 200
    obj = res.json()
    pred = obj["text"]
    _record_generated_tokens(_count_text_tokens(pred))
    return pred


def append_log(log_file_path: str, data: dict):
    if not log_file_path.endswith('.json'):
        raise ValueError("The log file path must end with '.json'")

    path = _resolve_log_path(log_file_path)

    path.parent.mkdir(parents=True, exist_ok=True)

    logs = []

    if path.is_file() and path.stat().st_size > 0:
        try:
            with path.open('r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if isinstance(existing_data, list):
                    logs = existing_data
        except json.JSONDecodeError:
            print(f"Warning: '{log_file_path}' contains invalid JSON. It will be overwritten.")

    # Allow wrapper scripts to attach baseline metadata without changing
    # every benchmark script signature.
    baseline_backend = os.environ.get("PIE_BASELINE_BACKEND")
    baseline_version = os.environ.get("PIE_BASELINE_VERSION")
    baseline_mode = os.environ.get("PIE_BASELINE_MODE")
    if baseline_backend or baseline_version or baseline_mode:
        data = dict(data)
        data["baseline"] = {
            "backend": baseline_backend,
            "version": baseline_version,
            "mode": baseline_mode,
        }

    auto_token_metrics = False
    if "total_time" in data and "total_generated_tokens" not in data:
        total_generated_tokens = _BASELINE_GENERATED_TOKENS + _read_int_env(
            PIE_BENCH_GENERATED_TOKENS_ENV
        )
        if total_generated_tokens > 0:
            data["total_generated_tokens"] = total_generated_tokens
            data["per_token_latency_ms"] = (float(data["total_time"]) * 1000.0) / float(
                total_generated_tokens
            )
            auto_token_metrics = True

    if (
        "total_time" in data
        and "total_generated_tokens" in data
        and "per_token_latency_ms" not in data
        and float(data["total_generated_tokens"]) > 0
    ):
        data["per_token_latency_ms"] = (float(data["total_time"]) * 1000.0) / float(
            data["total_generated_tokens"]
        )
        auto_token_metrics = True

    if auto_token_metrics:
        print(f"Total Generated Tokens: {int(data['total_generated_tokens'])}")
        print(f"Per-Token Latency:      {float(data['per_token_latency_ms']):.4f} ms/token")

    logs.append(data)

    with path.open('w', encoding='utf-8') as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)
