#!/usr/bin/env python3
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback for --help without deps
    def tqdm(iterable, *args, **kwargs):
        return iterable

from test_6_prefix_tree_baseline import (
    ASSISTANT_HEADER,
    ASSISTANT_SUFFIX,
    BOS_TOKEN,
    EOT_ID,
    PROMPT_L1_PHOTO,
    PROMPT_L1_RESP,
    PROMPT_L2_PHOTO_ELI5,
    PROMPT_L2_PHOTO_HS,
    PROMPT_L2_RESP_LOC,
    PROMPT_L2_RESP_PROD,
    SYSTEM_HEADER,
    SYSTEM_PROMPT,
    USER_HEADER,
    get_prefix_tree_prompts,
)
from test_utils import append_log, get_call_generate


def _send_one(prompt: str, call_generate: callable, max_tokens: int, temperature: float):
    started = time.monotonic()
    completion, completion_tokens = call_generate(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=ASSISTANT_SUFFIX,
        return_usage=True,
    )
    elapsed = time.monotonic() - started
    if completion_tokens is None:
        completion_tokens = len(str(completion).split())
    return elapsed, int(completion_tokens), completion


def _run_phase(
    prompts: list[str],
    call_generate: callable,
    *,
    max_tokens: int,
    temperature: float,
    max_workers: int,
    show_progress: bool,
    verbose: bool,
) -> tuple[float, int, int]:
    if not prompts:
        return 0.0, 0, 0

    phase_start = time.monotonic()
    generated_tokens = 0

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(prompts)))) as executor:
        futures = [
            executor.submit(_send_one, prompt, call_generate, max_tokens, temperature)
            for prompt in prompts
        ]
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(prompts))
        for future in iterator:
            _, completion_tokens, completion = future.result()
            generated_tokens += completion_tokens
            if verbose:
                print(f"Response: {completion}")

    phase_time = time.monotonic() - phase_start
    return phase_time, generated_tokens, len(prompts)


def build_staged_prefix_prompts() -> tuple[list[str], list[str], list[str]]:
    base_system = f"{BOS_TOKEN}{SYSTEM_HEADER}{SYSTEM_PROMPT}{EOT_ID}"

    # Level 1 (2 nodes)
    l1_photo = f"{base_system}{USER_HEADER}{PROMPT_L1_PHOTO}{EOT_ID}"
    l1_resp = f"{base_system}{USER_HEADER}{PROMPT_L1_RESP}{EOT_ID}"

    # Level 2 (4 nodes)
    l2_photo_eli5 = f"{l1_photo}{USER_HEADER}{PROMPT_L2_PHOTO_ELI5}{EOT_ID}"
    l2_photo_hs = f"{l1_photo}{USER_HEADER}{PROMPT_L2_PHOTO_HS}{EOT_ID}"
    l2_resp_loc = f"{l1_resp}{USER_HEADER}{PROMPT_L2_RESP_LOC}{EOT_ID}"
    l2_resp_prod = f"{l1_resp}{USER_HEADER}{PROMPT_L2_RESP_PROD}{EOT_ID}"

    # These are warmup-only staged prefixes.
    stage_l0 = [base_system]
    stage_l1 = [l1_photo, l1_resp]
    stage_l2 = [l2_photo_eli5, l2_photo_hs, l2_resp_loc, l2_resp_prod]
    return stage_l0, stage_l1, stage_l2


def main(args: argparse.Namespace):
    if args.backend != "vllm":
        raise ValueError("This staged benchmark is only supported for backend=vllm.")

    call_generate = get_call_generate(args.backend, args.host, args.port, args.model_path)
    stage_l0, stage_l1, stage_l2 = build_staged_prefix_prompts()
    leaf_prompts = get_prefix_tree_prompts()

    # Sanity check: the 8 leaves must end at assistant prompt.
    for prompt in leaf_prompts:
        if not prompt.endswith(ASSISTANT_HEADER):
            raise ValueError("Leaf prompt does not end with assistant header.")

    total_leaf_prompts = leaf_prompts * args.num_requests
    total_leaf_calls = len(total_leaf_prompts)

    print("Running staged prefix warmup: 1 -> 2 -> 4, then benchmarking 8-leaf requests")
    print(
        f"Dummy warmup tokens are excluded from token accounting; "
        f"timing includes warmup + leaves."
    )

    total_start = time.monotonic()

    print("Stage L0: warming 1 shared root prefix...")
    l0_time, l0_tokens, _ = _run_phase(
        stage_l0,
        call_generate,
        max_tokens=args.warmup_max_tokens,
        temperature=args.warmup_temperature,
        max_workers=args.num_max_workers,
        show_progress=False,
        verbose=args.verbose,
    )

    print("Stage L1: warming 2 child prefixes...")
    l1_time, l1_tokens, _ = _run_phase(
        stage_l1,
        call_generate,
        max_tokens=args.warmup_max_tokens,
        temperature=args.warmup_temperature,
        max_workers=args.num_max_workers,
        show_progress=False,
        verbose=args.verbose,
    )

    print("Stage L2: warming 4 child prefixes...")
    l2_time, l2_tokens, _ = _run_phase(
        stage_l2,
        call_generate,
        max_tokens=args.warmup_max_tokens,
        temperature=args.warmup_temperature,
        max_workers=args.num_max_workers,
        show_progress=False,
        verbose=args.verbose,
    )

    print(f"Stage L3: benchmarking {total_leaf_calls} leaf requests...")
    leaf_time, leaf_generated_tokens, _ = _run_phase(
        total_leaf_prompts,
        call_generate,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_workers=args.num_max_workers,
        show_progress=True,
        verbose=args.verbose,
    )

    total_time = time.monotonic() - total_start
    dummy_generated_tokens = l0_tokens + l1_tokens + l2_tokens

    # Throughput defined in "request groups / s" to stay comparable with existing baseline script.
    throughput = args.num_requests / total_time if total_time > 0 else 0.0
    calls_throughput = total_leaf_calls / total_time if total_time > 0 else 0.0
    per_token_latency_ms = (
        (total_time * 1000.0) / leaf_generated_tokens if leaf_generated_tokens > 0 else 0.0
    )

    print("\n--- ✅ Benchmark Complete ---")
    print(f"Total Time Taken: {total_time * 1000:.2f} milliseconds")
    print(f"Throughput:       {throughput:.2f} requests/second")
    print(f"Calls Throughput: {calls_throughput:.2f} calls/second")
    print(f"Total Generated Tokens: {leaf_generated_tokens}")
    print(f"Per-Token Latency:      {per_token_latency_ms:.4f} ms/token")
    print(
        f"Dummy Warmup: L0={l0_time * 1000:.2f}ms, "
        f"L1={l1_time * 1000:.2f}ms, L2={l2_time * 1000:.2f}ms, "
        f"dummy_tokens_excluded={dummy_generated_tokens}"
    )
    print(f"Leaf Phase Time:  {leaf_time * 1000:.2f} milliseconds")
    print("--------------------------")

    append_log(
        "./logs/test_6_prefix_tree_vllm_staged.json",
        {
            "total_time": total_time,
            "throughput": throughput,
            "calls_throughput": calls_throughput,
            # IMPORTANT: tokens from dummy warmup phases are intentionally excluded.
            "total_generated_tokens": leaf_generated_tokens,
            "per_token_latency_ms": per_token_latency_ms,
            "warmup_l0_time": l0_time,
            "warmup_l1_time": l1_time,
            "warmup_l2_time": l2_time,
            "warmup_dummy_generated_tokens": dummy_generated_tokens,
            "leaf_phase_time": leaf_time,
            "args": vars(args),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    server_group = parser.add_argument_group("Serving Engine Configuration")
    server_group.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm"],
        help="Backend to use. This script is vLLM-only.",
    )
    server_group.add_argument(
        "--host", type=str, default="http://127.0.0.1", help="The host of the LLM server."
    )
    server_group.add_argument(
        "--port", type=int, default=8000, help="The port of the LLM server."
    )
    server_group.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="The path or name of the model to use.",
    )

    benchmark_group = parser.add_argument_group("Benchmark Configuration")
    benchmark_group.add_argument(
        "--num-requests",
        type=int,
        default=64,
        help="Number of request groups. Each group contains 8 leaf requests.",
    )
    benchmark_group.add_argument(
        "--num-max-workers",
        type=int,
        default=64,
        help="Maximum number of threadpool workers to use.",
    )
    benchmark_group.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Maximum number of tokens to generate per leaf request.",
    )
    benchmark_group.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for leaf requests.",
    )

    warmup_group = parser.add_argument_group("Warmup Configuration")
    warmup_group.add_argument(
        "--warmup-max-tokens",
        type=int,
        default=1,
        help="Max tokens to generate for each dummy warmup request.",
    )
    warmup_group.add_argument(
        "--warmup-temperature",
        type=float,
        default=0.0,
        help="Temperature for dummy warmup requests.",
    )

    parser.add_argument("--verbose", action="store_true", help="Print each model response.")
    args = parser.parse_args()
    main(args)
