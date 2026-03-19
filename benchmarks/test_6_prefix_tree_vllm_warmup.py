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
    ASSISTANT_SUFFIX,
    BOS_TOKEN,
    EOT_ID,
    SYSTEM_HEADER,
    SYSTEM_PROMPT,
    get_prefix_tree_prompts,
)
from test_utils import append_log, get_call_generate


def get_shared_prefix_prompt() -> str:
    # This is the deepest prefix shared by all 8 leaf prompts.
    return f"{BOS_TOKEN}{SYSTEM_HEADER}{SYSTEM_PROMPT}{EOT_ID}"


def send_request(prompt: str, call_generate: callable, args: argparse.Namespace):
    start_time = time.monotonic()
    completion, completion_tokens = call_generate(
        prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stop=ASSISTANT_SUFFIX,
        return_usage=True,
    )
    latency = time.monotonic() - start_time
    if completion_tokens is None:
        completion_tokens = len(str(completion).split())
    if args.verbose:
        print(f"Response: {completion}")
    return latency, int(completion_tokens)


def warmup_shared_prefix(
    shared_prefix: str, call_generate: callable, args: argparse.Namespace
) -> tuple[float, int]:
    start_time = time.monotonic()
    _, completion_tokens = call_generate(
        shared_prefix,
        max_tokens=args.warmup_max_tokens,
        temperature=args.warmup_temperature,
        stop=ASSISTANT_SUFFIX,
        return_usage=True,
    )
    latency = time.monotonic() - start_time
    if completion_tokens is None:
        completion_tokens = args.warmup_max_tokens
    return latency, int(completion_tokens)


def main(args: argparse.Namespace):
    if args.backend != "vllm":
        raise ValueError("This warmup benchmark is only supported for backend=vllm.")

    call_generate = get_call_generate(args.backend, args.host, args.port, args.model_path)
    shared_prefix = get_shared_prefix_prompt()
    branch_prompts = get_prefix_tree_prompts()

    for prompt in branch_prompts:
        if not prompt.startswith(shared_prefix):
            raise ValueError("Branch prompt does not start with shared prefix.")

    print("Priming shared prefix cache with one warmup request...")
    warmup_time, warmup_generated_tokens = warmup_shared_prefix(
        shared_prefix, call_generate, args
    )
    print(
        f"Warmup complete in {warmup_time * 1000:.2f} ms; "
        f"generated tokens: {warmup_generated_tokens}"
    )

    prompts = []
    for _ in range(args.num_requests):
        prompts.extend(branch_prompts)

    total_calls = len(prompts)
    print(f"Starting benchmark with {args.num_requests} request groups.")
    print(f"Total branch API calls launched: {total_calls}")

    request_latencies = []
    total_generated_tokens = 0
    start_time = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.num_max_workers) as executor:
        futures = [executor.submit(send_request, p, call_generate, args) for p in prompts]
        for future in tqdm(as_completed(futures), total=total_calls):
            latency, completion_tokens = future.result()
            request_latencies.append(latency)
            total_generated_tokens += completion_tokens

    total_time = time.monotonic() - start_time
    throughput = args.num_requests / total_time if total_time > 0 else 0.0
    calls_throughput = total_calls / total_time if total_time > 0 else 0.0
    per_token_latency_ms = (
        (total_time * 1000.0) / total_generated_tokens if total_generated_tokens > 0 else 0.0
    )
    mean_request_latency_ms = (
        (sum(request_latencies) * 1000.0) / len(request_latencies)
        if request_latencies
        else 0.0
    )

    print("\n--- ✅ Benchmark Complete ---")
    print(f"Total Time Taken: {total_time * 1000:.2f} milliseconds")
    print(f"Throughput:       {throughput:.2f} requests/second")
    print(f"Calls Throughput: {calls_throughput:.2f} calls/second")
    print(f"Total Generated Tokens: {total_generated_tokens}")
    print(f"Per-Token Latency:      {per_token_latency_ms:.4f} ms/token")
    print(f"Mean Request Latency:   {mean_request_latency_ms:.2f} ms")
    print("--------------------------")

    append_log(
        "./logs/test_6_prefix_tree_vllm_warmup.json",
        {
            "warmup_time": warmup_time,
            "warmup_generated_tokens": warmup_generated_tokens,
            "total_time": total_time,
            "throughput": throughput,
            "calls_throughput": calls_throughput,
            "total_generated_tokens": total_generated_tokens,
            "per_token_latency_ms": per_token_latency_ms,
            "mean_request_latency_ms": mean_request_latency_ms,
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
        help="Number of request groups. Each group contains 8 branched requests.",
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
        help="Maximum number of tokens to generate per branched request.",
    )
    benchmark_group.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for branched requests.",
    )

    warmup_group = parser.add_argument_group("Warmup Configuration")
    warmup_group.add_argument(
        "--warmup-max-tokens",
        type=int,
        default=1,
        help="Max tokens to generate during the single warmup request.",
    )
    warmup_group.add_argument(
        "--warmup-temperature",
        type=float,
        default=0.0,
        help="Temperature for the warmup request.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print each model response.")

    args = parser.parse_args()
    main(args)
