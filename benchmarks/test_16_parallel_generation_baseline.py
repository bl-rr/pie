#!/usr/bin/env python3
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback for --help without deps
    def tqdm(iterable, *args, **kwargs):
        return iterable


from test_utils import append_log, get_call_generate


def one_request(call_generate, args):
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful, respectful and honest assistant.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Explain pulmonary embolism.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    _ = call_generate(
        prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stop=" <|eot_id|>",
        n=args.num_branches,
    )


def main(args):
    call_generate = get_call_generate(args.backend, args.host, args.port, args.model_path)

    print(f"Starting benchmark with {args.num_requests} total requests.")
    start_time = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.num_max_workers) as executor:
        list(
            tqdm(
                executor.map(lambda _: one_request(call_generate, args), range(args.num_requests)),
                total=args.num_requests,
            )
        )

    total_time = time.monotonic() - start_time
    throughput = args.num_requests / total_time if total_time > 0 else 0

    print("\n--- ✅ Benchmark Complete ---")
    print(f"Total Time Taken: {total_time * 1000:.2f} milliseconds")
    print(f"Throughput:       {throughput:.2f} requests/second")
    print("--------------------------")

    append_log(
        "./logs/test_16_parallel_generation_baseline.json",
        {
            "total_time": total_time,
            "throughput": throughput,
            "args": vars(args),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    server_group = parser.add_argument_group("Serving Engine Configuration")
    server_group.add_argument(
        "--backend", type=str, default="vllm", help="Backend to use: vllm or sglang."
    )
    server_group.add_argument(
        "--host", type=str, default="http://127.0.0.1", help="Host of the LLM server."
    )
    server_group.add_argument(
        "--port", type=int, default=8000, help="Port of the LLM server."
    )
    server_group.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name/path passed to OpenAI-compatible API.",
    )

    benchmark_group = parser.add_argument_group("Benchmark Configuration")
    benchmark_group.add_argument(
        "--num-requests", type=int, default=128, help="Total requests to send."
    )
    benchmark_group.add_argument(
        "--num-max-workers", type=int, default=128, help="Threadpool worker count."
    )
    benchmark_group.add_argument(
        "--num-branches",
        type=int,
        default=2,
        help="Number of parallel completions per request.",
    )
    benchmark_group.add_argument(
        "--max-tokens", type=int, default=128, help="Maximum output tokens per branch."
    )
    benchmark_group.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature."
    )

    args = parser.parse_args()
    main(args)
