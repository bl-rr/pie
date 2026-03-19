import argparse
import asyncio
import re
import time

from pie_eval_common import add_auth_args, connect_client, ensure_inferlet_installed, run_simple_instances
from test_utils import append_log

TOKEN_COUNT_RE = re.compile(r"Total\s+Generated\s+Tokens:\s*([0-9]+)", re.IGNORECASE)


def _extract_total_generated_tokens(instance_outputs: list[str]) -> int:
    total = 0
    for output in instance_outputs:
        match = TOKEN_COUNT_RE.search(output or "")
        if match:
            total += int(match.group(1))
    return total


async def main(args):
    client = await connect_client(args)
    try:
        inferlet = await ensure_inferlet_installed(client, args.program_name)

        print(f"Starting benchmark with {args.num_instances} total inferlet instances...")
        start_time = time.monotonic()

        def _build_instance_args(index: int):
            return [
                "--num-tokens", str(args.max_tokens),
            ]

        instance_outputs = await run_simple_instances(
            client,
            inferlet,
            args.num_instances,
            _build_instance_args,
            verbose=args.verbose,
            collect_outputs=True,
        )
        if instance_outputs is None:
            instance_outputs = []

        total_time = time.monotonic() - start_time
        throughput = args.num_instances / total_time if total_time > 0 else 0
        total_generated_tokens = _extract_total_generated_tokens(instance_outputs)
        if total_generated_tokens == 0:
            # Fallback when inferlet output does not expose token accounting.
            total_generated_tokens = args.num_instances * 8 * args.max_tokens
        per_token_latency_ms = (
            (total_time * 1000.0) / total_generated_tokens if total_generated_tokens > 0 else 0.0
        )

        print("\n--- ✅ Benchmark Complete ---")
        print(f"Total Time Taken:       {total_time * 1000:.2f} milliseconds")
        print(f"Throughput:             {throughput:.2f} requests/second")
        print(f"Total Generated Tokens: {total_generated_tokens}")
        print(f"Per-Token Latency:      {per_token_latency_ms:.4f} ms/token")
        print("--------------------------")

        append_log('./logs/test_6_prefix_tree_pie.json', {
            'total_time': total_time,
            'throughput': throughput,
            'total_generated_tokens': total_generated_tokens,
            'per_token_latency_ms': per_token_latency_ms,
            'args': vars(args),
        })
    finally:
        await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    server_group = parser.add_argument_group('Serving Engine Configuration')
    server_group.add_argument("--server-uri", type=str, default="ws://127.0.0.1:8080", help="PIE server URI")

    benchmark_group = parser.add_argument_group('Benchmark Configuration')
    benchmark_group.add_argument("--program-name", type=str, default="prefix_tree", help="Name of the WASM program file (without .wasm extension).")
    benchmark_group.add_argument("--num-instances", type=int, default=64, help="Total number of concurrent instances to launch.")
    benchmark_group.add_argument("--verbose", action="store_true", help="Enable verbose output for debugging.")

    wasm_args_group = parser.add_argument_group('Inferlet Arguments')
    wasm_args_group.add_argument("--prompt", type=str, default='Tell me about the number', help="Base prompt to send to the WASM program.")
    wasm_args_group.add_argument("--max-tokens", type=int, default=32, help="Argument for WASM program: max tokens to generate.")
    add_auth_args(parser)

    args = parser.parse_args()
    asyncio.run(main(args))
