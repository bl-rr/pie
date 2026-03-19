import argparse
import asyncio
import time

from pie_eval_common import add_auth_args, connect_client, ensure_inferlet_installed, run_simple_instances
from test_utils import append_log


async def main(args):
    client = await connect_client(args)
    try:
        inferlet = await ensure_inferlet_installed(client, args.program_name)

        print(f"Starting benchmark with {args.num_instances} total inferlet instances...")
        start_time = time.monotonic()

        def _build_instance_args(index: int):
            return [
                "--prompt", "any string",
                "--max-tokens", str(args.max_tokens),
            ]

        await run_simple_instances(
            client,
            inferlet,
            args.num_instances,
            _build_instance_args,
            verbose=args.verbose,
        )

        total_time = time.monotonic() - start_time
        throughput = args.num_instances / total_time if total_time > 0 else 0

        print("\n--- ✅ Benchmark Complete ---")
        print(f"Total Time Taken:       {total_time * 1000:.2f} milliseconds")
        print(f"Throughput:             {throughput:.2f} requests/second")
        print("--------------------------")

        append_log('./logs/test_11_cache_pie.json', {
            'total_time': total_time,
            'throughput': throughput,
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
    benchmark_group.add_argument("--program-name", type=str, default="prefix_caching", help="Name of the WASM program file (without .wasm extension).")
    benchmark_group.add_argument("--num-instances", type=int, default=128, help="Total number of concurrent instances to launch.")
    benchmark_group.add_argument("--verbose", action="store_true", help="Enable verbose output for debugging.")

    wasm_args_group = parser.add_argument_group('Inferlet Arguments')
    wasm_args_group.add_argument("--prompt", type=str, default='Tell me about the number', help="Base prompt to send to the WASM program.")
    wasm_args_group.add_argument("--max-tokens", type=int, default=64, help="Argument for WASM program: max tokens to generate.")
    add_auth_args(parser)

    args = parser.parse_args()
    asyncio.run(main(args))
