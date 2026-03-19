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
                "--tokens-between-calls", str(args.tokens_between_calls),
                "--function-call-delay", str(args.function_call_delay),
                *( ["--use-prefix-cache"] if args.use_prefix_cache else [] ),
                *( ["--drop-tool-cache"] if args.drop_tool_cache else [] ),
                *( ["--concurrent-calls"] if args.concurrent_calls else [] ),
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

        append_log('./logs/test_4_agent_case_study_pie.json', {
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
    benchmark_group.add_argument("--program-name", type=str, default="agent_react_bench", help="Name of the WASM program file (without .wasm extension).")
    benchmark_group.add_argument("--num-instances", type=int, default=128, help="Total number of concurrent instances to launch.")
    benchmark_group.add_argument("--verbose", action="store_true", help="Enable verbose output for debugging.")

    wasm_args_group = parser.add_argument_group('Inferlet Arguments')
    wasm_args_group.add_argument("-t", "--tokens-between-calls", type=int, default=32, help="Argument for WASM: Max tokens for each Thought/Action step.")
    wasm_args_group.add_argument("-d", "--function-call-delay", type=int, default=100, help="Argument for WASM: Simulated delay in milliseconds for tool execution.")
    wasm_args_group.add_argument("--use-prefix-cache", action="store_true", help="Argument for WASM: Enable caching of the initial system prompt.")
    wasm_args_group.add_argument("--drop-tool-cache", action="store_true", help="Argument for WASM: Drop the first tool's KV cache after use.")
    wasm_args_group.add_argument("--concurrent-calls", action="store_true", help="Argument for WASM: Simulate concurrent execution of function calls.")
    add_auth_args(parser)

    args = parser.parse_args()
    asyncio.run(main(args))
