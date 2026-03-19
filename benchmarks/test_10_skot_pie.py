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
                "--question", args.question,
                "--num-points", str(args.num_points),
                "--plan-tokens", str(args.plan_tokens),
                "--elab-tokens", str(args.elab_tokens),
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

        append_log('./logs/test_10_skot_pie.json', {
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
    benchmark_group.add_argument("--program-name", type=str, default="skeleton_of_thought", help="Name of the WASM program file (without .wasm extension).")
    benchmark_group.add_argument("--num-instances", type=int, default=128, help="Total number of concurrent instances to launch.")
    benchmark_group.add_argument("--verbose", action="store_true", help="Enable verbose output for debugging.")

    wasm_args_group = parser.add_argument_group('Inferlet Arguments')
    wasm_args_group.add_argument("--question", type=str, default='What is the significance of open source software?', help="Base question to send to the WASM program.")
    wasm_args_group.add_argument("--num-points", type=int, default=3, help="Argument for WASM: max key points to generate in the plan.")
    wasm_args_group.add_argument("--plan-tokens", type=int, default=80, help="Argument for WASM: max tokens for the planning step.")
    wasm_args_group.add_argument("--elab-tokens", type=int, default=80, help="Argument for WASM: max tokens for each elaboration step.")
    add_auth_args(parser)

    args = parser.parse_args()
    asyncio.run(main(args))
