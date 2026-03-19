#!/usr/bin/env python3
import argparse
import asyncio
import statistics
import time

from pie_eval_common import add_auth_args, connect_client, ensure_inferlet_installed
from test_utils import append_log


async def launch_and_measure(client, inferlet: str, idx: int) -> float:
    instance_args = ["--message", f"{idx}-th instance launched!!"]
    start_time_ns = time.monotonic_ns()
    await client.launch_instance(inferlet, arguments=instance_args)
    elapsed_time_ns = time.monotonic_ns() - start_time_ns
    return elapsed_time_ns / 1000.0


async def main(args):
    client = await connect_client(args)
    try:
        inferlet = await ensure_inferlet_installed(client, "bench_spawn_time")

        print(f"Starting benchmark with {args.num_instances} total inferlet instances...")

        tasks = [launch_and_measure(client, inferlet, idx) for idx in range(args.num_instances)]
        results_in_us = await asyncio.gather(*tasks)

        mean_latency = statistics.mean(results_in_us)
        median_latency = statistics.median(results_in_us)
        stdev_latency = statistics.stdev(results_in_us) if len(results_in_us) > 1 else 0.0

        print("\n--- ✅ Benchmark Complete ---")
        print(f"mean latency:   {mean_latency:.2f} μs")
        print(f"median latency: {median_latency:.2f} μs")
        print(f"stdev latency:  {stdev_latency:.2f} μs")

        append_log(
            "./logs/microbench_spawn_time.json",
            {
                "mean_latency": mean_latency,
                "median_latency": median_latency,
                "stdev_latency": stdev_latency,
                "args": vars(args),
            },
        )
    finally:
        await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    server_group = parser.add_argument_group("Server & Program Configuration")
    server_group.add_argument(
        "--server-uri",
        type=str,
        default="ws://127.0.0.1:8080",
        help="WebSocket URI for the Pie server.",
    )

    benchmark_group = parser.add_argument_group("Benchmark Configuration")
    benchmark_group.add_argument(
        "--num-instances",
        type=int,
        default=1000,
        help="Total number of concurrent instances to launch.",
    )

    add_auth_args(parser)

    args = parser.parse_args()
    asyncio.run(main(args))
