#!/usr/bin/env python3
import argparse
import asyncio
import json

from pie_eval_common import add_auth_args, connect_client, ensure_inferlet_installed
from test_utils import append_log


async def launch_and_handle(client, inferlet: str, layer: str, idx: int):
    instance_args = [
        "--index",
        str(idx),
        "--layer",
        layer,
    ]
    await client.launch_instance(inferlet, arguments=instance_args)


async def main(args):
    from pie_client import Event

    client = await connect_client(args)
    try:
        inferlet = await ensure_inferlet_installed(client, "bench_execution_latency")

        print(f"Starting benchmark with {args.num_instances} total inferlet instances...")

        tasks = [
            launch_and_handle(client, inferlet, args.layer, idx)
            for idx in range(args.num_instances)
        ]
        await asyncio.gather(*tasks)

        # Wait briefly so measurements are stored before aggregation.
        await asyncio.sleep(0.5)

        aggregator = await client.launch_instance(
            inferlet,
            arguments=["--aggregate-size", str(args.num_instances)],
        )

        result_payload = "{}"
        while True:
            event, message = await aggregator.recv()
            if event == Event.Completed:
                result_payload = message
                break
        result = json.loads(result_payload)

        print("\n--- ✅ Benchmark Complete ---")
        print(f"mean latency:   {result['mean']:.2f} μs")
        print(f"median latency: {result['median']:.2f} μs")
        print(f"stdev latency:  {result['std_dev']:.2f} μs")

        append_log(
            "./logs/microbench_execution_latency.json",
            {
                "mean_latency": result["mean"],
                "median_latency": result["median"],
                "stdev_latency": result["std_dev"],
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

    wasm_args_group = parser.add_argument_group("WASM Program Arguments")
    wasm_args_group.add_argument(
        "--layer", type=str, default="inference", help="control or inference"
    )

    add_auth_args(parser)

    args = parser.parse_args()
    asyncio.run(main(args))
