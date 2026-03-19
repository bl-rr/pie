#!/usr/bin/env python3
import argparse
import asyncio
import random
import time

from pie_eval_common import add_auth_args, connect_client, ensure_inferlet_installed
from test_utils import append_log

AGENT_ROLES = [
    "idea_generator",
    "plot_developer",
    "character_creator",
    "dialogue_writer",
]


async def main(args):
    from pie_client import Event

    client = await connect_client(args)
    try:
        inferlet = await ensure_inferlet_installed(client, args.program_name)

        print(
            f"Starting benchmark with {args.num_pipelines * len(AGENT_ROLES)} "
            "total inferlet instances..."
        )
        start_time = time.monotonic()

        pipelines = []
        for pipeline_idx in range(args.num_pipelines):
            futures = {}
            for role in AGENT_ROLES:
                instance_args = [
                    role,
                    "--group-id",
                    str(pipeline_idx),
                    "--tokens-per-step",
                    str(args.tokens_per_step),
                ]
                futures[role] = client.launch_instance(inferlet, arguments=instance_args)

            resolved = await asyncio.gather(*futures.values())
            pipelines.append(dict(zip(futures.keys(), resolved)))

        async def send_initial_prompt(pipeline_idx: int):
            first_agent = pipelines[pipeline_idx]["idea_generator"]
            prompt = random.choice(
                [
                    "a haunted spaceship",
                    "a detective who can talk to ghosts",
                    "a romance in a city that floats",
                ]
            )
            await first_agent.send(prompt)

        async def await_final_story(pipeline_idx: int):
            last_agent = pipelines[pipeline_idx]["dialogue_writer"]
            while True:
                event, message = await last_agent.recv()
                if event == Event.Completed:
                    return message
                if event in (
                    Event.Aborted,
                    Event.Exception,
                    Event.ServerError,
                    Event.OutOfResources,
                ):
                    raise RuntimeError(
                        f"Pipeline {pipeline_idx} failed with {event.name}: {message}"
                    )

        await asyncio.gather(*[send_initial_prompt(i) for i in range(args.num_pipelines)])
        final_stories = await asyncio.gather(*[await_final_story(i) for i in range(args.num_pipelines)])

        total_time = time.monotonic() - start_time
        throughput = args.num_pipelines / total_time if total_time > 0 else 0
        total_generated_tokens = sum(len((story or "").split()) for story in final_stories)
        per_token_latency_ms = (
            (total_time * 1000.0) / total_generated_tokens if total_generated_tokens > 0 else 0.0
        )

        print("\n--- ✅ Benchmark Complete ---")
        print(f"Total Time Taken: {total_time * 1000:.2f} milliseconds")
        print(f"Throughput:       {throughput:.2f} requests/second")
        print(f"Total Generated Tokens: {total_generated_tokens}")
        print(f"Per-Token Latency:      {per_token_latency_ms:.4f} ms/token")
        print("--------------------------")

        append_log(
            "./logs/test_3_agent_swarm_pie.json",
            {
                "total_time": total_time,
                "throughput": throughput,
                "total_generated_tokens": total_generated_tokens,
                "per_token_latency_ms": per_token_latency_ms,
                "args": vars(args),
            },
        )
    finally:
        await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    server_group = parser.add_argument_group("Serving Engine Configuration")
    server_group.add_argument(
        "--server-uri", type=str, default="ws://127.0.0.1:8080", help="PIE server URI"
    )

    benchmark_group = parser.add_argument_group("Benchmark Configuration")
    benchmark_group.add_argument(
        "--program-name",
        type=str,
        default="agent_swarm",
        help="Name of the WASM program file (without .wasm extension).",
    )
    benchmark_group.add_argument(
        "--num-pipelines",
        type=int,
        default=32,
        help="Total number of concurrent story-writing pipelines to run.",
    )

    wasm_args_group = parser.add_argument_group("Inferlet Arguments")
    wasm_args_group.add_argument(
        "--tokens-per-step",
        type=int,
        default=96,
        help="Argument for WASM program: max tokens each agent generates.",
    )

    add_auth_args(parser)

    args = parser.parse_args()
    asyncio.run(main(args))
