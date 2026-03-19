from __future__ import annotations

import argparse
import asyncio
import os
import random
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pie_client import PieClient
    from pie_client.crypto import ParsedPrivateKey
else:
    PieClient = Any
    ParsedPrivateKey = Any


@dataclass(frozen=True)
class ProgramSpec:
    program_name: str
    crate_dir: Path


# Legacy benchmark program names mapped to current inferlet crate locations.
PROGRAM_SPECS: dict[str, ProgramSpec] = {
    "agent_react": ProgramSpec("agent_react", Path("sdk/examples/agent-react")),
    "agent_codeact": ProgramSpec("agent_codeact", Path("sdk/examples/agent-codeact")),
    "agent_swarm": ProgramSpec("agent_swarm", Path("sdk/examples/agent-swarm")),
    "agent_react_bench": ProgramSpec(
        "agent_react_bench", Path("benchmarks/inferlets/agent-react-bench")
    ),
    "text_completion": ProgramSpec("text_completion", Path("std/text-completion")),
    "prefix_tree": ProgramSpec("prefix_tree", Path("sdk/examples/prefix-tree")),
    "tree_of_thought": ProgramSpec(
        "tree_of_thought", Path("sdk/examples/tree-of-thought")
    ),
    "recursion_of_thought": ProgramSpec(
        "recursion_of_thought", Path("sdk/examples/recursion-of-thought")
    ),
    "graph_of_thought": ProgramSpec(
        "graph_of_thought", Path("sdk/examples/graph-of-thought")
    ),
    "skeleton_of_thought": ProgramSpec(
        "skeleton_of_thought", Path("sdk/examples/skeleton-of-thought")
    ),
    "prefix_caching": ProgramSpec("prefix_caching", Path("sdk/examples/prefix-caching")),
    "constrained_decoding": ProgramSpec(
        "constrained_decoding", Path("sdk/examples/constrained-decoding")
    ),
    "speculative_decoding": ProgramSpec(
        "speculative_decoding", Path("benchmarks/inferlets/speculative-decoding")
    ),
    "beam_search": ProgramSpec("beam_search", Path("std/beam-search")),
    "attention_sink": ProgramSpec("attention_sink", Path("sdk/examples/attention-sink")),
    "parallel_generation": ProgramSpec(
        "parallel_generation", Path("sdk/examples/parallel-generation")
    ),
    "bench_spawn_time": ProgramSpec(
        "bench_spawn_time", Path("benchmarks/inferlets/bench-spawn-time")
    ),
    "bench_execution_latency": ProgramSpec(
        "bench_execution_latency", Path("benchmarks/inferlets/bench-execution-latency")
    ),
}

PIE_BENCH_GENERATED_TOKENS_ENV = "PIE_BENCH_GENERATED_TOKENS"
_TOTAL_GENERATED_TOKENS_RE = re.compile(
    r"Total\s+Generated\s+Tokens:\s*([0-9]+)", re.IGNORECASE
)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _estimate_generated_tokens_from_message(message: str) -> int:
    if not isinstance(message, str):
        return 0
    match = _TOTAL_GENERATED_TOKENS_RE.search(message)
    if match:
        return int(match.group(1))
    text = message.strip()
    if not text:
        return 0
    return len(text.split())


def _accumulate_generated_tokens_in_env(delta: int) -> None:
    if delta <= 0:
        return
    raw = os.environ.get(PIE_BENCH_GENERATED_TOKENS_ENV, "").strip()
    try:
        previous = int(raw) if raw else 0
    except ValueError:
        previous = 0
    os.environ[PIE_BENCH_GENERATED_TOKENS_ENV] = str(max(0, previous) + delta)


def resolve_program(program_name: str) -> tuple[Path, Path, str]:
    spec = PROGRAM_SPECS.get(program_name)
    if spec is None:
        raise ValueError(f"Unknown program name: {program_name}")

    crate_dir = (repo_root() / spec.crate_dir).resolve()
    manifest_path = crate_dir / "Pie.toml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    package = manifest.get("package", {})
    name = package.get("name")
    version = package.get("version")
    if not name or not version:
        raise ValueError(f"Invalid manifest, missing name/version: {manifest_path}")

    inferlet = f"{name}@{version}"
    wasm_stem = name.split("/")[-1].replace("-", "_")

    candidates = [
        crate_dir / "target" / "wasm32-wasip2" / "release" / f"{wasm_stem}.wasm",
        repo_root() / "target" / "wasm32-wasip2" / "release" / f"{wasm_stem}.wasm",
    ]

    if "sdk/examples" in str(spec.crate_dir):
        candidates.append(
            repo_root()
            / "sdk/examples"
            / "target"
            / "wasm32-wasip2"
            / "release"
            / f"{wasm_stem}.wasm"
        )
    if "benchmarks/inferlets" in str(spec.crate_dir):
        candidates.append(
            repo_root()
            / "benchmarks/inferlets"
            / "target"
            / "wasm32-wasip2"
            / "release"
            / f"{wasm_stem}.wasm"
        )

    for wasm_path in candidates:
        if wasm_path.exists():
            return wasm_path.resolve(), manifest_path, inferlet

    raise FileNotFoundError(
        "WASM binary not found for program "
        f"'{program_name}'. Expected one of: {', '.join(str(p) for p in candidates)}"
    )


def _get_event_enum():
    from pie_client import Event

    return Event


def add_auth_args(parser: argparse.ArgumentParser) -> None:
    auth_group = parser.add_argument_group("Authentication")
    auth_group.add_argument(
        "--username",
        type=str,
        default="benchmark-user",
        help="Username for Pie authentication",
    )
    auth_group.add_argument(
        "--private-key-path",
        type=str,
        default=None,
        help="Optional private key path for auth-enabled Pie servers",
    )


def load_private_key(path: str | None) -> ParsedPrivateKey | None:
    if not path:
        return None
    from pie_client.crypto import ParsedPrivateKey as ParsedKey

    return ParsedKey.from_file(path)


async def connect_client(args) -> PieClient:
    from pie_client import PieClient as _PieClient

    client = _PieClient(args.server_uri)
    await client.connect()
    try:
        await client.authenticate(args.username, load_private_key(args.private_key_path))
    except Exception:
        await client.close()
        raise
    return client


async def ensure_inferlet_installed(client: PieClient, program_name: str, verbose: bool = True) -> str:
    wasm_path, manifest_path, inferlet = resolve_program(program_name)

    exists = await client.program_exists(inferlet, wasm_path, manifest_path)
    if not exists:
        if verbose:
            print(f"Program '{inferlet}' not found on server, installing...")
        await client.install_program(wasm_path, manifest_path)
        if verbose:
            print("Install complete.")

    return inferlet


async def wait_until_terminated(instance, verbose: bool = False) -> str:
    event_enum = _get_event_enum()
    terminal_error_events = {
        event_enum.Aborted,
        event_enum.Exception,
        event_enum.ServerError,
        event_enum.OutOfResources,
    }

    while True:
        event, message = await instance.recv()

        if event == event_enum.Completed:
            if verbose:
                print(f"Instance {instance.instance_id} completed")
            return message if isinstance(message, str) else ""

        if event in terminal_error_events:
            raise RuntimeError(
                f"Instance {instance.instance_id} terminated with {event.name}: {message}"
            )

        if verbose:
            print(
                f"Instance {instance.instance_id} event={event.name} "
                f"message={str(message)[:200]}"
            )


async def run_simple_instances(
    client: PieClient,
    inferlet: str,
    num_instances: int,
    build_args,
    verbose: bool = False,
    instance_retries: int | None = None,
    collect_outputs: bool = False,
) -> list[str] | None:
    if instance_retries is None:
        instance_retries = int(os.environ.get("PIE_BENCH_INSTANCE_RETRIES", "0"))

    total_generated_tokens = 0

    async def _launch_one(index: int):
        nonlocal total_generated_tokens
        last_error: Exception | None = None
        for attempt in range(instance_retries + 1):
            instance_args = build_args(index)
            try:
                instance = await client.launch_instance(inferlet, arguments=instance_args)
                message = await wait_until_terminated(instance, verbose=verbose)
                total_generated_tokens += _estimate_generated_tokens_from_message(message)
                return message if collect_outputs else None
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < instance_retries:
                    if verbose:
                        print(
                            f"Instance {index} failed on attempt {attempt + 1}; retrying..."
                        )
                    await asyncio.sleep(0.1)
                    continue
                raise

        if last_error is not None:
            raise last_error

    tasks = [_launch_one(i) for i in range(num_instances)]
    outputs = await asyncio.gather(*tasks)
    _accumulate_generated_tokens_in_env(total_generated_tokens)
    if collect_outputs:
        return [output for output in outputs if isinstance(output, str)]
    return None


def random_math_prompt() -> str:
    num1 = random.randint(1, 100)
    num2 = random.randint(1, 100)
    return f"What is the sum of {num1} and {num2}?"
