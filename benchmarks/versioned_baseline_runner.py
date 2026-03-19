#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def run_versioned_baseline(
    *,
    target_script: str | None,
    backend: str,
    backend_version: str,
    mode: str,
    pass_backend_arg: bool = True,
    unsupported_reason: str | None = None,
) -> None:
    if unsupported_reason:
        script_name = Path(sys.argv[0]).name
        help_requested = any(arg in {"-h", "--help"} for arg in sys.argv[1:])
        message = (
            f"{script_name}: unsupported baseline variant\n"
            f"backend={backend} version={backend_version} mode={mode}\n"
            f"reason: {unsupported_reason}"
        )
        if help_requested:
            print(message)
            return
        print(message, file=sys.stderr)
        raise SystemExit(2)

    if target_script is None:
        raise ValueError("target_script must be provided for supported wrappers")

    bench_dir = Path(__file__).resolve().parent
    target_path = (bench_dir / target_script).resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"Target script not found: {target_path}")

    os.environ["PIE_BASELINE_BACKEND"] = backend
    os.environ["PIE_BASELINE_VERSION"] = backend_version
    os.environ["PIE_BASELINE_MODE"] = mode

    # Keep wrapper CLI behavior identical to `python <target>.py ...`.
    argv = [str(target_path), *sys.argv[1:]]
    if pass_backend_arg:
        # Append forced backend at the end to avoid accidental overrides.
        argv.extend(["--backend", backend])
    sys.argv = argv
    runpy.run_path(str(target_path), run_name="__main__")
