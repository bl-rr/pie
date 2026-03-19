#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


READY_SENTINEL = "Engine running. Press Ctrl+C to stop"
OOM_PATTERNS = (
    "cuda out of memory",
    "outofmemoryerror",
    "torch.outofmemoryerror",
)


@dataclass(frozen=True)
class Workload:
    script: str
    flag: str
    legacy_default: int


WORKLOADS: list[Workload] = [
    Workload("test_1_agent_react_pie.py", "--num-instances", 128),
    Workload("test_2_agent_codeact_pie.py", "--num-instances", 128),
    Workload("test_3_agent_swarm_pie.py", "--num-pipelines", 32),
    Workload("test_4_agent_case_study_pie.py", "--num-instances", 128),
    Workload("test_5_text_completion_pie.py", "--num-instances", 128),
    Workload("test_6_prefix_tree_pie.py", "--num-instances", 64),
    Workload("test_7_tot_pie.py", "--num-instances", 64),
    Workload("test_8_rot_pie.py", "--num-instances", 32),
    Workload("test_9_got_pie.py", "--num-instances", 128),
    Workload("test_10_skot_pie.py", "--num-instances", 128),
    Workload("test_11_cache_pie.py", "--num-instances", 128),
    Workload("test_12_ebnf_pie.py", "--num-instances", 128),
    Workload("test_13_specdec_pie.py", "--num-instances", 128),
    Workload("test_14_beamsearch_pie.py", "--num-instances", 128),
    Workload("test_15_attnsink_pie.py", "--num-instances", 128),
    Workload("test_16_parallel_generation_pie.py", "--num-instances", 128),
    Workload("microbench_spawn_time.py", "--num-instances", 1000),
    Workload("microbench_execution_latency.py", "--num-instances", 1000),
]


class PieServerManager:
    def __init__(self, bench_dir: Path, startup_timeout_s: int) -> None:
        self.bench_dir = bench_dir
        self.startup_timeout_s = startup_timeout_s
        self.proc: subprocess.Popen[str] | None = None
        self.log_path = bench_dir / "calibration_pie_backend.log"
        self._scan_pos = 0

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = self.log_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            ["./run_pie.sh"],
            cwd=self.bench_dir,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        self._wait_ready()
        self._scan_pos = self.log_path.stat().st_size

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 15
            while self.proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.2)
            if self.proc.poll() is None:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.proc = None

    def restart(self) -> None:
        self.stop()
        self.start()

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_s
        last_size = 0
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                raise RuntimeError(f"PIE exited during startup.\n{tail}")
            if self.log_path.exists():
                text = self.log_path.read_text(encoding="utf-8", errors="replace")
                if READY_SENTINEL in text:
                    return
                last_size = len(text)
            time.sleep(1)
        tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise TimeoutError(f"Timed out waiting for PIE readiness.\n{tail}")

    def check_new_oom(self) -> bool:
        if not self.log_path.exists():
            return False
        with self.log_path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(self._scan_pos)
            chunk = f.read()
            self._scan_pos = f.tell()
        low = chunk.lower()
        return any(pat in low for pat in OOM_PATTERNS)

    def ensure_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


def run_one(
    *,
    bench_dir: Path,
    python_bin: Path,
    server_uri: str,
    workload: Workload,
    value: int,
    timeout_s: int,
    env: dict[str, str],
    pie: PieServerManager,
    poll_interval_s: int,
) -> tuple[int, str, bool]:
    cmd = [
        str(python_bin),
        workload.script,
        "--server-uri",
        server_uri,
        workload.flag,
        str(value),
    ]
    out_path = bench_dir / ".calibration_last_attempt.log"
    oom_detected = False
    start = time.monotonic()
    next_poll = start + poll_interval_s

    with out_path.open("w", encoding="utf-8") as out_fp:
        proc = subprocess.Popen(
            cmd,
            cwd=bench_dir,
            env=env,
            stdout=out_fp,
            stderr=subprocess.STDOUT,
            text=True,
        )
        rc: int | None = None
        while rc is None:
            rc = proc.poll()
            now = time.monotonic()

            if rc is not None:
                break
            if timeout_s > 0 and (now - start > timeout_s):
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                rc = 124
                break
            if now >= next_poll:
                # Check backend health every poll interval.
                if not pie.ensure_alive():
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    rc = 125
                    break
                if pie.check_new_oom():
                    oom_detected = True
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    rc = 125
                    break
                next_poll = now + poll_interval_s
            time.sleep(0.5)

    output = out_path.read_text(encoding="utf-8", errors="replace")
    if oom_detected:
        output += "\n[calibration] detected OOM in backend log; forcing backend restart.\n"
        pie.restart()
    elif not pie.ensure_alive():
        output += "\n[calibration] backend exited; forcing backend restart.\n"
        pie.restart()
    elif rc != 0:
        # Keep each binary-search probe isolated from prior failed state.
        output += "\n[calibration] attempt failed; forcing backend restart before next probe.\n"
        pie.restart()

    return int(rc), output, oom_detected


def binary_calibrate(
    *,
    bench_dir: Path,
    python_bin: Path,
    server_uri: str,
    workload: Workload,
    timeout_s: int,
    env: dict[str, str],
    pie: PieServerManager,
    poll_interval_s: int,
) -> dict[str, object]:
    log_lines: list[str] = []
    legacy = workload.legacy_default

    def attempt(v: int) -> bool:
        rc, out, oom = run_one(
            bench_dir=bench_dir,
            python_bin=python_bin,
            server_uri=server_uri,
            workload=workload,
            value=v,
            timeout_s=timeout_s,
            env=env,
            pie=pie,
            poll_interval_s=poll_interval_s,
        )
        ok = rc == 0
        status = "ok" if ok else f"rc={rc}"
        if oom:
            status += " (oom)"
        log_lines.append(f"  - try {workload.flag}={v}: {status}")
        if not ok:
            tail = "\n".join(out.splitlines()[-12:])
            if tail:
                log_lines.append("    last log lines:")
                for ln in tail.splitlines():
                    log_lines.append(f"      {ln}")
        return ok

    candidate = legacy
    upper_fail = legacy + 1
    lower_success = 0
    while candidate >= 1:
        if attempt(candidate):
            lower_success = candidate
            break
        upper_fail = candidate
        candidate //= 2

    if lower_success == 0:
        return {
            "script": workload.script,
            "flag": workload.flag,
            "legacy_default": legacy,
            "max_safe": 0,
            "notes": "no successful value found",
            "trace": log_lines,
        }

    lo = lower_success + 1
    hi = upper_fail - 1
    best = lower_success
    while lo <= hi:
        mid = (lo + hi) // 2
        if attempt(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return {
        "script": workload.script,
        "flag": workload.flag,
        "legacy_default": legacy,
        "max_safe": best,
        "trace": log_lines,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Calibrate per-script safe concurrency for PIE Llama-3.1-8B: "
            "start from legacy default, halve on failure, then binary refine."
        ),
    )
    parser.add_argument("--server-uri", default="ws://127.0.0.1:10009")
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Per-attempt timeout (seconds); 0 disables timeout (default: unbounded).",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=600,
        help="PIE startup timeout (seconds)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=10,
        help="Seconds between backend OOM polls while each attempt is running.",
    )
    parser.add_argument(
        "--output",
        default="safe_31_8b_calibration.json",
        help="Output JSON filename under benchmarks/",
    )
    parser.add_argument(
        "--scripts",
        nargs="*",
        default=None,
        help="Optional subset of scripts to calibrate (by filename).",
    )
    args = parser.parse_args()

    bench_dir = Path(__file__).resolve().parent
    repo_root = bench_dir.parent
    python_bin = repo_root / "pie" / ".venv" / "bin" / "python"
    if not python_bin.exists():
        print(f"Missing python interpreter: {python_bin}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    py_client_src = repo_root / "client" / "python" / "src"
    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{py_client_src}:{old_pp}" if old_pp else str(py_client_src)

    selected = WORKLOADS
    if args.scripts:
        wanted = set(args.scripts)
        selected = [w for w in WORKLOADS if w.script in wanted]
        missing = sorted(wanted - {w.script for w in selected})
        if missing:
            print(f"Warning: requested scripts not found in calibration list: {missing}")

    if not selected:
        print("No workloads selected.", file=sys.stderr)
        return 2

    pie = PieServerManager(bench_dir=bench_dir, startup_timeout_s=args.startup_timeout)
    results: dict[str, dict[str, object]] = {}

    try:
        pie.start()
        print(f"Calibrating {len(selected)} workloads against {args.server_uri}")
        for idx, w in enumerate(selected, start=1):
            print(f"[{idx}/{len(selected)}] {w.script} ({w.flag}, legacy={w.legacy_default})")
            res = binary_calibrate(
                bench_dir=bench_dir,
                python_bin=python_bin,
                server_uri=args.server_uri,
                workload=w,
                timeout_s=args.timeout,
                env=env,
                pie=pie,
                poll_interval_s=args.poll_interval,
            )
            results[w.script] = res
            print(f"  -> max_safe={res['max_safe']}")
            trace = res.get("trace", [])
            if trace:
                print("\n".join(trace[-3:]))
    finally:
        pie.stop()

    out_path = bench_dir / args.output
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote calibration: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
