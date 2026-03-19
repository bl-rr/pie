#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent


BACKEND_VERSIONS = {
    "vllm": {
        "pinned": "0.6.0",
        "latest": "0.16.0",
    },
    "sglang": {
        "pinned": "0.4.4",
        "latest": "0.5.9",
    },
}


@dataclass(frozen=True)
class BackendSpec:
    target_script: str | None
    pass_backend_arg: bool = True
    supported: bool = True
    reason: str | None = None


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    kind: str
    pie_script: str
    backends: Dict[str, BackendSpec]


WORKLOADS: List[WorkloadSpec] = [
    WorkloadSpec(
        name="test_1_agent_react",
        kind="legacy",
        pie_script="test_1_agent_react_pie.py",
        backends={
            "vllm": BackendSpec("test_1_agent_react_baseline.py"),
            "sglang": BackendSpec("test_1_agent_react_baseline.py"),
        },
    ),
    WorkloadSpec(
        name="test_2_agent_codeact",
        kind="legacy",
        pie_script="test_2_agent_codeact_pie.py",
        backends={
            "vllm": BackendSpec("test_2_agent_codeact_baseline.py"),
            "sglang": BackendSpec("test_2_agent_codeact_baseline.py"),
        },
    ),
    WorkloadSpec(
        name="test_3_agent_swarm",
        kind="legacy",
        pie_script="test_3_agent_swarm_pie.py",
        backends={
            "vllm": BackendSpec("test_3_agent_swarm_baseline.py"),
            "sglang": BackendSpec("test_3_agent_swarm_baseline.py"),
        },
    ),
    WorkloadSpec(
        name="test_4_agent_case_study",
        kind="legacy",
        pie_script="test_4_agent_case_study_pie.py",
        backends={
            "vllm": BackendSpec("test_4_agent_case_study_baseline.py"),
            "sglang": BackendSpec("test_4_agent_case_study_baseline.py"),
        },
    ),
    WorkloadSpec(
        name="test_5_text_completion",
        kind="legacy",
        pie_script="test_5_text_completion_pie.py",
        backends={
            "vllm": BackendSpec("test_5_text_completion_baseline.py"),
            "sglang": BackendSpec("test_5_text_completion_baseline.py"),
        },
    ),
    WorkloadSpec(
        name="test_6_prefix_tree",
        kind="legacy",
        pie_script="test_6_prefix_tree_pie.py",
        backends={
            "vllm": BackendSpec("test_6_prefix_tree_baseline.py"),
            "sglang": BackendSpec(
                "test_6_prefix_tree_sglang.py",
                pass_backend_arg=False,
            ),
        },
    ),
    WorkloadSpec(
        name="test_7_tot",
        kind="legacy",
        pie_script="test_7_tot_pie.py",
        backends={
            "vllm": BackendSpec("test_7_tot_baseline.py"),
            "sglang": BackendSpec(
                "test_7_tot_sglang.py",
                pass_backend_arg=False,
            ),
        },
    ),
    WorkloadSpec(
        name="test_8_rot",
        kind="legacy",
        pie_script="test_8_rot_pie.py",
        backends={
            "vllm": BackendSpec("test_8_rot_baseline.py"),
            "sglang": BackendSpec("test_8_rot_baseline.py"),
        },
    ),
    WorkloadSpec(
        name="test_9_got",
        kind="legacy",
        pie_script="test_9_got_pie.py",
        backends={
            "vllm": BackendSpec("test_9_got_baseline.py"),
            "sglang": BackendSpec("test_9_got_baseline.py"),
        },
    ),
    WorkloadSpec(
        name="test_10_skot",
        kind="legacy",
        pie_script="test_10_skot_pie.py",
        backends={
            "vllm": BackendSpec("test_10_skot_baseline.py"),
            "sglang": BackendSpec("test_10_skot_baseline.py"),
        },
    ),
    WorkloadSpec(
        name="test_11_cache",
        kind="legacy",
        pie_script="test_11_cache_pie.py",
        backends={
            "vllm": BackendSpec("test_11_cache_baseline.py"),
            "sglang": BackendSpec("test_11_cache_baseline.py"),
        },
    ),
    WorkloadSpec(
        name="test_12_ebnf",
        kind="legacy",
        pie_script="test_12_ebnf_pie.py",
        backends={
            "vllm": BackendSpec("test_12_ebnf_baseline.py"),
            "sglang": BackendSpec("test_12_ebnf_baseline.py"),
        },
    ),
    WorkloadSpec(
        name="test_13_specdec",
        kind="legacy",
        pie_script="test_13_specdec_pie.py",
        backends={
            "vllm": BackendSpec("test_13_specdec_baseline.py"),
            "sglang": BackendSpec(
                target_script=None,
                supported=False,
                reason=(
                    "No equivalent n-gram speculative decoding API in SGLang "
                    "OpenAI-compatible server mode."
                ),
            ),
        },
    ),
    WorkloadSpec(
        name="test_14_beamsearch",
        kind="legacy",
        pie_script="test_14_beamsearch_pie.py",
        backends={
            "vllm": BackendSpec("test_14_beamsearch_baseline.py"),
            "sglang": BackendSpec(
                target_script=None,
                supported=False,
                reason=(
                    "No tested SGLang endpoint parity for vLLM `use_beam_search` "
                    "request contract."
                ),
            ),
        },
    ),
    WorkloadSpec(
        name="test_15_attnsink",
        kind="legacy",
        pie_script="test_15_attnsink_pie.py",
        backends={
            "vllm": BackendSpec("test_15_attnsink_baseline.py"),
            "sglang": BackendSpec("test_15_attnsink_baseline.py"),
        },
    ),
    WorkloadSpec(
        name="test_16_parallel_generation",
        kind="new-example",
        pie_script="test_16_parallel_generation_pie.py",
        backends={
            "vllm": BackendSpec("test_16_parallel_generation_baseline.py"),
            "sglang": BackendSpec("test_16_parallel_generation_baseline.py"),
        },
    ),
]


def version_tag(version: str) -> str:
    return version.replace(".", "_")


def wrapper_body(
    *,
    target_script: str | None,
    backend: str,
    version: str,
    mode: str,
    pass_backend_arg: bool,
    reason: str | None,
) -> str:
    lines = [
        "#!/usr/bin/env python3",
        "from versioned_baseline_runner import run_versioned_baseline",
        "",
        "if __name__ == \"__main__\":",
        "    run_versioned_baseline(",
        f"        target_script={target_script!r},",
        f"        backend={backend!r},",
        f"        backend_version={version!r},",
        f"        mode={mode!r},",
        f"        pass_backend_arg={pass_backend_arg!r},",
        f"        unsupported_reason={reason!r},",
        "    )",
        "",
    ]
    return "\n".join(lines)


def write_wrapper(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def build_coverage() -> dict:
    workloads_payload = []
    implemented = 0
    omitted = 0

    for workload in WORKLOADS:
        cells = []
        for backend in ("vllm", "sglang"):
            backend_spec = workload.backends[backend]
            for mode in ("pinned", "latest"):
                version = BACKEND_VERSIONS[backend][mode]
                script = f"{workload.name}_{backend}_{version_tag(version)}.py"
                status = "implemented" if backend_spec.supported else "omitted"
                if status == "implemented":
                    implemented += 1
                else:
                    omitted += 1
                cells.append(
                    {
                        "backend": backend,
                        "mode": mode,
                        "version": version,
                        "status": status,
                        "script": script,
                        "target_script": backend_spec.target_script,
                        "reason": backend_spec.reason,
                    }
                )

        workloads_payload.append(
            {
                "name": workload.name,
                "type": workload.kind,
                "pie_script": workload.pie_script,
                "cells": cells,
            }
        )

    return {
        "freeze_date": "2026-03-02",
        "backend_versions": BACKEND_VERSIONS,
        "summary": {
            "workloads": len(WORKLOADS),
            "legacy_workloads": len([w for w in WORKLOADS if w.kind == "legacy"]),
            "new_example_workloads": len([w for w in WORKLOADS if w.kind == "new-example"]),
            "matrix_cells": implemented + omitted,
            "implemented_cells": implemented,
            "omitted_cells": omitted,
        },
        "workloads": workloads_payload,
    }


def coverage_cell_text(cell: dict) -> str:
    if cell["status"] == "implemented":
        return f"`{cell['script']}`"
    return f"omitted: {cell['reason']}"


def write_coverage_markdown(coverage: dict) -> None:
    lines = [
        "# Benchmark Coverage Report",
        "",
        f"Freeze date: `{coverage['freeze_date']}`",
        "",
        "## Backend versions",
        "",
        "| Backend | Pinned | Latest |",
        "| --- | --- | --- |",
        f"| vLLM | `{BACKEND_VERSIONS['vllm']['pinned']}` | `{BACKEND_VERSIONS['vllm']['latest']}` |",
        f"| SGLang | `{BACKEND_VERSIONS['sglang']['pinned']}` | `{BACKEND_VERSIONS['sglang']['latest']}` |",
        "",
        "## Matrix",
        "",
        "| Workload | Type | PIE | vLLM pinned | vLLM latest | SGLang pinned | SGLang latest |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for workload in coverage["workloads"]:
        cells = workload["cells"]
        cell_index = {(c["backend"], c["mode"]): c for c in cells}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{workload['name']}`",
                    workload["type"],
                    f"`{workload['pie_script']}`",
                    coverage_cell_text(cell_index[("vllm", "pinned")]),
                    coverage_cell_text(cell_index[("vllm", "latest")]),
                    coverage_cell_text(cell_index[("sglang", "pinned")]),
                    coverage_cell_text(cell_index[("sglang", "latest")]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Omitted Cells",
            "",
        ]
    )

    for workload in coverage["workloads"]:
        for cell in workload["cells"]:
            if cell["status"] == "omitted":
                lines.append(
                    "- "
                    + f"`{workload['name']}` "
                    + f"{cell['backend']} {cell['mode']} ({cell['version']}): "
                    + f"{cell['reason']}"
                )

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Workloads: `{coverage['summary']['workloads']}`",
            f"- Matrix cells: `{coverage['summary']['matrix_cells']}`",
            f"- Implemented: `{coverage['summary']['implemented_cells']}`",
            f"- Omitted: `{coverage['summary']['omitted_cells']}`",
        ]
    )

    (ROOT / "coverage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    expected_wrapper_files = set()

    for workload in WORKLOADS:
        for backend in ("vllm", "sglang"):
            backend_spec = workload.backends[backend]
            for mode in ("pinned", "latest"):
                version = BACKEND_VERSIONS[backend][mode]
                exact_name = f"{workload.name}_{backend}_{version_tag(version)}.py"
                expected_wrapper_files.add(exact_name)
                exact_content = wrapper_body(
                    target_script=backend_spec.target_script,
                    backend=backend,
                    version=version,
                    mode=mode,
                    pass_backend_arg=backend_spec.pass_backend_arg,
                    reason=backend_spec.reason if not backend_spec.supported else None,
                )
                write_wrapper(ROOT / exact_name, exact_content)

    # Remove stale generated wrappers (old alias files and old version files).
    for path in ROOT.glob("test_*_vllm_*.py"):
        if path.name not in expected_wrapper_files:
            path.unlink()
    for path in ROOT.glob("test_*_sglang_*.py"):
        if path.name not in expected_wrapper_files:
            path.unlink()

    coverage = build_coverage()
    (ROOT / "coverage_report.json").write_text(
        json.dumps(coverage, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_coverage_markdown(coverage)


if __name__ == "__main__":
    main()
