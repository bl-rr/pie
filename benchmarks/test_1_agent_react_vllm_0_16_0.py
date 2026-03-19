#!/usr/bin/env python3
from versioned_baseline_runner import run_versioned_baseline

if __name__ == "__main__":
    run_versioned_baseline(
        target_script='test_1_agent_react_baseline.py',
        backend='vllm',
        backend_version='0.16.0',
        mode='latest',
        pass_backend_arg=True,
        unsupported_reason=None,
    )
