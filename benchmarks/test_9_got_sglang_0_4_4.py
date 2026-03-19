#!/usr/bin/env python3
from versioned_baseline_runner import run_versioned_baseline

if __name__ == "__main__":
    run_versioned_baseline(
        target_script='test_9_got_baseline.py',
        backend='sglang',
        backend_version='0.4.4',
        mode='pinned',
        pass_backend_arg=True,
        unsupported_reason=None,
    )
