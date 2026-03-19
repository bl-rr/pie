#!/usr/bin/env python3
from versioned_baseline_runner import run_versioned_baseline

if __name__ == "__main__":
    run_versioned_baseline(
        target_script=None,
        backend='sglang',
        backend_version='0.5.9',
        mode='latest',
        pass_backend_arg=True,
        unsupported_reason='No equivalent n-gram speculative decoding API in SGLang OpenAI-compatible server mode.',
    )
