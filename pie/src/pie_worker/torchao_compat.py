"""Compatibility helpers for torchao imports."""

from __future__ import annotations

import importlib
import warnings

_FILTER_CONFIGURED = False


def _configure_warning_filter() -> None:
    """Suppress known torchao SyntaxWarning on Python 3.12+."""
    global _FILTER_CONFIGURED
    if _FILTER_CONFIGURED:
        return

    warnings.filterwarnings(
        "ignore",
        message=r"invalid escape sequence.*",
        category=SyntaxWarning,
        module=r"torchao\.quantization\.quant_api",
    )
    _FILTER_CONFIGURED = True


def import_torchao():
    """Import torchao with compatibility warning filters in place."""
    _configure_warning_filter()
    return importlib.import_module("torchao")
