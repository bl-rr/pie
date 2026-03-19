"""Path utilities for Pie."""

import os
from pathlib import Path

DEFAULT_PIE_HOME_DIR = ".pie-eval"
DEFAULT_PIE_CACHE_DIR = "pie-eval"


def get_pie_home() -> Path:
    """Get the Pie home directory (~/.pie-eval or PIE_HOME env var)."""
    if pie_home := os.environ.get("PIE_HOME"):
        return Path(pie_home)
    return Path.home() / DEFAULT_PIE_HOME_DIR


def get_pie_cache_home() -> Path:
    """Get the Pie cache directory (~/.cache/pie-eval or PIE_HOME env var)."""
    if pie_home := os.environ.get("PIE_HOME"):
        return Path(pie_home)
    return Path.home() / ".cache" / DEFAULT_PIE_CACHE_DIR


def get_default_config_path() -> Path:
    """Get the default config file path (~/.pie-eval/config.toml)."""
    return get_pie_home() / "config.toml"


def get_authorized_users_path() -> Path:
    """Get the authorized users file path (~/.pie-eval/authorized_users.toml)."""
    return get_pie_home() / "authorized_users.toml"


def get_shell_history_path() -> Path:
    """Get the shell history file path (~/.pie-eval/.pie_history)."""
    return get_pie_home() / ".pie_history"


def expand_path(path_str: str) -> Path:
    """Expand ~ and environment variables in a path string."""
    return Path(os.path.expanduser(os.path.expandvars(path_str)))
