"""Minimal .env loader: no dependency, no magic.

Called at the top of every CLI entry point so `ANTHROPIC_API_KEY=...` in the
repo-root .env (gitignored) just works without a manual `source`. Library
code never calls this: importing telecom_aut must not touch the process
environment.

Rules: KEY=VALUE lines only. '#' comments and blanks are ignored, and
optional single/double quotes are stripped. Variables ALREADY set in the
environment are never overridden (the shell wins). Values are never logged.
"""

from __future__ import annotations

import os
from pathlib import Path

_KEY_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")


def load_dotenv(path: str | Path = ".env") -> list[str]:
    """Load variables from `path` into os.environ. Returns the names set."""
    env_path = Path(path)
    if not env_path.is_file():
        return []
    loaded: list[str] = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if not key or set(key) - _KEY_CHARS:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key not in os.environ and value:
            os.environ[key] = value
            loaded.append(key)
    return loaded
