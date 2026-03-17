"""Runtime metadata helpers for lightweight observability."""

from __future__ import annotations

import platform
import subprocess
from typing import Dict


def _resolve_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
    except Exception:
        return "unknown"
    return out.decode("utf-8").strip()


def collect_runtime_fingerprint() -> Dict[str, str]:
    """Return compact runtime metadata used in startup logging."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "commit": _resolve_commit(),
    }
