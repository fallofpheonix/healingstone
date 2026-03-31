"""Compatibility wrapper for legacy ``python -m healingstone.cli`` usage."""

from __future__ import annotations

import sys

from .api.cli import main as _main

__all__ = ["main"]


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        del sys.argv[1]
    _main()


if __name__ == "__main__":
    main()
