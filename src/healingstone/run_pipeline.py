"""Backward-compatible command entrypoint.

TODO: remove this compatibility module after external scripts migrate to
`healingstone.api.cli:main`.
"""

from __future__ import annotations

from .api.cli import main

__all__ = ["main"]
