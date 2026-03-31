"""Compatibility wrapper for legacy console scripts."""

from __future__ import annotations

from .pipeline.run_pipeline import main, parse_args, run_pipeline

__all__ = ["main", "parse_args", "run_pipeline"]


if __name__ == "__main__":
    main()
