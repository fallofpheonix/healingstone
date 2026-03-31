"""Compatibility wrapper for the API CLI entrypoint."""

from __future__ import annotations

import argparse

from ..services.reconstruction_service import execute_reconstruction

__all__ = ["execute_reconstruction", "main", "parse_args"]


def parse_args() -> argparse.Namespace:
    from ..pipeline.run_pipeline import parse_args as parse_runtime_args

    return parse_runtime_args()


def main() -> None:
    args = parse_args()
    execute_reconstruction(args)


if __name__ == "__main__":
    main()
