"""Primary CLI entrypoint for Healingstone."""

from __future__ import annotations


def parse_args():
    from ..pipeline.run_pipeline import parse_args as _parse_args

    return _parse_args()


def execute_reconstruction(args):
    from ..services.reconstruction_service import execute_reconstruction as _execute_reconstruction

    return _execute_reconstruction(args)


def main() -> None:
    """Parse CLI arguments and run reconstruction pipeline."""
    args = parse_args()
    execute_reconstruction(args)


if __name__ == "__main__":
    main()
