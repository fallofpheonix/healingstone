"""Compatibility wrapper for the API CLI entrypoint."""

from __future__ import annotations

import argparse

from ..services.reconstruction_service import execute_reconstruction

__all__ = ["execute_reconstruction", "main", "parse_args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Healingstone API entrypoint")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--dataset-manifest", default="configs/datasets.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--labels-csv", default=None)
    parser.add_argument("--allow-overwrite-run", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    execute_reconstruction(args)
