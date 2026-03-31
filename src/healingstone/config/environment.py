"""Environment default handling for CLI args."""

from __future__ import annotations

import os
from argparse import Namespace


def apply_env_defaults(args: Namespace) -> Namespace:
    """Fill CLI defaults from environment variables when missing."""
    if getattr(args, "data_dir", None) is None:
        env_data = os.environ.get("HEALINGSTONE_DATA_DIR")
        if env_data:
            args.data_dir = env_data

    if getattr(args, "output_dir", None) is None:
        env_out = os.environ.get("HEALINGSTONE_OUTPUT_DIR")
        if env_out:
            args.output_dir = env_out

    if getattr(args, "labels_csv", None) is None:
        env_labels = os.environ.get("HEALINGSTONE_LABELS_CSV")
        if env_labels:
            args.labels_csv = env_labels

    return args
