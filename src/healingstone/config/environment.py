"""Environment-backed defaults for runtime CLI arguments."""

from __future__ import annotations

import os
from argparse import Namespace


def apply_env_defaults(cli_args: Namespace) -> Namespace:
    """Fill selected CLI fields from environment without overriding explicit CLI input."""
    env_map = {
        "data_dir": "HEALINGSTONE_DATA_DIR",
        "output_dir": "HEALINGSTONE_OUTPUT_DIR",
        "labels_csv": "HEALINGSTONE_LABELS_CSV",
    }

    for field_name, env_name in env_map.items():
        current = getattr(cli_args, field_name, None)
        if current is None and env_name in os.environ:
            setattr(cli_args, field_name, os.environ[env_name])

    return cli_args
