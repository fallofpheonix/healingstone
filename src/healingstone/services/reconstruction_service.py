"""Service-level orchestration around the reconstruction pipeline."""

from __future__ import annotations

import logging
from argparse import Namespace

from ..config.environment import apply_env_defaults
from ..core.runtime_config import build_runtime_config, to_namespace
from ..pipeline.run_pipeline import run_pipeline
from ..utils.runtime_info import collect_runtime_fingerprint

LOG = logging.getLogger(__name__)


def execute_reconstruction(cli_args: Namespace) -> None:
    """Resolve runtime configuration and execute the selected reconstruction flow."""
    args_with_env = apply_env_defaults(cli_args)
    bundle = build_runtime_config(args_with_env)
    effective_args = to_namespace(bundle)

    LOG.info("runtime=%s", collect_runtime_fingerprint())
    run_pipeline(effective_args)
