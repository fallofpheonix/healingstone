"""Pipeline execution service wrapper."""

from __future__ import annotations

from argparse import Namespace

from ..core.runtime_config import build_runtime_config, to_namespace
from ..pipeline.run_pipeline import run_pipeline


def execute_reconstruction(args: Namespace) -> None:
    bundle = build_runtime_config(args)
    effective_args = to_namespace(bundle)
    setattr(effective_args, "_train_config", bundle.train)
    setattr(effective_args, "_dataset_aliases", bundle.datasets.get("aliases", {}))
    setattr(effective_args, "_config_source_map", bundle.source_map)
    setattr(effective_args, "_config_paths", bundle.config_paths)
    setattr(effective_args, "_config_hash", bundle.config_hash)
    run_pipeline(effective_args)
