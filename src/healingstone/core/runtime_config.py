"""Minimal runtime config loader for CLI/test compatibility."""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict

import yaml


@dataclass
class RuntimeConfigBundle:
    pipeline: SimpleNamespace
    train: Dict[str, Any]
    datasets: Dict[str, Any]
    resolved: Dict[str, Any]


def _load_yaml(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload if isinstance(payload, dict) else {}


def build_runtime_config(args: Any) -> RuntimeConfigBundle:
    """Resolve runtime config with CLI > env > YAML precedence."""
    pipeline_cfg = _load_yaml(getattr(args, "config", None))
    train_cfg = _load_yaml(getattr(args, "train_config", None))
    datasets_cfg = _load_yaml(getattr(args, "dataset_manifest", None))

    config_version = int(pipeline_cfg.get("config_version", 1))
    if config_version != 1:
        raise ValueError(f"Unsupported config_version={config_version}")

    env_seed = os.environ.get("HEALINGSTONE_SEED")
    seed = (
        getattr(args, "seed", None)
        if getattr(args, "seed", None) is not None
        else int(env_seed) if env_seed is not None else int(pipeline_cfg.get("seed", 42))
    )

    data_dir = (
        getattr(args, "data_dir", None)
        or os.environ.get("HEALINGSTONE_DATA_DIR")
        or pipeline_cfg.get("data_dir")
    )
    output_dir = (
        getattr(args, "output_dir", None)
        or os.environ.get("HEALINGSTONE_OUTPUT_DIR")
        or pipeline_cfg.get("output_dir")
    )
    dataset_alias = getattr(args, "dataset_alias", None) or pipeline_cfg.get("dataset_alias", "3d")

    resolved = dict(vars(args))
    resolved.update(
        {
            "config_version": config_version,
            "seed": seed,
            "data_dir": data_dir,
            "output_dir": output_dir,
            "dataset_alias": dataset_alias,
        }
    )

    pipeline_ns = SimpleNamespace(seed=seed, config_version=config_version)
    return RuntimeConfigBundle(
        pipeline=pipeline_ns,
        train=train_cfg,
        datasets=datasets_cfg,
        resolved=resolved,
    )


def to_namespace(bundle: RuntimeConfigBundle) -> Any:
    """Flatten the resolved config into an argparse-like namespace."""
    return SimpleNamespace(**bundle.resolved)
