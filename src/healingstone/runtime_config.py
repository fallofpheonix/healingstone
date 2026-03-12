"""Runtime configuration loading and precedence resolution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

SUPPORTED_CONFIG_VERSION = {1}
ENV_PREFIX = "HEALINGSTONE_"


class PipelineConfig(BaseModel):
    """Pipeline runtime configuration."""

    model_config = ConfigDict(extra="forbid")

    config_version: int = Field(default=1)
    dataset_alias: str = Field(default="3d")
    data_dir: str | None = None
    output_dir: str | None = None
    labels_csv: str | None = None
    allow_overwrite_run: bool = False

    sample_points: int = Field(default=40000, gt=0)
    voxel_size: float = Field(default=0.01, gt=0)
    normal_radius: float = Field(default=0.04, gt=0)
    normal_max_nn: int = Field(default=64, gt=0)
    outlier_nb_neighbors: int = Field(default=24, gt=0)
    outlier_std_ratio: float = Field(default=1.8, gt=0)

    k_neighbors: int = Field(default=24, gt=0)
    fpfh_radius: float = Field(default=0.06, gt=0)
    fpfh_max_nn: int = Field(default=100, gt=0)
    dbscan_eps: float = Field(default=0.04, gt=0)
    dbscan_min_samples: int = Field(default=24, gt=0)
    n_keypoints: int = Field(default=256, gt=0)

    candidate_top_k: int = Field(default=4, gt=0)
    align_top_n: int = Field(default=10, gt=0)
    label_suggestions_top_n: int = Field(default=50, gt=0)
    threshold_objective: str = Field(default="accuracy", pattern="^(accuracy|f1)$")
    min_match_accuracy: float = Field(default=0.0, ge=0.0)
    min_required_accuracy: float = Field(default=0.80, ge=0.0, le=1.0)
    evaluation_split: str = Field(default="test", pattern="^(train|validation|test)$")

    augment_rotations: bool = False
    augment_count: int = Field(default=2, gt=0)

    seed: int = 42
    device: str = Field(default="cpu", pattern="^(cpu|cuda)$")


class TrainConfig(BaseModel):
    """Training sub-configuration stored for reproducibility."""

    model_config = ConfigDict(extra="forbid")

    config_version: int = Field(default=1)
    emb_dim: int = Field(default=64, gt=0)
    epochs: int = Field(default=120, gt=0)
    batch_size: int = Field(default=64, gt=0)
    lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-5, ge=0)
    margin: float = Field(default=1.0, gt=0)


class DatasetManifest(BaseModel):
    """Dataset alias mapping."""

    model_config = ConfigDict(extra="forbid")
    aliases: Dict[str, str] = Field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeConfigBundle:
    pipeline: PipelineConfig
    train: TrainConfig
    dataset_manifest: DatasetManifest
    config_hash: str
    config_paths: Dict[str, Path]
    source_map: Dict[str, str]


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return raw


def _parse_env_value(raw: str) -> Any:
    low = raw.strip().lower()
    if low in {"true", "1", "yes", "y", "on"}:
        return True
    if low in {"false", "0", "no", "n", "off"}:
        return False
    try:
        if "." in raw or "e" in low:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _collect_env_overrides(model_cls: type[BaseModel]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for field_name in model_cls.model_fields:
        env_key = ENV_PREFIX + field_name.upper()
        if env_key in os.environ:
            overrides[field_name] = _parse_env_value(os.environ[env_key])
    return overrides


def _collect_cli_overrides(cli_args: argparse.Namespace, model_cls: type[BaseModel]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    arg_map = vars(cli_args)
    for field_name in model_cls.model_fields:
        val = arg_map.get(field_name)
        if val is not None:
            overrides[field_name] = val
    return overrides


def _infer_source(field: str, cli: Dict[str, Any], env: Dict[str, Any], yaml_cfg: Dict[str, Any], fallback: str) -> str:
    if field in cli:
        return "cli"
    if field in env:
        return "env"
    if field in yaml_cfg:
        return "yaml"
    return fallback


def _validate_config_version(cfg_name: str, version: int) -> None:
    if version not in SUPPORTED_CONFIG_VERSION:
        raise ValueError(
            f"Unsupported {cfg_name} config_version={version}; supported versions={sorted(SUPPORTED_CONFIG_VERSION)}"
        )


def build_runtime_config(cli_args: argparse.Namespace) -> RuntimeConfigBundle:
    pipeline_path = Path(cli_args.config).expanduser().resolve()
    train_path = Path(cli_args.train_config).expanduser().resolve()
    datasets_path = Path(cli_args.dataset_manifest).expanduser().resolve()

    pipeline_yaml = _load_yaml(pipeline_path)
    train_yaml = _load_yaml(train_path)
    manifest_yaml = _load_yaml(datasets_path)

    pipeline_env = _collect_env_overrides(PipelineConfig)
    train_env = _collect_env_overrides(TrainConfig)
    pipeline_cli = _collect_cli_overrides(cli_args, PipelineConfig)

    pipeline_raw = {**pipeline_yaml, **pipeline_env, **pipeline_cli}
    train_raw = {**train_yaml, **train_env}

    try:
        pipeline_cfg = PipelineConfig(**pipeline_raw)
        train_cfg = TrainConfig(**train_raw)
        manifest_cfg = DatasetManifest(**manifest_yaml)
    except ValidationError as exc:
        raise ValueError(f"Configuration validation failed: {exc}") from exc

    _validate_config_version("pipeline", pipeline_cfg.config_version)
    _validate_config_version("train", train_cfg.config_version)

    source_map = {
        "data_dir": _infer_source("data_dir", pipeline_cli, pipeline_env, pipeline_yaml, "default"),
        "output_dir": _infer_source("output_dir", pipeline_cli, pipeline_env, pipeline_yaml, "default"),
        "labels_csv": _infer_source("labels_csv", pipeline_cli, pipeline_env, pipeline_yaml, "default"),
    }

    hash_blob = {
        "pipeline": pipeline_cfg.model_dump(mode="json"),
        "train": train_cfg.model_dump(mode="json"),
        "datasets": manifest_cfg.model_dump(mode="json"),
    }
    config_hash = hashlib.sha256(json.dumps(hash_blob, sort_keys=True).encode("utf-8")).hexdigest()

    return RuntimeConfigBundle(
        pipeline=pipeline_cfg,
        train=train_cfg,
        dataset_manifest=manifest_cfg,
        config_hash=config_hash,
        config_paths={
            "pipeline": pipeline_path,
            "train": train_path,
            "datasets": datasets_path,
        },
        source_map=source_map,
    )


def to_namespace(bundle: RuntimeConfigBundle) -> argparse.Namespace:
    """Create argparse-compatible namespace from resolved config."""
    payload = bundle.pipeline.model_dump(mode="python")
    payload["_config_hash"] = bundle.config_hash
    payload["_config_paths"] = bundle.config_paths
    payload["_config_source_map"] = bundle.source_map
    payload["_train_config"] = bundle.train.model_dump(mode="python")
    payload["_dataset_aliases"] = bundle.dataset_manifest.aliases
    return argparse.Namespace(**payload)
