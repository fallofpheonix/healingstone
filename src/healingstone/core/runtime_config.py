"""Minimal runtime config loader for CLI/test compatibility."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict

import yaml


@dataclass
class RuntimeConfigBundle:
    pipeline: SimpleNamespace
    train: Dict[str, Any]
    datasets: Dict[str, Any]
    resolved: Dict[str, Any]
    source_map: Dict[str, str]
    config_paths: Dict[str, str]
    config_hash: str


@dataclass(frozen=True)
class FieldSpec:
    default: Any
    cast: Callable[[Any], Any]
    env: str | None = None


def _identity(value: Any) -> Any:
    return value


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _parse_choice(choices: set[str]) -> Callable[[Any], str]:
    def _cast(value: Any) -> str:
        text = str(value)
        if text not in choices:
            allowed = ", ".join(sorted(choices))
            raise ValueError(f"Expected one of {{{allowed}}}, got {value!r}")
        return text

    return _cast


FIELD_SPECS: Dict[str, FieldSpec] = {
    "config_version": FieldSpec(1, int),
    "dataset_alias": FieldSpec("3d", str, "HEALINGSTONE_DATASET_ALIAS"),
    "data_dir": FieldSpec(None, _identity, "HEALINGSTONE_DATA_DIR"),
    "output_dir": FieldSpec("artifacts", _identity, "HEALINGSTONE_OUTPUT_DIR"),
    "labels_csv": FieldSpec(None, _identity, "HEALINGSTONE_LABELS_CSV"),
    "allow_overwrite_run": FieldSpec(False, _parse_bool, "HEALINGSTONE_ALLOW_OVERWRITE_RUN"),
    "sample_points": FieldSpec(40000, int),
    "voxel_size": FieldSpec(0.01, float),
    "normal_radius": FieldSpec(0.04, float),
    "normal_max_nn": FieldSpec(64, int),
    "outlier_nb_neighbors": FieldSpec(24, int),
    "outlier_std_ratio": FieldSpec(1.8, float),
    "k_neighbors": FieldSpec(24, int),
    "fpfh_radius": FieldSpec(0.06, float),
    "fpfh_max_nn": FieldSpec(100, int),
    "dbscan_eps": FieldSpec(0.04, float),
    "dbscan_min_samples": FieldSpec(24, int),
    "n_keypoints": FieldSpec(256, int),
    "candidate_top_k": FieldSpec(4, int),
    "align_top_n": FieldSpec(10, int),
    "label_suggestions_top_n": FieldSpec(50, int),
    "threshold_objective": FieldSpec("accuracy", _parse_choice({"accuracy", "f1"})),
    "min_match_accuracy": FieldSpec(0.0, float),
    "min_required_accuracy": FieldSpec(0.80, float),
    "evaluation_split": FieldSpec("test", _parse_choice({"train", "validation", "test"})),
    "augment_rotations": FieldSpec(False, _parse_bool),
    "augment_count": FieldSpec(2, int),
    "seed": FieldSpec(42, int, "HEALINGSTONE_SEED"),
    "device": FieldSpec("cpu", _parse_choice({"cpu", "cuda"}), "HEALINGSTONE_DEVICE"),
}


def _load_yaml(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload if isinstance(payload, dict) else {}


def _normalize_path(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _resolve_field(args: Any, name: str, pipeline_cfg: Dict[str, Any]) -> tuple[Any, str]:
    spec = FIELD_SPECS[name]
    cli_value = getattr(args, name, None)
    if cli_value is not None:
        return spec.cast(cli_value), "cli"

    if spec.env:
        env_value = os.environ.get(spec.env)
        if env_value is not None:
            return spec.cast(env_value), "env"

    yaml_value = pipeline_cfg.get(name)
    if yaml_value is not None:
        return spec.cast(yaml_value), "yaml"

    return spec.cast(spec.default), "default"


def _compute_config_hash(
    pipeline_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    datasets_cfg: Dict[str, Any],
    resolved: Dict[str, Any],
) -> str:
    non_semantic_fields = {"output_dir", "allow_overwrite_run"}
    payload = {
        "pipeline": pipeline_cfg,
        "train": train_cfg,
        "datasets": datasets_cfg,
        "resolved": {
            key: _normalize_path(value)
            if key in {"config", "train_config", "dataset_manifest", "data_dir", "output_dir", "labels_csv"}
            else value
            for key, value in resolved.items()
            if not key.startswith("_") and key not in non_semantic_fields
        },
    }
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _validate_pipeline_cfg(pipeline_cfg: Dict[str, Any]) -> None:
    unknown = sorted(set(pipeline_cfg) - set(FIELD_SPECS))
    if unknown:
        raise ValueError(f"Unknown pipeline config keys: {unknown}")

    for field_name, raw_value in pipeline_cfg.items():
        if raw_value is None:
            continue
        spec = FIELD_SPECS[field_name]
        try:
            spec.cast(raw_value)
        except Exception as exc:
            raise ValueError(f"Invalid value for pipeline config field '{field_name}': {raw_value!r}") from exc


def build_runtime_config(args: Any) -> RuntimeConfigBundle:
    """Resolve runtime config with CLI > env > YAML precedence."""
    pipeline_cfg = _load_yaml(getattr(args, "config", None))
    train_cfg = _load_yaml(getattr(args, "train_config", None))
    datasets_cfg = _load_yaml(getattr(args, "dataset_manifest", None))

    config_version = int(pipeline_cfg.get("config_version", FIELD_SPECS["config_version"].default))
    if config_version != 1:
        raise ValueError(f"Unsupported config_version={config_version}")
    _validate_pipeline_cfg(pipeline_cfg)

    resolved = dict(vars(args))
    source_map: Dict[str, str] = {}
    for field_name in FIELD_SPECS:
        value, source = _resolve_field(args, field_name, pipeline_cfg)
        resolved[field_name] = value
        source_map[field_name] = source

    config_paths = {
        "config": _normalize_path(getattr(args, "config", None)) or "",
        "train_config": _normalize_path(getattr(args, "train_config", None)) or "",
        "dataset_manifest": _normalize_path(getattr(args, "dataset_manifest", None)) or "",
    }
    config_hash = _compute_config_hash(pipeline_cfg, train_cfg, datasets_cfg, resolved)

    pipeline_ns = SimpleNamespace(
        seed=resolved["seed"],
        config_version=resolved["config_version"],
    )
    return RuntimeConfigBundle(
        pipeline=pipeline_ns,
        train=train_cfg,
        datasets=datasets_cfg,
        resolved=resolved,
        source_map=source_map,
        config_paths=config_paths,
        config_hash=config_hash,
    )


def to_namespace(bundle: RuntimeConfigBundle) -> Any:
    """Flatten the resolved config into an argparse-like namespace."""
    return SimpleNamespace(**bundle.resolved)
