"""Schema validation for alignment metrics outputs."""

from __future__ import annotations

from typing import Any, Dict

METRICS_SCHEMA_VERSION = 1
REQUIRED_METRICS: Dict[str, type] = {
    "pairwise_match_accuracy": float,
    "min_required_accuracy": float,
    "evaluation_split": str,
    "aligned_pairs": int,
    "successful_alignments": int,
    "mean_icp_rmse": float,
    "mean_chamfer_distance": float,
    "reconstruction_completeness": float,
    "assembled_fragments": int,
    "graph_nodes": int,
    "graph_edges": int,
}


class MetricsSchemaError(ValueError):
    pass


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_metrics_schema(metrics: Dict[str, Any]) -> None:
    if not isinstance(metrics, dict):
        raise MetricsSchemaError("metrics must be a dictionary")

    missing = [k for k in REQUIRED_METRICS if k not in metrics]
    if missing:
        raise MetricsSchemaError(f"missing required metrics keys: {missing}")

    for key, expected in REQUIRED_METRICS.items():
        val = metrics[key]
        if expected is int:
            if not isinstance(val, int) or isinstance(val, bool):
                raise MetricsSchemaError(f"metrics['{key}'] must be int, got {type(val).__name__}")
        elif expected is float:
            if not _is_number(val):
                raise MetricsSchemaError(f"metrics['{key}'] must be numeric, got {type(val).__name__}")
        elif expected is str:
            if not isinstance(val, str):
                raise MetricsSchemaError(f"metrics['{key}'] must be str, got {type(val).__name__}")


def attach_schema_version(report: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(report)
    out["metrics_schema_version"] = METRICS_SCHEMA_VERSION
    return out
