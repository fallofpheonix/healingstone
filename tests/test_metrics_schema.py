from __future__ import annotations

import pytest

from healingstone.core.metrics_schema import MetricsSchemaError, validate_metrics_schema


def test_metrics_schema_valid() -> None:
    metrics = {
        "pairwise_match_accuracy": 0.8,
        "min_required_accuracy": 0.8,
        "evaluation_split": "test",
        "aligned_pairs": 10,
        "successful_alignments": 8,
        "mean_icp_rmse": 0.012,
        "mean_chamfer_distance": 0.21,
        "reconstruction_completeness": 0.92,
        "assembled_fragments": 7,
        "graph_nodes": 8,
        "graph_edges": 9,
    }
    validate_metrics_schema(metrics)


def test_metrics_schema_missing_key() -> None:
    metrics = {
        "pairwise_match_accuracy": 0.8,
    }
    with pytest.raises(MetricsSchemaError):
        validate_metrics_schema(metrics)
