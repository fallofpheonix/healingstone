from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import pytest

from healingstone.run_pipeline import enforce_accuracy_requirement, summarize_metrics


def test_summarize_metrics_stable() -> None:
    diagnostics = {"pairwise_match_accuracy": 0.75}
    alignments = {
        (0, 1): SimpleNamespace(success=True, inlier_rmse=0.01, chamfer=0.2),
        (1, 2): SimpleNamespace(success=False, inlier_rmse=0.02, chamfer=0.4),
    }
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2])
    g.add_edge(0, 1)
    assembly = SimpleNamespace(completeness=0.9, global_transforms={0: None, 1: None}, graph=g)

    m1 = summarize_metrics(diagnostics, alignments, assembly)
    m2 = summarize_metrics(diagnostics, alignments, assembly)

    assert m1 == m2
    assert abs(m1["mean_icp_rmse"] - 0.01) < 1e-12
    assert abs(m1["reconstruction_completeness"] - 0.9) < 1e-12


def test_accuracy_gate_passes() -> None:
    enforce_accuracy_requirement(
        metrics={"pairwise_match_accuracy": 0.81},
        min_required_accuracy=0.80,
        evaluation_split="test",
    )


def test_accuracy_gate_fails_on_non_test_split() -> None:
    with pytest.raises(RuntimeError):
        enforce_accuracy_requirement(
            metrics={"pairwise_match_accuracy": 0.99},
            min_required_accuracy=0.80,
            evaluation_split="validation",
        )


def test_accuracy_gate_fails_on_low_accuracy() -> None:
    with pytest.raises(RuntimeError):
        enforce_accuracy_requirement(
            metrics={"pairwise_match_accuracy": 0.79},
            min_required_accuracy=0.80,
            evaluation_split="test",
        )
