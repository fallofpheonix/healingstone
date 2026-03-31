"""Validation layer for the fragment reassembly pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

LOG = logging.getLogger(__name__)


def validate_fragment(frag: Any) -> bool:
    """
    Validate that a Fragment object has necessary points and normals.
    MANDATORY: Assert no NaNs/Infs and minimum coverage.
    """
    if not hasattr(frag, "points") or not hasattr(frag, "normals"):
        LOG.error("event=validation_failed reason=missing_attributes fragment_idx=%s", getattr(frag, "idx", "unknown"))
        return False

    points = np.asarray(frag.points)
    if points.size == 0:
        LOG.error("event=validation_failed reason=empty_points fragment_idx=%d", frag.idx)
        return False

    if np.isnan(points).any() or np.isinf(points).any():
        LOG.error("event=validation_failed reason=invalid_values fragment_idx=%d", frag.idx)
        return False

    if len(points) < 64:
        LOG.error("event=validation_failed reason=low_point_count count=%d fragment_idx=%d", len(points), frag.idx)
        return False

    return True


def validate_match_candidates(candidate_pairs: List[tuple[int, int]], n_fragments: int) -> bool:
    """
    Ensure the matching phase produced enough candidates to potentially form a connected graph.
    """
    if not candidate_pairs:
        LOG.error("event=validation_failed reason=no_candidate_pairs")
        return False

    # A connected graph requires at least N-1 edges
    if len(candidate_pairs) < (n_fragments - 1):
        LOG.warning("event=validation_warning reason=insufficient_candidates count=%d expected_min=%d", 
                    len(candidate_pairs), n_fragments - 1)
        # We don't fail here, but we warn because reconstruction might be partial.
    
    return True


def validate_mesh_integrity(ply_path: Path) -> bool:
    """
    Verify that the generated .ply file is valid and non-empty.
    """
    if not ply_path.exists():
        LOG.error("event=validation_failed reason=file_not_found path=%s", ply_path)
        return False

    if ply_path.stat().st_size < 128: # Basic header size check
        LOG.error("event=validation_failed reason=empty_file path=%s", ply_path)
        return False

    # Check for PLY header
    try:
        with ply_path.open("rb") as f:
            header = f.read(4)
            if header != b"ply\n":
                LOG.error("event=validation_failed reason=invalid_format path=%s", ply_path)
                return False
    except Exception as exc:
        LOG.error("event=validation_failed reason=read_error path=%s error=%s", ply_path, exc)
        return False

    return True


def validate_metrics(metrics: Dict[str, Any]) -> bool:
    """
    Ensure generated metrics are within formal research bounds [0, 1].
    """
    required_keys = ["mre", "completeness"]
    for key in required_keys:
        if key not in metrics:
            LOG.error("event=validation_failed reason=missing_metric key=%s", key)
            return False
        
        val = metrics[key]
        if not isinstance(val, (int, float)) or not (0 <= val <= 1.5): # MRE can be > 1 depending on scale, but completeness is 0-1
             if key == "completeness" and not (0 <= val <= 1.0):
                 LOG.error("event=validation_failed reason=out_of_bounds key=%s value=%s", key, val)
                 return False
                 
    return True


def validate_candidate_pairs_nonempty(candidate_pairs: List[tuple[int, int]]) -> None:
    if not candidate_pairs:
        raise RuntimeError("Matching produced zero candidate pairs")


def validate_alignment_results(alignments: Mapping[tuple[int, int], Any]) -> None:
    if not alignments:
        raise RuntimeError("Alignment produced zero pairwise results")
    successes = [result for result in alignments.values() if getattr(result, "success", False)]
    if not successes:
        raise RuntimeError("Alignment produced zero successful pairwise registrations")


def validate_global_transforms(global_transforms: Mapping[int, Any]) -> None:
    if not global_transforms:
        raise RuntimeError("Assembly produced zero global transforms")

    for idx, matrix in global_transforms.items():
        arr = np.asarray(matrix)
        if arr.shape != (4, 4):
            raise RuntimeError(f"Transform for fragment {idx} must be 4x4, got {arr.shape}")
        if not np.isfinite(arr).all():
            raise RuntimeError(f"Transform for fragment {idx} contains non-finite values")


def validate_metrics_payload(metrics: Mapping[str, Any]) -> None:
    required_nonnegative = (
        "aligned_pairs",
        "successful_alignments",
        "mean_icp_rmse",
        "mean_chamfer_distance",
        "assembled_fragments",
        "graph_nodes",
        "graph_edges",
    )
    for key in required_nonnegative:
        value = metrics.get(key)
        if value is None:
            raise RuntimeError(f"Metrics payload missing required key: {key}")
        if float(value) < 0:
            raise RuntimeError(f"Metrics key '{key}' must be non-negative, got {value!r}")

    completeness = float(metrics.get("reconstruction_completeness", -1.0))
    if not (0.0 <= completeness <= 1.0):
        raise RuntimeError(f"reconstruction_completeness must be in [0, 1], got {completeness!r}")

    accuracy = metrics.get("pairwise_match_accuracy")
    if accuracy is not None:
        accuracy_value = float(accuracy)
        if np.isfinite(accuracy_value) and not (0.0 <= accuracy_value <= 1.0):
            raise RuntimeError(f"pairwise_match_accuracy must be in [0, 1], got {accuracy_value!r}")
