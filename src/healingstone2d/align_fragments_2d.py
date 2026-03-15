"""Rigid transform estimation for 2D fragment alignment."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors  # type: ignore[import]

from .edge_detection import EdgeBundle
from .preprocess_2d import Fragment2D

LOG = logging.getLogger(__name__)

_RANSAC_MAX_ITERATIONS = 1000
_RANSAC_INLIER_DISTANCE = 5.0  # pixels


@dataclass
class AlignmentResult2D:
    """Alignment output for one 2D pair."""

    i: int
    j: int
    transform: np.ndarray  # 3x3 homogeneous rigid-body matrix
    inlier_ratio: float
    rmse: float
    success: bool


def _kabsch_2d(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Compute optimal rigid 2D transform (rotation + translation) via SVD (Kabsch).

    Args:
        src: (N, 2) source points
        dst: (N, 2) destination points

    Returns:
        (3, 3) homogeneous transform matrix mapping src → dst
    """
    c_src = src.mean(axis=0)
    c_dst = dst.mean(axis=0)
    s = src - c_src
    d = dst - c_dst
    H = s.T @ d
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Ensure proper rotation (det = +1).
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = c_dst - R @ c_src
    T = np.eye(3, dtype=np.float32)
    T[:2, :2] = R
    T[:2, 2] = t
    return T


def _apply_transform_2d(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 3x3 homogeneous 2D transform to (N, 2) points."""
    n = points.shape[0]
    homo = np.hstack([points, np.ones((n, 1), dtype=np.float32)])
    return (homo @ T.T)[:, :2]


def _ransac_rigid_2d(
    src: np.ndarray,
    dst: np.ndarray,
    max_iter: int = _RANSAC_MAX_ITERATIONS,
    inlier_dist: float = _RANSAC_INLIER_DISTANCE,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """RANSAC rigid 2D alignment between two point sets.

    Returns:
        best_T: (3, 3) transform
        inlier_mask: (N,) boolean mask
    """
    rng = np.random.default_rng(seed)
    n = src.shape[0]
    best_T = np.eye(3, dtype=np.float32)
    best_inliers = np.zeros(n, dtype=bool)

    if n < 3:
        return best_T, best_inliers

    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")

    for _ in range(max_iter):
        sample_idx = rng.choice(n, size=3, replace=False)
        T = _kabsch_2d(src[sample_idx], dst[sample_idx])
        aligned = _apply_transform_2d(src, T)
        nn.fit(dst)
        dists, _ = nn.kneighbors(aligned)
        inliers = (dists.flatten() < inlier_dist)
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_T = T
            if inliers.sum() > 0.7 * n:
                break

    # Refine on all inliers.
    if best_inliers.sum() >= 3:
        best_T = _kabsch_2d(src[best_inliers], dst[best_inliers])
        aligned = _apply_transform_2d(src, best_T)
        nn.fit(dst)
        dists, _ = nn.kneighbors(aligned)
        best_inliers = (dists.flatten() < inlier_dist)

    return best_T, best_inliers


def _rmse(src: np.ndarray, dst: np.ndarray, T: np.ndarray, inliers: np.ndarray) -> float:
    if inliers.sum() == 0:
        return float("inf")
    aligned = _apply_transform_2d(src[inliers], T)
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(dst)
    dists, _ = nn.kneighbors(aligned)
    return float(np.sqrt(np.mean(dists.flatten() ** 2)))


def align_pair_2d(
    frag_i: Fragment2D,
    edge_i: EdgeBundle,
    frag_j: Fragment2D,
    edge_j: EdgeBundle,
    score_prior: float,
) -> AlignmentResult2D:
    """Estimate rigid 2D alignment between fragment i and fragment j.

    Uses RANSAC on boundary point correspondences (nearest-neighbour seeding).
    """
    bp_i = edge_i.boundary_points
    bp_j = edge_j.boundary_points

    if bp_i.shape[0] < 5 or bp_j.shape[0] < 5:
        LOG.warning("Fragment %s or %s has too few boundary points for alignment", frag_i.name, frag_j.name)
        return AlignmentResult2D(
            i=frag_i.idx,
            j=frag_j.idx,
            transform=np.eye(3, dtype=np.float32),
            inlier_ratio=0.0,
            rmse=float("inf"),
            success=False,
        )

    # Subsample for efficiency.
    rng = np.random.default_rng(42)
    max_pts = 512
    if bp_i.shape[0] > max_pts:
        bp_i = bp_i[rng.choice(bp_i.shape[0], size=max_pts, replace=False)]
    if bp_j.shape[0] > max_pts:
        bp_j = bp_j[rng.choice(bp_j.shape[0], size=max_pts, replace=False)]

    # Seed correspondences via nearest-neighbour matching.
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(bp_j)
    _, nn_idx = nn.kneighbors(bp_i)
    src = bp_i
    dst = bp_j[nn_idx.flatten()]

    T, inliers = _ransac_rigid_2d(src, dst, seed=42)
    inlier_ratio = float(inliers.sum()) / max(1, len(inliers))
    rmse = _rmse(src, dst, T, inliers)

    success = inlier_ratio >= 0.2 and np.isfinite(rmse)
    LOG.debug(
        "Align 2D pair (%s, %s): inlier_ratio=%.3f rmse=%.3f success=%s",
        frag_i.name,
        frag_j.name,
        inlier_ratio,
        rmse if np.isfinite(rmse) else -1.0,
        success,
    )
    return AlignmentResult2D(
        i=frag_i.idx,
        j=frag_j.idx,
        transform=T.astype(np.float32),
        inlier_ratio=inlier_ratio,
        rmse=rmse,
        success=success,
    )


def align_candidate_pairs_2d(
    fragments: List[Fragment2D],
    edges: Dict[int, EdgeBundle],
    candidate_pairs: List[Tuple[int, int]],
    pair_scores: Dict[Tuple[int, int], float],
) -> Dict[Tuple[int, int], AlignmentResult2D]:
    """Run pairwise alignment for all candidate pairs."""
    by_idx = {f.idx: f for f in fragments}
    results: Dict[Tuple[int, int], AlignmentResult2D] = {}

    for i, j in candidate_pairs:
        score = float(pair_scores.get((i, j), pair_scores.get((j, i), 0.0)))
        result = align_pair_2d(
            frag_i=by_idx[i],
            edge_i=edges[i],
            frag_j=by_idx[j],
            edge_j=edges[j],
            score_prior=score,
        )
        results[(i, j)] = result

    successful = sum(1 for r in results.values() if r.success)
    LOG.info("2D alignment: %d/%d pairs succeeded", successful, len(results))
    return results
