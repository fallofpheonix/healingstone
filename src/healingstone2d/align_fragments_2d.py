"""Rigid transform estimation for 2D fragment alignment.

Provides:
- AlignmentResult2D         – dataclass holding alignment outcome
- estimate_rigid_transform() – estimate rotation + translation from contours
- align_pair_2d()           – full alignment for one fragment pair
- align_candidate_pairs_2d() – batch alignment
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from healingstone2d.match_fragments_2d import FragmentMatch

LOG = logging.getLogger(__name__)

# Maximum iterations for the point-matching loop.
_MAX_ICP_ITER = 50
# Convergence tolerance (pixels).
_ICP_TOL = 0.5


@dataclass
class AlignmentResult2D:
    """Outcome of a 2D rigid alignment attempt."""

    frag_idx_i: int
    frag_idx_j: int
    transform: np.ndarray   # (3, 3) homogeneous transform matrix (float64)
    angle_deg: float        # estimated rotation angle in degrees
    tx: float               # translation x (pixels)
    ty: float               # translation y (pixels)
    score: float            # similarity score from matching stage
    rmse: float             # final contour-point RMSE after alignment
    success: bool           # whether the alignment converged


def _contour_to_points(contour: np.ndarray) -> np.ndarray:
    """Convert an OpenCV contour to an (N, 2) float64 point array."""
    return contour.squeeze().astype(np.float64)


def _centroid(pts: np.ndarray) -> np.ndarray:
    """Return the centroid of a point set, shape (2,)."""
    return pts.mean(axis=0)


def _sample_contour(pts: np.ndarray, n: int = 200) -> np.ndarray:
    """Sub-sample or up-sample a contour to exactly *n* points via interpolation."""
    if len(pts) == 0:
        return pts
    if len(pts) == n:
        return pts

    # Compute cumulative arc-length parameterisation.
    diffs = np.diff(pts, axis=0)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])
    total = arc[-1]
    if total < 1e-10:
        return np.tile(pts[0], (n, 1))

    t_new = np.linspace(0.0, total, n)
    resampled = np.column_stack([
        np.interp(t_new, arc, pts[:, 0]),
        np.interp(t_new, arc, pts[:, 1]),
    ])
    return resampled


def _nearest_neighbours(src: np.ndarray, tgt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For each point in *src* find the nearest point in *tgt*.

    Returns
    -------
    indices : np.ndarray, shape (N,)
    distances : np.ndarray, shape (N,)
    """
    # Vectorised nearest-neighbour (O(N*M) – acceptable for small contours).
    diff = src[:, np.newaxis, :] - tgt[np.newaxis, :, :]   # (N, M, 2)
    dists = np.linalg.norm(diff, axis=2)                    # (N, M)
    indices = np.argmin(dists, axis=1)
    distances = dists[np.arange(len(src)), indices]
    return indices, distances


def _icp_2d(
    src: np.ndarray,
    tgt: np.ndarray,
    max_iter: int = _MAX_ICP_ITER,
    tol: float = _ICP_TOL,
) -> Tuple[float, float, float, float]:
    """Iterative Closest Point for 2D point sets.

    Estimates the rigid transform ``(angle_rad, tx, ty)`` that minimises the
    mean squared distance from *src* to *tgt*.

    Parameters
    ----------
    src, tgt:
        Point arrays of shape ``(N, 2)`` and ``(M, 2)``.

    Returns
    -------
    (angle_rad, tx, ty, final_rmse) : Tuple[float, float, float, float]
    """
    src = src.copy()
    angle_total = 0.0
    t_total = np.zeros(2, dtype=np.float64)

    prev_rmse = np.inf

    for _ in range(max_iter):
        nn_idx, nn_dist = _nearest_neighbours(src, tgt)
        rmse = float(np.sqrt(np.mean(nn_dist ** 2)))

        matched_tgt = tgt[nn_idx]

        # Compute optimal rotation + translation using SVD.
        mu_src = src.mean(axis=0)
        mu_tgt = matched_tgt.mean(axis=0)

        src_c = src - mu_src
        tgt_c = matched_tgt - mu_tgt

        H = src_c.T @ tgt_c
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (det = +1).
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = mu_tgt - R @ mu_src
        angle_rad = float(np.arctan2(R[1, 0], R[0, 0]))

        # Apply to source.
        src = (R @ src.T).T + t

        angle_total += angle_rad
        t_total += t

        if abs(prev_rmse - rmse) < tol:
            break
        prev_rmse = rmse

    final_rmse = prev_rmse
    return float(angle_total), float(t_total[0]), float(t_total[1]), float(final_rmse)


def estimate_rigid_transform(
    contour_i: np.ndarray,
    contour_j: np.ndarray,
    n_points: int = 200,
) -> Tuple[np.ndarray, float, float, float, float]:
    """Estimate the rigid transform aligning *contour_i* onto *contour_j*.

    Parameters
    ----------
    contour_i, contour_j:
        OpenCV contour arrays, shape ``(N, 1, 2)``.
    n_points:
        Number of points to resample each contour to before ICP.

    Returns
    -------
    (transform, angle_deg, tx, ty, rmse)
        *transform* is a ``(3, 3)`` homogeneous matrix.
    """
    pts_i = _sample_contour(_contour_to_points(contour_i), n=n_points)
    pts_j = _sample_contour(_contour_to_points(contour_j), n=n_points)

    if len(pts_i) < 3 or len(pts_j) < 3:
        identity = np.eye(3, dtype=np.float64)
        return identity, 0.0, 0.0, 0.0, float("inf")

    angle_rad, tx, ty, rmse = _icp_2d(pts_i, pts_j)

    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    transform = np.array([
        [cos_a, -sin_a, tx],
        [sin_a,  cos_a, ty],
        [0.0,    0.0,   1.0],
    ], dtype=np.float64)

    return transform, float(np.degrees(angle_rad)), float(tx), float(ty), float(rmse)


def align_pair_2d(
    frag_i,
    frag_j,
    match: FragmentMatch,
    n_points: int = 200,
    rmse_threshold: float = 20.0,
) -> AlignmentResult2D:
    """Align one pair of 2D fragments.

    Parameters
    ----------
    frag_i, frag_j:
        :class:`~healingstone2d.preprocess_2d.Fragment2D` objects.
    match:
        Corresponding :class:`~healingstone2d.match_fragments_2d.FragmentMatch`.
    n_points:
        Contour resampling size.
    rmse_threshold:
        Maximum RMSE (pixels) for an alignment to be considered successful.

    Returns
    -------
    AlignmentResult2D
    """
    ci = frag_i.main_contour
    cj = frag_j.main_contour

    if ci is None or cj is None or len(ci) < 5 or len(cj) < 5:
        return AlignmentResult2D(
            frag_idx_i=frag_i.idx,
            frag_idx_j=frag_j.idx,
            transform=np.eye(3, dtype=np.float64),
            angle_deg=0.0, tx=0.0, ty=0.0,
            score=match.score, rmse=float("inf"), success=False,
        )

    transform, angle_deg, tx, ty, rmse = estimate_rigid_transform(ci, cj, n_points=n_points)
    success = rmse < rmse_threshold

    LOG.debug(
        "Align pair (%d, %d): angle=%.1f°, tx=%.1f, ty=%.1f, rmse=%.2f, success=%s",
        frag_i.idx, frag_j.idx, angle_deg, tx, ty, rmse, success,
    )
    return AlignmentResult2D(
        frag_idx_i=frag_i.idx,
        frag_idx_j=frag_j.idx,
        transform=transform,
        angle_deg=angle_deg, tx=tx, ty=ty,
        score=match.score, rmse=rmse, success=success,
    )


def align_candidate_pairs_2d(
    fragments: list,
    candidate_matches: List[FragmentMatch],
    n_points: int = 200,
    rmse_threshold: float = 20.0,
    top_n: int = 20,
) -> Dict[Tuple[int, int], AlignmentResult2D]:
    """Align the top-N candidate pairs.

    Parameters
    ----------
    fragments:
        List of :class:`~healingstone2d.preprocess_2d.Fragment2D` (indexed by ``idx``).
    candidate_matches:
        Candidate pairs sorted by score (descending).
    n_points:
        Contour resampling size for ICP.
    rmse_threshold:
        RMSE threshold for alignment success classification.
    top_n:
        Maximum number of pairs to attempt.

    Returns
    -------
    Dict[Tuple[int, int], AlignmentResult2D]
        Mapping from ``(frag_idx_i, frag_idx_j)`` to alignment result.
    """
    frag_by_idx = {f.idx: f for f in fragments}
    results: Dict[Tuple[int, int], AlignmentResult2D] = {}

    for match in candidate_matches[:top_n]:
        fi = match.frag_idx_i
        fj = match.frag_idx_j
        if fi not in frag_by_idx or fj not in frag_by_idx:
            continue
        key = (min(fi, fj), max(fi, fj))
        if key in results:
            continue

        result = align_pair_2d(
            frag_by_idx[fi], frag_by_idx[fj], match,
            n_points=n_points, rmse_threshold=rmse_threshold,
        )
        results[key] = result

    n_success = sum(1 for r in results.values() if r.success)
    LOG.info(
        "2D alignment: %d pairs attempted, %d successful", len(results), n_success
    )
    return results
