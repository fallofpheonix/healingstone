"""Rigid transform estimation for 2D fragment alignment.

Provides:
- AlignmentResult2D         – dataclass holding alignment outcome
- estimate_rigid_transform() – estimate rotation + translation from contours
- align_pair_2d()           – full alignment for one fragment pair
- align_candidate_pairs_2d() – batch alignment
"""
"""Rigid transform estimation for 2D fragment alignment."""

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
from sklearn.neighbors import NearestNeighbors  # type: ignore[import]

from .edge_detection import EdgeBundle
from .preprocess_2d import Fragment2D

LOG = logging.getLogger(__name__)

_RANSAC_MAX_ITERATIONS = 1000
_RANSAC_INLIER_DISTANCE = 5.0  # pixels


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
    # Track accumulated rotation so we can compose translations correctly.
    R_total = np.eye(2, dtype=np.float64)

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
        U, S, Vt = np.linalg.svd(H)
        del S  # singular values not needed; named for readability
        R = Vt.T @ U.T

        # Ensure proper rotation (det = +1).
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = mu_tgt - R @ mu_src
        angle_rad = float(np.arctan2(R[1, 0], R[0, 0]))

        # Apply to source.
        src = (R @ src.T).T + t

        # Accumulate total transform. In 2D, angles add, but translations must
        # be composed through the current rotation.
        angle_total += angle_rad
        R_total = R @ R_total
        t_total = R @ t_total + t

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
    nn.fit(dst)

    for _ in range(max_iter):
        sample_idx = rng.choice(n, size=3, replace=False)
        T = _kabsch_2d(src[sample_idx], dst[sample_idx])
        aligned = _apply_transform_2d(src, T)
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
        "Align 2D pair (%s, %s): score_prior=%.3f inlier_ratio=%.3f rmse=%.3f success=%s",
        frag_i.name,
        frag_j.name,
        score_prior,
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
        key = (fi, fj)
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
