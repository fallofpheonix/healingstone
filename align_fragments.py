"""Pairwise geometric alignment using RANSAC + ICP."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

from features import FeatureBundle
from preprocess import Fragment

LOG = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """Alignment output for one pair."""

    i: int
    j: int
    transform_ij: np.ndarray
    score_prior: float
    fitness: float
    inlier_rmse: float
    chamfer: float
    success: bool


def _build_break_pcd(fragment: Fragment, features: FeatureBundle) -> o3d.geometry.PointCloud:
    max_len = min(features.break_mask.shape[0], fragment.points.shape[0], fragment.normals.shape[0])
    idx = np.where(features.break_mask[:max_len])[0]
    if idx.size < 64:
        idx = np.arange(fragment.points.shape[0])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fragment.points[idx])
    pcd.normals = o3d.utility.Vector3dVector(fragment.normals[idx])
    return pcd


def _build_feature_from_masked_fpfh(fpfh: np.ndarray, mask: np.ndarray) -> o3d.pipelines.registration.Feature:
    max_len = min(mask.shape[0], fpfh.shape[0])
    idx = np.where(mask[:max_len])[0]
    if idx.size < 64:
        idx = np.arange(fpfh.shape[0])
    idx = idx[idx < fpfh.shape[0]]
    data = fpfh[idx].T
    feat = o3d.pipelines.registration.Feature()
    feat.data = data.astype(np.float64)
    return feat


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homo = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float32)])
    out = (homo @ transform.T)[:, :3]
    return out.astype(np.float32)


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer distance (mean NN distance both directions)."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("inf")
    nn_ab = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(b)
    d_ab, _ = nn_ab.kneighbors(a)
    nn_ba = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(a)
    d_ba, _ = nn_ba.kneighbors(b)
    return float(np.mean(d_ab) + np.mean(d_ba))


def align_pair(
    frag_i: Fragment,
    feat_i: FeatureBundle,
    frag_j: Fragment,
    feat_j: FeatureBundle,
    score_prior: float,
    voxel_size: float,
) -> AlignmentResult:
    """Align one fragment pair with RANSAC + point-to-plane ICP."""
    src_break = _build_break_pcd(frag_i, feat_i)
    tgt_break = _build_break_pcd(frag_j, feat_j)

    src_f = _build_feature_from_masked_fpfh(feat_i.fpfh, feat_i.break_mask)
    tgt_f = _build_feature_from_masked_fpfh(feat_j.fpfh, feat_j.break_mask)

    dist_thresh = max(1.5 * voxel_size, 0.02)
    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
    ]

    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_break,
        tgt_break,
        src_f,
        tgt_f,
        mutual_filter=True,
        max_correspondence_distance=dist_thresh,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=checkers,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(120000, 0.999),
    )

    src_full = frag_i.to_point_cloud()
    tgt_full = frag_j.to_point_cloud()

    src_full.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(0.03, 2 * voxel_size), max_nn=48)
    )
    tgt_full.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(0.03, 2 * voxel_size), max_nn=48)
    )

    icp = o3d.pipelines.registration.registration_icp(
        src_full,
        tgt_full,
        max_correspondence_distance=max(1.0 * voxel_size, 0.015),
        init=ransac.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
    )

    transformed_src = _apply_transform(frag_i.points, icp.transformation)
    ch = chamfer_distance(transformed_src[::2], frag_j.points[::2])

    success = bool(icp.fitness > 0.02 and np.isfinite(icp.inlier_rmse))
    return AlignmentResult(
        i=frag_i.idx,
        j=frag_j.idx,
        transform_ij=np.asarray(icp.transformation, dtype=np.float32),
        score_prior=float(score_prior),
        fitness=float(icp.fitness),
        inlier_rmse=float(icp.inlier_rmse),
        chamfer=ch,
        success=success,
    )


def align_candidate_pairs(
    fragments: List[Fragment],
    features: Dict[int, FeatureBundle],
    candidate_pairs: List[Tuple[int, int]],
    pair_scores: Dict[Tuple[int, int], float],
    voxel_size: float,
    top_n: int,
) -> Dict[Tuple[int, int], AlignmentResult]:
    """Align top candidate pairs sorted by matching score."""
    ordered = sorted(candidate_pairs, key=lambda p: pair_scores.get(p, -1.0), reverse=True)
    ordered = ordered[: min(top_n, len(ordered))]

    by_idx = {f.idx: f for f in fragments}
    results: Dict[Tuple[int, int], AlignmentResult] = {}

    for rank, (i, j) in enumerate(ordered, start=1):
        LOG.info("Aligning pair %d/%d: (%d, %d)", rank, len(ordered), i, j)
        try:
            result = align_pair(
                frag_i=by_idx[i],
                feat_i=features[i],
                frag_j=by_idx[j],
                feat_j=features[j],
                score_prior=pair_scores[(i, j)],
                voxel_size=voxel_size,
            )
            results[(i, j)] = result
            LOG.info(
                "Pair (%d,%d): fitness=%.4f rmse=%.5f chamfer=%.5f success=%s",
                i,
                j,
                result.fitness,
                result.inlier_rmse,
                result.chamfer,
                result.success,
            )
        except Exception as exc:
            LOG.warning("Alignment failed for pair (%d,%d): %s", i, j, exc)

    return results
