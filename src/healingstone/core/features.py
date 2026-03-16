"""Feature extraction and break-surface detection with caching."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None  # type: ignore[assignment]

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from .preprocess import Fragment

LOG = logging.getLogger(__name__)


@dataclass
class FeatureBundle:
    """Features extracted for one fragment."""

    descriptor: np.ndarray
    break_mask: np.ndarray
    break_score: np.ndarray
    curvature: np.ndarray
    normal_var: np.ndarray
    roughness: np.ndarray
    fpfh: np.ndarray


def _to_point_cloud(points: np.ndarray, normals: np.ndarray) -> o3d.geometry.PointCloud:
    if o3d is None:
        raise ImportError(
            "open3d is required for point cloud operations. "
            "Install with: pip install open3d"
        )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def _normalize01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _otsu_threshold(x: np.ndarray, bins: int = 128) -> float:
    hist, edges = np.histogram(x, bins=bins)
    hist = hist.astype(np.float64)
    if hist.sum() <= 0:
        return float(np.mean(x))
    probs = hist / hist.sum()
    mids = 0.5 * (edges[:-1] + edges[1:])

    best_var = -1.0
    best_thr = float(np.mean(x))
    w0 = 0.0
    m0 = 0.0
    mt = float(np.sum(probs * mids))

    for i in range(bins):
        w0 += probs[i]
        if w0 <= 1e-12 or w0 >= 1.0 - 1e-12:
            continue
        m0 += probs[i] * mids[i]
        mu0 = m0 / w0
        mu1 = (mt - m0) / (1.0 - w0)
        var_between = w0 * (1.0 - w0) * (mu0 - mu1) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thr = float(mids[i])
    return best_thr


def estimate_geometry_features(points: np.ndarray, normals: np.ndarray, k_neighbors: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate curvature, normal variance, and roughness from local neighborhoods."""
    n = points.shape[0]
    k = min(k_neighbors + 1, n)
    nn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree")
    nn.fit(points)
    _, indices = nn.kneighbors(points)

    local_pts = points[indices]  # (n, k, 3)
    local_n = normals[indices]   # (n, k, 3)

    # Vectorized covariance computation
    centered = local_pts - local_pts.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', centered, centered) / max(1, k - 1)

    # Vectorized eigendecomposition (eigh returns eigenvalues in ascending order)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    
    denom = np.sum(eigvals, axis=1) + 1e-12
    curvature = (eigvals[:, 0] / denom).astype(np.float32)

    # Vectorized normal variance
    normal_var = np.mean(np.var(local_n, axis=1), axis=1).astype(np.float32)

    # Vectorized roughness
    plane_normal = eigvecs[:, :, 0]  # smallest eigenvector (n, 3)
    plane_normal /= np.linalg.norm(plane_normal, axis=1, keepdims=True) + 1e-12
    roughness = np.mean(np.abs(np.einsum('nki,ni->nk', centered, plane_normal)), axis=1).astype(np.float32)

    return curvature, normal_var, roughness


def detect_break_surface(
    points: np.ndarray,
    normals: np.ndarray,
    k_neighbors: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Detect break surfaces via curvature + normal variance + roughness + DBSCAN cleanup."""
    curvature, normal_var, roughness = estimate_geometry_features(points, normals, k_neighbors=k_neighbors)

    score = (
        0.45 * _normalize01(curvature)
        + 0.35 * _normalize01(normal_var)
        + 0.20 * _normalize01(roughness)
    ).astype(np.float32)

    threshold = _otsu_threshold(score)
    candidate = score >= threshold

    if np.count_nonzero(candidate) >= dbscan_min_samples:
        clusters = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(points[candidate]).labels_
        valid = clusters >= 0
        if np.any(valid):
            labels, counts = np.unique(clusters[valid], return_counts=True)
            best = labels[np.argmax(counts)]
            selected = np.zeros(points.shape[0], dtype=bool)
            selected_indices = np.where(candidate)[0][clusters == best]
            selected[selected_indices] = True
            candidate = selected

    if np.count_nonzero(candidate) < max(32, int(0.01 * points.shape[0])):
        topk = np.argsort(score)[::-1][: max(64, int(0.1 * points.shape[0]))]
        candidate = np.zeros(points.shape[0], dtype=bool)
        candidate[topk] = True

    features = {
        "curvature": curvature,
        "normal_var": normal_var,
        "roughness": roughness,
    }
    return candidate.astype(bool), score, features


def compute_fpfh(points: np.ndarray, normals: np.ndarray, radius: float, max_nn: int) -> np.ndarray:
    """Compute FPFH descriptors with Open3D."""
    if o3d is None:
        raise ImportError(
            "open3d is required for FPFH feature computation. "
            "Install with: pip install open3d"
        )
    pcd = _to_point_cloud(points, normals)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn),
    )
    return np.asarray(fpfh.data, dtype=np.float32).T


def build_fragment_descriptor(
    points: np.ndarray,
    normals: np.ndarray,
    break_mask: np.ndarray,
    break_score: np.ndarray,
    fpfh: np.ndarray,
    n_keypoints: int,
) -> np.ndarray:
    """Build robust fragment descriptor for matching."""
    idx = np.where(break_mask)[0]
    if idx.size < 16:
        idx = np.arange(points.shape[0])

    rng = np.random.default_rng(42)
    if idx.size > n_keypoints:
        idx = rng.choice(idx, size=n_keypoints, replace=False)

    sample_fpfh = fpfh[idx]
    sample_scores = break_score[idx]
    sample_normals = normals[idx]
    sample_pts = points[idx]

    center = sample_pts.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(sample_pts - center, axis=1)
    radius_n = radius / (np.max(radius) + 1e-12)

    mean_n = sample_normals.mean(axis=0)
    mean_n = mean_n / (np.linalg.norm(mean_n) + 1e-12)
    normal_align = np.abs(sample_normals @ mean_n)

    rad_hist, _ = np.histogram(radius_n, bins=12, range=(0, 1), density=True)
    aln_hist, _ = np.histogram(normal_align, bins=12, range=(0, 1), density=True)

    descriptor = np.concatenate(
        [
            sample_fpfh.mean(axis=0),
            sample_fpfh.std(axis=0),
            sample_fpfh.max(axis=0),
            sample_fpfh.min(axis=0),
            np.array(
                [
                    float(np.mean(sample_scores)),
                    float(np.std(sample_scores)),
                    float(np.percentile(sample_scores, 90)),
                    float(np.mean(radius_n)),
                    float(np.std(radius_n)),
                    float(np.mean(normal_align)),
                    float(np.std(normal_align)),
                    float(np.mean(break_mask.astype(np.float32))),
                ],
                dtype=np.float32,
            ),
            rad_hist.astype(np.float32),
            aln_hist.astype(np.float32),
        ]
    )
    return descriptor.astype(np.float32)


def _fragment_signature(fragment: Fragment) -> dict:
    """Stable signature for current preprocessed geometry."""
    n = int(fragment.points.shape[0])
    sample_n = min(2048, n)
    if sample_n == 0:
        return {"n_points": 0, "digest": "empty"}
    idx = np.linspace(0, n - 1, num=sample_n, dtype=np.int64)
    sample = np.hstack([fragment.points[idx], fragment.normals[idx]]).astype(np.float32, copy=False)
    digest = hashlib.md5(sample.tobytes()).hexdigest()
    return {"n_points": n, "digest": digest}


def _cache_key(path: Path, params: dict, signature: dict) -> str:
    payload = {
        "path": str(path.resolve()),
        "mtime": path.stat().st_mtime,
        "params": params,
        "signature": signature,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def _cache_paths(cache_dir: Path, fragment_path: Path, key: str) -> Tuple[Path, Path]:
    stem = fragment_path.stem
    npz_path = cache_dir / f"{stem}_{key}.npz"
    meta_path = cache_dir / f"{stem}_{key}.json"
    return npz_path, meta_path


def extract_fragment_features(
    fragment: Fragment,
    cache_dir: Path,
    k_neighbors: int = 24,
    fpfh_radius: float = 0.06,
    fpfh_max_nn: int = 100,
    dbscan_eps: float = 0.04,
    dbscan_min_samples: int = 24,
    n_keypoints: int = 256,
) -> FeatureBundle:
    """Extract features with persistent cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "k_neighbors": k_neighbors,
        "fpfh_radius": fpfh_radius,
        "fpfh_max_nn": fpfh_max_nn,
        "dbscan_eps": dbscan_eps,
        "dbscan_min_samples": dbscan_min_samples,
        "n_keypoints": n_keypoints,
    }
    signature = _fragment_signature(fragment)
    key = _cache_key(fragment.path, params, signature)
    npz_path, meta_path = _cache_paths(cache_dir, fragment.path, key)

    if npz_path.exists() and meta_path.exists():
        data = np.load(npz_path)
        n = fragment.points.shape[0]
        cache_ok = (
            data["break_mask"].shape[0] == n
            and data["break_score"].shape[0] == n
            and data["curvature"].shape[0] == n
            and data["normal_var"].shape[0] == n
            and data["roughness"].shape[0] == n
            and data["fpfh"].shape[0] == n
        )
        if cache_ok:
            LOG.info("Using cached features for %s", fragment.name)
            return FeatureBundle(
                descriptor=data["descriptor"],
                break_mask=data["break_mask"].astype(bool),
                break_score=data["break_score"],
                curvature=data["curvature"],
                normal_var=data["normal_var"],
                roughness=data["roughness"],
                fpfh=data["fpfh"],
            )
        LOG.warning("Cache mismatch for %s; recomputing features", fragment.name)

    break_mask, break_score, geom = detect_break_surface(
        fragment.points,
        fragment.normals,
        k_neighbors=k_neighbors,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
    )
    fpfh = compute_fpfh(
        fragment.points,
        fragment.normals,
        radius=fpfh_radius,
        max_nn=fpfh_max_nn,
    )
    descriptor = build_fragment_descriptor(
        fragment.points,
        fragment.normals,
        break_mask,
        break_score,
        fpfh,
        n_keypoints=n_keypoints,
    )

    np.savez_compressed(
        npz_path,
        descriptor=descriptor,
        break_mask=break_mask.astype(np.uint8),
        break_score=break_score,
        curvature=geom["curvature"],
        normal_var=geom["normal_var"],
        roughness=geom["roughness"],
        fpfh=fpfh,
    )
    meta_path.write_text(json.dumps(params, indent=2), encoding="utf-8")

    return FeatureBundle(
        descriptor=descriptor,
        break_mask=break_mask,
        break_score=break_score,
        curvature=geom["curvature"],
        normal_var=geom["normal_var"],
        roughness=geom["roughness"],
        fpfh=fpfh,
    )


def extract_all_features(
    fragments: List[Fragment],
    cache_dir: Path,
    k_neighbors: int,
    fpfh_radius: float,
    fpfh_max_nn: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    n_keypoints: int,
) -> Dict[int, FeatureBundle]:
    """Extract features for all fragments."""
    bundles: Dict[int, FeatureBundle] = {}
    for frag in fragments:
        bundles[frag.idx] = extract_fragment_features(
            fragment=frag,
            cache_dir=cache_dir,
            k_neighbors=k_neighbors,
            fpfh_radius=fpfh_radius,
            fpfh_max_nn=fpfh_max_nn,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            n_keypoints=n_keypoints,
        )
    return bundles


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Generate a random SO(3) rotation matrix."""
    q = rng.normal(size=4)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ],
        dtype=np.float32,
    )


def augment_fragment_geometry(
    points: np.ndarray,
    normals: np.ndarray,
    rng: np.random.Generator,
    noise_std: float = 0.004,
    keep_ratio: float = 0.85,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random rotation, Gaussian noise, and partial point removal."""
    rot = random_rotation_matrix(rng)
    pts = (rot @ points.T).T
    nrm = (rot @ normals.T).T

    pts = pts + rng.normal(0.0, noise_std, size=pts.shape).astype(np.float32)

    n_keep = max(64, int(keep_ratio * pts.shape[0]))
    sel = rng.choice(np.arange(pts.shape[0]), size=n_keep, replace=False)
    pts = pts[sel]
    nrm = nrm[sel]

    # Re-normalize after augmentation.
    pts = pts - pts.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-12
    pts = pts / scale
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
    return pts.astype(np.float32), nrm.astype(np.float32)


def build_augmented_descriptor(
    fragment: Fragment,
    k_neighbors: int,
    fpfh_radius: float,
    fpfh_max_nn: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    n_keypoints: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute descriptor on randomly augmented geometry."""
    pts, nrm = augment_fragment_geometry(fragment.points, fragment.normals, rng=rng)
    break_mask, break_score, _ = detect_break_surface(
        pts,
        nrm,
        k_neighbors=k_neighbors,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=max(12, dbscan_min_samples // 2),
    )
    fpfh = compute_fpfh(pts, nrm, radius=fpfh_radius, max_nn=fpfh_max_nn)
    return build_fragment_descriptor(
        pts,
        nrm,
        break_mask,
        break_score,
        fpfh,
        n_keypoints=min(n_keypoints, pts.shape[0]),
    )
