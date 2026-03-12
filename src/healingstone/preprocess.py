"""Data loading and preprocessing for fragmented mesh reconstruction."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import open3d as o3d
import torch

LOG = logging.getLogger(__name__)


@dataclass
class Fragment:
    """Container for one fragment point cloud and metadata."""

    idx: int
    name: str
    path: Path
    points: np.ndarray
    normals: np.ndarray

    def to_point_cloud(self) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.normals = o3d.utility.Vector3dVector(self.normals)
        return pcd


def set_deterministic_seed(seed: int = 42) -> None:
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def discover_fragment_files(data_dir: Path) -> List[Path]:
    """Discover supported mesh files recursively."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    patterns = ("*.ply", "*.PLY", "*.obj", "*.OBJ")
    files = []
    for pattern in patterns:
        files.extend(data_dir.rglob(pattern))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No .PLY/.OBJ fragments found in: {data_dir}")
    return files


def _load_fragment_geometry(path: Path, sample_points: int) -> o3d.geometry.PointCloud:
    """Load file as mesh or point cloud and return sampled point cloud."""
    ext = path.suffix.lower()

    if ext == ".obj":
        mesh = o3d.io.read_triangle_mesh(str(path))
        if mesh.is_empty() or len(mesh.vertices) == 0:
            raise ValueError(f"Empty OBJ mesh: {path}")
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=sample_points)
    else:
        mesh = o3d.io.read_triangle_mesh(str(path))
        if not mesh.is_empty() and len(mesh.vertices) > 0:
            mesh.compute_vertex_normals()
            pcd = mesh.sample_points_uniformly(number_of_points=sample_points)
        else:
            pcd = o3d.io.read_point_cloud(str(path))
            if pcd.is_empty() or len(pcd.points) == 0:
                raise ValueError(f"Cannot load PLY as mesh or point cloud: {path}")

    return pcd


def _normalize_points(points: np.ndarray) -> np.ndarray:
    """Center and scale to unit sphere."""
    centered = points - points.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale <= 1e-12:
        raise ValueError("Degenerate fragment (zero scale)")
    return centered / scale


def preprocess_fragment(
    idx: int,
    path: Path,
    sample_points: int,
    voxel_size: float,
    normal_radius: float,
    normal_max_nn: int,
    outlier_nb_neighbors: int,
    outlier_std_ratio: float,
) -> Fragment:
    """Load and preprocess one fragment."""
    pcd = _load_fragment_geometry(path, sample_points=sample_points)

    if len(pcd.points) < 64:
        raise ValueError(f"Too few points in fragment: {path}")

    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=outlier_nb_neighbors,
        std_ratio=outlier_std_ratio,
    )
    if pcd.is_empty() or len(pcd.points) < 64:
        raise ValueError(f"Point cloud became empty after denoising: {path}")

    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    if pcd.is_empty() or len(pcd.points) < 64:
        raise ValueError(f"Too few points after voxel downsampling: {path}")

    points = np.asarray(pcd.points, dtype=np.float64)
    points = _normalize_points(points)
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=normal_max_nn,
        )
    )
    pcd.orient_normals_consistent_tangent_plane(k=min(30, len(pcd.points) - 1))

    normals = np.asarray(pcd.normals, dtype=np.float64)
    if normals.shape[0] != points.shape[0]:
        raise ValueError(f"Normal estimation failed for: {path}")

    return Fragment(
        idx=idx,
        name=path.stem,
        path=path,
        points=points.astype(np.float32),
        normals=normals.astype(np.float32),
    )


def load_and_preprocess_fragments(
    data_dir: Path,
    sample_points: int = 40000,
    voxel_size: float = 0.01,
    normal_radius: float = 0.04,
    normal_max_nn: int = 64,
    outlier_nb_neighbors: int = 24,
    outlier_std_ratio: float = 1.8,
) -> List[Fragment]:
    """Load all fragments from directory and preprocess robustly."""
    files = discover_fragment_files(data_dir)
    LOG.info("Discovered %d mesh fragments", len(files))

    fragments: List[Fragment] = []
    for idx, path in enumerate(files):
        try:
            frag = preprocess_fragment(
                idx=idx,
                path=path,
                sample_points=sample_points,
                voxel_size=voxel_size,
                normal_radius=normal_radius,
                normal_max_nn=normal_max_nn,
                outlier_nb_neighbors=outlier_nb_neighbors,
                outlier_std_ratio=outlier_std_ratio,
            )
            fragments.append(frag)
            LOG.info("Loaded %s: %d points", path.name, frag.points.shape[0])
        except Exception as exc:
            LOG.warning("Skipping %s due to preprocessing failure: %s", path.name, exc)

    if len(fragments) < 2:
        raise RuntimeError("Need at least 2 valid fragments after preprocessing")

    # Reindex contiguous IDs after skipping invalid files.
    for i, fragment in enumerate(fragments):
        fragment.idx = i

    return fragments
