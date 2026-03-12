"""Global graph assembly and reconstruction utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import open3d as o3d

from .align_fragments import AlignmentResult
from .preprocess import Fragment

LOG = logging.getLogger(__name__)


@dataclass
class AssemblyResult:
    """Output of global assembly."""

    graph: nx.Graph
    mst: nx.Graph
    global_transforms: Dict[int, np.ndarray]
    completeness: float


def _edge_score(prior: float, fitness: float, rmse: float) -> float:
    rmse_term = float(np.exp(-rmse / 0.06)) if np.isfinite(rmse) else 0.0
    return float(0.55 * prior + 0.30 * fitness + 0.15 * rmse_term)


def build_fragment_graph(
    n_fragments: int,
    pair_scores: Dict[Tuple[int, int], float],
    alignments: Dict[Tuple[int, int], AlignmentResult],
    min_fitness: float = 0.06,
    max_rmse: float = 0.02,
    max_chamfer: float = 0.45,
) -> nx.Graph:
    """Build weighted graph with confidence edges."""
    g = nx.Graph()
    g.add_nodes_from(range(n_fragments))

    # Prefer alignment-validated edges only.
    for (i, j), a in alignments.items():
        prior = float(pair_scores.get((i, j), pair_scores.get((j, i), 0.0)))
        if not a.success:
            continue
        if not np.isfinite(a.inlier_rmse) or a.inlier_rmse > max_rmse:
            continue
        if not np.isfinite(a.chamfer) or a.chamfer > max_chamfer:
            continue
        if a.fitness < min_fitness:
            continue
        score = _edge_score(prior=prior, fitness=a.fitness, rmse=a.inlier_rmse)
        g.add_edge(i, j, score=score, prior=prior, fitness=a.fitness, rmse=a.inlier_rmse, chamfer=a.chamfer)

    # If too sparse, backfill with strongest prior-only edges.
    if g.number_of_edges() < max(1, n_fragments - 1):
        ranked = sorted(pair_scores.items(), key=lambda kv: kv[1], reverse=True)
        for (i, j), prior in ranked:
            if g.has_edge(i, j):
                continue
            g.add_edge(i, j, score=float(0.35 * prior), prior=float(prior), fitness=0.0, rmse=float("inf"), chamfer=float("inf"))
            if g.number_of_edges() >= max(1, n_fragments - 1):
                break

    return g


def _alignment_transform_lookup(
    alignments: Dict[Tuple[int, int], AlignmentResult],
) -> Dict[Tuple[int, int], np.ndarray]:
    tf = {}
    for (i, j), a in alignments.items():
        t_ij = np.asarray(a.transform_ij, dtype=np.float32)
        tf[(i, j)] = t_ij
        tf[(j, i)] = np.linalg.inv(t_ij).astype(np.float32)
    return tf


def compute_global_transforms(
    mst: nx.Graph,
    alignments: Dict[Tuple[int, int], AlignmentResult],
    root: int,
) -> Dict[int, np.ndarray]:
    """Compute transforms that map each fragment to root coordinates."""
    tf_lookup = _alignment_transform_lookup(alignments)

    transforms: Dict[int, np.ndarray] = {root: np.eye(4, dtype=np.float32)}
    queue = [root]

    while queue:
        u = queue.pop(0)
        for v in mst.neighbors(u):
            if v in transforms:
                continue

            # Need T_vu: maps v -> u.
            if (v, u) in tf_lookup:
                t_vu = tf_lookup[(v, u)]
            elif (u, v) in tf_lookup:
                t_vu = np.linalg.inv(tf_lookup[(u, v)]).astype(np.float32)
            else:
                t_vu = np.eye(4, dtype=np.float32)

            transforms[v] = transforms[u] @ t_vu
            queue.append(v)

    return transforms


def assemble_global_reconstruction(
    fragments: List[Fragment],
    pair_scores: Dict[Tuple[int, int], float],
    alignments: Dict[Tuple[int, int], AlignmentResult],
) -> AssemblyResult:
    """Build graph, extract MST, and compute global transforms."""
    g = build_fragment_graph(len(fragments), pair_scores, alignments)

    # Keep largest connected component for stable assembly.
    components = sorted(nx.connected_components(g), key=len, reverse=True)
    largest_nodes = components[0] if components else set()
    g_largest = g.subgraph(largest_nodes).copy()

    mst = nx.maximum_spanning_tree(g_largest, weight="score")
    if mst.number_of_nodes() == 0:
        raise RuntimeError("Assembly graph is empty")

    degree_weight = {
        n: sum(g_largest[n][m]["score"] for m in g_largest.neighbors(n))
        for n in g_largest.nodes
    }
    root = max(degree_weight, key=degree_weight.get)

    transforms = compute_global_transforms(mst, alignments, root=root)

    connectivity = len(transforms) / max(1, len(fragments))
    avg_edge = float(np.mean([d["score"] for _, _, d in mst.edges(data=True)])) if mst.number_of_edges() > 0 else 0.0
    completeness = float(np.clip(0.7 * connectivity + 0.3 * avg_edge, 0.0, 1.0))

    return AssemblyResult(
        graph=g,
        mst=mst,
        global_transforms=transforms,
        completeness=completeness,
    )


def merge_and_save_reconstruction(
    fragments: List[Fragment],
    global_transforms: Dict[int, np.ndarray],
    output_path: Path,
    voxel_size: float = 0.008,
) -> o3d.geometry.PointCloud:
    """Merge transformed fragments and save to .ply."""
    all_points = []
    by_idx = {f.idx: f for f in fragments}

    for idx, tf in global_transforms.items():
        frag = by_idx[idx]
        homo = np.hstack([frag.points, np.ones((frag.points.shape[0], 1), dtype=np.float32)])
        pts = (homo @ tf.T)[:, :3]
        all_points.append(pts)

    if not all_points:
        raise RuntimeError("No transformed fragments available for reconstruction")

    merged = np.vstack(all_points).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(output_path), pcd)
    if not ok:
        raise RuntimeError(f"Failed to save reconstructed point cloud: {output_path}")

    LOG.info("Saved reconstructed model: %s (%d points)", output_path, len(pcd.points))
    return pcd
