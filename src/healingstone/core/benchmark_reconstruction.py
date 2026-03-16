"""Reconstruction accuracy benchmarking tool.

Computes geometric quality metrics between reconstructed meshes and
reference ground-truth data (or pairwise between aligned fragments).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import open3d as o3d

LOG = logging.getLogger(__name__)


def chamfer_distance(
    source: np.ndarray, target: np.ndarray
) -> Dict[str, float]:
    """Compute asymmetric and symmetric Chamfer distance."""
    from scipy.spatial import cKDTree

    tree_s = cKDTree(source)
    tree_t = cKDTree(target)

    d_st, _ = tree_t.query(source, k=1)
    d_ts, _ = tree_s.query(target, k=1)

    return {
        "chamfer_s2t_mean": float(np.mean(d_st)),
        "chamfer_t2s_mean": float(np.mean(d_ts)),
        "chamfer_symmetric": float((np.mean(d_st) + np.mean(d_ts)) / 2.0),
    }


def hausdorff_distance(
    source: np.ndarray, target: np.ndarray
) -> Dict[str, float]:
    """Compute directed and symmetric Hausdorff distances."""
    from scipy.spatial import cKDTree

    tree_s = cKDTree(source)
    tree_t = cKDTree(target)

    d_st, _ = tree_t.query(source, k=1)
    d_ts, _ = tree_s.query(target, k=1)

    return {
        "hausdorff_s2t": float(np.max(d_st)),
        "hausdorff_t2s": float(np.max(d_ts)),
        "hausdorff_symmetric": float(max(np.max(d_st), np.max(d_ts))),
        "hausdorff_rms_s2t": float(np.sqrt(np.mean(d_st ** 2))),
        "hausdorff_rms_t2s": float(np.sqrt(np.mean(d_ts ** 2))),
    }


def point_to_point_rmse(
    source: np.ndarray, target: np.ndarray
) -> float:
    """Compute point-to-point RMS error."""
    from scipy.spatial import cKDTree

    tree = cKDTree(target)
    dists, _ = tree.query(source, k=1)
    return float(np.sqrt(np.mean(dists ** 2)))


def geometric_completeness(
    source: np.ndarray, target: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute geometric completeness: fraction of target covered by source."""
    from scipy.spatial import cKDTree

    tree_s = cKDTree(source)
    dists, _ = tree_s.query(target, k=1)
    covered = np.sum(dists < threshold)
    return {
        "completeness": float(covered / max(1, len(target))),
        "threshold": float(threshold),
        "n_covered": int(covered),
        "n_total": int(len(target)),
    }


def benchmark_pair(
    source_path: Path,
    target_path: Path,
    n_sample: int = 100_000,
    completeness_threshold: float = 0.5,
) -> Dict[str, any]:
    """Run all benchmark metrics on a source-target pair."""
    t0 = time.time()

    mesh_s = o3d.io.read_triangle_mesh(str(source_path))
    mesh_t = o3d.io.read_triangle_mesh(str(target_path))

    pcd_s = mesh_s.sample_points_uniformly(number_of_points=n_sample)
    pcd_t = mesh_t.sample_points_uniformly(number_of_points=n_sample)

    pts_s = np.asarray(pcd_s.points).astype(np.float64)
    pts_t = np.asarray(pcd_t.points).astype(np.float64)

    results: Dict[str, any] = {
        "source": source_path.name,
        "target": target_path.name,
        "n_sample": n_sample,
    }

    results.update(chamfer_distance(pts_s, pts_t))
    results.update(hausdorff_distance(pts_s, pts_t))
    results["point_to_point_rmse"] = point_to_point_rmse(pts_s, pts_t)
    results.update(geometric_completeness(pts_s, pts_t, threshold=completeness_threshold))
    results["elapsed_seconds"] = round(time.time() - t0, 3)

    return results


def write_markdown_report(
    all_results: List[Dict[str, any]], output_path: Path
):
    """Write RECONSTRUCTION_BENCHMARK_REPORT.md."""
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Reconstruction Benchmark Report\n\n")
        f.write("## Pairwise Metrics\n\n")
        f.write("| Source | Target | Chamfer (sym) | Hausdorff (sym) | P2P RMSE | Completeness | Time (s) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in all_results:
            f.write(
                f"| {r['source']} | {r['target']} "
                f"| {r['chamfer_symmetric']:.4f} "
                f"| {r['hausdorff_symmetric']:.4f} "
                f"| {r['point_to_point_rmse']:.4f} "
                f"| {r['completeness']:.2%} "
                f"| {r['elapsed_seconds']:.1f} |\n"
            )
        f.write("\n## Metric Definitions\n")
        f.write("- **Chamfer Distance**: Average nearest-neighbor distance (symmetric).\n")
        f.write("- **Hausdorff Distance**: Maximum nearest-neighbor distance (worst case).\n")
        f.write("- **P2P RMSE**: Root-mean-square of nearest-neighbor distances.\n")
        f.write("- **Completeness**: Fraction of target points within threshold of source.\n")


def write_json_report(
    all_results: List[Dict[str, any]], output_path: Path
):
    """Write benchmark results as JSON."""
    output_path.write_text(
        json.dumps(all_results, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruction accuracy benchmarks.")
    parser.add_argument("--source", type=Path, required=True, help="Source mesh")
    parser.add_argument("--target", type=Path, required=True, help="Target mesh")
    parser.add_argument("--n-sample", type=int, default=100_000,
                        help="Points to sample for comparison")
    parser.add_argument("--output", type=Path,
                        default=Path("RECONSTRUCTION_BENCHMARK_REPORT.md"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = benchmark_pair(args.source, args.target, n_sample=args.n_sample)
    write_markdown_report([result], args.output)
    write_json_report([result], args.output.with_suffix(".json"))
    LOG.info("Benchmark report written to %s", args.output)
