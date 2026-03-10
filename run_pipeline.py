"""End-to-end 3D fragment reconstruction pipeline.

Run:
    python run_pipeline.py --data-dir DataSet/3D --output-dir results
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from align_fragments import AlignmentResult, align_candidate_pairs
from features import FeatureBundle, extract_all_features
from match_fragments import train_and_match_fragments
from preprocess import Fragment, load_and_preprocess_fragments, set_deterministic_seed
from reconstruct import AssemblyResult, assemble_global_reconstruction, merge_and_save_reconstruction

LOG = logging.getLogger(__name__)


def _json_safe(obj):
    """Recursively convert NumPy/Python objects into JSON-serializable primitives."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _count_binary_labels(labels_csv: Path) -> int:
    """Count rows where label is explicitly 0 or 1."""
    if not labels_csv.exists():
        return 0
    count = 0
    with labels_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get("label") or "").strip()
            if label in {"0", "1"}:
                count += 1
    return count


def configure_logging(output_dir: Path) -> None:
    """Configure console and file logging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
    )


def plot_similarity_matrix(similarity: np.ndarray, fragments: List[Fragment], out_path: Path) -> None:
    """Save similarity matrix heatmap."""
    labels = [f.name for f in fragments]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(similarity, cmap="viridis", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="Learned Similarity")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Fragment Similarity Matrix")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homo = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float32)])
    return (homo @ transform.T)[:, :3]


def plot_alignment_snapshots(
    fragments: List[Fragment],
    alignments: Dict[Tuple[int, int], AlignmentResult],
    output_dir: Path,
    max_plots: int = 4,
) -> None:
    """Plot before/after alignment for top aligned pairs."""
    ordered = sorted(
        alignments.values(),
        key=lambda r: (r.success, -r.score_prior, -r.fitness),
        reverse=True,
    )[:max_plots]

    by_idx = {f.idx: f for f in fragments}
    for rank, result in enumerate(ordered, start=1):
        src = by_idx[result.i].points
        tgt = by_idx[result.j].points
        aligned = _apply_transform(src, result.transform_ij)

        fig = plt.figure(figsize=(12, 5))

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.scatter(src[::8, 0], src[::8, 1], src[::8, 2], s=1, alpha=0.5, c="tomato", label=f"{by_idx[result.i].name}")
        ax1.scatter(tgt[::8, 0], tgt[::8, 1], tgt[::8, 2], s=1, alpha=0.5, c="steelblue", label=f"{by_idx[result.j].name}")
        ax1.set_title("Before Alignment")
        ax1.legend(fontsize=7)

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.scatter(aligned[::8, 0], aligned[::8, 1], aligned[::8, 2], s=1, alpha=0.5, c="tomato", label=f"{by_idx[result.i].name} aligned")
        ax2.scatter(tgt[::8, 0], tgt[::8, 1], tgt[::8, 2], s=1, alpha=0.5, c="steelblue", label=f"{by_idx[result.j].name}")
        ax2.set_title(f"After Alignment\nRMSE={result.inlier_rmse:.4f}, fitness={result.fitness:.3f}")
        ax2.legend(fontsize=7)

        plt.tight_layout()
        out = output_dir / f"alignment_pair_{rank}_{by_idx[result.i].name}_{by_idx[result.j].name}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)


def plot_final_reconstruction(points: np.ndarray, out_path: Path) -> None:
    """Plot merged reconstructed point cloud."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    sub = points[:: max(1, points.shape[0] // 15000)]
    ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=1, alpha=0.6, c=sub[:, 2], cmap="viridis")
    ax.set_title("Final Reconstructed Artifact")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summarize_metrics(
    diagnostics: Dict[str, float],
    alignments: Dict[Tuple[int, int], AlignmentResult],
    assembly: AssemblyResult,
) -> Dict[str, float]:
    """Compute aggregate evaluation metrics."""
    rmses = [r.inlier_rmse for r in alignments.values() if r.success and np.isfinite(r.inlier_rmse)]
    chs = [r.chamfer for r in alignments.values() if r.success and np.isfinite(r.chamfer)]

    metrics = {
        "pairwise_match_accuracy": diagnostics.get("pairwise_match_accuracy", float("nan")),
        "aligned_pairs": int(len(alignments)),
        "successful_alignments": int(sum(1 for r in alignments.values() if r.success)),
        "mean_icp_rmse": float(np.mean(rmses)) if rmses else float("nan"),
        "mean_chamfer_distance": float(np.mean(chs)) if chs else float("nan"),
        "reconstruction_completeness": float(assembly.completeness),
        "assembled_fragments": int(len(assembly.global_transforms)),
        "graph_nodes": int(assembly.graph.number_of_nodes()),
        "graph_edges": int(assembly.graph.number_of_edges()),
    }
    return metrics


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute full reconstruction pipeline."""
    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(output_dir)
    set_deterministic_seed(args.seed)

    data_dir = Path(args.data_dir)
    labels_csv = Path(args.labels_csv) if args.labels_csv else None

    enforce_accuracy_gate = (
        args.min_match_accuracy is not None and float(args.min_match_accuracy) > 0.0
    )
    if enforce_accuracy_gate:
        if labels_csv is None:
            raise RuntimeError(
                f"Minimum accuracy {args.min_match_accuracy:.2f} required, but --labels-csv was not provided."
            )
        n_labeled_rows = _count_binary_labels(labels_csv)
        if n_labeled_rows == 0:
            raise RuntimeError(
                f"Minimum accuracy {args.min_match_accuracy:.2f} required, but {labels_csv} has 0 labeled rows. "
                "Fill the 'label' column with 0/1 values first."
            )

    LOG.info("Starting reconstruction pipeline")
    fragments = load_and_preprocess_fragments(
        data_dir=data_dir,
        sample_points=args.sample_points,
        voxel_size=args.voxel_size,
        normal_radius=args.normal_radius,
        normal_max_nn=args.normal_max_nn,
        outlier_nb_neighbors=args.outlier_nb_neighbors,
        outlier_std_ratio=args.outlier_std_ratio,
    )

    features = extract_all_features(
        fragments=fragments,
        cache_dir=cache_dir,
        k_neighbors=args.k_neighbors,
        fpfh_radius=args.fpfh_radius,
        fpfh_max_nn=args.fpfh_max_nn,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        n_keypoints=args.n_keypoints,
    )

    similarity, candidate_pairs, pair_scores, diagnostics, _ = train_and_match_fragments(
        fragments=fragments,
        features=features,
        models_dir=models_dir,
        output_dir=output_dir,
        labels_csv=labels_csv,
        augment_rotations=args.augment_rotations,
        augment_count=args.augment_count,
        candidate_top_k=args.candidate_top_k,
        label_suggestions_top_n=args.label_suggestions_top_n,
        threshold_objective=args.threshold_objective,
        k_neighbors=args.k_neighbors,
        fpfh_radius=args.fpfh_radius,
        fpfh_max_nn=args.fpfh_max_nn,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        n_keypoints=args.n_keypoints,
        seed=args.seed,
        device=args.device,
    )
    if "selected_threshold" in diagnostics:
        LOG.info("Selected match threshold: %.4f", diagnostics["selected_threshold"])
    if "metrics_at_selected_threshold" in diagnostics:
        LOG.info("Labeled metrics @selected threshold: %s", diagnostics["metrics_at_selected_threshold"])

    selected_metrics = diagnostics.get("metrics_at_selected_threshold", {})
    selected_acc = selected_metrics.get("accuracy", float("nan"))
    labeled_pairs = int(diagnostics.get("n_labeled_pairs", 0))
    if enforce_accuracy_gate:
        if labeled_pairs == 0:
            raise RuntimeError(
                f"Minimum accuracy {args.min_match_accuracy:.2f} required, but no labeled pairs were provided. "
                f"Provide --labels-csv and annotate {output_dir / 'labeling_candidates.csv'}."
            )
        if (not np.isfinite(selected_acc)) or (float(selected_acc) < float(args.min_match_accuracy)):
            raise RuntimeError(
                f"Minimum pairwise match accuracy not met: got {selected_acc:.4f}, "
                f"required >= {args.min_match_accuracy:.4f}."
            )

    plot_similarity_matrix(similarity, fragments, output_dir / "similarity_matrix.png")

    alignments = align_candidate_pairs(
        fragments=fragments,
        features=features,
        candidate_pairs=candidate_pairs,
        pair_scores=pair_scores,
        voxel_size=args.voxel_size,
        top_n=args.align_top_n,
    )

    plot_alignment_snapshots(fragments, alignments, output_dir, max_plots=min(5, args.align_top_n))

    assembly = assemble_global_reconstruction(
        fragments=fragments,
        pair_scores=pair_scores,
        alignments=alignments,
    )

    reconstructed_pcd = merge_and_save_reconstruction(
        fragments=fragments,
        global_transforms=assembly.global_transforms,
        output_path=output_dir / "reconstructed_model.ply",
        voxel_size=max(args.voxel_size * 0.8, 0.006),
    )

    merged_pts = np.asarray(reconstructed_pcd.points)
    if merged_pts.size > 0:
        plot_final_reconstruction(merged_pts, output_dir / "final_reconstruction.png")

    metrics = summarize_metrics(
        diagnostics=diagnostics,
        alignments=alignments,
        assembly=assembly,
    )

    report = {
        "config": vars(args),
        "n_fragments": len(fragments),
        "candidate_pairs": [[int(a), int(b)] for (a, b) in candidate_pairs],
        "diagnostics": diagnostics,
        "metrics": metrics,
        "alignment_results": {
            f"{i}_{j}": {
                "prior_score": r.score_prior,
                "fitness": r.fitness,
                "icp_rmse": r.inlier_rmse,
                "chamfer": r.chamfer,
                "success": r.success,
            }
            for (i, j), r in alignments.items()
        },
    }

    report_path = output_dir / "alignment_metrics.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")

    LOG.info("Pipeline complete")
    LOG.info("Reconstructed model: %s", output_dir / "reconstructed_model.ply")
    LOG.info("Metrics report: %s", report_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct fragmented 3D artifact from .PLY/.OBJ fragments")
    parser.add_argument("--data-dir", default="DataSet/3D", help="Directory containing fragment .PLY/.OBJ files")
    parser.add_argument("--output-dir", default="results", help="Directory for outputs and artifacts")
    parser.add_argument("--labels-csv", default=None, help="Optional labeled pair CSV (frag_a,frag_b,label)")

    parser.add_argument("--sample-points", type=int, default=40000)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--normal-radius", type=float, default=0.04)
    parser.add_argument("--normal-max-nn", type=int, default=64)
    parser.add_argument("--outlier-nb-neighbors", type=int, default=24)
    parser.add_argument("--outlier-std-ratio", type=float, default=1.8)

    parser.add_argument("--k-neighbors", type=int, default=24)
    parser.add_argument("--fpfh-radius", type=float, default=0.06)
    parser.add_argument("--fpfh-max-nn", type=int, default=100)
    parser.add_argument("--dbscan-eps", type=float, default=0.04)
    parser.add_argument("--dbscan-min-samples", type=int, default=24)
    parser.add_argument("--n-keypoints", type=int, default=256)

    parser.add_argument("--candidate-top-k", type=int, default=4)
    parser.add_argument("--align-top-n", type=int, default=10)
    parser.add_argument("--label-suggestions-top-n", type=int, default=50)
    parser.add_argument("--threshold-objective", choices=["accuracy", "f1"], default="accuracy")
    parser.add_argument("--min-match-accuracy", type=float, default=0.80)

    parser.add_argument("--augment-rotations", action="store_true", help="Enable random-rotation augmentation")
    parser.add_argument("--augment-count", type=int, default=2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"]) 

    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
