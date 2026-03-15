"""End-to-end fragment reconstruction pipeline (3D and 2D).

Automatically detects whether the input data directory contains 3D mesh files
(.PLY / .OBJ) or 2D image files (.PNG / .JPG) and routes to the appropriate
sub-pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import platform
import subprocess
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .metrics_schema import attach_schema_version, validate_metrics_schema
from .runtime_paths import (
    ResolvedRunPaths,
    initialize_run_layout,
    resolve_artifact_root,
    resolve_data_dir,
    write_resolved_paths_metadata,
)

LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .align_fragments import AlignmentResult
    from .preprocess import Fragment
    from .reconstruct import AssemblyResult

# ---------------------------------------------------------------------------
# Input-type detection
# ---------------------------------------------------------------------------

_MESH_EXTENSIONS = frozenset({".ply", ".obj"})
_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})


def detect_pipeline_mode(data_dir: Path) -> str:
    """Detect whether to use the 3D or 2D pipeline from files in *data_dir*.

    Scans *data_dir* recursively for mesh files (.ply, .obj) and image files
    (.png, .jpg, …).  Mesh files take precedence: if both types are present
    the 3D pipeline is selected.

    Parameters
    ----------
    data_dir:
        Directory to scan.

    Returns
    -------
    str
        ``"3d"`` or ``"2d"``.

    Raises
    ------
    FileNotFoundError
        If no supported files are found.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    has_mesh = False
    has_image = False
    for p in data_dir.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in _MESH_EXTENSIONS:
            has_mesh = True
            break  # mesh files → 3D pipeline, stop immediately
        if ext in _IMAGE_EXTENSIONS:
            has_image = True

    if has_mesh:
        LOG.info("Detected 3D mesh fragments in %s → running 3D pipeline", data_dir)
        return "3d"
    if has_image:
        LOG.info("Detected 2D image fragments in %s → running 2D pipeline", data_dir)
        return "2d"
    raise FileNotFoundError(
        f"No supported fragment files (.ply/.obj or .png/.jpg) found in: {data_dir}"
    )


def _json_safe(obj):
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


def _runtime_deps() -> Dict[str, str]:
    deps = ["numpy", "matplotlib", "networkx", "scikit-learn", "torch", "open3d", "pydantic", "PyYAML"]
    out: Dict[str, str] = {}
    for dep in deps:
        try:
            out[dep] = version(dep)
        except PackageNotFoundError:
            out[dep] = "not-installed"
    return out


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _count_binary_labels(labels_csv: Path) -> int:
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


def configure_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
    )
    return log_path


def plot_similarity_matrix(similarity: np.ndarray, fragments: List[Fragment], out_path: Path) -> None:
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
    diagnostics: Mapping[str, float],
    alignments: Dict[Tuple[int, int], AlignmentResult],
    assembly: AssemblyResult,
) -> Dict[str, Any]:
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


def enforce_accuracy_requirement(metrics: Mapping[str, Any], min_required_accuracy: float, evaluation_split: str) -> None:
    """Enforce mandatory pairwise accuracy threshold on test split."""
    if evaluation_split != "test":
        raise RuntimeError(
            f"Accuracy gate requires evaluation_split='test', got '{evaluation_split}'."
        )

    acc = float(metrics.get("pairwise_match_accuracy", float("nan")))
    if (not np.isfinite(acc)) or acc < float(min_required_accuracy):
        raise RuntimeError(
            f"Required pairwise_match_accuracy >= {min_required_accuracy:.2f} not met; got {acc:.4f}."
        )


def _serialize_effective_config(args: argparse.Namespace) -> Dict[str, object]:
    return {k: v for k, v in vars(args).items() if not k.startswith("_")}


def _write_run_metadata(args: argparse.Namespace, run_paths: ResolvedRunPaths, log_path: Path) -> None:
    payload = {
        "run_id": run_paths.run_id,
        "timestamp_utc": run_paths.run_id.split("_")[0],
        "git_commit": _git_commit(),
        "config_hash": getattr(args, "_config_hash", "unknown"),
        "config_version": int(args.config_version),
        "dataset_alias": args.dataset_alias,
        "resolved_data_dir": str(run_paths.data_dir),
        "used_legacy_data": run_paths.used_legacy_data,
        "used_legacy_output": run_paths.used_legacy_output,
        "seed": int(args.seed),
        "device": args.device,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependency_versions": _runtime_deps(),
        "evaluation_split": args.evaluation_split,
        "min_required_accuracy": float(args.min_required_accuracy),
        "config_paths": {k: str(v) for k, v in getattr(args, "_config_paths", {}).items()},
        "source_map": getattr(args, "_config_source_map", {}),
        "train_config": getattr(args, "_train_config", {}),
        "log_file": str(log_path),
    }
    (run_paths.run_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_resolved_paths_metadata(run_paths, run_paths.run_dir / "resolved_paths.json")


def _write_error_log(run_paths: ResolvedRunPaths | None, exc: Exception) -> None:
    if run_paths is None:
        return
    payload = {
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    (run_paths.logs_dir / "run_error.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute full reconstruction pipeline.

    Automatically selects the 3D or 2D sub-pipeline based on the fragment
    files found in the resolved data directory.
    """
    from .align_fragments import align_candidate_pairs
    from .features import extract_all_features
    from .match_fragments import train_and_match_fragments
    from .preprocess import load_and_preprocess_fragments, set_deterministic_seed
    from .reconstruct import assemble_global_reconstruction, merge_and_save_reconstruction

    run_paths: ResolvedRunPaths | None = None
    try:
        source_map = getattr(args, "_config_source_map", {})
        aliases = getattr(args, "_dataset_aliases", {})

        data_dir, used_legacy_data = resolve_data_dir(
            configured_data_dir=args.data_dir,
            data_dir_source=source_map.get("data_dir", "default"),
            dataset_alias=args.dataset_alias,
            aliases=aliases,
        )
        artifact_root, used_legacy_output = resolve_artifact_root(
            configured_output_dir=args.output_dir,
            output_dir_source=source_map.get("output_dir", "default"),
        )
        run_paths = initialize_run_layout(
            data_dir=data_dir,
            labels_csv=args.labels_csv,
            artifact_root=artifact_root,
            allow_overwrite_run=bool(args.allow_overwrite_run),
            used_legacy_data=used_legacy_data,
            used_legacy_output=used_legacy_output,
        )

        log_path = configure_logging(run_paths.logs_dir)
        _write_run_metadata(args, run_paths, log_path)

        if run_paths.used_legacy_data:
            LOG.warning("Using legacy dataset path fallback: %s", run_paths.data_dir)
        if run_paths.used_legacy_output:
            LOG.warning("Using legacy artifact root fallback: %s", run_paths.artifact_root)

        set_deterministic_seed(args.seed)

        # ------------------------------------------------------------------
        # Auto-detect input type and dispatch to the appropriate sub-pipeline.
        # ------------------------------------------------------------------
        pipeline_mode = detect_pipeline_mode(run_paths.data_dir)

        if pipeline_mode == "2d":
            _run_2d_pipeline(args, run_paths)
            return

        # ------------------------------------------------------------------
        # 3D pipeline (original implementation below).
        # ------------------------------------------------------------------
        labels_csv = run_paths.labels_csv
        enforce_accuracy_gate = args.min_match_accuracy is not None and float(args.min_match_accuracy) > 0.0
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
        LOG.info("Run directory: %s", run_paths.run_dir)

        fragments = load_and_preprocess_fragments(
            data_dir=run_paths.data_dir,
            sample_points=args.sample_points,
            voxel_size=args.voxel_size,
            normal_radius=args.normal_radius,
            normal_max_nn=args.normal_max_nn,
            outlier_nb_neighbors=args.outlier_nb_neighbors,
            outlier_std_ratio=args.outlier_std_ratio,
        )

        features = extract_all_features(
            fragments=fragments,
            cache_dir=run_paths.cache_dir,
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
            models_dir=run_paths.models_dir,
            output_dir=run_paths.results_dir,
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

        selected_metrics_raw: Any = diagnostics.get("metrics_at_selected_threshold", {})
        selected_metrics: Dict[str, float] = (
            selected_metrics_raw if isinstance(selected_metrics_raw, dict) else {}
        )
        selected_acc = float(selected_metrics.get("accuracy", float("nan")))
        labeled_pairs = int(diagnostics.get("n_labeled_pairs", 0))
        if enforce_accuracy_gate:
            if labeled_pairs == 0:
                raise RuntimeError(
                    f"Minimum accuracy {args.min_match_accuracy:.2f} required, but no labeled pairs were provided. "
                    f"Provide --labels-csv and annotate {run_paths.results_dir / 'labeling_candidates.csv'}."
                )
            if (not np.isfinite(selected_acc)) or (float(selected_acc) < float(args.min_match_accuracy)):
                raise RuntimeError(
                    f"Minimum pairwise match accuracy not met: got {selected_acc:.4f}, "
                    f"required >= {args.min_match_accuracy:.4f}."
                )

        plot_similarity_matrix(similarity, fragments, run_paths.results_dir / "similarity_matrix.png")

        alignments = align_candidate_pairs(
            fragments=fragments,
            features=features,
            candidate_pairs=candidate_pairs,
            pair_scores=pair_scores,
            voxel_size=args.voxel_size,
            top_n=args.align_top_n,
        )

        plot_alignment_snapshots(fragments, alignments, run_paths.results_dir, max_plots=min(5, args.align_top_n))

        assembly = assemble_global_reconstruction(
            fragments=fragments,
            pair_scores=pair_scores,
            alignments=alignments,
        )

        reconstructed_pcd = merge_and_save_reconstruction(
            fragments=fragments,
            global_transforms=assembly.global_transforms,
            output_path=run_paths.results_dir / "reconstructed_model.ply",
            voxel_size=max(args.voxel_size * 0.8, 0.006),
        )

        merged_pts = np.asarray(reconstructed_pcd.points)
        if merged_pts.size > 0:
            plot_final_reconstruction(merged_pts, run_paths.results_dir / "final_reconstruction.png")

        metrics = summarize_metrics(
            diagnostics=diagnostics,
            alignments=alignments,
            assembly=assembly,
        )
        metrics["min_required_accuracy"] = float(args.min_required_accuracy)
        metrics["evaluation_split"] = str(args.evaluation_split)
        validate_metrics_schema(metrics)
        enforce_accuracy_requirement(
            metrics=metrics,
            min_required_accuracy=float(args.min_required_accuracy),
            evaluation_split=str(args.evaluation_split),
        )

        report = {
            "config": _serialize_effective_config(args),
            "run": {
                "run_id": run_paths.run_id,
                "artifact_root": str(run_paths.artifact_root),
                "run_dir": str(run_paths.run_dir),
                "results_dir": str(run_paths.results_dir),
                "models_dir": str(run_paths.models_dir),
                "logs_dir": str(run_paths.logs_dir),
                "cache_dir": str(run_paths.cache_dir),
            },
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
        report = attach_schema_version(report)

        report_path = run_paths.results_dir / "alignment_metrics.json"
        report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")

        LOG.info("Pipeline complete")
        LOG.info("Reconstructed model: %s", run_paths.results_dir / "reconstructed_model.ply")
        LOG.info("Metrics report: %s", report_path)

    except Exception as exc:
        _write_error_log(run_paths, exc)
        raise


def _run_2d_pipeline(args: argparse.Namespace, run_paths: ResolvedRunPaths) -> None:
    """Delegate to the healingstone2d pipeline and write a minimal report."""
    try:
        from healingstone2d.reconstruct_2d import run_2d_pipeline
    except ImportError as exc:
        raise ImportError(
            "The healingstone2d package is required for 2D fragment reconstruction. "
            "Install it with: pip install 'healingstone[runtime]'"
        ) from exc

    LOG.info("Starting 2D reconstruction pipeline")
    LOG.info("Run directory: %s", run_paths.run_dir)

    metrics = run_2d_pipeline(
        data_dir=run_paths.data_dir,
        output_dir=run_paths.results_dir,
        seed=args.seed,
    )

    # Write a minimal JSON report so downstream tooling finds a report file.
    report = {
        "pipeline_mode": "2d",
        "run": {
            "run_id": run_paths.run_id,
            "results_dir": str(run_paths.results_dir),
        },
        "metrics": metrics,
    }
    report_path = run_paths.results_dir / "alignment_metrics.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")

    LOG.info("2D pipeline complete. Metrics report: %s", report_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct fragmented 3D artifact from .PLY/.OBJ fragments")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Pipeline config YAML path")
    parser.add_argument("--train-config", default="configs/train.yaml", help="Training config YAML path")
    parser.add_argument("--dataset-manifest", default="configs/datasets.yaml", help="Dataset alias manifest YAML path")

    parser.add_argument("--data-dir", default=None, help="Directory containing fragment .PLY/.OBJ files")
    parser.add_argument("--output-dir", default=None, help="Artifact root directory")
    parser.add_argument("--labels-csv", default=None, help="Optional labeled pair CSV (frag_a,frag_b,label)")
    parser.add_argument("--allow-overwrite-run", action="store_true", default=None, help="Allow reuse of existing run-id directory")

    parser.add_argument("--sample-points", type=int, default=None)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--normal-radius", type=float, default=None)
    parser.add_argument("--normal-max-nn", type=int, default=None)
    parser.add_argument("--outlier-nb-neighbors", type=int, default=None)
    parser.add_argument("--outlier-std-ratio", type=float, default=None)

    parser.add_argument("--k-neighbors", type=int, default=None)
    parser.add_argument("--fpfh-radius", type=float, default=None)
    parser.add_argument("--fpfh-max-nn", type=int, default=None)
    parser.add_argument("--dbscan-eps", type=float, default=None)
    parser.add_argument("--dbscan-min-samples", type=int, default=None)
    parser.add_argument("--n-keypoints", type=int, default=None)

    parser.add_argument("--candidate-top-k", type=int, default=None)
    parser.add_argument("--align-top-n", type=int, default=None)
    parser.add_argument("--label-suggestions-top-n", type=int, default=None)
    parser.add_argument("--threshold-objective", choices=["accuracy", "f1"], default=None)
    parser.add_argument("--min-match-accuracy", type=float, default=None)
    parser.add_argument("--min-required-accuracy", type=float, default=None)
    parser.add_argument("--evaluation-split", choices=["train", "validation", "test"], default=None)

    parser.add_argument("--augment-rotations", action="store_true", default=None, help="Enable random-rotation augmentation")
    parser.add_argument("--augment-count", type=int, default=None)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from .runtime_config import build_runtime_config, to_namespace

    bundle = build_runtime_config(args)
    effective_args = to_namespace(bundle)
    run_pipeline(effective_args)


if __name__ == "__main__":
    main()
