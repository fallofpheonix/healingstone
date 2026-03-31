"""End-to-end fragment reconstruction pipeline (3D and 2D)."""

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
from typing import Any, Dict, Mapping

import numpy as np

from ..core.metrics_collector import summarize_3d_metrics
from ..core.metrics_schema import attach_schema_version, validate_metrics_schema
from ..core.runtime_paths import (
    ResolvedRunPaths,
    _contains_fragments,
    _contains_images,
    initialize_run_layout,
    project_root,
    resolve_artifact_root,
    resolve_data_dir,
    write_resolved_paths_metadata,
)
from ..utils.visualization import (
    plot_alignment_snapshots,
    plot_final_reconstruction,
    plot_similarity_matrix,
)

LOG = logging.getLogger(__name__)


def detect_pipeline_mode(data_dir: Path) -> str:
    """Detect whether to use the 3D or 2D pipeline from files in *data_dir*."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    if _contains_fragments(data_dir):
        LOG.info("Detected 3D mesh fragments in %s -> running 3D pipeline", data_dir)
        return "3d"
    if _contains_images(data_dir):
        LOG.info("Detected 2D image fragments in %s -> running 2D pipeline", data_dir)
        return "2d"
    raise FileNotFoundError(
        f"No supported fragment files (.ply/.obj/.png/.jpg/.jpeg/.tif/.tiff/.bmp) found in: {data_dir}"
    )


def _detect_input_type(data_dir: Path) -> str:
    """Backward-compatible wrapper around the public pipeline detector."""
    return detect_pipeline_mode(data_dir)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.bool_):
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
    with labels_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
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


def enforce_accuracy_requirement(metrics: Mapping[str, Any], min_required_accuracy: float, evaluation_split: str) -> None:
    """Enforce mandatory pairwise accuracy threshold on test split."""
    if min_required_accuracy <= 0:
        return

    n_labeled_pairs = int(metrics.get("n_labeled_pairs", 0) or 0)
    if n_labeled_pairs <= 0:
        LOG.warning(
            "Skipping required pairwise accuracy gate because no labeled pairs were provided."
        )
        return

    if evaluation_split != "test":
        raise RuntimeError(
            f"Accuracy gate requires evaluation_split='test', got '{evaluation_split}'."
        )

    accuracy = float(metrics.get("pairwise_match_accuracy", float("nan")))
    if np.isnan(accuracy):
        raise RuntimeError(
            "Required pairwise_match_accuracy is undefined despite labeled pairs being present."
        )

    if accuracy < float(min_required_accuracy):
        raise RuntimeError(
            f"Required pairwise_match_accuracy >= {min_required_accuracy:.2f} not met; got {accuracy:.4f}."
        )


def summarize_metrics(
    diagnostics: Mapping[str, Any],
    alignments: Dict[tuple[int, int], Any],
    assembly: Any,
) -> Dict[str, Any]:
    """Backward-compatible metrics helper forwarded to the core collector."""
    return summarize_3d_metrics(diagnostics=diagnostics, alignments=alignments, assembly=assembly)


def _serialize_effective_config(args: argparse.Namespace) -> Dict[str, object]:
    return {key: value for key, value in vars(args).items() if not key.startswith("_")}


def _write_run_metadata(args: argparse.Namespace, run_paths: ResolvedRunPaths, log_path: Path) -> None:
    payload = {
        "run_id": run_paths.run_id,
        "timestamp_utc": run_paths.run_id.split("_")[0],
        "project_root": str(project_root()),
        "git_commit": _git_commit(),
        "config_hash": getattr(args, "_config_hash", "unknown"),
        "config_version": int(args.config_version),
        "dataset_alias": args.dataset_alias,
        "resolved_data_dir": str(run_paths.data_dir),
        "seed": int(args.seed),
        "device": args.device,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependency_versions": _runtime_deps(),
        "evaluation_split": args.evaluation_split,
        "min_required_accuracy": float(args.min_required_accuracy),
        "config_paths": {key: str(value) for key, value in getattr(args, "_config_paths", {}).items()},
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


def _resolve_run_paths(args: argparse.Namespace) -> ResolvedRunPaths:
    source_map = getattr(args, "_config_source_map", {})
    aliases = getattr(args, "_dataset_aliases", {})
    data_dir = resolve_data_dir(
        configured_data_dir=args.data_dir,
        data_dir_source=source_map.get("data_dir", "default"),
        dataset_alias=args.dataset_alias,
        aliases=aliases,
    )
    if isinstance(data_dir, tuple):
        data_dir = data_dir[0]
    artifact_root = resolve_artifact_root(
        configured_output_dir=args.output_dir,
        output_dir_source=source_map.get("output_dir", "default"),
    )
    if isinstance(artifact_root, tuple):
        artifact_root = artifact_root[0]
    return initialize_run_layout(
        data_dir=data_dir,
        labels_csv=args.labels_csv,
        artifact_root=artifact_root,
        allow_overwrite_run=bool(args.allow_overwrite_run),
    )


def _run_2d_pipeline(args: argparse.Namespace, run_paths: ResolvedRunPaths) -> None:
    """Delegate to the healingstone2d pipeline and write a minimal report."""
    try:
        from ..healingstone2d.reconstruct_2d import run_2d_pipeline
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

    report = {
        "pipeline_mode": "2d",
        "config": _serialize_effective_config(args),
        "run": {
            "run_id": run_paths.run_id,
            "artifact_root": str(run_paths.artifact_root),
            "run_dir": str(run_paths.run_dir),
            "results_dir": str(run_paths.results_dir),
            "logs_dir": str(run_paths.logs_dir),
        },
        "metrics": metrics,
    }
    report_path = run_paths.results_dir / "alignment_metrics.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")

    LOG.info("2D pipeline complete. Metrics report: %s", report_path)


def _run_3d_pipeline(args: argparse.Namespace, run_paths: ResolvedRunPaths) -> None:
    from ..alignment.align_fragments import align_candidate_pairs
    from ..alignment.reconstruct import assemble_global_reconstruction, merge_and_save_reconstruction
    from ..core.features import extract_all_features
    from ..core.preprocess import load_and_preprocess_fragments, set_deterministic_seed
    from ..ml_models.match_fragments import train_and_match_fragments

    set_deterministic_seed(args.seed)

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

    LOG.info("Starting 3D reconstruction pipeline")
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
        emb_dim=int(getattr(args, "_train_config", {}).get("emb_dim", 64)),
        epochs=int(getattr(args, "_train_config", {}).get("epochs", 120)),
        batch_size=int(getattr(args, "_train_config", {}).get("batch_size", 64)),
        lr=float(getattr(args, "_train_config", {}).get("lr", 1e-3)),
        weight_decay=float(getattr(args, "_train_config", {}).get("weight_decay", 1e-5)),
        margin=float(getattr(args, "_train_config", {}).get("margin", 1.0)),
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
    selected_metrics: Dict[str, float] = selected_metrics_raw if isinstance(selected_metrics_raw, dict) else {}
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

    merged_points = np.asarray(reconstructed_pcd.points)
    if merged_points.size > 0:
        plot_final_reconstruction(merged_points, run_paths.results_dir / "final_reconstruction.png")

    metrics = summarize_3d_metrics(
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
                "prior_score": result.score_prior,
                "fitness": result.fitness,
                "icp_rmse": result.inlier_rmse,
                "chamfer": result.chamfer,
                "success": result.success,
            }
            for (i, j), result in alignments.items()
        },
    }
    report = attach_schema_version(report)

    report_path = run_paths.results_dir / "alignment_metrics.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")

    LOG.info("Pipeline complete")
    LOG.info("Reconstructed model: %s", run_paths.results_dir / "reconstructed_model.ply")
    LOG.info("Metrics report: %s", report_path)


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute full reconstruction pipeline."""
    run_paths: ResolvedRunPaths | None = None
    try:
        run_paths = _resolve_run_paths(args)
        log_path = configure_logging(run_paths.logs_dir)
        _write_run_metadata(args, run_paths, log_path)

        pipeline_mode = detect_pipeline_mode(run_paths.data_dir)
        LOG.info("Detected input type: %s", pipeline_mode)

        if pipeline_mode == "2d":
            _run_2d_pipeline(args, run_paths)
            return

        _run_3d_pipeline(args, run_paths)
    except Exception as exc:
        _write_error_log(run_paths, exc)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct fragmented artifacts from 3D meshes or 2D images")
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"), help="Pipeline config YAML path")
    parser.add_argument("--train-config", type=Path, default=Path("configs/train.yaml"), help="Training config YAML path")
    parser.add_argument("--dataset-manifest", type=Path, default=Path("configs/datasets.yaml"), help="Dataset alias manifest YAML path")

    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing fragment .PLY/.OBJ/.PNG/.JPG files")
    parser.add_argument("--output-dir", type=Path, default=None, help="Artifact root directory")
    parser.add_argument("--labels-csv", type=Path, default=None, help="Optional labeled pair CSV (frag_a,frag_b,label)")
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
    from ..services.reconstruction_service import execute_reconstruction

    execute_reconstruction(args)


if __name__ == "__main__":
    main()
