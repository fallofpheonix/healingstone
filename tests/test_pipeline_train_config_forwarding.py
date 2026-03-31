from __future__ import annotations

import argparse
import importlib
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path

import numpy as np


def test_run_pipeline_forwards_train_config(monkeypatch, tmp_path: Path) -> None:
    pipeline = importlib.import_module("healingstone.pipeline.run_pipeline")

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "fragment_a.ply").touch()

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    captured: dict[str, float | int | str] = {}
    fragments = [
        SimpleNamespace(idx=0, name="a", points=np.zeros((1, 3), dtype=np.float32)),
        SimpleNamespace(idx=1, name="b", points=np.zeros((1, 3), dtype=np.float32)),
    ]

    fake_preprocess = ModuleType("healingstone.core.preprocess")
    fake_preprocess.load_and_preprocess_fragments = lambda **_: fragments
    fake_preprocess.set_deterministic_seed = lambda seed: None

    fake_features = ModuleType("healingstone.core.features")
    fake_features.extract_all_features = lambda **_: {0: object(), 1: object()}

    fake_match = ModuleType("healingstone.ml_models.match_fragments")

    def fake_train_and_match_fragments(**kwargs):
        captured.update(
            {
                "emb_dim": kwargs["emb_dim"],
                "epochs": kwargs["epochs"],
                "batch_size": kwargs["batch_size"],
                "lr": kwargs["lr"],
                "weight_decay": kwargs["weight_decay"],
                "margin": kwargs["margin"],
                "device": kwargs["device"],
            }
        )
        return (
            np.eye(2, dtype=np.float32),
            [(0, 1)],
            {(0, 1): 0.95},
            {
                "n_labeled_pairs": 0,
                "pairwise_match_accuracy": float("nan"),
                "metrics_at_selected_threshold": {"accuracy": float("nan")},
            },
            object(),
        )

    fake_match.train_and_match_fragments = fake_train_and_match_fragments

    fake_align = ModuleType("healingstone.core.geometry.align_fragments")
    fake_align.align_candidate_pairs = lambda **_: {
        (0, 1): SimpleNamespace(
            score_prior=0.95,
            fitness=0.8,
            inlier_rmse=0.01,
            chamfer=0.02,
            success=True,
        )
    }

    fake_reconstruct = ModuleType("healingstone.core.geometry.reconstruct")
    fake_reconstruct.assemble_global_reconstruction = lambda **_: SimpleNamespace(
        completeness=1.0,
        global_transforms={0: np.eye(4, dtype=np.float32)},
        graph=SimpleNamespace(number_of_nodes=lambda: 1, number_of_edges=lambda: 0),
    )
    fake_reconstruct.merge_and_save_reconstruction = lambda **_: SimpleNamespace(
        points=np.zeros((0, 3), dtype=np.float32)
    )

    monkeypatch.setitem(sys.modules, "healingstone.core.preprocess", fake_preprocess)
    monkeypatch.setitem(sys.modules, "healingstone.core.features", fake_features)
    monkeypatch.setitem(sys.modules, "healingstone.ml_models.match_fragments", fake_match)
    monkeypatch.setitem(sys.modules, "healingstone.core.geometry.align_fragments", fake_align)
    monkeypatch.setitem(sys.modules, "healingstone.core.geometry.reconstruct", fake_reconstruct)

    monkeypatch.setattr(pipeline, "resolve_data_dir", lambda **_: (data_dir, False))
    monkeypatch.setattr(pipeline, "resolve_artifact_root", lambda **_: (artifact_root, False))
    monkeypatch.setattr(pipeline, "_write_run_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "plot_similarity_matrix", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "plot_alignment_snapshots", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "plot_final_reconstruction", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "validate_mesh_integrity", lambda path: True)
    monkeypatch.setattr(pipeline, "_publish_reconstruction_alias", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_write_metrics_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "summarize_metrics",
        lambda **_: {
            "pairwise_match_accuracy": 0.9,
            "aligned_pairs": 0,
            "successful_alignments": 0,
            "mean_icp_rmse": 0.0,
            "mean_chamfer_distance": 0.0,
            "reconstruction_completeness": 1.0,
            "assembled_fragments": 1,
            "graph_nodes": 1,
            "graph_edges": 0,
        },
    )
    monkeypatch.setattr(pipeline, "validate_metrics_schema", lambda metrics: None)
    monkeypatch.setattr(pipeline, "enforce_accuracy_requirement", lambda **_: None)

    args = argparse.Namespace(
        config_version=1,
        dataset_alias="3d",
        data_dir=str(data_dir),
        output_dir=str(artifact_root),
        labels_csv=None,
        allow_overwrite_run=True,
        sample_points=100,
        voxel_size=0.01,
        normal_radius=0.02,
        normal_max_nn=16,
        outlier_nb_neighbors=8,
        outlier_std_ratio=1.0,
        k_neighbors=8,
        fpfh_radius=0.05,
        fpfh_max_nn=16,
        dbscan_eps=0.03,
        dbscan_min_samples=4,
        n_keypoints=32,
        candidate_top_k=2,
        align_top_n=2,
        label_suggestions_top_n=10,
        threshold_objective="accuracy",
        min_match_accuracy=0.0,
        min_required_accuracy=0.0,
        evaluation_split="test",
        augment_rotations=False,
        augment_count=1,
        seed=42,
        device="cpu",
        _config_hash="hash",
        _config_paths={},
        _config_source_map={},
        _dataset_aliases={},
        _train_config={
            "emb_dim": 96,
            "epochs": 7,
            "batch_size": 11,
            "lr": 0.0025,
            "weight_decay": 0.123,
            "margin": 1.7,
        },
    )

    pipeline.run_pipeline(args)

    assert captured == {
        "emb_dim": 96,
        "epochs": 7,
        "batch_size": 11,
        "lr": 0.0025,
        "weight_decay": 0.123,
        "margin": 1.7,
        "device": "cpu",
    }
