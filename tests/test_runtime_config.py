from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from healingstone.runtime_config import build_runtime_config


FIELDS = [
    "config_version",
    "dataset_alias",
    "data_dir",
    "output_dir",
    "labels_csv",
    "allow_overwrite_run",
    "sample_points",
    "voxel_size",
    "normal_radius",
    "normal_max_nn",
    "outlier_nb_neighbors",
    "outlier_std_ratio",
    "k_neighbors",
    "fpfh_radius",
    "fpfh_max_nn",
    "dbscan_eps",
    "dbscan_min_samples",
    "n_keypoints",
    "candidate_top_k",
    "align_top_n",
    "label_suggestions_top_n",
    "threshold_objective",
    "min_match_accuracy",
    "augment_rotations",
    "augment_count",
    "seed",
    "device",
]


def _empty_cli(config: Path, train: Path, datasets: Path) -> argparse.Namespace:
    payload = {k: None for k in FIELDS}
    payload.update(
        {
            "config": str(config),
            "train_config": str(train),
            "dataset_manifest": str(datasets),
        }
    )
    return argparse.Namespace(**payload)


def test_precedence_cli_over_env_over_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text("config_version: 1\nseed: 11\n", encoding="utf-8")
    train = tmp_path / "train.yaml"
    train.write_text("config_version: 1\n", encoding="utf-8")
    datasets = tmp_path / "datasets.yaml"
    datasets.write_text("aliases:\n  3d: data/raw/v1\n", encoding="utf-8")

    cli = _empty_cli(cfg, train, datasets)
    cli.seed = 33

    monkeypatch.setenv("HEALINGSTONE_SEED", "22")

    bundle = build_runtime_config(cli)
    assert bundle.pipeline.seed == 33


def test_invalid_config_version_fails(tmp_path: Path) -> None:
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text("config_version: 99\n", encoding="utf-8")
    train = tmp_path / "train.yaml"
    train.write_text("config_version: 1\n", encoding="utf-8")
    datasets = tmp_path / "datasets.yaml"
    datasets.write_text("aliases: {}\n", encoding="utf-8")

    cli = _empty_cli(cfg, train, datasets)
    with pytest.raises(ValueError):
        build_runtime_config(cli)
