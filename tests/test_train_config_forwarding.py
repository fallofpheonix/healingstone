from __future__ import annotations

from pathlib import Path

import numpy as np

from healingstone.ml_models import match_fragments


def test_train_and_match_forwards_train_hyperparameters(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, float | int | str] = {}

    monkeypatch.setattr(
        match_fragments,
        "_build_self_supervised_pairs",
        lambda **_: (
            np.ones((4, 3), dtype=np.float32),
            np.ones((4, 3), dtype=np.float32),
            np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
            {"n_train_pairs": 4, "n_pos_pairs": 2, "n_neg_pairs": 2},
        ),
    )
    monkeypatch.setattr(
        match_fragments,
        "_descriptor_matrix",
        lambda features, n: np.vstack([features[i].descriptor for i in range(n)]).astype(np.float32),
    )
    monkeypatch.setattr(
        match_fragments,
        "train_siamese_model",
        lambda **kwargs: captured.update(
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
        or object(),
    )
    monkeypatch.setattr(
        match_fragments,
        "encode_descriptors",
        lambda descriptors, bundle, device: descriptors,
    )
    monkeypatch.setattr(
        match_fragments,
        "cosine_similarity_matrix",
        lambda embeddings: np.eye(len(embeddings), dtype=np.float32),
    )
    monkeypatch.setattr(
        match_fragments,
        "reciprocal_topk_pairs",
        lambda similarity, top_k: [(0, 1)],
    )
    monkeypatch.setattr(match_fragments, "write_labeling_candidates", lambda **_: None)
    monkeypatch.setattr(match_fragments, "calibrate_threshold", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(
        match_fragments,
        "evaluate_pair_metrics",
        lambda *args, **kwargs: {
            "accuracy": float("nan"),
            "f1": float("nan"),
        },
    )

    fragment = type("Fragment", (), {"idx": 0, "name": "a", "path": tmp_path / "a.ply"})()
    fragment_b = type("Fragment", (), {"idx": 1, "name": "b", "path": tmp_path / "b.ply"})()
    bundle = type("FeatureBundle", (), {"descriptor": np.array([1.0, 2.0, 3.0], dtype=np.float32)})()
    features = {0: bundle, 1: bundle}

    match_fragments.train_and_match_fragments(
        fragments=[fragment, fragment_b],
        features=features,
        models_dir=tmp_path / "models",
        output_dir=tmp_path / "results",
        emb_dim=96,
        epochs=7,
        batch_size=11,
        lr=0.0025,
        weight_decay=0.123,
        margin=1.7,
        device="cpu",
    )

    assert captured == {
        "emb_dim": 96,
        "epochs": 7,
        "batch_size": 11,
        "lr": 0.0025,
        "weight_decay": 0.123,
        "margin": 1.7,
        "device": "cpu",
    }
