"""Break surface classifier and Siamese matching models.

Requires the ``runtime`` optional dependencies (open3d, torch).  Install with::

    pip install 'healingstone[runtime]'
"""

from __future__ import annotations

from healingstone.ml_models.surface_model import (
    BreakSurfaceClassifier,
    SurfaceModelBundle,
    predict_break_surface,
    train_break_surface_classifier,
)

try:
    from healingstone.match_fragments import (
        calibrate_threshold,
        evaluate_pair_accuracy,
        evaluate_pair_metrics,
        load_pair_labels,
        reciprocal_topk_pairs,
        train_and_match_fragments,
        write_labeling_candidates,
    )
    from healingstone.train_model import (
        ContrastiveLoss,
        PairDataset,
        SiameseEncoder,
        SiameseModelBundle,
        cosine_similarity_matrix,
        encode_descriptors,
        train_siamese_model,
    )
except ImportError:
    pass  # torch not installed; individual modules still importable.

__all__ = [
    "BreakSurfaceClassifier",
    "SurfaceModelBundle",
    "train_break_surface_classifier",
    "predict_break_surface",
    "SiameseEncoder",
    "ContrastiveLoss",
    "PairDataset",
    "SiameseModelBundle",
    "train_siamese_model",
    "encode_descriptors",
    "cosine_similarity_matrix",
    "load_pair_labels",
    "reciprocal_topk_pairs",
    "evaluate_pair_accuracy",
    "evaluate_pair_metrics",
    "calibrate_threshold",
    "write_labeling_candidates",
    "train_and_match_fragments",
]
