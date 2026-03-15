"""Re-export shim: healingstone.ml_models.train_model → healingstone.train_model."""

from healingstone.train_model import (  # noqa: F401
    ContrastiveLoss,
    PairDataset,
    SiameseEncoder,
    SiameseModelBundle,
    cosine_similarity_matrix,
    encode_descriptors,
    train_siamese_model,
)

__all__ = [
    "PairDataset",
    "SiameseEncoder",
    "ContrastiveLoss",
    "SiameseModelBundle",
    "train_siamese_model",
    "encode_descriptors",
    "cosine_similarity_matrix",
]
