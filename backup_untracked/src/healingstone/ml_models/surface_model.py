from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from ..core.preprocess import Fragment

LOG = logging.getLogger(__name__)


class SurfacePointDataset(Dataset):
    """Dataset for point-wise surface classification."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.x = torch.from_numpy(features.astype(np.float32))
        self.y = torch.from_numpy(labels.astype(np.float32)).unsqueeze(1)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class PointSurfaceClassifier(nn.Module):
    """MLP for classifying individual points based on local geometry."""

    def __init__(self, in_dim: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_surface_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    models_dir: Path,
    epochs: int = 50,
    batch_size: int = 1024,
    lr: float = 1e-3,
    device: str = "cpu",
) -> PointSurfaceClassifier:
    """Train the classifier on provided features and labels (e.g., pseudo-labels)."""
    ds = SurfacePointDataset(features, labels)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = PointSurfaceClassifier(in_dim=features.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x_batch, y_batch in dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)

        if (epoch + 1) % 10 == 0:
            LOG.info("Surface model epoch %d/%d loss: %.6f", epoch + 1, epochs, total_loss / len(ds))

    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), models_dir / "surface_classifier.pt")
    return model


def collect_pseudo_labels(
    fragments: List[Fragment],
    k_neighbors: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    high_conf_threshold: float = 0.85,
    low_conf_threshold: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect high-confidence pseudo-labels from heuristic scores."""
    from ..core.features import build_point_surface_features, detect_break_surface

    rng = np.random.default_rng(seed)
    all_ft = []
    all_y = []
    for frag in fragments:
        _, score, geom = detect_break_surface(
            frag.points,
            frag.normals,
            k_neighbors=k_neighbors,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
        )
        ft = build_point_surface_features(
            frag.points, frag.normals, geom["curvature"], geom["normal_var"], geom["roughness"]
        )

        # Sampling high and low confidence points
        pos_idx = np.where(score >= high_conf_threshold)[0]
        neg_idx = np.where(score <= low_conf_threshold)[0]

        # Limit per fragment to balance
        n_pos = min(len(pos_idx), 2000)
        n_neg = min(len(neg_idx), 1500)  # slightly more break points usually? actually usually original is more.

        if n_pos > 0:
            pos_sel = rng.choice(pos_idx, size=n_pos, replace=False)
            all_ft.append(ft[pos_sel])
            all_y.append(np.ones(n_pos))

        if n_neg > 0:
            neg_sel = rng.choice(neg_idx, size=n_neg, replace=False)
            all_ft.append(ft[neg_sel])
            all_y.append(np.zeros(n_neg))

    if not all_ft:
        raise ValueError("No pseudo-labels collected; check features/heuristics")

    return np.vstack(all_ft), np.concatenate(all_y)


def predict_surface_labels(
    model: PointSurfaceClassifier,
    features: np.ndarray,
    device: str = "cpu",
    batch_size: int = 2048,
) -> np.ndarray:
    """Predict probabilities for each point."""
    model.eval()
    model.to(device)
    probs = []
    with torch.no_grad():
        for i in range(0, features.shape[0], batch_size):
            x_batch = torch.from_numpy(features[i : i + batch_size].astype(np.float32)).to(device)
            pred = model(x_batch)
            probs.append(pred.cpu().numpy())
    return np.vstack(probs).flatten()
