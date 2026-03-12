"""Siamese embedding model training with contrastive loss."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

LOG = logging.getLogger(__name__)


class PairDataset(Dataset):
    """Dataset for Siamese pair training."""

    def __init__(self, x1: np.ndarray, x2: np.ndarray, y: np.ndarray):
        self.x1 = torch.from_numpy(x1.astype(np.float32))
        self.x2 = torch.from_numpy(x2.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        return self.x1[idx], self.x2[idx], self.y[idx]


class SiameseEncoder(nn.Module):
    """MLP encoder mapping descriptor vectors to compact embeddings."""

    def __init__(self, in_dim: int, emb_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.10),
            nn.Linear(128, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return nn.functional.normalize(z, p=2, dim=1)


class ContrastiveLoss(nn.Module):
    """Contrastive loss on embedding distances."""

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y=1 for positive pairs, y=0 for negative pairs.
        d = torch.norm(z1 - z2, dim=1)
        pos = y * d.pow(2)
        neg = (1.0 - y) * torch.clamp(self.margin - d, min=0.0).pow(2)
        return (pos + neg).mean()


@dataclass
class SiameseModelBundle:
    """Trained Siamese model + scaler for inference."""

    model: SiameseEncoder
    scaler: StandardScaler
    train_loss: np.ndarray


def train_siamese_model(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    models_dir: Path,
    emb_dim: int = 64,
    epochs: int = 120,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    margin: float = 1.0,
    device: str = "cpu",
) -> SiameseModelBundle:
    """Train Siamese model with contrastive loss."""
    if x1.shape[0] < 4:
        raise ValueError("Not enough pairs to train Siamese model")

    scaler = StandardScaler()
    stacked = np.vstack([x1, x2])
    scaler.fit(stacked)
    x1n = scaler.transform(x1)
    x2n = scaler.transform(x2)

    ds = PairDataset(x1n, x2n, y)
    dl = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, drop_last=False)

    model = SiameseEncoder(in_dim=x1.shape[1], emb_dim=emb_dim).to(device)
    criterion = ContrastiveLoss(margin=margin)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for xa, xb, lbl in dl:
            xa = xa.to(device)
            xb = xb.to(device)
            lbl = lbl.to(device)

            za = model(xa)
            zb = model(xb)
            loss = criterion(za, zb, lbl)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * xa.shape[0]

        epoch_loss = total / max(1, len(ds))
        losses.append(epoch_loss)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            LOG.info("Siamese epoch %d/%d loss=%.6f", epoch + 1, epochs, epoch_loss)

    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / "siamese_encoder.pt"
    torch.save({"state_dict": model.state_dict(), "in_dim": x1.shape[1], "emb_dim": emb_dim}, ckpt_path)

    plt.figure(figsize=(7, 4))
    plt.plot(losses, color="tab:blue", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Contrastive Loss")
    plt.title("Siamese Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(models_dir / "training_loss.png", dpi=140)
    plt.close()

    return SiameseModelBundle(model=model, scaler=scaler, train_loss=np.array(losses, dtype=np.float32))


def encode_descriptors(
    descriptors: np.ndarray,
    bundle: SiameseModelBundle,
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Encode descriptors into learned embedding space."""
    x = bundle.scaler.transform(descriptors)
    model = bundle.model.to(device)
    model.eval()

    all_emb = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch = torch.from_numpy(x[start : start + batch_size].astype(np.float32)).to(device)
            z = model(batch).cpu().numpy()
            all_emb.append(z)
    return np.vstack(all_emb).astype(np.float32)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix for embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    normed = embeddings / norms
    sim = normed @ normed.T
    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float32)
