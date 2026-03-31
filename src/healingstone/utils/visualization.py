"""Plotting helpers for pipeline diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..core.geometry.align_fragments import AlignmentResult
    from ..core.geometry.preprocess import Fragment


def _plt():
    import matplotlib.pyplot as plt

    return plt


def plot_similarity_matrix(similarity: np.ndarray, fragments: List["Fragment"], out_path: Path) -> None:
    plt = _plt()
    labels = [fragment.name for fragment in fragments]
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(similarity, cmap="viridis", vmin=-1, vmax=1)
    fig.colorbar(image, ax=ax, label="Learned Similarity")
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
    fragments: List["Fragment"],
    alignments: Dict[Tuple[int, int], "AlignmentResult"],
    output_dir: Path,
    max_plots: int = 4,
) -> None:
    plt = _plt()
    ordered = sorted(
        alignments.values(),
        key=lambda result: (result.success, -result.score_prior, -result.fitness),
        reverse=True,
    )[:max_plots]

    by_idx = {fragment.idx: fragment for fragment in fragments}
    for rank, result in enumerate(ordered, start=1):
        source_points = by_idx[result.i].points
        target_points = by_idx[result.j].points
        aligned_points = _apply_transform(source_points, result.transform_ij)

        fig = plt.figure(figsize=(12, 5))
        ax_before = fig.add_subplot(1, 2, 1, projection="3d")
        ax_before.scatter(
            source_points[::8, 0],
            source_points[::8, 1],
            source_points[::8, 2],
            s=1,
            alpha=0.5,
            c="tomato",
            label=by_idx[result.i].name,
        )
        ax_before.scatter(
            target_points[::8, 0],
            target_points[::8, 1],
            target_points[::8, 2],
            s=1,
            alpha=0.5,
            c="steelblue",
            label=by_idx[result.j].name,
        )
        ax_before.set_title("Before Alignment")
        ax_before.legend(fontsize=7)

        ax_after = fig.add_subplot(1, 2, 2, projection="3d")
        ax_after.scatter(
            aligned_points[::8, 0],
            aligned_points[::8, 1],
            aligned_points[::8, 2],
            s=1,
            alpha=0.5,
            c="tomato",
            label=f"{by_idx[result.i].name} aligned",
        )
        ax_after.scatter(
            target_points[::8, 0],
            target_points[::8, 1],
            target_points[::8, 2],
            s=1,
            alpha=0.5,
            c="steelblue",
            label=by_idx[result.j].name,
        )
        ax_after.set_title(f"After Alignment\nRMSE={result.inlier_rmse:.4f}, fitness={result.fitness:.3f}")
        ax_after.legend(fontsize=7)

        plt.tight_layout()
        output_path = output_dir / (
            f"alignment_pair_{rank}_{by_idx[result.i].name}_{by_idx[result.j].name}.png"
        )
        fig.savefig(output_path, dpi=140)
        plt.close(fig)


def plot_final_reconstruction(points: np.ndarray, out_path: Path) -> None:
    plt = _plt()
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    sampled = points[:: max(1, points.shape[0] // 15000)]
    ax.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2], s=1, alpha=0.6, c=sampled[:, 2], cmap="viridis")
    ax.set_title("Final Reconstructed Artifact")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
