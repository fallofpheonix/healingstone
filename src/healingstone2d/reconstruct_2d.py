"""2D fragment assembly and image reconstruction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .align_fragments_2d import AlignmentResult2D
from .preprocess_2d import Fragment2D

LOG = logging.getLogger(__name__)


@dataclass
class Assembly2DResult:
    """Output of global 2D assembly."""

    global_transforms: Dict[int, np.ndarray]  # fragment idx → 3×3 homogeneous transform
    completeness: float  # fraction of fragments successfully placed


def _spanning_tree_transforms(
    n: int,
    alignments: Dict[Tuple[int, int], AlignmentResult2D],
    pair_scores: Dict[Tuple[int, int], float],
) -> Dict[int, np.ndarray]:
    """Compute global transforms via greedy BFS on the alignment graph."""
    import networkx as nx  # type: ignore[import]

    g: nx.Graph = nx.Graph()
    g.add_nodes_from(range(n))
    for (i, j), a in alignments.items():
        prior = float(pair_scores.get((i, j), pair_scores.get((j, i), 0.0)))
        weight = float(a.inlier_ratio * (prior + 1e-6)) if a.success else 0.0
        if weight > 0:
            g.add_edge(i, j, weight=weight, transform=a.transform)

    global_transforms: Dict[int, np.ndarray] = {}
    # Anchor first connected component at identity.
    for start in range(n):
        if start in global_transforms:
            continue
        if start not in g:
            global_transforms[start] = np.eye(3, dtype=np.float32)
            continue
        global_transforms[start] = np.eye(3, dtype=np.float32)
        for u, v in nx.bfs_edges(g, source=start):
            edge_data = g.edges[u, v]
            T_edge = np.asarray(edge_data.get("transform", np.eye(3)), dtype=np.float32)
            # Compose: global[v] = global[u] @ T_edge
            parent_T = global_transforms.get(u, np.eye(3, dtype=np.float32))
            global_transforms[v] = parent_T @ T_edge

    # Any remaining isolated nodes get identity.
    for i in range(n):
        if i not in global_transforms:
            global_transforms[i] = np.eye(3, dtype=np.float32)

    return global_transforms


def assemble_fragments_2d(
    fragments: List[Fragment2D],
    alignments: Dict[Tuple[int, int], AlignmentResult2D],
    pair_scores: Dict[Tuple[int, int], float],
) -> Assembly2DResult:
    """Compute global 2D transforms for all fragments."""
    n = len(fragments)
    global_transforms = _spanning_tree_transforms(n, alignments, pair_scores)
    # Count fragments that are connected to at least one other via a successful alignment.
    connected: set[int] = set()
    for (i, j), a in alignments.items():
        if a.success:
            connected.add(i)
            connected.add(j)
    placed = len(connected) if connected else n
    completeness = float(placed) / max(1, n)
    LOG.info("2D assembly: %d/%d fragments placed (completeness=%.2f)", placed, n, completeness)
    return Assembly2DResult(
        global_transforms=global_transforms,
        completeness=completeness,
    )


def _apply_transform_2d(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    homo = np.hstack([points, np.ones((n, 1), dtype=np.float32)])
    return (homo @ T.T)[:, :2]


def _warp_image(image: np.ndarray, T: np.ndarray, canvas_size: Tuple[int, int]) -> np.ndarray:
    """Warp a grayscale image by a 2D homogeneous transform onto a canvas."""
    try:
        import cv2  # type: ignore[import]

        h_out, w_out = canvas_size
        M = T[:2, :].astype(np.float32)
        warped = cv2.warpAffine(image, M, (w_out, h_out), flags=cv2.INTER_LINEAR, borderValue=0)
        return warped
    except ImportError:
        from skimage.transform import AffineTransform, warp  # type: ignore[import]

        h_out, w_out = canvas_size
        # skimage uses inverse mapping.
        try:
            T_inv = np.linalg.inv(T)
        except np.linalg.LinAlgError:
            T_inv = np.eye(3, dtype=np.float32)
        aff = AffineTransform(matrix=T_inv.astype(np.float64))
        warped = warp(
            image.astype(np.float32) / 255.0,
            aff,
            output_shape=(h_out, w_out),
            order=1,
            preserve_range=True,
        )
        return (warped * 255).clip(0, 255).astype(np.uint8)


def render_reconstruction(
    fragments: List[Fragment2D],
    global_transforms: Dict[int, np.ndarray],
    output_path: Path,
) -> np.ndarray:
    """Render assembled fragments onto a shared canvas and save."""
    if not fragments:
        return np.zeros((64, 64), dtype=np.uint8)

    # Determine canvas size: transform corners of each image, find bounding box.
    all_corners: List[np.ndarray] = []
    for frag in fragments:
        h, w = frag.image.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        T = global_transforms.get(frag.idx, np.eye(3, dtype=np.float32))
        transformed = _apply_transform_2d(corners, T)
        all_corners.append(transformed)

    all_pts = np.vstack(all_corners)
    min_x = float(np.floor(np.min(all_pts[:, 0])))
    min_y = float(np.floor(np.min(all_pts[:, 1])))
    max_x = float(np.ceil(np.max(all_pts[:, 0])))
    max_y = float(np.ceil(np.max(all_pts[:, 1])))

    canvas_w = max(1, int(max_x - min_x))
    canvas_h = max(1, int(max_y - min_y))
    # Clamp canvas to a reasonable size.
    canvas_w = min(canvas_w, 4096)
    canvas_h = min(canvas_h, 4096)

    # Translate transforms so (min_x, min_y) maps to (0, 0).
    offset = np.eye(3, dtype=np.float32)
    offset[0, 2] = -min_x
    offset[1, 2] = -min_y

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    counts = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for frag in fragments:
        T = global_transforms.get(frag.idx, np.eye(3, dtype=np.float32))
        T_canvas = offset @ T
        warped = _warp_image(frag.image, T_canvas, canvas_size=(canvas_h, canvas_w))
        mask = warped > 0
        canvas[mask] += warped[mask].astype(np.float32)
        counts[mask] += 1.0

    # Average overlapping pixels.
    with np.errstate(invalid="ignore"):
        result = np.where(counts > 0, canvas / np.maximum(counts, 1.0), 0.0)
    result = result.clip(0, 255).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_image(result, output_path)
    LOG.info("Reconstruction saved to %s", output_path)
    return result


def _save_image(image: np.ndarray, path: Path) -> None:
    try:
        import cv2  # type: ignore[import]

        cv2.imwrite(str(path), image)
        return
    except ImportError:
        pass

    from PIL import Image  # type: ignore[import]

    Image.fromarray(image).save(str(path))
