"""2D fragment reconstruction: canvas assembly and image output.

Provides:
- assemble_reconstruction()   – place aligned fragments on a shared canvas
- save_reconstruction()       – write the assembled image to disk
- run_2d_pipeline()           – end-to-end 2D pipeline entry point
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple
from collections import deque

import cv2
import numpy as np

LOG = logging.getLogger(__name__)

# Padding (pixels) added around each side of the assembled canvas.
_CANVAS_PADDING = 64


def _apply_transform_to_image(
    img: np.ndarray,
    transform: np.ndarray,
    canvas_size: Tuple[int, int],
) -> np.ndarray:
    """Warp a grayscale image using a (3, 3) homogeneous rigid transform.

    Parameters
    ----------
    img:
        Source grayscale image (uint8).
    transform:
        (3, 3) homogeneous transform matrix.
    canvas_size:
        ``(width, height)`` of the output canvas.

    Returns
    -------
    np.ndarray
        Warped image on the canvas, uint8.
    """
    # OpenCV warpAffine takes a (2, 3) affine matrix.
    M = transform[:2, :]
    warped = cv2.warpAffine(
        img, M, canvas_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,    # white background
    )
    return warped


def _build_canvas_transform(
    frag,
    base_transform: np.ndarray,
    canvas_offset: np.ndarray,
) -> np.ndarray:
    """Compose the fragment-specific transform with the canvas offset translation.

    Parameters
    ----------
    frag:
        :class:`~healingstone2d.preprocess_2d.Fragment2D` (used for shape only).
    base_transform:
        (3, 3) homogeneous rigid transform for this fragment.
    canvas_offset:
        (2,) translation vector (e.g. padding offset) applied uniformly to
        all fragments on the canvas.

    Returns
    -------
    np.ndarray
        Composed (3, 3) homogeneous transform.
    """
    T_offset = np.eye(3, dtype=np.float64)
    T_offset[0, 2] = canvas_offset[0]
    T_offset[1, 2] = canvas_offset[1]
    return T_offset @ base_transform


def assemble_reconstruction(
    fragments: list,
    alignments: dict,
    root_idx: Optional[int] = None,
) -> np.ndarray:
    """Place all fragments on a shared canvas using estimated transforms.

    A root fragment (default: the one with the most successful alignments)
    is chosen, and all fragments are placed on a shared canvas using the
    estimated transforms that align them relative to this root.

    Parameters
    ----------
    fragments:
        List of :class:`~healingstone2d.preprocess_2d.Fragment2D`.
    alignments:
        Mapping ``(fi, fj) → AlignmentResult2D`` from the alignment stage.
    root_idx:
        Fragment index to treat as the root.  Inferred automatically if *None*.

    Returns
    -------
    np.ndarray
        Assembled grayscale image (uint8).
    """
    if not fragments:
        return np.full((256, 256), 255, dtype=np.uint8)

    frag_by_idx = {f.idx: f for f in fragments}
    n_frags = len(fragments)

    # ------------------------------------------------------------------ #
    # Select root: fragment involved in most successful alignments.        #
    # ------------------------------------------------------------------ #
    if root_idx is None:
        success_count: Dict[int, int] = {f.idx: 0 for f in fragments}
        for (fi, fj), res in alignments.items():
            if res.success:
                # Use directed fragment indices from the alignment result when available.
                frag_i = getattr(res, "frag_idx_i", fi)
                frag_j = getattr(res, "frag_idx_j", fj)
                success_count[frag_i] = success_count.get(frag_i, 0) + 1
                success_count[frag_j] = success_count.get(frag_j, 0) + 1
        root_idx = max(success_count, key=lambda k: success_count[k])

    LOG.info("Assembling 2D reconstruction with root fragment idx=%d", root_idx)

    # ------------------------------------------------------------------ #
    # Compute global transforms via BFS from root.                        #
    # ------------------------------------------------------------------ #
    global_transforms: Dict[int, np.ndarray] = {root_idx: np.eye(3, dtype=np.float64)}
    queue: Deque[int] = deque([root_idx])
    visited: set = {root_idx}

    while queue:
        current = queue.popleft()
        for (fi, fj), res in alignments.items():
            if not res.success:
                continue
            neighbour: Optional[int] = None
            # Determine directed endpoints from the alignment result. Fall back to the
            # key ordering if explicit fragment indices are not present.
            frag_i = getattr(res, "frag_idx_i", fi)
            frag_j = getattr(res, "frag_idx_j", fj)
            if frag_i == current and frag_j not in visited:
                neighbour = frag_j
                T_neighbour = res.transform @ global_transforms[current]
            elif frag_j == current and frag_i not in visited:
                neighbour = frag_i
                T_neighbour = np.linalg.inv(res.transform) @ global_transforms[current]

            if neighbour is not None and neighbour in frag_by_idx:
                global_transforms[neighbour] = T_neighbour
                visited.add(neighbour)
                queue.append(neighbour)

    # Fragments not reached by BFS keep the identity transform.
    for frag in fragments:
        if frag.idx not in global_transforms:
            global_transforms[frag.idx] = np.eye(3, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Estimate canvas size from all transformed fragment extents.          #
    # ------------------------------------------------------------------ #
    all_h = max(f.gray.shape[0] for f in fragments)
    all_w = max(f.gray.shape[1] for f in fragments)
    canvas_w = all_w * int(math.ceil(math.sqrt(n_frags))) + 2 * _CANVAS_PADDING
    canvas_h = all_h * int(math.ceil(math.sqrt(n_frags))) + 2 * _CANVAS_PADDING

    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    offset = np.array([_CANVAS_PADDING, _CANVAS_PADDING], dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Place each fragment.                                                 #
    # ------------------------------------------------------------------ #
    for frag in fragments:
        T = _build_canvas_transform(frag, global_transforms[frag.idx], offset)
        warped = _apply_transform_to_image(frag.gray, T, canvas_size=(canvas_w, canvas_h))

        # Composite: only write non-white (fragment) pixels.
        mask = warped < 250
        canvas[mask] = np.minimum(canvas[mask], warped[mask])

    return canvas


def save_reconstruction(assembled: np.ndarray, output_path: Path) -> None:
    """Write the assembled reconstruction image to disk.

    Parameters
    ----------
    assembled:
        Grayscale reconstruction canvas (uint8).
    output_path:
        Destination file path (e.g. ``*.png``).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), assembled)
    LOG.info("Saved 2D reconstruction to %s", output_path)


def _summarize_2d_metrics(
    fragments: list,
    alignments: dict,
    n_descriptors: int,
) -> Dict[str, Any]:
    """Compute and return summary metrics for the 2D pipeline run."""
    n_success = sum(1 for r in alignments.values() if r.success)
    rmse_vals = [r.rmse for r in alignments.values() if r.success and math.isfinite(r.rmse)]
    return {
        "n_fragments": len(fragments),
        "n_descriptors": n_descriptors,
        "n_candidate_pairs": len(alignments),
        "n_successful_alignments": n_success,
        "mean_alignment_rmse": float(np.mean(rmse_vals)) if rmse_vals else float("nan"),
        "pipeline_mode": "2d",
    }


def run_2d_pipeline(
    data_dir: Path,
    output_dir: Path,
    seed: int = 42,
    target_size: Optional[Tuple[int, int]] = None,
    top_k: int = 3,
    align_top_n: int = 20,
    rmse_threshold: float = 20.0,
) -> Dict[str, Any]:
    """End-to-end 2D reconstruction pipeline.

    Parameters
    ----------
    data_dir:
        Directory containing fragment ``.png`` images.
    output_dir:
        Directory where outputs are written.
    seed:
        Random seed for deterministic execution.
    target_size:
        Optional ``(W, H)`` to resize all images to a common size.
    top_k:
        Reciprocal top-k for candidate pair selection.
    align_top_n:
        Maximum number of pairs to align.
    rmse_threshold:
        RMSE threshold (pixels) for alignment success.

    Returns
    -------
    Dict[str, Any]
        Metrics and paths summary.
    """
    import random

    import numpy as np

    from healingstone2d.align_fragments_2d import align_candidate_pairs_2d
    from healingstone2d.match_fragments_2d import match_all_fragments
    from healingstone2d.preprocess_2d import load_and_preprocess_fragments_2d
    from healingstone2d.shape_descriptors import extract_all_descriptors

    # Deterministic seed.
    random.seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("[2D pipeline] Loading and preprocessing fragments from %s", data_dir)
    fragments = load_and_preprocess_fragments_2d(data_dir, target_size=target_size)

    LOG.info("[2D pipeline] Extracting shape descriptors")
    descriptors = extract_all_descriptors(fragments)

    LOG.info("[2D pipeline] Matching fragments")
    similarity, candidate_matches, pair_scores = match_all_fragments(descriptors, top_k=top_k)

    LOG.info("[2D pipeline] Aligning candidate pairs")
    alignments = align_candidate_pairs_2d(
        fragments=fragments,
        candidate_matches=candidate_matches,
        rmse_threshold=rmse_threshold,
        top_n=align_top_n,
    )

    LOG.info("[2D pipeline] Assembling reconstruction")
    assembled = assemble_reconstruction(fragments, alignments)

    out_image = output_dir / "reconstructed_2d.png"
    save_reconstruction(assembled, out_image)

    metrics = _summarize_2d_metrics(fragments, alignments, n_descriptors=len(descriptors))
    metrics_path = output_dir / "metrics_2d.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    LOG.info("[2D pipeline] Complete. Outputs in %s", output_dir)
    return metrics
"""2D fragment assembly and image reconstruction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
    """Compute global transforms via maximum spanning tree on the alignment graph."""
    import networkx as nx  # type: ignore[import]

    g: nx.Graph = nx.Graph()
    g.add_nodes_from(range(n))
    for (i, j), a in alignments.items():
        prior = float(pair_scores.get((i, j), pair_scores.get((j, i), 0.0)))
        weight = float(a.inlier_ratio * (prior + 1e-6)) if a.success else 0.0
        if weight > 0:
            g.add_edge(i, j, weight=weight, transform=a.transform)

    # Build maximum spanning tree to select best edges.
    if g.number_of_edges() > 0:
        mst = nx.maximum_spanning_tree(g, weight="weight")
    else:
        mst = nx.Graph()
        mst.add_nodes_from(range(n))

    global_transforms: Dict[int, np.ndarray] = {}
    # Traverse MST using BFS to propagate transforms.
    for start in range(n):
        if start in global_transforms:
            continue
        if start not in mst:
            global_transforms[start] = np.eye(3, dtype=np.float32)
            continue
        global_transforms[start] = np.eye(3, dtype=np.float32)
        for u, v in nx.bfs_edges(mst, source=start):
            # Determine the transform in the traversal direction u → v.
            if (u, v) in alignments:
                a = alignments[(u, v)]
                T_edge = np.asarray(a.transform, dtype=np.float32)
            elif (v, u) in alignments:
                a = alignments[(v, u)]
                T_raw = np.asarray(a.transform, dtype=np.float32)
                try:
                    T_edge = np.linalg.inv(T_raw)
                except np.linalg.LinAlgError:
                    LOG.warning(
                        "Failed to invert transform between fragments %d and %d; using identity.",
                        v,
                        u,
                    )
                    T_edge = np.eye(3, dtype=np.float32)
            else:
                # Edge exists in MST but not in alignments; shouldn't happen.
                LOG.warning("MST edge (%d, %d) missing in alignments; using identity.", u, v)
                T_edge = np.eye(3, dtype=np.float32)
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
    placed = len(connected)
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

        success = cv2.imwrite(str(path), image)
        if not success:
            LOG.warning("cv2.imwrite failed for %s; falling back to PIL", path)
            raise RuntimeError("cv2.imwrite failed")
        return
    except (ImportError, RuntimeError):
        pass

    from PIL import Image  # type: ignore[import]

    Image.fromarray(image).save(str(path))
