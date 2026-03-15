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
