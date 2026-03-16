"""2D fragment reconstruction: canvas assembly and image output.

Provides:
- assemble_fragments_2d()   – place aligned fragments on a shared canvas (MST)
- render_reconstruction()     – write the assembled image to disk
- run_2d_pipeline()           – end-to-end 2D pipeline entry point
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .align_fragments_2d import AlignmentResult2D
from .preprocess_2d import Fragment2D

LOG = logging.getLogger(__name__)


def _spanning_tree_transforms(
    n: int,
    alignments: Dict[Tuple[int, int], AlignmentResult2D],
    pair_scores: Dict[Tuple[int, int], float],
) -> Dict[int, np.ndarray]:
    """Compute global transforms via maximum spanning tree on the alignment graph."""
    import networkx as nx

    g: nx.Graph = nx.Graph()
    g.add_nodes_from(range(n))
    for (i, j), a in alignments.items():
        prior = float(pair_scores.get((i, j), pair_scores.get((j, i), 0.0)))
        weight = float(a.inlier_ratio * (prior + 1e-6)) if a.success else 0.0
        if weight > 0:
            g.add_edge(i, j, weight=weight, transform=a.transform)

    if g.number_of_edges() > 0:
        mst = nx.maximum_spanning_tree(g, weight="weight")
    else:
        mst = nx.Graph()
        mst.add_nodes_from(range(n))

    global_transforms: Dict[int, np.ndarray] = {}
    for start in range(n):
        if start in global_transforms:
            continue
        global_transforms[start] = np.eye(3, dtype=np.float32)
        for u, v in nx.bfs_edges(mst, source=start):
            if (u, v) in alignments:
                a = alignments[(u, v)]
                T_edge = a.transform
            else:
                a = alignments[(v, u)]
                T_edge = np.linalg.inv(a.transform)
            
            global_transforms[v] = global_transforms[u] @ T_edge.astype(np.float32)

    return global_transforms


def assemble_fragments_2d(
    fragments: List[Fragment2D],
    alignments: Dict[Tuple[int, int], AlignmentResult2D],
    pair_scores: Dict[Tuple[int, int], float],
) -> Any:
    """Compute global 2D transforms for all fragments."""
    from dataclasses import dataclass

    @dataclass
    class Assembly2DResult:
        global_transforms: Dict[int, np.ndarray]
        completeness: float

    n = len(fragments)
    global_transforms = _spanning_tree_transforms(n, alignments, pair_scores)
    
    connected: set[int] = set()
    for (i, j), a in alignments.items():
        if a.success:
            connected.add(i)
            connected.add(j)
    
    return Assembly2DResult(
        global_transforms=global_transforms,
        completeness=len(connected) / max(1, n)
    )


def _apply_transform_2d(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    homo = np.hstack([points, np.ones((n, 1), dtype=np.float32)])
    return (homo @ T.T)[:, :2]


def render_reconstruction(
    fragments: List[Fragment2D],
    global_transforms: Dict[int, np.ndarray],
    output_path: Path,
) -> np.ndarray:
    """Render assembled fragments onto a shared canvas and save."""
    if not fragments:
        return np.zeros((64, 64), dtype=np.uint8)

    all_corners: List[np.ndarray] = []
    for frag in fragments:
        h, w = frag.gray.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        T = global_transforms.get(frag.idx, np.eye(3, dtype=np.float32))
        all_corners.append(_apply_transform_2d(corners, T))

    all_pts = np.vstack(all_corners)
    min_x, min_y = np.floor(np.min(all_pts, axis=0))
    max_x, max_y = np.ceil(np.max(all_pts, axis=0))

    canvas_w: int, canvas_h = int(max_x - min_x) + 1, int(max_y - min_y) + 1
    canvas_w, canvas_h = min(canvas_w, 4096), min(canvas_h, 4096)

    offset: np.ndarray = np.eye(3, dtype=np.float32)
    offset[0, 2], offset[1, 2] = -min_x, -min_y

    canvas: np.ndarray = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    for frag in fragments:
        T = offset @ global_transforms.get(frag.idx, np.eye(3, dtype=np.float32))
        warped = cv2.warpAffine(frag.gray, T[:2, :], (canvas_w, canvas_h), borderValue=255)
        mask = warped < 250
        canvas[mask] = np.minimum(canvas[mask], warped[mask])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return canvas


def run_2d_pipeline(
    data_dir: Path,
    output_dir: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """End-to-end 2D reconstruction pipeline."""
    from .align_fragments_2d import align_candidate_pairs_2d
    from .match_fragments_2d import match_all_fragments
    from .preprocess_2d import load_and_preprocess_fragments_2d, set_deterministic_seed_2d
    from .shape_descriptors import extract_all_descriptors

    set_deterministic_seed_2d(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fragments = load_and_preprocess_fragments_2d(data_dir)
    descriptors = extract_all_descriptors(fragments)
    _, candidate_matches, pair_scores = match_all_fragments(descriptors)
    
    candidate_pairs = [(m.frag_idx_i, m.frag_idx_j) for m in candidate_matches]
    # Need edge bundles for align_candidate_pairs_2d in the actual API
    from .edge_detection import EdgeBundle
    edge_bundles = {f.idx: EdgeBundle(f.edges, f.contours, f.edges.reshape(-1, 2), f.main_contour) for f in fragments}

    alignments = align_candidate_pairs_2d(fragments, edge_bundles, candidate_pairs, pair_scores)
    assembly = assemble_fragments_2d(fragments, alignments, pair_scores)
    render_reconstruction(fragments, assembly.global_transforms, output_dir / "reconstructed_2d.png")
    
    return {"status": "success", "completeness": assembly.completeness}
