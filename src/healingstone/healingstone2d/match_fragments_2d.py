"""Fragment similarity matching for the 2D pipeline.

Provides:
- FragmentMatch           – Data container for candidate pairs
- match_all_fragments()    – Full matching pipeline
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .shape_descriptors import ShapeDescriptor

LOG = logging.getLogger(__name__)


@dataclass
class FragmentMatch:
    """A candidate fragment pair with associated similarity score."""

    i: int          # index into descriptors list
    j: int
    frag_idx_i: int  # original Fragment2D.idx
    frag_idx_j: int
    score: float


def cosine_similarity_2d(descriptors: List[ShapeDescriptor]) -> np.ndarray:
    """Compute an (N, N) pairwise cosine similarity matrix."""
    if not descriptors:
        return np.zeros((0, 0), dtype=np.float32)

    X = np.vstack([d.descriptor for d in descriptors]).astype(np.float32)

    # L2-normalise rows.
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    X_norm = X / norms

    similarity = X_norm @ X_norm.T
    np.clip(similarity, -1.0, 1.0, out=similarity)
    return similarity


def reciprocal_topk_2d(
    similarity: np.ndarray,
    top_k: int = 3,
) -> List[Tuple[int, int]]:
    """Return reciprocal top-k candidate pairs."""
    n = similarity.shape[0]
    if n < 2:
        return []

    # Build top-k neighbour sets (exclude self).
    topk_sets: List[set] = []
    for i in range(n):
        row = similarity[i].copy()
        row[i] = -np.inf  # exclude self
        neighbours = set(np.argsort(row)[::-1][:top_k])
        topk_sets.append(neighbours)

    pairs: List[Tuple[int, int]] = []
    seen: set = set()
    for i in range(n):
        for j in topk_sets[i]:
            if i in topk_sets[j]:
                key = (min(i, j), max(i, j))
                if key not in seen:
                    seen.add(key)
                    pairs.append(key)

    return sorted(pairs)


def match_all_fragments(
    descriptors: List[ShapeDescriptor],
    top_k: int = 3,
) -> Tuple[np.ndarray, List[FragmentMatch], Dict[Tuple[int, int], float]]:
    """Run full 2D matching: compute similarity and select candidate pairs."""
    similarity = cosine_similarity_2d(descriptors)
    raw_pairs = reciprocal_topk_2d(similarity, top_k=top_k)

    candidate_matches: List[FragmentMatch] = []
    pair_scores: Dict[Tuple[int, int], float] = {}

    for i, j in raw_pairs:
        score = float(similarity[i, j])
        fi = descriptors[i].idx
        fj = descriptors[j].idx
        key = (min(fi, fj), max(fi, fj))
        pair_scores[key] = score
        candidate_matches.append(
            FragmentMatch(i=i, j=j, frag_idx_i=fi, frag_idx_j=fj, score=score)
        )

    # Sort by descending score.
    candidate_matches.sort(key=lambda m: m.score, reverse=True)

    LOG.info(
        "2D matching: %d descriptors → %d candidate pairs",
        len(descriptors),
        len(candidate_matches),
    )
    return similarity, candidate_matches, pair_scores
