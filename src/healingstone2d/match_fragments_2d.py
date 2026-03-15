"""Fragment similarity matching for the 2D pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import]

from .preprocess_2d import Fragment2D
from .shape_descriptors import ShapeDescriptor

LOG = logging.getLogger(__name__)


@dataclass
class MatchResult2D:
    """Pairwise match score for two 2D fragments."""

    i: int
    j: int
    score: float  # in [0, 1]; higher is more similar


def _descriptor_matrix(descriptors: Dict[int, ShapeDescriptor], n: int) -> np.ndarray:
    rows = []
    for i in range(n):
        if i not in descriptors:
            raise KeyError(f"Missing descriptor for fragment idx {i}")
        rows.append(descriptors[i].descriptor.astype(np.float32))
    return np.vstack(rows)


def compute_similarity_matrix(
    descriptors: Dict[int, ShapeDescriptor],
    n: int,
) -> np.ndarray:
    """Compute pairwise cosine-similarity matrix for all fragment descriptors."""
    mat = _descriptor_matrix(descriptors, n)
    sim = cosine_similarity(mat).astype(np.float32)
    # Clip to [0, 1] since descriptors are non-negative.
    sim = np.clip((sim + 1.0) / 2.0, 0.0, 1.0)
    return sim


def select_candidate_pairs(
    similarity: np.ndarray,
    top_k: int = 4,
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
    """Select top-k candidate pairs per fragment using similarity matrix.

    Returns:
        candidate_pairs: sorted list of (i, j) unique pairs
        pair_scores: mapping from (i, j) to similarity score
    """
    n = similarity.shape[0]
    top_k = min(top_k, n - 1)
    candidates: set[Tuple[int, int]] = set()
    pair_scores: Dict[Tuple[int, int], float] = {}

    for i in range(n):
        row = similarity[i].copy()
        row[i] = -1.0  # exclude self
        ranked = np.argsort(row)[::-1][:top_k]
        for j in ranked:
            pair = (min(int(i), int(j)), max(int(i), int(j)))
            candidates.add(pair)
            score = float(similarity[i, j])
            if pair not in pair_scores or pair_scores[pair] < score:
                pair_scores[pair] = score

    candidate_pairs = sorted(candidates)
    LOG.info("Selected %d candidate pairs from %d fragments", len(candidate_pairs), n)
    return candidate_pairs, pair_scores


def match_fragments(
    fragments: List[Fragment2D],
    descriptors: Dict[int, ShapeDescriptor],
    top_k: int = 4,
) -> Tuple[np.ndarray, List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
    """Run full matching pipeline.

    Returns:
        similarity: (N, N) float32 matrix
        candidate_pairs: list of (i, j) pairs to align
        pair_scores: similarity score per pair
    """
    n = len(fragments)
    similarity = compute_similarity_matrix(descriptors, n)
    candidate_pairs, pair_scores = select_candidate_pairs(similarity, top_k=top_k)
    return similarity, candidate_pairs, pair_scores
