"""Naive baseline matchers for context and evaluation."""

from __future__ import annotations

import random
import numpy as np
from typing import Any, Dict, List, Set, Tuple

class RandomMatcher:
    """
    MANDATORY: Random baseline comparison.
    """
    def __init__(self, fragments: List[Any]):
        self.fragments = fragments

    def match(self) -> List[Tuple[int, int]]:
        """Predict random fragment pairs."""
        pairs = []
        for i in range(len(self.fragments)):
            j = random.randint(0, len(self.fragments) - 1)
            if i != j:
                pairs.append((i, j))
        return pairs

class CentroidDistanceMatcher:
    """
    MANDATORY: Heuristic baseline comparison based on centroid distance.
    """
    def __init__(self, fragments: List[Any], threshold: float = 1.0):
        self.fragments = fragments
        self.threshold = threshold

    def match(self) -> List[Tuple[int, int]]:
        """Predict matches based on centroid proximity."""
        pairs = []
        for i in range(len(self.fragments)):
            for j in range(i + 1, len(self.fragments)):
                # Assume fragments have a .centroid attribute or (N,3) points
                # For this implementation, we use the mean of points
                c_i = np.mean(self.fragments[i].points, axis=0) if hasattr(self.fragments[i], 'points') else np.zeros(3)
                c_j = np.mean(self.fragments[j].points, axis=0) if hasattr(self.fragments[j], 'points') else np.zeros(3)
                
                dist = np.linalg.norm(c_i - c_j)
                if dist < self.threshold:
                    pairs.append((i, j))
        return pairs
