"""Canny edge detection and contour extraction for 2D fragments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

LOG = logging.getLogger(__name__)


@dataclass
class EdgeBundle:
    """Edge and contour data extracted from one fragment image."""

    edge_map: np.ndarray  # uint8 binary edge map, shape (H, W)
    contours: List[np.ndarray]  # list of (N, 2) float32 contour point arrays
    boundary_points: np.ndarray  # (M, 2) float32 array of all boundary points


def _canny_cv2(image: np.ndarray, low: int, high: int) -> np.ndarray:
    import cv2  # type: ignore[import]

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, low, high)
    return edges


def _canny_skimage(image: np.ndarray, sigma: float = 1.4) -> np.ndarray:
    from skimage.feature import canny  # type: ignore[import]

    edge_bool = canny(image.astype(np.float32) / 255.0, sigma=sigma)
    return (edge_bool.astype(np.uint8)) * 255


def compute_edge_map(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """Compute Canny edge map from a grayscale image."""
    try:
        return _canny_cv2(image, low=low_threshold, high=high_threshold)
    except ImportError:
        LOG.debug("cv2 not available; falling back to skimage Canny")
        return _canny_skimage(image)


def _extract_contours_cv2(edge_map: np.ndarray) -> List[np.ndarray]:
    import cv2  # type: ignore[import]

    contours_raw, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    result: List[np.ndarray] = []
    for c in contours_raw:
        pts = c.reshape(-1, 2).astype(np.float32)
        if pts.shape[0] >= 5:
            result.append(pts)
    return result


def _extract_contours_skimage(edge_map: np.ndarray) -> List[np.ndarray]:
    from skimage.measure import find_contours  # type: ignore[import]

    raw = find_contours(edge_map.astype(float), level=127.5)
    result: List[np.ndarray] = []
    for c in raw:
        pts = c[:, ::-1].astype(np.float32)  # (row, col) → (x, y)
        if pts.shape[0] >= 5:
            result.append(pts)
    return result


def extract_contours(edge_map: np.ndarray) -> List[np.ndarray]:
    """Extract contours from a binary edge map."""
    try:
        return _extract_contours_cv2(edge_map)
    except ImportError:
        LOG.debug("cv2 not available; falling back to skimage contour extraction")
        return _extract_contours_skimage(edge_map)


def detect_edges(image: np.ndarray) -> EdgeBundle:
    """Run full edge detection and contour extraction pipeline on one fragment."""
    edge_map = compute_edge_map(image)
    contours = extract_contours(edge_map)

    if contours:
        boundary_points = np.vstack(contours)
    else:
        # Fallback: use all non-zero edge pixels.
        ys, xs = np.where(edge_map > 0)
        if ys.size > 0:
            boundary_points = np.column_stack([xs, ys]).astype(np.float32)
        else:
            boundary_points = np.zeros((0, 2), dtype=np.float32)

    return EdgeBundle(
        edge_map=edge_map,
        contours=contours,
        boundary_points=boundary_points,
    )
