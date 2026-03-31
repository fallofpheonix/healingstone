"""Canny edge detection and contour extraction for 2D fragment images.

Provides:
- EdgeBundle            - Dataclass for edge/contour data
- detect_edges()       - Canny edge map constrained to the fragment mask
- extract_contours()   - Contour extraction and selection of the main contour
- extract_break_contour()- Heuristic identification of break (fracture) edges
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

LOG = logging.getLogger(__name__)

# Default Canny thresholds; auto-calibrated if sigma parameter is provided.
_CANNY_LOW_DEFAULT = 50
_CANNY_HIGH_DEFAULT = 150

# Minimum contour arc-length (in pixels) to be considered significant.
_MIN_CONTOUR_LENGTH = 20


@dataclass
class EdgeBundle:
    """Edge and contour data extracted from one fragment image."""

    edge_map: np.ndarray  # uint8 binary edge map, shape (H, W)
    contours: List[np.ndarray]  # list of (N, 1, 2) int32 contour point arrays (OpenCV format)
    boundary_points: np.ndarray  # (M, 2) float32 array of all boundary points
    main_contour: Optional[np.ndarray] = None


def _auto_canny(gray: np.ndarray, sigma: float = 0.33) -> Tuple[int, int]:
    """Compute Canny thresholds automatically from image median (Cohen's rule)."""
    med = float(np.median(gray))
    low = max(0, int((1.0 - sigma) * med))
    high = min(255, int((1.0 + sigma) * med))
    return low, high


def compute_edge_map(
    gray: np.ndarray,
    binary_mask: Optional[np.ndarray] = None,
    low_threshold: Optional[int] = None,
    high_threshold: Optional[int] = None,
    aperture_size: int = 3,
    auto_sigma: float = 0.33,
) -> np.ndarray:
    """Detect edges in *gray* using the Canny algorithm."""
    if low_threshold is None or high_threshold is None:
        low_threshold, high_threshold = _auto_canny(gray, sigma=auto_sigma)

    edges = cv2.Canny(gray, low_threshold, high_threshold, apertureSize=aperture_size)

    if binary_mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=binary_mask)

    LOG.debug(
        "Canny edge detection: thresholds=(%d, %d), edge pixels=%d",
        low_threshold,
        high_threshold,
        int(np.sum(edges > 0)),
    )
    return edges


def extract_contours(
    edges: np.ndarray,
    binary_mask: Optional[np.ndarray] = None,
    min_length: int = _MIN_CONTOUR_LENGTH,
) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """Extract contours from an edge map."""
    search_image = edges.copy()
    if binary_mask is not None:
        # Dilate mask slightly so the fragment boundary itself is included.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        search_image = cv2.bitwise_and(search_image, search_image, mask=dilated_mask)

    raw_contours, _ = cv2.findContours(search_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours = [c for c in raw_contours if cv2.arcLength(c, closed=False) >= min_length]
    contours = sorted(contours, key=lambda c: cv2.arcLength(c, closed=False), reverse=True)

    main_contour: Optional[np.ndarray] = contours[0] if contours else None
    LOG.debug("Extracted %d contours (min_length=%d)", len(contours), min_length)
    return contours, main_contour


def extract_break_contour(
    contours: List[np.ndarray],
    binary_mask: np.ndarray,
) -> Optional[np.ndarray]:
    """Heuristically select the break (fracture) contour from all contours."""
    if not contours:
        return None

    hull_contours, _ = cv2.findContours(
        binary_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not hull_contours:
        return contours[0]

    hull_len = max(cv2.arcLength(hc, closed=True) for hc in hull_contours)

    for c in contours:
        arc = cv2.arcLength(c, closed=False)
        if arc < hull_len * 0.9:
            return c

    return contours[0]


def detect_edges(
    gray: np.ndarray,
    binary_mask: Optional[np.ndarray] = None,
) -> EdgeBundle:
    """Run full edge detection and contour extraction pipeline on one fragment."""
    edge_map = compute_edge_map(gray, binary_mask=binary_mask)
    contours, main_contour = extract_contours(edge_map, binary_mask=binary_mask)

    if contours:
        # Convert OpenCV contours (N, 1, 2) to flat point sets for boundary_points
        boundary_pts_list = [c.reshape(-1, 2) for c in contours]
        boundary_points = np.vstack(boundary_pts_list).astype(np.float32)
    else:
        ys, xs = np.where(edge_map > 0)
        if ys.size > 0:
            boundary_points = np.column_stack([xs, ys]).astype(np.float32)
        else:
            boundary_points = np.zeros((0, 2), dtype=np.float32)

    return EdgeBundle(
        edge_map=edge_map,
        contours=contours,
        boundary_points=boundary_points,
        main_contour=main_contour,
    )
