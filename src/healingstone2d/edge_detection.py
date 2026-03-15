"""Canny edge detection and contour extraction for 2D fragment images.

Provides:
- detect_edges()       – Canny edge map constrained to the fragment mask
- extract_contours()   – Contour extraction and selection of the main contour
- extract_break_contour() – Heuristic identification of break (fracture) edges
"""
"""Canny edge detection and contour extraction for 2D fragments."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
from dataclasses import dataclass
from typing import List

import numpy as np

LOG = logging.getLogger(__name__)

# Default Canny thresholds; auto-calibrated if sigma parameter is provided.
_CANNY_LOW_DEFAULT = 50
_CANNY_HIGH_DEFAULT = 150

# Minimum contour arc-length (in pixels) to be considered significant.
_MIN_CONTOUR_LENGTH = 20


def _auto_canny(gray: np.ndarray, sigma: float = 0.33) -> Tuple[int, int]:
    """Compute Canny thresholds automatically from image median (Cohen's rule).

    Parameters
    ----------
    gray:
        Grayscale image (uint8).
    sigma:
        Spread factor around the median intensity.

    Returns
    -------
    (low, high) : Tuple[int, int]
        Canny lower and upper thresholds.
    """
    med = float(np.median(gray))
    low = max(0, int((1.0 - sigma) * med))
    high = min(255, int((1.0 + sigma) * med))
    return low, high


def detect_edges(
    gray: np.ndarray,
    binary_mask: Optional[np.ndarray] = None,
    low_threshold: Optional[int] = None,
    high_threshold: Optional[int] = None,
    aperture_size: int = 3,
    auto_sigma: float = 0.33,
) -> np.ndarray:
    """Detect edges in *gray* using the Canny algorithm.

    Parameters
    ----------
    gray:
        Grayscale uint8 image.
    binary_mask:
        Optional fragment mask (uint8, 255 = foreground).  Edges outside the
        mask are zeroed to avoid background noise.
    low_threshold, high_threshold:
        Explicit Canny thresholds.  When *None*, derived automatically using
        *auto_sigma* from the image median.
    aperture_size:
        Sobel kernel size for Canny (3, 5, or 7).
    auto_sigma:
        Sigma for automatic threshold estimation (ignored if explicit
        thresholds are provided).

    Returns
    -------
    np.ndarray
        Binary edge map (uint8, 0 or 255).
    """
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
    """Extract contours from an edge map.

    Uses ``cv2.RETR_EXTERNAL`` to retrieve only outer contours and
    ``cv2.CHAIN_APPROX_NONE`` to preserve every contour point (needed for
    Fourier descriptors).

    Parameters
    ----------
    edges:
        Canny edge map (uint8).
    binary_mask:
        Optional fragment mask.  If provided, the contour search is limited
        to the masked region.
    min_length:
        Minimum contour arc-length (pixels) to retain.

    Returns
    -------
    contours : List[np.ndarray]
        Filtered list of contour arrays (each shape ``(N, 1, 2)``).
    main_contour : Optional[np.ndarray]
        The single longest contour (best candidate for break surface), or
        *None* if no contours were found.
    """
    # Combine edge map with the binary mask to confine search region.
    search_image = edges.copy()
    if binary_mask is not None:
        # Dilate mask slightly so the fragment boundary itself is included.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        search_image = cv2.bitwise_and(search_image, search_image, mask=dilated_mask)

    raw_contours, _hierarchy = cv2.findContours(search_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours = [c for c in raw_contours if cv2.arcLength(c, closed=False) >= min_length]
    contours = sorted(contours, key=lambda c: cv2.arcLength(c, closed=False), reverse=True)

    main_contour: Optional[np.ndarray] = contours[0] if contours else None
    LOG.debug("Extracted %d contours (min_length=%d)", len(contours), min_length)
    return contours, main_contour


def extract_break_contour(
    contours: List[np.ndarray],
    binary_mask: np.ndarray,
) -> Optional[np.ndarray]:
    """Heuristically select the break (fracture) contour from all contours.

    The break contour is distinguished from the smooth exterior boundary by
    being the longest contour that does *not* closely follow the convex hull.
    If no interior contour exists, the longest contour is returned.

    Parameters
    ----------
    contours:
        All contours extracted from the fragment (sorted by length, descending).
    binary_mask:
        Binary fragment mask used to obtain the fragment's convex-hull boundary.

    Returns
    -------
    Optional[np.ndarray]
        Break contour array, shape ``(N, 1, 2)``, or *None*.
    """
    if not contours:
        return None

    # Compute fragment boundary contours from the mask.
    hull_contours, _hierarchy = cv2.findContours(
        binary_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not hull_contours:
        return contours[0]

    # Find the main hull boundary arc-length.
    hull_len = max(cv2.arcLength(hc, closed=True) for hc in hull_contours)

    for c in contours:
        arc = cv2.arcLength(c, closed=False)
        # A break contour is shorter than the full outer hull.
        if arc < hull_len * 0.9:
            return c

    # Fall back to the longest available contour.
    return contours[0]

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
