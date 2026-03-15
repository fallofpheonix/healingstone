"""Canny edge detection and contour extraction for 2D fragment images.

Provides:
- detect_edges()       – Canny edge map constrained to the fragment mask
- extract_contours()   – Contour extraction and selection of the main contour
- extract_break_contour() – Heuristic identification of break (fracture) edges
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
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
