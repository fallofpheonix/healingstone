"""Shape descriptor extraction for 2D fragment matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .edge_detection import EdgeBundle

# Number of angular bins for the radial profile histogram.
_N_ANGLE_BINS = 36
# Number of radial bins for the shape context.
_N_RADIAL_BINS = 5


@dataclass
class ShapeDescriptor:
    """Shape descriptors for one 2D fragment."""

    hu_moments: np.ndarray  # (7,) float64
    contour_histogram: np.ndarray  # (N_ANGLE_BINS,) float32 shape context
    radial_histogram: np.ndarray  # (N_RADIAL_BINS,) float32 radial distribution
    perimeter: float
    area_ratio: float  # fraction of image bounding box covered
    descriptor: np.ndarray  # concatenated normalised descriptor vector


def _hu_moments_cv2(binary: np.ndarray) -> np.ndarray:
    import cv2  # type: ignore[import]

    moments = cv2.moments(binary)
    hu = cv2.HuMoments(moments).flatten()
    # Log-scale to reduce dynamic range.
    with np.errstate(divide="ignore"):
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu_log.astype(np.float64)


def _hu_moments_skimage(binary: np.ndarray) -> np.ndarray:
    from skimage.measure import moments, moments_hu  # type: ignore[import]

    m = moments(binary.astype(float))
    hu = moments_hu(m)
    with np.errstate(divide="ignore"):
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu_log.astype(np.float64)


def compute_hu_moments(binary: np.ndarray) -> np.ndarray:
    """Compute log-scaled Hu moments from a binary mask."""
    try:
        return _hu_moments_cv2(binary)
    except ImportError:
        return _hu_moments_skimage(binary)


def _build_binary_mask(image: np.ndarray) -> np.ndarray:
    """Threshold grayscale image to binary mask."""
    try:
        import cv2  # type: ignore[import]

        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    except ImportError:
        from skimage.filters import threshold_otsu  # type: ignore[import]

        thr = threshold_otsu(image)
        return (image >= thr).astype(np.uint8) * 255


def _angular_histogram(points: np.ndarray, centroid: np.ndarray, n_bins: int) -> np.ndarray:
    """Histogram of angles from centroid to boundary points."""
    if points.shape[0] == 0:
        return np.zeros(n_bins, dtype=np.float32)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    hist, _ = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi))
    denom = float(hist.sum()) + 1e-12
    return (hist / denom).astype(np.float32)


def _radial_histogram(points: np.ndarray, centroid: np.ndarray, n_bins: int) -> np.ndarray:
    """Histogram of normalised radial distances from centroid."""
    if points.shape[0] == 0:
        return np.zeros(n_bins, dtype=np.float32)
    radii = np.linalg.norm(points - centroid, axis=1)
    max_r = float(np.max(radii)) + 1e-12
    normed = radii / max_r
    hist, _ = np.histogram(normed, bins=n_bins, range=(0.0, 1.0))
    denom = float(hist.sum()) + 1e-12
    return (hist / denom).astype(np.float32)


def _perimeter_from_contours(contours: List[np.ndarray]) -> float:
    """Sum of contour arc lengths (closed perimeter)."""
    total = 0.0
    for c in contours:
        if c.shape[0] < 2:
            continue
        # Add segment lengths for consecutive points
        diffs = np.diff(c, axis=0)
        total += float(np.sum(np.linalg.norm(diffs, axis=1)))
        # Add closing segment from last point back to first
        closing = np.linalg.norm(c[-1] - c[0])
        total += float(closing)
    return total


def _normalise(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v)) + 1e-12
    return v / norm


def compute_shape_descriptor(image: np.ndarray, edges: EdgeBundle) -> ShapeDescriptor:
    """Compute shape descriptors from a preprocessed fragment image."""
    binary = _build_binary_mask(image)

    hu = compute_hu_moments(binary)

    h, w = image.shape[:2]
    image_area = max(1.0, float(h * w))
    fragment_area = float(np.count_nonzero(binary))
    area_ratio = float(np.clip(fragment_area / image_area, 0.0, 1.0))

    bp = edges.boundary_points
    if bp.shape[0] >= 2:
        centroid = bp.mean(axis=0)
        angular_hist = _angular_histogram(bp, centroid, _N_ANGLE_BINS)
        radial_hist = _radial_histogram(bp, centroid, _N_RADIAL_BINS)
    else:
        angular_hist = np.zeros(_N_ANGLE_BINS, dtype=np.float32)
        radial_hist = np.zeros(_N_RADIAL_BINS, dtype=np.float32)

    perimeter = _perimeter_from_contours(edges.contours)

    # Build final descriptor.
    hu_norm = _normalise(hu.astype(np.float32))
    descriptor = np.concatenate(
        [
            hu_norm,
            angular_hist,
            radial_hist,
            np.array([area_ratio, np.log1p(perimeter) / (np.log1p(float(max(h, w))) + 1e-12)], dtype=np.float32),
        ]
    ).astype(np.float32)

    return ShapeDescriptor(
        hu_moments=hu,
        contour_histogram=angular_hist,
        radial_histogram=radial_hist,
        perimeter=perimeter,
        area_ratio=area_ratio,
        descriptor=descriptor,
    )
