"""Shape descriptor extraction for 2D fragment contours.

Provides:
- compute_hu_moments()        – Log-scaled Hu invariant moments (7-D)
- compute_fourier_descriptors() – Fourier shape descriptors (2*n_coeffs-D)
- compute_shape_descriptor()  – Combined normalised descriptor
- extract_all_descriptors()   – Batch extraction for all Fragment2D objects
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

LOG = logging.getLogger(__name__)

# Dimension of Hu moment vector.
_HU_DIM = 7
# Default number of Fourier coefficients to retain (excluding DC).
_N_FOURIER = 32
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
    """Combined shape descriptor for one fragment."""

    idx: int
    name: str
    hu_moments: np.ndarray       # shape (_HU_DIM,)
    fourier: np.ndarray          # shape (2 * n_fourier,)  real + imag
    descriptor: np.ndarray       # concatenated + normalised, shape (D,)
    contour_length: float        # arc-length of the source contour (pixels)
    area: float                  # contour-enclosed area (pixels²)


def compute_hu_moments(contour: np.ndarray) -> np.ndarray:
    """Compute log-scaled Hu invariant moments from a contour.

    Parameters
    ----------
    contour:
        Array of shape ``(N, 1, 2)`` (OpenCV contour format).

    Returns
    -------
    np.ndarray
        Shape ``(7,)``, float32.  Log-transformed Hu moments as
        ``sign(h) * log10(1 + |h|)`` for numerical stability.
    """
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).ravel()  # shape (7,)
    # Log-scale for improved discriminability.
    hu_log = np.sign(hu) * np.log10(1.0 + np.abs(hu))
    return hu_log.astype(np.float32)


def compute_fourier_descriptors(
    contour: np.ndarray,
    n_coeffs: int = _N_FOURIER,
) -> np.ndarray:
    """Compute normalised Fourier shape descriptors.

    The contour is represented as a complex signal z = x + jy.  The DFT is
    computed and the low-frequency coefficients (excluding DC) are kept.
    Normalisation is performed by dividing by the magnitude of the first
    non-DC coefficient (scale invariance).

    Parameters
    ----------
    contour:
        Array of shape ``(N, 1, 2)`` (OpenCV contour format).
    n_coeffs:
        Number of Fourier coefficients to retain (excluding DC term).

    Returns
    -------
    np.ndarray
        Shape ``(2 * n_coeffs,)``, float32 (interleaved real and imag parts).
    """
    pts = contour.squeeze()  # (N, 2)
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]

    # Represent the contour as a complex signal.
    z = pts[:, 0].astype(np.float64) + 1j * pts[:, 1].astype(np.float64)
    dft = np.fft.fft(z)

    # If the contour is too short to provide a first non-DC coefficient,
    # return a zero descriptor to avoid indexing errors and keep API stable.
    if dft.size < 2:
        return np.zeros(2 * n_coeffs, dtype=np.float32)

    # Take low-frequency coefficients (skip DC at index 0).
    low = dft[1 : n_coeffs + 1]
    if len(low) < n_coeffs:
        # Pad with zeros if contour is shorter than requested coefficients.
        low = np.pad(low, (0, n_coeffs - len(low)), constant_values=0.0)

    # Scale-invariant normalisation: divide by magnitude of first coefficient.
    norm = np.abs(dft[1]) if np.abs(dft[1]) > 1e-10 else 1.0
    low = low / norm

    # Interleave real and imaginary parts → (2 * n_coeffs,).
    result = np.empty(2 * n_coeffs, dtype=np.float32)
    result[0::2] = low.real.astype(np.float32)
    result[1::2] = low.imag.astype(np.float32)
    return result


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    """L2-normalise a 1-D array; returns zero vector if norm is tiny."""
    norm = float(np.linalg.norm(x))
    if norm < 1e-12:
        return np.zeros_like(x)
    return x / norm


def compute_shape_descriptor(
    contour: np.ndarray,
    n_fourier: int = _N_FOURIER,
) -> ShapeDescriptor:
    """Compute combined Hu + Fourier shape descriptor for a single contour.

    The two descriptor parts are individually L2-normalised and then
    concatenated to form the final descriptor vector.

    Parameters
    ----------
    contour:
        OpenCV contour array, shape ``(N, 1, 2)``.
    n_fourier:
        Number of Fourier coefficients to retain.

    Returns
    -------
    ShapeDescriptor
        Filled descriptor object (without ``idx`` / ``name`` – set externally).
    """
    hu = compute_hu_moments(contour)
    fourier = compute_fourier_descriptors(contour, n_coeffs=n_fourier)

    descriptor = np.concatenate([_l2_normalise(hu), _l2_normalise(fourier)]).astype(np.float32)

    arc = float(cv2.arcLength(contour, closed=False))
    area = float(cv2.contourArea(contour))

    return ShapeDescriptor(
        idx=-1,
        name="",
        hu_moments=hu,
        fourier=fourier,
        descriptor=descriptor,
        contour_length=arc,
        area=area,
    )


def extract_all_descriptors(
    fragments: list,
    n_fourier: int = _N_FOURIER,
) -> List[ShapeDescriptor]:
    """Extract shape descriptors for every :class:`~healingstone2d.preprocess_2d.Fragment2D`.

    Fragments without a valid main contour are skipped with a warning.

    Parameters
    ----------
    fragments:
        List of ``Fragment2D`` objects.
    n_fourier:
        Number of Fourier coefficients to retain.

    Returns
    -------
    List[ShapeDescriptor]
        One descriptor per valid fragment (in the same order).
    """
    descriptors: List[ShapeDescriptor] = []
    for frag in fragments:
        if frag.main_contour is None or len(frag.main_contour) < 5:
            LOG.warning("Fragment %s has no usable contour – skipping descriptor.", frag.name)
            continue
        try:
            desc = compute_shape_descriptor(frag.main_contour, n_fourier=n_fourier)
            desc.idx = frag.idx
            desc.name = frag.name
            descriptors.append(desc)
            LOG.debug(
                "Descriptor for %s: contour_length=%.1f, area=%.1f",
                frag.name,
                desc.contour_length,
                desc.area,
            )
        except Exception as exc:
            LOG.warning("Descriptor extraction failed for %s: %s", frag.name, exc)

    LOG.info("Extracted descriptors for %d / %d fragments", len(descriptors), len(fragments))
    return descriptors
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
