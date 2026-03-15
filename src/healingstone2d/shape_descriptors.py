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
