"""2D fragment image loading and preprocessing.

Provides:
- Fragment2D dataclass
- discover_image_files()
- load_fragment_image()
- preprocess_image()
- load_and_preprocess_fragments_2d()
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

LOG = logging.getLogger(__name__)

# Supported image extensions (lower-case; checked case-insensitively).
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class Fragment2D:
    """Container for one 2D fragment image and its derived data."""

    idx: int
    name: str
    path: Path
    gray: np.ndarray        # uint8 grayscale, shape (H, W)
    binary: np.ndarray      # uint8 binary mask (0/255), shape (H, W)
    edges: np.ndarray       # uint8 Canny edge map, shape (H, W)
    contours: List[np.ndarray] = field(default_factory=list)
    main_contour: Optional[np.ndarray] = None   # largest contour, shape (N, 1, 2)


def discover_image_files(data_dir: Path) -> List[Path]:
    """Return sorted list of image files found recursively under *data_dir*."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    files: List[Path] = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() in _IMAGE_EXTENSIONS:
            files.append(path)
    if not files:
        raise FileNotFoundError(f"No image files found in: {data_dir}")
    return files


def load_fragment_image(path: Path) -> np.ndarray:
    """Load an image file as a BGR numpy array (uint8)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def _remove_background(gray: np.ndarray) -> np.ndarray:
    """Return a binary mask separating fragment from background using Otsu thresholding.

    Assumes fragments are darker than the background (typical scan convention).
    If Otsu produces an inverted result, it is flipped automatically.
    """
    otsu_threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    LOG.debug("Otsu threshold: %d", int(otsu_threshold))
    # Ensure the fragment (foreground) covers less than 85 % of the image.
    if np.mean(binary > 0) > 0.85:
        binary = cv2.bitwise_not(binary)
    # Morphological closing to fill small holes inside the fragment.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def preprocess_image(
    img: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    denoise_strength: int = 7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess a raw BGR fragment image.

    Steps
    -----
    1. Convert to grayscale.
    2. Optional resize to *target_size* (W, H).
    3. Bilateral filter denoising.
    4. Background removal via Otsu thresholding → binary mask.
    5. Apply mask to the grayscale image (background → 255).

    Returns
    -------
    gray : np.ndarray
        Preprocessed grayscale image (uint8).
    binary : np.ndarray
        Binary fragment mask (uint8, 0 = background, 255 = fragment).
    masked : np.ndarray
        Grayscale image with background set to 255 (uint8).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if target_size is not None:
        gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    # Bilateral filter: preserves edges while smoothing noise.
    gray = cv2.bilateralFilter(gray, d=denoise_strength, sigmaColor=50, sigmaSpace=50)
    binary = _remove_background(gray)
    # Set background pixels to 255 (white) so contour detection is unaffected.
    masked = gray.copy()
    masked[binary == 0] = 255
    return gray, binary, masked


def load_and_preprocess_fragments_2d(
    data_dir: Path,
    target_size: Optional[Tuple[int, int]] = None,
    denoise_strength: int = 7,
) -> List[Fragment2D]:
    """Discover and preprocess all 2D fragment images in *data_dir*.

    Parameters
    ----------
    data_dir:
        Directory containing ``.png`` (or other supported) fragment images.
    target_size:
        Optional ``(width, height)`` to resize every image to a common canvas.
    denoise_strength:
        Bilateral filter diameter for denoising.

    Returns
    -------
    List[Fragment2D]
        Preprocessed fragments, at least 2.

    Raises
    ------
    RuntimeError
        If fewer than 2 valid fragments can be loaded.
    """
    from healingstone2d.edge_detection import detect_edges, extract_contours

    files = discover_image_files(data_dir)
    LOG.info("Discovered %d image fragments", len(files))

    fragments: List[Fragment2D] = []
    for idx, path in enumerate(files):
        try:
            raw = load_fragment_image(path)
            gray, binary, _masked = preprocess_image(raw, target_size=target_size, denoise_strength=denoise_strength)
            edges = detect_edges(gray, binary)
            contours, main_contour = extract_contours(edges, binary)
            frag = Fragment2D(
                idx=idx,
                name=path.stem,
                path=path,
                gray=gray,
                binary=binary,
                edges=edges,
                contours=contours,
                main_contour=main_contour,
            )
            fragments.append(frag)
            LOG.info("Loaded %s: %dx%d", path.name, gray.shape[1], gray.shape[0])
        except Exception as exc:
            LOG.warning("Skipping %s due to preprocessing failure: %s", path.name, exc)

    if len(fragments) < 2:
        raise RuntimeError("Need at least 2 valid image fragments after preprocessing")

    # Reindex contiguously after dropping invalid files.
    for i, frag in enumerate(fragments):
        frag.idx = i

    return fragments


def set_deterministic_seed_2d(seed: int = 42) -> None:
    """Set deterministic seeds for the 2D pipeline (numpy and random)."""
    random.seed(seed)
    np.random.seed(seed)
