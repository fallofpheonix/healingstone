"""2D fragment image loading and preprocessing."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

LOG = logging.getLogger(__name__)

# Maximum edge length for resizing to avoid excessive memory use.
_MAX_EDGE = 1024


@dataclass
class Fragment2D:
    """Container for one 2D fragment image and metadata."""

    idx: int
    name: str
    path: Path
    image: np.ndarray  # uint8 grayscale, shape (H, W)


def set_deterministic_seed(seed: int = 42) -> None:
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def discover_image_files(data_dir: Path) -> List[Path]:
    """Discover supported image files recursively."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    patterns = ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG")
    files: List[Path] = []
    for pattern in patterns:
        files.extend(data_dir.rglob(pattern))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No image fragments (.PNG/.JPG) found in: {data_dir}")
    return files


def _load_grayscale(path: Path) -> np.ndarray:
    """Load an image as uint8 grayscale array using Pillow."""
    from PIL import Image  # type: ignore[import]

    with Image.open(path) as img:
        gray = img.convert("L")
        arr = np.asarray(gray, dtype=np.uint8)
    return arr


def _resize_if_large(image: np.ndarray) -> np.ndarray:
    """Downsample if the largest dimension exceeds _MAX_EDGE."""
    h, w = image.shape[:2]
    if max(h, w) <= _MAX_EDGE:
        return image

    scale = _MAX_EDGE / float(max(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    try:
        import cv2  # type: ignore[import]

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except ImportError:
        from skimage.transform import resize  # type: ignore[import]

        resized = resize(image, (new_h, new_w), anti_aliasing=True, preserve_range=True)
        return resized.astype(np.uint8)


def preprocess_fragment(idx: int, path: Path) -> Fragment2D:
    """Load and preprocess one 2D fragment image."""
    image = _load_grayscale(path)
    if image.size == 0:
        raise ValueError(f"Empty image: {path}")

    image = _resize_if_large(image)
    return Fragment2D(idx=idx, name=path.stem, path=path, image=image)


def load_and_preprocess_fragments(data_dir: Path) -> List[Fragment2D]:
    """Load all 2D fragment images from directory and preprocess."""
    files = discover_image_files(data_dir)
    LOG.info("Discovered %d image fragments", len(files))

    fragments: List[Fragment2D] = []
    for idx, path in enumerate(files):
        try:
            frag = preprocess_fragment(idx, path)
            fragments.append(frag)
            LOG.info("Loaded %s: %dx%d", path.name, frag.image.shape[1], frag.image.shape[0])
        except Exception as exc:
            LOG.warning("Skipping %s due to preprocessing failure: %s", path.name, exc)

    if len(fragments) < 2:
        raise RuntimeError("Need at least 2 valid image fragments after preprocessing")

    for i, frag in enumerate(fragments):
        frag.idx = i

    return fragments
