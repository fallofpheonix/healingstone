"""Tests for 2D image discovery in runtime_paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from healingstone.core.runtime_paths import _contains_images, resolve_data_dir


def _write_image(path: Path) -> None:
    """Create a minimal PNG file."""
    # Minimal 1x1 PNG file (grayscale)
    png_data = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00:~\x9bU"
        b"\x00\x00\x00\nIDATx\x9cc\x00\x00\x00\x02\x00\x01\xe2!\xbc3"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    path.write_bytes(png_data)


def test_contains_images_true(tmp_path: Path) -> None:
    """Test _contains_images detects PNG files."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    _write_image(img_dir / "frag.png")
    assert _contains_images(img_dir) is True


def test_contains_images_false(tmp_path: Path) -> None:
    """Test _contains_images returns False for empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert _contains_images(empty_dir) is False


def test_contains_images_jpg(tmp_path: Path) -> None:
    """Test _contains_images detects JPG files."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "frag.jpg").write_bytes(b"fake jpg data")
    assert _contains_images(img_dir) is True


def test_contains_images_jpeg(tmp_path: Path) -> None:
    """Test _contains_images detects JPEG files."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "frag.jpeg").write_bytes(b"fake jpeg data")
    assert _contains_images(img_dir) is True


def test_resolve_data_dir_images_cli(tmp_path: Path) -> None:
    """Test resolve_data_dir with explicit image directory."""
    img_dir = tmp_path / "fragments"
    img_dir.mkdir()
    _write_image(img_dir / "a.png")

    resolved, used_legacy = resolve_data_dir(
        configured_data_dir=str(img_dir),
        data_dir_source="cli",
        dataset_alias="2d",
        aliases={},
    )
    assert resolved == img_dir.resolve()
    assert used_legacy is False


def test_resolve_data_dir_mixed_fails(tmp_path: Path) -> None:
    """Test resolve_data_dir fails when directory is empty (no meshes or images)."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No dataset fragments found"):
        resolve_data_dir(
            configured_data_dir=str(empty_dir),
            data_dir_source="yaml",
            dataset_alias="2d",
            aliases={},
        )


def test_resolve_data_dir_images_default(tmp_path: Path) -> None:
    """Test resolve_data_dir with image-only directory as configured path."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    _write_image(img_dir / "frag.png")

    resolved, used_legacy = resolve_data_dir(
        configured_data_dir=str(img_dir),
        data_dir_source="yaml",
        dataset_alias="2d",
        aliases={},
    )
    assert resolved == img_dir.resolve()
    assert used_legacy is False
