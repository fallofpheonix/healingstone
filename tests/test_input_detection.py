"""Tests for 2D/3D input type detection and pipeline dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from healingstone.run_pipeline import _detect_input_type


def _write_ply(path: Path) -> None:
    """Create a minimal PLY file."""
    path.write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n", encoding="utf-8")


def _write_png(path: Path) -> None:
    """Create a minimal PNG file."""
    png_data = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00:~\x9bU"
        b"\x00\x00\x00\nIDATx\x9cc\x00\x00\x00\x02\x00\x01\xe2!\xbc3"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    path.write_bytes(png_data)


def test_detect_input_type_3d(tmp_path: Path) -> None:
    """Test _detect_input_type returns '3d' for PLY files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_ply(data_dir / "frag.ply")
    assert _detect_input_type(data_dir) == "3d"


def test_detect_input_type_2d(tmp_path: Path) -> None:
    """Test _detect_input_type returns '2d' for PNG files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_png(data_dir / "frag.png")
    assert _detect_input_type(data_dir) == "2d"


def test_detect_input_type_3d_priority(tmp_path: Path) -> None:
    """Test _detect_input_type prefers 3D when both types are present."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_ply(data_dir / "frag.ply")
    _write_png(data_dir / "frag.png")
    # 3D takes priority.
    assert _detect_input_type(data_dir) == "3d"


def test_detect_input_type_empty_raises(tmp_path: Path) -> None:
    """Test _detect_input_type raises FileNotFoundError for empty directory."""
    data_dir = tmp_path / "empty"
    data_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No fragment files found"):
        _detect_input_type(data_dir)


def test_detect_input_type_jpeg(tmp_path: Path) -> None:
    """Test _detect_input_type detects JPEG files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "frag.jpeg").write_bytes(b"fake jpeg data")
    assert _detect_input_type(data_dir) == "2d"
