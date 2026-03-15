"""Tests for pipeline mode auto-detection."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from healingstone.run_pipeline import detect_pipeline_mode


def test_detect_3d_pipeline_with_ply_files() -> None:
    """Test that PLY files trigger the 3D pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        # Create mock PLY files
        (data_dir / "fragment1.ply").touch()
        (data_dir / "fragment2.ply").touch()
        
        mode = detect_pipeline_mode(data_dir)
        assert mode == "3d"


def test_detect_3d_pipeline_with_obj_files() -> None:
    """Test that OBJ files trigger the 3D pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        # Create mock OBJ files
        (data_dir / "fragment1.obj").touch()
        (data_dir / "fragment2.OBJ").touch()  # Test case-insensitivity
        
        mode = detect_pipeline_mode(data_dir)
        assert mode == "3d"


def test_detect_2d_pipeline_with_png_files() -> None:
    """Test that PNG files trigger the 2D pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        # Create mock PNG files
        (data_dir / "fragment1.png").touch()
        (data_dir / "fragment2.PNG").touch()  # Test case-insensitivity
        
        mode = detect_pipeline_mode(data_dir)
        assert mode == "2d"


def test_detect_2d_pipeline_with_jpg_files() -> None:
    """Test that JPG/JPEG files trigger the 2D pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        # Create mock JPG files
        (data_dir / "fragment1.jpg").touch()
        (data_dir / "fragment2.jpeg").touch()
        
        mode = detect_pipeline_mode(data_dir)
        assert mode == "2d"


def test_detect_2d_pipeline_with_mixed_image_formats() -> None:
    """Test that various image formats all trigger the 2D pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        # Create mock files of different image formats
        (data_dir / "frag1.png").touch()
        (data_dir / "frag2.jpg").touch()
        (data_dir / "frag3.tif").touch()
        (data_dir / "frag4.bmp").touch()
        
        mode = detect_pipeline_mode(data_dir)
        assert mode == "2d"


def test_mesh_files_take_precedence_over_images() -> None:
    """Test that mesh files take precedence when both mesh and image files are present."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        # Create both mesh and image files
        (data_dir / "fragment1.ply").touch()
        (data_dir / "fragment2.png").touch()
        
        mode = detect_pipeline_mode(data_dir)
        assert mode == "3d", "Mesh files should take precedence over image files"


def test_recursive_directory_search() -> None:
    """Test that detection searches subdirectories recursively."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        # Create nested directory structure
        subdir = data_dir / "fragments" / "set1"
        subdir.mkdir(parents=True)
        (subdir / "fragment1.ply").touch()
        
        mode = detect_pipeline_mode(data_dir)
        assert mode == "3d"


def test_no_supported_files_raises_error() -> None:
    """Test that an error is raised when no supported files are found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        # Create unsupported files
        (data_dir / "readme.txt").touch()
        (data_dir / "data.csv").touch()
        
        with pytest.raises(FileNotFoundError, match="No supported fragment files"):
            detect_pipeline_mode(data_dir)


def test_empty_directory_raises_error() -> None:
    """Test that an error is raised when the directory is empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        
        with pytest.raises(FileNotFoundError, match="No supported fragment files"):
            detect_pipeline_mode(data_dir)


def test_nonexistent_directory_raises_error() -> None:
    """Test that an error is raised when the directory does not exist."""
    nonexistent = Path("/nonexistent/path/to/data")
    
    with pytest.raises(FileNotFoundError, match="Data directory does not exist"):
        detect_pipeline_mode(nonexistent)


def test_case_insensitive_extension_matching() -> None:
    """Test that file extension matching is case-insensitive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        # Create files with various case combinations
        (data_dir / "frag1.PLY").touch()
        (data_dir / "frag2.Obj").touch()
        (data_dir / "frag3.PnG").touch()
        
        # Mesh files should be detected first
        mode = detect_pipeline_mode(data_dir)
        assert mode == "3d"
        
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        # Only images with mixed case
        (data_dir / "frag1.PnG").touch()
        (data_dir / "frag2.JpG").touch()
        
        mode = detect_pipeline_mode(data_dir)
        assert mode == "2d"
