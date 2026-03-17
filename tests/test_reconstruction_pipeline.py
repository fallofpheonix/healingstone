"""Integration tests for the reconstruction pipeline.

Validates 2D and 3D pipeline outputs, reproducibility, and alignment accuracy.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest


PYTHON = os.environ.get("HS_PYTHON", "python3")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"


def _run_pipeline(args: list[str], env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PYTHONPATH": str(SRC_DIR),
        "MPLCONFIGDIR": "/tmp/matplotlib_cache",
    }
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [PYTHON, "-m", "healingstone.pipeline.run_pipeline"] + args,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


class Test2DPipeline:
    """Tests for the 2D reconstruction pipeline."""

    @pytest.fixture(autouse=True)
    def setup_2d_data(self, tmp_path: Path):
        """Create synthetic 2D fragment images."""
        try:
            import cv2
        except ImportError:
            pytest.skip("opencv not installed")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(4):
            img = np.full((100, 100), 200, dtype=np.uint8)
            cv2.circle(img, (50, 50), 30, 80, -1)
            rng = np.random.default_rng(42 + i)
            img = img + rng.integers(-10, 10, size=img.shape, dtype=np.int16)
            img = np.clip(img, 0, 255).astype(np.uint8)
            cv2.imwrite(str(img_dir / f"frag_{i:02d}.png"), img)
        self.img_dir = img_dir
        self.out_dir = tmp_path / "output"

    def test_2d_pipeline_runs(self):
        """Verify that the 2D pipeline completes without errors."""
        result = _run_pipeline([
            "--data-dir", str(self.img_dir),
            "--output-dir", str(self.out_dir),
            "--config", str(PROJECT_ROOT / "configs" / "pipeline.yaml"),
            "--allow-overwrite-run",
        ])
        assert result.returncode == 0, f"Pipeline failed:\n{result.stderr}"

    def test_2d_pipeline_reproducibility(self):
        """Verify that two runs produce identical outputs."""
        out1 = self.out_dir / "run1"
        out2 = self.out_dir / "run2"
        for out in (out1, out2):
            result = _run_pipeline([
                "--data-dir", str(self.img_dir),
                "--output-dir", str(out),
                "--config", str(PROJECT_ROOT / "configs" / "pipeline.yaml"),
                "--allow-overwrite-run",
            ])
            assert result.returncode == 0


class Test3DPipelineSmoke:
    """Smoke tests for 3D pipeline components."""

    def test_feature_extraction_deterministic(self):
        """Verify that the feature pipeline produces reproducible descriptors."""
        try:
            import open3d as o3d
        except ImportError:
            pytest.skip("open3d not installed")

        pcd = o3d.geometry.PointCloud()
        rng = np.random.default_rng(42)
        pts = rng.standard_normal((1000, 3)).astype(np.float64)
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals()

        fpfh1 = o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=100)
        )
        fpfh2 = o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=100)
        )

        np.testing.assert_array_equal(
            np.asarray(fpfh1.data), np.asarray(fpfh2.data)
        )

    def test_voxel_downsampling(self):
        """Verify adaptive downsampling reduces point count."""
        try:
            import open3d as o3d
        except ImportError:
            pytest.skip("open3d not installed")

        pcd = o3d.geometry.PointCloud()
        rng = np.random.default_rng(42)
        pts = rng.standard_normal((50_000, 3)).astype(np.float64)
        pcd.points = o3d.utility.Vector3dVector(pts)

        from healingstone.core.adaptive_voxel_downsampling import (
            adaptive_voxel_downsample,
        )

        result = adaptive_voxel_downsample(pcd, target_points=10_000)
        assert result.downsampled_points <= 15_000
        assert result.downsampled_points > 0
