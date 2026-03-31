"""Integration test for pipeline determinism and reproducibility."""

import json
import os
import subprocess
import sys
from pathlib import Path
import pytest

def run_pipeline(config_path: str, data_dir: str, output_dir: str) -> Path:
    """Run the canonical CLI and return the generated run directory."""
    cmd = [
        sys.executable, "-m", "healingstone.api.cli",
        "--config", config_path,
        "--data-dir", data_dir,
        "--output-dir", output_dir,
        "--min-required-accuracy", "0.0",
        "--allow-overwrite-run",
    ]
    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "MPLCONFIGDIR": "/tmp/matplotlib_cache",
    }
    subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)

    runs_root = Path(output_dir) / "runs"
    run_dirs = sorted(runs_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0]

def test_pipeline_reproducibility(tmp_path):
    """Assert that two identical runs produce identical normalized metrics."""
    project_root = Path(__file__).parents[2]
    config_path = str(project_root / "configs/pipeline.yaml")
    data_dir = str(project_root / "data/sample/3d")

    run_dir_1 = run_pipeline(config_path, data_dir, str(tmp_path / "experiments_1"))
    with (run_dir_1 / "metrics.json").open(encoding="utf-8") as f:
        metrics_1 = json.load(f)

    run_dir_2 = run_pipeline(config_path, data_dir, str(tmp_path / "experiments_2"))
    with (run_dir_2 / "metrics.json").open(encoding="utf-8") as f:
        metrics_2 = json.load(f)

    assert run_dir_1.name == run_dir_2.name, "Run IDs must be deterministic for identical inputs"
    assert metrics_1 == metrics_2, "Metrics must be identical for identical runs"

if __name__ == "__main__":
    # For manual execution
    pytest.main([__file__])
