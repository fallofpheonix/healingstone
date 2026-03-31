"""Integration test for pipeline determinism and reproducibility."""

import json
import subprocess
import sys
from pathlib import Path
import pytest

def run_pipeline(config_path: str, data_dir: str, output_dir: str) -> str:
    """Run the pipeline via CLI and return the run_id."""
    cmd = [
        sys.executable, "-m", "healingstone.cli", "run",
        "--config", config_path,
        "--data-dir", data_dir,
        "--output-dir", output_dir
    ]
    # Set PYTHONPATH to include the src directory
    env = {"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"}
    subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    
    # Extract run_id from logs or stdout
    # In a real system, we might parse the output. For simplicity, we'll look at the FS.
    run_dirs = sorted(Path(output_dir).iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0].name

def test_pipeline_reproducibility(tmp_path):
    """MANDATORY: Assert that two identical runs produce identical outcomes."""
    project_root = Path(__file__).parents[2]
    config_path = str(project_root / "configs/pipeline.yaml")
    data_dir = str(project_root / "data/sample")
    output_dir = str(tmp_path / "experiments")
    
    # Create mock config if it doesn't exist
    Path("configs").mkdir(exist_ok=True)
    if not Path(config_path).exists():
        with open(config_path, "w") as f:
            f.write("output_dir: experiments\n")

    # 1. First Run
    run_id_1 = run_pipeline(config_path, data_dir, output_dir)
    with open(Path(output_dir) / run_id_1 / "metrics.json") as f:
        metrics_1 = json.load(f)
        
    # 2. Second Run (Identical config and data)
    run_id_2 = run_pipeline(config_path, data_dir, output_dir)
    with open(Path(output_dir) / run_id_2 / "metrics.json") as f:
        metrics_2 = json.load(f)
        
    # Hard Determinism Assertions
    assert run_id_1 == run_id_2, "Run IDs must be identical for identical inputs (hashing contract)"
    assert metrics_1 == metrics_2, "Metrics must be identical for identical runs (determinism contract)"

if __name__ == "__main__":
    # For manual execution
    pytest.main([__file__])
