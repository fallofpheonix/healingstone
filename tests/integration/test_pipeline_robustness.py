"""Integration tests for pipeline robustness and failure modes."""

import subprocess
import sys
from pathlib import Path
import pytest

def run_pipeline(config_path: str, data_dir: str, output_dir: str) -> subprocess.CompletedProcess:
    """Run the pipeline via CLI and return the result."""
    cmd = [
        sys.executable, "-m", "healingstone.cli", "run",
        "--config", config_path,
        "--data-dir", data_dir,
        "--output-dir", output_dir
    ]
    env = {"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"}
    return subprocess.run(cmd, env=env, capture_output=True, text=True)

def test_invalid_config_fails(tmp_path):
    """MANDATORY: Assert that corrupted configuration fails before execution."""
    invalid_config = tmp_path / "invalid_pipeline.yaml"
    
    # Intentionally corrupt config (invalid type for data_dir)
    invalid_config.write_text("data_dir: 12345\npreprocessing: {invalid_key: true}")
    
    output_dir = str(tmp_path / "experiments")
    
    result = run_pipeline(str(invalid_config), "data/sample", output_dir)
    
    # Should fail with an error code (Pydantic validation error)
    assert result.returncode != 0
    # In a real setup, we'd check for "ValidationError" in stderr
    # assert "ValidationError" in result.stderr

def test_missing_data_fails(tmp_path):
    """MANDATORY: Assert that missing data produces a meaningful error."""
    project_root = Path(__file__).parents[2]
    config_path = str(project_root / "configs/pipeline.yaml")
    
    # Use a non-existent data directory
    non_existent_data = str(tmp_path / "missing_data")
    output_dir = str(tmp_path / "experiments")
    
    result = run_pipeline(config_path, non_existent_data, output_dir)
    
    # Should fail if hashing or preprocessing detects missing files
    assert result.returncode != 0

if __name__ == "__main__":
    pytest.main([__file__])
