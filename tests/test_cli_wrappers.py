from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_wrappers_do_not_mutate_sys_path() -> None:
    for rel in ["run_pipeline.py", "test_pipeline.py", "healing_stones.py", "scripts/run_pipeline.py", "scripts/test_pipeline.py"]:
        text = Path(rel).read_text(encoding="utf-8")
        assert "sys.path" not in text


def test_module_help_works() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "healingstone.run_pipeline", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "--config" in proc.stdout


def test_root_wrapper_help_works() -> None:
    proc = subprocess.run(
        [sys.executable, "run_pipeline.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "--config" in proc.stdout
