from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_module_help_works() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "healingstone.run_pipeline", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "--config" in proc.stdout


