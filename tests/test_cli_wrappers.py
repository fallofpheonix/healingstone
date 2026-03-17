from __future__ import annotations

import subprocess
import sys


def test_module_help_works() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "healingstone.pipeline.run_pipeline", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "--config" in proc.stdout


