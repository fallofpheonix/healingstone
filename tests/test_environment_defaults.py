from __future__ import annotations

from argparse import Namespace
from _pytest.monkeypatch import MonkeyPatch

from healingstone.config.environment import apply_env_defaults


def test_apply_env_defaults_fills_missing_values(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("HEALINGSTONE_DATA_DIR", "data/raw/v1")
    monkeypatch.setenv("HEALINGSTONE_OUTPUT_DIR", "artifacts")

    args = Namespace(data_dir=None, output_dir=None, labels_csv=None)
    out = apply_env_defaults(args)

    assert out.data_dir == "data/raw/v1"
    assert out.output_dir == "artifacts"


def test_apply_env_defaults_preserves_cli_values(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("HEALINGSTONE_DATA_DIR", "from-env")

    args = Namespace(data_dir="from-cli", output_dir=None, labels_csv=None)
    out = apply_env_defaults(args)

    assert out.data_dir == "from-cli"
