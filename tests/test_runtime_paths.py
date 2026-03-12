from __future__ import annotations

from pathlib import Path

import pytest

from healingstone.runtime_paths import (
    initialize_run_layout,
    resolve_artifact_root,
    resolve_data_dir,
)


def _write_fragment(path: Path) -> None:
    path.write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n", encoding="utf-8")


def test_data_dir_cli_strict(tmp_path: Path) -> None:
    frag_dir = tmp_path / "input"
    frag_dir.mkdir(parents=True)
    _write_fragment(frag_dir / "a.ply")

    resolved, used_legacy = resolve_data_dir(
        configured_data_dir=str(frag_dir),
        data_dir_source="cli",
        dataset_alias="3d",
        aliases={"3d": str(tmp_path / "alias")},
    )
    assert resolved == frag_dir.resolve()
    assert used_legacy is False


def test_data_dir_cli_missing_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_data_dir(
            configured_data_dir=str(tmp_path / "missing"),
            data_dir_source="cli",
            dataset_alias="3d",
            aliases={},
        )


def test_artifact_root_cli(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    resolved, used_legacy = resolve_artifact_root(str(root), output_dir_source="cli")
    assert resolved == root.resolve()
    assert resolved.is_absolute()
    assert used_legacy is False


def test_run_layout_collision(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    _write_fragment(data_dir / "a.ply")
    root = tmp_path / "artifacts"

    initialize_run_layout(
        data_dir=data_dir,
        labels_csv=None,
        artifact_root=root,
        allow_overwrite_run=False,
        run_id="fixed",
    )

    with pytest.raises(FileExistsError):
        initialize_run_layout(
            data_dir=data_dir,
            labels_csv=None,
            artifact_root=root,
            allow_overwrite_run=False,
            run_id="fixed",
        )
