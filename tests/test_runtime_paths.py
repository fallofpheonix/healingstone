from __future__ import annotations

from pathlib import Path

import pytest

import healingstone.core.runtime_paths as runtime_paths
from healingstone.core.runtime_paths import (
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

    resolved = resolve_data_dir(
        configured_data_dir=str(frag_dir),
        data_dir_source="cli",
        dataset_alias="3d",
        aliases={"3d": str(tmp_path / "alias")},
    )
    assert resolved == frag_dir.resolve()


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
    resolved = resolve_artifact_root(str(root), output_dir_source="cli")
    assert resolved == root.resolve()
    assert resolved.is_absolute()


def test_default_paths_resolve_from_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = tmp_path / "repo"
    data_dir = project_root / "data" / "raw" / "3d"
    data_dir.mkdir(parents=True)
    _write_fragment(data_dir / "a.ply")

    monkeypatch.setattr(runtime_paths, "PROJECT_ROOT", project_root)

    resolved_data = resolve_data_dir(
        configured_data_dir=None,
        data_dir_source="yaml",
        dataset_alias="3d",
        aliases={},
    )
    resolved_artifacts = resolve_artifact_root(None, output_dir_source="yaml")

    assert resolved_data == data_dir.resolve()
    assert resolved_artifacts == (project_root / "artifacts").resolve()


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
