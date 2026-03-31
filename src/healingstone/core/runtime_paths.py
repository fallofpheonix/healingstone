"""Compatibility re-exports for runtime path helpers."""

from __future__ import annotations

from ..io import runtime_paths as _runtime_paths
from ..io.runtime_paths import (
    CANONICAL_ARTIFACT_ROOT,
    CANONICAL_DATA_DIR,
    ResolvedRunPaths,
    _contains_fragments,
    _contains_images,
    make_deterministic_run_id,
    project_root,
    write_resolved_paths_metadata,
)

__all__ = [
    "CANONICAL_ARTIFACT_ROOT",
    "CANONICAL_DATA_DIR",
    "PROJECT_ROOT",
    "ResolvedRunPaths",
    "_contains_fragments",
    "_contains_images",
    "make_deterministic_run_id",
    "project_root",
    "initialize_run_layout",
    "resolve_artifact_root",
    "resolve_data_dir",
    "write_resolved_paths_metadata",
]

PROJECT_ROOT = _runtime_paths.PROJECT_ROOT


def _sync_project_root() -> None:
    _runtime_paths.PROJECT_ROOT = PROJECT_ROOT


def resolve_data_dir(
    configured_data_dir: str | None,
    data_dir_source: str,
    dataset_alias: str,
    aliases: dict[str, str],
):
    _sync_project_root()
    return _runtime_paths.resolve_data_dir(
        configured_data_dir=configured_data_dir,
        data_dir_source=data_dir_source,
        dataset_alias=dataset_alias,
        aliases=aliases,
    )


def resolve_artifact_root(configured_output_dir: str | None, output_dir_source: str):
    _sync_project_root()
    return _runtime_paths.resolve_artifact_root(
        configured_output_dir=configured_output_dir,
        output_dir_source=output_dir_source,
    )


def initialize_run_layout(
    data_dir,
    labels_csv,
    artifact_root,
    allow_overwrite_run,
    run_id=None,
):
    _sync_project_root()
    return _runtime_paths.initialize_run_layout(
        data_dir=data_dir,
        labels_csv=labels_csv,
        artifact_root=artifact_root,
        allow_overwrite_run=allow_overwrite_run,
        run_id=run_id,
    )
