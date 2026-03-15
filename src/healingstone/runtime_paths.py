"""Path resolution and run-scoped artifact layout."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

CANONICAL_DATA_DIR = Path("data/raw/3d")
LEGACY_DATA_DIR = Path("DataSet/3D")
CANONICAL_ARTIFACT_ROOT = Path("artifacts")
LEGACY_ARTIFACT_ROOT = Path("results")


@dataclass(frozen=True)
class ResolvedRunPaths:
    data_dir: Path
    labels_csv: Optional[Path]
    artifact_root: Path
    run_id: str
    run_dir: Path
    results_dir: Path
    models_dir: Path
    logs_dir: Path
    cache_dir: Path
    used_legacy_data: bool
    used_legacy_output: bool


def _normalize(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _contains_fragments(path: Path) -> bool:
    if not path.exists():
        return False
    patterns = ("*.ply", "*.PLY", "*.obj", "*.OBJ")
    for pattern in patterns:
        if next(path.rglob(pattern), None) is not None:
            return True
    return False


def _contains_images(path: Path) -> bool:
    """Check whether *path* contains any supported 2D image files."""
    if not path.exists():
        return False
    patterns = ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG")
    for pattern in patterns:
        if next(path.rglob(pattern), None) is not None:
            return True
    return False


def _check_writable_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write_probe"
    with probe.open("w", encoding="utf-8") as f:
        f.write("ok")
    probe.unlink(missing_ok=True)


def _git_short_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        val = out.decode("utf-8").strip()
        return val or "nogit"
    except Exception:
        return "nogit"


def make_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{_git_short_commit()}"


def resolve_data_dir(
    configured_data_dir: str | None,
    data_dir_source: str,
    dataset_alias: str,
    aliases: Dict[str, str],
) -> tuple[Path, bool]:
    """Resolve dataset path with strict precedence semantics.

    Supports both 3D mesh fragments (.PLY/.OBJ) and 2D image fragments
    (.PNG/.JPG/.JPEG).
    """
    used_legacy = False

    if data_dir_source in {"cli", "env"}:
        if configured_data_dir is None:
            raise FileNotFoundError("Explicit data_dir source provided but value is empty")
        candidate = _normalize(configured_data_dir)
        if not _contains_fragments(candidate) and not _contains_images(candidate):
            raise FileNotFoundError(
                f"Explicit data_dir has no .PLY/.OBJ/.PNG/.JPG fragments: {candidate}"
            )
        return candidate, used_legacy

    alias_target = aliases.get(dataset_alias)
    if configured_data_dir:
        candidate = _normalize(configured_data_dir)
    elif alias_target:
        candidate = _normalize(alias_target)
    else:
        candidate = _normalize(CANONICAL_DATA_DIR)

    if _contains_fragments(candidate) or _contains_images(candidate):
        return candidate, used_legacy

    legacy_candidate = _normalize(LEGACY_DATA_DIR)
    if _contains_fragments(legacy_candidate) or _contains_images(legacy_candidate):
        used_legacy = True
        return legacy_candidate, used_legacy

    raise FileNotFoundError(
        f"No dataset fragments found. Checked candidate={candidate}, legacy={legacy_candidate}, alias={dataset_alias}."
    )


def resolve_artifact_root(configured_output_dir: str | None, output_dir_source: str) -> tuple[Path, bool]:
    """Resolve artifact root path with fallback only for non-explicit sources."""
    used_legacy = False
    if output_dir_source in {"cli", "env"}:
        if configured_output_dir is None:
            raise FileNotFoundError("Explicit output_dir source provided but value is empty")
        root = _normalize(configured_output_dir)
        _check_writable_dir(root)
        return root, used_legacy

    canonical = _normalize(configured_output_dir or CANONICAL_ARTIFACT_ROOT)
    if canonical.exists():
        _check_writable_dir(canonical)
        return canonical, used_legacy

    legacy = _normalize(LEGACY_ARTIFACT_ROOT)
    if legacy.exists():
        _check_writable_dir(legacy)
        used_legacy = True
        return legacy, used_legacy

    _check_writable_dir(canonical)
    return canonical, used_legacy


def initialize_run_layout(
    data_dir: Path,
    labels_csv: str | None,
    artifact_root: Path,
    allow_overwrite_run: bool,
    run_id: str | None = None,
    used_legacy_data: bool = False,
    used_legacy_output: bool = False,
) -> ResolvedRunPaths:
    rid = run_id or make_run_id()
    runs_root = artifact_root / "runs"
    run_dir = runs_root / rid

    if run_dir.exists() and not allow_overwrite_run:
        raise FileExistsError(f"Run directory already exists: {run_dir}. Use --allow-overwrite-run to reuse.")

    run_dir.mkdir(parents=True, exist_ok=allow_overwrite_run)
    results_dir = run_dir / "results"
    models_dir = run_dir / "models"
    logs_dir = run_dir / "logs"
    cache_dir = run_dir / "cache"
    for path in (results_dir, models_dir, logs_dir, cache_dir):
        path.mkdir(parents=True, exist_ok=True)

    labels_path = _normalize(labels_csv) if labels_csv else None
    if labels_path is not None and not labels_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_path}")

    resolved = ResolvedRunPaths(
        data_dir=_normalize(data_dir),
        labels_csv=labels_path,
        artifact_root=_normalize(artifact_root),
        run_id=rid,
        run_dir=_normalize(run_dir),
        results_dir=_normalize(results_dir),
        models_dir=_normalize(models_dir),
        logs_dir=_normalize(logs_dir),
        cache_dir=_normalize(cache_dir),
        used_legacy_data=used_legacy_data,
        used_legacy_output=used_legacy_output,
    )
    _update_latest_pointer(resolved)
    return resolved


def _update_latest_pointer(paths: ResolvedRunPaths) -> None:
    latest = paths.artifact_root / "latest"
    if latest.is_symlink() or latest.is_file():
        latest.unlink(missing_ok=True)
    elif latest.exists():
        shutil.rmtree(latest)
    try:
        latest.symlink_to(paths.run_dir, target_is_directory=True)
    except OSError:
        latest.write_text(str(paths.run_dir), encoding="utf-8")


def write_resolved_paths_metadata(paths: ResolvedRunPaths, out_file: Path) -> None:
    payload = asdict(paths)
    payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in payload.items()}
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
