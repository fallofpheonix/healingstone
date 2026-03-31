"""Path resolution and run-scoped artifact layout."""

from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

CANONICAL_DATA_DIR = Path("data/raw/3d")
CANONICAL_ARTIFACT_ROOT = Path("artifacts")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


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


def _normalize(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def project_root() -> Path:
    """Return the repository root used for resolving relative runtime paths."""
    return PROJECT_ROOT


def _contains_fragments(path: Path) -> bool:
    if not path.exists():
        return False
    supported_suffixes = {".ply", ".obj"}
    for candidate in path.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() in supported_suffixes:
            return True
    return False


def _contains_images(path: Path) -> bool:
    """Check whether *path* contains any supported 2D image files."""
    if not path.exists():
        return False
    supported_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    for candidate in path.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() in supported_suffixes:
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


def make_deterministic_run_id(
    *,
    data_dir: Path,
    config_hash: str,
    labels_csv: Path | None = None,
) -> str:
    payload = {
        "data_dir": str(_normalize(data_dir)),
        "labels_csv": str(_normalize(labels_csv)) if labels_csv is not None else None,
        "config_hash": config_hash,
        "git_commit": _git_short_commit(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def resolve_data_dir(
    configured_data_dir: str | None,
    data_dir_source: str,
    dataset_alias: str,
    aliases: Dict[str, str],
) -> Path:
    """Resolve dataset path with strict precedence semantics.

    Supports both 3D mesh fragments (.PLY/.OBJ) and 2D image fragments
    (.PNG/.JPG/.JPEG/.TIF/.TIFF/.BMP).
    """
    if data_dir_source in {"cli", "env"}:
        if configured_data_dir is None:
            raise FileNotFoundError("Explicit data_dir source provided but value is empty")
        candidate = _normalize(configured_data_dir)
        if not _contains_fragments(candidate) and not _contains_images(candidate):
            raise FileNotFoundError(
                f"Explicit data_dir has no supported fragments (.PLY/.OBJ/.PNG/.JPG/.JPEG/.TIF/.TIFF/.BMP): {candidate}"
            )
        return candidate

    alias_target = aliases.get(dataset_alias)
    if configured_data_dir:
        candidate = _normalize(configured_data_dir)
    elif alias_target:
        candidate = _normalize(alias_target)
    else:
        candidate = _normalize(CANONICAL_DATA_DIR)

    if _contains_fragments(candidate) or _contains_images(candidate):
        return candidate

    raise FileNotFoundError(
        f"No dataset fragments found under {candidate}. dataset_alias={dataset_alias!r}, configured_data_dir={configured_data_dir!r}."
    )


def resolve_artifact_root(configured_output_dir: str | None, output_dir_source: str) -> Path:
    """Resolve artifact root path relative to the repository root."""
    if output_dir_source in {"cli", "env"}:
        if configured_output_dir is None:
            raise FileNotFoundError("Explicit output_dir source provided but value is empty")
        root = _normalize(configured_output_dir)
        _check_writable_dir(root)
        return root

    root = _normalize(configured_output_dir or CANONICAL_ARTIFACT_ROOT)
    _check_writable_dir(root)
    return root


def initialize_run_layout(
    data_dir: Path,
    labels_csv: str | None,
    artifact_root: Path,
    allow_overwrite_run: bool,
    run_id: str | None = None,
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
