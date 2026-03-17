#!/usr/bin/env python3
"""Verify that pyproject dependencies are represented as pinned entries in requirements.lock."""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
LOCK = ROOT / "requirements.lock"


def normalize_name(spec: str) -> str:
    raw = spec.split(";", 1)[0].strip()
    raw = raw.split("[", 1)[0].strip()
    parts = re.split(r"[<>=!~ ]+", raw, maxsplit=1)
    return parts[0].lower().replace("_", "-")


def parse_lock_names(lines: list[str]) -> set[str]:
    out: set[str] = set()
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("-"):
            continue
        name = normalize_name(s)
        if name:
            out.add(name)
    return out


def main() -> int:
    if not PYPROJECT.exists() or not LOCK.exists():
        print("missing pyproject.toml or requirements.lock", file=sys.stderr)
        return 1

    cfg = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = cfg.get("project", {})
    deps = list(project.get("dependencies", []))
    extras = project.get("optional-dependencies", {})
    deps.extend(extras.get("dev", []))

    required = {normalize_name(dep) for dep in deps}
    lock_names = parse_lock_names(LOCK.read_text(encoding="utf-8").splitlines())

    missing = sorted(name for name in required if name and name not in lock_names)
    if missing:
        print("requirements.lock is missing pinned entries for:", file=sys.stderr)
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        return 1

    print("lock consistency check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
