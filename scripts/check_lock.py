#!/usr/bin/env python3
"""Verify that pyproject dependencies are represented as pinned entries in requirements.lock."""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None

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


def _extract_array_strings(section_text: str, key: str) -> list[str]:
    pattern = re.compile(
        rf"^\s*{re.escape(key)}\s*=\s*\[(?P<body>.*?)^\s*\]",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(section_text)
    if not match:
        return []
    return re.findall(r'"((?:[^"\\]|\\.)*)"', match.group("body"))


def _extract_section(text: str, section_name: str) -> str:
    section_pattern = re.compile(r"^\[(?P<name>[^\]]+)\]\s*$", re.MULTILINE)
    matches = list(section_pattern.finditer(text))
    for index, match in enumerate(matches):
        if match.group("name") != section_name:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        return text[start:end]
    return ""


def load_pyproject_dependencies(pyproject_text: str) -> tuple[list[str], list[str]]:
    if tomllib is not None:
        cfg = tomllib.loads(pyproject_text)
        project = cfg.get("project", {})
        deps = list(project.get("dependencies", []))
        extras = project.get("optional-dependencies", {})
        return deps, list(extras.get("dev", []))

    project_section = _extract_section(pyproject_text, "project")
    optional_section = _extract_section(pyproject_text, "project.optional-dependencies")
    deps = _extract_array_strings(project_section, "dependencies")
    dev_deps = _extract_array_strings(optional_section, "dev")
    return deps, dev_deps


def main() -> int:
    if not PYPROJECT.exists() or not LOCK.exists():
        print("missing pyproject.toml or requirements.lock", file=sys.stderr)
        return 1

    deps, dev_deps = load_pyproject_dependencies(PYPROJECT.read_text(encoding="utf-8"))
    deps.extend(dev_deps)

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
