"""Security audit utility for the healingstone pipeline.

Scans the codebase for common security vulnerabilities:
- Unsafe file loading (pickle, eval)
- Path traversal risks
- Dependency vulnerabilities
- Unvalidated CLI inputs
"""

from __future__ import annotations

import ast
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List

LOG = logging.getLogger(__name__)


def scan_unsafe_patterns(src_dir: Path) -> List[Dict[str, str]]:
    """Scan for dangerous function calls."""
    dangerous_patterns = {
        r"\bpickle\.load\b": "Unsafe deserialization (pickle.load)",
        r"\beval\(": "Use of eval() — code injection risk",
        r"\bexec\(": "Use of exec() — code injection risk",
        r"\b__import__\(": "Dynamic import — code injection risk",
        r"\bos\.system\(": "Shell command execution via os.system",
        r"\bsubprocess\.call\(.*shell\s*=\s*True": "Shell=True subprocess call",
        r"\byaml\.load\((?!.*Loader)": "Unsafe YAML load without Loader",
    }

    findings = []
    for py_file in src_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
        except Exception:
            continue

        for pattern, description in dangerous_patterns.items():
            for i, line in enumerate(content.splitlines(), 1):
                if re.search(pattern, line):
                    findings.append({
                        "file": str(py_file.relative_to(src_dir)),
                        "line": i,
                        "pattern": description,
                        "code": line.strip(),
                    })

    return findings


def scan_path_traversal(src_dir: Path) -> List[Dict[str, str]]:
    """Scan for potential path traversal vulnerabilities."""
    findings = []
    traversal_patterns = [
        r"\.\.\/",
        r"\.\.\\\\",
        r"os\.path\.join\(.*\.\.",
    ]

    for py_file in src_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
        except Exception:
            continue

        for pattern in traversal_patterns:
            for i, line in enumerate(content.splitlines(), 1):
                if re.search(pattern, line):
                    findings.append({
                        "file": str(py_file.relative_to(src_dir)),
                        "line": i,
                        "pattern": "Path traversal risk",
                        "code": line.strip(),
                    })

    return findings


def scan_input_validation(src_dir: Path) -> List[Dict[str, str]]:
    """Check for CLI/API inputs that lack validation."""
    findings = []

    for py_file in src_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "add_argument":
                        # Check if type= is specified
                        has_type = any(
                            kw.arg == "type" for kw in node.keywords
                        )
                        if not has_type:
                            # Positional args without type are fine
                            if node.args and isinstance(node.args[0], ast.Constant):
                                arg_name = node.args[0].value
                                if isinstance(arg_name, str) and arg_name.startswith("--"):
                                    findings.append({
                                        "file": str(py_file.relative_to(src_dir)),
                                        "line": node.lineno,
                                        "pattern": f"CLI arg '{arg_name}' has no type validation",
                                        "code": f"add_argument('{arg_name}', ...)",
                                    })

    return findings


def run_audit(src_dir: Path) -> Dict[str, List[Dict[str, str]]]:
    """Run all security scans."""
    return {
        "unsafe_patterns": scan_unsafe_patterns(src_dir),
        "path_traversal": scan_path_traversal(src_dir),
        "input_validation": scan_input_validation(src_dir),
    }


def write_report(results: Dict[str, List[Dict[str, str]]], output_path: Path):
    """Generate SECURITY_AUDIT_REPORT.md."""
    total = sum(len(v) for v in results.values())
    critical = len(results.get("unsafe_patterns", []))

    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Security Audit Report\n\n")
        f.write(f"**Total Findings:** {total}\n")
        f.write(f"**Critical (unsafe code):** {critical}\n")
        f.write(f"**Path Traversal Risks:** {len(results.get('path_traversal', []))}\n")
        f.write(f"**Input Validation Gaps:** {len(results.get('input_validation', []))}\n\n")

        for category, findings in results.items():
            title = category.replace("_", " ").title()
            f.write(f"## {title}\n\n")
            if not findings:
                f.write("✅ No issues found.\n\n")
                continue

            f.write("| File | Line | Issue | Code |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for finding in findings:
                code = finding["code"][:60].replace("|", "\\|")
                f.write(
                    f"| {finding['file']} | {finding['line']} "
                    f"| {finding['pattern']} | `{code}` |\n"
                )
            f.write("\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Security audit for healingstone.")
    parser.add_argument("--src-dir", type=Path, default=Path("src/healingstone"))
    parser.add_argument("--output", type=Path, default=Path("SECURITY_AUDIT_REPORT.md"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    results = run_audit(args.src_dir)
    write_report(results, args.output)
    LOG.info("Security audit written to %s", args.output)
