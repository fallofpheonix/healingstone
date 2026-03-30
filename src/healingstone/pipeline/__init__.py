"""Pipeline orchestration entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .run_pipeline import (  # noqa: F401
        configure_logging,
        detect_pipeline_mode,
        enforce_accuracy_requirement,
        main,
        parse_args,
        plot_alignment_snapshots,
        plot_final_reconstruction,
        plot_similarity_matrix,
        run_pipeline,
        summarize_metrics,
    )


def __getattr__(name: str) -> object:
    if name in {
        "configure_logging",
        "detect_pipeline_mode",
        "enforce_accuracy_requirement",
        "main",
        "parse_args",
        "plot_alignment_snapshots",
        "plot_final_reconstruction",
        "plot_similarity_matrix",
        "run_pipeline",
        "summarize_metrics",
    }:
        from . import run_pipeline as _rp
        return getattr(_rp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "configure_logging",
    "detect_pipeline_mode",
    "parse_args",
    "run_pipeline",
    "summarize_metrics",
    "enforce_accuracy_requirement",
    "plot_similarity_matrix",
    "plot_alignment_snapshots",
    "plot_final_reconstruction",
    "main",
]
