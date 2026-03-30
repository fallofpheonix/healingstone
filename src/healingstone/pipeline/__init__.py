"""Pipeline orchestration package exports."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

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


def __getattr__(name: str) -> Any:
    if name in __all__:
        run_pipeline_module = importlib.import_module(".run_pipeline", __name__)
        return getattr(run_pipeline_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
