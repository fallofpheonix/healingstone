"""Re-export shim: healingstone.pipeline.run_pipeline → healingstone.run_pipeline."""

from healingstone.run_pipeline import (  # noqa: F401
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
