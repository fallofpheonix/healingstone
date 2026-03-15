"""Core preprocessing, feature extraction and metrics for the healingstone pipeline.

This package re-exports the core modules.  Importing from this package
requires the ``runtime`` optional dependencies (open3d, torch).  Install with::

    pip install 'healingstone[runtime]'
"""

from __future__ import annotations

try:
    from healingstone.features import (
        FeatureBundle,
        augment_fragment_geometry,
        build_augmented_descriptor,
        build_fragment_descriptor,
        compute_fpfh,
        detect_break_surface,
        estimate_geometry_features,
        extract_all_features,
        extract_fragment_features,
    )
    from healingstone.preprocess import (
        Fragment,
        discover_fragment_files,
        load_and_preprocess_fragments,
        preprocess_fragment,
        set_deterministic_seed,
    )
except ImportError:
    pass  # open3d / torch not installed; individual modules still importable.

from healingstone.metrics_schema import (
    METRICS_SCHEMA_VERSION,
    attach_schema_version,
    validate_metrics_schema,
)

__all__ = [
    "Fragment",
    "set_deterministic_seed",
    "discover_fragment_files",
    "preprocess_fragment",
    "load_and_preprocess_fragments",
    "FeatureBundle",
    "estimate_geometry_features",
    "detect_break_surface",
    "compute_fpfh",
    "build_fragment_descriptor",
    "augment_fragment_geometry",
    "build_augmented_descriptor",
    "extract_fragment_features",
    "extract_all_features",
    "METRICS_SCHEMA_VERSION",
    "validate_metrics_schema",
    "attach_schema_version",
]
