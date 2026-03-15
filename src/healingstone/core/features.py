"""Re-export shim: healingstone.core.features → healingstone.features."""

from healingstone.features import (  # noqa: F401
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

__all__ = [
    "FeatureBundle",
    "estimate_geometry_features",
    "detect_break_surface",
    "compute_fpfh",
    "build_fragment_descriptor",
    "augment_fragment_geometry",
    "build_augmented_descriptor",
    "extract_fragment_features",
    "extract_all_features",
]
