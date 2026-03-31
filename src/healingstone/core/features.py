"""Compatibility re-exports for geometry feature extraction."""

from __future__ import annotations

from .geometry.features import (
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
    "augment_fragment_geometry",
    "build_augmented_descriptor",
    "build_fragment_descriptor",
    "compute_fpfh",
    "detect_break_surface",
    "estimate_geometry_features",
    "extract_all_features",
    "extract_fragment_features",
]
