"""Compatibility re-exports for adaptive voxel downsampling."""

from __future__ import annotations

from .geometry.adaptive_voxel_downsampling import (
    DownsampleResult,
    adaptive_voxel_downsample,
    benchmark_downsampling,
    estimate_voxel_size,
    load_and_downsample,
)

__all__ = [
    "DownsampleResult",
    "adaptive_voxel_downsample",
    "benchmark_downsampling",
    "estimate_voxel_size",
    "load_and_downsample",
]
