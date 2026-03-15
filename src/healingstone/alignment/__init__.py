"""RANSAC + ICP geometric alignment and global reconstruction assembly.

Requires the ``runtime`` optional dependencies (open3d, torch).  Install with::

    pip install 'healingstone[runtime]'
"""

from __future__ import annotations

try:
    from healingstone.align_fragments import (
        AlignmentResult,
        align_candidate_pairs,
        align_pair,
        chamfer_distance,
    )
    from healingstone.reconstruct import (
        AssemblyResult,
        assemble_global_reconstruction,
        build_fragment_graph,
        compute_global_transforms,
        merge_and_save_reconstruction,
    )
except ImportError:
    pass  # open3d not installed; individual modules still importable.

__all__ = [
    "AlignmentResult",
    "align_pair",
    "chamfer_distance",
    "align_candidate_pairs",
    "AssemblyResult",
    "build_fragment_graph",
    "compute_global_transforms",
    "assemble_global_reconstruction",
    "merge_and_save_reconstruction",
]
