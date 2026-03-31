"""RANSAC + ICP geometric alignment and global reconstruction assembly.

Requires the ``runtime`` optional dependencies (open3d, torch).  Install with::

    pip install 'healingstone[runtime]'
"""

from __future__ import annotations

try:
    from .align_fragments import (
        AlignmentResult,
        align_candidate_pairs,
        align_pair,
        chamfer_distance,
    )
    from .reconstruct import (
        AssemblyResult,
        assemble_global_reconstruction,
        build_fragment_graph,
        compute_global_transforms,
        merge_and_save_reconstruction,
    )
except ImportError:
    pass

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
