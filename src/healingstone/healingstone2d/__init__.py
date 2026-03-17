"""2D fragment reconstruction pipeline for PNG image fragments.

This package implements the full 2D reconstruction pipeline:

    fragment images (.PNG)
    → preprocess_2d      – image loading, normalisation, background removal
    → edge_detection     – Canny edges and contour extraction
    → shape_descriptors  – Hu moments + Fourier descriptors per fragment
    → match_fragments_2d – shape-similarity matching, candidate selection
    → align_fragments_2d – rigid transform estimation (rotation + translation)
    → reconstruct_2d     – canvas assembly and output image

Entry point::

    from healingstone2d.reconstruct_2d import run_2d_pipeline
    run_2d_pipeline(data_dir, output_dir, seed=42)

Requires: opencv-python, numpy.  Install with::

    pip install 'healingstone[runtime]'
"""

from __future__ import annotations

try:
    from .align_fragments_2d import AlignmentResult2D, align_candidate_pairs_2d
    from .edge_detection import detect_edges, extract_contours
    from .match_fragments_2d import FragmentMatch, match_all_fragments
    from .preprocess_2d import Fragment2D, load_and_preprocess_fragments_2d
    from .reconstruct_2d import run_2d_pipeline
    from .shape_descriptors import ShapeDescriptor, extract_all_descriptors
except ImportError:
    pass

__all__ = [
    "Fragment2D",
    "load_and_preprocess_fragments_2d",
    "detect_edges",
    "extract_contours",
    "ShapeDescriptor",
    "extract_all_descriptors",
    "FragmentMatch",
    "match_all_fragments",
    "AlignmentResult2D",
    "align_candidate_pairs_2d",
    "assemble_fragments_2d",
    "run_2d_pipeline",
]
