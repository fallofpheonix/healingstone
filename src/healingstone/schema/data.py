"""Data contracts for the healingstone reassembly pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class SchemaBase(BaseModel):
    """Base class for all schemas with common configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)


class InputSample(SchemaBase):
    """MANDATORY: Formal input schema for the pipeline."""
    fragments: List[Dict[str, Any]]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    labels_csv_path: Optional[Path] = None


class IntermediateRepresentation(SchemaBase):
    """MANDATORY: Data passed between internal stages."""
    stage_name: str
    payload: Dict[str, Any]
    metrics: Dict[str, Any] = Field(default_factory=dict)


class MatchResult(SchemaBase):
    """MANDATORY: Result of a pairwise registration between two fragments."""
    frag_a_id: int
    frag_b_id: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    transformation: List[List[float]] = Field(..., description="4x4 rigid matrix")
    alignment_error: float = Field(..., description="MRE for this specific pair")


class OutputResult(SchemaBase):
    """MANDATORY: Final validated output schema with scientific metrics."""
    run_id: str
    reconstruction_path: Path
    # Formalized Scientific Metrics
    mre: float = Field(..., description="Mean Registration Error across all matched interfaces")
    completeness: float = Field(..., ge=0.0, le=1.0, description="Ratio of total fragments integrated")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Auxiliary metrics")
    global_transforms: Dict[int, List[List[float]]]


class FragmentMetadata(SchemaBase):
    """Metadata for a single fragment."""
    id: int
    name: str
    source_path: Path
    num_points: int


class Fragment(SchemaBase):
    """Container for fragment point cloud and metadata."""
    metadata: FragmentMetadata
    points: np.ndarray = Field(..., description="Numpy array (N, 3)")
    normals: Optional[np.ndarray] = Field(None, description="Numpy array (N, 3)")
