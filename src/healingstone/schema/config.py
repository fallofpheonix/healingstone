"""Configuration schemas and validation logic for the pipeline."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PreprocessingConfig(BaseModel):
    """Configuration for fragment preprocessing."""
    sample_points: int = 40000
    voxel_size: float = 0.01
    normal_radius: float = 0.04
    normal_max_nn: int = 64
    outlier_nb_neighbors: int = 24
    outlier_std_ratio: float = 1.8


class MatchingConfig(BaseModel):
    """Configuration for pairwise fragment matching."""
    candidate_top_k: int = 50
    min_match_accuracy: float = 0.8
    device: str = "cpu"
    seed: int = 42


class PipelineConfig(BaseModel):
    """Global configuration for the full reassembly pipeline."""
    project_name: str = "healingstone"
    run_mode: str = Field("3d", pattern="^(3d|2d)$")
    data_dir: str
    output_dir: str = "experiments"
    
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)

    def validate_paths(self) -> None:
        """Custom validation for paths if needed."""
        pass
