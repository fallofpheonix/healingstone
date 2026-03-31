"""Implementation of the Preprocessing Stage."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from healingstone.core.preprocess import load_and_preprocess_fragments
from healingstone.pipeline.stage import Stage
from healingstone.pipeline.validators import validate_fragment

LOG = logging.getLogger(__name__)


class PreprocessingStage(Stage):
    """MANDATORY: Stage for loading and initial cleaning of 3D fragments."""

    def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Load, denoise, and downsample fragment point clouds."""
        metadata = input_data.get("metadata", {})
        data_dir = metadata.get("source")
        
        if not data_dir:
            raise ValueError("Preprocessing requires a 'source' data directory in metadata")
        
        LOG.info("event=preprocessing_started data_dir=%s", data_dir)
        
        # MANDATORY: Delegate to core.preprocess with config
        fragments = load_and_preprocess_fragments(
            data_dir=Path(data_dir),
            sample_points=self.config.sample_points,
            voxel_size=self.config.voxel_size,
            normal_radius=self.config.normal_radius,
            normal_max_nn=self.config.normal_max_nn,
            outlier_nb_neighbors=self.config.outlier_nb_neighbors,
            outlier_std_ratio=self.config.outlier_std_ratio,
        )
        
        # MANDATORY: Validate every preprocessed fragment
        validated_fragments = []
        for frag in fragments:
            if validate_fragment(frag):
                validated_fragments.append(frag)
            else:
                LOG.warning("event=preprocessing_warning reason=fragment_validation_failed idx=%d", frag.idx)
        
        if not validated_fragments:
            raise RuntimeError("Preprocessing stage produced zero valid fragments")
            
        LOG.info("event=preprocessing_success num_fragments=%d", len(validated_fragments))
        
        return {
            "stage": "preprocessing",
            "fragments": validated_fragments,
            "status": "success"
        }
