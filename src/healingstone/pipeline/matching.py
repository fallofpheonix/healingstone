"""Implementation of the Matching Stage."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from healingstone.core.geometry.features import extract_all_features
from healingstone.ml_models.match_fragments import train_and_match_fragments
from healingstone.pipeline.stage import Stage
from healingstone.pipeline.validators import validate_match_candidates

LOG = logging.getLogger(__name__)


class MatchingStage(Stage):
    """MANDATORY: Stage for pairwise fragment matching logic."""

    def __init__(self, name: str, config: Any, results_dir: Path, models_dir: Path, cache_dir: Path):
        super().__init__(name, config)
        self.results_dir = results_dir
        self.models_dir = models_dir
        self.cache_dir = cache_dir

    def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform pairwise matching using the configured model."""
        fragments = input_data.get("fragments", [])
        if not fragments:
            raise ValueError("MatchingStage requires a non-empty list of fragments")
            
        LOG.info("event=matching_started num_fragments=%d", len(fragments))
        
        # 1. Feature extraction with caching
        LOG.info("event=feature_extraction_started")
        features = extract_all_features(
            fragments=fragments,
            cache_dir=self.cache_dir,
            k_neighbors=24, # Default params from core
            fpfh_radius=0.06,
            fpfh_max_nn=100,
            dbscan_eps=0.04,
            dbscan_min_samples=24,
            n_keypoints=256,
        )
        
        # 2. Train and match
        # Function: train_and_match_fragments
        similarity, candidate_pairs, pair_scores, diagnostics, _ = train_and_match_fragments(
            fragments=fragments,
            features=features,
            models_dir=self.models_dir,
            output_dir=self.results_dir,
            seed=self.config.seed,
            device=self.config.device,
            candidate_top_k=self.config.candidate_top_k,
        )
        
        # 3. MANDATORY validation
        if not validate_match_candidates(candidate_pairs, len(fragments)):
            raise RuntimeError("MatchingStage validation failed: No candidates produced")
        
        LOG.info("event=matching_success num_candidates=%d", len(candidate_pairs))
        
        return {
            "stage": "matching",
            "matches": candidate_pairs,
            "pair_scores": pair_scores,
            "diagnostics": diagnostics,
            "fragments": fragments
        }
