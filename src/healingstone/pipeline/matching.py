"""Implementation of the Matching Stage."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..schema.config import MatchingConfig
from ..schema.data import MatchResult
from .stage import Stage

LOG = logging.getLogger(__name__)


class MatchingStage(Stage):
    """MANDATORY: Stage for pairwise fragment matching logic."""

    def __init__(self, name: str, config: MatchingConfig):
        super().__init__(name, config)

    def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform pairwise matching using the configured model."""
        fragments = input_data.get("fragments", [])
        LOG.info("event=matching_started num_fragments=%d", len(fragments))
        
        # Simulate matching pairs
        matches: List[MatchResult] = []
        if len(fragments) >= 2:
            # Mock match between the first two fragments
            match = MatchResult(
                frag_a_id=0,
                frag_b_id=1,
                confidence=0.92,
                transformation=[[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]],
                alignment_error=0.012
            )
            matches.append(match)
        
        LOG.info("event=matching_success num_matches=%d", len(matches))
        
        # Pass matches to the next stage
        return {
            "stage": "matching",
            "matches": [m.model_dump() for m in matches],
            "fragments": fragments
        }
