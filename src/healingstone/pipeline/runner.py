"""Pipeline runner with deterministic experiment tracking."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel

from ..schema.config import PipelineConfig
from ..core.utils.determinism import set_seed
from .stage import Stage

LOG = logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrates the sequence of pipeline stages with hard determinism."""

    def __init__(
        self,
        stages: List[Stage],
        config: PipelineConfig,
        input_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.stages = stages
        self.config = config
        self.input_metadata = input_metadata or {}
        
        # Enforce seeding at runner initialization
        set_seed(config.matching.seed)
        
        self.run_id = self._generate_run_id()
        self.run_dir = Path(config.output_dir) / self.run_id
        self.results_dir = self.run_dir / "outputs"
        self.logs_dir = self.run_dir / "logs"

    def _generate_run_id(self) -> str:
        """MANDATORY: run_id = hash(config + input_metadata) [CANONICAL]."""
        # Ensure canonical representation for hashing (sorting keys, ascii, no spaces)
        config_payload = self.config.model_dump()
        payload = {
            "config": config_payload,
            "input": self.input_metadata,
            "system_version": "1.0.0" 
        }
        # Canonical JSON: sorted keys, no whitespace after delimiters, ascii only
        payload_str = json.dumps(
            payload, 
            sort_keys=True, 
            ensure_ascii=True, 
            separators=(',', ':')
        )
        return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()[:16]

    def _initialize_run_dir(self) -> None:
        """Create structured execution directory."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot the configuration
        with (self.run_dir / "config_snapshot.yaml").open("w") as f:
            yaml.dump(self.config.model_dump(), f)
            
        # Metadata snapshot
        metadata_payload = {
            "run_id": self.run_id,
            "input_metadata": self.input_metadata,
            "system_info": {
                "hash_id": self.run_id,
                "config_seed": int(self.config.matching.seed)
            }
        }
        with (self.run_dir / "run_metadata.json").open("w") as f:
            json.dump(metadata_payload, f, indent=2)

    def execute(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """Sequence orchestration with strict determinism guarantees."""
        self._initialize_run_dir()
        LOG.info("event=run_start run_id=%s run_dir=%s", self.run_id, self.run_dir)

        current_data = initial_input
        for stage in self.stages:
            current_data = stage.run(current_data)

        # Final metrics
        metrics = {"run_id": self.run_id, "status": "completed"}
        with (self.run_dir / "metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2)

        LOG.info("event=run_success run_id=%s outputs=%s", self.run_id, self.results_dir)
        return current_data
