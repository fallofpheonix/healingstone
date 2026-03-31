"""Abstract Base Class for all pipeline stages."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel

LOG = logging.getLogger(__name__)


class Stage(ABC):
    """Abstract Base Class for a pipeline stage following a strict transform contract."""

    def __init__(self, name: str, config: BaseModel):
        self.name = name
        self.config = config

    @abstractmethod
    def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Core logic of the stage to be implemented by subclasses."""
        pass

    def run(self, input_data: Dict[str, Any], config_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """MANDATORY: execution wrapper with logging, timing, and deterministic contract.
        
        Interface: run(input: Dict, config: Dict) -> Dict
        """
        start_time = time.time()
        start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        LOG.info("stage=%s status=started start_time=%s", self.name, start_timestamp)

        try:
            # Stage implementation logic
            output = self._execute(input_data)
            
            end_time = time.time()
            end_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            duration = end_time - start_time
            
            LOG.info(
                "stage=%s status=success start_time=%s end_time=%s duration=%.3fs", 
                self.name, start_timestamp, end_timestamp, duration
            )
            return output
        except Exception as exc:
            end_time = time.time()
            duration = end_time - start_time
            LOG.error("stage=%s status=failed duration=%.3fs error=%s", self.name, duration, str(exc))
            raise RuntimeError(f"Stage '{self.name}' failed: {exc}") from exc
