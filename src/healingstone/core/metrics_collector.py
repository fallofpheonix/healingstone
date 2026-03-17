"""Pipeline metrics collector for performance monitoring.

Tracks execution timing, memory usage, and mesh processing statistics
across pipeline stages.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List

LOG = logging.getLogger(__name__)


@dataclass
class StageMetric:
    """Timing and resource data for a single pipeline stage."""

    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    elapsed_ms: float = 0.0
    memory_mb: float = 0.0
    metadata: Dict[str, any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and reports pipeline execution metrics."""

    def __init__(self):
        self.stages: List[StageMetric] = []
        self._active: Dict[str, StageMetric] = {}

    @contextmanager
    def track_stage(self, name: str) -> Generator[StageMetric, None, None]:
        """Context manager to track a pipeline stage's timing and memory."""
        metric = StageMetric(name=name, start_time=time.time())
        self._active[name] = metric

        try:
            yield metric
        finally:
            metric.end_time = time.time()
            metric.elapsed_ms = round((metric.end_time - metric.start_time) * 1000, 2)
            metric.memory_mb = _get_memory_mb()

            LOG.info(
                "Stage '%s' completed: %.1fms, %.1f MB",
                name,
                metric.elapsed_ms,
                metric.memory_mb,
                extra={"stage": name, "elapsed_ms": metric.elapsed_ms},
            )

            self.stages.append(metric)
            self._active.pop(name, None)

    def record(self, name: str, value: float, unit: str = ""):
        """Record a custom metric."""
        self.stages.append(
            StageMetric(
                name=name,
                metadata={"value": value, "unit": unit},
            )
        )

    def total_elapsed_ms(self) -> float:
        """Total wall-clock time across all stages."""
        return sum(s.elapsed_ms for s in self.stages)

    def summary(self) -> Dict[str, any]:
        """Generate a metrics summary dictionary."""
        return {
            "total_elapsed_ms": round(self.total_elapsed_ms(), 2),
            "n_stages": len(self.stages),
            "stages": [
                {
                    "name": s.name,
                    "elapsed_ms": s.elapsed_ms,
                    "memory_mb": s.memory_mb,
                    **s.metadata,
                }
                for s in self.stages
            ],
        }

    def write_report(self, output_path: Path):
        """Write metrics to a JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.summary(), indent=2, default=str),
            encoding="utf-8",
        )
        LOG.info("Metrics report written to %s", output_path)


def _get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return round(usage.ru_maxrss / 1024 / 1024, 1)  # macOS: bytes -> MB
    except Exception:
        return 0.0
