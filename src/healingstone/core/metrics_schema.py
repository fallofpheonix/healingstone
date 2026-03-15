"""Re-export shim: healingstone.core.metrics_schema → healingstone.metrics_schema."""

from healingstone.metrics_schema import (  # noqa: F401
    METRICS_SCHEMA_VERSION,
    attach_schema_version,
    validate_metrics_schema,
)

__all__ = [
    "METRICS_SCHEMA_VERSION",
    "validate_metrics_schema",
    "attach_schema_version",
]
