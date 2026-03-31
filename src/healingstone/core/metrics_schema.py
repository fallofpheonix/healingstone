"""Compatibility re-exports for metrics schema helpers."""

from __future__ import annotations

from ..schema.metrics_schema import (
    METRICS_SCHEMA_VERSION,
    MetricsSchemaError,
    attach_schema_version,
    validate_metrics_schema,
)

__all__ = [
    "METRICS_SCHEMA_VERSION",
    "MetricsSchemaError",
    "attach_schema_version",
    "validate_metrics_schema",
]
