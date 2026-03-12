"""Compatibility wrapper for the legacy monolithic entrypoint."""

import warnings

from healingstone.healing_stones import main, run_pipeline

__all__ = ["main", "run_pipeline"]


if __name__ == "__main__":
    warnings.warn(
        "Root wrapper healing_stones.py is deprecated; use 'python -m healingstone.healing_stones' or 'healingstone-legacy'.",
        DeprecationWarning,
        stacklevel=1,
    )
    main()
