"""Compatibility wrapper for the packaged pipeline entrypoint."""

import warnings

from healingstone.run_pipeline import main


if __name__ == "__main__":
    warnings.warn(
        "Root wrapper run_pipeline.py is deprecated; use 'python -m healingstone.run_pipeline' or 'healingstone-run'.",
        DeprecationWarning,
        stacklevel=1,
    )
    main()
