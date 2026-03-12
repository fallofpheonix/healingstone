"""Compatibility wrapper for the packaged test entrypoint."""

import warnings

from healingstone.test_pipeline import main


if __name__ == "__main__":
    warnings.warn(
        "Root wrapper test_pipeline.py is deprecated; use 'python -m healingstone.test_pipeline' or 'healingstone-test'.",
        DeprecationWarning,
        stacklevel=1,
    )
    main()
