from __future__ import annotations

import healingstone.pipeline as pipeline


def test_pipeline_package_lazy_exports() -> None:
    assert callable(pipeline.main)
    assert callable(pipeline.run_pipeline)
    assert "main" in dir(pipeline)
    assert "run_pipeline" in dir(pipeline)
