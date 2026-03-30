# Current State

## Snapshot

- Date: `2026-03-30`
- Branch: `push-clean`
- Verified code baseline: `e4ec6cb`
- Canonical dataset root: `data/raw/3d`
- Canonical artifact root: `artifacts`

## Verified Now

- repository layout normalized to project-root-relative paths
- root dataset/output clutter removed
- `run_pipeline.py` reduced to orchestration
- plotting isolated in `src/healingstone/utils/visualization.py`
- 3D metrics summarization isolated in `src/healingstone/core/metrics_collector.py`
- default developer loop passes in current `.venv`:
  - `pytest -q`
  - `ruff check .`
  - `mypy`

## Verified With Environment Gating

- `pytest -q -rs` passes with Open3D-dependent tests skipped
- skipped tests:
  - `tests/test_train_config_forwarding.py`
  - `tests/test_reconstruction_pipeline.py` 3D smoke checks

## Historically Verified Evidence

- local 17-fragment 3D run exists in project history
- recorded metrics from prior run:
  - candidate pairs: `46`
  - successful alignments: `15`
  - mean ICP RMSE: `0.00915`
  - mean Chamfer distance: `0.24294`
  - reconstruction completeness: `0.49094`
  - assembled fragments: `8`

## Completed Components

- package-first CLI entrypoint
- config precedence resolution: `CLI > ENV > YAML`
- deterministic path resolution and run layout
- 3D preprocessing and feature extraction
- 3D pair scoring and alignment
- 2D preprocessing, matching, alignment, and rendering
- schema-validated metrics report generation
- run metadata and resolved-path metadata
- diagnostic plotting

## Partial Components

- real labeled-pair evaluation workflow exists, but labeled data is not present in-repo
- benchmark reports exist, but acceptance thresholds are not calibrated on supervised data
- submission bundle exists under `artifacts/submission`, but should be treated as generated output

## Broken Or Environment-Blocked

- full 3D execution is blocked in the current `.venv` because `open3d` is not installed
- full 3D runtime requires Python `3.10` to `3.12` plus runtime extras

## Known Bugs Or Risks

- no current supervised accuracy signal for the local 17-fragment dataset
- large meshes remain CPU-heavy even after downsampling
- historical docs under `docs/` contain generated claims and should not be treated as canonical context
