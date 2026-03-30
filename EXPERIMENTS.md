# Experiments

## Log

### 2026-03-16: Local 17-Fragment 3D Pipeline Run

- setup:
  - dataset: `data/raw/3d`
  - labels: none
- result:
  - candidate pairs: `46`
  - successful alignments: `15`
  - mean ICP RMSE: `0.00915`
  - mean Chamfer distance: `0.24294`
  - reconstruction completeness: `0.49094`
  - assembled fragments: `8`
- conclusion:
  - pipeline produces a partial coherent assembly
  - supervised accuracy cannot be measured without labeled pairs

### 2026-03-16: Unaligned Pair Benchmark

- source: `docs/RECONSTRUCTION_BENCHMARK_REPORT.md`
- result:
  - FR_01 vs FR_02 Chamfer: `12.4832`
  - FR_01 vs FR_02 RMSE: `14.7621`
  - completeness ranges observed: `38.70%` to `55.10%`
- conclusion:
  - raw fragment distances are high before alignment
  - benchmark output is useful for regression tracking, not acceptance alone

### 2026-03-16: Adaptive Downsampling Benchmark

- source: `docs/MESH_PERFORMANCE_REPORT.md`
- result:
  - large fragments reduced by `93%` to `96%`
  - target point count: about `500K`
  - average CPU preprocessing time: about `3.5s` per fragment
- conclusion:
  - CPU preprocessing is feasible
  - aggressive downsampling is mandatory for large meshes

### 2026-03-30: Developer Environment Verification

- commands:
  - `pytest -q -rs`
  - `ruff check .`
  - `mypy`
- result:
  - default test suite passes
  - skipped tests: `3`
  - skip reason: `open3d` absent in current `.venv`
- conclusion:
  - default development loop is stable
  - full 3D runtime verification still requires Python `3.12` + runtime extras

## Repeated Failure To Avoid

- do not use Python `3.13` for full 3D runtime validation
- do not assume pairwise accuracy is meaningful without curated labels
- do not upload generated artifacts or raw meshes as primary GPT context
