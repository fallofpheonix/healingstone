# Architecture & Roadmap Plan: Healing Stones

## Objective

Deliver a deterministic, package-first reconstruction pipeline for fragmented 3D cultural-heritage scans that runs from CLI without manual intervention and produces run-scoped artifacts.

## Verified Data Flow

```text
Fragments (.PLY/.OBJ)
  -> discovery + Open3D load
  -> denoise + voxel downsample + unit-sphere normalization + normals
  -> heuristic break-surface scoring
  -> pseudo-label extraction
  -> point-wise surface classifier training
  -> cached FPFH + fragment descriptor extraction
  -> Siamese descriptor training
  -> pairwise similarity matrix + reciprocal top-k candidates
  -> RANSAC + ICP alignment for top pairs
  -> weighted graph assembly + MST transform propagation
  -> merged reconstruction + metrics + metadata + plots
```

## Code Mapping

- `healingstone/src/healingstone/core/preprocess.py`
- `healingstone/src/healingstone/core/features.py`
- `healingstone/src/healingstone/ml_models/surface_model.py`
- `healingstone/src/healingstone/ml_models/train_model.py`
- `healingstone/src/healingstone/ml_models/match_fragments.py`
- `healingstone/src/healingstone/alignment/align_fragments.py`
- `healingstone/src/healingstone/alignment/reconstruct.py`
- `healingstone/src/healingstone/pipeline/run_pipeline.py`

## Operational Invariants

- Config precedence is `CLI > ENV > YAML`.
- `config_version` must be supported for both pipeline and training configs.
- Data path resolution is strict for explicit CLI/ENV paths.
- Artifacts are run-scoped under `artifacts/runs/<run_id>/`.
- Metrics must satisfy the schema in `core/metrics_schema.py`.
- Accuracy hard-gate requires `evaluation_split=test`.

## Current Implemented State

Implemented:
- package-first CLI entrypoint `python -m healingstone.run_pipeline`
- compatibility wrappers for legacy import paths
- preprocessing and normal estimation
- heuristic break-surface scoring
- pseudo-label-driven surface classifier training
- descriptor extraction with persistent feature cache
- Siamese embedding matcher
- reciprocal candidate selection
- pairwise RANSAC + ICP alignment
- graph-based global assembly
- run metadata, resolved-path metadata, metrics schema, plots
- automated checks: `ruff`, `mypy`, `pytest`
- verified packaged run on the local 17-fragment `healingstone/3D/` dataset

Not yet production-complete:
- no validated labeled-pair dataset in-repo for real accuracy measurement
- no benchmarked real-data acceptance threshold beyond schema plumbing
- no documented dataset ingestion path for the broken top-level archive
- legacy monolithic pipeline still coexists with package-first pipeline
- artifact tree contains both deprecated flat outputs and canonical run-scoped outputs

Observed baseline on local 17-fragment run:
- candidate pairs: `46`
- successful alignments: `15`
- mean ICP RMSE: `0.00915`
- mean Chamfer distance: `0.24294`
- reconstruction completeness: `0.49094`
- assembled fragments: `8`
- pairwise accuracy unavailable because no labels were supplied

## Near-Term Roadmap

### Phase 1: Repository Stabilization
- document canonical dataset placement and execution path
- remove stale claims from docs
- treat `data/3D_fragments.zip` as invalid until replaced or repaired
- decide retirement plan for `src/healingstone/healing_stones.py`

### Phase 2: Dataset and Evaluation
- add a verified real-data manifest or retrieval instructions
- provide labeled pair CSV workflow for supervised evaluation
- calibrate `min_match_accuracy` and `min_required_accuracy` on real fragments
- define acceptance thresholds for completeness and Chamfer distance

### Phase 3: Runtime Robustness
- exercise the packaged pipeline end-to-end on the included `healingstone/3D/` set
- capture failure cases for sparse or degenerate fragments
- add tests around report generation and artifact layout

### Phase 4: Model/Alignment Quality
- benchmark heuristic break detection versus trained classifier contribution
- improve graph construction criteria using alignment diagnostics
- quantify false-positive pair generation on larger fragment sets

## Final Deliverables

1. Automated package-first pipeline.
2. Reproducible run-scoped artifacts and metadata.
3. Reconstruction metrics with schema validation.
4. Visual diagnostics for similarity, alignment, and final reconstruction.
5. Documentation aligned with actual repository state.
