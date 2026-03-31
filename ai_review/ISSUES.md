# AI Review: HealingStones

## Current Status

Documentation: was stale, now partially reconciled with code
Repository Structure: package-first implementation exists under `healingstone/src/healingstone/`
Core Pipeline Code: packaged pipeline, tests, and compatibility wrappers are present
Baseline Runtime: packaged pipeline completed on local 17-fragment set

## Verified Findings

- `python -m healingstone.run_pipeline` exists and is exercised by tests for CLI help behavior
- package compatibility modules exist for `healingstone.run_pipeline`, `healingstone.runtime_config`, `healingstone.runtime_paths`, and `healingstone.metrics_schema`
- FPFH extraction is implemented in `core/features.py`
- graph-based global assembly is implemented in `alignment/reconstruct.py`
- the repo contains both packaged and legacy monolithic pipelines; canonical path is the packaged one
- baseline packaged run produced 15 successful alignments with mean ICP RMSE `0.00915`

## Real Problems

- `data/3D_fragments.zip` is not a valid zip file; it is a truncated tar archive
- deprecated flat artifacts under `healingstone/artifacts/results/` coexist with canonical run-scoped artifacts
- the legacy monolith `src/healingstone/healing_stones.py` duplicates pipeline logic and increases maintenance surface
- no verified in-repo labeled real-data pair dataset is available for hard accuracy calibration

## Next Actions

1. Replace or remove the broken top-level archive and document the canonical dataset path.
2. Run the packaged pipeline end-to-end on the included `healingstone/3D/` fragment set.
3. Decide whether to freeze or delete the legacy monolithic pipeline.
4. Add labeled-pair evaluation data or clear instructions for generating it.
