# GPT Context Pack

## Upload Order

1. `SPEC.md`
2. `ARCH.md`
3. `STATE.md`
4. `TASKS.md`
5. `API.md`
6. `DATA.md`
7. `SETUP.md`
8. `CONSTRAINTS.md`
9. `EXPERIMENTS.md`
10. `ROADMAP.md`
11. `DECISIONS.md`

## Mandatory Code Files

- `src/healingstone/pipeline/run_pipeline.py`
- `src/healingstone/core/runtime_paths.py`
- `src/healingstone/core/runtime_config.py`
- `src/healingstone/core/metrics_schema.py`
- `src/healingstone/ml_models/match_fragments.py`
- `src/healingstone/alignment/align_fragments.py`
- `src/healingstone/alignment/reconstruct.py`
- `src/healingstone/core/preprocess.py`
- `src/healingstone/core/features.py`

## Mandatory Config Files

- `configs/pipeline.yaml`
- `configs/train.yaml`
- `configs/datasets.yaml`
- `pyproject.toml`

## Exclude By Default

- `artifacts/**`
- `artifacts/submission/**`
- `data/raw/3d/*.PLY`
- historical docs under `docs/**`
- `.venv/**`
- generated caches and reports

## Include Historical Evidence Only If Needed

- `docs/DATASET_INTEGRITY_REPORT.md`
- `docs/RECONSTRUCTION_BENCHMARK_REPORT.md`
- `docs/MESH_PERFORMANCE_REPORT.md`
- `docs/FINAL_SYSTEM_CERTIFICATION.md`

## Purpose Of This Pack

- `SPEC.md`: system boundary
- `ARCH.md`: structure and flow
- `STATE.md`: current reality
- `TASKS.md`: execution frontier
- `API.md`: integration contracts
- `DATA.md`: input and artifact schemas
- `SETUP.md`: reproducible environment
- `CONSTRAINTS.md`: non-negotiable limits
- `EXPERIMENTS.md`: evidence and failed loops
- `ROADMAP.md`: milestone plan
- `DECISIONS.md`: rationale and tradeoffs
