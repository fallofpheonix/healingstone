# GPT Context Pack

## 1. Primary Documentation (Upload Order)

1. `specification.md`: system boundary and approach.
2. `project_status.md`: current reality and current tasks.
3. `user_guide.md`: reproducible environment and setup.
4. `decisions_and_constraints.md`: rationale and non-negotiable limits.
5. `data_and_experiments.md`: input and artifact schemas and logs.

---

## 2. Mandatory Code Files

- `src/healingstone/pipeline/run_pipeline.py`
- `src/healingstone/core/runtime_paths.py`
- `src/healingstone/core/runtime_config.py`
- `src/healingstone/core/metrics_schema.py`
- `src/healingstone/ml_models/match_fragments.py`
- `src/healingstone/alignment/align_fragments.py`
- `src/healingstone/alignment/reconstruct.py`
- `src/healingstone/core/preprocess.py`
- `src/healingstone/core/features.py`

---

## 3. Mandatory Config Files

- `configs/pipeline.yaml`
- `configs/train.yaml`
- `configs/datasets.yaml`
- `pyproject.toml`

---

## 4. Exclude By Default

- `artifacts/**`
- `data/raw/3d/*.PLY`
- `.venv/**`
- generated caches and reports

---

## 5. Historical Evidence (Only if needed)

- `reports/DATASET_INTEGRITY_REPORT.md` (Already merged, but can be found as historical)
- `reports/RECONSTRUCTION_BENCHMARK_REPORT.md`
- `reports/MESH_PERFORMANCE_REPORT.md`
- `reports/FINAL_SYSTEM_CERTIFICATION.md`
- `submission/SUBMISSION_MANIFEST.md`

---

## Purpose of this Pack
This index organizes the repository context into structured layers to prevent information entropy when interacting with GPT-based assistants.
- **Specification**: boundaries and system logic.
- **Status**: execution frontier.
- **Guide**: operational knowledge.
- **Decisions**: trade-offs and rationale.
- **Data**: input/output reality.
