# Healing Stones 3D Reconstruction Pipeline

Reproducible, package-first pipeline for fragmented 3D cultural heritage reconstruction (`.PLY/.OBJ`).

## Install

Python `3.10-3.12` is recommended for full runtime dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e .[runtime]   # open3d/torch for full 3D runtime
pip install -e .[dev]       # tests + lint + type checks
```

Reproducible lock-based install:

```bash
pip install -r requirements.lock
```

## Canonical Execution

```bash
python -m healingstone.run_pipeline
```

Console script entrypoints (after install):

```bash
healingstone-run
healingstone-test
healingstone-legacy
```

## Compatibility Wrappers (Deprecated)

These remain supported but emit deprecation warnings:

```bash
python run_pipeline.py
python test_pipeline.py
python healing_stones.py
```

## Runtime Config Precedence

Precedence order is strict:

```text
CLI > ENV > YAML
```

- Config files:
  - `configs/pipeline.yaml`
  - `configs/train.yaml`
  - `configs/datasets.yaml`
- Environment override prefix: `HEALINGSTONE_`
- Example:

```bash
HEALINGSTONE_OUTPUT_DIR=/tmp/artifacts python -m healingstone.run_pipeline
```

## Path Resolution Policy

Data path resolution:
1. Explicit CLI/ENV `data_dir` -> use or fail (no fallback).
2. YAML `data_dir` or `dataset_alias` target.
3. Legacy fallback `DataSet/3D` with warning.
4. Hard error.

Artifact root resolution:
1. Explicit CLI/ENV `output_dir` -> use or fail.
2. YAML/default canonical root `artifacts`.
3. Legacy fallback `results` with warning.
4. Hard error.

All resolved paths are normalized to absolute paths.

## Run-Scoped Artifacts

Outputs are run-scoped:

```text
artifacts/
  runs/
    <run_id>/
      results/
      models/
      logs/
      cache/
```

Convenience pointer:

```text
artifacts/latest -> artifacts/runs/<run_id>
```

No overwrite by default. Use `--allow-overwrite-run` to reuse an existing `run_id` directory.

## Repository Layout

```text
.
├── src/healingstone/
├── scripts/
├── configs/
├── tests/
├── data/
│   ├── raw/
│   │   └── v1/
│   ├── interim/
│   └── processed/
├── artifacts/
│   └── runs/
├── pyproject.toml
├── requirements.lock
└── README.md
```

## Metadata and Schema

Each run writes:

- `run_metadata.json` (run id, commit, config hash/version, dataset alias/path, seed/device, dependency versions)
- `resolved_paths.json`
- `alignment_metrics.json` (validated against required metrics schema)

Metrics report includes `metrics_schema_version`.

## Accuracy Requirement

Hard gate policy:

- `pairwise_match_accuracy >= min_required_accuracy`
- default `min_required_accuracy: 0.80`
- enforcement requires `evaluation_split: test`

Gate is evaluated after final metrics computation and fails the run if unmet.

## Config Versioning

`config_version` is required in both pipeline and train configs.
Unsupported versions fail fast.

## Tests and Quality Gates

```bash
ruff check .
mypy
pytest
```

CI runs `ruff`, `mypy`, and `pytest` on Python `3.10`, `3.11`, `3.12`.

## Regenerate Lock File

```bash
pip-compile pyproject.toml --extra dev -o requirements.lock
```

## Migration Window

Legacy wrappers and legacy fallback paths are still supported in this release.
Planned next step is removal after migration stabilization.
