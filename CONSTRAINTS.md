# Constraints

## Runtime Constraints

- full 3D runtime support: Python `3.10` to `3.12`
- Python `3.13` is acceptable for light checks only
- `open3d` and `torch` are optional dependencies but required for full 3D execution

## Data Constraints

- 3D fragments must be `.ply` or `.obj`
- 2D fragments must be image files supported by the 2D pipeline
- local canonical 3D dataset root is `data/raw/3d`
- no canonical in-repo 2D dataset root exists

## Compute Constraints

- expected hardware:
  - CPU-only operation must remain possible
  - optional single GPU acceleration
  - memory budget typically `16` to `32` GB RAM
- large meshes require decimation and point sampling

## Reproducibility Constraints

- config precedence must remain `CLI > ENV > YAML`
- paths must resolve relative to project root
- artifacts must be isolated under `artifacts/runs/<run_id>/`
- metrics must satisfy schema version `1`
- accuracy gate applies only when `evaluation_split=test`

## Project Constraints

- package-first entrypoint is canonical
- generated artifacts are not source of truth
- submission requirements still target GSoC Healing Stones deliverables
- mentors must not be contacted directly

## Current Hard Blockers

- no labeled-pair dataset in repo
- full 3D validation blocked in current `.venv` due missing `open3d`
