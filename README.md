# Healingstone

Production-oriented reconstruction pipeline for fragmented artifacts. It supports:
- 3D mesh fragments (`.ply`, `.obj`)
- 2D image fragments (`.png`, `.jpg`, `.tif`)

The runtime detects input type and routes to the appropriate pipeline.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
cp .env.example .env
healingstone-run --help
```

Run with explicit inputs:

```bash
healingstone-run --data-dir 3D --output-dir artifacts
```

## Architecture

The codebase uses a small layered design:

- `src/healingstone/api`: command entrypoints and process boundaries
- `src/healingstone/services`: orchestration and runtime handoff
- `src/healingstone/core`: domain config, schema, and path policies
- `src/healingstone/pipeline`: end-to-end 2D/3D execution
- `src/healingstone/config`: environment-to-runtime adaptation
- `src/healingstone/utils`: shared operational helpers

## Configuration

Resolution order is intentional:

```text
CLI > ENV > YAML
```

- YAML: `configs/pipeline.yaml`, `configs/train.yaml`, `configs/datasets.yaml`
- ENV prefix: `HEALINGSTONE_`
- Example: `HEALINGSTONE_OUTPUT_DIR=/tmp/artifacts healingstone-run`

## Testing

Run fast quality checks:

```bash
pytest -q
ruff check .
mypy
```

## Key Decisions

- Keep run artifacts isolated under `artifacts/runs/<run_id>` to avoid accidental overwrite.
- Preserve a compatibility entrypoint (`healingstone.run_pipeline`) for existing automation.
- Keep strict schema checks on metrics output even in minimal runs.
