# Healingstone

Fragment reconstruction pipeline for 3D mesh fragments and 2D image fragments.

## Runtime

- Canonical CLI: `healingstone-run`
- Python: `3.10` to `3.12` for full 3D runtime (`open3d`, `torch`)
- Resolution order: `CLI > ENV > YAML`

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,runtime]'
```

## Smoke Run

Included sample dataset:

- `data/sample/3d/fragment_a.ply`
- `data/sample/3d/fragment_b.ply`

Run:

```bash
healingstone-run --data-dir data/sample/3d --output-dir artifacts --min-required-accuracy 0 --allow-overwrite-run
```

2D inputs are auto-detected from image extensions and routed through the 2D pipeline.

## Output Contract

Each run writes a scoped artifact directory:

```text
artifacts/runs/<run_id>/
├── metrics.json
├── reconstruction.ply        # 3D runs
├── logs.json
├── run_metadata.json
├── resolved_paths.json
├── logs/
│   └── pipeline.log
└── results/
    ├── alignment_metrics.json
    ├── metrics.json
    ├── reconstructed_model.ply
    ├── similarity_matrix.png
    ├── alignment_pair_*.png
    └── final_reconstruction.png
```

`metrics.json` is the compact contract surface. `results/alignment_metrics.json` is the detailed report.

## Quality Gates

- `pytest -q`
- `ruff check .`
- `mypy`
- sample CLI smoke on `data/sample/3d`

The matcher falls back to deterministic descriptor cosine scoring when the sample dataset is too small to train the Siamese model safely.

## Repository Layout

```text
configs/                  Runtime configuration
data/sample/3d/           Included 3D smoke dataset
data/raw/3d/              Local large-fragment dataset root
src/healingstone/api/     CLI entrypoint
src/healingstone/services/Runtime orchestration
src/healingstone/core/    Geometry, metrics, config, path policy
src/healingstone/pipeline/End-to-end execution
tests/                    Unit and integration tests
```
