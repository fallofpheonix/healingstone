# Tasks

## TODO

- create a Python `3.12` runtime environment and install `pip install -e '.[dev,runtime]'`
- run `healingstone-run --data-dir data/raw/3d --output-dir artifacts --min-required-accuracy 0`
- capture the new run ID and refresh `STATE.md` and `EXPERIMENTS.md`
- generate and curate `labeling_candidates.csv` into a real `labels.csv`
- calibrate `min_match_accuracy` and `min_required_accuracy` on labeled data
- define acceptance thresholds for completeness and Chamfer distance

## IN PROGRESS

- none

## BLOCKED

- full 3D runtime verification in current `.venv`
  - reason: `open3d` missing
- supervised pairwise accuracy measurement
  - reason: no labeled pair CSV in repo

## DONE

- consolidate `3D/` into `data/raw/3d/`
- consolidate root outputs into `artifacts/`
- remove legacy path fallbacks from runtime path policy
- isolate plotting into `utils/visualization.py`
- isolate 3D metrics summarization into `core/metrics_collector.py`
- normalize docs and tests so default `pytest -q` works in the default environment
- push verified restructure branch to GitHub
