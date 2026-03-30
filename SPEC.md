# Specification

## System

- Name: `healingstone`
- Type: package-first reconstruction pipeline
- Primary mode: 3D fragmented mesh reconstruction
- Secondary mode: 2D fragmented image reconstruction
- Invocation: `healingstone-run` or `python -m healingstone.pipeline.run_pipeline`

## Problem Definition

- Input: unordered fragment set
- Required inference:
  - detect break surfaces or equivalent fragment boundaries
  - score candidate fragment pairs
  - estimate pairwise alignments
  - assemble a global reconstruction
- Output: machine-readable reconstruction artifacts, diagnostics, and reproducibility metadata

## Inputs

| Input | Type | Required | Notes |
| --- | --- | --- | --- |
| `--data-dir` | directory | no | Defaults via dataset alias manifest |
| fragment files | `.ply` / `.obj` / image formats | yes | Mode auto-detected from files present |
| `--labels-csv` | CSV | no | Supervised pair labels with `frag_a,frag_b,label` |
| pipeline config | YAML | yes | `configs/pipeline.yaml` |
| train config | YAML | yes | `configs/train.yaml` |
| dataset manifest | YAML | yes | `configs/datasets.yaml` |

## Outputs

| Output | Path | Contract |
| --- | --- | --- |
| run root | `artifacts/runs/<run_id>/` | isolated per execution |
| results | `artifacts/runs/<run_id>/results/` | reports, plots, reconstructed outputs |
| models | `artifacts/runs/<run_id>/models/` | trained model weights |
| logs | `artifacts/runs/<run_id>/logs/` | pipeline log and error log |
| cache | `artifacts/runs/<run_id>/cache/` | reusable feature/cache payloads |
| latest pointer | `artifacts/latest` | symlink or text fallback |

## Success Conditions

- pipeline runs end-to-end from CLI without manual interaction
- paths resolve deterministically from project root
- outputs are run-scoped and non-destructive
- metrics report satisfies schema version `1`
- `pairwise_match_accuracy` gate is enforced only on `evaluation_split=test`
- default developer checks pass:
  - `pytest -q`
  - `ruff check .`
  - `mypy`

## Non-Goals

- GUI or manual reconstruction tooling
- archaeological interpretation
- guaranteed perfect reconstruction under severe erosion or missing geometry
- shipping raw datasets inside the GPT context pack
