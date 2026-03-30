# API And Interface Contracts

## CLI Surface

| Command | Contract |
| --- | --- |
| `healingstone-run` | primary packaged entrypoint |
| `python -m healingstone.pipeline.run_pipeline` | direct module entrypoint |
| `python -m healingstone.run_pipeline` | compatibility entrypoint |

## Core Python Entry Points

| Symbol | Signature | Side Effects |
| --- | --- | --- |
| `healingstone.api.cli.main` | `() -> None` | parses CLI args and executes reconstruction |
| `execute_reconstruction` | `(cli_args: Namespace) -> None` | merges env/config and dispatches pipeline |
| `build_runtime_config` | `(cli_args: Namespace) -> RuntimeConfigBundle` | validates YAML + ENV + CLI config |
| `to_namespace` | `(bundle: RuntimeConfigBundle) -> Namespace` | expands config bundle for runtime use |
| `resolve_data_dir` | `(configured_data_dir, data_dir_source, dataset_alias, aliases) -> Path` | validates fragment presence |
| `resolve_artifact_root` | `(configured_output_dir, output_dir_source) -> Path` | validates writable artifact root |
| `initialize_run_layout` | `(data_dir, labels_csv, artifact_root, allow_overwrite_run, run_id=None) -> ResolvedRunPaths` | creates run directories and latest pointer |
| `detect_pipeline_mode` | `(data_dir: Path) -> "3d" | "2d"` | inspects file suffixes recursively |
| `run_pipeline` | `(args: Namespace) -> None` | executes full 2D or 3D pipeline |

## Config Precedence Contract

```text
CLI > ENV > YAML
```

- ENV prefix: `HEALINGSTONE_`
- pipeline config: `configs/pipeline.yaml`
- train config: `configs/train.yaml`
- dataset aliases: `configs/datasets.yaml`

## Important CLI Arguments

| Group | Arguments |
| --- | --- |
| config | `--config`, `--train-config`, `--dataset-manifest` |
| IO | `--data-dir`, `--output-dir`, `--labels-csv`, `--allow-overwrite-run` |
| geometry | `--sample-points`, `--voxel-size`, `--normal-radius`, `--normal-max-nn` |
| features | `--k-neighbors`, `--fpfh-radius`, `--fpfh-max-nn`, `--dbscan-eps`, `--n-keypoints` |
| matching | `--candidate-top-k`, `--align-top-n`, `--label-suggestions-top-n`, `--threshold-objective` |
| evaluation | `--min-match-accuracy`, `--min-required-accuracy`, `--evaluation-split` |
| runtime | `--seed`, `--device` |

## Report Contract

### 3D

- file: `artifacts/runs/<run_id>/results/alignment_metrics.json`
- required `metrics` keys:
  - `pairwise_match_accuracy`
  - `min_required_accuracy`
  - `evaluation_split`
  - `aligned_pairs`
  - `successful_alignments`
  - `mean_icp_rmse`
  - `mean_chamfer_distance`
  - `reconstruction_completeness`
  - `assembled_fragments`
  - `graph_nodes`
  - `graph_edges`

### 2D

- file: `artifacts/runs/<run_id>/results/alignment_metrics.json`
- contract:
  - `pipeline_mode = "2d"`
  - `config`
  - `run`
  - `metrics`

## Path Contract

- relative paths resolve from project root
- canonical dataset root: `data/raw/3d`
- canonical artifact root: `artifacts`
- run root: `artifacts/runs/<run_id>`
- latest pointer: `artifacts/latest`
