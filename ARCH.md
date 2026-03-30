# Architecture

## Runtime Call Path

```text
CLI
  -> healingstone.api.cli.main
  -> healingstone.services.reconstruction_service.execute_reconstruction
  -> env defaults
  -> config merge and validation
  -> project-root-relative path resolution
  -> mode detection
  -> 2D or 3D reconstruction pipeline
  -> run-scoped artifacts
```

## Module Graph

```mermaid
graph TD
    CLI["api/cli.py"] --> SERVICE["services/reconstruction_service.py"]
    SERVICE --> ENV["config/environment.py"]
    SERVICE --> CFG["core/runtime_config.py"]
    SERVICE --> PIPE["pipeline/run_pipeline.py"]

    PIPE --> PATHS["core/runtime_paths.py"]
    PIPE --> METRICS["core/metrics_schema.py"]
    PIPE --> VIZ["utils/visualization.py"]

    PIPE --> PRE3D["core/preprocess.py"]
    PIPE --> FEAT["core/features.py"]
    PIPE --> MATCH3D["ml_models/match_fragments.py"]
    PIPE --> ALIGN3D["alignment/align_fragments.py"]
    PIPE --> RECON3D["alignment/reconstruct.py"]

    PIPE --> PRE2D["healingstone2d/preprocess_2d.py"]
    PIPE --> MATCH2D["healingstone2d/match_fragments_2d.py"]
    PIPE --> ALIGN2D["healingstone2d/align_fragments_2d.py"]
    PIPE --> RECON2D["healingstone2d/reconstruct_2d.py"]
```

## Layers

| Layer | Responsibility | Key Files |
| --- | --- | --- |
| CLI | parse arguments and enter service boundary | `src/healingstone/api/cli.py` |
| Service | merge env/config/runtime and invoke pipeline | `src/healingstone/services/reconstruction_service.py` |
| Core policy | config, paths, metrics schema, deterministic runtime contracts | `src/healingstone/core/*.py` |
| 3D engine | preprocess, feature extraction, matching, alignment, reconstruction | `core/preprocess.py`, `core/features.py`, `ml_models/match_fragments.py`, `alignment/*.py` |
| 2D engine | preprocess, descriptor matching, alignment, canvas reconstruction | `src/healingstone/healingstone2d/*.py` |
| Utilities | plotting and runtime fingerprints | `src/healingstone/utils/*.py` |

## 3D Data Flow

```text
mesh fragments
  -> Open3D load
  -> denoise + downsample + normals
  -> feature extraction
  -> Siamese pair scoring
  -> candidate selection
  -> RANSAC + ICP alignment
  -> graph assembly
  -> merged point cloud + metrics + plots
```

## 2D Data Flow

```text
image fragments
  -> preprocessing + edge extraction
  -> shape descriptors
  -> candidate matching
  -> rigid 2D alignment
  -> canvas assembly
  -> reconstructed image + minimal report
```

## External Dependencies

| Dependency | Role | Scope |
| --- | --- | --- |
| `numpy` | numerical operations | all modes |
| `matplotlib` | diagnostic plots | all modes |
| `PyYAML` + `pydantic` | config loading and validation | all modes |
| `networkx` | graph assembly | all modes |
| `open3d` | mesh IO, geometry, ICP, FPFH | 3D only |
| `torch` | Siamese and classifier training | 3D only |
| `opencv-python` | 2D preprocessing and rendering | 2D only |
