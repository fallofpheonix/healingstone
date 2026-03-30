# Data

## Input Data Types

| Mode | Location | File Types | Notes |
| --- | --- | --- | --- |
| 3D | `data/raw/3d` | `.ply`, `.obj` | canonical local sample set |
| 2D | explicit external directory | `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp` | no canonical in-repo root |

## In-Repo 3D Sample Reality

- count: `17` fragments
- naming pattern: `NAR_ST_43B_FR_*`
- example file: `NAR_ST_43B_FR_01_F_01_R_02.PLY`
- scale reality: large meshes, up to `11M+` vertices

## Optional Supervised Labels CSV

- path: user-provided via `--labels-csv`
- required columns:
  - `frag_a`
  - `frag_b`
  - `label`
- label semantics:
  - `1`: positive pair
  - `0`: negative pair

## Generated Labeling Candidates

- file: `artifacts/runs/<run_id>/results/labeling_candidates.csv`
- columns:
  - `frag_a`
  - `frag_b`
  - `label`
  - `score`
  - `source`

## Artifact Layout

```text
artifacts/
  latest
  runs/<run_id>/
    cache/
    logs/
    models/
    results/
```

## Important Generated Files

| File | Meaning |
| --- | --- |
| `run_metadata.json` | config provenance and runtime fingerprint |
| `resolved_paths.json` | concrete resolved data/output paths |
| `alignment_metrics.json` | primary machine-readable report |
| `reconstructed_model.ply` | merged 3D reconstruction |
| `reconstructed_2d.png` | rendered 2D reconstruction |
| `similarity_matrix.png` | pairwise score visualization |
| `final_reconstruction.png` | final 3D scatter diagnostic |

## Edge Cases

- mixed mesh and image inputs: 3D takes precedence
- empty `--data-dir`: hard failure
- missing labels CSV: allowed unless accuracy gate requires labels
- malformed labels CSV: hard failure on schema load
- sparse or tiny fragments: may reduce alignment completeness
