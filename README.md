# Healing Stones 3D Reconstruction Pipeline

Robust end-to-end reconstruction pipeline for fragmented 3D cultural heritage artifacts (.PLY/.OBJ).

## Single-command Run

```bash
python run_pipeline.py --data-dir DataSet/3D --output-dir results --augment-rotations
```

If your fragments are in a different folder:

```bash
python run_pipeline.py --data-dir /path/to/fragments --output-dir results
```

## Installation

Python 3.10-3.12 recommended (Open3D wheels are not available on some 3.13 builds).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline Stages

1. Load and validate `.PLY/.OBJ` fragments.
2. Preprocess point clouds:
   - denoise
   - voxel downsample
   - normalize scale/center
   - estimate consistent normals
3. Detect break surfaces using:
   - curvature
   - normal variance
   - roughness
   - DBSCAN clustering
4. Extract descriptors:
   - FPFH
   - break-surface geometric statistics
5. Train Siamese embedding model with contrastive loss using:
   - random rotations
   - Gaussian noise
   - partial surface removal
6. Build similarity matrix and candidate fragment pairs.
7. Align candidate pairs with:
   - RANSAC (FPFH)
   - point-to-plane ICP refinement
8. Build fragment graph and compute global assembly (maximum spanning tree).
9. Save reconstructed model and automatic visualizations/metrics.

## Output Artifacts

Saved under `--output-dir`:

- `reconstructed_model.ply`
- `alignment_metrics.json`
- `similarity_matrix.png`
- `final_reconstruction.png`
- `alignment_pair_*.png`
- `models/training_loss.png`
- `pipeline.log`
- `cache/*.npz` feature cache

## Optional Labels

For pairwise match accuracy reporting, provide labeled pairs CSV:

```csv
frag_a,frag_b,label
fragment_01,fragment_02,1
fragment_01,fragment_05,0
```

Run:

```bash
python run_pipeline.py --data-dir DataSet/3D --labels-csv labels.csv
```

Accuracy gate:

- `--min-match-accuracy` defaults to `0.80`.
- If no labeled pairs are provided, the pipeline stops with an explicit error.
- Use generated `results/labeling_candidates.csv` for fast annotation rounds.

Temporary bypass (debug only):

```bash
python run_pipeline.py --data-dir DataSet/3D --min-match-accuracy 0
```

## Notes

- Deterministic seeds are set for `random`, `numpy`, and `torch`.
- Without labels, match accuracy is reported as `NaN` and the pipeline remains fully automatic.
- The pipeline is designed to be reusable on other fragmented datasets with identical file formats.
