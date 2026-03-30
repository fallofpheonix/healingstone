# Data and Experiments: Healingstone

## 1. Data Schema and Types
The system operates on three primary data types:

- **3D Meshes**: `.PLY` (canonical), `.OBJ` (supported).
- **2D Images**: `.PNG`, `.JPG`, `.TIF`.
- **Metrics**: JSON payloads satisfying version `1` of the metrics schema.

### Canonical Path Policies
- **Input**: `data/raw/3d/` (fragment dataset).
- **Interim**: `data/interim/` (preprocessed point clouds).
- **Processed**: `data/processed/` (labeled samples, if any).
- **Output**: `artifacts/runs/<run_id>/` (isolated per run).

---

## 2. Dataset Integrity Status
- **Verification Summary**: 
  - Canonical 3D run (17 fragments) verified for zero data loss.
  - Fragment meshes validated for normal estimation and point density.
  - No data overlaps or corruption detected in `data/raw/3d`.

---

## 3. Experiment Log
Tracking significant baseline results and ablation studies.

| ID | Configuration | Result | Note |
| --- | --- | --- | --- |
| `base_3d_01` | Default 3D pipeline | Success | Verified baseline reconstruction (71.5% completeness). |
| `noise_robustness_01` | Gaussian noise injection | Success | Pipeline maintains 65%+ completeness with 1mm noise. |
| `downsampling_test_01` | Adaptive Voxel vs. Uniform | - | Ongoing testing. |

---

## 4. Evaluation and Metrics Portfolio
All reconstruction results are quantitatively evaluated using:
- **Registration RMSE**: Root Mean Square Error of aligned point clouds.
- **Chamfer Distance**: Bi-directional average of closest-point distances.
- **Match Accuracy**: Ratio of correctly predicted fragment pairs (Target ≥ 80% with labels).
- **Reconstruction Completeness**: Proportion of fragments successfully assembled.
