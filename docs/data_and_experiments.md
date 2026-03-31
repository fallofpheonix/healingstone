# Data and Evaluation: Healing Stones

## 1. Data Schema and Types
The system operates on three primary data types:
- **3D Meshes**: `.PLY` (canonical), `.OBJ` (supported).
- **2D Images**: `.PNG`, `.JPG`, `.TIF`.
- **Metrics**: Versioned JSON payloads for tracking pipeline performance.

### Canonical Path Policies
- **Raw Input**: `data/raw/3d/` (fragment datasets).
- **Processing Artifacts**: `artifacts/runs/<run_id>/` (isolated per run).
- **Caching**: `data/cache/` (reusable feature payloads).

---

## 2. Dataset Integrity & Ground Truth
- **Verification**: All fragments are scanned for mesh integrity, normal consistency, and point density.
- **Ground Truth**: Evaluation requires annotated fragment correspondences and reference alignments (if available) to verify pipeline accuracy.

---

## 3. Evaluation Framework
The system is evaluated across four key dimensions:

### 1. Matching Performance
- **Matching Accuracy**: Correctly identified fragment matches compared to ground truth pairs.
- **Precision**: $\frac{\text{correct matches}}{\text{predicted matches}}$
- **Recall**: $\frac{\text{correct matches}}{\text{true matches}}$
- **F1-Score**: Harmonic mean of Precision and Recall.

### 2. Alignment Quality
- **RMSE (Root Mean Square Error)**: Average point-to-point distance between aligned fragments.
- **Chamfer Distance**: Bi-directional average of closest-neighbor distances.
- **Normal Consistency**: Agreement between surface normals of aligned fracture patches.

### 3. Reconstruction Completeness
- **Completeness**: Fraction of the original artifact successfully reconstructed from the input fragments.
- **Global Consistency**: Degree to which the full assembly maintains coherent spatial relationships without transformation conflicts or physical overlaps.

### 4. System Reliability
- Performance under real-world noise (sensor error, erosion gaps, and partial scans).
- Scaling efficiency relative to the number of fragments ($O(n^2)$ filtering).

---

## 4. Experiment Log (Key Milestones)

| ID | Configuration | Success | Note |
| --- | --- | --- | --- |
| `base_3d_v1` | Default 3D FPFH pipeline | Success | Verified baseline reconstruction (71.5% completeness). |
| `noise_test_v1` | 1mm Gaussian noise | Success | Robustness confirmed; completeness maintained at 65%+. |
| `patch_match_v1` | Patch-level fragment matching | - | Ongoing testing on eroded datasets. |
