# GSoC 2026 Proposal: HealingStones
## *Automated 3D Archaeological Fragment Reconstruction via PointNet++ and Pose Graph Optimization*

---

## 1. Contact Information

| Field | Value |
|-------|-------|
| **Student Name** | Ujjwal Singh |
| **Email** | ujjosing@gmail.com |
| **GitHub** | github.com/fallofpheonix |
| **Time Zone** | IST (UTC+5:30) |

---

## 2. Synopsis

Reconstructing broken archaeological artifacts from 3D scans requires solving a complex geometry problem: classify break surfaces, match fragment pairs by geometry, and align them in 3D space — all without human interaction. This project delivers a fully automated pipeline executing `python -m healingstone.run_pipeline` end-to-end: starting from raw `.PLY`/`.OBJ` mesh files and producing aligned fragment assemblies with quantitative accuracy metrics.

---

## 3. Benefits to Community

- Reduces manual reconstruction effort at CERN and partner archaeological institutions.
- The automated pipeline is generalizable to any 3D fragment dataset.
- Contributes a reusable Open3D + PointNet++ toolkit for computer vision researchers.

---

## 4. Technical Approach

### Pipeline

```
.PLY / .OBJ Fragment Meshes
        ↓
Open3D Preprocessing (normal estimation, decimation)
        ↓
PointNet++ Break Surface Classifier
  Input: N×3 point cloud → Output: N×2 label (break / non-break)
        ↓
FPFH Descriptor Extraction on break regions
        ↓
Cosine similarity → candidate fragment pairs
        ↓
RANSAC + ICP pose estimation → T = [R | t] ∈ SE(3)
        ↓
Open3D Pose Graph Optimization → global assembly
        ↓
Evaluation: RMSE, match accuracy, overlap ratio
```

### Break Surface Classifier

```python
# PointNet++ applied to break region point clouds
# Loss: CrossEntropyLoss (break vs. non-break)
```

### Evaluation Metrics

| Metric | Formula |
|--------|---------|
| Match accuracy | `correct_matches / total_matches` (target ≥ 80%) |
| Alignment RMSE | `√(Σ ‖pᵢ - qᵢ‖² / N)` |
| Overlap ratio | `matched_points / total_points` |

---

## 5. Deliverables

| # | Deliverable | Required/Optional |
|---|------------|-------------------|
| 1 | Mesh preprocessing module | Required |
| 2 | PointNet++ break surface classifier | Required |
| 3 | FPFH feature extraction + fragment matching | Required |
| 4 | RANSAC + ICP alignment | Required |
| 5 | Pose graph global reconstruction | Required |
| 6 | Evaluation notebook with metrics | Required |
| 7 | `run_pipeline` end-to-end entrypoint | Required |

---

## 6. Timeline (175 hours)

| Period | Activity |
|--------|----------|
| **Pre-bonding** | Study Open3D API, PointNet++ architecture |
| **Weeks 1–2** | Mesh loading, preprocessing, normal estimation |
| **Weeks 3–5** | Train PointNet++ break surface classifier |
| **Weeks 6–7** | FPFH extraction, fragment matching module |
| **Weeks 8–9** | RANSAC + ICP + pose graph optimization |
| **Week 10** | End-to-end `run_pipeline` integration |
| **Weeks 11–12** | Evaluation notebook, documentation, PR |

---

## 7. Related Work

- **PointNet++** (Qi et al., 2017): hierarchical point cloud learning — our classifier backbone.
- **FPFH** (Rusu et al., 2009): fast point feature histograms for geometric matching.
- **Open3D** (Zhou et al., 2018): used for pose graph optimization.

---

## 8. About Me

**Ujjwal Singh** | ujjosing@gmail.com | [GitHub](https://github.com) | [LinkedIn](https://linkedin.com) | VIT University, B.Tech CS (2023–2027) | IST (UTC+5:30)

- **3D/Spatial Experience:** Built *UDIE* iOS app with MapKit spatial heatmap visualization — foundational experience in spatial data structures. Actively studying Open3D and point cloud processing in preparation for this project.
- **Deep Learning:** CNN development via *TerraHerb* (TensorFlow/Keras) — transferable to training the PointNet++ break surface classifier.
- **Scientific Computing:** Python, NumPy, SciPy — used extensively for data pipeline work.
- **Technical Skills:** Python, TensorFlow, Keras, NumPy, Pandas, C++, Git, Java.
- **Certifications:** AI and Machine Learning Fundamentals.
