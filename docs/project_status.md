# Project Status: Healing Stones

## 1. Current State Summary (Pre-GSoC)
The project has established a foundational architectural baseline. A minimal functional prototype exists that handles basic 3D reconstruction tasks.

### Pre-GSoC Accomplishments
- **Ingestion**: Functional loading of `.PLY` and `.OBJ` files via Open3D.
- **Preprocessing**: Implementation of point cloud loading, downsampling, and normal estimation.
- **Feature Extraction**: Preliminary feature extraction using FPFH descriptors.
- **Alignment**: Basic pairwise alignment using ICP.
- **Workability**: CLI entry points established; test suite and lint/type checks (pytest, ruff, mypy) passing.

---

## 2. Detailed Project Timeline

### Community Bonding Period
- **Dataset Analysis**: Understand fragment size distribution and noise levels.
- **Strategy Finalization**: Finalize fracture surface detection (Heuristics vs. ML) and assembly logic (MST vs. Pose Graph).
- **Evaluation Setup**: Define metrics (Chamfer, RMSE) and prepare a small annotated ground truth dataset.

### Phase 1: Ingestion & Preprocessing (Weeks 1–2)
- [ ] Support batch processing and heterogeneous scan properties.
- [ ] Standardize normalization (scale, denoising, sampling).

### Phase 2: Surface Classification (Weeks 3–4)
- [ ] Implement curvature and normal variance-based break surface detection.
- [ ] **ML Extension**: Pseudo-label fracture regions and train a PointNet++ classifier.

### Phase 3: Feature Extraction & Matching (Weeks 5–6)
- [ ] Implement learned embeddings (PointNet/DGCNN) to improve robustness.
- [ ] Develop similarity-based candidate matching using nearest-neighbor search.

### Phase 4: Alignment & Pose Estimation (Weeks 7–8)
- [ ] Robust pairwise alignment under noise and partial overlap using RANSAC.
- [ ] Refinement metrics for SE(3) estimation quality.

### Phase 5: Global Assembly (Weeks 9–10)
- [ ] Build the fragment graph (Nodes = Fragments, Edges = Validated Matches).
- [ ] Implement Pose Graph Optimization to resolve global consistency and conflicts.

### Phase 6: Final Evaluation & Refinement (Weeks 11–12)
- [ ] Full pipeline validation on annotated datasets.
- [ ] System documentation, refactoring, and final submission-ready bundle.

---

## 3. Known Issues & Blockers
- **Open3D-Torch Coupling**: Learned features require specific PyTorch-Open3D interaction environments.
- **Missing Global Assembly**: Current assembly is primarily pairwise; global consistency logic is the current focus.
- **Erosion Modeling**: Handling large gaps in severely eroded fragments remains a technical challenge.
