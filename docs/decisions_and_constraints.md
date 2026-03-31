# Decisions and Constraints: Healing Stones

## 1. Key Architectural Decisions

- **Modular Reconstruction**: Deconstructing the pipeline into independent, testable stages (Ingestion $\rightarrow$ Preprocessing $\rightarrow$ Classification $\rightarrow$ Matching $\rightarrow$ Alignment $\rightarrow$ Assembly).
- **Patch-Level Matching**: Deciding to match fracture surface patches instead of whole fragments to improve robustness to size differences and partial overlap.
- **Cycle Consistency in Assembly**: Using graph-based optimization with consistency checks rather than simple greedy methods (e.g., MST) to ensure global coherence.
- **Path Resolution**: Enforcing root-relative paths via `healingstone.core.runtime_paths` to ensure environment-agnostic execution.

---

## 3. Risks and Mitigation Strategies

| Risk | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Weak Matching** | Low precision, failed assembly. | Combine geometric (FPFH) with learned embeddings (PointNet++); use patch-level matching. |
| **False Positives** | Globally inconsistent assemblies. | Apply strict confidence thresholds; use validation metrics (overlap, normal consistency). |
| **Alignment Instability** | ICP converging to local minima. | Use RANSAC-based coarse alignment; reject alignments with high residuals. |
| **Assembly Conflicts** | Conflicting fragment placements. | Enforce graph-level cycle consistency; remove conflicting edges during optimization. |
| **Complexity ($O(n^2)$)** | Poor scaling with fragment count. | Use embedding-based filtering and Top-k candidate selection to prune search space. |

---

## 4. System Constraints

### Runtime Constraints
- **Python Support**: Optimized for Python `3.10`–`3.12`.
- **Dependencies**: `open3d` and `torch` are required for full 3D features.
- **Compute**: Recommended 16+ GB RAM for dense mesh processing.

### Data Constraints
- **Formats**: 3D meshes must be `.ply` or `.obj`.
- **Root Policy**: Data must reside in `data/raw/3d/` or be explicitly pointed via CLI.
- **Reproducibility**: `CLI > ENV > YAML` precedence for all configuration overrides.
