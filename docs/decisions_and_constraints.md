# Decisions and Constraints: Healingstone

## 1. Key Architectural Decisions
The following documented decisions guide the development of the system:

### Repository Layout
- **Decision**: Keep run artifacts isolated under `artifacts/runs/<run_id/>`.
- **Rationale**: Prevent accidental overwrite and ensure clear data lineage.

### Path Resolution
- **Decision**: Resolve relative runtime paths from the project root.
- **Rationale**: Ensures the code works regardless of the caller's working directory.

### Compatibility Entrypoint
- **Decision**: Preserve a CLI compatible entrypoint (`healingstone.run_pipeline`).
- **Rationale**: Support legacy automation without a breaking change.

### Metrics Schema
- **Decision**: Use strict pydantic-based schema checks on metrics outputs.
- **Rationale**: Maintain high data quality and schema-validated deliverables.

---

## 2. System Constraints

### Runtime Constraints
- **Python Support**: Full 3D runtime requires Python `3.10`–`3.12`. Python `3.13` is acceptable only for light checks.
- **Dependencies**: `open3d` and `torch` are optional but required for 3D reconstruction.

### Data Constraints
- **Formats**: 3D meshes must be `.ply` or `.obj`.
- **Roots**: Local canonical data root is `data/raw/3d`.

### Compute Constraints
- **RAM**: Typically `16` to `32` GB RAM for large meshes.
- **CPU**: CPU-only operation must remain possible.
- **GPU**: Optional single GPU acceleration.

### Reproducibility Constraints
- **Precedence**: `CLI > ENV > YAML`.
- **Schemas**: Metrics must satisfy version `1` of the metrics schema.
- **Gates**: `pairwise_match_accuracy` gate applies only when `evaluation_split=test`.

---

## 3. Project Constraints
- **Deliverables**: Must target GSoC Healing Stones deliverables.
- **Ingestion**: GPT context pack ingestion is limited to 11 canonical files via `CONTEXT_INDEX.md`.
