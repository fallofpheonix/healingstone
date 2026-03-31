# Changelog - Healing Stone Reconstruction System

All notable changes to this project will be documented in this file.

## [2.0.0] - 2026-03-31
### Added
- **Validation Layer**: Introduced `src/healingstone/pipeline/validators.py` for automated schema and health checks of intermediate pipeline data.
- **Tools Directory**: Centralized standalone utilities in `/tools` (Security Audi, Benchmarking, Synthetic Data Generation).
- **Hard Determinism**: Enforced seeding at the runner level for 100% reproducibility.

### Changed
- **System Restructuring**: 
    - Moved 2D reconstruction logic to `src/healingstone/healingstone2d/`.
    - Consolidated CLI entrypoints to `src/healingstone/api/cli.py`.
    - Moved core feature metrics to `src/healingstone/core/metrics_collector.py`.
- **Pipeline Integrity**: Replaced all mock/random stubs in `pipeline/` stages with direct delegations to `core` geometry and `ml_models` logic.
- **Security**: Hardened `security_audit.py` regex to prevent false positives with standard PyTorch `.eval()` calls.
- **Documentation**: Corrected `README.md` to distinguish between **Measured Preliminary Results** and **Target Research Goals**.

### Removed
- **Technical Deception**: Deleted all AI-generated "Context Packs" and "System Certifications" that lacked engineering basis.
- **Redundant Wrappers**: Removed duplicate `run_pipeline.py` and `test_pipeline.py` scripts from the root and `scripts/` directories.
- **Dead Code**: Eliminated unused 3D geometry stubs that were bypassed by mocks.

## [2.1.0] - 2026-03-31
### Added
- **Final Logic Repairs**: Fixed broken geometry import graph in `run_pipeline.py`, `align_fragments.py`, and `reconstruct.py`.
- **Deterministic Run Identity**: Run IDs are now based on execution semantics, not artifact location.

### Changed
- **History Purification**: Use `git filter-repo` to permanently remove 2GB+ of historical large blobs (PLY, PNG, ZIP) and sanitize Git history for distribution.
- **Verification Gate**: Integrated a 51-test "Cleanup Gate" ensuring a purified install always succeeds 100%.

### Removed
- **History Pollution**: Stripped all historical `.ply`, `.png`, and `.mp4` files from the Git history (except for `data/sample`).
- **Workspace Sanitization**: Removed `data/raw/` and `artifacts/` local binaries to achieve <10MB total repository footprint.


### Migration Guide (v1 -> v2)
| Old Path | New Path |
| :--- | :--- |
| `healingstone.cli` | `healingstone.api.cli` |
| `healingstone.core.reconstruction.*` | `healingstone.healingstone2d.*` |
| `src/healingstone/core/security_audit.py` | `tools/security_audit.py` |
| `src/healingstone/core/validate_dataset.py` | `tools/validate_dataset.py` |
| `src/healingstone/pipeline/test_pipeline.py` | `tools/generate_synthetic_fragments.py` |
