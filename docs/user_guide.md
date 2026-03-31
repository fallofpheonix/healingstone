# Developer and User Guide: Healing Stones

## 1. Quick Start and Installation

The project is optimized for Python `3.10`–`3.12`. For full 3D execution, `open3d` and `torch` (gated extras) are required.

```bash
# Set up environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,runtime]'

# Run the pipeline
healingstone-run --data-dir data/raw/3d --output-dir artifacts
```

---

## 3. Usage Instructions
`healingstone-run` is the primary entry point. It automatically configures paths and detects input types.

- **Option 1**: Use dataset aliases in `configs/datasets.yaml`.
- **Option 2**: Provide explicit directory paths via CLI.
- **Option 3**: Configure via environment variables (`HEALINGSTONE_OUTPUT_DIR`, etc.).

---

## 4. Contributing Guidelines
We follow a modular, "package-first" approach.

### Code Style
- **Formatting**: We use `ruff` (linting/formatting) and `mypy` (static type checking).
- **Style**: Follow `PEP 8` standards.

### Pull Requests
Before submitting a PR, ensure all checks pass:
```bash
pytest -q
ruff check .
mypy
```

---

## 5. Developer Takeover & Architecture

### Core Layers
- `src/healingstone/api`: CLI entrypoints.
- `src/healingstone/core`: Path policies and runtime configuration.
- `src/healingstone/services`: Orchestration and processing logic.
- `src/healingstone/pipeline`: End-to-end 3D and 2D execution pipelines.
- `src/healingstone/alignment`: 3D match and registration algorithms.

### Path Responsibility
All relative paths must resolve from the project root. The `healingstone.core.runtime_paths` module enforces this to ensure environment-agnostic execution.

### Artifact Policy
Outputs are isolated under `artifacts/runs/<run_id>/` to prevent accidental overwrites. The `latest` pointer provides quick access to the most recent run results.
