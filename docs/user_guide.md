# Developer and User Guide: Healingstone

## 1. Quick Start and Installation
The project is optimized for Python `3.10`–`3.12`. For full 3D execution, `open3d` and `torch` are required.

```bash
# Set up environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,runtime]'

# Initialize configuration
cp .env.example .env

# Run the pipeline
healingstone-run --data-dir data/raw/3d --output-dir artifacts
```

---

## 2. Usage Instructions
`healingstone-run` is the primary entry point. It automatically configures paths and detects input types.

- **Option 1**: Use dataset aliases in `configs/datasets.yaml`.
- **Option 2**: Provide explicit directory paths.
- **Option 3**: Configure via environment variables (`HEALINGSTONE_OUTPUT_DIR`, etc.).

---

## 3. Contributing Guidelines
We follow a modular, package-first approach.

### Code Style
- **Naming**: `snake_case` for variables/functions, `CamelCase` for classes.
- **Formatting**: We use `ruff` for linting and formatting.
- **Typing**: `mypy` for static type checking.

### Pull Requests
Before submitting a PR, ensure all checks pass:
```bash
pytest -q
ruff check .
mypy
```

---

## 4. Developer Takeover & Architecture
If you are taking over the repository:

### Core Layers
- `src/healingstone/api`: CLI entrypoints.
- `src/healingstone/core`: Path policies, metrics schema, and runtime config.
- `src/healingstone/services`: Process boundaries and orchestration.
- `src/healingstone/pipeline`: End-to-end 3D and 2D execution.
- `src/healingstone/alignment`: 3D-specific geometric matching and registration.

### Runtime Paths
All relative paths must resolve from the project root. The `healingstone.core.runtime_paths` module enforces this policy to ensure environment-agnostic execution.

### Artifact Policy
Outputs are isolated under `artifacts/runs/<run_id>/` to prevent accidental overwrites. The `artifacts/latest` pointer provides quick access to the most recent execution.
