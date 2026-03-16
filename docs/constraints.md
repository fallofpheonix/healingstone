# Constraints: Healing Stones

## Technical Constraints

### Programming Language

Implementation must use Python.

Primary libraries in the packaged project:
- Open3D
- NumPy
- PyTorch
- scikit-learn
- matplotlib
- networkx
- pydantic
- PyYAML

Notes:
- Open3D and Torch are currently optional runtime dependencies
- documented full runtime support is Python `3.10` to `3.12`
- Python `3.13` is suitable for lightweight checks, but full Open3D and Torch runtime should not be assumed there

### Input Data Constraints

Pipeline must process `.PLY` or `.OBJ` files. Meshes may contain holes, noise, irregular topology, and missing geometry.

Repository reality:
- `healingstone/3D/` contains local `.PLY` fragments
- `healingstone/2D/` contains local `.png` references
- `data/3D_fragments.zip` is invalid as distributed: it is a truncated tar archive mislabeled as zip

### Automation Requirement

Pipeline must execute end-to-end without manual interaction.

Canonical execution:

```bash
cd healingstone
python -m healingstone.run_pipeline
```

Required packaged stages:
- load data
- preprocess meshes
- collect pseudo-labels
- train or apply point-wise surface classifier
- extract cached features
- train Siamese matcher
- generate candidate pairs
- align fragments
- assemble reconstruction
- generate metrics and artifacts

### Dataset Limitations

Available local 3D sample set in this repo contains 17 `.PLY` fragments under `healingstone/3D/`.
The external evaluation description references 12 rotated sections at uniform scale.

Algorithms must tolerate:
- incomplete surface overlap
- irregular fracture boundaries
- missing topology
- scan noise

### Computational Constraints

Typical runtime hardware:
- 1 GPU optional
- 16 to 32 GB RAM

Large meshes require decimation and point sampling to prevent memory overflow.

### Reproducibility Constraints

- config precedence must remain `CLI > ENV > YAML`
- run artifacts must be isolated under `artifacts/runs/<run_id>/`
- metrics must satisfy the schema in `core/metrics_schema.py`
- acceptance gating for `pairwise_match_accuracy` only applies on `evaluation_split=test`

### General Project Constraints

Applicants must follow submission rules:
- email: `human-ai@cern.ch`
- subject: `GSoC Healing Stones`
- mentors must not be contacted directly
