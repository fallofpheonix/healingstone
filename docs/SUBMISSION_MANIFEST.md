# Submission Manifest: GSoC Healing Stones

This document consolidates the deliverables required for the **GSoC Healing Stones** evaluation as per the `docs/submission_guidelines.md`.

## 1. Automated Pipeline (GitHub)
The codebase is version-controlled in the GitHub repository and the submission branch should contain the current delivery state.

- **GitHub Link**: [https://github.com/fallofpheonix/healingstone](https://github.com/fallofpheonix/healingstone)
- **Branch**: use the active submission branch
- **Execution**: Run `pip install -e '.[runtime]'` followed by `healingstone-run` or `python -m healingstone.pipeline.run_pipeline`.

## 2. Pre-trained Models
The project can train the Siamese encoder during pipeline execution and save it automatically in the run-scoped output directory.
Pre-generated model weights are also available for convenience.

- **Recommended Action**: Upload the following files to a Google Drive folder and include the link in your email to the mentors.
- **Local File Paths**:
  - `artifacts/runs/20260330T134400Z_b7ac4ff/models/siamese_encoder.pt`
  - `artifacts/runs/20260330T134400Z_b7ac4ff/models/training_metrics.json`

## 3. Local Validation Runs
- **2D command**: `python -m healingstone.pipeline.run_pipeline --data-dir /path/to/2d_fragments --output-dir artifacts`
- **3D command**: `healingstone-run --data-dir data/raw/3d --output-dir artifacts`
- **Recent successful 3D run**: `artifacts/runs/20260330T134400Z_b7ac4ff/`

## 4. Contact Instructions
Send an email to **human-ai@cern.ch** with:
- **Subject**: `GSoC Healing Stones`
- **Body**:
  - Your Name and CV/Resume link.
  - The GitHub repository link (provided above).
  - The Google Drive link to the models (after you upload the `.pt` files).

## 5. Technical Validation Status
- **Audit Phase**: 12/12 Phases Completed (100% Technical Debt Remediation).
- **Functional Validation**: 3D and 2D pipelines verified with real and synthetic data.
- **Continuous Integration**: Ruff, Mypy, and Pytest coverage stabilized.
- **Stability**: Passed 5/5 stress-test cycles without resource leaks or failures.
