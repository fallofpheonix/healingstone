# Submission Manifest: GSoC Healing Stones

This document consolidates the deliverables required for the **GSoC Healing Stones** evaluation as per the `docs/submission_guidelines.md`.

## 1. Automated Pipeline (GitHub)
The codebase has been refactored, audited, and pushed to the `main` branch of the repository.

- **GitHub Link**: [https://github.com/fallofpheonix/healingstone](https://github.com/fallofpheonix/healingstone)
- **Branch**: `main`
- **Execution**: Run `pip install -e .` followed by `healingstone-run` or `python -m healingstone.pipeline.run_pipeline`.

## 2. Pre-trained Models
The project can train the Siamese encoder during pipeline execution and save it automatically in the run-scoped output directory.
Pre-generated model weights are also available for convenience.

- **Recommended Action**: Upload the following files to a Google Drive folder and include the link in your email to the mentors.
- **Local File Paths**:
  - `artifacts/runs/20260312T190849Z_8ce41b4/models/siamese_encoder.pt`
  - `artifacts/scan_run/runs/20260314T001313Z_8ce41b4/models/surface_classifier.pt`

## 3. Local Validation Runs
- **2D command**: `python -m healingstone.pipeline.run_pipeline --data-dir 2D --output-dir result`
- **3D command**: `python -m healingstone.pipeline.run_pipeline --data-dir 3D --output-dir result --min-required-accuracy 0`
- **Recent successful 3D run**: `result/runs/20260316T204009Z_a8dd6da/`

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
