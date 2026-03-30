# Roadmap

## Milestones

### M1: Full Runtime Validation

- build Python `3.12` environment with runtime extras
- run end-to-end 3D pipeline on `data/raw/3d`
- capture new run ID and refresh state docs
- exit criterion: successful 3D run from current branch

### M2: Supervised Evaluation

- label candidate fragment pairs
- create committed or externally tracked `labels.csv`
- calibrate `min_match_accuracy` and `min_required_accuracy`
- exit criterion: measured pairwise accuracy on labeled data

### M3: Quality Gates

- add regression assertions around report generation
- add artifact layout assertions for 3D and 2D
- define acceptable completeness and Chamfer bands
- exit criterion: reproducible acceptance thresholds

### M4: Packaging Cleanup

- retire compatibility surfaces that no longer serve external automation
- separate generated submission bundle policy from source tree policy
- document dataset ingestion for external evaluators
- exit criterion: no ambiguity around canonical runtime path

### M5: Delivery Hardening

- refresh submission bundle from a current verified run
- pin environment instructions for evaluators
- verify GitHub branch and deliverable links
- exit criterion: evaluator can reproduce run without local tribal knowledge
