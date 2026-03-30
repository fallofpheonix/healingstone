# Security Audit Report

**Total Findings:** 16
**Critical (unsafe code):** 4
**Path Traversal Risks:** 0
**Input Validation Gaps:** 12

## Unsafe Patterns

| File | Line | Issue | Code |
| :--- | :--- | :--- | :--- |
| core/security_audit.py | 26 | Unsafe deserialization (pickle.load) | `r"\bpickle\.load\b": "Unsafe deserialization (pickle.load)",` |
| core/security_audit.py | 27 | Use of eval() — code injection risk | `r"\beval\(": "Use of eval() — code injection risk",` |
| core/security_audit.py | 28 | Use of exec() — code injection risk | `r"\bexec\(": "Use of exec() — code injection risk",` |
| ml_models/train_model.py | 170 | Use of eval() — code injection risk | `model.eval()` |

## Path Traversal

✅ No issues found.

## Input Validation

| File | Line | Issue | Code |
| :--- | :--- | :--- | :--- |
| pipeline/run_pipeline.py | 684 | CLI arg '--config' has no type validation | `add_argument('--config', ...)` |
| pipeline/run_pipeline.py | 685 | CLI arg '--train-config' has no type validation | `add_argument('--train-config', ...)` |
| pipeline/run_pipeline.py | 686 | CLI arg '--dataset-manifest' has no type validation | `add_argument('--dataset-manifest', ...)` |
| pipeline/run_pipeline.py | 688 | CLI arg '--data-dir' has no type validation | `add_argument('--data-dir', ...)` |
| pipeline/run_pipeline.py | 689 | CLI arg '--output-dir' has no type validation | `add_argument('--output-dir', ...)` |
| pipeline/run_pipeline.py | 690 | CLI arg '--labels-csv' has no type validation | `add_argument('--labels-csv', ...)` |
| pipeline/run_pipeline.py | 691 | CLI arg '--allow-overwrite-run' has no type validation | `add_argument('--allow-overwrite-run', ...)` |
| pipeline/run_pipeline.py | 710 | CLI arg '--threshold-objective' has no type validation | `add_argument('--threshold-objective', ...)` |
| pipeline/run_pipeline.py | 713 | CLI arg '--evaluation-split' has no type validation | `add_argument('--evaluation-split', ...)` |
| pipeline/run_pipeline.py | 715 | CLI arg '--augment-rotations' has no type validation | `add_argument('--augment-rotations', ...)` |
| pipeline/run_pipeline.py | 719 | CLI arg '--device' has no type validation | `add_argument('--device', ...)` |
| pipeline/test_pipeline.py | 335 | CLI arg '--output' has no type validation | `add_argument('--output', ...)` |

