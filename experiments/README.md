# Experiments

This directory is reserved for **non-production** experimental code and research prototypes.

## Rules

1. **No production dependency**: Code in `experiments/` must NEVER be imported by the core pipeline (`src/healingstone/`).
2. **No guarantees**: Experiments are not covered by CI, type checking, or determinism contracts.
3. **Self-contained**: Each experiment should include its own README or docstring explaining purpose and usage.
4. **Ephemeral**: Experiments may be archived or deleted at any time without affecting system correctness.

## Structure

```
experiments/
├── README.md           # This file
├── notebook_*.py       # Standalone experiment scripts
└── <experiment_name>/  # Multi-file experiments in subdirectories
    ├── README.md
    └── ...
```

## Creating an Experiment

```bash
mkdir experiments/my_experiment
# Add a README.md explaining hypothesis, setup, and expected output
```
