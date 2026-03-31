# Design Decisions & Tradeoffs: HealingStone Reassembly

This document outlines the critical choices made during the engineering of the HealingStone system, explaining the rationales and tradeoffs.

## 1. Modular Execution Model vs. Scripting
**Decision**: Adopt a formal `abc.Stage` based orchestration model.
**Rationale**: 
- **Separation of Concerns**: Decouples algorithmic logic (in `core/`) from orchestration and data management.
- **Testability**: Individual stages can be unit-tested in isolation with fixed inputs.
- **Maintainability**: New reassembly methods can be added by implementing a single `Stage` subclass without modifying the runner.
**Tradeoff**: Slight overhead in complexity compared to simple scripts.

## 2. Pydantic-based Data Contracts
**Decision**: Every stage boundary is enforced via `Pydantic` schema validation.
**Rationale**: 
- **Type Safety**: Catches subtle data format errors early (e.g., mismatched numpy shapes).
- **Self-Documentation**: Schemas serve as live documentation for the data pipeline.
**Tradeoff**: Performance penalty for data validation at runtime (mitigated by only validating at stage boundaries).

## 3. Local Experiment Tracking
**Decision**: Use a structured `experiments/<run_id>/` filesystem-based tracking system.
**Rationale**: 
- **Portability**: No dependency on external services (WandB/MLflow) which can break during local development.
- **Minimalism**: Results are stored adjacent to the code that produced them.
**Tradeoff**: Lacks high-level dashboard features of dedicated tracking platforms.

## 4. Single-Entrypoint CLI
**Decision**: Unified `healingstone.cli` with subcommands (`run`, `eval`, `inspect`).
**Rationale**: 
- **Consistency**: All interactions follow the same pattern.
- **Discoverability**: `--help` provides a complete map of system capabilities.
**Tradeoff**: More boilerplate code than multiple standalone scripts.

## 5. Determinism Layer
**Decision**: Global seed enforcement at the beginning of every run.
**Rationale**: 
- **Reproducibility**: Identical configs must yield identical results for mentor-grade review.
**Tradeoff**: Adds deterministic constraints that may slightly impact execution speed on some GPU operations.
