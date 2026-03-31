# System Architecture: HealingStone Reassembly Pipeline

The HealingStone system is designed as a modular, stage-based pipeline for the automated reconstruction of fragmented 3D artifacts.

## System Overview

The architecture follows a strict **linear orchestration model** where data moves through a sequence of transformation stages. Each stage is isolated and communicates via well-defined data contracts.

```mermaid
graph TD
    A[Data Ingestion / IO] -->|InputData| B(Preprocessing Stage)
    B -->|List[Fragment]| C(Matching Stage)
    C -->|List[MatchResult]| D(Alignment Stage)
    D -->|List[AlignmentResult]| E(Global Assembly Stage)
    E -->|ReconstructionResult| F[Data Export / Persistence]
```

## Core Components

### 1. Cli Interface (`src/healingstone/cli.py`)
The primary entrypoint. It handles configuration loading, validation, and triggers the orchestrator. It supports subcommands for `run`, `eval`, `inspect`, and `validate-config`.

### 2. Pipeline Orchestrator (`src/healingstone/pipeline/`)
- **`Stage` (ABC)**: Defines the interface for all pipeline steps. Each stage must implement `_execute(input_data)` and is wrapped with logging and timing instrumentation.
- **`PipelineRunner`**: Manages the execution sequence, experiment directory initialization, and configuration snapshotting.

### 3. Core Logic (`src/healingstone/core/`)
- **`geometry/`**: Contains algorithmic logic for point cloud preprocessing, normal estimation, and rigid-body alignment (ICP).
- **`models/`**: Houses ML models for feature extraction (PointNet++) and pairwise match prediction.
- **`reconstruction/`**: Implements global graph optimization to assemble multiple pairwise matches into a single coherent model.

### 4. Schema & Contracts (`src/healingstone/schema/`)
Uses `Pydantic` to enforce typed boundaries.
- **`config.py`**: Validates the `pipeline.yaml` structure.
- **`data.py`**: Defines `InputData`, `Fragment`, `MatchResult`, and `ReconstructionResult`.

## Data Strategy & Determinism

### Experiment Tracking
Every execution is assigned a unique `run_id`. All artifacts (logs, metrics, snapshots, models) are stored in `experiments/<run_id>/`, ensuring that results are always traceable to their specific configuration.

### Determinism Guarantees
Random seeds for `numpy`, `torch`, and `random` are fixed via the global pipeline configuration and applied at the start of every run. No non-deterministic operations are permitted within core algorithmic stages.

## Failure Model
The system employs a **fail-fast** strategy:
1. **Config Validation**: The pipeline will not start if the configuration is invalid.
2. **Stage Isolation**: If a stage fails, the error is logged with full context, and the pipeline terminates immediately to prevent the propagation of corrupted data.
