# Problem Statement: Healing Stones

## Problem Statement

Fragmented archaeological artifacts present a difficult reconstruction problem due to:

- unknown relative orientation of fragments
- missing material along break surfaces
- erosion and scan noise
- variable fragment sizes

Given a set of 3D mesh fragments `F = {f_1, f_2, ..., f_n}` derived from `.PLY` or `.OBJ` scans, the objective is to automatically determine:

1. break surfaces on each fragment
2. matching fragment pairs
3. relative pose transformations
4. global assembly of fragments

Each fragment is represented as a mesh:

```text
f_i = (V_i, E_i, F_i)
```

and the reconstruction pipeline must estimate transforms:

```text
T_i in SE(3)
```

such that transformed fragments minimize geometric misalignment along probable break surfaces.

## Key Technical Challenges

### 1. Surface Classification

Fragments contain multiple surface types:
- original carved surface
- break surface
- erosion artifacts

The system must separate likely break surfaces from non-break surfaces without assuming clean labels.

### 2. Imperfect Geometry

Break edges often contain:
- chipped material
- erosion
- scanning noise
- incomplete overlap

Therefore exact matching is not realistic. The algorithm must tolerate partial overlap, missing volume, and topology gaps.

### 3. Combinatorial Search

For `n` fragments, naive matching requires `O(n^2)` pairwise comparisons. Candidate pruning is required before geometric registration.

### 4. Automation Constraint

The final system must run without human interaction and accept arbitrary `.PLY` or `.OBJ` fragment datasets.

## Repository-Specific Interpretation

For this repository, the assignment is narrower than open-ended reconstruction research:

- canonical runtime is the packaged CLI in `healingstone/src/healingstone/`
- the pipeline must emit machine-readable artifacts, not only visual outputs
- configuration, resolved paths, and metrics must be reproducible across runs
- the system must remain usable when no supervised pair labels are available

## Required Outputs

Given an input fragment directory, the system must produce:

- candidate pair scores
- per-pair alignment diagnostics
- global reconstruction output
- run metadata and resolved paths
- schema-validated metrics
- visual diagnostics for similarity, alignment, and final assembly

## Non-Goals

The current assignment does not require:

- manual interactive reconstruction tooling
- a GUI
- archaeological interpretation of the reconstructed object
- guaranteed perfect assembly under severe erosion or missing fragments
