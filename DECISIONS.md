# Decisions

## D001: Package-First Runtime

- decision:
  - treat the packaged CLI as canonical
- why:
  - stable import boundaries and reproducible entrypoints
- tradeoff:
  - some compatibility shims remain for older automation

## D002: Project-Root-Relative Paths

- decision:
  - resolve relative data and output paths from project root
- why:
  - removes caller-working-directory ambiguity
- tradeoff:
  - external scripts with old hardcoded paths must be updated

## D003: Canonical Data And Artifact Roots

- decision:
  - use `data/raw/3d` for local 3D data and `artifacts` for generated outputs
- why:
  - cleaner separation between source data and generated state
- tradeoff:
  - old `3D/`, `result/`, and `submission/` conventions are retired

## D004: Run-Scoped Artifact Isolation

- decision:
  - write all runtime output under `artifacts/runs/<run_id>/`
- why:
  - preserves reproducibility and avoids destructive overwrite
- tradeoff:
  - requires explicit navigation to inspect the latest run unless using `artifacts/latest`

## D005: Single Entry Point, Dual Mode

- decision:
  - auto-detect 2D versus 3D input within one pipeline entrypoint
- why:
  - keeps operational surface area small
- tradeoff:
  - mode detection must remain exact and case-insensitive

## D006: Versioned Metrics Schema

- decision:
  - enforce a strict schema for machine-readable metrics output
- why:
  - downstream tooling depends on stable keys and types
- tradeoff:
  - report evolution requires explicit schema version management
