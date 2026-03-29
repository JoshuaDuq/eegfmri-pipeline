# ADR-0001: Module Boundaries

- Status: Accepted
- Date: 2026-02-22

## Context

The repository contains multiple layers (`cli`, `analysis`, `utils`) and has grown
large enough that accidental cross-layer imports can increase coupling and reduce
maintainability.

## Decision

Enforce the following import boundaries in tests:

1. `eeg_pipeline.analysis.*` must not import `eeg_pipeline.cli.*`.
2. `eeg_pipeline.utils.*` must not import `eeg_pipeline.cli.*`.
3. `eeg_pipeline.analysis.features.*` must not import
   `eeg_pipeline.analysis.machine_learning.*`.

## Consequences

- Pros: clearer ownership, lower coupling, easier refactoring.
- Cons: some convenience imports are disallowed and may require moving shared
  helpers into neutral modules.
- Enforcement: `tests/utils/test_architecture_import_boundaries.py` via
  `make verify-architecture`.
