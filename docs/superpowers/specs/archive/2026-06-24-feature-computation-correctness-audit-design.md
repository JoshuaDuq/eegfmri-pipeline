# Feature Computation Correctness Audit Design

## Objective

Audit the highest-risk EEG feature-computation paths, reproduce confirmed defects with
focused tests, and make the smallest behavior-preserving fixes that improve scientific
correctness, validation, numerical stability, determinism, and failure transparency.

The audit prioritizes scientific defects over cosmetic style findings. It does not add
fallback behavior, compatibility shims, or silent recovery paths.

## Current Baseline

The repository baseline was established before implementation:

- the configured Ruff checks pass;
- all 16 maintainability checks pass;
- all 1,281 tests pass when `MNE_DONTWRITE_HOME=true` gives MNE a writable temporary
  configuration directory;
- the suite reports 21 warnings, including project-owned datetime and pandas
  deprecations; and
- the feature-computation and shared statistical modules contain approximately 44,000
  lines, with several individual modules exceeding 3,000 lines.

Passing regression tests establish the current behavior but do not independently validate
all estimator assumptions or boundary conditions.

## Scope

### Primary production code

- `eeg_pipeline/analysis/features/`
- `eeg_pipeline/utils/analysis/`
- `eeg_pipeline/context/features.py`
- `eeg_pipeline/pipelines/features.py`
- `eeg_pipeline/domain/features/`

### Primary tests

- `tests/features/`
- feature-related tests in `tests/pipelines/`, `tests/utils/`, and `tests/cli/`

### Feature families

Audit in descending risk order:

1. shared preparation, trial alignment, window selection, and feature-result assembly;
2. spectral power, ERDS, aperiodic fitting, and periodic peak metrics;
3. phase, ITPC, PAC, surrogate generation, and cross-trial estimators;
4. static and dynamic connectivity, graph summaries, and source connectivity;
5. complexity, microstates, ERP, burst, and signal-quality features;
6. source-localization feature boundaries where they share feature-computation contracts;
7. naming, provenance, serialization, and canonical trial-table export.

The Go TUI and unrelated pipeline code are out of scope unless inspection proves that a
configuration value reaching feature computation is represented incorrectly there.

Existing uncommitted changes in ICA preprocessing and the pain-study issue log are
user-owned and will not be altered by this audit.

## Audit Method

### 1. Boundary and invariant review

Trace data from the feature pipeline entry point through precomputation, extractors, result
assembly, and persistence. Verify that entry points explicitly validate:

- array dimensionality and axis meaning;
- trial, event, condition, channel, frequency, and time-axis alignment;
- sampling rates, frequency ranges, time windows, and baseline placement;
- required sample, epoch, and channel counts;
- finite values and allowed missingness;
- train-only masks for cross-trial or learned quantities; and
- consistent feature-row counts and identifiers before merging or saving.

Invalid required inputs must raise descriptive exceptions. Optional feature unavailability
may produce an explicit structured skip only where the public contract already defines the
feature as optional. Broad exception handlers that obscure programming or data errors are
not acceptable.

### 2. Independent scientific checks

For high-risk estimators, compare implementation behavior with small deterministic signals
whose expected properties can be derived independently. Examples include:

- sinusoids and mixtures with known spectral peaks and band-power ordering;
- zero or phase-shifted oscillators with known phase-locking and connectivity behavior;
- coupled and uncoupled signals for PAC and surrogate checks;
- constant, random, and periodic signals for complexity boundary cases;
- known state sequences for microstate duration, occurrence, and transition metrics; and
- deliberately misaligned trial metadata to verify fail-fast behavior.

Tests should assert scientifically meaningful invariants rather than duplicating internal
implementation steps.

### 3. Numerical and reproducibility review

Inspect divisions, logarithms, normalizations, regressions, matrix operations, and NaN-aware
aggregations. Confirm that degenerate inputs either have a scientifically defined result or
fail explicitly. Do not coerce invalid results to plausible finite values.

All stochastic computations must use an explicit, traceable seed or generator. Repeated
runs with the same inputs and configuration must produce identical outputs where the
underlying libraries support determinism. Cross-validation and learned reference quantities
must not use held-out data.

### 4. Output and provenance review

Verify that feature names, rows, columns, units, window labels, condition labels, and
provenance match the computed quantity. Serialization must preserve non-finite-value policy,
configuration values that materially affect computation, random seeds, and subject/trial
identity. Merges must reject ambiguous keys, duplicate columns, and inconsistent row counts.

## Repair Policy

Each confirmed issue follows one isolated red-green-refactor cycle:

1. document the concrete failure and trace it to its root cause;
2. add the smallest regression test that fails for the correct reason;
3. run the test and record the expected failure;
4. implement one minimal fix;
5. run the focused test and the relevant feature-family suite;
6. refactor only the touched path while tests remain green; and
7. inspect the diff before proceeding to the next issue.

No production change is justified only by a lint suggestion. Extended Ruff findings are
triage inputs; they are fixed only when they expose ambiguity, error masking, misalignment,
or unnecessary complexity in an already touched path.

## Verification

Verification is proportional to each repair and culminates in a repository-wide gate:

1. focused regression test;
2. related feature-family test module;
3. all tests in `tests/features/` plus affected pipeline and utility tests;
4. configured Ruff checks;
5. `make verify-maintainability`;
6. `git diff --check`; and
7. `MNE_DONTWRITE_HOME=true make test` for all 1,281 baseline tests plus new tests.

Warnings introduced or changed by the audit are treated as failures unless an upstream
library warning is intentionally exercised and asserted or filtered at the narrowest test
boundary.

## Delivery Strategy

Work proceeds in independently reviewable batches, ordered by risk and shared impact:

1. shared invariants and feature-pipeline boundaries;
2. spectral and aperiodic computation;
3. phase, ITPC, and PAC computation;
4. connectivity computation;
5. remaining feature families and provenance;
6. final cross-family verification and residual-risk report.

The audit stops short of speculative module rewrites. If correctness fixes reveal that a
large module boundary itself prevents safe repair, the architectural change will be proposed
separately with explicit behavior-preservation tests.

## Success Criteria

- Every production edit addresses a reproduced defect, a demonstrated scientific-risk
  condition, or an in-scope project warning.
- Every defect fix has a regression test observed failing before the implementation change.
- Invalid required inputs fail fast with specific messages.
- Feature outputs remain compatible unless the previous output is scientifically incorrect;
  any intentional correction is documented explicitly.
- Deterministic computations remain reproducible for identical inputs and configuration.
- Existing user-owned worktree changes remain intact.
- The final configured lint, maintainability, feature, and full-test gates pass.
- Remaining unverified scientific assumptions and deferred architectural risks are reported
  explicitly rather than hidden behind successful tests.
