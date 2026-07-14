# Native Cohort Scanner-Spectrum QC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically publish a participant-first raw, gradient-corrected, and final cohort scanner-spectrum figure and its numerical TSV at the end of native EEG-fMRI correction.

**Architecture:** Preserve each completed run's already-computed 15–90 Hz channel-median spectra in a small immutable record, aggregate runs within participants before cohort estimation, and use the shared deterministic participant bootstrap. Keep aggregation in a focused core QC module, rendering in the existing correction plotting module, and orchestration in the fixed pain-study runner.

**Tech Stack:** Python 3.11, NumPy, MNE-Python, Matplotlib, PyYAML, pytest, Ruff.

---

### Task 1: Share the deterministic participant bootstrap

**Files:**
- Create: `eeg_pipeline/analysis/participant_bootstrap.py`
- Modify: `studies/pain_study/study1/figures/spectral_statistics.py`
- Modify: `studies/pain_study/study1/figures/scanner_harmonic_spectrum.py`
- Modify: `studies/pain_study/study1/figures/cohort_power_spectral_density.py`
- Test: `studies/tests/pipelines/test_study1_scanner_harmonic_spectrum.py`

- [ ] **Step 1: Move the existing bootstrap test to the core import**

```python
from eeg_pipeline.analysis.participant_bootstrap import paired_participant_bootstrap
```

- [ ] **Step 2: Run the focused test and verify the new module is absent**

Run: `PYTHONPATH=. .venv/bin/python -m pytest studies/tests/pipelines/test_study1_scanner_harmonic_spectrum.py -q`

Expected: collection fails with `ModuleNotFoundError` for `participant_bootstrap`.

- [ ] **Step 3: Move the implementation without changing its algorithm**

```python
BOOTSTRAP_BATCH_SIZE = 256


def paired_participant_bootstrap(
    values: np.ndarray,
    *,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value_matrix = np.asarray(values, dtype=float)
    if value_matrix.ndim != 2 or value_matrix.shape[0] < 1:
        raise ValueError("Participant bootstrap requires a non-empty two-dimensional matrix.")
    if not np.isfinite(value_matrix).all():
        raise ValueError("Participant bootstrap values must be finite.")
    if iterations < 1:
        raise ValueError("Participant bootstrap iterations must be positive.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("Participant bootstrap confidence_level must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, value_matrix.shape[0], size=(iterations, value_matrix.shape[0]))
    estimates = np.empty((iterations, value_matrix.shape[1]), dtype=float)
    for start in range(0, iterations, BOOTSTRAP_BATCH_SIZE):
        stop = min(start + BOOTSTRAP_BATCH_SIZE, iterations)
        estimates[start:stop] = np.median(value_matrix[indices[start:stop]], axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return (
        np.median(value_matrix, axis=0),
        np.quantile(estimates, alpha, axis=0),
        np.quantile(estimates, 1.0 - alpha, axis=0),
    )
```

Update all callers to import this function from the core module. Leave only the Study 1 bootstrap configuration dataclass and loader in `spectral_statistics.py`.

- [ ] **Step 4: Run the existing Study 1 spectrum tests**

Run: `PYTHONPATH=. .venv/bin/python -m pytest studies/tests/pipelines/test_study1_scanner_harmonic_spectrum.py studies/tests/pipelines/test_study1_cohort_power_spectral_density.py -q`

Expected: all tests pass with identical seeded estimates.

- [ ] **Step 5: Commit the bootstrap extraction**

```bash
git add eeg_pipeline/analysis/participant_bootstrap.py studies/pain_study/study1/figures studies/tests/pipelines/test_study1_scanner_harmonic_spectrum.py
git commit -m "refactor: share participant spectral bootstrap"
```

### Task 2: Implement participant-first cohort scanner spectra

**Files:**
- Create: `eeg_pipeline/preprocessing/eeg_fmri/cohort_spectrum.py`
- Create: `tests/preprocessing/test_eeg_fmri_cohort_spectrum.py`

- [ ] **Step 1: Write failing tests for extraction, participant weighting, and validation**

```python
def test_cohort_scanner_spectrum_aggregates_runs_within_participants_first() -> None:
    runs = (
        run_spectrum("sub-0001", 1, offset=0.0),
        run_spectrum("sub-0001", 2, offset=20.0),
        run_spectrum("sub-0002", 1, offset=10.0),
    )
    cohort = aggregate_cohort_scanner_spectra(
        runs,
        bootstrap_iterations=50,
        confidence_level=0.95,
        bootstrap_seed=42,
    )
    assert cohort.participant_count == 2
    assert cohort.run_count == 3
    assert cohort.raw.median_power_db == pytest.approx(expected_participant_first_median)


def test_cohort_scanner_spectrum_rejects_inconsistent_frequency_bins() -> None:
    with pytest.raises(ValueError, match="frequency bins"):
        aggregate_cohort_scanner_spectra(
            inconsistent_runs,
            bootstrap_iterations=50,
            confidence_level=0.95,
            bootstrap_seed=42,
        )
```

- [ ] **Step 2: Run the new tests and verify imports fail**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/preprocessing/test_eeg_fmri_cohort_spectrum.py -q`

Expected: collection fails because `cohort_spectrum.py` does not exist.

- [ ] **Step 3: Add immutable run, stage, and cohort data structures**

```python
@dataclass(frozen=True)
class RunScannerSpectra:
    subject: str
    run: int
    channel_count: int
    raw: HarmonicSpectrum
    gradient_corrected: HarmonicSpectrum
    final: HarmonicSpectrum


@dataclass(frozen=True)
class CohortStageSpectrum:
    frequencies_hz: np.ndarray
    median_power_db: np.ndarray
    confidence_low_power_db: np.ndarray
    confidence_high_power_db: np.ndarray


@dataclass(frozen=True)
class CohortScannerSpectra:
    participant_count: int
    run_count: int
    channel_count: int
    raw: CohortStageSpectrum
    gradient_corrected: CohortStageSpectrum
    final: CohortStageSpectrum
```

Implement `extract_run_scanner_spectra` to crop every stage to 15–90 Hz immediately and
`aggregate_cohort_scanner_spectra` to validate identities, channels, stages, bins, and finite
values before participant-first median aggregation and paired bootstrap estimation.

- [ ] **Step 4: Add deterministic TSV serialization tests and implementation**

```python
write_cohort_scanner_spectra_tsv(cohort, output_path)
rows = list(csv.DictReader(output_path.open(), delimiter="\t"))
assert set(rows[0]) == {
    "stage",
    "frequency_hz",
    "median_psd_db_v2_hz",
    "ci_low_psd_db_v2_hz",
    "ci_high_psd_db_v2_hz",
    "n_participants",
    "n_runs",
}
```

- [ ] **Step 5: Run the new module tests**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/preprocessing/test_eeg_fmri_cohort_spectrum.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit the aggregation module**

```bash
git add eeg_pipeline/preprocessing/eeg_fmri/cohort_spectrum.py tests/preprocessing/test_eeg_fmri_cohort_spectrum.py
git commit -m "feat: aggregate native cohort scanner spectra"
```

### Task 3: Add the five-panel cohort spectral figure

**Files:**
- Modify: `eeg_pipeline/preprocessing/eeg_fmri/plotting.py`
- Modify: `tests/scripts/test_run_native_eeg_fmri_artifact_correction.py`

- [ ] **Step 1: Write a failing five-panel figure test**

```python
figure = build_cohort_scanner_spectrum_qc_figure(cohort)
assert len(figure.axes) == 5
assert figure.axes[0].get_xlim() == pytest.approx((15.0, 90.0))
assert [axis.get_title() for axis in figure.axes[1:]] == [
    "20.0 Hz raw reference",
    "41.0 Hz raw reference",
    "61.0 Hz raw reference",
    "82.0 Hz raw reference",
]
assert all(len(axis.collections) >= 3 for axis in figure.axes)
```

- [ ] **Step 2: Run the figure test and verify the builder is missing**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/scripts/test_run_native_eeg_fmri_artifact_correction.py::test_cohort_scanner_spectrum_qc_matches_run_layout -q`

Expected: failure because the cohort spectrum figure builder does not exist.

- [ ] **Step 3: Reuse the run-level layout for cohort curves**

Add `build_cohort_scanner_spectrum_qc_figure` and `save_cohort_scanner_spectrum_qc_figure`.
The builder must draw stage medians and confidence bands, locate raw cohort references in the four
prespecified windows, annotate AAS attenuation and final prominence, share local y limits, and
state participant, run, and channel counts in the title.

- [ ] **Step 4: Run plotting tests**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/scripts/test_run_native_eeg_fmri_artifact_correction.py -q`

Expected: all plotting and runner tests pass.

- [ ] **Step 5: Commit the cohort renderer**

```bash
git add eeg_pipeline/preprocessing/eeg_fmri/plotting.py tests/scripts/test_run_native_eeg_fmri_artifact_correction.py
git commit -m "feat: plot cohort scanner spectra"
```

### Task 4: Integrate automatic v3 publication

**Files:**
- Modify: `eeg_pipeline/preprocessing/eeg_fmri/config.py`
- Modify: `studies/pain_study/scripts/config/native_eeg_fmri_artifact_correction.yaml`
- Modify: `studies/pain_study/scripts/run_native_eeg_fmri_artifact_correction.py`
- Modify: `tests/preprocessing/test_eeg_fmri_config.py`
- Modify: `tests/scripts/test_run_native_eeg_fmri_artifact_correction.py`

- [ ] **Step 1: Write failing configuration and publication tests**

```python
assert parameters.qc_bootstrap_iterations == 10_000
assert parameters.qc_bootstrap_confidence_level == 0.95
assert parameters.qc_bootstrap_seed == 42

assert (output_root / "cohort_scanner_spectrum_qc.png").is_file()
assert (output_root / "cohort_scanner_spectrum_qc.tsv").is_file()
assert runner.DEFAULT_OUTPUT_ROOT.name == "native_eeg_fmri_correction-v3"
```

- [ ] **Step 2: Run the focused tests and confirm the new contract fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/preprocessing/test_eeg_fmri_config.py tests/scripts/test_run_native_eeg_fmri_artifact_correction.py -q`

Expected: failures for missing bootstrap parameters, completed-run spectra, cohort artifacts, and v3 root.

- [ ] **Step 3: Add strict YAML bootstrap parameters**

```yaml
qc:
  bootstrap:
    iterations: 10000
    confidence_level: 0.95
    seed: 42
```

Require exactly these keys and validate iterations, confidence level, and seed at configuration
load time.

- [ ] **Step 4: Return a completed-run record from `process_recording`**

```python
@dataclass(frozen=True)
class CompletedRun:
    manifest_row: dict[str, object]
    scanner_spectra: RunScannerSpectra
```

Create the cropped `RunScannerSpectra` from `result.harmonic_stages` before releasing the result.
Update the cohort loop to collect completed runs and derive manifest rows explicitly.

- [ ] **Step 5: Publish both cohort spectral artifacts before the atomic rename**

```python
cohort_spectra = aggregate_cohort_scanner_spectra(
    [completed.scanner_spectra for completed in completed_runs],
    bootstrap_iterations=parameters.qc_bootstrap_iterations,
    confidence_level=parameters.qc_bootstrap_confidence_level,
    bootstrap_seed=parameters.qc_bootstrap_seed,
)
write_cohort_scanner_spectra_tsv(
    cohort_spectra,
    incomplete_root / "cohort_scanner_spectrum_qc.tsv",
)
save_cohort_scanner_spectrum_qc_figure(
    cohort_spectra,
    incomplete_root / "cohort_scanner_spectrum_qc.png",
)
```

Set the default derivative root and generated-by version to v3 without changing corrected FIF
behavior.

- [ ] **Step 6: Run focused configuration and runner tests**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/preprocessing/test_eeg_fmri_config.py tests/preprocessing/test_eeg_fmri_cohort_spectrum.py tests/scripts/test_run_native_eeg_fmri_artifact_correction.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit automatic publication**

```bash
git add eeg_pipeline/preprocessing/eeg_fmri/config.py studies/pain_study/scripts/config/native_eeg_fmri_artifact_correction.yaml studies/pain_study/scripts/run_native_eeg_fmri_artifact_correction.py tests/preprocessing/test_eeg_fmri_config.py tests/scripts/test_run_native_eeg_fmri_artifact_correction.py
git commit -m "feat: publish cohort scanner spectrum QC"
```

### Task 5: Document and verify the completed feature

**Files:**
- Modify: `docs/native_eeg_fmri_artifact_correction.md`
- Modify: `studies/pain_study/scripts/README.md`

- [ ] **Step 1: Document the participant-first plot and v3 outputs**

State the aggregation order, bootstrap settings, exact output filenames, units, fail-fast checks,
and distinction between the distribution summary and spectral cohort figure.

- [ ] **Step 2: Run formatting and focused verification**

Run: `black eeg_pipeline/analysis/participant_bootstrap.py eeg_pipeline/preprocessing/eeg_fmri/cohort_spectrum.py eeg_pipeline/preprocessing/eeg_fmri/plotting.py eeg_pipeline/preprocessing/eeg_fmri/config.py studies/pain_study/scripts/run_native_eeg_fmri_artifact_correction.py tests/preprocessing/test_eeg_fmri_cohort_spectrum.py tests/preprocessing/test_eeg_fmri_config.py tests/scripts/test_run_native_eeg_fmri_artifact_correction.py`

Run: `ruff check eeg_pipeline/analysis/participant_bootstrap.py eeg_pipeline/preprocessing/eeg_fmri studies/pain_study/scripts/run_native_eeg_fmri_artifact_correction.py tests/preprocessing/test_eeg_fmri_cohort_spectrum.py tests/preprocessing/test_eeg_fmri_config.py tests/scripts/test_run_native_eeg_fmri_artifact_correction.py`

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/preprocessing/test_eeg_fmri_config.py tests/preprocessing/test_eeg_fmri_qc.py tests/preprocessing/test_eeg_fmri_cohort_spectrum.py tests/scripts/test_run_native_eeg_fmri_artifact_correction.py studies/tests/pipelines/test_study1_scanner_harmonic_spectrum.py studies/tests/pipelines/test_study1_cohort_power_spectral_density.py -q`

Run: `make verify-architecture`

Expected: every command exits successfully.

- [ ] **Step 3: Run the complete 83-run v3 cohort**

Run: `PYTHONPATH=. .venv/bin/python studies/pain_study/scripts/run_native_eeg_fmri_artifact_correction.py`

Expected: `/Volumes/KINGSTON/EEG_fMRI_data/derivatives/native_eeg_fmri_correction-v3` is atomically
published with 83 run outputs and both cohort scanner-spectrum artifacts.

- [ ] **Step 4: Verify numerical and visual artifacts**

Validate the TSV row count, stages, shared frequencies, participant/run counts, finite ordered
confidence bounds, and PNG metadata. Inspect the PNG at original resolution for five panels,
legible labels, confidence bands, harmonic references, and unclipped content.

- [ ] **Step 5: Commit documentation and final verification state**

```bash
git add docs/native_eeg_fmri_artifact_correction.md studies/pain_study/scripts/README.md
git commit -m "docs: describe cohort scanner spectrum QC"
```
