# Feature Computation Correctness Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate confirmed silent-failure and scientific-validity defects in the shared EEG feature-computation contracts and highest-risk estimators.

**Architecture:** Enforce shape and trial-identity invariants at shared data boundaries, then correct estimator-specific failure semantics with small focused changes. Every production change is preceded by a regression test that demonstrates the current defect; no module rewrite or compatibility shim is included.

**Tech Stack:** Python 3.14, NumPy, pandas, MNE-Python, antropy, pytest, Ruff

---

## File Structure

- Create `tests/features/test_feature_data_contracts.py` for `PrecomputedData`,
  `FeatureSet`, and `ExtractionResult` boundary contracts.
- Modify `eeg_pipeline/types.py` to validate precomputed axes and reject empty crops.
- Modify `eeg_pipeline/analysis/features/results.py` to reject inconsistent result frames.
- Modify `eeg_pipeline/utils/data/feature_alignment.py` to require exact event/table alignment.
- Modify `eeg_pipeline/utils/data/features.py` to validate block lengths before masking.
- Modify `eeg_pipeline/pipelines/features.py` to reject ambiguous dataframe merges.
- Modify `tests/features/test_feature_alignment_masking.py` and
  `tests/pipelines/test_pipeline_features.py` for alignment/merge regressions.
- Modify `eeg_pipeline/analysis/features/quality.py` and
  `tests/features/test_feature_scientific_validity_guards.py` for strict Welch errors.
- Modify `eeg_pipeline/analysis/features/phase.py` and
  `tests/features/test_feature_pac_surrogates.py` for distinct surrogate semantics.
- Modify `eeg_pipeline/analysis/features/spectral.py` and
  `tests/features/test_feature_scientific_validity_issues.py` for descriptor validation.
- Modify `eeg_pipeline/analysis/features/connectivity.py` and
  `tests/features/test_feature_connectivity_validity_guards.py` for strict configuration
  and wavelet failures.
- Modify `eeg_pipeline/utils/analysis/signal_metrics.py` and
  `tests/utils/test_signal_metrics_complexity_entropy.py` to remove entropy fallbacks.
- Modify `eeg_pipeline/domain/features/naming.py`,
  `eeg_pipeline/cli/commands/base_feature_availability.py`,
  `tests/features/test_feature_provenance.py`, and
  `tests/cli/test_cli_info_subject_status_optimizations.py` for timezone-aware UTC metadata.

### Task 1: Validate Precomputed Feature Data

**Files:**
- Create: `tests/features/test_feature_data_contracts.py`
- Modify: `eeg_pipeline/types.py`

- [ ] **Step 1: Write failing shape and crop contract tests**

Add tests that construct a valid three-dimensional `PrecomputedData` fixture, then assert:

```python
def test_precomputed_data_rejects_time_axis_mismatch():
    with pytest.raises(ValueError, match="times length"):
        PrecomputedData(
            data=np.zeros((2, 3, 5)),
            times=np.arange(4, dtype=float),
            sfreq=100.0,
            ch_names=["C3", "C4", "Pz"],
            picks=np.arange(3),
        )


def test_precomputed_data_rejects_metadata_trial_mismatch():
    with pytest.raises(ValueError, match="metadata rows"):
        _valid_precomputed(metadata=pd.DataFrame({"trial": [1]}))


def test_precomputed_crop_rejects_nonoverlapping_range():
    precomputed = _valid_precomputed()
    with pytest.raises(ValueError, match="does not overlap"):
        precomputed.crop(10.0, 11.0)
```

Also cover non-3D data, nonpositive sampling rate, channel/pick count mismatch,
condition-label length mismatch, train-mask length mismatch, and window-mask length mismatch.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
env MNE_DONTWRITE_HOME=true /Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python \
  -m pytest tests/features/test_feature_data_contracts.py -q
```

Expected: failures because `PrecomputedData` currently accepts mismatched axes and `crop()`
returns the original object for a nonoverlapping interval.

- [ ] **Step 3: Implement explicit dataclass invariants**

Add `PrecomputedData.__post_init__` helpers that require:

```python
def __post_init__(self) -> None:
    self.data = np.asarray(self.data)
    self.times = np.asarray(self.times, dtype=float)
    self.picks = np.asarray(self.picks)
    self._validate_core_axes()
    self._validate_trial_metadata()
    self._validate_windows()


def _validate_core_axes(self) -> None:
    if self.data.ndim != 3:
        raise ValueError(
            "PrecomputedData.data must have shape (epochs, channels, times); "
            f"got {self.data.shape}."
        )
    n_epochs, n_channels, n_times = self.data.shape
    if self.times.ndim != 1 or len(self.times) != n_times:
        raise ValueError(
            f"PrecomputedData times length ({len(self.times)}) does not match "
            f"data time axis ({n_times})."
        )
    if not np.isfinite(self.sfreq) or float(self.sfreq) <= 0:
        raise ValueError("PrecomputedData.sfreq must be a positive finite number.")
    if len(self.ch_names) != n_channels or len(self.picks) != n_channels:
        raise ValueError("PrecomputedData channel metadata does not match data channels.")
```

Validate optional metadata, condition labels, train mask, and every stored window mask
against `n_epochs` or `n_times` as appropriate. Change a nonoverlapping crop to raise a
descriptive `ValueError`. Remove `_recompute_windows` exception suppression and call
`time_windows_from_spec(spec, logger=self.logger, strict=True)`.

- [ ] **Step 4: Run contract and related precomputed tests**

Run:

```bash
env MNE_DONTWRITE_HOME=true /Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python \
  -m pytest tests/features/test_feature_data_contracts.py \
  tests/features/test_feature_api_precomputed_microstates_context.py \
  tests/features/test_feature_complexity_entropy.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the precomputed-data contract**

```bash
git add eeg_pipeline/types.py tests/features/test_feature_data_contracts.py
git commit -m "fix: validate precomputed feature data"
```

### Task 2: Reject Ambiguous Feature Alignment and Result Assembly

**Files:**
- Modify: `tests/features/test_feature_data_contracts.py`
- Modify: `tests/features/test_feature_alignment_masking.py`
- Modify: `tests/pipelines/test_pipeline_features.py`
- Modify: `eeg_pipeline/analysis/features/results.py`
- Modify: `eeg_pipeline/utils/data/feature_alignment.py`
- Modify: `eeg_pipeline/utils/data/features.py`
- Modify: `eeg_pipeline/pipelines/features.py`

- [ ] **Step 1: Write failing alignment tests**

Add tests proving that the current code silently accepts or ambiguously handles:

```python
def test_attach_alignment_columns_rejects_row_mismatch():
    features = pd.DataFrame({"power": [1.0, 2.0]})
    events = pd.DataFrame({"trial_id": [1, 2, 3]})
    with pytest.raises(ValueError, match="row count"):
        attach_feature_alignment_columns(features, events)


def test_extraction_result_rejects_misaligned_feature_indices():
    result = ExtractionResult(
        features={
            "a": FeatureSet(pd.DataFrame({"a": [1.0, 2.0]}), ["a"], "a"),
            "b": FeatureSet(
                pd.DataFrame({"b": [3.0, 4.0]}, index=[1, 2]), ["b"], "b"
            ),
        }
    )
    with pytest.raises(ValueError, match="row index"):
        result.get_combined_df()


def test_merge_dataframes_rejects_duplicate_payload_columns():
    with pytest.raises(ValueError, match="duplicate feature columns"):
        _merge_dataframes(
            [pd.DataFrame({"trial_id": [1], "alpha": [1.0]}),
             pd.DataFrame({"trial_id": [1], "alpha": [2.0]})]
        )
```

Add cases for duplicate/null event trial IDs, mismatched raw block lengths before finite
masking, mismatched condition length, `FeatureSet.columns` disagreement, duplicate columns,
mixed presence of `trial_id`, and differing trial-ID sets.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
env MNE_DONTWRITE_HOME=true /Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python \
  -m pytest tests/features/test_feature_data_contracts.py \
  tests/features/test_feature_alignment_masking.py \
  tests/pipelines/test_pipeline_features.py -q
```

Expected: the new cases fail because current code returns unaligned frames, broadcasts
masks before length validation, aligns by arbitrary indexes, or drops duplicate columns.

- [ ] **Step 3: Implement shared alignment guards**

Implement focused validators with contracts equivalent to:

```python
def attach_feature_alignment_columns(df, events_df):
    if events_df is None or df is None or df.empty:
        return df
    if len(df) != len(events_df):
        raise ValueError(
            f"Feature/event row count mismatch: features={len(df)}, events={len(events_df)}."
        )
    trial_ids = require_trial_id_column(events_df, context="Aligned events")
    if trial_ids.isna().any() or trial_ids.duplicated().any():
        raise ValueError("Aligned events trial_id values must be non-null and unique.")
    out = df.copy()
    event_series = trial_ids.reset_index(drop=True)
    if TRIAL_ID_COLUMN in out.columns:
        current = out[TRIAL_ID_COLUMN].reset_index(drop=True)
        if not current.equals(event_series):
            raise ValueError("Feature table trial_id values conflict with aligned events.")
        return out
    out.insert(0, TRIAL_ID_COLUMN, event_series)
    return out
```

Validate all nonempty feature-block lengths before `_create_finite_mask`, make
`_apply_drop_mask` raise on mismatch, validate `FeatureSet` columns in `__post_init__`, and
make `ExtractionResult` require equal row counts/indexes and unique columns before concat.

In `_merge_dataframes`, require either all or no frames to contain `trial_id`; require unique
IDs and identical ID sets when IDs exist; require equal row count/index when they do not;
and raise on duplicate payload columns instead of keeping the first silently.

- [ ] **Step 4: Run alignment, result, pipeline, and feature-I/O tests**

Run:

```bash
env MNE_DONTWRITE_HOME=true /Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python \
  -m pytest tests/features/test_feature_data_contracts.py \
  tests/features/test_feature_alignment_masking.py \
  tests/pipelines/test_pipeline_features.py tests/utils/test_feature_io_fail_fast.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the alignment contract**

```bash
git add eeg_pipeline/analysis/features/results.py \
  eeg_pipeline/utils/data/feature_alignment.py eeg_pipeline/utils/data/features.py \
  eeg_pipeline/pipelines/features.py tests/features/test_feature_data_contracts.py \
  tests/features/test_feature_alignment_masking.py tests/pipelines/test_pipeline_features.py
git commit -m "fix: enforce feature trial alignment"
```

### Task 3: Make Quality PSD Configuration Fail Fast

**Files:**
- Modify: `tests/features/test_feature_scientific_validity_guards.py`
- Modify: `eeg_pipeline/analysis/features/quality.py`

- [ ] **Step 1: Write failing Welch and PSD error tests**

Add direct tests of `_compute_psd` and `_compute_spectral_metrics`:

```python
def test_quality_rejects_overlap_equal_to_segment_length():
    data = np.ones((2, 100), dtype=float)
    with pytest.raises(ValueError, match="n_overlap"):
        _compute_psd(
            data, 100.0,
            {"psd_method": "welch", "n_per_seg": 50, "n_fft": 64,
             "n_overlap": 50, "exclude_line_noise": False},
        )


def test_quality_surfaces_psd_failure():
    with patch(
        "eeg_pipeline.analysis.features.quality._compute_psd",
        side_effect=ValueError("bad PSD"),
    ):
        with pytest.raises(ValueError, match="bad PSD"):
            _compute_spectral_metrics(np.ones((2, 100)), 100.0, {})
```

Also cover explicit `n_per_seg > n_times`, `n_fft < n_per_seg`, negative overlap, and
invalid finite line-noise frequencies.

- [ ] **Step 2: Run the focused tests and verify RED**

Run the new test node IDs with `pytest -q`; expect clamping or NaN substitution to make the
tests fail.

- [ ] **Step 3: Replace clamping and NaN substitution with validation**

Parse explicit Welch integers, require `2 <= n_per_seg <= n_times`, `n_fft >= n_per_seg`,
and `0 <= n_overlap < n_per_seg`. Preserve adaptive defaults only when the user did not
provide a value. Remove `_compute_spectral_metrics` exception handling so PSD errors retain
their original type and message. Reject nonfinite/nonpositive line-noise frequencies in
`_get_line_noise_parameters`.

- [ ] **Step 4: Run quality and strict-computation tests**

```bash
env MNE_DONTWRITE_HOME=true /Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python \
  -m pytest tests/features/test_feature_scientific_validity_guards.py \
  tests/features/test_feature_scientific_validity_issues.py \
  tests/features/test_eeg_compute_strict_failures.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit strict quality computation**

```bash
git add eeg_pipeline/analysis/features/quality.py \
  tests/features/test_feature_scientific_validity_guards.py
git commit -m "fix: surface quality PSD errors"
```

### Task 4: Separate PAC Surrogate Methods and Undefined Normalization

**Files:**
- Modify: `tests/features/test_feature_pac_surrogates.py`
- Modify: `eeg_pipeline/analysis/features/phase.py`

- [ ] **Step 1: Write failing surrogate-semantics tests**

Use a deterministic fake generator or known donor trace to assert that
`surrogate_method="trial_shuffle"` pairs cross-epoch amplitude without circularly shifting
it, while `"circular_shift"` shifts the same epoch. Add a zero-amplitude normalized PAC test:

```python
def test_normalized_pac_is_nan_when_amplitude_mass_is_zero():
    data = np.zeros((1, 1, 2, 8), dtype=np.complex128)
    values = _compute_pac_for_channel_band_pair(
        data,
        channel_idx=0,
        phase_freqs=np.array([6.0]),
        amp_freqs=np.array([40.0]),
        phase_indices=np.array([0]),
        amp_indices=np.array([1]),
        phase_band_range=(4.0, 8.0),
        amp_band_range=(30.0, 50.0),
        normalize=True,
        epsilon=1e-12,
        n_times=8,
    )
    assert np.isnan(values[0])
```

- [ ] **Step 2: Run PAC tests and verify RED**

Run `tests/features/test_feature_pac_surrogates.py`; expect trial-shuffle values to show the
unrequested additional roll and zero-amplitude PAC to equal zero.

- [ ] **Step 3: Correct PAC method semantics**

Only draw/apply a nonzero time shift when `method == "circular_shift"`. For normalized PAC
and normalized surrogates, divide only where the finite amplitude sum exceeds `epsilon`;
emit `NaN` for undefined zero-amplitude estimates rather than a plausible zero.

- [ ] **Step 4: Run PAC, phase, and API tests**

```bash
env MNE_DONTWRITE_HOME=true /Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python \
  -m pytest tests/features/test_feature_pac_surrogates.py \
  tests/features/test_feature_scientific_validity_guards.py \
  tests/features/test_feature_api_precomputed_microstates_context.py \
  tests/plotting/test_plotting_phase.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit PAC corrections**

```bash
git add eeg_pipeline/analysis/features/phase.py \
  tests/features/test_feature_pac_surrogates.py
git commit -m "fix: separate PAC surrogate methods"
```

### Task 5: Validate Spectral Descriptor Inputs

**Files:**
- Modify: `tests/features/test_feature_scientific_validity_issues.py`
- Modify: `eeg_pipeline/analysis/features/spectral.py`

- [ ] **Step 1: Write failing spectral-descriptor tests**

Add tests requiring equal one-dimensional PSD/frequency shapes, strictly increasing finite
frequencies, `0 < percentile <= 1`, nonnegative finite PSD support, and correct entropy
normalization when one frequency bin is missing:

```python
def test_spectral_entropy_normalizes_over_finite_bins():
    value = compute_spectral_entropy(
        np.array([1.0, np.nan, 1.0]),
        np.array([8.0, 9.0, 10.0]),
        8.0,
        10.0,
    )
    assert value == pytest.approx(1.0)
```

- [ ] **Step 2: Run the new descriptor tests and verify RED**

Run the added node IDs; expect invalid percentiles/shapes to return values and entropy to
normalize by three bins instead of the two observed bins.

- [ ] **Step 3: Add one shared descriptor-input validator**

Create a small helper that returns the selected finite support and bin widths after checking
the public descriptor contract. Use it in center, bandwidth, edge, and entropy calculations.
Keep zero-power bins in entropy's possible support but exclude missing bins from the
normalization denominator. Raise on negative PSD rather than clipping it.

- [ ] **Step 4: Run spectral and aperiodic tests**

```bash
env MNE_DONTWRITE_HOME=true /Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python \
  -m pytest tests/features/test_feature_scientific_validity_issues.py \
  tests/features/test_feature_aperiodic_periodic_peaks.py \
  tests/features/test_spectral_strict_failures.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit spectral validation**

```bash
git add eeg_pipeline/analysis/features/spectral.py \
  tests/features/test_feature_scientific_validity_issues.py
git commit -m "fix: validate spectral descriptors"
```

### Task 6: Surface Connectivity Configuration and Wavelet Errors

**Files:**
- Modify: `tests/features/test_feature_connectivity_validity_guards.py`
- Modify: `eeg_pipeline/analysis/features/connectivity.py`

- [ ] **Step 1: Write failing connectivity failure tests**

Add one test with `measures=["unknown"]` that expects a specific `ValueError`, and one
within-epoch wPLI test that patches `spectral_connectivity_time` to raise the known
wavelet-too-long `ValueError` and expects that error to surface with segment/band context.

- [ ] **Step 2: Run the new connectivity tests and verify RED**

Current behavior warns and returns an empty dataframe, so both strict tests must fail.

- [ ] **Step 3: Reject unsupported measures and re-raise contextual wavelet errors**

Replace unknown-measure filtering with:

```python
if unknown:
    raise ValueError(
        "Connectivity: unsupported measures: " + ", ".join(sorted(unknown))
    )
```

Replace the known-wavelet skip branch with a descriptive `ValueError` raised from the
original exception. Its message must include method, segment, band, sample count, duration,
and the corrective configuration options. Do not return an empty dataframe for a requested
estimator failure.

- [ ] **Step 4: Run all connectivity feature tests**

```bash
env MNE_DONTWRITE_HOME=true /Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python \
  -m pytest tests/features/test_feature_connectivity_dynamic.py \
  tests/features/test_feature_connectivity_validity_guards.py \
  tests/features/test_feature_source_connectivity_validity.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit connectivity failure semantics**

```bash
git add eeg_pipeline/analysis/features/connectivity.py \
  tests/features/test_feature_connectivity_validity_guards.py
git commit -m "fix: surface connectivity failures"
```

### Task 7: Remove Sample-Entropy Fallbacks

**Files:**
- Modify: `tests/utils/test_signal_metrics_complexity_entropy.py`
- Modify: `eeg_pipeline/utils/analysis/signal_metrics.py`

- [ ] **Step 1: Write a failing unexpected-error propagation test**

Patch `antropy.sample_entropy` to raise `RuntimeError("entropy boom")` and assert
`compute_sample_entropy` raises the same error. The current broad handler silently runs a
different estimator, so the test must fail.

- [ ] **Step 2: Run the focused test and verify RED**

Run `tests/utils/test_signal_metrics_complexity_entropy.py -q`; expect the new propagation
assertion to fail.

- [ ] **Step 3: Use the required antropy estimator directly**

Remove `_sample_entropy_fallback` and both fallback exception handlers. Import and call the
required `antropy.sample_entropy` implementation directly. Preserve the existing explicit
short-signal and tolerance behavior; unexpected dependency or estimator failures surface.

- [ ] **Step 4: Run signal-metric and complexity tests**

```bash
env MNE_DONTWRITE_HOME=true /Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python \
  -m pytest tests/utils/test_signal_metrics_complexity_entropy.py \
  tests/features/test_feature_complexity_entropy.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit entropy failure semantics**

```bash
git add eeg_pipeline/utils/analysis/signal_metrics.py \
  tests/utils/test_signal_metrics_complexity_entropy.py
git commit -m "fix: surface sample entropy errors"
```

### Task 8: Remove Feature Metadata UTC Deprecations

**Files:**
- Modify: `tests/features/test_feature_provenance.py`
- Modify: `tests/cli/test_cli_info_subject_status_optimizations.py`
- Modify: `eeg_pipeline/domain/features/naming.py`
- Modify: `eeg_pipeline/cli/commands/base_feature_availability.py`

- [ ] **Step 1: Write warning-as-error metadata tests**

Wrap `generate_manifest` and feature-availability timestamp creation in
`warnings.catch_warnings()` with `simplefilter("error", DeprecationWarning)`. Assert produced
timestamps retain the existing UTC representation ending in `Z`.

- [ ] **Step 2: Run the focused tests and verify RED**

On Python 3.14, current `datetime.utcnow()` and `utcfromtimestamp()` calls raise the promoted
deprecation warning.

- [ ] **Step 3: Use timezone-aware UTC and preserve serialized format**

Import `UTC` and use:

```python
datetime.now(UTC).isoformat().replace("+00:00", "Z")
datetime.fromtimestamp(timestamp, UTC).isoformat().replace("+00:00", "Z")
```

Use one small private formatter in each module where it removes duplication.

- [ ] **Step 4: Run feature provenance and CLI info tests**

```bash
env MNE_DONTWRITE_HOME=true /Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python \
  -m pytest tests/features/test_feature_provenance.py \
  tests/cli/test_cli_info_subject_status_optimizations.py \
  tests/features/test_feature_io_pac_outputs.py -q
```

Expected: PASS with no project-owned datetime deprecation warning.

- [ ] **Step 5: Commit timestamp modernization**

```bash
git add eeg_pipeline/domain/features/naming.py \
  eeg_pipeline/cli/commands/base_feature_availability.py \
  tests/features/test_feature_provenance.py \
  tests/cli/test_cli_info_subject_status_optimizations.py
git commit -m "fix: use timezone-aware feature timestamps"
```

### Task 9: Run Cross-Family and Repository Verification

**Files:**
- Verify all modified files.

- [ ] **Step 1: Run the complete feature and affected utility suites**

```bash
env MNE_DONTWRITE_HOME=true /Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python \
  -m pytest tests/features tests/pipelines/test_pipeline_features.py \
  tests/utils/test_feature_io_fail_fast.py \
  tests/utils/test_signal_metrics_complexity_entropy.py \
  tests/cli/test_cli_info_subject_status_optimizations.py -q
```

Expected: PASS.

- [ ] **Step 2: Run configured lint and maintainability gates**

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/ruff check \
  eeg_pipeline fmri_pipeline studies tests scripts local_workflows
make verify-maintainability
git diff --check dd22a9d..HEAD
```

Expected: all commands exit 0.

- [ ] **Step 3: Run the full repository suite**

```bash
env MNE_DONTWRITE_HOME=true /Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python \
  -m pytest
```

Expected: all baseline tests plus new regression tests pass. Review every warning and verify
that the project-owned feature datetime warnings are gone.

- [ ] **Step 4: Review the complete diff against the design**

Confirm that every production edit has a red-green regression, no user-owned ICA or study-log
file is present, no fallback or compatibility code was introduced, and no unrelated style
rewrite is included.

- [ ] **Step 5: Confirm the audit branch is clean**

Run `git status --short` and expect no output. If verification finds a defect, return to the
task that owns the affected code and repeat its red-green cycle before running Task 9 again.
