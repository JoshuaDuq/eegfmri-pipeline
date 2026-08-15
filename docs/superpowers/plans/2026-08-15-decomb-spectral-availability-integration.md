# Decomb Spectral Availability Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Decomb's recording-specific unavailable-frequency intervals an explicit, epoch-aligned analysis contract while preserving exact behavior when no manifest is configured.

**Architecture:** A NumPy-only domain model represents BIDS recording keys, merged closed frequency intervals, and epoch-aligned exclusions. A Decomb adapter validates provenance and translates the TSV into that model; the feature pipeline aligns it with `run_id`, threads it through shared intermediates, applies estimator-support-aware masking or contiguous-band eligibility at centralized computation seams, and writes one atomic subject audit.

**Tech Stack:** Python 3.11+, NumPy, SciPy, pandas, MNE-Python 1.12, PyYAML, pytest, Ruff.

---

## File structure

- Create `eeg_pipeline/spectral_availability/model.py`: immutable generic recording, interval, and epoch-aligned availability types.
- Create `eeg_pipeline/spectral_availability/estimators.py`: Welch, multitaper, and Morlet spectral half-support calculations.
- Create `eeg_pipeline/spectral_availability/decomb.py`: strict Decomb provenance and TSV parser.
- Create `eeg_pipeline/spectral_availability/alignment.py`: strict clean-event to BIDS-recording alignment.
- Create `eeg_pipeline/spectral_availability/audit.py`: idempotent audit collector and atomic TSV writer.
- Create `eeg_pipeline/spectral_availability/__init__.py`: intentionally small public API.
- Modify `eeg_pipeline/types.py`: attach eligibility and validity metadata to shared intermediates.
- Modify `eeg_pipeline/context/features.py`: carry epoch availability and the audit collector.
- Modify `eeg_pipeline/analysis/features/preparation.py`: pass availability into PSD and band precomputation.
- Modify `eeg_pipeline/utils/analysis/spectral.py`: mask Welch PSD and retained-width-normalize band power; invalidate overlapping Hilbert bands.
- Modify `eeg_pipeline/utils/analysis/tfr.py`: expose exact Morlet cycle metadata and mask unavailable TFR cells.
- Modify `eeg_pipeline/analysis/features/api.py`: preserve availability across family-specific and TFR computations.
- Modify `eeg_pipeline/analysis/features/spectral.py`: keep unavailable bins out of PSD descriptors, peaks, and IAF-like outputs.
- Modify `eeg_pipeline/analysis/features/aperiodic.py`: reject disconnected or insufficient retained fit support.
- Modify `eeg_pipeline/analysis/features/phase.py`: exclude ineligible epochs from phase/PAC/ITPC bands and cross-trial estimates.
- Modify `eeg_pipeline/analysis/features/bursts.py`: leave ineligible trial-band outputs unavailable.
- Modify `eeg_pipeline/analysis/features/connectivity.py`: exclude ineligible epochs before trial, condition, or subject connectivity.
- Modify `eeg_pipeline/analysis/features/precomputed/extras.py`: pass availability into PSD band ratios and asymmetry.
- Modify `eeg_pipeline/analysis/features/quality.py`: mask direct quality-feature PSD estimates.
- Modify `eeg_pipeline/analysis/features/source_localization.py`: enforce availability in source power, envelopes, TFR, and connectivity.
- Modify `eeg_pipeline/pipelines/features.py`: activate, cache, align, thread, audit, and write the contract.
- Modify `eeg_pipeline/pipelines/preprocessing.py`: enforce the no-second-notch contract before MNE preprocessing.
- Modify `eeg_pipeline/utils/data/preflight.py`: reject Decomb plus a downstream notch and validate configured provenance early.
- Modify `eeg_pipeline/utils/config/eeg_config.yaml`: document the explicit nullable manifest path.
- Create focused tests under `tests/spectral_availability/` and extend existing spectral, feature, pipeline, and preflight tests.

### Task 1: Generic availability model and estimator support

**Files:**
- Create: `eeg_pipeline/spectral_availability/model.py`
- Create: `eeg_pipeline/spectral_availability/estimators.py`
- Create: `eeg_pipeline/spectral_availability/__init__.py`
- Test: `tests/spectral_availability/test_model.py`
- Test: `tests/spectral_availability/test_estimators.py`

- [ ] **Step 1: Write failing model tests**

Cover strict key/interval validation, duplicate/overlap/touch merging, closed-interval overlap, per-epoch frequency masks, retained bandwidth, contiguous-band eligibility, and zero-support errors. The desired public API is:

```python
key = RecordingKey(subject="0001", task="thermalactive", run="1")
availability = EpochSpectralAvailability(
    recording_keys=(key, key),
    exclusions_by_epoch=(
        (FrequencyInterval(59.0, 61.0),),
        (FrequencyInterval(39.0, 41.0),),
    ),
)

valid = availability.valid_frequency_mask(
    np.array([10.0, 40.0, 60.0]),
    half_support_hz=np.array([0.5, 0.5, 0.5]),
)
assert valid.tolist() == [[True, True, False], [True, False, True]]
assert availability.contiguous_band_eligible(8.0, 13.0).tolist() == [True, True]
assert availability.contiguous_band_eligible(35.0, 45.0).tolist() == [True, False]
```

- [ ] **Step 2: Run the model tests and verify RED**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/spectral_availability/test_model.py -q`

Expected: collection fails because `eeg_pipeline.spectral_availability` does not exist.

- [ ] **Step 3: Implement the immutable model**

Implement frozen `RecordingKey`, `FrequencyInterval`, `RecordingExclusions`, and `EpochSpectralAvailability`. Validate non-empty canonical BIDS entities, finite non-negative interval edges, positive widths, exact epoch-axis lengths, finite strictly increasing frequency grids, and scalar-or-vector non-negative half-support. Merge duplicate, overlapping, and touching intervals in one constructor helper. Use the closed-overlap rule:

```python
invalid = (centres + half_support >= interval.low_hz) & (
    centres - half_support <= interval.high_hz
)
```

Expose `valid_frequency_mask`, `contiguous_band_eligible`, `intersections`, and `retained_bandwidth`. Do not import pandas, MNE, or Decomb.

- [ ] **Step 4: Run the model tests and verify GREEN**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/spectral_availability/test_model.py -q`

Expected: all model tests pass.

- [ ] **Step 5: Write failing estimator-support tests**

Test these exact contracts:

```python
assert multitaper_half_support(2.0) == pytest.approx(1.0)
assert morlet_half_support(np.array([10.0]), np.array([5.0])) == pytest.approx(
    np.array([10.0 / 5.0 * np.sqrt(np.log(2.0))])
)
assert welch_half_support(500.0, 1000, "hann") > 0
assert welch_half_support(500.0, 1000, "hann") < 1.0
assert multitaper_tfr_half_support(
    frequencies=np.array([10.0]),
    n_cycles=np.array([5.0]),
    time_bandwidth=4.0,
) == pytest.approx(np.array([4.0]))
```

Also reject non-finite bandwidths, mismatched Morlet arrays, invalid sampling rates, unsupported windows, and fewer than two Welch samples.

- [ ] **Step 6: Run estimator tests and verify RED**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/spectral_availability/test_estimators.py -q`

Expected: imports fail because estimator-support functions do not exist.

- [ ] **Step 7: Implement estimator-support calculations**

Use MNE's documented full multitaper bandwidth convention, the Morlet Gaussian power half-height expression, and `scipy.signal.get_window` plus a deterministic zero-padded DTFT main-lobe search for Welch. Raise when the first half-power crossing cannot be measured. Return half-support in Hz, never a bin count.

- [ ] **Step 8: Run Task 1 verification and commit**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/spectral_availability/test_model.py \
  tests/spectral_availability/test_estimators.py -q
.venv/bin/ruff check eeg_pipeline/spectral_availability tests/spectral_availability
```

Expected: all tests pass and Ruff reports no errors.

Commit:

```bash
git add eeg_pipeline/spectral_availability tests/spectral_availability
git commit -m "feat: add spectral availability domain model"
```

### Task 2: Strict Decomb ingestion, alignment, and activation validation

**Files:**
- Create: `eeg_pipeline/spectral_availability/decomb.py`
- Create: `eeg_pipeline/spectral_availability/alignment.py`
- Modify: `eeg_pipeline/spectral_availability/__init__.py`
- Modify: `eeg_pipeline/utils/data/preflight.py`
- Modify: `eeg_pipeline/utils/config/eeg_config.yaml`
- Modify: `eeg_pipeline/pipelines/preprocessing.py`
- Test: `tests/spectral_availability/test_decomb.py`
- Test: `tests/spectral_availability/test_alignment.py`
- Modify: `tests/utils/test_study_preflight.py`
- Modify: `tests/config/test_scientific_defaults.py`
- Modify: `tests/pipelines/test_pipeline_preprocessing.py`

- [ ] **Step 1: Write failing Decomb adapter tests**

Build temporary `dataset_description.json` and TSV fixtures. Require exactly one `GeneratedBy` item whose `Name` case-insensitively equals `decomb`, all required columns, one `outcome=no_line_detected` terminal-null row per recording, paired interval fields, finite positive geometry, and parseable subject/task/run/optional-session entities. Verify all finite rows from all removal rounds are deduplicated and merged and that `sha256` equals the file bytes.

- [ ] **Step 2: Run adapter tests and verify RED**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/spectral_availability/test_decomb.py -q`

Expected: imports fail because the Decomb adapter does not exist.

- [ ] **Step 3: Implement strict Decomb loading**

Define:

```python
@dataclass(frozen=True)
class DecombManifest:
    path: Path
    sha256: str
    exclusions: tuple[RecordingExclusions, ...]

def load_decomb_manifest(path: str | Path) -> DecombManifest: ...
```

Read JSON with `json.loads`, TSV with `pandas.read_csv(sep="\t", dtype=str, keep_default_na=False)`, and hash with `hashlib.sha256(path.read_bytes())`. Surface JSON, TSV, provenance, entity, and geometry errors with recording/row context. Do not search for alternate files or infer malformed values.

- [ ] **Step 4: Run adapter tests and verify GREEN**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/spectral_availability/test_decomb.py -q`

Expected: all adapter tests pass.

- [ ] **Step 5: Write failing alignment tests**

Test float `run_id` values `1.0` through `6.0`, rejection of missing/fractional/infinite runs, optional `session_id`, duplicate/ambiguous keys, unmatched epochs, and allowance for extra manifest recordings outside the selected subject. Desired API:

```python
aligned = align_decomb_to_epochs(
    manifest,
    subject="0001",
    task="thermalactive",
    events=pd.DataFrame({"run_id": [1.0, 2.0]}),
)
assert [key.run for key in aligned.recording_keys] == ["1", "2"]
```

- [ ] **Step 6: Run alignment tests and verify RED**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/spectral_availability/test_alignment.py -q`

Expected: import fails because alignment is not implemented.

- [ ] **Step 7: Implement exact epoch alignment**

Canonicalize only finite integer-valued numeric identifiers, strip one BIDS entity prefix from explicit strings, and require session identity when matching manifest candidates contain sessions. Construct one exclusion tuple per event row in original order. Raise for every ambiguity or mismatch; do not use event order, participant unions, or cohort unions.

- [ ] **Step 8: Write failing activation/preflight tests**

Add tests proving:

```python
config["paths.decomb_manifest"] = str(manifest_path)
config["preprocessing.notch_freq"] = 60
report = run_preflight(config)
assert _by_key(report)["preprocessing.notch_freq"].status == "error"
```

Also test missing manifest, invalid provenance, `notch_freq=None`, and the packaged default `paths.decomb_manifest is None`.

- [ ] **Step 9: Run activation tests and verify RED**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/utils/test_study_preflight.py \
  tests/config/test_scientific_defaults.py \
  tests/pipelines/test_pipeline_preprocessing.py -q
```

Expected: the new Decomb assertions fail.

- [ ] **Step 10: Implement explicit activation and preflight**

Add `paths.decomb_manifest: null` with concise documentation. In preflight, do nothing when the value is null; otherwise require a file, require `preprocessing.notch_freq is None`, and call the strict adapter so provenance/manifest errors surface before processing. Apply the same validation at the start of `PreprocessingPipeline.process_subject` so direct preprocessing cannot bypass the contract. Keep imports local to the configured branch and do not catch adapter errors into warnings.

- [ ] **Step 11: Run Task 2 verification and commit**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/spectral_availability/test_decomb.py \
  tests/spectral_availability/test_alignment.py \
  tests/utils/test_study_preflight.py \
  tests/config/test_scientific_defaults.py \
  tests/pipelines/test_pipeline_preprocessing.py -q
.venv/bin/ruff check \
  eeg_pipeline/spectral_availability \
  eeg_pipeline/utils/data/preflight.py \
  eeg_pipeline/pipelines/preprocessing.py \
  tests/spectral_availability \
  tests/utils/test_study_preflight.py
```

Expected: all focused tests pass and Ruff reports no errors.

Commit:

```bash
git add \
  eeg_pipeline/spectral_availability \
  eeg_pipeline/utils/data/preflight.py \
  eeg_pipeline/pipelines/preprocessing.py \
  eeg_pipeline/utils/config/eeg_config.yaml \
  tests/spectral_availability \
  tests/utils/test_study_preflight.py \
  tests/config/test_scientific_defaults.py \
  tests/pipelines/test_pipeline_preprocessing.py
git commit -m "feat: load and align Decomb exclusions"
```

### Task 3: Shared PSD and contiguous-band intermediates

**Files:**
- Modify: `eeg_pipeline/types.py`
- Modify: `eeg_pipeline/context/features.py`
- Modify: `eeg_pipeline/utils/analysis/spectral.py`
- Modify: `eeg_pipeline/analysis/features/preparation.py`
- Modify: `eeg_pipeline/analysis/features/precomputed/extras.py`
- Test: `tests/spectral_availability/test_shared_intermediates.py`
- Modify: `tests/features/test_feature_data_contracts.py`
- Modify: `tests/features/test_spectral_strict_failures.py`

- [ ] **Step 1: Write failing shared-type and crop tests**

Require `FeatureContext.spectral_availability`, `PrecomputedData.spectral_availability`, `BandData.eligible_epochs`, and `PSDData.valid_frequency_mask`. Verify crop/with-window operations preserve epoch availability and eligibility while clearing time-dependent PSD exactly as before.

- [ ] **Step 2: Run type tests and verify RED**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/features/test_feature_data_contracts.py \
  tests/spectral_availability/test_shared_intermediates.py -q
```

Expected: constructors reject the new keywords or fields are absent.

- [ ] **Step 3: Add optional shared metadata fields**

Add fields with `None` defaults so no-manifest constructor behavior and array values are unchanged. Validate new masks against epoch/frequency axes only when present. Preserve them in `BandData.crop`, `PrecomputedData.crop`, and all internal clone constructors.

- [ ] **Step 4: Write failing Welch PSD masking tests**

Generate two epochs with different exclusions and assert `compute_psd(..., spectral_availability=...)` returns the same frequency grid as the unmasked call, stores a `(epochs, freqs)` mask, and sets only invalid epoch/channel/frequency cells to `NaN`. Assert no-manifest output is bitwise equal to the existing call.

- [ ] **Step 5: Run Welch tests and verify RED**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/spectral_availability/test_shared_intermediates.py -k welch -q`

Expected: `compute_psd` does not accept the availability argument.

- [ ] **Step 6: Implement estimator-aware Welch masking**

Resolve `n_per_seg` from the exact call (`n_fft` in the shared Welch path), compute its half-power support from the configured window, build the per-epoch mask, replace invalid cells with `NaN`, and store support metadata in `PSDData`. When availability is absent, do not calculate support or copy/mask the PSD.

- [ ] **Step 7: Write failing retained-bandwidth tests**

Use monkeypatched deterministic PSD arrays and irregular frequencies. For each epoch, assert integration uses `sum(psd * weight)` over valid bins and normalization divides by that epoch's retained `sum(weight)`, not nominal width. Assert one exhausted epoch returns `NaN`, all exhausted epochs raise, and fixed configured line masks combine with manifest masks.

- [ ] **Step 8: Run bandpower tests and verify RED**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/spectral_availability/test_shared_intermediates.py -k bandpower -q`

Expected: current bandpower uses one global mask and denominator.

- [ ] **Step 9: Implement retained-width PSD band power**

Add keyword-only `spectral_availability=None` to `compute_psd_bandpower`. Use multitaper `bandwidth / 2` or measured Welch support, combine per-epoch validity with any configured fixed line mask, integrate with `compute_frequency_weights`, divide by per-epoch retained width, and raise only when no epoch has support for a requested band. Keep the return dictionary and no-manifest values unchanged.

- [ ] **Step 10: Write failing contiguous-band tests**

Assert `compute_band_data` filters only eligible epochs, fills all arrays for ineligible epochs with `NaN`, stores `eligible_epochs`, and raises if every epoch overlaps the requested band. Assert a non-overlapping band is numerically identical to the no-manifest call.

- [ ] **Step 11: Run contiguous-band tests and verify RED**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/spectral_availability/test_shared_intermediates.py -k band_data -q`

Expected: current Hilbert computation processes every epoch.

- [ ] **Step 12: Implement and thread shared availability**

Add keyword-only availability parameters to `compute_band_data`, `_compute_single_band`, `_compute_and_store_bands`, `_compute_psd_with_qc`, and `precompute_data`. Store the same epoch-aligned object on `PrecomputedData`; pass it to ratios/asymmetry PSD bandpower. Filtering only eligible epoch slices avoids representing notch attenuation as a computed oscillatory feature.

- [ ] **Step 13: Run Task 3 verification and commit**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/spectral_availability/test_shared_intermediates.py \
  tests/features/test_feature_data_contracts.py \
  tests/features/test_spectral_strict_failures.py -q
.venv/bin/ruff check \
  eeg_pipeline/types.py \
  eeg_pipeline/context/features.py \
  eeg_pipeline/utils/analysis/spectral.py \
  eeg_pipeline/analysis/features/preparation.py \
  eeg_pipeline/analysis/features/precomputed/extras.py \
  tests/spectral_availability/test_shared_intermediates.py
```

Expected: all focused tests pass and Ruff reports no errors.

Commit:

```bash
git add \
  eeg_pipeline/types.py \
  eeg_pipeline/context/features.py \
  eeg_pipeline/utils/analysis/spectral.py \
  eeg_pipeline/analysis/features/preparation.py \
  eeg_pipeline/analysis/features/precomputed/extras.py \
  tests/spectral_availability/test_shared_intermediates.py \
  tests/features/test_feature_data_contracts.py \
  tests/features/test_spectral_strict_failures.py
git commit -m "feat: enforce availability in shared spectral data"
```

### Task 4: TFR, peaks, aperiodic, phase, burst, and connectivity consumers

**Files:**
- Modify: `eeg_pipeline/utils/analysis/tfr.py`
- Modify: `eeg_pipeline/analysis/features/api.py`
- Modify: `eeg_pipeline/analysis/features/spectral.py`
- Modify: `eeg_pipeline/analysis/features/aperiodic.py`
- Modify: `eeg_pipeline/analysis/features/phase.py`
- Modify: `eeg_pipeline/analysis/features/bursts.py`
- Modify: `eeg_pipeline/analysis/features/connectivity.py`
- Modify: `eeg_pipeline/analysis/features/quality.py`
- Modify: `eeg_pipeline/analysis/features/source_localization.py`
- Test: `tests/spectral_availability/test_tfr_consumers.py`
- Test: `tests/spectral_availability/test_contiguous_consumers.py`
- Modify: `tests/features/test_feature_aperiodic_periodic_peaks.py`
- Modify: `tests/features/test_feature_connectivity_validity_guards.py`
- Modify: `tests/features/test_feature_source_connectivity_validity.py`

- [ ] **Step 1: Write failing TFR masking tests**

Use an `EpochsTFRArray` with known frequencies and adaptive `n_cycles`. Assert a helper applies Morlet half-support per frequency, preserves the 4-D shape, writes `NaN` only to invalid epoch/frequency cells, exposes the validity mask and contributing epoch counts, and raises for averaged 3-D TFR data when recording-specific availability is active.

- [ ] **Step 2: Run TFR tests and verify RED**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/spectral_availability/test_tfr_consumers.py -k mask -q`

Expected: no TFR availability helper exists.

- [ ] **Step 3: Implement exact TFR metadata and masking**

Keep the exact post-length-filtering `freqs` and `n_cycles` used by MNE together. Apply the Morlet half-power mask immediately after each power or complex TFR computation and after reuse/crop paths. Store private provenance attributes only on the in-memory TFR object; the scientific output remains explicit `NaN` plus validity/count metadata consumed by feature extraction.

- [ ] **Step 4: Write failing PSD descriptor and aperiodic tests**

Assert peak/IAF/centre/bandwidth/entropy never select masked bins. Assert aperiodic fitting accepts retained contiguous support, returns unavailable for an internal masked gap or too few points, and does not interpolate. Test epoch-specific masks, not a cohort union.

- [ ] **Step 5: Run spectral-model tests and verify RED**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/spectral_availability/test_tfr_consumers.py -k 'peak or aperiodic' \
  tests/features/test_feature_aperiodic_periodic_peaks.py -q
```

Expected: masked support is not yet enforced consistently.

- [ ] **Step 6: Enforce retained support in spectral models**

In direct PSD extraction, including quality and source-ROI power paths, apply the same estimator-support mask before all channel/global/ROI reductions. Make descriptor helpers select finite retained bins. Before each aperiodic/specparam fit, require retained indices to be consecutive within the requested fit range and retain existing minimum point/range requirements; return the existing unavailable representation and record the reason rather than filling gaps.

- [ ] **Step 7: Write failing contiguous-consumer tests**

Construct two-run synthetic precomputed/TFR inputs where only run 2 overlaps alpha. Assert:

- burst and Hilbert PAC trial rows for run 2 are `NaN`;
- band-level phase/ITPC excludes run 2 from cross-trial estimation and reports the eligible count;
- trial connectivity leaves run 2 unavailable;
- condition/subject connectivity computes only from eligible epochs and enforces its existing minimum count after selection;
- source envelopes/connectivity exclude run 2 while source PSD uses retained-bin integration;
- all-overlap inputs raise instead of returning attenuated numbers.

- [ ] **Step 8: Run contiguous-consumer tests and verify RED**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/spectral_availability/test_contiguous_consumers.py \
  tests/features/test_feature_connectivity_validity_guards.py \
  tests/features/test_feature_source_connectivity_validity.py -q
```

Expected: current cross-trial paths include every epoch.

- [ ] **Step 9: Enforce band eligibility at consumer boundaries**

Use `BandData.eligible_epochs` for Hilbert-derived burst/PAC/connectivity paths and `EpochSpectralAvailability.contiguous_band_eligible` for band reductions of complex TFR phase, source envelopes, and source connectivity. Source ROI PSD power uses retained-bin integration rather than whole-band rejection. Preserve trial axis rows with `NaN`; subset only the estimator input for cross-trial calculations, then broadcast/align results using original epoch indices. Re-run existing minimum-epoch guards after eligibility selection. Never silently substitute a different band.

- [ ] **Step 10: Run Task 4 verification and commit**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/spectral_availability/test_tfr_consumers.py \
  tests/spectral_availability/test_contiguous_consumers.py \
  tests/features/test_feature_aperiodic_periodic_peaks.py \
  tests/features/test_feature_connectivity_validity_guards.py \
  tests/features/test_feature_source_connectivity_validity.py \
  tests/features/test_feature_scientific_validity_guards.py -q
.venv/bin/ruff check \
  eeg_pipeline/utils/analysis/tfr.py \
  eeg_pipeline/analysis/features/api.py \
  eeg_pipeline/analysis/features/spectral.py \
  eeg_pipeline/analysis/features/aperiodic.py \
  eeg_pipeline/analysis/features/phase.py \
  eeg_pipeline/analysis/features/bursts.py \
  eeg_pipeline/analysis/features/connectivity.py \
  eeg_pipeline/analysis/features/quality.py \
  eeg_pipeline/analysis/features/source_localization.py \
  tests/spectral_availability/test_tfr_consumers.py \
  tests/spectral_availability/test_contiguous_consumers.py
```

Expected: all focused tests pass and Ruff reports no errors.

Commit:

```bash
git add \
  eeg_pipeline/utils/analysis/tfr.py \
  eeg_pipeline/analysis/features/api.py \
  eeg_pipeline/analysis/features/spectral.py \
  eeg_pipeline/analysis/features/aperiodic.py \
  eeg_pipeline/analysis/features/phase.py \
  eeg_pipeline/analysis/features/bursts.py \
  eeg_pipeline/analysis/features/connectivity.py \
  eeg_pipeline/analysis/features/quality.py \
  eeg_pipeline/analysis/features/source_localization.py \
  tests/spectral_availability/test_tfr_consumers.py \
  tests/spectral_availability/test_contiguous_consumers.py \
  tests/features/test_feature_aperiodic_periodic_peaks.py \
  tests/features/test_feature_connectivity_validity_guards.py \
  tests/features/test_feature_source_connectivity_validity.py
git commit -m "feat: mask unavailable spectral feature support"
```

### Task 5: Pipeline activation, audit, integration, and compatibility

**Files:**
- Create: `eeg_pipeline/spectral_availability/audit.py`
- Modify: `eeg_pipeline/spectral_availability/__init__.py`
- Modify: `eeg_pipeline/pipelines/features.py`
- Modify: `eeg_pipeline/analysis/features/api.py`
- Modify: `eeg_pipeline/analysis/features/preparation.py`
- Test: `tests/spectral_availability/test_audit.py`
- Test: `tests/spectral_availability/test_pipeline_integration.py`
- Modify: `tests/pipelines/test_pipeline_features.py`

- [ ] **Step 1: Write failing audit tests**

Require one row per recording and analysis target with subject/task/run/session, intersecting intervals, nominal/retained bandwidth, retained share, estimator/support rule, PSD and contiguous eligibility, aligned/eligible/ineligible epoch counts, and manifest checksum. Verify idempotent registration of an identical target, rejection of conflicting duplicate rows, deterministic sort order, and atomic replacement without a residual temporary file.

- [ ] **Step 2: Run audit tests and verify RED**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/spectral_availability/test_audit.py -q`

Expected: audit types do not exist.

- [ ] **Step 3: Implement the subject audit collector**

Define a focused collector whose registration methods accept the actual estimator grid/mask or contiguous band eligibility. Aggregate epoch rows by `RecordingKey`, serialize intervals deterministically, and treat a repeated identical registration as one output row while raising on conflicting content. Write through `tempfile.NamedTemporaryFile` in the destination directory followed by `os.replace`.

- [ ] **Step 4: Write failing two-run pipeline integration tests**

Create temporary Decomb provenance/manifest, two-run clean events with float `run_id`, lightweight synthetic epochs, one broad PSD band, one TFR grid, and one alpha Hilbert/connectivity request. Assert exact run-specific masks, retained-width values, rejected contiguous run, and the final file name:

```text
sub-0001_task-thermalactive_desc-spectralavailability.tsv
```

Also monkeypatch the Decomb loader and prove it is not called and existing outputs are unchanged when `paths.decomb_manifest` is absent or null.

- [ ] **Step 5: Run integration tests and verify RED**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/spectral_availability/test_pipeline_integration.py \
  tests/pipelines/test_pipeline_features.py -q
```

Expected: the feature pipeline neither loads nor writes spectral availability.

- [ ] **Step 6: Wire activation once at the pipeline boundary**

In `FeaturePipeline`, keep a per-instance manifest cache. After strict epoch/event loading and before shared TFR/precompute work, locally import the Decomb adapter only when the configured path is non-null, validate the no-notch contract, align all epochs, create one audit collector, and pass both through every full-range and per-range context. Write the audit once after all requested ranges succeed. Do not change existing feature-table schemas.

- [ ] **Step 7: Register actual analysis targets**

Have shared PSD, TFR, and contiguous-band seams register their actual frequency grid/band, estimator type, support rule, per-epoch validity, and counts with the collector. Include range/family identity in the analysis-target key so different requested outputs cannot collide. Add manifest path/checksum to extraction provenance so non-spectral outputs still identify their filtered source; non-spectral features add no masking target and continue to consume the already filtered samples.

- [ ] **Step 8: Validate the mounted 90-recording structure read-only**

Run a read-only script through the production adapter and aligner against:

```text
/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg_decombed_auto/line_notch_manifest.tsv
/Volumes/KINGSTON/EEG_fMRI_data/derivatives/preprocessed/eeg
```

Assert 90 manifest keys, one terminal-null row per key, and exact subject/run coverage for the 15 clean-events tables. Do not use these paths in automated tests and do not rewrite current derivatives because they contain an additional 60 Hz notch.

- [ ] **Step 9: Run compatibility and full verification**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/spectral_availability \
  tests/features \
  tests/pipelines/test_pipeline_features.py \
  tests/utils/test_study_preflight.py \
  tests/config/test_scientific_defaults.py -q
.venv/bin/ruff check \
  eeg_pipeline/spectral_availability \
  eeg_pipeline/types.py \
  eeg_pipeline/context/features.py \
  eeg_pipeline/utils/analysis/spectral.py \
  eeg_pipeline/utils/analysis/tfr.py \
  eeg_pipeline/analysis/features/preparation.py \
  eeg_pipeline/analysis/features/api.py \
  eeg_pipeline/analysis/features/spectral.py \
  eeg_pipeline/analysis/features/aperiodic.py \
  eeg_pipeline/analysis/features/phase.py \
  eeg_pipeline/analysis/features/bursts.py \
  eeg_pipeline/analysis/features/connectivity.py \
  eeg_pipeline/analysis/features/quality.py \
  eeg_pipeline/analysis/features/source_localization.py \
  eeg_pipeline/analysis/features/precomputed/extras.py \
  eeg_pipeline/pipelines/features.py \
  eeg_pipeline/utils/data/preflight.py \
  eeg_pipeline/pipelines/preprocessing.py \
  tests/spectral_availability
git diff --check
```

Expected: all focused/feature tests pass, Ruff reports no errors, and the diff has no whitespace errors.

- [ ] **Step 10: Commit the integrated feature**

```bash
git add \
  eeg_pipeline/spectral_availability \
  eeg_pipeline/types.py \
  eeg_pipeline/context/features.py \
  eeg_pipeline/utils/analysis/spectral.py \
  eeg_pipeline/utils/analysis/tfr.py \
  eeg_pipeline/analysis/features/preparation.py \
  eeg_pipeline/analysis/features/api.py \
  eeg_pipeline/analysis/features/spectral.py \
  eeg_pipeline/analysis/features/aperiodic.py \
  eeg_pipeline/analysis/features/phase.py \
  eeg_pipeline/analysis/features/bursts.py \
  eeg_pipeline/analysis/features/connectivity.py \
  eeg_pipeline/analysis/features/quality.py \
  eeg_pipeline/analysis/features/source_localization.py \
  eeg_pipeline/analysis/features/precomputed/extras.py \
  eeg_pipeline/pipelines/features.py \
  eeg_pipeline/utils/data/preflight.py \
  eeg_pipeline/pipelines/preprocessing.py \
  eeg_pipeline/utils/config/eeg_config.yaml \
  tests/spectral_availability \
  tests/features \
  tests/pipelines/test_pipeline_features.py \
  tests/utils/test_study_preflight.py \
  tests/config/test_scientific_defaults.py
git commit -m "feat: integrate Decomb spectral availability"
```

- [ ] **Step 11: Run final branch verification**

Run:

```bash
MPLBACKEND=Agg MNE_DONTWRITE_HOME=true PYTHONPATH=. .venv/bin/python -m pytest -q
.venv/bin/ruff check eeg_pipeline tests
git status --short
```

Expected: the full suite passes, Ruff reports no errors, and only intentional ignored/runtime artifacts may remain untracked.
