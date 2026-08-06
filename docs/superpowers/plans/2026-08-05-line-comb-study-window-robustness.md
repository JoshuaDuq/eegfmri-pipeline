# Line-Comb Study-Window Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make line-comb planning and validation use correct BIDS channel types, automatically
remove independently resolved comb-adjacent electrical lines, and guarantee artifact suppression
and signal preservation in every `Trig_therm/T  1` interval from -5 to +15 seconds.

**Architecture:** Read each recording with MNE-BIDS so `channels.tsv` is authoritative and only
scalp EEG enters EEG endpoints. Continue whole-run and overlapping 54-second evidence, add raw-data
20-second study-window evidence, and translate any accepted study/window source into narrow targets
on every continuous overlap-add window contributing samples to its evidence interval. Benchmark the
unchanged continuous transform both over the full run and over the exact study windows.

**Tech Stack:** Python 3.11+, NumPy, pandas, MNE-Python, MNE-BIDS, pytest, Ruff, Black.

---

## File structure

- `studies/pain_study/scripts/line_comb/remove.py`: BIDS loading, study-window extraction,
  evidence planning, continuous filtering, and benchmark orchestration.
- `studies/pain_study/analysis/line_comb/removal.py`: preservation-gate decisions only.
- `studies/pain_study/scripts/line_comb/config.yaml`: explicit event, epoch, and expected-trial
  contract.
- `tests/scripts/line_comb/test_remove.py`: unit tests for BIDS typing and study-window bounds.
- `tests/scripts/line_comb/test_run_plan.py`: unit tests for narrow adjacent targets and overlap
  propagation.
- `tests/analysis/line_comb/test_removal_gates.py`: tests proving study-window endpoints can fail.
- `docs/scanner_harmonic_removal.md` and the line-comb READMEs: scientific method and output
  interpretation.

The current working tree contains pre-existing line-comb work plus unrelated fMRI changes. All git
inspection and any eventual staging must be path-scoped; no worktree reset or broad staging is
allowed.

### Task 1: Make BIDS metadata authoritative

**Files:**
- Modify: `studies/pain_study/scripts/line_comb/remove.py`
- Test: `tests/scripts/line_comb/test_remove.py`

- [ ] **Step 1: Write failing tests**

  Add a minimal BIDS fixture whose BrainVision header describes every channel as EEG while
  `channels.tsv` declares `ECG` correctly. Assert that `read_bids_raw()` returns EEG picks that
  exclude `ECG`, retains the ECG channel as type `ecg`, and raises on a sidecar/header mismatch.

- [ ] **Step 2: Verify RED**

  Run:

  ```bash
  _MNE_FAKE_HOME_DIR=/tmp/codex-mne-home MPLCONFIGDIR=/tmp/codex-mpl \
    .venv/bin/python -m pytest tests/scripts/line_comb/test_remove.py \
    -k 'read_bids_raw or study_epoch' -q
  ```

  Expected: fail because `read_bids_raw` and the study-window contract do not exist.

- [ ] **Step 3: Implement strict loading and window bounds**

  Add `read_bids_raw(vhdr)` using:

  ```python
  bids_path = get_bids_path_from_fname(vhdr)
  return read_raw_bids(
      bids_path,
      extra_params={"preload": True},
      on_ch_mismatch="raise",
      verbose="ERROR",
  )
  ```

  Add settings for exact event name, `(-5.0, 15.0)` seconds, and 11 events per run. Implement
  `study_epoch_bounds(raw, settings)` with exact annotation matching, sample-index conversion,
  fixed stop-exclusive intervals, and fail-fast checks for count, bounds, and equal length.
  Replace every production `read_raw_brainvision` call in planning, benchmark, apply, round-trip
  verification, and cohort verification with `read_bids_raw`.

- [ ] **Step 4: Verify GREEN**

  Run the focused command from Step 2 and expect all selected tests to pass.

### Task 2: Detect and route comb-adjacent electrical lines

**Files:**
- Modify: `studies/pain_study/scripts/line_comb/remove.py`
- Test: `tests/scripts/line_comb/test_remove.py`
- Test: `tests/scripts/line_comb/test_run_plan.py`

- [ ] **Step 1: Write failing detector tests**

  Construct synthetic spectra containing a validated 1.2 Hz comb plus a distinct 10 dB narrow
  summit inside the existing 0.15 Hz residual-responsibility region. Assert:

  ```python
  assert adjacent == pytest.approx((27.72,))
  ```

  Also assert rejection when the summit is broad, below 10 dB, already covered by the parent
  target, outside 0.15 Hz, or inside the excluded mains range.

- [ ] **Step 2: Verify RED**

  Run:

  ```bash
  _MNE_FAKE_HOME_DIR=/tmp/codex-mne-home MPLCONFIGDIR=/tmp/codex-mpl \
    .venv/bin/python -m pytest tests/scripts/line_comb/test_remove.py \
    tests/scripts/line_comb/test_run_plan.py -k 'adjacent or overlap' -q
  ```

  Expected: fail because adjacent-line evidence and target provenance are absent.

- [ ] **Step 3: Implement the detector**

  Use only raw spectra. A candidate must be a local summit, meet the configured 10 dB
  channel-median prominence floor, have a 3 dB width no larger than 0.25 Hz, lie within
  `RESIDUAL_SEARCH_HZ` of an in-range fitted comb target, and lie outside that target's existing
  half-width or one spectrum bin, whichever is larger. Deduplicate candidates at the 27-second
  spectrum-fit resolution. Do not widen the parent notch and do not change gate thresholds.

- [ ] **Step 4: Add study-window evidence and interval routing**

  Extend `SessionRunSpectra` with study spectra and sample bounds. Add candidates from both
  adaptive 54-second windows and every -5/+15 second epoch. Cluster the same source within one
  line-claim distance and add its median frequency to every adaptive filtering window whose
  sample bounds overlap at least one supporting interval:

  ```python
  max(source_start, filter_start) < min(source_stop, filter_stop)
  ```

  Keep these targets distinct from ordinary isolated targets so their width remains
  `max(freq / 450, 0.05)` rather than the 0.337 Hz isolated-source width. Count them against the
  existing per-run source cap and fail if the cap is exceeded.

- [ ] **Step 5: Verify GREEN and regression safety**

  Run the focused command from Step 2, then:

  ```bash
  _MNE_FAKE_HOME_DIR=/tmp/codex-mne-home MPLCONFIGDIR=/tmp/codex-mpl \
    .venv/bin/python -m pytest tests/scripts/line_comb tests/analysis/line_comb -q
  ```

  Expected: all line-comb tests pass.

### Task 3: Gate the exact study samples

**Files:**
- Modify: `studies/pain_study/scripts/line_comb/remove.py`
- Modify: `studies/pain_study/analysis/line_comb/removal.py`
- Test: `tests/analysis/line_comb/test_removal_gates.py`
- Test: `tests/scripts/line_comb/test_remove.py`

- [ ] **Step 1: Write failing endpoint tests**

  Assert that `PreservationGate.evaluate()` fails independently when any of these exceed the
  existing corresponding full-run limit:

  ```python
  study_residual_excess_db = 1.01
  study_focal_residual_excess_db = 1.01
  study_max_probe_deviation_db = 0.51
  study_max_nonline_change_db = 0.21
  ```

  Assert that clean study-window values pass.

- [ ] **Step 2: Verify RED**

  Run:

  ```bash
  .venv/bin/python -m pytest tests/analysis/line_comb/test_removal_gates.py \
    -k study -q
  ```

  Expected: fail because study-window metrics are not evaluated.

- [ ] **Step 3: Implement study-window metrics**

  Build per-epoch channel-median and per-channel Hann spectra using the exact bounds from Task 1.
  For each epoch, audit the union of plan targets from continuous windows that overlap it. Reuse
  the existing matched local-control algorithms for aggregate and focal residual excess. In the
  benchmark, calculate probe preservation and non-line spectral change on only these epochs.
  Prefix every result with `study_` and add four explicit gate decisions without weakening any
  existing endpoint.

- [ ] **Step 4: Verify GREEN**

  Run the focused test from Step 2 and all line-comb tests. Expect all to pass.

### Task 4: Configuration, provenance, documentation, and real-data verification

**Files:**
- Modify: `studies/pain_study/scripts/line_comb/config.yaml`
- Modify: `docs/scanner_harmonic_removal.md`
- Modify: `studies/pain_study/analysis/line_comb/README.md`
- Modify: `studies/pain_study/scripts/line_comb/README.md`
- Modify: relevant provenance and report tests under `tests/scripts/line_comb/`

- [ ] **Step 1: Add explicit configuration**

  Store exactly:

  ```yaml
  study_event_name: "Trig_therm/T  1"
  study_epoch_s: [-5.0, 15.0]
  expected_study_events_per_run: 11
  ```

  Include these values in the settings fingerprint and fail on invalid ordering, non-finite
  bounds, or fewer than one expected event.

- [ ] **Step 2: Update scientific documentation**

  State that detection covers the continuous run and exact study epochs, filtering is continuous,
  ECG/EOG are preserved but excluded from EEG endpoints, adjacent sources need spectral narrowness
  plus channel-median spatial replication plus proximity to a validated electrical comb, and no
  benchmark threshold was moved.

- [ ] **Step 3: Run static and focused verification**

  ```bash
  .venv/bin/python -m ruff check studies/pain_study/analysis/line_comb \
    studies/pain_study/scripts/line_comb tests/analysis/line_comb tests/scripts/line_comb
  .venv/bin/python -m black --check studies/pain_study/analysis/line_comb \
    studies/pain_study/scripts/line_comb tests/analysis/line_comb tests/scripts/line_comb
  _MNE_FAKE_HOME_DIR=/tmp/codex-mne-home MPLCONFIGDIR=/tmp/codex-mpl \
    .venv/bin/python -m pytest tests/analysis/line_comb tests/scripts/line_comb -q
  ```

- [ ] **Step 4: Run the complete 90-recording benchmark**

  ```bash
  cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline
  _MNE_FAKE_HOME_DIR=/tmp/codex-mne-home MPLCONFIGDIR=/tmp/codex-mpl \
    .venv/bin/python -m studies.pain_study.scripts.line_comb.remove --stage benchmark
  ```

  Require 90 complete rows, every run-level gate true, the cohort seam randomization criterion
  true, and no partial output substituted for the authoritative table.

- [ ] **Step 5: Apply only the benchmarked immutable plans and verify**

  Archive the existing cleaned output directory to a timestamped sibling, run `--stage apply`,
  then `--stage verify`. Confirm 90 manifests, byte-identical sidecars, unchanged channel/sample
  geometry, and passing full-run and study-window residual reports.

- [ ] **Step 6: Produce the in-depth report**

  Report per-participant and per-run fundamentals, drift, isolated and adjacent sources, before/
  after prominence, band cost, preservation, study-window results, exclusions (none), scanner-
  artifact limitations, and all provenance digests. Cite the authoritative TSV/NPZ outputs and
  official MNE/MNE-BIDS methods.

## Self-review

- Spec coverage: BIDS typing, full-duration evidence, -5/+15 second evidence, continuous
  filtering, automatic adjacent-source removal, unchanged thresholds, immutable benchmark/apply,
  and final reporting are each assigned to a task.
- Placeholder scan: no implementation placeholder or deferred error handling remains.
- Type consistency: study intervals are stop-exclusive sample tuples throughout; adjacent targets
  remain separate from ordinary isolated targets throughout planning and width calculation.

