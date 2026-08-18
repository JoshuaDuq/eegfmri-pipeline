# Scanner Evidence Relocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move every scanner- and fMRI-derived measurement out of `eeg_pipeline/` into `studies/pain_study/`, so the shared MNE preprocessing pipeline and its report describe EEG alone.

**Architecture:** Relocation, not deletion. Measurement halves move to `studies/pain_study/analysis/`, I/O and plotting to `studies/pain_study/scripts/`, commands to `studies/pain_study/cli/` — the shape `line_comb` and `cardiac_gaps` already use. Report-rendering halves are deleted, because the study side draws PNGs and TSVs instead of MNE report sections. Two capabilities are *generalized* rather than moved: the aperiodic comb exclusion becomes a config-supplied window list, and the ECG beat-marker path becomes a configurable marker name.

**Tech Stack:** Python 3, MNE-Python, MNE-BIDS-Pipeline, NumPy, pandas, Matplotlib, pytest.

**Spec:** [`docs/superpowers/specs/2026-08-17-scanner-evidence-relocation-design.md`](../specs/2026-08-17-scanner-evidence-relocation-design.md)

## Global Constraints

- **No docstrings in code this plan writes.** New modules, classes, functions and tests get none. Where a line genuinely needs explaining — a threshold, an ordering, a decision that will look wrong later — use a short `#` comment above it. This applies to code *authored* here; a file that is moved keeps the contents it already had, because a move is a move. Every code block below already follows this rule; do not reintroduce docstrings to match surrounding style.
- **Never run the full pytest suite.** It takes ~9 minutes. Run the targeted subsets named in each task.
- **Work directly in the main checkout.** Do not create git worktrees.
- `test_config_loader_paths` fails at HEAD in this environment because of external-drive paths. That is pre-existing and not a regression — ignore it.
- **Core must never import from `studies/`.** Enforced by `test_core_does_not_import_the_study`.
- **Two tasks are behaviour-preserving and must be proven so:** Task 3 (aperiodic exponents bit-identical) and Task 8 (beat detections identical). Everything else is code and config changing address.
- Reports present measurements, not pass/fail verdicts from invented thresholds. Do not add thresholds to any moved code.
- Bulk outputs go on the external drive, never into the repo.

---

## File Structure

**Created under `studies/pain_study/`:**

| Path | Responsibility |
|---|---|
| `analysis/gradient/__init__.py` | Package marker |
| `analysis/gradient/comb.py` | Comb residual measurement — volume timing, per-channel excess |
| `analysis/gradient/locked.py` | Volume-locked average and its floor-corrected amplitude |
| `analysis/gradient/cohort.py` | Cohort pooling of the comb across participants |
| `analysis/noise_floor.py` | Odd-even averaging-floor estimator, shared by `gradient/` and `bcg/` |
| `scripts/gradient/__init__.py` | Package marker |
| `scripts/gradient/plot.py` | The three figures, written as PNG |
| `scripts/gradient/tables.py` | The five tables, written as TSV |
| `scripts/gradient/config.yaml` | Comb range, Welch window, TR tolerance, volume marker, harmonic QC |
| `scripts/gradient/README.md` | What the measurements are and are not |
| `scripts/bcg/config.yaml` | Analyzer block, BCG windows, pulse marker, marker-per-volume reference |
| `cli/gradient.py` | `eeg-pipeline gradient <mode>` |

**Created under `eeg_pipeline/`:**

| Path | Responsibility |
|---|---|
| `preprocessing/report/rr_intervals.py` | Beat-to-beat interval evidence, extracted from `analyzer_qc.py` |

**Deleted from `eeg_pipeline/`:** `utils/config/acquisition.py`, `analysis/qc/` (whole package), `cli/commands/harmonics{,_parser,_orchestrator}.py`, `utils/config/presets/eeg_only.yaml`.

---

## Phase 0 — The ECG coupling gate

### Task 1: Gate ECG coupling on the ECG channel, not on the scanner

`utils/data/preprocessing.py` disables the ECG coupling QC when `preprocessing.eeg_fmri` is false. ECG coupling correlates EEG against a recorded ECG lead — it needs the lead, never a scanner. `coherence.py:228` documents this as issue #14 and checks `eeg.ecg_channels`; that fix was never applied here.

**Files:**
- Modify: `eeg_pipeline/utils/data/preprocessing.py:501`, `:575`
- Test: `tests/preprocessing/test_clean_events_qc_gating.py` (create)

**Interfaces:**
- Consumes: `get_config_value` (already imported at `utils/data/preprocessing.py:32`)
- Produces: nothing new — behaviour change only

- [ ] **Step 1: Write the failing test**

Create `tests/preprocessing/test_clean_events_qc_gating.py`:

```python
# ECG coupling needs a recorded lead, not a scanner.

from __future__ import annotations

from eeg_pipeline.utils.data.preprocessing import CleanEventsQCConfig


def _config(*, eeg_fmri: bool, ecg_channels: list[str]) -> dict:
    return {
        "preprocessing": {
            "eeg_fmri": eeg_fmri,
            "clean_events_qc": {
                "enabled": True,
                "ecg_coupling": {"enabled": True, "channels": ["ECG"]},
                "peripheral_low_gamma": {"enabled": False},
            },
        },
        "eeg": {"ecg_channels": ecg_channels},
    }


def test_ecg_coupling_runs_outside_a_scanner_when_a_lead_is_named():
    config = _config(eeg_fmri=False, ecg_channels=["ECG"])
    parsed = CleanEventsQCConfig.from_config(config)
    assert parsed.ecg_coupling.enabled is True


def test_ecg_coupling_is_disabled_when_no_lead_is_named():
    config = _config(eeg_fmri=True, ecg_channels=[])
    parsed = CleanEventsQCConfig.from_config(config)
    assert parsed.ecg_coupling.enabled is False
```

If `CleanEventsQCConfig` is not the class name or `from_config` is not the constructor, read `eeg_pipeline/utils/data/preprocessing.py` around line 480 and use the real names. Do not invent them.

- [ ] **Step 2: Run the test and confirm the first case fails**

```bash
python -m pytest tests/preprocessing/test_clean_events_qc_gating.py -v
```

Expected: `test_ecg_coupling_runs_outside_a_scanner_when_a_lead_is_named` FAILS (asserts True, gets False). The second test passes only by accident today — it will still pass after the fix, for the right reason.

- [ ] **Step 3: Replace the scanner gate with the channel gate**

At `eeg_pipeline/utils/data/preprocessing.py:501`, replace:

```python
        ecg_coupling_enabled = bool(ecg_raw.get("enabled", True)) and is_eeg_fmri(config)
```

with:

```python
        # The ECG coupling metric correlates each EEG channel against a recorded ECG
        # lead. What it needs is the lead. Gated on the scanner declaration until now,
        # which switched it off for anyone recording ECG outside a bore -- issue #14,
        # fixed in config coherence and missed here. Left on with no channel named, it
        # fails at ``pick_channels`` after PyPREP, ICA and epoching have already run,
        # so the guard stays; only its question changes.
        ecg_coupling_enabled = bool(ecg_raw.get("enabled", True)) and bool(
            get_config_value(config, "eeg.ecg_channels", None)
        )
```

At `:575`, replace:

```python
            if not is_eeg_fmri(config) and bool(ecg_raw.get("enabled", True)):
                return replace(cfg, enabled=False)
```

with:

```python
            if not get_config_value(config, "eeg.ecg_channels", None) and bool(
                ecg_raw.get("enabled", True)
            ):
                return replace(cfg, enabled=False)
```

Remove the now-unused `from eeg_pipeline.utils.config.acquisition import is_eeg_fmri` import at line 31 **only if** no other use of `is_eeg_fmri` remains in the file. Check with:

```bash
grep -n "is_eeg_fmri" eeg_pipeline/utils/data/preprocessing.py
```

- [ ] **Step 4: Run the tests and confirm they pass**

```bash
python -m pytest tests/preprocessing/test_clean_events_qc_gating.py tests/preprocessing/test_clean_events_alignment.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/preprocessing/test_clean_events_qc_gating.py eeg_pipeline/utils/data/preprocessing.py
git commit -m "fix(qc): gate ECG coupling on the recorded lead, not on the scanner

Issue #14 was fixed in config coherence and missed in the config parser,
so a montage with an ECG channel and no scanner had its coupling metric
switched off underneath it."
```

---

## Phase 1 — Aperiodic exclusion, then the gradient move

### Task 2: Add `report.analysis.aperiodic_exclude_hz`, honoured alongside the existing exclusions

`compute_run_spectra` already withholds notch stopbands and decomb-unavailable intervals from the aperiodic fit. This adds a third, config-supplied source. `gradient_windows` stays for now — this task is purely additive, so nothing changes until Task 3.

**Files:**
- Modify: `eeg_pipeline/preprocessing/report/spectra.py:271-303`
- Modify: `eeg_pipeline/preprocessing/report/settings.py`
- Modify: `eeg_pipeline/utils/config/eeg_config.yaml` (the `report.analysis` block)
- Modify: `eeg_pipeline/preprocessing/report/run_evidence.py:242` (pass the new setting through)
- Test: `tests/preprocessing/test_report_spectra.py`

**Interfaces:**
- Produces: `ReportSettings.aperiodic_exclude_hz: tuple[tuple[float, float], ...]`, default `()`
- Produces: `compute_run_spectra(..., aperiodic_exclude_hz: Sequence[tuple[float, float]] = ())`

- [ ] **Step 1: Write the failing tests**

Append to `tests/preprocessing/test_report_spectra.py`:

```python
def test_aperiodic_exclude_hz_withholds_a_named_window():
    raw = _raw(focal_channels=(), n_channels=8, exponent=1.0)
    baseline = compute_run_spectra(raw, raw, recording_id="sub-01_run-1", fmax=100.0)
    excluded = compute_run_spectra(
        raw,
        raw,
        recording_id="sub-01_run-1",
        fmax=100.0,
        aperiodic_exclude_hz=((20.0, 25.0),),
    )
    assert excluded.before.aperiodic.exponent != baseline.before.aperiodic.exponent


def test_aperiodic_exclude_hz_empty_leaves_the_fit_untouched():
    raw = _raw(focal_channels=(), n_channels=8, exponent=1.0)
    baseline = compute_run_spectra(raw, raw, recording_id="sub-01_run-1", fmax=100.0)
    empty = compute_run_spectra(
        raw, raw, recording_id="sub-01_run-1", fmax=100.0, aperiodic_exclude_hz=()
    )
    assert empty.before.aperiodic.exponent == baseline.before.aperiodic.exponent


def test_aperiodic_exclude_hz_composes_with_notch_windows():
    raw = _raw(focal_channels=(), n_channels=8, exponent=1.0)
    both = compute_run_spectra(
        raw,
        raw,
        recording_id="sub-01_run-1",
        fmax=100.0,
        line_frequency=60.0,
        aperiodic_exclude_hz=((20.0, 25.0),),
    )
    notch_only = compute_run_spectra(
        raw, raw, recording_id="sub-01_run-1", fmax=100.0, line_frequency=60.0
    )
    assert both.before.aperiodic.exponent != notch_only.before.aperiodic.exponent
```

If `RunSpectra.before.aperiodic.exponent` is not the accessor, read the `RunSpectra` and `summarize_stage` definitions in `report/spectra.py` and use the real path.

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
python -m pytest tests/preprocessing/test_report_spectra.py -k aperiodic_exclude -v
```

Expected: FAIL with `TypeError: compute_run_spectra() got an unexpected keyword argument 'aperiodic_exclude_hz'`.

- [ ] **Step 3: Accept and honour the new parameter**

In `eeg_pipeline/preprocessing/report/spectra.py`, add the parameter to `compute_run_spectra`'s signature after `notch_half_width_hz`:

```python
    aperiodic_exclude_hz: Sequence[tuple[float, float]] = (),
```

Leave the existing docstring alone. Explain the new parameter with a comment where it is consumed instead:

```python
    # Windows withheld from the aperiodic fit beyond the notch stopbands and anything a
    # decomb manifest reports unavailable. For a persistent narrowband feature that is
    # instrumental rather than neural -- an equipment line, a residual comb -- which a
    # robust fit would otherwise tilt toward. Empty unless a study names a reason.


Extend the `excluded` tuple:

```python
    excluded = (
        tuple(
            notch_windows(
                line_frequency,
                fmax=upper,
                half_width=notch_half_width_hz,
                unavailable_intervals=unavailable_intervals,
            )
        )
        + gradient_windows(gradient_fundamental_hz, frequencies=frequencies)
        + tuple((float(low), float(high)) for low, high in aperiodic_exclude_hz)
    )
```

- [ ] **Step 4: Add the setting and the config key**

In `eeg_pipeline/preprocessing/report/settings.py`, add to the `ReportSettings` dataclass beside `aperiodic_fit_range_hz`:

```python
    # Withheld from the aperiodic fit, on top of the notch stopbands and the
    # decomb-unavailable intervals.
    aperiodic_exclude_hz: tuple[tuple[float, float], ...] = ()
```

Add `"aperiodic_exclude_hz"` to the supported-keys set for the `analysis` block, and parse it in the `from_*` classmethod alongside `aperiodic_fit_range_hz`, using whatever float-pair-sequence helper that module already provides. Read the surrounding parsing code and match it; do not invent a new helper.

In `eeg_pipeline/utils/config/eeg_config.yaml`, add inside the `report: analysis:` block after `aperiodic_fit_range_hz`:

```yaml
    # Frequency windows withheld from the aperiodic fit, beyond the notch stopbands and
    # anything a decomb manifest reports as unavailable. For a persistent narrowband
    # feature that is instrumental rather than neural -- an equipment line, a residual
    # comb -- which a robust fit would otherwise tilt toward. Empty by default: a fit
    # should sit on the data unless there is a named reason it cannot.
    aperiodic_exclude_hz: []
```

In `eeg_pipeline/preprocessing/report/run_evidence.py`, pass it through at the `compute_run_spectra` call (line ~242):

```python
                aperiodic_exclude_hz=settings.aperiodic_exclude_hz,
```

- [ ] **Step 5: Run the tests and confirm they pass**

```bash
python -m pytest tests/preprocessing/test_report_spectra.py tests/preprocessing/test_report_setup_settings.py tests/preprocessing/test_report_settings_new_keys.py tests/preprocessing/test_report_settings_shipped_configs.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add eeg_pipeline/preprocessing/report/spectra.py eeg_pipeline/preprocessing/report/settings.py eeg_pipeline/preprocessing/report/run_evidence.py eeg_pipeline/utils/config/eeg_config.yaml tests/preprocessing/test_report_spectra.py
git commit -m "feat(report): let config name frequency windows the aperiodic fit withholds"
```

### Task 3: Migrate the comb exclusion to config and remove the volume-rate derivation

Atomic: the study's harmonics go into its override in the same commit that stops deriving them. Split either way, there is a commit whose aperiodic slopes are wrong.

**Files:**
- Modify: `studies/pain_study/scripts/config/thermal_pain_eeg_overrides.yaml`
- Modify: `eeg_pipeline/preprocessing/report/spectra.py` (remove `gradient_windows`, `_HARMONIC_SKIRT_BINS`, `gradient_fundamental_hz`)
- Modify: `eeg_pipeline/preprocessing/report/run_evidence.py:242`
- Test: `tests/preprocessing/test_report_spectra.py`

**Interfaces:**
- Consumes: `compute_run_spectra(..., aperiodic_exclude_hz=...)` from Task 2
- Produces: `compute_run_spectra` no longer accepts `gradient_fundamental_hz`

- [ ] **Step 1: Compute the window list the derivation currently produces**

Do not hand-derive this. Call the existing function with the study's real inputs and print what it returns, so the migrated list is the same list by construction:

```python
# scratch script -- run, copy the output, then discard
import numpy as np
from eeg_pipeline.preprocessing.report.spectra import gradient_windows

# The frequency grid compute_run_spectra fits on: Welch bins from 1 Hz to the low-pass.
# Read sfreq and h_freq from a real filtered run rather than assuming them.
frequencies = np.arange(1.0, 100.0 + 1e-9, <welch_bin_width>)
fundamental = 1.0 / <repetition_time_s>
for low, high in gradient_windows(fundamental, frequencies=frequencies):
    if low >= 2.0 and high <= 45.0:      # the aperiodic fit range
        print(f"      - [{low:.4f}, {high:.4f}]")
```

Get `<repetition_time_s>` and `<welch_bin_width>` from real data, not from memory. The repetition time is in any existing sidecar or cohort audit TSV:

```bash
find . -name "*_comb_qc.tsv" -o -name "*_desc-cohort_qc.json" 2>/dev/null | head
```

The bin width is `1 / welch_seconds` for the window `compute_run_spectra` uses — read it from `_channel_spectra_db` in `report/spectra.py`, **not** from `COMB_WELCH_SECONDS`, which belongs to the comb measurement and is a different window.

Also record the current `aperiodic_*` values for one subject now, so Step 6 has a baseline to compare against.

- [ ] **Step 2: Write the failing test**

Append to `tests/preprocessing/test_report_spectra.py`:

```python
def test_compute_run_spectra_no_longer_takes_a_volume_rate():
    import inspect

    from eeg_pipeline.preprocessing.report import spectra

    assert "gradient_fundamental_hz" not in inspect.signature(
        spectra.compute_run_spectra
    ).parameters
    assert not hasattr(spectra, "gradient_windows")
```

- [ ] **Step 3: Run it and confirm it fails**

```bash
python -m pytest tests/preprocessing/test_report_spectra.py::test_compute_run_spectra_no_longer_takes_a_volume_rate -v
```

Expected: FAIL — both assertions are false today.

- [ ] **Step 4: Populate the study override**

In `studies/pain_study/scripts/config/thermal_pain_eeg_overrides.yaml`, add under `report: analysis:` the window list computed in Step 1, with a comment recording where it came from:

```yaml
  analysis:
    # The residual gradient comb, withheld from the aperiodic fit. Previously derived
    # inside the report from each run's volume markers; named here now that core does not
    # read a volume rate. Harmonics of the 0.9 s repetition time, widened by 1.5 spectral
    # bins either side, which is what the derivation used.
    aperiodic_exclude_hz:
      - [<low>, <high>]
      # ... one pair per harmonic inside the fit range
```

Replace `<low>`/`<high>` with the real numbers from Step 1. **Do not commit placeholder values.**

- [ ] **Step 5: Remove the derivation**

In `eeg_pipeline/preprocessing/report/spectra.py`, delete `gradient_windows` (lines ~237-260), `_HARMONIC_SKIRT_BINS` (~234), the `gradient_fundamental_hz` parameter and its docstring paragraph, and the `+ gradient_windows(...)` term from `excluded`. Delete `gradient_windows` from `__all__` if listed. Remove the gradient clause from the figure caption at line ~444.

In `eeg_pipeline/preprocessing/report/run_evidence.py`, delete the `gradient_fundamental_hz=` argument at line ~242.

Remove `gradient_windows` from the import block at the top of `tests/preprocessing/test_report_spectra.py`, and delete any existing test that exercised it directly.

- [ ] **Step 6: Verify the exponents are bit-identical on a real subject**

This is the gate for the whole phase. Build one subject's report with the study override and compare `aperiodic_*` values against those recorded in Step 1.

```bash
python -m pytest tests/preprocessing/test_report_spectra.py -v
```

Expected: PASS. If any exponent moved, the window list in Step 4 is wrong — fix the list, not the test.

- [ ] **Step 7: Commit**

```bash
git add eeg_pipeline/preprocessing/report/spectra.py eeg_pipeline/preprocessing/report/run_evidence.py tests/preprocessing/test_report_spectra.py studies/pain_study/scripts/config/thermal_pain_eeg_overrides.yaml
git commit -m "refactor(report): name the comb exclusion in config instead of deriving it

The windows are the same windows; core stops reading a volume rate to
find them. Verified exponent-identical on a real subject."
```

### Task 3A: Give the study a config the loader actually reads

**Added during execution.** Task 3's review found that
`studies/pain_study/scripts/config/thermal_pain_eeg_overrides.yaml` is loaded by nothing —
its own README says override templates are not applied automatically, `load_config()`
resolves `report.analysis.aperiodic_exclude_hz` to `[]`, and a report built the previous day
carried core defaults rather than the template's values. The live config for this study is
`eeg_pipeline/utils/config/eeg_config.yaml`, which holds the study's own `bids_root`,
`decomb_manifest`, `bids_fmri_root` and `deriv_root`.

That invalidates the premise Task 12 rests on. This task builds the destination Task 12
needs, before Task 12 runs.

**Scope discipline: this task is additive.** Core's defaults keep working exactly as they do
today, so none of the 94 test files touching `bids_root`/`deriv_root` changes. Emptying core
is Task 12's job, once there is somewhere for the values to go.

**Files:**
- Create: `studies/pain_study/config/pain_study.yaml`
- Create: `studies/tests/config/test_pain_study_config.py`
- Modify: `studies/pain_study/scripts/README.md`

**Interfaces:**
- Produces: `studies/pain_study/config/pain_study.yaml`, loadable as
  `load_config("studies/pain_study/config/pain_study.yaml")` and via
  `eeg-pipeline --config studies/pain_study/config/pain_study.yaml <command>`

- [ ] **Step 1: Write the failing test**

Create `studies/tests/config/test_pain_study_config.py`:

```python
# The study's own config must load, inherit from core, and win where it disagrees.

from __future__ import annotations

from pathlib import Path

from eeg_pipeline.utils.config.loader import load_config

STUDY_CONFIG = Path("studies/pain_study/config/pain_study.yaml")


def test_the_study_config_loads():
    assert STUDY_CONFIG.exists()
    assert load_config(STUDY_CONFIG) is not None


def test_it_inherits_keys_it_does_not_set():
    config = load_config(STUDY_CONFIG)
    # Set in core, not overridden here, so inheritance is what supplies it.
    assert config.get("report.analysis.aperiodic_fit_range_hz", None) == [2.0, 45.0]


def test_it_names_the_study_data_roots():
    config = load_config(STUDY_CONFIG)
    for key in ("paths.bids_root", "paths.deriv_root", "paths.decomb_manifest"):
        assert config.get(key, None), f"{key} must be set by the study config"
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
.venv/bin/python -m pytest studies/tests/config/test_pain_study_config.py -v
```

Expected: FAIL — the file does not exist.

- [ ] **Step 3: Write the study config**

Model it on `eeg_pipeline/utils/config/presets/rest.yaml`, which already uses the
inheritance machinery: an `extends:` key naming the core config by relative path, then the
values that differ. Copy across the four path keys that are unambiguously this study's,
with their explanatory comments: `bids_root`, `decomb_manifest`, `bids_fmri_root`,
`deriv_root`.

Head the file with a comment stating what it is — the pain study's configuration, inheriting
everything generic from core and naming only what belongs to this study.

**Verify the relative path resolves by loading, not by reading.** `rest.yaml` sits in
`presets/` and uses `../eeg_config.yaml`; this file sits under `studies/pain_study/config/`,
a different depth, so the path is different.

- [ ] **Step 4: Confirm inheritance merges rather than replaces**

A config that silently dropped every unlisted core key would satisfy Step 1's tests and
break everything downstream:

```bash
.venv/bin/python -c "
from eeg_pipeline.utils.config.loader import load_config
core = load_config()
study = load_config('studies/pain_study/config/pain_study.yaml')
keys = ('report.analysis.aperiodic_fit_range_hz', 'ica.cardiac_review.enabled', 'preprocessing.h_freq', 'report.thresholds.min_roi_channels')
print('changed unexpectedly:', [k for k in keys if study.get(k, None) != core.get(k, None)])
"
```

Expected: an empty list.

- [ ] **Step 5: Update the documented invocation**

`studies/pain_study/scripts/README.md` shows bare `eeg-pipeline preprocessing ...` around
line 234. Add `--config studies/pain_study/config/pain_study.yaml` to those examples, with a
short note above the block: core's defaults are not this study's, so the study config has to
be named.

- [ ] **Step 6: Run the tests**

```bash
.venv/bin/python -m pytest studies/tests/config/ -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add studies/pain_study/config/pain_study.yaml studies/tests/config/test_pain_study_config.py studies/pain_study/scripts/README.md
git commit -m "feat(config): give the pain study a config the loader actually reads

The override template under scripts/config/ is loaded by nothing, so
every study-specific value has been living in the shared core config.
This is the destination the scanner keys move to in Task 12."
```

### Task 4: Delete the gradient report wiring from core

**Files:**
- Modify: `eeg_pipeline/preprocessing/report/organize.py:301`
- Modify: `eeg_pipeline/preprocessing/report/run_evidence.py`
- Modify: `eeg_pipeline/preprocessing/report/cohort/report.py`
- Modify: `eeg_pipeline/preprocessing/report/continuity.py`
- Modify: `eeg_pipeline/preprocessing/report/cohort/spectra.py`
- Test: `tests/preprocessing/test_report_organize.py`, `test_report_run_evidence.py`, `test_report_continuity.py`, `tests/preprocessing/report/test_cohort_report.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/preprocessing/test_report_organize.py`:

```python
def test_section_order_names_no_scanner_section():
    from eeg_pipeline.preprocessing.report.organize import SECTION_ORDER

    assert "Residual scanner gradient" not in SECTION_ORDER
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
python -m pytest tests/preprocessing/test_report_organize.py::test_section_order_names_no_scanner_section -v
```

Expected: FAIL.

- [ ] **Step 3: Remove the wiring**

- `organize.py`: delete `"Residual scanner gradient"` from `SECTION_ORDER`. Update the group comment above it — `"What came in, and what the upstream correction left in it"` now covers only the coverage and data-quality sections.
- `run_evidence.py`: delete the `scanner` import block, `MARKED_GRADIENT_HARMONICS`, `gradient_marks_hz`, `has_scanner_evidence`, the `combs`/`locked_averages`/`declined_combs` fields, the `add_scanner_residual_section` call, and `MARKED_GRADIENT_HARMONICS` from `__all__`. In the `add_spectra_section` call, `marked_frequencies` becomes `tuple(settings.spectra_marked_frequencies)` alone.
- `cohort/report.py`: delete the `add_gradient_section` import and call, and the `comb` entry from `_audit_tables`.
- `continuity.py`: delete `VOLUME_GAP_FACTOR`, `_volume_gaps`, `has_volume_markers`, the `volume_gaps` field, the `volume_description` parameter, and the volume-marker rug.
- `cohort/spectra.py`: delete `MARKED_GRADIENT_HARMONICS` and the harmonic marks derived from it.

- [ ] **Step 4: Run the affected tests**

```bash
python -m pytest tests/preprocessing/test_report_organize.py tests/preprocessing/test_report_run_evidence.py tests/preprocessing/test_report_continuity.py tests/preprocessing/report/test_cohort_report.py tests/preprocessing/report/test_cohort_continuity.py tests/preprocessing/report/test_cohort_spectra.py tests/preprocessing/test_report_modes.py -v
```

Expected: PASS. Delete test cases that asserted on the removed sections; do not weaken assertions to keep them alive.

- [ ] **Step 5: Commit**

```bash
git add -A eeg_pipeline/preprocessing/report tests/preprocessing
git commit -m "refactor(report): drop the residual gradient section and its wiring"
```

---

### Task 5: Move the gradient measurement to the study

**Files:**
- Create: `studies/pain_study/analysis/gradient/__init__.py`, `comb.py`, `locked.py`, `cohort.py`
- Create: `studies/pain_study/analysis/noise_floor.py`
- Delete: `eeg_pipeline/preprocessing/report/scanner.py`, `eeg_pipeline/preprocessing/report/cohort/gradient.py`
- **Not** deleted here: `eeg_pipeline/preprocessing/report/cohort/noise_floor.py` — copied, not moved. See Step 1.
- Move: `tests/preprocessing/test_report_scanner.py` → `studies/tests/analysis/test_gradient_comb.py`
- Move: `tests/preprocessing/report/test_cohort_gradient.py` → `studies/tests/analysis/test_gradient_cohort.py`

**Interfaces:**
- Produces: `studies.pain_study.analysis.gradient.comb` — `measure_volume_timing`, `compute_comb_residual`, `CombResidual`, `VolumeTiming`, `CombNotMeasured`
- Produces: `studies.pain_study.analysis.gradient.locked` — `compute_volume_locked_average`, `VolumeLockedAverage`
- Produces: `studies.pain_study.analysis.gradient.cohort` — `participant_comb`, `cohort_comb`, `comb_attenuation`, `comb_audit`, `CohortComb`
- Produces: `studies.pain_study.analysis.noise_floor` — `measure_locked_average`

- [ ] **Step 1: Create the packages and move the measurement code**

Split `eeg_pipeline/preprocessing/report/scanner.py` by responsibility:

- `analysis/gradient/comb.py` — `VOLUME_MARKER_DESCRIPTION`, `COMB_WELCH_SECONDS`, `HARMONIC_PEAK_FRACTION`, `BACKGROUND_FRACTION_RANGE`, `MINIMUM_BINS_PER_HARMONIC`, `MINIMUM_VOLUMES`, `VolumeTiming`, `CombResidual`, `CombNotMeasured`, `measure_volume_timing`, `_channel_spectrum`, `_excess_db_per_channel`, `compute_comb_residual`
- `analysis/gradient/locked.py` — `VolumeLockedAverage`, `_locked_average_rms_uv`, `compute_volume_locked_average`
- `analysis/gradient/cohort.py` — everything from `report/cohort/gradient.py` **except** `timing_table`, `attenuation_table`, `plot_cohort_comb` and `add_gradient_section`

**Copy** `report/cohort/noise_floor.py` to `analysis/noise_floor.py`; do not delete the core copy yet. It has two importers leaving in different tasks — `scanner.py` here, and `analyzer_qc.py` in Task 10 — and core must never import from `studies/`, so repointing the survivor is not an option. Two commits of duplication is the price of never leaving the tree broken. Task 10 deletes the core copy once its last importer is gone.

Do **not** carry over: `scanner_residual_html`, `_comb_table`, `_locked_table`, `_declined_table`, `_COMB_INTRO`, `_COMB_NOTE`, `volume_locked_note_html`, `add_scanner_residual_section`, `_comb_run_label`, `legend_columns`, `plot_comb_residual`, `plot_volume_locked_average`. Plots and tables are Task 5; their prose goes into the README there.

Rewrite imports: `eeg_pipeline.preprocessing.report.filtering` (`in_notch`, `notch_windows`, `NOTCH_EXCLUSION_HALF_WIDTH_HZ`) and `report.annotations` stay as core imports — the study may import core. `report.style` imports are only needed by the plotting code, which is not moving here.

- [ ] **Step 2: Move the tests and strip their rendering cases**

`git mv` both test files to `studies/tests/analysis/`. Update their imports to the new module paths. Delete every test that asserted on HTML — anything calling `scanner_residual_html`, `add_scanner_residual_section`, `timing_table` or `attenuation_table`. Keep every test of the measurements.

- [ ] **Step 3: Run the moved tests**

```bash
python -m pytest studies/tests/analysis/test_gradient_comb.py studies/tests/analysis/test_gradient_cohort.py -v
```

Expected: PASS.

- [ ] **Step 4: Confirm core no longer references the moved modules**

```bash
grep -rn "report\.scanner\|report\.cohort\.gradient\|cohort\.noise_floor" --include="*.py" eeg_pipeline/
python -c "import eeg_pipeline.preprocessing.report.run_evidence, eeg_pipeline.preprocessing.report.cohort.report"
```

Task 4 already removed the report wiring, so three hits remain, all expected and all resolved later:

| Hit | Removed in |
|---|---|
| `settings.py` importing `VOLUME_MARKER_DESCRIPTION` from `scanner` | Task 12 |
| `cohort/record.py` importing `CombResidual`, `VolumeLockedAverage` | Task 17 |
| `analyzer_qc.py` importing `measure_locked_average` from `cohort/noise_floor` | Task 10 |

The first two import from `scanner.py`, which this task deletes — so both must be fixed **now**, ahead of their own tasks, or the tree will not import. In `settings.py` inline the literal (`volume_marker_description: str = "Volume/V  1"`); in `cohort/record.py` replace the type import with `typing.Any` annotations. Both lines disappear entirely in their own tasks; this is the minimum to keep the tree green in between.

The third needs nothing: the core `noise_floor.py` is still there by design.

```bash
python -m pytest tests/preprocessing/ -q 2>&1 | tail -5
```

Expected: collection succeeds and the suite passes.

- [ ] **Step 5: Commit**

```bash
git add -A studies/pain_study/analysis/gradient studies/pain_study/analysis/noise_floor.py studies/tests/analysis eeg_pipeline/preprocessing/report/
git commit -m "refactor(gradient): move the comb and volume-locked measurement to the study"
```

### Task 6: Build the study-side gradient outputs and command

**Files:**
- Create: `studies/pain_study/scripts/gradient/{__init__.py,plot.py,tables.py,config.yaml,README.md}`
- Create: `studies/pain_study/cli/gradient.py`
- Modify: `studies/pain_study/cli/command_registry.py`
- Test: `studies/tests/scripts/test_gradient_outputs.py` (create)

**Interfaces:**
- Consumes: everything Task 5 produced
- Produces: `eeg-pipeline gradient <mode>` writing `*_comb.tsv`, `*_locked.tsv`, `*_comb.png`, `*_locked.png`, `*_cohort_comb.png`

- [ ] **Step 1: Write the failing test**

Create `studies/tests/scripts/test_gradient_outputs.py`:

```python
# The gradient workflow writes tables and figures, not report sections.

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from studies.pain_study.scripts.gradient import tables  # noqa: E402


def test_comb_table_carries_one_row_per_run(tmp_path):
    # Build two CombResidual fixtures with the same helper the moved tests use.
    from studies.tests.analysis.test_gradient_comb import _comb_residual

    frame = tables.comb_frame([_comb_residual("sub-01_run-1"), _comb_residual("sub-01_run-2")])
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 2
    assert {"run", "repetition_time_s", "median_before_excess_db"} <= set(frame.columns)
```

Read `studies/tests/analysis/test_gradient_comb.py` for the real fixture helper name and reuse it rather than building a `CombResidual` by hand.

- [ ] **Step 2: Run it and confirm it fails**

```bash
python -m pytest studies/tests/scripts/test_gradient_outputs.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `tables.py`**

One function per departed HTML table, each returning a `DataFrame` carrying the same columns the table had:

- `comb_frame(combs)` — run, repetition_time_s, n_volumes, marker_jitter_ms, n_harmonics, median_before_excess_db, median_after_excess_db, worst_excess_db, worst_harmonic_hz, worst_channel
- `locked_frame(averages)` — run, stage, n_volumes, observed_locked_rms_uv, noise_floor_uv, signed_excess_power_uv2, half_correlation, resolved_amplitude_uv
- `declined_frame(declined)` — run, n_harmonics, n_notched, reason
- `attenuation_frame(comb)` — subject, before_db, after_db, removed_db
- `timing_frame(cohort)` — subject, repetition_time_s, worst_jitter_ms, n_volumes, before_uv, after_uv, removed_uv, observed_after_uv, floor_after_uv

Preserve the `unresolved` sentinel: a non-positive excess is written as the string `unresolved`, never as `0`.

- [ ] **Step 4: Write `plot.py`**

Port `plot_comb_residual`, `plot_volume_locked_average` (from the deleted `report/scanner.py`) and `plot_cohort_comb` (from the deleted `report/cohort/gradient.py`), plus the `legend_columns` and `separated_labels` helpers they need. Follow `studies/pain_study/scripts/line_comb/plot.py`: `matplotlib.use("Agg")` before the pyplot import, module-level `DPI = 200`, save to PNG. Replace `report.style` imports with the study's `figure_style.py` or local colour constants.

- [ ] **Step 5: Write `config.yaml` and `README.md`**

`config.yaml` carries the keys leaving core in Phase 4 — `comb_frequency_range_hz: [15.0, 90.0]`, `comb_welch_seconds: 8.0`, `repetition_time_tolerance_s: 0.001`, `volume_marker_description: "Volume/V  1"`, and the `scanner_harmonic_qc` block.

`README.md` carries the prose from the deleted table notes verbatim — the `_COMB_NOTE` paragraph on excess being uncalibrated and focal, and the `volume_locked_note_html` paragraphs on the odd-even floor, the `unresolved` sentinel, what `halves agree` distinguishes, and why the after figure is not guaranteed smaller. That prose is the only written account of what these measurements mean; do not paraphrase it away.

- [ ] **Step 6: Wire the CLI**

Write `studies/pain_study/cli/gradient.py` modelled on `studies/pain_study/cli/line_comb.py` — `setup_gradient(subparsers)` and `run_gradient(args, subjects, config)`, with modes `measure` and `plot`. Register it in `command_registry.py` beside `line_comb_command()`.

- [ ] **Step 7: Run the tests**

```bash
python -m pytest studies/tests/scripts/test_gradient_outputs.py studies/tests/ -k gradient -v
eeg-pipeline gradient --help
```

Expected: PASS, and the help text lists both modes.

- [ ] **Step 8: Commit**

```bash
git add -A studies/pain_study/scripts/gradient studies/pain_study/cli studies/tests/scripts
git commit -m "feat(gradient): draw the comb and locked residual as figures and tables"
```

## Phase 2 — Extract and genericize what stays

### Task 7: Make the ECG beat source configurable

**Files:**
- Modify: `eeg_pipeline/preprocessing/ica_cardiac_review.py:12`, `:216-253`, `:255-310`, and `CardiacReviewSettings`
- Modify: `eeg_pipeline/utils/config/eeg_config.yaml` (the `ica.cardiac_review` block)
- Modify: `eeg_pipeline/utils/config/loader.py:50-57`
- Test: `tests/preprocessing/test_ica_cardiac_beat_source.py` (create)

**Interfaces:**
- Produces: `CardiacReviewSettings.beat_source: str = "auto"` and `.marker_description: str | None = None`
- Produces: `_marker_beats(raw, description)` — takes the description rather than importing a constant

- [ ] **Step 1: Write the failing tests**

Create `tests/preprocessing/test_ica_cardiac_beat_source.py`:

```python
# Beat times come from markers, the channel, or whichever is available.

from __future__ import annotations

import pytest

from eeg_pipeline.preprocessing.ica_cardiac_review import CardiacReviewSettings


def test_beat_source_defaults_to_auto():
    assert CardiacReviewSettings().beat_source == "auto"


def test_marker_description_defaults_to_none():
    assert CardiacReviewSettings().marker_description is None


def test_beat_source_rejects_an_unknown_value():
    with pytest.raises(ValueError, match="beat_source"):
        CardiacReviewSettings.from_mapping({"beat_source": "guess"})


def test_marker_description_survives_a_slash():
    settings = CardiacReviewSettings.from_mapping(
        {"marker_description": "Pulse Artifact/R"}
    )
    assert settings.marker_description == "Pulse Artifact/R"
```

- [ ] **Step 2: Run them and confirm they fail**

```bash
python -m pytest tests/preprocessing/test_ica_cardiac_beat_source.py -v
```

Expected: FAIL — the fields do not exist.

- [ ] **Step 3: Add the settings**

In `ica_cardiac_review.py`, add to `CardiacReviewSettings`:

```python
    # markers | detect | auto. "auto" is what this module did unconditionally before the
    # choice existed: prefer a marker train, fall back to the channel. That preference was
    # fixed because a QRS detector locks onto the magnetohydrodynamic deflection in a
    # magnet, reporting 8 and 2 bpm where markers report 61 and 60. Elsewhere it is wrong.
    beat_source: str = "auto"
    # The beat annotation, spelled as the recording spells it. None means there is none,
    # which is ordinary: a lead without a marker train is detected from the channel.
    marker_description: str | None = None
```

Add both names to the `supported` set in `from_mapping`, and parse them:

```python
            beat_source=_beat_source(values.get("beat_source", cls.beat_source)),
            marker_description=_marker_description(
                values.get("marker_description", cls.marker_description)
            ),
```

Add the two validators beside `_ctps_threshold`:

```python
BEAT_SOURCES = ("markers", "detect", "auto")


def _beat_source(value: Any) -> str:
    text = str(value).strip()
    if text not in BEAT_SOURCES:
        raise ValueError(
            "ica.cardiac_review.beat_source must be one of " + ", ".join(BEAT_SOURCES)
        )
    return text


def _marker_description(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
```

- [ ] **Step 4: Honour the setting**

Change `_marker_beats` to take the description:

```python
def _marker_beats(raw: mne.io.BaseRaw, description: str) -> np.ndarray:
```

and use `event_id={description: PULSE_EVENT_ID}`. Delete the `PULSE_MARKER_DESCRIPTION` import at line 12.

Rewrite the top of `detect_ecg_events`:

```python
    _validate_ecg_channel(raw, settings.ecg_channel)
    sfreq = float(raw.info["sfreq"])

    if settings.beat_source in {"markers", "auto"} and settings.marker_description:
        markers = _marker_beats(raw, settings.marker_description)
        if markers.shape[0] >= MINIMUM_BEATS:
            rate = _rate_from_beats(markers, sfreq)
            if np.isfinite(rate) and rate > 0:
                return EcgDetection(
                    events=markers,
                    average_pulse_bpm=rate,
                    source=ANALYZER_MARKER_SOURCE,
                )
        if settings.beat_source == "markers":
            raise UnusableEcg(
                f"beat_source is 'markers' and {settings.marker_description!r} yielded "
                f"fewer than {MINIMUM_BEATS} usable beats. Set beat_source to 'auto' to "
                "fall back to channel detection."
            )
    if settings.beat_source == "markers":
        raise UnusableEcg(
            "beat_source is 'markers' but ica.cardiac_review.marker_description names no "
            "annotation."
        )
```

leaving the existing `find_ecg_events` block below it unchanged.

Rename the constant itself, not only its value: `ANALYZER_MARKER_SOURCE = "analyzer-markers"` becomes `MARKER_TRAIN_SOURCE = "annotation-markers"`, with a comment describing a marker train generally rather than Analyzer's. **The name matters as much as the value** — `analyzer` is on the forbidden-word list the Task 18 gate applies to `eeg_pipeline/preprocessing/`, so leaving the old name would fail that gate later. Update every reader:

```bash
grep -rn "ANALYZER_MARKER_SOURCE" --include="*.py" eeg_pipeline/ studies/ tests/
```

- [ ] **Step 5: Add the config keys**

In `eeg_config.yaml`, inside `ica: cardiac_review:` after `ecg_channel`:

```yaml
    # Where beat times come from.
    #   markers  read the annotation named below; fail if it yields too few beats
    #   detect   run find_ecg_events on the ECG channel, ignoring any markers
    #   auto     prefer markers where present, fall back to detection
    beat_source: auto
    # Annotation carrying one mark per heartbeat, exactly as the recording spells it.
    # Null means the recording carries none, which is the ordinary case: a montage with
    # an ECG lead and no marker train detects from the channel.
    marker_description: null
```

Correct the block comment above `cardiac_review:` — it currently claims R peaks "do not require Analyzer R annotations", while the code prefers a marker train when one is named.

In `utils/config/loader.py`, add `"marker_description"` to `_NON_PATH_KEYS`. **This is required, not cosmetic:** `"Pulse Artifact/R"` contains a slash, and without the entry the loader resolves it as a filesystem path.

Register `beat_source` in the two lists that track settings which move a measurement:
`cohort/homogeneity.py` `COMPARED_SETTINGS` (bare setting names) and `provenance.py`
`PROVENANCE_KEYS` (`(dotted.key, "Human label")` tuples). Changing the beat source changes
which beats are detected and therefore every cardiac number downstream, so a cohort mixing
subjects processed under different values must raise a homogeneity flag, and a subject's
provenance table must say which source produced its beats. `marker_description` belongs in
provenance for the same reason — two runs whose markers were spelled differently are not
comparable — but not in `COMPARED_SETTINGS`, since a study may legitimately carry several
spellings across sites without that making the numbers incomparable.

*Added during execution: Task 2's review caught exactly this omission for
`aperiodic_exclude_hz`. The rule generalises — any config key this plan adds that changes a
measured value goes in both lists.*

- [ ] **Step 6: Prove behaviour is preserved**

```bash
python -m pytest tests/preprocessing/test_ica_cardiac_beat_source.py tests/preprocessing/test_ica_cardiac_report_figures.py tests/preprocessing/test_ica_cardiac_promotion_wiring.py tests/preprocessing/test_report_settings_shipped_configs.py -v
```

Then, on a run carrying `Pulse Artifact/R`, confirm `detect_ecg_events` with `beat_source="auto"` and `marker_description="Pulse Artifact/R"` returns the **same event array** as today's code. Identical beats, not merely a similar rate.

- [ ] **Step 7: Commit**

```bash
git add -A eeg_pipeline/preprocessing/ica_cardiac_review.py eeg_pipeline/utils/config tests/preprocessing/test_ica_cardiac_beat_source.py
git commit -m "feat(ica): make the ECG beat source and marker name configurable

Any EEG study may carry beat markers; what was vendor-specific was the
hardcoded name. auto reproduces the previous preference exactly."
```

### Task 8: Extract the RR interval section to its own core module

**Files:**
- Create: `eeg_pipeline/preprocessing/report/rr_intervals.py`
- Modify: `eeg_pipeline/preprocessing/report/analyzer_qc.py` (remove the extracted code)
- Modify: `eeg_pipeline/preprocessing/report/run_evidence.py` (import from the new module)
- Move: `tests/preprocessing/test_report_rr_intervals.py` (stays; imports change)

**Interfaces:**
- Produces: `report.rr_intervals` — `add_rr_interval_section(*, report, series, missing, plausible_rr_range_s)`, `RRSeries`, `MISSED_BEAT_FACTOR`

- [ ] **Step 1: Write the failing test**

Append to `tests/preprocessing/test_report_rr_intervals.py`:

```python
def test_rr_section_lives_outside_the_analyzer_module():
    from eeg_pipeline.preprocessing.report import rr_intervals

    assert hasattr(rr_intervals, "add_rr_interval_section")
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
python -m pytest tests/preprocessing/test_report_rr_intervals.py::test_rr_section_lives_outside_the_analyzer_module -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Move the code**

Move `add_rr_interval_section` (`analyzer_qc.py:1308`), its table builder (`:1258`), its two figures (`:1146`, `:1237`), `MISSED_BEAT_FACTOR` and the RR dataclasses into `report/rr_intervals.py`. Head the module with a one-line `#` comment: it is ECG physiology and reads no scanner quantity.

Its section name changes from `"Scanner artifact correction (Analyzer)"` to `"Cardiac rhythm"`, and its tag from the Analyzer tag to `"rr-intervals"`. Add `"Cardiac rhythm"` to `SECTION_ORDER` in `organize.py`, positioned with the ICA cardiac review.

Take the beat train from `detect_ecg_events` rather than reading annotations directly, so it inherits the configured beat source.

Update `run_evidence.py` to import `add_rr_interval_section` from the new module.

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/preprocessing/test_report_rr_intervals.py tests/preprocessing/test_report_run_evidence.py tests/preprocessing/test_report_organize.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A eeg_pipeline/preprocessing/report tests/preprocessing
git commit -m "refactor(report): move the tachogram out of the Analyzer module

Beat-to-beat intervals are ECG physiology and stay in core; they now take
their beats from the configured source rather than a marker train."
```

---

## Phase 3 — Analyzer and BCG

### Task 9: Delete the Analyzer report wiring from core

**Files:**
- Modify: `eeg_pipeline/preprocessing/report/at_a_glance.py:113-140`
- Modify: `eeg_pipeline/preprocessing/report/organize.py`
- Modify: `eeg_pipeline/preprocessing/report/cohort/report.py`
- Modify: `eeg_pipeline/preprocessing/report/run_evidence.py`
- Test: `tests/preprocessing/test_report_at_a_glance.py`, `tests/preprocessing/report/test_cohort_at_a_glance.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/preprocessing/test_report_at_a_glance.py`:

```python
def test_headlines_point_at_no_scanner_section():
    from eeg_pipeline.preprocessing.report.at_a_glance import HEADLINES

    assert all("Analyzer" not in headline.section for headline in HEADLINES)
    assert len(HEADLINES) == 14
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
python -m pytest tests/preprocessing/test_report_at_a_glance.py::test_headlines_point_at_no_scanner_section -v
```

Expected: FAIL — 17 headlines, three naming the Analyzer section.

- [ ] **Step 3: Remove the wiring**

- `at_a_glance.py`: delete the three `Headline` entries for `worst_marker_agreement`, `worst_marker_agreement_lag_ms`, `worst_marker_agreement_lag_iqr_ms` and their comment block.
- `organize.py`: delete `"Scanner artifact correction (Analyzer)"` from `SECTION_ORDER`.
- `cohort/report.py`: delete the `add_analyzer_section` import and call.
- `run_evidence.py`: delete the `add_marker_agreement_section` import and call, and the `marker_agreements` field.

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/preprocessing/test_report_at_a_glance.py tests/preprocessing/report/test_cohort_at_a_glance.py tests/preprocessing/test_report_organize.py tests/preprocessing/report/test_cohort_report.py tests/preprocessing/test_report_run_evidence.py tests/preprocessing/test_report_modes.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A eeg_pipeline/preprocessing/report tests/preprocessing
git commit -m "refactor(report): drop the Analyzer correction section and its headlines"
```

---

### Task 10: Move the Analyzer and BCG modules to the study

**Files:**
- Move: `eeg_pipeline/preprocessing/report/analyzer_qc.py` → `studies/pain_study/analysis/bcg/report.py`
- Move: `eeg_pipeline/preprocessing/report/cohort/analyzer.py` → `studies/pain_study/analysis/bcg/cohort.py`
- Move: `eeg_pipeline/preprocessing/report/cohort_qc.py` → `studies/pain_study/analysis/bcg/cohort_qc.py`
- Move: `eeg_pipeline/preprocessing/pulse_artifact_qc.py` → `studies/pain_study/analysis/bcg/pulse_markers.py`
- Move: `eeg_pipeline/preprocessing/cardiac_artifact_qc.py` → `studies/pain_study/analysis/bcg/attenuation.py`
- Create: `studies/pain_study/scripts/bcg/{__init__.py,config.yaml,README.md,plot.py,tables.py}`
- Move the matching test files to `studies/tests/analysis/`

- [ ] **Step 1: Move the modules and rewrite imports**

`git mv` each file. Update every internal import. `measure_locked_average` now comes from `studies.pain_study.analysis.noise_floor`, the copy Task 5 made — and **`eeg_pipeline/preprocessing/report/cohort/noise_floor.py` is deleted in this commit**, now that its last importer has left core. `PULSE_MARKER_DESCRIPTION` keeps its home in `pulse_markers.py`.

Confirm the core copy has no readers before deleting it:

```bash
grep -rn "cohort\.noise_floor\|measure_locked_average" --include="*.py" eeg_pipeline/
```

Expected: no hits.

Strip the MNE-report rendering as in Task 4 — `add_*_section` functions and `grid_table` callers become DataFrame builders in `scripts/bcg/tables.py` and figures in `scripts/bcg/plot.py`. `add_rr_interval_section` is already gone (Task 8) and the wiring that imported these modules is already gone (Task 9); do not move or re-delete either.

- [ ] **Step 2: Move the tests**

`git mv` into `studies/tests/analysis/`:

```bash
git mv tests/preprocessing/test_report_analyzer_qc.py studies/tests/analysis/
git mv tests/preprocessing/report/test_cohort_analyzer.py studies/tests/analysis/
git mv tests/preprocessing/test_pulse_artifact_qc.py studies/tests/analysis/
git mv tests/preprocessing/test_cardiac_artifact_qc.py studies/tests/analysis/
git mv tests/preprocessing/test_report_marker_agreement.py studies/tests/analysis/
git mv tests/preprocessing/test_report_cohort_qc.py studies/tests/analysis/
git mv tests/preprocessing/report/test_cohort_noise_floor.py studies/tests/analysis/
```

The last two cover `cohort_qc.py` and the noise-floor estimator whose core copy this task deletes; both were unnamed in earlier drafts of this plan. Update imports; drop HTML assertions.

- [ ] **Step 3: Run them**

```bash
python -m pytest studies/tests/analysis/ -v
```

Expected: PASS.

- [ ] **Step 4: Confirm core has no dangling imports**

```bash
grep -rn "analyzer_qc\|cohort_qc\|pulse_artifact_qc\|cardiac_artifact_qc\|cohort\.analyzer" --include="*.py" eeg_pipeline/
python -c "import eeg_pipeline.preprocessing.report.run_evidence, eeg_pipeline.preprocessing.report.cohort.report"
```

Expected: no hits from the grep, and both imports succeed. A hit means Task 9 was skipped; do it before continuing.

- [ ] **Step 5: Commit**

```bash
git add -A studies eeg_pipeline tests
git commit -m "refactor(bcg): move the Analyzer and pulse-artifact QC to the study"
```

### Task 11: Delete the Analyzer fallback branch from the band ICA report

This one rots silently rather than failing: the read is guarded by `if "analyzer_marker_ctps_fallback" in components.columns`, and the column's writer has just left core.

**Files:**
- Modify: `eeg_pipeline/preprocessing/band_ica_report.py:1932`, `:1946`, `:2353-2354`
- Test: `tests/preprocessing/test_band_ica_report.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/preprocessing/test_band_ica_report.py`:

```python
def test_band_ica_report_names_no_vendor_fallback():
    from pathlib import Path

    source = Path("eeg_pipeline/preprocessing/band_ica_report.py").read_text(encoding="utf-8")
    assert "analyzer_marker_ctps_fallback" not in source
    assert "0.21s" not in source
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
python -m pytest tests/preprocessing/test_band_ica_report.py::test_band_ica_report_names_no_vendor_fallback -v
```

Expected: FAIL.

- [ ] **Step 3: Delete the branch**

Remove the panel constant at ~1932, the warning text at ~1946 naming Analyzer's 0.21 s default, and the `has_fallback_runs` branch at 2353-2354 along with whatever panel it gated.

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/preprocessing/test_band_ica_report.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A eeg_pipeline/preprocessing/band_ica_report.py tests/preprocessing/test_band_ica_report.py
git commit -m "refactor(report): drop the vendor fallback panel the moved QC fed"
```

## Phase 4 — Configuration

### Task 12: Move the scanner keys out of the core config

**Files:**
- Modify: `eeg_pipeline/utils/config/eeg_config.yaml`
- Modify: `studies/pain_study/scripts/gradient/config.yaml`, `studies/pain_study/scripts/bcg/config.yaml`
- Modify: `eeg_pipeline/preprocessing/report/settings.py`, `provenance.py`
- Test: `tests/preprocessing/test_report_settings_shipped_configs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/preprocessing/test_report_settings_shipped_configs.py`:

```python
def test_core_config_names_no_scanner_key():
    from pathlib import Path

    text = Path("eeg_pipeline/utils/config/eeg_config.yaml").read_text(encoding="utf-8")
    for key in (
        "eeg_fmri:",
        "brainvision_analyzer:",
        "scanner_harmonic_qc:",
        "comb_frequency_range_hz:",
        "comb_welch_seconds:",
        "repetition_time_tolerance_s:",
        "volume_marker_description:",
        "pulse_marker_description:",
        "min_r_markers_per_volume:",
        "bcg_residual_window_s:",
        "trim_to_volume_bounds:",
    ):
        assert key not in text, f"{key} is still in the core config"
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
python -m pytest tests/preprocessing/test_report_settings_shipped_configs.py::test_core_config_names_no_scanner_key -v
```

Expected: FAIL, listing the first key found.

- [ ] **Step 3: Move the keys**

**Destination corrected during execution.** Earlier drafts sent these keys to
`studies/pain_study/scripts/config/thermal_pain_eeg_overrides.yaml`. Nothing loads that file
— Task 3's review established this — so anything the *report* reads goes to
`studies/pain_study/config/pain_study.yaml`, created in Task 3A, which the loader does read.
The workflow configs under `scripts/gradient/` and `scripts/bcg/` are read by their own
workflow code through `workflow_config.py` and remain the right home for settings only those
workflows use.

Delete each from `eeg_config.yaml`, moving its value and its comment to whichever of those
two destinations reads it.

**Also settle the fate of the dead template.** `studies/pain_study/scripts/config/thermal_pain_eeg_overrides.yaml`
is loaded by nothing, yet it sits in `SHIPPED_CONFIGS` and is therefore parsed and
validated by the test suite — authoritative-looking, test-covered, inert. It cost this plan
a whole detour: Task 3 wrote its windows there in good faith and they had no effect. Now
that `pain_study.yaml` exists and is real, that file is a trap with a live duplicate.

Decide explicitly and say why in the commit message. Deleting it is the honest option if
nothing reads it. Keeping it requires a reason that survives the question "what reads
this?". Check the sibling templates in the same directory
(`thermal_pain_fmri_overrides.yaml`, the two `t1_*.yaml`) the same way — `workflow_config.py`
may genuinely read some of them, in which case they stay and only the unread ones go. `preprocessing.eeg_fmri` and `alignment.trim_to_volume_bounds` are deleted outright — nothing reads either after Phase 3, and `trim_to_volume_bounds` was never read at all.

Change `paths.decomb_manifest`'s default to `null` with a comment saying a study configures it; the key itself **stays**, because `pipelines/features.py:1089` reads it and the mechanism is general.

Delete the matching fields from `report/settings.py` — `volume_marker_description`, `pulse_marker_description`, `comb_frequency_range_hz`, `comb_welch_seconds`, `repetition_time_tolerance_s`, `bcg_residual_window_s`, `bcg_residual_baseline_s`, `bcg_residual_measurement_s`, `DEFAULT_REPETITION_TIME_TOLERANCE_S` — and the five scanner rows from `provenance.py`.

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/preprocessing/test_report_settings_shipped_configs.py tests/preprocessing/test_report_settings_new_keys.py tests/preprocessing/test_report_setup_settings.py tests/preprocessing/test_report_setup_settings_recorded.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A eeg_pipeline/utils/config eeg_pipeline/preprocessing/report studies/pain_study/scripts tests/preprocessing
git commit -m "refactor(config): move the scanner keys to the study workflow configs"
```

### Task 13: Delete the acquisition module and trim the config machinery

**Files:**
- Delete: `eeg_pipeline/utils/config/acquisition.py`
- Modify: `eeg_pipeline/utils/config/coherence.py`, `loader.py`
- Modify: `eeg_pipeline/cli/commands/preprocessing_overrides.py:137-138`
- Modify: `eeg_pipeline/pipelines/preprocessing.py`
- Test: `tests/` config coherence tests

- [ ] **Step 1: Write the failing test**

Append to `tests/preprocessing/test_report_settings_shipped_configs.py`:

```python
def test_the_acquisition_module_is_gone():
    import importlib

    import pytest

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("eeg_pipeline.utils.config.acquisition")
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
python -m pytest tests/preprocessing/test_report_settings_shipped_configs.py::test_the_acquisition_module_is_gone -v
```

Expected: FAIL — the module imports fine.

- [ ] **Step 3: Delete and trim**

- Delete `utils/config/acquisition.py`.
- `coherence.py`: delete `_check_scanner_settings`, the `is_eeg_fmri` import and its call site. **Keep `_check_ecg_settings` and `_check_decomb_notch`** — the first wants a channel, the second guards a key that stays.
- `loader.py`: remove `volume_marker_description` and `pulse_marker_description` from `_NON_PATH_KEYS`. `marker_description` was added in Task 7 and stays.
- `preprocessing_overrides.py`: delete the `--trim-to-volume-bounds` override and its argparse flag.
- `pipelines/preprocessing.py`: delete `STEP_SCANNER_HARMONIC_QC`, `STEP_PULSE_MARKER_QC`, `STEP_CARDIAC_ATTENUATION_QC`, their `_get_steps_for_mode` insertions, `_run_scanner_harmonic_qc`, `_is_eeg_fmri`, `_validate_eeg_fmri_declaration`, and the `_is_eeg_fmri()` gate on the ICA cardiac review at line 1140 — that stage now runs whenever `ica.cardiac_review.enabled` is set.

- [ ] **Step 4: Retire the gating test whose subject just disappeared**

`tests/pipelines/test_eeg_only_gating.py` (243 lines) exists to prove that
`preprocessing.eeg_fmri` and `preprocessing.brainvision_analyzer.enabled` are two switches
rather than one — a distinction that stops existing when the first key is deleted. Most of
it must go, but **read it before deleting it**: anything asserting that the ordinary path
still runs — filtering, PyPREP, ICA, the ocular review, epoching — is now asserting the
*only* path, and that is worth more than it was before, not less.

Move those cases into `tests/pipelines/test_pipeline_preprocessing.py` with the scanner
premise stripped from their names and setup, then delete the file. Do not delete wholesale
and do not keep it limping with its scanner cases commented out.

- [ ] **Step 5: Run the tests**

```bash
python -m pytest tests/pipelines/ tests/preprocessing/test_report_settings_shipped_configs.py tests/preprocessing/test_report_modes.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A eeg_pipeline tests
git commit -m "refactor(config): delete the scanner declaration and the stages it gated"
```

### Task 14: Retire the eeg_only preset and update the study configs

**Files:**
- Delete: `eeg_pipeline/utils/config/presets/eeg_only.yaml`
- Modify: `eeg_pipeline/utils/config/presets/rest.yaml`, `eeg_pipeline/utils/config/eeg_config.yaml`
- Modify: `studies/pain_study/scripts/config/thermal_pain_eeg_overrides.yaml`, `studies/pain_study/study{1,2,3}/config/*.yaml`

- [ ] **Step 1: Write the failing test**

Append to `tests/preprocessing/test_report_settings_shipped_configs.py`:

```python
def test_the_eeg_only_preset_is_gone():
    from pathlib import Path

    assert not Path("eeg_pipeline/utils/config/presets/eeg_only.yaml").exists()
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
python -m pytest tests/preprocessing/test_report_settings_shipped_configs.py::test_the_eeg_only_preset_is_gone -v
```

Expected: FAIL.

- [ ] **Step 3: Rescue the prose, then delete the preset**

Before deleting, move two things from `eeg_only.yaml` into `eeg_config.yaml`: the `task: null` deliberate-unset comment, and the long "DO YOU HAVE AN ECG LEAD?" block explaining how to name the channel and switch the ECG stages on. Place the latter beside `eeg.ecg_channels`. Then delete the file.

In `rest.yaml`, delete the sentence naming `preprocessing.eeg_fmri`.

- [ ] **Step 4: Update the study configs**

In `thermal_pain_eeg_overrides.yaml`, confirm all four study-side keys are set:

```yaml
paths:
  decomb_manifest: "<the real manifest path>"
ica:
  cardiac_review:
    marker_description: "Pulse Artifact/R"
report:
  analysis:
    aperiodic_exclude_hz: [...]   # set in Task 3
```

`marker_description` is required. Without it the review silently switches to channel detection and meets the MHD lock-on.

Grep every study config for keys core no longer defines and remove them:

```bash
grep -rn "eeg_fmri\|brainvision_analyzer\|scanner_harmonic_qc\|trim_to_volume_bounds" --include="*.yaml" studies/
```

- [ ] **Step 5: Run the tests**

```bash
python -m pytest tests/preprocessing/test_report_settings_shipped_configs.py studies/tests/config/ -v
eeg-pipeline validate --config-only
```

Expected: PASS, and validation reports no unknown keys.

- [ ] **Step 6: Commit**

```bash
git add -A eeg_pipeline/utils/config studies
git commit -m "refactor(config): retire the eeg_only preset

Core is EEG-only by construction now, so every key the preset switched
off no longer exists. Its ECG-lead explainer moves beside the keys it
describes."
```

---

## Phase 5 — The harmonics subsystem and the conversion helpers

### Task 15: Move the scanner-harmonic subsystem to the study

1,583 lines across eight files, exposed as `eeg-pipeline harmonics`.

**Files:**
- Move: `eeg_pipeline/analysis/qc/scanner_harmonics.py`, `scanner_harmonic_comb.py`, `__init__.py` → `studies/pain_study/analysis/gradient/`
- Move: `eeg_pipeline/plotting/scanner_harmonic_comb.py`, `eeg_pipeline/preprocessing/pipeline/scanner_harmonic_qc.py` → `studies/pain_study/scripts/gradient/`
- Move: `eeg_pipeline/cli/commands/harmonics{,_parser,_orchestrator}.py` → `studies/pain_study/cli/harmonics.py`
- Modify: `eeg_pipeline/cli/commands/__init__.py:98`, `studies/pain_study/cli/command_registry.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/architecture/test_paradigm_code_stays_in_studies.py`:

```python
def test_the_harmonics_subsystem_left_core() -> None:
    for relative_path in (
        "eeg_pipeline/analysis/qc",
        "eeg_pipeline/plotting/scanner_harmonic_comb.py",
        "eeg_pipeline/preprocessing/pipeline/scanner_harmonic_qc.py",
        "eeg_pipeline/cli/commands/harmonics.py",
        "eeg_pipeline/cli/commands/harmonics_parser.py",
        "eeg_pipeline/cli/commands/harmonics_orchestrator.py",
    ):
        assert not (REPO_ROOT / relative_path).exists(), f"{relative_path} is scanner code"
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
python -m pytest tests/architecture/test_paradigm_code_stays_in_studies.py::test_the_harmonics_subsystem_left_core -v
```

Expected: FAIL on the first path.

- [ ] **Step 3: Move the files**

`git mv` each. `eeg_pipeline/analysis/qc/` holds nothing else, so remove the directory. Rewrite imports to the new paths.

Delete the `("harmonics", f"{_EEG}.harmonics", "harmonics", False)` row from `cli/commands/__init__.py:98`, and register the command study-side in `command_registry.py` alongside `line_comb_command()` so `eeg-pipeline harmonics` keeps working for the study.

- [ ] **Step 4: Reverse the protection assertion that guards the plot**

`tests/architecture/test_tui_plotting_removal.py:13` declares:

```python
PROTECTED_PLOTS = ("eeg_pipeline/plotting/scanner_harmonic_comb.py",)
```

That file asserts this plot **must exist in core** — it was deliberately kept when the rest of the plotting tree was removed. Moving it is a reversal of that decision, not an oversight, so make the reversal explicit: drop the entry from `PROTECTED_PLOTS`, and if the tuple empties, delete it and its test rather than leaving an assertion over nothing. Add a line to that file's rationale recording that the plot moved to the study rather than being deleted.

Also check whether `eeg_pipeline/plotting/` still warrants being a package: after this move it holds only `component_tfr.py`. Leave it — one module is a thin package, not a wrong one.

- [ ] **Step 5: Move the tests**

Four files, named rather than grepped for:

```bash
git mv tests/preprocessing/test_scanner_harmonic_qc.py studies/tests/analysis/
git mv tests/analysis/test_scanner_harmonics.py studies/tests/analysis/
git mv tests/analysis/test_scanner_harmonic_comb.py studies/tests/analysis/
git mv tests/plotting/test_scanner_harmonic_comb_plot.py studies/tests/scripts/
```

Two study-side tests already import these modules from core and need their imports repointed, not moved: `studies/tests/pipelines/test_study1_scanner_harmonic_figure.py` and `test_study1_scanner_harmonic_spectrum.py`.

Confirm nothing was missed:

```bash
grep -rln "analysis\.qc\|scanner_harmonic" tests/
```

Expected: no hits.

- [ ] **Step 6: Run the tests**

```bash
python -m pytest tests/architecture/ studies/tests/ tests/plotting/ -v
eeg-pipeline harmonics --help
```

Expected: PASS, and the command still resolves.

- [ ] **Step 7: Commit**

```bash
git add -A eeg_pipeline studies tests
git commit -m "refactor(harmonics): move the scanner-harmonic QC subsystem to the study"
```

### Task 16: Move the BrainVision marker sanitation and the volume trim

`brainvision_markers.py` hardcodes this study's `Vas_on` marker colliding with the scanner's `Volume/V  1`. Both its consumers are already study conversion scripts.

**Files:**
- Move: `eeg_pipeline/preprocessing/brainvision_markers.py` → `studies/pain_study/scripts/conversion/brainvision_markers.py`
- Move: `trim_to_volume_bounds` from `eeg_pipeline/utils/data/preprocessing.py:278` → `studies/pain_study/scripts/conversion/eeg_raw_to_bids.py`
- Move: `tests/preprocessing/test_brainvision_markers.py` → `studies/tests/scripts/`

- [ ] **Step 1: Write the failing test**

Append to `tests/architecture/test_paradigm_code_stays_in_studies.py`:

```python
def test_the_marker_sanitation_left_core() -> None:
    assert not (REPO_ROOT / "eeg_pipeline/preprocessing/brainvision_markers.py").exists()
    assert (
        REPO_ROOT / "studies/pain_study/scripts/conversion/brainvision_markers.py"
    ).exists()
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
python -m pytest tests/architecture/test_paradigm_code_stays_in_studies.py::test_the_marker_sanitation_left_core -v
```

Expected: FAIL.

- [ ] **Step 3: Move both**

`git mv` the marker module and update the two importers, `sanitize_brainvision_vas_markers.py` and `export_brainvision_matlab.py`.

Move the `trim_to_volume_bounds` function body into `eeg_raw_to_bids.py`, its only caller, and remove it from `utils/data/preprocessing.py` and that module's `__all__`.

`git mv tests/preprocessing/test_brainvision_markers.py studies/tests/scripts/` and update its imports.

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/architecture/ studies/tests/scripts/ -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A eeg_pipeline studies tests
git commit -m "refactor(conversion): move the marker sanitation and volume trim to the study"
```

---

## Phase 6 — Sidecar, gates, and the prose sweep

### Task 17: Drop the scanner columns from the cohort sidecar

**Files:**
- Modify: `eeg_pipeline/preprocessing/report/cohort/sidecar.py:39-49`, `:93-104`
- Modify: `eeg_pipeline/preprocessing/report/cohort/record.py:51`, `:175-220`
- Modify: `eeg_pipeline/preprocessing/report/cohort/multiplicity.py:141-179`
- Modify: `eeg_pipeline/preprocessing/report/cohort/composition.py:53`
- Test: `tests/preprocessing/report/test_cohort_sidecar.py`, `test_cohort_record.py`, `test_cohort_multiplicity.py`, `test_cohort_composition.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/preprocessing/report/test_cohort_sidecar.py`:

```python
def test_schema_version_is_four_and_scanner_columns_are_gone():
    from eeg_pipeline.preprocessing.report.cohort import sidecar

    assert sidecar.SCHEMA_VERSION == 4
    assert not hasattr(sidecar, "SCANNER_RUN_COLUMNS")
    assert not hasattr(sidecar, "AcquisitionContext")
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
python -m pytest tests/preprocessing/report/test_cohort_sidecar.py::test_schema_version_is_four_and_scanner_columns_are_gone -v
```

Expected: FAIL — version 3, both attributes present.

- [ ] **Step 3: Remove the columns**

- `sidecar.py`: bump `SCHEMA_VERSION` to `4` and add a comment recording that version 4 dropped the eleven scanner columns. Delete `SCANNER_RUN_COLUMNS` and `AcquisitionContext`. The existing guard at `:483` already refuses a mismatched version — leave it alone.
- `record.py`: delete the `scanner` import at line 51, `_acquisition_context`, and the in-scanner row block.
- `multiplicity.py`: delete `SCANNER_FAMILY`, its two `MetricSource` entries, and its slot in `FAMILY_ORDER`. `FAMILY_ORDER` is already filtered to families present and each metric's decile is computed independently, so no other placement moves.
- `composition.py`: delete the In scanner / Outside scanner strata and the `AcquisitionContext` label map. Its participant table and the rest of the section are untouched.
- `cohort/spectra.py`: deleting `AcquisitionContext` breaks this module too, so it must change in the same commit. Remove the dashed/solid in-scanner linestyle split and its two legend entries (~lines 507-535), the `scanner_note` suffix on the aperiodic figure title, and the two-branch interpretation prose at ~612-650 that reads a downward exponent shift one way inside a bore and the other way outside. What remains is one unconditional reading of the shift.

Verify nothing else referenced the removed symbols:

```bash
grep -rn "AcquisitionContext\|SCANNER_RUN_COLUMNS\|SCANNER_FAMILY" --include="*.py" eeg_pipeline/
```

Expected: no hits.

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/preprocessing/report/ -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A eeg_pipeline/preprocessing/report/cohort tests/preprocessing/report
git commit -m "refactor(cohort): drop the scanner columns and bump the sidecar schema to 4"
```

### Task 18: Write the new boundary contract into the architecture test

**Files:**
- Modify: `tests/architecture/test_paradigm_code_stays_in_studies.py`

- [ ] **Step 1: Write the failing tests**

Replace the module docstring's first line — this file already has one and keeps it, since the contract is what the file exists to state:

```python
"""``eeg_pipeline/`` holds what is true of any EEG study. This checks it stayed that way.
```

and append:

```python
# Compound terms, deliberately. Bare "gradient" would hit np.gradient,
# GradientBoostingRegressor and cnn_gradient_clip_norm, all legitimate and all under
# eeg_pipeline/analysis/. Bare "fmri" would hit resolve_fmri_bids_root and the
# eeg-pipeline fmri commands, which drive this repo's separate fMRI pipeline.
SCANNER_MARKERS = (
    "volume_locked",
    "repetition_time_s",
    "volume_marker",
    "Pulse Artifact/R",
    "scanner gradient",
    "brainvision_analyzer",
)

# Bare words, safe in this tree and nowhere else: eeg_pipeline/preprocessing/ has no
# np.gradient, no GradientBoostingRegressor, no gradient_clip, and no "scanner RAS" --
# those all live under eeg_pipeline/analysis/. "volume" is deliberately absent: ordinary
# English, and an MNE source-space term.
PREPROCESSING_FORBIDDEN_WORDS = (
    "scanner",
    "gradient",
    "bore",
    "analyzer",
    "ballistocardiogram",
    "bcg",
)


@pytest.mark.parametrize("marker", SCANNER_MARKERS)
def test_core_python_does_not_name_a_scanner_concept(marker: str) -> None:
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "eeg_pipeline").rglob("*.py")
        if "__pycache__" not in path.parts
        and marker.lower() in path.read_text(encoding="utf-8").lower()
    ]

    assert not offenders, f"core modules naming {marker!r}: {offenders}"


@pytest.mark.parametrize("word", PREPROCESSING_FORBIDDEN_WORDS)
def test_the_preprocessing_tree_is_free_of_scanner_prose(word: str) -> None:
    pattern = re.compile(rf"\b{word}\b", re.IGNORECASE)
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "eeg_pipeline/preprocessing").rglob("*.py")
        if "__pycache__" not in path.parts and pattern.search(path.read_text(encoding="utf-8"))
    ]

    assert not offenders, f"preprocessing modules naming {word!r}: {offenders}"


@pytest.mark.parametrize("marker", SCANNER_MARKERS)
def test_the_core_config_does_not_name_a_scanner_concept(marker: str) -> None:
    text = (REPO_ROOT / "eeg_pipeline/utils/config/eeg_config.yaml").read_text(encoding="utf-8")

    assert marker.lower() not in text.lower()
```

Add `import re` at the top. Extend `RELOCATED` with the moved core paths and `RELOCATED_TO` with `studies/pain_study/analysis/gradient`, `studies/pain_study/scripts/gradient` and `studies/pain_study/scripts/bcg`.

- [ ] **Step 2: Run them and record what fails**

```bash
python -m pytest tests/architecture/test_paradigm_code_stays_in_studies.py -v
```

Expected: `test_the_preprocessing_tree_is_free_of_scanner_prose` fails with a file list — that list is Task 19's worklist. Save it.

- [ ] **Step 3: Commit the contract**

Commit the test file even though one parametrization is red, so the worklist is recorded:

```bash
git add tests/architecture/test_paradigm_code_stays_in_studies.py
git commit -m "test(architecture): write the EEG-only boundary contract

The prose gate is red until the sweep in the next commit."
```

### Task 19: Sweep the surviving scanner prose

About 130 references sit in modules that stay. This is the "clean report" deliverable: a report whose sections explain themselves by reference to sections that no longer exist is the defect this change exists to cure.

**Files:** every file listed by the failing gate in Task 18. Expect `cohort/spectra.py`, `continuity.py`, `cohort/sidecar.py`, `run_evidence.py`, `at_a_glance.py`, `ica_cardiac_review.py`, `settings.py`, `cohort/record.py`, `report/organize.py`, `report/filtering.py`, `report/summary.py`, `report/tables.py`, `report/style.py`, `report/preservation.py`, `cohort/ica.py`, `cohort/aggregate.py`, `cohort/composition.py`, `ica_cardiac_report.py`.

- [ ] **Step 1: Work the list**

For each file, rewrite the prose so it describes what the module now does. Categories, with real examples:

- **Promises about sections that no longer exist.** `settings.py:322` — "a dataset that spells it differently, or has none, simply gets no gradient section." Delete; the setting is gone.
- **References to departed tables.** `sidecar.py:145` and `:421` — "a comb table without it cannot tell…", "The comb table". Rewrite in terms of what the sidecar still carries.
- **Cross-references into removed sections.** `cohort/composition.py:17` — a reader meeting something "in the middle of the gradient section". Repoint at a section that exists.
- **Justifications resting on a scanner fact.** `report/filtering.py:50` and `:274` — the notch prose explaining itself by reference to the gradient comb's deepest line. Keep the filtering argument, drop the comb.
- **Context that has changed underneath the module.** `ica_cardiac_review.py` and `ica_cardiac_report.py` — written throughout in terms of ballistocardiogram and bore, because that was their only context. They are now the general ECG review, and their prose must say so. **This is rewriting, not find-and-replace**: the MHD lock-on note in `detect_ecg_events` becomes a general note on why a configurable beat source exists, not a deleted sentence.

Preserve every judgement the prose records. Where a comment explains *why* a threshold or an ordering was chosen, keep the reasoning and change only the scanner framing.

- [ ] **Step 2: Run the gate until it is green**

```bash
python -m pytest tests/architecture/test_paradigm_code_stays_in_studies.py -v
```

Expected: PASS, all parametrizations.

- [ ] **Step 3: Check the reading order still reads**

Open `organize.py` and confirm `SECTION_ORDER` is 22 entries and each group comment still describes the sections beneath it. The group headed "What came in, and what the upstream correction left in it" has lost two of its four members — reword or merge it.

- [ ] **Step 4: Run the report test suites**

```bash
python -m pytest tests/preprocessing/ tests/architecture/ -v
```

Expected: PASS.

- [ ] **Step 5: Build one real subject report and read it**

Build a report end to end and open it. Confirm: no section heading over empty content; no sentence pointing at a section that is not there; the contents list reads as one document. This is a human check the test suite cannot make.

- [ ] **Step 6: Commit**

```bash
git add -A eeg_pipeline
git commit -m "docs(report): sweep the scanner prose out of the preprocessing tree

The last commit made the boundary a test; this makes the report read as
one document about EEG rather than a pipeline assembled from parts."
```

---

## Final verification

- [ ] **Run the targeted suites**

```bash
python -m pytest tests/preprocessing/ tests/architecture/ tests/spectral_availability/ studies/tests/ -v
```

Expected: PASS, except `test_config_loader_paths`, which fails at HEAD in this environment for unrelated external-drive reasons.

- [ ] **Confirm the commands still resolve**

```bash
eeg-pipeline --help
eeg-pipeline gradient --help
eeg-pipeline harmonics --help
eeg-pipeline line-comb --help
eeg-pipeline validate --config-only
```

- [ ] **Confirm the two behavioural seams held**

Aperiodic exponents identical to the Task 3 baseline on a real subject; beat detections identical on a run carrying `Pulse Artifact/R`.
