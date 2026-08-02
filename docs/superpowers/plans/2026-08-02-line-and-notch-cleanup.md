# Line and Notch Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the narrowband lines still standing in 45-100 Hz, replace the 0.97 Hz mains FIR notch with a 0.13 Hz spectrum_fit notch so 45-95 Hz becomes continuous, re-run MNE-BIDS-Pipeline once on the result, and add a CSD gamma variant that must prove it reduces the muscle confound.

**Architecture:** Three configuration changes feed one `line-comb apply` pass that rewrites the cleaned BIDS root, then one MNE-BIDS-Pipeline run rebuilds the derivatives, then CSD is added at feature level with no further pipeline run. A new repo-level audit script makes the before/after comparison a command rather than ad-hoc analysis, and is written first so the later verification steps have something to call.

**Tech Stack:** Python 3.11+, MNE 1.12.1, mne-bids-pipeline 1.10.1, numpy, pandas, pytest.

**Source spec:** `docs/superpowers/specs/2026-08-02-epoch-cleanup-design.md`

**Explicitly out of scope:** Stage 2 of the spec (ECG-driven BCG correction) gets its own plan. Nothing here improves 1-45 Hz, and no task below should be extended to attempt it.

## Global Constraints

- Python 3.11+, 100-character lines, Black formatting, Ruff clean.
- No fallback behaviour and no backward-compatibility shims unless explicitly requested; fail fast on invalid input (`AGENTS.md`).
- Volume repetition time is exactly `0.900000` s. Verified from the Volume markers in all 90 runs; do not re-derive it from spectral peaks.
- Spectral work uses 21.6 s TR-commensurate segments (24 TR at 500 Hz = 10800 samples), bin width `0.0462963` Hz, Hann window.
- Background half-width is `100.0 / 21.6` = `4.6296` Hz at and above 13 Hz, and `1.0` Hz below 13 Hz. The wide window returns NaN within 4.6296 Hz of DC and cannot see delta.
- Uncleaned BIDS root: `/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg`. Cleaned root: `/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg_linecleaned`. Derivatives: `/Volumes/KINGSTON/EEG_fMRI_data/derivatives/preprocessed/eeg`.
- Bulk outputs go on the drive, not in the repo. Only small analysis tables belong in `outputs/`.
- sub-0008 is excluded from *analysis* only. Every pipeline stage still runs on all 15 participants.
- Work directly in the main checkout. Do not create git worktrees.
- Never run the full pytest suite; it takes about 9 minutes. Run targeted subsets.

---

### Task 1: Full-band epoch audit as a repo script

The verification in Tasks 7 and 10 needs a repeatable measurement. This task adds it. It computes, per participant and band, three separately-reported quantities: independent-line excess, gradient-comb sideband excess, and holes (power missing below background).

**Files:**
- Create: `studies/pain_study/analysis/band_audit.py`
- Create: `tests/analysis/test_band_audit.py`

**Interfaces:**
- Consumes: `studies.pain_study.analysis.line_comb.diagnosis` (`to_db`, `local_background_db`, `hann_periodogram`, `tr_commensurate_length`).
- Produces:
  - `adaptive_background_db(spectrum_db: np.ndarray, freqs: np.ndarray, *, wide_hz: float = 4.6296, narrow_hz: float = 1.0, switch_hz: float = 13.0) -> np.ndarray`
  - `band_costs(freqs, psd, *, low_hz, high_hz, independent_hz, comb_hz, sideband_hz=0.20) -> dict[str, float]` returning keys `independent_pct`, `sideband_pct`, `hole_pct`.

- [ ] **Step 1: Write the failing tests**

```python
"""The audit must see delta, and must not confuse removal with contamination."""

from __future__ import annotations

import numpy as np
import pytest

from studies.pain_study.analysis import band_audit as ba


def _flat_spectrum(bin_width: float = 0.0462963, high_hz: float = 100.0):
    freqs = np.arange(0.0, high_hz, bin_width)
    psd = np.full(freqs.size, 1e-12)
    return freqs, psd


class TestAdaptiveBackground:
    def test_is_finite_in_delta_where_the_wide_window_is_not(self):
        freqs, psd = _flat_spectrum()
        spectrum_db = ba.hd.to_db(psd)
        wide = ba.hd.local_background_db(spectrum_db, half_width_bins=100)
        adaptive = ba.adaptive_background_db(spectrum_db, freqs)

        delta = (freqs >= 1.0) & (freqs <= 4.0)
        assert not np.any(np.isfinite(wide[delta])), "the 4.63 Hz window should not reach delta"
        assert np.all(np.isfinite(adaptive[delta])), "the adaptive window must reach delta"

    def test_uses_the_wide_window_above_the_switch(self):
        freqs, psd = _flat_spectrum()
        spectrum_db = ba.hd.to_db(psd)
        wide = ba.hd.local_background_db(spectrum_db, half_width_bins=100)
        adaptive = ba.adaptive_background_db(spectrum_db, freqs)

        upper = freqs >= 20.0
        assert np.allclose(adaptive[upper], wide[upper], equal_nan=True)

    def test_tracks_a_one_over_f_slope_without_reading_it_as_a_line(self):
        bin_width = 0.0462963
        freqs = np.arange(bin_width, 100.0, bin_width)
        psd = 1e-12 / freqs  # pure 1/f, no lines anywhere
        adaptive = ba.adaptive_background_db(ba.hd.to_db(psd), freqs)
        excess = ba.hd.to_db(psd) - adaptive

        delta = np.isfinite(excess) & (freqs >= 1.5) & (freqs <= 4.0)
        assert np.nanmax(np.abs(excess[delta])) < 0.5, (
            "a running median over a monotone background must not manufacture a line"
        )


class TestBandCosts:
    def test_a_clean_flat_band_costs_nothing(self):
        freqs, psd = _flat_spectrum()
        costs = ba.band_costs(
            freqs, psd, low_hz=30.1, high_hz=45.0, independent_hz=[], comb_hz=[]
        )
        assert costs["independent_pct"] == pytest.approx(0.0, abs=1e-6)
        assert costs["sideband_pct"] == pytest.approx(0.0, abs=1e-6)
        assert costs["hole_pct"] == pytest.approx(0.0, abs=1e-6)

    def test_an_added_line_is_charged_to_independent_not_to_sidebands(self):
        freqs, psd = _flat_spectrum()
        psd = psd.copy()
        peak = int(round(40.0 / (freqs[1] - freqs[0])))
        psd[peak] *= 100.0

        costs = ba.band_costs(
            freqs, psd, low_hz=30.1, high_hz=45.0, independent_hz=[40.0], comb_hz=[]
        )
        assert costs["independent_pct"] > 1.0
        assert costs["sideband_pct"] == pytest.approx(0.0, abs=1e-6)

    def test_a_nulled_bin_is_charged_to_holes_not_to_lines(self):
        freqs, psd = _flat_spectrum()
        psd = psd.copy()
        notch = int(round(40.0 / (freqs[1] - freqs[0])))
        psd[notch] *= 0.01  # 20 dB below background: removal, not contamination

        costs = ba.band_costs(
            freqs, psd, low_hz=30.1, high_hz=45.0, independent_hz=[40.0], comb_hz=[]
        )
        assert costs["hole_pct"] > 0.0
        assert costs["independent_pct"] == pytest.approx(0.0, abs=1e-6), (
            "a bin dug below background must never count as positive artifact"
        )

    def test_comb_sidebands_exclude_the_nulled_centre_bin(self):
        freqs, psd = _flat_spectrum()
        psd = psd.copy()
        centre = int(round(82.2222 / (freqs[1] - freqs[0])))
        psd[centre] *= 0.01           # comb centre nulled by the gradient correction
        psd[centre + 3] *= 50.0       # shoulder carrying the residual

        costs = ba.band_costs(
            freqs, psd, low_hz=62.0, high_hz=95.0, independent_hz=[], comb_hz=[82.2222]
        )
        assert costs["sideband_pct"] > 0.0, "the shoulder must be counted"
        assert costs["hole_pct"] > 0.0, "the nulled centre must be counted as a hole"

    def test_rejects_a_band_with_no_usable_background(self):
        freqs, psd = _flat_spectrum()
        with pytest.raises(ValueError, match="no bin with a usable background"):
            ba.band_costs(
                freqs, psd, low_hz=0.0, high_hz=0.2, independent_hz=[], comb_hz=[]
            )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/analysis/test_band_audit.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'studies.pain_study.analysis.band_audit'`

- [ ] **Step 3: Write the implementation**

```python
"""Per-band accounting of what is still in the delivered epochs.

Three costs are reported separately because they need different fixes. Independent lines
are what `line-comb apply` can remove. Comb sidebands are residual volume-to-volume
gradient variability in the shoulders of each k/TR harmonic, which a fixed-frequency notch
centred on the nulled comb bin cannot reach. Holes are power *missing* below background
from removal already applied. Pooling them makes every band look equally contaminated and
hides which fix is the one that helps.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from studies.pain_study.analysis.line_comb import diagnosis as hd

WIDE_HALF_WIDTH_HZ = 100.0 / 21.6
NARROW_HALF_WIDTH_HZ = 1.0
SCALE_SWITCH_HZ = 13.0
SIDEBAND_HALF_WIDTH_HZ = 0.20
HOLE_THRESHOLD_DB = 1.0


def adaptive_background_db(
    spectrum_db: np.ndarray,
    freqs: np.ndarray,
    *,
    wide_hz: float = WIDE_HALF_WIDTH_HZ,
    narrow_hz: float = NARROW_HALF_WIDTH_HZ,
    switch_hz: float = SCALE_SWITCH_HZ,
) -> np.ndarray:
    """Running-median background that also reaches delta.

    The production 4.63 Hz half-width returns NaN within 4.63 Hz of DC, so it cannot see
    below about 5 Hz at all -- the estimator trap that once read the 1/f slope as a 4 dB
    line below 3 Hz. Below `switch_hz` the half-width drops to 1.0 Hz, still fifteen times
    a line's width, and a running median over a monotone background is exact, so the 1/f
    slope is not mistaken for structure.
    """
    values = np.asarray(spectrum_db, dtype=float)
    frequency = np.asarray(freqs, dtype=float)
    if values.shape != frequency.shape:
        raise ValueError("spectrum_db and freqs must have the same shape.")

    bin_width = float(frequency[1] - frequency[0])
    wide = hd.local_background_db(values, half_width_bins=int(round(wide_hz / bin_width)))
    narrow = hd.local_background_db(values, half_width_bins=int(round(narrow_hz / bin_width)))
    return np.where(frequency < switch_hz, narrow, wide)


def _mask_near(freqs: np.ndarray, centres: Iterable[float], half_width_hz: float) -> np.ndarray:
    mask = np.zeros(freqs.size, dtype=bool)
    for centre in centres:
        mask |= np.abs(freqs - centre) <= half_width_hz
    return mask


def band_costs(
    freqs: Sequence[float],
    psd: Sequence[float],
    *,
    low_hz: float,
    high_hz: float,
    independent_hz: Iterable[float],
    comb_hz: Iterable[float],
    sideband_hz: float = SIDEBAND_HALF_WIDTH_HZ,
) -> dict[str, float]:
    """Independent-line excess, comb-sideband excess and hole loss, each as a percentage."""
    frequency = np.asarray(freqs, dtype=float)
    spectrum = np.asarray(psd, dtype=float)
    if frequency.shape != spectrum.shape:
        raise ValueError("freqs and psd must have the same shape.")
    if low_hz >= high_hz:
        raise ValueError("low_hz must be below high_hz.")

    bin_width = float(frequency[1] - frequency[0])
    spectrum_db = hd.to_db(spectrum)
    background_db = adaptive_background_db(spectrum_db, frequency)
    background = 10.0 ** (background_db / 10.0)

    inside = (frequency >= low_hz) & (frequency <= high_hz) & np.isfinite(background_db)
    if not np.any(inside):
        raise ValueError("The band holds no bin with a usable background.")

    at_independent = _mask_near(frequency, independent_hz, sideband_hz)
    at_sideband = _mask_near(frequency, comb_hz, sideband_hz)
    # The comb centre is nulled by the volume-average subtraction; charging it as a line
    # would report the correction as contamination.
    for centre in comb_hz:
        at_sideband &= ~(np.abs(frequency - centre) <= 1.5 * bin_width)
    at_sideband &= ~at_independent

    total = float(np.sum(spectrum[inside]))
    costs = {}
    for name, mask in (("independent_pct", at_independent), ("sideband_pct", at_sideband)):
        selected = inside & mask
        excess = float(np.sum(np.clip(spectrum[selected] - background[selected], 0.0, None)))
        costs[name] = 100.0 * excess / total if total > 0 else 0.0

    deep = inside & ((background_db - spectrum_db) > HOLE_THRESHOLD_DB)
    expected = float(np.sum(background[inside]))
    missing = float(np.sum(background[deep] - spectrum[deep]))
    costs["hole_pct"] = 100.0 * missing / expected if expected > 0 else 0.0
    return costs
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/analysis/test_band_audit.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Lint and format**

Run: `.venv/bin/python -m black studies/pain_study/analysis/band_audit.py tests/analysis/test_band_audit.py && .venv/bin/python -m ruff check studies/pain_study/analysis/band_audit.py tests/analysis/test_band_audit.py`
Expected: reformatted or unchanged, and `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add studies/pain_study/analysis/band_audit.py tests/analysis/test_band_audit.py
git commit -m "feat(audit): account for lines, comb sidebands and holes separately

Pooling them makes every band look equally contaminated. Splitting them shows
1-45 Hz has no removable lines at all, which is what decides where each fix
helps. The adaptive background also reaches delta, which the production 4.63 Hz
window cannot."
```

---

### Task 2: Make the mains exclusion configurable

`removal_frequencies` hard-excludes 59.5-60.5 Hz, with a docstring saying it does so because the downstream pipeline notches mains. Stage 1b moves that job here, so the exclusion has to become a setting rather than a constant.

**Files:**
- Modify: `studies/pain_study/analysis/line_comb/removal.py:360-388` (`removal_frequencies`)
- Modify: `studies/pain_study/scripts/line_comb/remove.py:75-116` (`RemovalSettings`), `:210-231` (`estimate_and_targets`)
- Test: `tests/analysis/line_comb/test_removal.py` (extend `TestRemovalFrequencies`)

**Interfaces:**
- Consumes: `lr.CombEstimate`, `lr.MAINS_NOTCH_HZ` from Task 0 state (already present).
- Produces:
  - `RemovalSettings.exclude_mains: bool = True` — read from config key `line_comb_removal.exclude_mains`.
  - `removal_frequencies(..., excluded_hz: Iterable[tuple[float, float]] = (MAINS_NOTCH_HZ,))` unchanged in signature; callers now pass `()` when `exclude_mains` is false.

- [ ] **Step 1: Write the failing tests**

Append to `tests/analysis/line_comb/test_removal.py` inside `class TestRemovalFrequencies`:

```python
    def test_keeps_mains_when_the_exclusion_is_turned_off(self):
        targets = lr.removal_frequencies(
            self._estimate(isolated=(57.22, 60.0)), harmonic_range=(24, 79), excluded_hz=()
        )
        assert any(f == pytest.approx(60.0) for f in targets), (
            "with the FIR notch disabled the removal must take mains itself"
        )
```

And in `tests/scripts/line_comb/test_remove.py`:

```python
def test_removal_settings_reads_the_mains_exclusion_flag():
    from studies.pain_study.scripts.line_comb.remove import RemovalSettings

    class _Config:
        def __init__(self, block):
            self._block = block

        def get(self, key):
            return self._block if key == "line_comb_removal" else None

    assert RemovalSettings.from_config(_Config({})).exclude_mains is True
    assert RemovalSettings.from_config(_Config({"exclude_mains": False})).exclude_mains is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/analysis/line_comb/test_removal.py::TestRemovalFrequencies -v tests/scripts/line_comb/test_remove.py -v`
Expected: FAIL — `TypeError: removal_frequencies() got an unexpected keyword argument` is not expected (the kwarg exists); the failing one is `AttributeError: 'RemovalSettings' object has no attribute 'exclude_mains'`

- [ ] **Step 3: Add the field to `RemovalSettings`**

In `studies/pain_study/scripts/line_comb/remove.py`, add to the dataclass body after `high_hz`:

```python
    exclude_mains: bool = True
    """Leave 59.5-60.5 Hz to the pipeline's own notch.

    False moves mains into this pass, which is the point of doing so: the pipeline's FIR
    notch occupies 0.97 Hz against 0.133 Hz for spectrum_fit at freq/450. Exactly one of
    the two must remove mains; `preprocessing.notch_freq` has to be null when this is
    False, and tests/scripts/line_comb/test_config_pairing.py enforces that.
    """
```

And in `from_config`, add to the constructor call:

```python
            exclude_mains=bool(block.get("exclude_mains", defaults.exclude_mains)),
```

- [ ] **Step 4: Thread it through `estimate_and_targets`**

In `studies/pain_study/scripts/line_comb/remove.py`, change the `lr.removal_frequencies` call:

```python
    targets = lr.removal_frequencies(
        estimate,
        harmonic_range=settings.removal_harmonic_range,
        low_hz=settings.low_hz,
        high_hz=settings.high_hz,
        excluded_hz=(lr.MAINS_NOTCH_HZ,) if settings.exclude_mains else (),
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/analysis/line_comb/test_removal.py tests/scripts/line_comb/test_remove.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add studies/pain_study/analysis/line_comb/removal.py studies/pain_study/scripts/line_comb/remove.py tests/analysis/line_comb/test_removal.py tests/scripts/line_comb/test_remove.py
git commit -m "feat(line-comb): make the mains exclusion a setting

removal_frequencies excluded 59.5-60.5 Hz because the pipeline notched it. That
is about to stop being true, so the reason becomes a flag instead of a constant."
```

---

### Task 3: Add the four isolated lines

**Files:**
- Modify: `studies/pain_study/scripts/line_comb/config.yaml` (`line_comb_removal.isolated_hz`)
- Test: `tests/scripts/line_comb/test_config_pairing.py`

**Interfaces:**
- Consumes: `load_workflow_config("line_comb", core_config=core)` from `studies.pain_study.scripts.workflow_config`.
- Produces: no new symbols; `isolated_hz` gains `23.7776, 29.6854, 61.0353, 81.1111`.

- [ ] **Step 1: Write the failing test**

Append to `tests/scripts/line_comb/test_config_pairing.py`:

```python
import pytest

from studies.pain_study.analysis.line_comb import removal as lr

#: Measured on the delivered epochs, 14 participants, sub-0008 excluded. Each entry is the
#: frequency, the number of participants carrying it, and what it is.
AUDITED_RESIDUALS = {
    23.7776: "narrow, off both combs, 7/14, beta",
    29.6854: "narrow, off both combs, 9/14, 0.41 Hz below the gamma_low edge",
    61.0353: "mains +1.02 Hz sideband, 11/14",
    81.1111: "gradient harmonic 73 at TR = 0.9 s, 3/14, inside 62-95 Hz",
}


def test_the_audited_residual_lines_are_all_targeted():
    _, workflow = _configs()
    isolated = workflow.get("line_comb_removal.isolated_hz")
    search = float(workflow.get("line_comb_removal.isolated_search_hz"))

    for frequency, why in AUDITED_RESIDUALS.items():
        nearest = min(isolated, key=lambda seed: abs(seed - frequency))
        assert abs(nearest - frequency) <= search, (
            f"{frequency} Hz ({why}) has no seed within the {search} Hz search window; "
            f"nearest is {nearest}"
        )


def test_no_isolated_seed_collides_with_a_benchmark_probe():
    _, workflow = _configs()
    isolated = workflow.get("line_comb_removal.isolated_hz")
    probe = lr.Probe()

    for frequency in isolated:
        for sinusoid in probe.sinusoid_hz + (probe.burst_hz,):
            assert abs(frequency - sinusoid) > 0.3, (
                f"seed {frequency} Hz sits on probe tone {sinusoid} Hz; the benchmark would "
                "remove the probe by design and report it as signal loss"
            )


def test_the_gradient_harmonics_are_derived_from_a_nine_tenths_second_tr():
    _, workflow = _configs()
    isolated = workflow.get("line_comb_removal.isolated_hz")

    for harmonic in (73, 74):
        expected = harmonic / 0.9
        nearest = min(isolated, key=lambda seed: abs(seed - expected))
        assert nearest == pytest.approx(expected, abs=0.01), (
            f"gradient harmonic {harmonic} should be seeded at {expected:.4f} Hz "
            f"(TR is exactly 0.9 s by the Volume markers), found {nearest}"
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/scripts/line_comb/test_config_pairing.py -v`
Expected: FAIL — `23.7776 Hz ... has no seed within the 0.15 Hz search window` and the harmonic-73 assertion

- [ ] **Step 3: Edit the config**

In `studies/pain_study/scripts/line_comb/config.yaml`, replace the `isolated_hz` line and add to the comment block above it:

```yaml
  # Added 2026-08-02 from an audit of the delivered epochs (14 participants, sub-0008
  # excluded). 23.7776 and 29.6854 are narrow, on neither comb, and present in 7 and 9 of
  # 14; 29.6854 matters because it sits 0.41 Hz below the 30.1 Hz gamma_low edge, close
  # enough that any spectral smoothing pulls it in. 61.0353 is the mains +1.02 Hz sideband,
  # 11 of 14. 81.1111 is the imaging gradient's 73rd harmonic at TR = 0.9 s -- the partner
  # of 82.2228 below, with the same nulled centre and elevated shoulders, and like it must
  # be re-derived if the sequence TR ever changes.
  #
  # 93.76 Hz is deliberately absent: 16.6 dB, but in one participant only, and a cohort
  # notch for one participant is the wrong trade. 59.02 Hz is absent too -- 4 of 14, and
  # inside the 60 Hz notch skirt.
  isolated_hz: [23.7776, 29.6854, 47.0362, 57.2247, 57.3485, 58.1807, 58.3442, 61.0353, 81.1111, 82.2228, 94.0748]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/scripts/line_comb/test_config_pairing.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add studies/pain_study/scripts/line_comb/config.yaml tests/scripts/line_comb/test_config_pairing.py
git commit -m "feat(line-comb): target the four residual lines the audit found

23.7776 and 29.6854 Hz are narrow and on neither comb; 61.0353 Hz is a mains
sideband; 81.1111 Hz is the gradient's 73rd harmonic, partner to the 82.2228
entry already here. The test pins each to the audit that found it."
```

---

### Task 4: Move mains into the removal and disable the FIR notch

This is one change across two configs. Splitting it would leave a commit where mains is notched twice or not at all.

**Files:**
- Modify: `studies/pain_study/scripts/line_comb/config.yaml` (`isolated_hz`, new `exclude_mains`)
- Modify: `eeg_pipeline/utils/config/eeg_config.yaml:284` (`preprocessing.notch_freq`)
- Test: `tests/scripts/line_comb/test_config_pairing.py`

**Interfaces:**
- Consumes: `RemovalSettings.exclude_mains` from Task 2; `AUDITED_RESIDUALS` and `_configs()` from Task 3.
- Produces: no new symbols.

- [ ] **Step 1: Write the failing test**

Append to `tests/scripts/line_comb/test_config_pairing.py`:

```python
def test_exactly_one_stage_removes_mains():
    core, workflow = _configs()

    fir_notch = core.get("preprocessing.notch_freq")
    exclude_mains = bool(workflow.get("line_comb_removal.exclude_mains"))
    isolated = workflow.get("line_comb_removal.isolated_hz")
    removal_takes_mains = (not exclude_mains) and any(
        abs(seed - 60.0) <= float(workflow.get("line_comb_removal.isolated_search_hz"))
        for seed in isolated
    )

    assert bool(fir_notch) != removal_takes_mains, (
        f"preprocessing.notch_freq={fir_notch!r} and the line-comb pass "
        f"{'takes' if removal_takes_mains else 'does not take'} mains. Exactly one must: "
        "both means a second bite of the spectrum, neither means 60 Hz survives."
    )


def test_the_narrow_mains_notch_is_the_one_in_use():
    core, workflow = _configs()

    assert core.get("preprocessing.notch_freq") in (None, False), (
        "the FIR notch occupies 0.97 Hz against 0.133 Hz for spectrum_fit at freq/450; "
        "leaving it on forfeits 0.84 Hz and keeps 58-62 Hz unusable"
    )
    ratio = float(workflow.get("line_comb_removal.notch_width_ratio"))
    assert 60.0 / ratio < 0.2, (
        f"mains notch would be {60.0 / ratio:.3f} Hz wide; the point of the move is a "
        "notch narrower than 0.2 Hz"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/scripts/line_comb/test_config_pairing.py -v`
Expected: FAIL — `preprocessing.notch_freq=60 and the line-comb pass does not take mains`

- [ ] **Step 3: Add mains to the line-comb config**

In `studies/pain_study/scripts/line_comb/config.yaml`, add `60.0` to `isolated_hz` and add the new key below `high_hz`:

```yaml
  isolated_hz: [23.7776, 29.6854, 47.0362, 57.2247, 57.3485, 58.1807, 58.3442, 60.0, 61.0353, 81.1111, 82.2228, 94.0748]

  # Take mains here instead of in MNE-BIDS-Pipeline. Its notch is FIR and measured 0.97 Hz
  # wide on the delivered epochs (59.537-60.463, -54.6 dB at centre); spectrum_fit at
  # freq/450 takes 0.133 Hz. That recovers about 0.84 Hz and makes 45-95 Hz continuous
  # rather than 45-58 plus 62-95.
  #
  # preprocessing.notch_freq MUST be null while this is false, or mains is removed twice.
  # tests/scripts/line_comb/test_config_pairing.py fails if the two ever disagree.
  exclude_mains: false
```

- [ ] **Step 4: Disable the FIR notch**

In `eeg_pipeline/utils/config/eeg_config.yaml`, replace the `notch_freq: 60` line:

```yaml
  # Mains is removed by the line-comb spectrum_fit pass before the pipeline reads the data
  # (studies/pain_study/scripts/line_comb/config.yaml, exclude_mains: false). The FIR notch
  # here measured 0.97 Hz wide against 0.133 Hz for spectrum_fit. Setting this back to 60
  # without also setting exclude_mains true removes mains twice.
  notch_freq: null
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/scripts/line_comb/test_config_pairing.py tests/config -v`
Expected: PASS. `tests/config/test_config_loader_paths.py` fails at HEAD on this machine for unrelated external-drive path reasons; that one failure is pre-existing and not caused by this change.

- [ ] **Step 6: Commit**

```bash
git add studies/pain_study/scripts/line_comb/config.yaml eeg_pipeline/utils/config/eeg_config.yaml tests/scripts/line_comb/test_config_pairing.py
git commit -m "feat(mains): trade a 0.97 Hz FIR notch for a 0.13 Hz spectrum_fit one

Measured on the delivered epochs, the pipeline's FIR notch empties
59.537-60.463 Hz. spectrum_fit at freq/450 takes 0.133 Hz. Moving mains into
the line-comb pass recovers ~0.84 Hz and makes 45-95 Hz continuous. The two
configs are coupled and a test now fails if they ever disagree."
```

---

### Task 5: Benchmark the new settings before writing anything

The gates are stated before the measurement. A failure means the settings are wrong, not that the gates should move.

**Files:**
- Read only: `studies/pain_study/scripts/line_comb/config.yaml`
- Output: `outputs/line_comb_mains/benchmark.csv`

**Interfaces:**
- Consumes: every config change from Tasks 2-4.
- Produces: a benchmark table later tasks cite; no code symbols.

- [ ] **Step 1: Run the benchmark**

```bash
.venv/bin/eeg-pipeline line-comb benchmark --report-dir outputs/line_comb_mains --limit 13
```

Expected: 13 runs, each with `gate_passed` true across all 7 gates.

- [ ] **Step 2: Confirm every gate passed**

```bash
.venv/bin/python -c "
import pandas as pd
f = pd.read_csv('outputs/line_comb_mains/benchmark.csv')
gates = [c for c in f.columns if c.startswith('gate_') and c != 'gate_passed']
print(f[['recording', 'n_targets', 'median_suppression_db', 'removed_band_fraction']].to_string(index=False))
print()
for g in gates:
    print(f'{g}: {int(f[g].sum())}/{len(f)}')
print('ALL PASSED' if f['gate_passed'].all() else 'FAILED')
"
```

Expected: `ALL PASSED`, and `removed_band_fraction` at or below 0.15 on every run.

If `band_mostly_untouched` fails, the added targets have pushed the removed fraction past 15%. Do not raise the threshold. Drop `61.0353` first — it sits in the unanalysed 58-62 gap and is the only addition that serves hygiene rather than an analysis band — then re-run.

- [ ] **Step 3: Commit the benchmark table**

```bash
git add outputs/line_comb_mains/benchmark.csv
git commit -m "test(line-comb): benchmark the mains move and the four new lines

13/13 runs on all 7 preservation gates with mains taken by spectrum_fit."
```

---

### Task 6: Write the cleaned BIDS copy

**Files:**
- Writes: `/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg_linecleaned` (90 runs, `.eeg` binaries rewritten, sidecars mirrored)

**Interfaces:**
- Consumes: the benchmarked config from Task 5.
- Produces: the cleaned BIDS root that Task 8 preprocesses.

- [ ] **Step 1: Record what is being replaced**

```bash
.venv/bin/python -c "
from pathlib import Path
root = Path('/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg_linecleaned')
files = sorted(p for p in root.glob('sub-*/eeg/*_eeg.eeg') if not p.name.startswith('._'))
print(f'{len(files)} binaries, {sum(p.stat().st_size for p in files) / 1e9:.1f} GB')
"
```

Expected: 90 binaries. This root is regenerated, not edited in place — the previous generation is reproducible by checking out the prior config and re-running, so no backup copy is kept.

- [ ] **Step 2: Apply**

```bash
.venv/bin/eeg-pipeline line-comb apply --report-dir outputs/line_comb_mains
```

Expected: 90 runs written. Runtime is roughly an hour.

- [ ] **Step 3: Confirm the sidecars are untouched and only binaries changed**

```bash
.venv/bin/python -c "
from pathlib import Path
root = Path('/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg_linecleaned')
for kind in ('*_eeg.vhdr', '*_events.tsv', '*_channels.tsv'):
    n = len([p for p in root.glob(f'sub-*/eeg/{kind}') if not p.name.startswith('._')])
    print(f'{kind}: {n}')
"
```

Expected: 90 of each. Events must not have been regenerated — the events chain is not part of this plan.

- [ ] **Step 4: Commit the apply report**

```bash
git add outputs/line_comb_mains/
git commit -m "chore(line-comb): rewrite the cleaned BIDS root with mains and four new lines"
```

---

### Task 7: Verify the cleaned BIDS against the audit

**Files:**
- Create: `studies/pain_study/scripts/band_audit_report.py`
- Output: `outputs/line_comb_mains/band_costs.csv`

**Interfaces:**
- Consumes: `studies.pain_study.analysis.band_audit.band_costs` from Task 1.
- Produces: `main(argv: list[str] | None = None) -> None`, writing one row per participant and band with columns `subject, band, independent_pct, sideband_pct, hole_pct`.

- [ ] **Step 1: Write the report script**

```python
"""Per-band costs on a set of preprocessed epochs, for before/after comparison.

Reads the delivered epochs, takes the channel-median spectrum of 21.6 s TR-commensurate
segments, and reports the three costs from band_audit for each analysis band.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from studies.pain_study.analysis import band_audit as ba
from studies.pain_study.analysis.line_comb import diagnosis as hd

TR = 0.9
INDEPENDENT_HZ = (23.7776, 29.6854, 46.5839, 57.1925, 59.0168, 61.0353, 61.4039, 99.5982)
COMB_HZ = tuple(k / TR for k in range(1, 91))
BANDS = {
    "delta 1-4": (1.0, 4.0),
    "theta 4-8": (4.0, 8.0),
    "alpha 8-13": (8.0, 13.0),
    "beta 13-30": (13.0, 30.0),
    "gamma 30.1-45": (30.1, 45.0),
    "gamma 45-58": (45.0, 58.0),
    "notch gap 58-62": (58.0, 62.0),
    "gamma 62-95": (62.0, 95.0),
    "top 95-100": (95.0, 100.0),
}


def subject_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Channel-median spectrum over good EEG channels, on the TR-commensurate grid."""
    epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
    bads = set(epochs.info["bads"])
    names = [
        n for n, t in zip(epochs.ch_names, epochs.get_channel_types())
        if t == "eeg" and n not in bads
    ]
    data = epochs.copy().pick(names).get_data(copy=False)
    keep = hd.tr_commensurate_length(data.shape[-1], epochs.info["sfreq"])
    freqs, psd = hd.hann_periodogram(data[..., :keep], epochs.info["sfreq"])
    band = freqs <= 125.0
    return freqs[band], np.median(psd.mean(axis=0)[:, band], axis=0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deriv-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude", nargs="*", default=["sub-0008"])
    args = parser.parse_args(argv)

    mne.set_log_level("ERROR")
    rows = []
    subjects = sorted(
        p.name
        for p in args.deriv_root.glob("sub-*")
        if p.is_dir() and not p.name.startswith("._") and p.name not in args.exclude
    )
    if not subjects:
        raise SystemExit(f"No subjects found under {args.deriv_root}")

    for subject in subjects:
        path = args.deriv_root / subject / "eeg" / f"{subject}_task-thermalactive_epo.fif"
        freqs, spectrum = subject_spectrum(path)
        for band, (low, high) in BANDS.items():
            costs = ba.band_costs(
                freqs, spectrum, low_hz=low, high_hz=high,
                independent_hz=INDEPENDENT_HZ, comb_hz=COMB_HZ,
            )
            rows.append({"subject": subject.replace("sub-", ""), "band": band, **costs})
        print(f"  {subject} done", flush=True)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = frame.groupby("band")[["independent_pct", "sideband_pct", "hole_pct"]].median()
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Record the "before" measurement on the current derivatives**

```bash
.venv/bin/python -m studies.pain_study.scripts.band_audit_report \
  --deriv-root /Volumes/KINGSTON/EEG_fMRI_data/derivatives/preprocessed/eeg \
  --out outputs/line_comb_mains/band_costs_before.csv
```

Expected, matching the spec's table to within rounding: `independent_pct` median 0.00 for delta/theta/alpha/gamma 30.1-45 and gamma 62-95, 0.73 for beta, 4.78 for gamma 45-58, 11.75 for the notch gap, 7.44 for 95-100. `hole_pct` median 34.50 for the notch gap.

If these do not reproduce, stop: the audit script disagrees with the measurement the whole plan rests on, and that must be resolved before Task 8 spends an hour of compute.

- [ ] **Step 3: Commit the before table and the script**

```bash
git add studies/pain_study/scripts/band_audit_report.py outputs/line_comb_mains/band_costs_before.csv
git commit -m "feat(audit): report per-band costs on a set of epochs

Reproduces the spec's before-table from the delivered derivatives, so the
after-comparison in this plan is a command rather than ad-hoc analysis."
```

---

### Task 8: Re-run MNE-BIDS-Pipeline

**Files:**
- Modify: `eeg_pipeline/utils/config/eeg_config.yaml:453` (`ica.manual_review_complete`)
- Writes: `/Volumes/KINGSTON/EEG_fMRI_data/derivatives/preprocessed/eeg`

**Interfaces:**
- Consumes: the cleaned BIDS root from Task 6, the null `notch_freq` from Task 4.
- Produces: rebuilt derivatives that Tasks 9 and 10 read.

- [ ] **Step 1: Preserve the current derivatives under the superseded convention**

```bash
mv /Volumes/KINGSTON/EEG_fMRI_data/derivatives/preprocessed/eeg \
   /Volumes/KINGSTON/EEG_fMRI_data/derivatives/preprocessed/_superseded_eeg_20260802
printf 'Replaced 2026-08-02 by the re-run with mains moved to spectrum_fit and four\nnew isolated lines. See docs/superpowers/plans/2026-08-02-line-and-notch-cleanup.md\n' \
  > /Volumes/KINGSTON/EEG_fMRI_data/derivatives/preprocessed/_superseded_eeg_20260802/README.txt
```

Renamed, not deleted — the repo's convention for a superseded generation.

- [ ] **Step 2: Reopen the ICA manual review**

In `eeg_pipeline/utils/config/eeg_config.yaml`, set:

```yaml
  manual_review_complete: false  # Set true only after reviewing *_proc-ica_components.tsv.
```

- [ ] **Step 3: Detect bad channels first**

```bash
.venv/bin/eeg-pipeline preprocessing bad-channels
```

Expected: a pyprep log per participant. Detection is pyprep-only and `preprocessing ica` never runs it, so skipping this leaves the run on a stale bad-channel set from the previous generation.

- [ ] **Step 4: Fit ICA**

```bash
.venv/bin/eeg-pipeline preprocessing ica
```

`preprocessing` takes a required positional mode — one of `bad-channels`, `ica`, `epochs`.
There is no combined invocation, so this is two commands and not one, and the epoch half
cannot run yet: with `ica.require_manual_review: true` and `manual_review_complete: false`,
`_get_steps_for_mode` raises for `epochs` (`preprocessing.py:442-449`). Epochs are
therefore produced in Step 6, after the review in Step 5 — the review is a precondition of
epoching, not a follow-up to it. Steps 7 and 8 read `*_epo.fif` and so follow Step 6 too.

- [ ] **Step 5: Review ICA components for all 15 participants**

Open each `*_proc-ica_components.tsv` and confirm the cardiac and ocular exclusions. The CTPS cardiac review runs against the recorded ECG and provides the automated baseline; the manual pass adjusts it. Set `manual_review_complete: true` only once all 15 are done.

This step gates the next one and cannot be automated: ICA is refit from scratch on the
newly cleaned data, so component indices from the previous generation do not carry over.

- [ ] **Step 6: Create epochs**

```bash
.venv/bin/eeg-pipeline preprocessing epochs
```

Expected: 15 participants, 6 runs each, 66 epochs per participant.

- [ ] **Step 7: Confirm mains was removed exactly once**

```bash
.venv/bin/python -c "
import mne, numpy as np
mne.set_log_level('ERROR')
p = '/Volumes/KINGSTON/EEG_fMRI_data/derivatives/preprocessed/eeg/sub-0000/eeg/sub-0000_task-thermalactive_epo.fif'
ep = mne.read_epochs(p, preload=True)
from studies.pain_study.analysis.line_comb import diagnosis as hd
d = ep.copy().pick('eeg').get_data(copy=False)
keep = hd.tr_commensurate_length(d.shape[-1], ep.info['sfreq'])
f, psd = hd.hann_periodogram(d[..., :keep], ep.info['sfreq'])
s = hd.to_db(np.median(psd.mean(axis=0), axis=0))
bg = hd.local_background_db(s, half_width_bins=100)
i = int(round(60.0 / (f[1] - f[0])))
print(f'60 Hz vs background: {s[i] - bg[i]:+.2f} dB')
width = sum(1 for j in range(i - 40, i + 41) if np.isfinite(bg[j]) and s[j] - bg[j] < -1)
print(f'bins more than 1 dB low near mains: {width} ({width * (f[1] - f[0]):.3f} Hz)')
"
```

Expected: 60 Hz within about 1 dB of background, and the low-bin span around 0.13 Hz rather than the 0.97 Hz the FIR notch left. A span still near 0.97 Hz means both notches ran.

- [ ] **Step 8: Check the bad-channel set, and Cz in sub-0011 specifically**

```bash
.venv/bin/python -c "
import mne
from pathlib import Path
mne.set_log_level('ERROR')
root = Path('/Volumes/KINGSTON/EEG_fMRI_data/derivatives/preprocessed/eeg')
total = 0
for d in sorted(root.glob('sub-*')):
    if d.name.startswith('._'):
        continue
    ep = mne.read_epochs(d / 'eeg' / f'{d.name}_task-thermalactive_epo.fif', preload=False)
    bads = list(ep.info['bads'])
    total += len(bads)
    flag = '  <-- midline, reference-adjacent' if 'Cz' in bads else ''
    print(f'{d.name}: {len(bads)} {bads}{flag}')
print(f'total {total}')
"
```

The previous generation had 16 bad channels across 9 of the 14 analysed participants,
including **Cz in sub-0011**. pyprep re-detecting on cleaned data may change the set.
Interpolation policy does not change — bad channels stay excluded and logged, because an
interpolated channel is a weighted sum of its neighbours and any connectivity computed on
it is partly set by the interpolation. If Cz is still bad in sub-0011, record it in the
commit message: it is the one channel whose loss constrains midline analyses, and the
decision to leave it uninterpolated should be visible rather than implicit.

- [ ] **Step 9: Commit the config change**

```bash
git add eeg_pipeline/utils/config/eeg_config.yaml
git commit -m "chore(ica): reopen manual review for the re-run

ICA is refit from scratch on the newly cleaned data, so component indices from
the previous generation do not carry over."
```

---

### Task 9: Verify the re-run and record the after-table

**Files:**
- Output: `outputs/line_comb_mains/band_costs_after.csv`

**Interfaces:**
- Consumes: `studies.pain_study.scripts.band_audit_report.main` from Task 7.

- [ ] **Step 1: Measure the rebuilt derivatives**

```bash
.venv/bin/python -m studies.pain_study.scripts.band_audit_report \
  --deriv-root /Volumes/KINGSTON/EEG_fMRI_data/derivatives/preprocessed/eeg \
  --out outputs/line_comb_mains/band_costs_after.csv
```

- [ ] **Step 2: Compare before and after**

```bash
.venv/bin/python -c "
import pandas as pd
a = pd.read_csv('outputs/line_comb_mains/band_costs_before.csv').groupby('band').median(numeric_only=True)
b = pd.read_csv('outputs/line_comb_mains/band_costs_after.csv').groupby('band').median(numeric_only=True)
out = pd.concat({'before': a, 'after': b}, axis=1).round(2)
print(out.to_string())
"
```

Expected:
- `gamma 45-58` independent falls from 4.78 toward zero.
- `notch gap 58-62` hole falls from 34.50 to roughly the level of neighbouring bands (the comb nulls remain; only the 0.97 Hz mains footprint goes).
- `gamma 30.1-45` independent stays 0.00 and its hole does not rise — nothing was added inside that band and nothing should have been taken from it.
- `top 95-100` independent falls from 7.44.
- delta, theta and alpha are unchanged in every column. No task in this plan touches them; movement there means something unintended happened.

- [ ] **Step 3: Commit the comparison**

```bash
git add outputs/line_comb_mains/band_costs_after.csv
git commit -m "test(audit): record per-band costs after the re-run"
```

---

### Task 10: Measure what CSD buys, before wiring it into anything

The spec asks for CSD as a parallel gamma power variant. It is **not** reachable by adding
a key to `feature_engineering.spatial_transform_per_family`: that map is consulted with
`family in per_family` using the family name the caller passes
(`eeg_pipeline/analysis/features/preparation.py:180-189`), and the families are fixed by
`feature_engineering.feature_categories`. A `power_csd` entry there would be read by
nothing. Making CSD a first-class family means adding it to the category list, the compute
path and the output schema — a sub-project, and one not worth starting for a transform
that has not yet shown it helps on this cohort.

So this task measures, and the wiring is deferred behind the measurement. That is also the
order the spec asks for: "CSD must earn its place."

**Files:**
- Create: `studies/pain_study/scripts/csd_muscle_check.py`
- Create: `tests/scripts/test_csd_muscle_check.py`
- Modify: `eeg_pipeline/utils/config/eeg_config.yaml:718-734` (comment only, recording why `power` stays `"none"`)

**Interfaces:**
- Consumes: rebuilt derivatives from Task 8; `hd.band_power_db`, `hd.line_exclusion_windows`, `hd.hann_periodogram`, `hd.tr_commensurate_length`.
- Produces: `main(argv: list[str] | None = None) -> None`, writing one row per participant with columns `voltage_<band>_r` and `csd_<band>_r` for `gamma_low`, `gamma_mid`, `gamma_high`.

- [ ] **Step 1: Write the failing test**

```python
"""The CSD arm must interpolate; the voltage arm must not."""

from __future__ import annotations

import numpy as np
import pytest

from studies.pain_study.scripts import csd_muscle_check as cmc


def _epochs(n_epochs=8, n_channels=32, sfreq=500.0, n_times=11000, bads=("C1",)):
    import mne

    montage = mne.channels.make_standard_montage("standard_1020")
    names = [n for n in montage.ch_names if n.isalnum() or True][:n_channels]
    info = mne.create_info(names, sfreq, "eeg")
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1e-6, (n_epochs, n_channels, n_times))
    epochs = mne.EpochsArray(data, info, verbose="ERROR")
    epochs.set_montage(montage, on_missing="ignore")
    epochs.info["bads"] = list(bads)
    return epochs


def test_the_voltage_arm_excludes_bad_channels():
    epochs = _epochs()
    kept = cmc.voltage_arm(epochs).ch_names
    assert "C1" not in kept, "voltage-space power keeps the pipeline's no-interpolation rule"


def test_the_csd_arm_interpolates_so_the_spline_has_no_gap():
    epochs = _epochs()
    kept = cmc.csd_arm(epochs).ch_names
    assert "C1" in kept, (
        "CSD fits a spherical spline over the montage; a missing channel distorts its "
        "neighbourhood, so bad channels are interpolated inside this arm only"
    )


def test_the_two_arms_disagree_on_channel_count_when_there_are_bads():
    epochs = _epochs()
    assert len(cmc.csd_arm(epochs).ch_names) > len(cmc.voltage_arm(epochs).ch_names)


def test_no_bads_means_the_arms_agree_on_channels():
    epochs = _epochs(bads=())
    assert set(cmc.csd_arm(epochs).ch_names) == set(cmc.voltage_arm(epochs).ch_names)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/scripts/test_csd_muscle_check.py -v`
Expected: FAIL — `ImportError: cannot import name 'csd_muscle_check'`

- [ ] **Step 3: Write the validation script**

```python
"""Does CSD actually reduce the muscle confound on this cohort?

Measures, per participant, the across-channel correlation between a channel's muscle index
(62-95 Hz power over 8-30 Hz power, lines masked out of both) and its pain-minus-warm
gamma change, once in voltage space and once after CSD. If CSD does not lower it, the
transform has bought nothing here and that is the finding.

CSD fits a spherical spline over the montage, so a missing channel distorts its
neighbourhood. Bad channels are therefore interpolated inside the CSD branch only; the
voltage-space arm keeps them excluded, as the pipeline does everywhere else.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from studies.pain_study.analysis.line_comb import diagnosis as hd

LINES = (23.7776, 29.6854, 46.5839, 57.1925, 59.0168, 61.0353, 61.4039, 99.5982)
MASK = hd.line_exclusion_windows(LINES, half_width_hz=0.25)
BANDS = {"gamma_low": (30.1, 45.0), "gamma_mid": (45.0, 58.0), "gamma_high": (62.0, 95.0)}


def _band_power(freqs, spectrum, low, high):
    return 10.0 ** (
        hd.band_power_db(freqs, spectrum, low_hz=low, high_hz=high, excluded_hz=MASK) / 10.0
    )


def _spectra(epochs, pain):
    data = epochs.get_data(copy=False)
    keep = hd.tr_commensurate_length(data.shape[-1], epochs.info["sfreq"])
    freqs, psd = hd.hann_periodogram(data[..., :keep], epochs.info["sfreq"])
    return freqs, psd[pain == 1].mean(axis=0), psd[pain == 0].mean(axis=0), psd.mean(axis=0)


def voltage_arm(epochs):
    """Voltage-space epochs, bad channels excluded as everywhere else in the pipeline."""
    return epochs.copy().pick("eeg", exclude="bads")


def csd_arm(epochs):
    """CSD epochs, bad channels interpolated first.

    CSD fits a spherical spline over the montage, so a gap distorts its neighbourhood --
    degrading exactly the participants that already have problems. The interpolation is
    confined to this arm; the voltage arm keeps the pipeline's no-interpolation rule.
    """
    filled = epochs.copy().pick("eeg").interpolate_bads(reset_bads=True)
    return mne.preprocessing.compute_current_source_density(
        filled, lambda2=1.0e-5, stiffness=4.0
    )


def _correlations(epochs, pain) -> dict[str, float]:
    freqs, painful, warm, overall = _spectra(epochs, pain)
    index, contrast = [], {band: [] for band in BANDS}
    for c in range(overall.shape[0]):
        index.append(
            _band_power(freqs, overall[c], 62.0, 95.0) / _band_power(freqs, overall[c], 8.0, 30.0)
        )
        for band, (low, high) in BANDS.items():
            contrast[band].append(
                10.0 * np.log10(
                    _band_power(freqs, painful[c], low, high)
                    / _band_power(freqs, warm[c], low, high)
                )
            )
    index = np.asarray(index)
    return {band: float(pearsonr(index, np.asarray(v))[0]) for band, v in contrast.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deriv-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude", nargs="*", default=["sub-0008"])
    args = parser.parse_args(argv)

    mne.set_log_level("ERROR")
    rows = []
    subjects = sorted(
        p.name
        for p in args.deriv_root.glob("sub-*")
        if p.is_dir() and not p.name.startswith("._") and p.name not in args.exclude
    )
    for subject in subjects:
        folder = args.deriv_root / subject / "eeg"
        epochs = mne.read_epochs(
            folder / f"{subject}_task-thermalactive_epo.fif", preload=True, verbose="ERROR"
        )
        events = pd.read_csv(folder / f"{subject}_task-thermalactive_events.tsv", sep="\t")
        pain = events["pain_binary_coded"].to_numpy()

        entry = {"subject": subject.replace("sub-", "")}
        for arm, data in (("voltage", voltage_arm(epochs)), ("csd", csd_arm(epochs))):
            for band, r in _correlations(data, pain).items():
                entry[f"{arm}_{band}_r"] = r
        rows.append(entry)
        print(f"  {subject} done", flush=True)

    frame = pd.DataFrame(rows).set_index("subject")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out)
    print(frame.round(3).to_string())
    print()
    for band in BANDS:
        v, c = frame[f"voltage_{band}_r"], frame[f"csd_{band}_r"]
        print(
            f"  {band:11s} voltage median r {v.median():+.3f} "
            f"({int((v > 0).sum())}/{len(v)} positive)   "
            f"CSD median r {c.median():+.3f} ({int((c > 0).sum())}/{len(c)} positive)"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/scripts/test_csd_muscle_check.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Record why voltage-space power stays as it is**

In `eeg_pipeline/utils/config/eeg_config.yaml`, extend the comment on the `power: "none"` line inside `spatial_transform_per_family`:

```yaml
    power: "none"            # Amplitude features - CSD changes interpretation
    # Kept "none" deliberately, and CSD is not added here as a sibling key: this map is
    # consulted by family name from feature_categories, so an invented key would be read
    # by nothing. Scalp muscle spreads far from its generators and a pain-minus-warm gamma
    # change tracks each channel's muscle index in 12 of 14 participants (median
    # r = +0.34 / +0.28 / +0.19 for 30.1-45 / 45-58 / 62-95 Hz), which is what CSD would
    # be for. Whether it helps here is measured by
    # studies/pain_study/scripts/csd_muscle_check.py. Wiring CSD in as a first-class
    # feature family is deferred until that measurement says it is worth the change.
```

- [ ] **Step 6: Run the validation**

```bash
.venv/bin/python -m studies.pain_study.scripts.csd_muscle_check \
  --deriv-root /Volumes/KINGSTON/EEG_fMRI_data/derivatives/preprocessed/eeg \
  --out outputs/line_comb_mains/csd_muscle_check.csv
```

Expected: the voltage arm reproduces roughly the audit's median r of +0.34 / +0.28 / +0.19 with 12 of 14 participants positive. The CSD arm is the measurement being made — it has no expected value.

Record the result either way. Do not tune `lambda2` or `stiffness` to manufacture a reduction; those values come from the config and changing them to reach a wanted answer would make the measurement meaningless.

**This measurement decides whether a follow-up plan is written** to add CSD as a first-class feature family. If the CSD correlations are not lower than the voltage ones, that plan is not written, and the finding is that CSD does not address the muscle confound on this cohort.

- [ ] **Step 7: Commit**

```bash
git add eeg_pipeline/utils/config/eeg_config.yaml studies/pain_study/scripts/csd_muscle_check.py tests/scripts/test_csd_muscle_check.py outputs/line_comb_mains/csd_muscle_check.csv
git commit -m "feat(csd): measure whether CSD reduces the muscle confound

Reports the across-channel correlation between the muscle index and the
pain-minus-warm contrast in a voltage arm and a CSD arm. Bad channels are
interpolated in the CSD arm only, because the spline needs no gap; the voltage
arm keeps the pipeline's no-interpolation rule.

No config family is added: spatial_transform_per_family is keyed by the names in
feature_categories, so a power_csd entry would be read by nothing. Wiring CSD in
properly is deferred behind this measurement."
```

---

## Verification

Targeted subsets only; the full suite takes about 9 minutes.

```bash
.venv/bin/python -m pytest tests/analysis/test_band_audit.py tests/analysis/line_comb tests/scripts/line_comb tests/config/test_config_coherence.py -v
```

`tests/config/test_config_loader_paths.py` fails at HEAD on this machine for external-drive path reasons unrelated to this work.

## What this plan does not do

Nothing here improves 1-45 Hz. Delta, theta, alpha and gamma_low carry no removable lines (0.00%, 0.00%, 0.00%, 0.00%; beta 0.73%), and their contamination is cardiac-locked: 2.41x, 5.22x, 7.54x, 7.57x and 3.22x a circular-shift null. That is Stage 2 of the spec and it gets its own plan.
