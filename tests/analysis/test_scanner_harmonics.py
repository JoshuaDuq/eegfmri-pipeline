from __future__ import annotations

import numpy as np
import pytest

from eeg_pipeline.analysis.qc.scanner_harmonics import (
    DEFAULT_GAMMA_EXCLUSIONS,
    DEFAULT_GAMMA_WINDOW,
    FrequencyWindow,
    build_frequency_mask,
    summarize_scanner_harmonics,
)
from studies.pain_study.scanner_contamination import (
    SCANNER_CLEAN_GAMMA_RANGES_HZ,
    SCANNER_HARMONIC_EXCLUSION_RANGES_HZ,
)


def test_gamma_mask_excludes_scanner_harmonic_windows() -> None:
    freqs = np.arange(1.0, 101.0)

    mask = build_frequency_mask(
        freqs,
        include=DEFAULT_GAMMA_WINDOW,
        exclusions=DEFAULT_GAMMA_EXCLUSIONS,
    )

    included = set(freqs[mask])
    assert 35.0 in included
    assert 50.0 in included
    assert 70.0 in included
    assert 25.0 not in included
    assert 40.0 not in included
    assert 61.0 not in included
    assert 78.0 not in included
    assert 90.0 not in included


def test_scanner_harmonic_defaults_match_study1_clean_gamma_definition() -> None:
    clean_ranges = tuple(SCANNER_CLEAN_GAMMA_RANGES_HZ.values())

    assert (DEFAULT_GAMMA_WINDOW.low_hz, DEFAULT_GAMMA_WINDOW.high_hz) == (30.1, 80.0)
    assert tuple((window.low_hz, window.high_hz) for window in DEFAULT_GAMMA_EXCLUSIONS) == (
        SCANNER_HARMONIC_EXCLUSION_RANGES_HZ
    )
    assert clean_ranges == ((30.1, 38.0), (43.0, 56.0), (67.0, 77.0))


def test_scanner_harmonic_summary_reports_masked_gamma_and_peak_prominence() -> None:
    freqs = np.arange(1.0, 101.0)
    psd = np.ones((3, freqs.size), dtype=float)
    psd[:, freqs == 41.0] = 1000.0
    psd[:, freqs == 61.0] = 2000.0
    psd[:, freqs == 78.0] = 1500.0

    summary = summarize_scanner_harmonics(
        freqs=freqs,
        psd=psd,
        source_file="sub-0008_task-pain_run-01_eeg.vhdr",
        sfreq=1000.0,
        n_samples=10_000,
        channel_names=["Fz", "Cz", "Pz"],
    )

    assert summary["subject"] == "0008"
    assert summary["task"] == "pain"
    assert summary["run"] == "01"
    assert summary["n_channels"] == 3
    assert summary["gamma_masked_power_db"] < summary["gamma_full_power_db"]
    assert summary["gamma_masked_ranges_hz"] == "30.1-38;43-56;67-77"
    assert summary["harmonic_38_43_peak_hz"] == pytest.approx(41.0)
    assert summary["harmonic_38_43_prominence_db"] > 20.0


def test_scanner_harmonic_summary_pairs_peak_power_with_reported_peak_frequency() -> None:
    freqs = np.arange(1.0, 101.0)
    psd = np.ones((3, freqs.size), dtype=float)
    psd[:, freqs == 39.0] = 10 ** (5.0 / 10.0)
    psd[:, freqs == 41.0] = 10 ** (6.0 / 10.0)
    psd[:, freqs == 42.0] = 10 ** (5.9 / 10.0)
    psd[:, freqs == 43.0] = 10 ** (5.8 / 10.0)

    summary = summarize_scanner_harmonics(
        freqs=freqs,
        psd=psd,
        source_file="sub-0008_task-pain_run-01_eeg.vhdr",
        sfreq=1000.0,
        n_samples=10_000,
        channel_names=["Fz", "Cz", "Pz"],
        harmonic_windows=(FrequencyWindow("scanner_38_43", 38.0, 43.0),),
    )

    assert summary["harmonic_38_43_peak_hz"] == pytest.approx(39.0)
    assert summary["harmonic_38_43_peak_power_db"] == pytest.approx(5.0)


def test_frequency_window_rejects_invalid_range() -> None:
    with pytest.raises(ValueError, match="low_hz must be less than high_hz"):
        FrequencyWindow("bad", 80.0, 30.0)
