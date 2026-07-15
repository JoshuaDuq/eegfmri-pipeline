from __future__ import annotations

import mne
import numpy as np
import pytest

from eeg_pipeline.analysis.qc.scanner_harmonic_comb import (
    ParticipantSpectrum,
    ScannerCombParameters,
    Spectrum,
    combine_participant_runs,
    compute_epoch_comb_spectrum,
    compute_raw_comb_spectrum,
    summarize_scanner_comb,
)


def _signal(sfreq: float, duration_seconds: float) -> np.ndarray:
    times = np.arange(int(sfreq * duration_seconds)) / sfreq
    return 2e-6 * np.sin(2.0 * np.pi * 20.0 * times) + 1e-6 * np.sin(2.0 * np.pi * 41.0 * times)


def _raw(sfreq: float = 400.0) -> mne.io.RawArray:
    signal = _signal(sfreq, 8.0)
    data = np.vstack([signal, 0.8 * signal, 0.2 * signal])
    info = mne.create_info(
        ["Fz", "Cz", "ECG"],
        sfreq=sfreq,
        ch_types=["eeg", "eeg", "ecg"],
    )
    return mne.io.RawArray(data, info, verbose=False)


def _epochs(sfreq: float = 200.0) -> mne.EpochsArray:
    signal = _signal(sfreq, 4.0)
    epoch = np.vstack([signal, 0.8 * signal, 0.2 * signal])
    data = np.stack([epoch, 1.1 * epoch])
    info = mne.create_info(
        ["Fz", "Cz", "ECG"],
        sfreq=sfreq,
        ch_types=["eeg", "eeg", "ecg"],
    )
    return mne.EpochsArray(data, info, verbose=False)


def _parameters(**overrides: object) -> ScannerCombParameters:
    values = {
        "frequency_min_hz": 15.0,
        "frequency_max_hz": 90.0,
        "welch_duration_seconds": 2.0,
        "frequency_resolution_hz": 0.5,
        "bootstrap_resamples": 100,
        "confidence_level": 0.95,
        "random_seed": 7,
    }
    values.update(overrides)
    return ScannerCombParameters(**values)


def _participant(participant: str, offset_db: float) -> ParticipantSpectrum:
    frequencies = np.arange(15.0, 90.5, 0.5)
    power = offset_db + np.sin(frequencies / 10.0)
    for frequency in (20.0, 41.0, 61.0, 82.0):
        power[np.argmin(np.abs(frequencies - frequency))] += 10.0
    return ParticipantSpectrum(participant, frequencies, power)


def test_raw_and_epochs_use_identical_frequency_grid() -> None:
    parameters = _parameters()

    raw_spectrum = compute_raw_comb_spectrum(_raw(), parameters)
    epoch_spectrum = compute_epoch_comb_spectrum(_epochs(), parameters)

    np.testing.assert_array_equal(
        raw_spectrum.frequencies_hz,
        epoch_spectrum.frequencies_hz,
    )
    assert raw_spectrum.power_db.shape == raw_spectrum.frequencies_hz.shape
    assert epoch_spectrum.power_db.shape == epoch_spectrum.frequencies_hz.shape


def test_epoch_spectrum_requires_retained_epochs() -> None:
    epochs = _epochs()[[]]

    with pytest.raises(ValueError, match="at least one retained epoch"):
        compute_epoch_comb_spectrum(epochs, _parameters())


def test_spectra_exclude_channels_marked_bad() -> None:
    raw = _raw()
    raw.info["bads"] = ["Cz"]

    observed = compute_raw_comb_spectrum(raw, _parameters())

    expected_raw = raw.copy().pick(["Fz"])
    expected = compute_raw_comb_spectrum(expected_raw, _parameters())
    np.testing.assert_allclose(observed.power_db, expected.power_db)


def test_combine_participant_runs_weights_runs_equally() -> None:
    frequencies = np.array([20.0, 21.0])
    runs = [
        Spectrum(frequencies, np.array([0.0, 2.0])),
        Spectrum(frequencies, np.array([10.0, 6.0])),
    ]

    combined = combine_participant_runs("0001", runs)

    np.testing.assert_allclose(combined.power_db, [5.0, 4.0])


def test_cohort_summary_weights_participants_equally() -> None:
    inputs = [_participant("0001", 0.0), _participant("0002", 10.0)]
    finals = [_participant("0001", -4.0), _participant("0002", 6.0)]

    summary = summarize_scanner_comb(inputs, finals, _parameters())

    expected_input = np.median(np.stack([item.power_db for item in inputs]), axis=0)
    expected_final = np.median(np.stack([item.power_db for item in finals]), axis=0)
    np.testing.assert_allclose(summary.input_median_db, expected_input)
    np.testing.assert_allclose(summary.final_median_db, expected_final)
    assert summary.participant_ids == ("0001", "0002")
    assert len(summary.harmonic_frequencies_hz) == 4


def test_bootstrap_is_deterministic() -> None:
    inputs = [_participant(f"{index:04d}", float(index)) for index in range(1, 5)]
    finals = [_participant(f"{index:04d}", float(index) - 3.0) for index in range(1, 5)]

    first = summarize_scanner_comb(inputs, finals, _parameters())
    second = summarize_scanner_comb(inputs, finals, _parameters())

    np.testing.assert_array_equal(first.input_ci_low_db, second.input_ci_low_db)
    np.testing.assert_array_equal(first.final_ci_high_db, second.final_ci_high_db)


def test_cohort_summary_requires_exact_participant_pairing() -> None:
    with pytest.raises(ValueError, match="identical participants"):
        summarize_scanner_comb(
            [_participant("0001", 0.0)],
            [_participant("0002", -2.0)],
            _parameters(),
        )


def test_parameters_reject_incompatible_frequency_resolution() -> None:
    with pytest.raises(ValueError, match="frequency span"):
        _parameters(frequency_resolution_hz=0.4)
