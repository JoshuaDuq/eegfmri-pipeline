from __future__ import annotations

from dataclasses import replace

import mne
import numpy as np
import pytest

from eeg_pipeline.preprocessing.eeg_fmri.qc import (
    compare_cardiac_locked_summaries,
    summarize_cardiac_locked_eeg,
)


def _cardiac_raw(*, artifact_scale: float) -> tuple[mne.io.RawArray, np.ndarray]:
    sampling_frequency = 1_000.0
    duration_seconds = 10.0
    qrs_times = np.arange(1.0, 10.0)
    n_samples = int(duration_seconds * sampling_frequency)
    data = np.zeros((3, n_samples), dtype=float)
    cardiac_times = np.arange(-0.2, 0.6, 1.0 / sampling_frequency)
    pulse = np.exp(-(((cardiac_times - 0.11) / 0.035) ** 2))
    pulse -= 0.65 * np.exp(-(((cardiac_times - 0.24) / 0.06) ** 2))
    for qrs_time in qrs_times:
        start = int(round((qrs_time - 0.2) * sampling_frequency))
        stop = start + pulse.size
        data[0, start:stop] += artifact_scale * pulse
        data[1, start:stop] -= 0.6 * artifact_scale * pulse
    info = mne.create_info(
        ["C3", "C4", "ECG"],
        sampling_frequency,
        ["eeg", "eeg", "ecg"],
    )
    return mne.io.RawArray(data, info, verbose=False), qrs_times


def test_cardiac_locked_qc_quantifies_obs_attenuation() -> None:
    before_raw, qrs_times = _cardiac_raw(artifact_scale=4.0)
    after_raw, _ = _cardiac_raw(artifact_scale=1.0)

    before = summarize_cardiac_locked_eeg(before_raw, qrs_times=qrs_times)
    after = summarize_cardiac_locked_eeg(after_raw, qrs_times=qrs_times)
    comparison = compare_cardiac_locked_summaries(before, after)

    assert before.channel_names == ("C3", "C4")
    assert before.valid_epoch_count == qrs_times.size
    assert before.median_evoked_rms > 0
    assert before.median_evoked_peak_to_peak > 0
    np.testing.assert_allclose(before.median_evoked, 4.0 * after.median_evoked)
    assert comparison.rms_attenuation_db == pytest.approx(20.0 * np.log10(4.0))
    assert comparison.peak_to_peak_attenuation_db == pytest.approx(20.0 * np.log10(4.0))


def test_cardiac_locked_qc_rejects_insufficient_valid_epochs() -> None:
    raw, _ = _cardiac_raw(artifact_scale=1.0)

    with pytest.raises(ValueError, match="at least 3 valid epochs"):
        summarize_cardiac_locked_eeg(raw, qrs_times=np.array([0.05, 9.9]))


@pytest.mark.parametrize("field", ["channel_names", "times"])
def test_cardiac_locked_comparison_rejects_mismatched_summaries(field: str) -> None:
    raw, qrs_times = _cardiac_raw(artifact_scale=1.0)
    summary = summarize_cardiac_locked_eeg(raw, qrs_times=qrs_times)
    if field == "channel_names":
        mismatched = replace(summary, channel_names=("C3", "Cz"))
    else:
        mismatched = replace(summary, times=summary.times + 0.001)

    with pytest.raises(ValueError, match=field.replace("_", " ")):
        compare_cardiac_locked_summaries(summary, mismatched)
