"""The single expensive pass must also record what a cohort will need.

``measure_runs`` reads each run once, applies the ICA and measures everything. Anything a
cohort needs that is not captured there has to be bought again from a gigabyte of filtered
raw per participant, so this file pins that the sensor positions, the bad-channel record,
the acquisition date, and the posterior rhythm either side of the exclusions all come out
of that one pass.
"""

from __future__ import annotations

import matplotlib
import mne
import numpy as np
import pytest

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.run_evidence import measure_runs  # noqa: E402
from eeg_pipeline.preprocessing.report.scanner import (  # noqa: E402
    VOLUME_MARKER_DESCRIPTION,
)
from eeg_pipeline.preprocessing.report.settings import ReportSettings  # noqa: E402

SFREQ = 200.0
TR = 1.0
DURATION = 120.0
N_CHANNELS = 8


def _raw(*, with_markers: bool, locked_amplitude=4e-6, seed=0):
    rng = np.random.default_rng(seed)
    info = mne.create_info([f"C{index}" for index in range(N_CHANNELS)], SFREQ, "eeg")
    n_samples = int(DURATION * SFREQ)
    data = rng.normal(0, 1e-5, (N_CHANNELS, n_samples))
    epoch_samples = int(round(TR * SFREQ))
    onsets = np.arange(0.0, DURATION - 2 * TR, TR)
    if with_markers:
        latency = np.arange(epoch_samples)
        waveform = locked_amplitude * np.sin(2 * np.pi * latency / epoch_samples)
        for onset in onsets:
            start = int(round(onset * SFREQ))
            data[:, start : start + epoch_samples] += waveform
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    if with_markers:
        raw.set_annotations(
            mne.Annotations(
                onset=onsets,
                duration=0.0,
                description=[VOLUME_MARKER_DESCRIPTION] * len(onsets),
            )
        )
    return raw


class _NullIca:
    """An ICA that excludes nothing, so before and after are the same recording."""

    exclude: list[int] = []

    def apply(self, raw, *, exclude=None, verbose=None):
        return raw


def _write_run(tmp_path, name: str, raw) -> None:
    raw.save(tmp_path / f"{name}_proc-filt_raw.fif", overwrite=True, verbose="ERROR")


def test_the_measuring_pass_records_where_the_sensors_were(tmp_path) -> None:
    """Taken while the recording is open, because a cohort topography needs the real ones.

    Recovering a position later from a montage name would place every electrode plausibly
    and, wherever the name did not match the cap, silently wrongly -- which is the one
    failure a topography must not have. Reading the run again to get them would cost more
    than every measurement in this pass put together.
    """
    raw = _raw(with_markers=False)
    for index in range(1, len(raw.ch_names)):
        raw.info["chs"][index]["loc"][:3] = (index * 0.01, 0.02, 0.03)
    # The first channel is left at the origin, which is not a position: it is a channel
    # whose position was never digitised, and storing it would stack every such sensor on
    # top of the others in the middle of the head.
    _write_run(tmp_path, "sub-0014_task-x_run-1", raw)

    evidence = measure_runs(
        filtered_raw_paths=sorted(tmp_path.glob("*_proc-filt_raw.fif")),
        ica=_NullIca(),
        settings=ReportSettings(),
    )

    assert set(evidence.channel_positions) == set(raw.ch_names[1:])
    assert evidence.channel_positions[raw.ch_names[1]] == pytest.approx((0.01, 0.02, 0.03))


def test_a_recording_with_no_digitised_positions_records_none(tmp_path) -> None:
    _write_run(tmp_path, "sub-0014_task-x_run-1", _raw(with_markers=False))

    evidence = measure_runs(
        filtered_raw_paths=sorted(tmp_path.glob("*_proc-filt_raw.fif")),
        ica=_NullIca(),
        settings=ReportSettings(),
    )

    assert evidence.channel_positions == {}


def test_the_measuring_pass_records_the_bad_channels_of_each_run(tmp_path) -> None:
    """Per run, so a cohort can take the union its own sync policy implies."""
    raw = _raw(with_markers=False)
    raw.info["bads"] = [raw.ch_names[0]]
    _write_run(tmp_path, "sub-0014_task-x_run-1", raw)

    evidence = measure_runs(
        filtered_raw_paths=sorted(tmp_path.glob("*_proc-filt_raw.fif")),
        ica=_NullIca(),
        settings=ReportSettings(),
    )

    assert evidence.bad_channels_by_run == {"sub-0014_task-x_run-1": (raw.ch_names[0],)}


def test_an_anonymised_recording_carries_no_acquisition_date(tmp_path) -> None:
    """An absent date must not arrive in a drift panel as some default day."""
    _write_run(tmp_path, "sub-0014_task-x_run-1", _raw(with_markers=False))

    evidence = measure_runs(
        filtered_raw_paths=sorted(tmp_path.glob("*_proc-filt_raw.fif")),
        ica=_NullIca(),
        settings=ReportSettings(),
    )

    assert evidence.acquisition_date is None


def _posterior_raw(*, alpha_amplitude: float, seed: int = 3):
    """A recording with posterior sensors and an injected alpha rhythm on them."""
    rng = np.random.default_rng(seed)
    names = ["Fp1", "Cz", "P3", "Pz", "P4", "O1", "Oz", "O2"]
    info = mne.create_info(names, SFREQ, "eeg")
    n_samples = int(DURATION * SFREQ)
    data = rng.normal(0, 1e-5, (len(names), n_samples))
    times = np.arange(n_samples) / SFREQ
    for index, name in enumerate(names):
        if name.startswith(("P", "O")):
            data[index] += alpha_amplitude * np.sin(2 * np.pi * 10.0 * times)
    return mne.io.RawArray(data, info, verbose="ERROR")


def test_the_rhythm_is_measured_on_both_sides_of_the_exclusions(tmp_path) -> None:
    """The paired preservation panel is unanswerable without a before, and only this pass has one.

    The preservation section measures alpha on the cleaned epochs. There is no pre-ICA
    epochs file, so that measurement has no counterpart and cannot say whether cleaning
    cost a participant their rhythm -- which is the question the cohort panel exists for.
    Here the same run exists both before and after, so one estimator produces both sides.
    """
    _write_run(tmp_path, "sub-0014_task-x_run-1", _posterior_raw(alpha_amplitude=3e-5))

    evidence = measure_runs(
        filtered_raw_paths=sorted(tmp_path.glob("*_proc-filt_raw.fif")),
        ica=_NullIca(),
        settings=ReportSettings(),
    )

    assert len(evidence.posterior_alpha_before) == 1
    assert len(evidence.posterior_alpha_after) == 1
    before = evidence.posterior_alpha_before[0]
    assert before.is_resolvable()
    assert before.peak_frequency_hz == pytest.approx(10.0, abs=1.0)
    # This ICA excludes nothing, so cleaning cost the rhythm exactly nothing.
    assert evidence.posterior_alpha_after[0].prominence_db == pytest.approx(
        before.prominence_db
    )


def test_a_recording_with_no_posterior_sensors_carries_no_rhythm(tmp_path) -> None:
    """A montage with nothing posterior has no alpha to measure, and reports none."""
    _write_run(tmp_path, "sub-0014_task-x_run-1", _raw(with_markers=False))

    evidence = measure_runs(
        filtered_raw_paths=sorted(tmp_path.glob("*_proc-filt_raw.fif")),
        ica=_NullIca(),
        settings=ReportSettings(),
    )

    assert evidence.posterior_alpha_before == []
    assert evidence.posterior_alpha_after == []


def test_a_recording_with_no_rhythm_is_not_credited_with_one(tmp_path) -> None:
    """The argmax of noise is still an argmax, so resolvability is what must say no."""
    _write_run(tmp_path, "sub-0014_task-x_run-1", _posterior_raw(alpha_amplitude=0.0))

    evidence = measure_runs(
        filtered_raw_paths=sorted(tmp_path.glob("*_proc-filt_raw.fif")),
        ica=_NullIca(),
        settings=ReportSettings(),
    )

    assert not evidence.posterior_alpha_before[0].is_resolvable()
