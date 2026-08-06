"""Tests for the line-comb removal runner."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from eeg_pipeline.utils.config.loader import load_config
from studies.pain_study.analysis.line_comb import removal as lr
from studies.pain_study.scripts.line_comb import remove as rlc
from studies.pain_study.scripts.workflow_config import load_workflow_config


@pytest.fixture
def bids_brainvision_run(tmp_path):
    """BIDS metadata, rather than BrainVision, carries auxiliary channel types."""
    import mne
    from mne_bids import BIDSPath, write_raw_bids

    sfreq = 100.0
    info = mne.create_info(
        ["Fp1", "Cz", "ECG"],
        sfreq,
        ["eeg", "eeg", "ecg"],
    )
    raw = mne.io.RawArray(np.zeros((3, int(360 * sfreq))), info, verbose="ERROR")
    raw.set_annotations(
        mne.Annotations(
            onset=np.arange(20.0, 350.0, 30.0),
            duration=0.001,
            description=["Trig_therm/T  1"] * 11,
        )
    )
    bids_path = BIDSPath(
        subject="0001",
        task="thermalactive",
        run="1",
        datatype="eeg",
        root=tmp_path,
    )
    write_raw_bids(
        raw,
        bids_path,
        format="BrainVision",
        allow_preload=True,
        overwrite=True,
        verbose="ERROR",
    )
    return bids_path.copy().update(suffix="eeg", extension=".vhdr").fpath


def test_read_bids_raw_applies_auxiliary_channel_types(bids_brainvision_run):
    import mne

    raw = rlc.read_bids_raw(bids_brainvision_run)

    assert raw.get_channel_types() == ["eeg", "eeg", "ecg"]
    eeg_names = [raw.ch_names[index] for index in mne.pick_types(raw.info, eeg=True)]
    assert eeg_names == ["Fp1", "Cz"]


def test_study_epoch_bounds_match_the_exact_analysis_interval(bids_brainvision_run):
    raw = rlc.read_bids_raw(bids_brainvision_run)

    bounds = rlc.study_epoch_bounds(raw, rlc.RemovalSettings())

    assert len(bounds) == 11
    assert bounds[0] == (1500, 3500)
    assert {stop - start for start, stop in bounds} == {2000}


def test_study_line_metrics_use_only_eeg_and_the_exact_event_windows():
    import mne

    settings = rlc.RemovalSettings()
    sfreq = 100.0
    n_times = int(360 * sfreq)
    times = np.arange(n_times) / sfreq
    inside_study = np.zeros(n_times, dtype=bool)
    for onset in np.arange(20.0, 350.0, 30.0):
        inside_study |= (times >= onset - 5.0) & (times < onset + 15.0)
    line = np.sin(2 * np.pi * 27.7 * times) * inside_study
    before_data = np.vstack((line, line, 100.0 * line)) * 1e-6
    after_data = np.vstack((np.zeros_like(line), np.zeros_like(line), 100.0 * line)) * 1e-6
    info = mne.create_info(["Cz", "Pz", "ECG"], sfreq, ["eeg", "eeg", "ecg"])
    before = mne.io.RawArray(before_data, info, verbose="ERROR")
    after = mne.io.RawArray(after_data, info, verbose="ERROR")
    annotations = mne.Annotations(
        onset=np.arange(20.0, 350.0, 30.0),
        duration=0.001,
        description=[settings.study_event_name] * settings.expected_study_events_per_run,
    )
    before.set_annotations(annotations)
    after.set_annotations(annotations)
    estimate = _estimate_for_study_test()
    plan = rlc.RunRemovalPlan(
        model=None,
        windows=(
            rlc.AdaptiveWindowRemovalPlan(
                bounds=(0, n_times),
                estimate=estimate,
                targets_hz=(27.6,),
                notch_widths_hz=(0.06,),
                narrow_targets_hz=(),
            ),
        ),
        study_windows=tuple(
            rlc.StudyWindowRemovalPlan(
                bounds=bounds,
                targets_hz=(27.6,),
                notch_widths_hz=(0.06,),
            )
            for bounds in rlc.study_epoch_bounds(before, settings)
        ),
    )

    metrics = rlc.study_line_metrics(before, after, plan, settings)

    assert metrics["study_residual_excess_db"] <= 0.0
    assert metrics["study_focal_residual_excess_db"] <= 0.0


def test_study_samples_use_their_exact_transform_with_transitions_outside(monkeypatch):
    import mne

    sfreq = 10.0
    raw = mne.io.RawArray(
        np.zeros((1, 200)),
        mne.create_info(["Cz"], sfreq, "eeg"),
        verbose="ERROR",
    )
    estimate = _estimate_for_study_test()
    continuous_windows = (
        rlc.AdaptiveWindowRemovalPlan(
            bounds=(0, 150),
            estimate=estimate,
            targets_hz=(27.6,),
            notch_widths_hz=(0.1,),
            narrow_targets_hz=(),
        ),
        rlc.AdaptiveWindowRemovalPlan(
            bounds=(50, 200),
            estimate=estimate,
            targets_hz=(27.6,),
            notch_widths_hz=(0.1,),
            narrow_targets_hz=(),
        ),
    )
    study_window = rlc.StudyWindowRemovalPlan(
        bounds=(80, 120),
        targets_hz=(27.6,),
        notch_widths_hz=(0.1,),
    )
    plan = rlc.RunRemovalPlan(
        model=None,
        windows=continuous_windows,
        study_windows=(study_window,),
    )

    def fake_clean(segment, *args, **kwargs):
        cleaned = segment.copy()
        cleaned._data += 3.0 if segment.n_times == 40 else 1.0
        return cleaned

    monkeypatch.setattr(rlc, "clean_raw", fake_clean)

    cleaned = rlc.clean_adaptive_raw(raw, plan, rlc.RemovalSettings())

    values = cleaned.get_data()[0]
    assert values[0] == pytest.approx(1.0)
    assert np.all(np.diff(values[:81]) >= 0.0)
    assert np.allclose(cleaned.get_data()[:, 80:120], 3.0)
    assert np.all(np.diff(values[119:]) <= 0.0)
    assert values[-1] == pytest.approx(1.0)
    assert np.max(np.abs(np.diff(values))) < 0.1


def test_channel_local_projector_removes_only_planned_sinusoids():
    import mne

    sampling_frequency_hz = 100.0
    times = np.arange(1_000) / sampling_frequency_hz
    rng = np.random.default_rng(31)
    data = rng.normal(scale=0.01, size=(2, times.size))
    data[0] += np.sin(2.0 * np.pi * 20.04 * times)
    data[0] += 0.5 * np.sin(2.0 * np.pi * 30.0 * times)
    data[1] += 0.5 * np.sin(2.0 * np.pi * 20.0 * times)
    data[1] += np.sin(2.0 * np.pi * 29.96 * times)
    info = mne.create_info(["Cz", "Pz"], sampling_frequency_hz, "eeg")
    targets = ((20.0,), (30.0,))
    widths = ((0.1,), (0.1,))
    settings = rlc.RemovalSettings(
        filter_length="5s",
        mt_bandwidth=1.2,
        filter_jobs=1,
    )

    actual = rlc._clean_channel_residuals(data, info, targets, widths, settings)

    def amplitude(values, frequency_hz):
        basis = np.exp(-2j * np.pi * frequency_hz * times)
        return 2.0 * abs(np.vdot(basis, values)) / values.size

    assert amplitude(actual[0], 20.04) < 0.01 * amplitude(data[0], 20.04)
    assert amplitude(actual[1], 29.96) < 0.01 * amplitude(data[1], 29.96)
    assert amplitude(actual[0], 30.0) == pytest.approx(amplitude(data[0], 30.0), rel=0.01)
    assert amplitude(actual[1], 20.0) == pytest.approx(amplitude(data[1], 20.0), rel=0.01)


def test_study_refinement_encodes_an_unmasked_focal_line(monkeypatch):
    import mne

    raw = mne.io.RawArray(
        np.zeros((4, 200)),
        mne.create_info(["Cz", "Pz", "Fz", "Oz"], 100.0, "eeg"),
        verbose="ERROR",
    )
    estimate = _estimate_for_study_test()
    plan = rlc.RunRemovalPlan(
        model=None,
        windows=(
            rlc.AdaptiveWindowRemovalPlan(
                bounds=(0, 200),
                estimate=estimate,
                targets_hz=(27.6,),
                notch_widths_hz=(0.1,),
                narrow_targets_hz=(),
            ),
        ),
        study_windows=(
            rlc.StudyWindowRemovalPlan(
                bounds=(80, 120),
                targets_hz=(27.6,),
                notch_widths_hz=(0.1,),
            ),
        ),
    )
    calls = 0

    def fake_statistics(data, **kwargs):
        nonlocal calls
        calls += 1
        freqs = np.arange(0.0, 50.1, 0.1)
        statistic = np.zeros((4, freqs.size))
        if calls == 1:
            statistic[0, np.argmin(np.abs(freqs - 27.7))] = 20.0
        return freqs, statistic, 10.0, np.where(statistic > 10.0, 1e-9, 0.5)

    monkeypatch.setattr(rlc.lr, "thomson_f_statistics", fake_statistics)
    monkeypatch.setattr(
        rlc,
        "_clean_study_segment",
        lambda data, *args, **kwargs: np.asarray(data),
    )

    refined = rlc._refine_study_residual_plans(raw, plan, rlc.RemovalSettings())

    assert calls == 1
    assert refined.study_windows[0].channel_residual_targets_hz[0] == pytest.approx((27.7,))
    assert refined.study_windows[0].channel_residual_targets_hz[1:] == ((), (), ())
    assert refined.study_windows[0].aggregate_residual_targets_hz == ()


def test_continuous_refinement_encodes_an_unmasked_focal_line(monkeypatch):
    import mne

    raw = mne.io.RawArray(
        np.zeros((4, 200)),
        mne.create_info(["Cz", "Pz", "Fz", "Oz"], 100.0, "eeg"),
        verbose="ERROR",
    )
    estimate = _estimate_for_study_test()
    plan = rlc.RunRemovalPlan(
        model=None,
        windows=(
            rlc.AdaptiveWindowRemovalPlan(
                bounds=(0, 200),
                estimate=estimate,
                targets_hz=(27.6,),
                notch_widths_hz=(0.1,),
                narrow_targets_hz=(),
            ),
        ),
    )
    calls = 0

    def fake_statistics(data, **kwargs):
        nonlocal calls
        calls += 1
        freqs = np.arange(0.0, 50.1, 0.1)
        statistic = np.zeros((4, freqs.size))
        if calls == 1:
            statistic[0, np.argmin(np.abs(freqs - 27.7))] = 20.0
        return freqs, statistic, 10.0, np.where(statistic > 10.0, 1e-9, 0.5)

    monkeypatch.setattr(rlc.lr, "thomson_f_statistics", fake_statistics)
    monkeypatch.setattr(
        rlc,
        "_clean_planned_segment",
        lambda data, *args, **kwargs: np.asarray(data),
    )

    refined = rlc._refine_continuous_residual_plans(raw, plan, rlc.RemovalSettings())

    assert calls == 1
    assert refined.windows[0].channel_residual_targets_hz[0] == pytest.approx((27.7,))
    assert refined.windows[0].channel_residual_targets_hz[1:] == ((), (), ())
    assert refined.windows[0].aggregate_residual_targets_hz == ()


def test_planned_segment_cleaning_leaves_the_callers_array_untouched():
    """The cleaner must not consume the data it was asked to clean.

    MNE filters a RawArray in place and the array can share the caller's buffer, so a
    caller that keeps its input to compare against -- which is exactly what residual
    detection does -- would otherwise compare cleaned data with itself and see nothing.
    """
    import mne

    sampling_frequency_hz = 1000.0
    times = np.arange(20_000) / sampling_frequency_hz
    rng = np.random.default_rng(0)
    data = rng.normal(size=(2, times.size)) * 1e-6
    data += 3e-6 * np.sin(2.0 * np.pi * 57.25 * times)
    pristine = data.copy()
    info = mne.create_info(["Cz", "Pz"], sampling_frequency_hz, "eeg")
    window = rlc.StudyWindowRemovalPlan(
        bounds=(0, times.size),
        targets_hz=(57.25,),
        notch_widths_hz=(0.06,),
    )

    cleaned = rlc._clean_planned_segment(data, info, window, rlc.RemovalSettings())

    assert np.array_equal(data, pristine)
    assert not np.array_equal(cleaned, pristine)


def _single_window_plan(n_times: int) -> rlc.RunRemovalPlan:
    """One continuous window and one study window over the same 27.6 Hz target."""
    estimate = _estimate_for_study_test()
    return rlc.RunRemovalPlan(
        model=None,
        windows=(
            rlc.AdaptiveWindowRemovalPlan(
                bounds=(0, n_times),
                estimate=estimate,
                targets_hz=(27.6,),
                notch_widths_hz=(0.1,),
                narrow_targets_hz=(),
            ),
        ),
        study_windows=(
            rlc.StudyWindowRemovalPlan(
                bounds=(0, n_times),
                targets_hz=(27.6,),
                notch_widths_hz=(0.1,),
            ),
        ),
    )


def _flat_f_statistics(_data, **_kwargs):
    """A frequency grid on which nothing is a significant sinusoid."""
    freqs = np.arange(0.0, 50.1, 0.1)
    return freqs, np.zeros((2, freqs.size)), 10.0, np.full((2, freqs.size), 0.5)


def test_residual_planning_never_consults_the_preservation_gate(monkeypatch):
    """The gate judges the removal, so it must not also choose what the removal takes.

    Selecting residual targets by the acceptance tolerance removes exactly what the gate
    would flag. The gate can then only fail where the search's own subtraction fell
    short, never because a line was missed, which is not a test of the method.
    """
    import mne

    class ForbiddenGate:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Residual planning consulted the preservation gate.")

    raw = mne.io.RawArray(
        np.zeros((2, 200)),
        mne.create_info(["Cz", "Pz"], 100.0, "eeg"),
        verbose="ERROR",
    )
    plan = _single_window_plan(200)
    monkeypatch.setattr(rlc.lr, "thomson_f_statistics", _flat_f_statistics)
    monkeypatch.setattr(rlc.lr, "PreservationGate", ForbiddenGate)
    monkeypatch.setattr(rlc, "_clean_study_segment", lambda data, *a, **k: np.asarray(data))
    monkeypatch.setattr(rlc, "_clean_planned_segment", lambda data, *a, **k: np.asarray(data))
    monkeypatch.setattr(rlc, "clean_continuous_raw", lambda raw, *a, **k: raw)

    study = rlc._refine_study_residual_plans(raw, plan, rlc.RemovalSettings())
    continuous = rlc._refine_continuous_residual_plans(raw, plan, rlc.RemovalSettings())

    assert study.study_windows[0].channel_residual_targets_hz == ((), ())
    assert continuous.windows[0].channel_residual_targets_hz == ((), ())


def test_residual_planning_leaves_power_without_a_significant_sinusoid(monkeypatch):
    """Elevated power alone does not authorise a subtraction.

    A residual that carries power but is not a resolvable sinusoid -- a drifting or
    nonstationary one -- is left in place for the gate to report. Removing it because it
    exceeds the gate's tolerance is what makes the gate unfalsifiable.
    """
    import mne

    sampling_frequency_hz = 100.0
    n_times = 1_000
    times = np.arange(n_times) / sampling_frequency_hz
    rng = np.random.default_rng(0)
    original = rng.normal(size=(2, n_times))
    bump = np.zeros_like(original)
    bump[0] = 50.0 * np.sin(2.0 * np.pi * 27.7 * times)

    raw = mne.io.RawArray(
        original,
        mne.create_info(["Cz", "Pz"], sampling_frequency_hz, "eeg"),
        verbose="ERROR",
    )
    plan = _single_window_plan(n_times)
    monkeypatch.setattr(rlc.lr, "thomson_f_statistics", _flat_f_statistics)
    monkeypatch.setattr(rlc, "_clean_study_segment", lambda data, *a, **k: np.asarray(data) + bump)

    refined = rlc._refine_study_residual_plans(raw, plan, rlc.RemovalSettings())

    assert refined.study_windows[0].aggregate_residual_targets_hz == ()
    assert refined.study_windows[0].channel_residual_targets_hz == ((), ())


def test_residual_removal_never_reaches_a_channel_without_evidence(monkeypatch):
    """Even a line the whole array carries is subtracted channel by channel.

    There is no shared route. Adding a frequency to every channel's plan would have
    _clean_channel_residuals search each channel independently and subtract whatever
    fluctuation is largest there, which in a channel without the artifact may be signal.
    """
    import mne

    raw = mne.io.RawArray(
        np.zeros((2, 200)),
        mne.create_info(["Cz", "Pz"], 100.0, "eeg"),
        verbose="ERROR",
    )
    plan = _single_window_plan(200)

    def shared_statistics(_data, **_kwargs):
        freqs = np.arange(0.0, 50.1, 0.1)
        statistic = np.zeros((2, freqs.size))
        statistic[:, np.argmin(np.abs(freqs - 27.7))] = 20.0
        return freqs, statistic, 10.0, np.where(statistic > 10.0, 1e-9, 0.5)

    monkeypatch.setattr(rlc.lr, "thomson_f_statistics", shared_statistics)
    monkeypatch.setattr(rlc, "_clean_study_segment", lambda data, *a, **k: np.asarray(data))

    refined = rlc._refine_study_residual_plans(raw, plan, rlc.RemovalSettings())

    assert refined.study_windows[0].aggregate_residual_targets_hz == ()
    channels = refined.study_windows[0].channel_residual_targets_hz
    assert len(channels) == 2
    assert all(values == pytest.approx((27.7,)) for values in channels)


def test_continuous_residual_support_reaches_every_overlapping_synthesis_window():
    estimate = _estimate_for_study_test()
    windows = (
        rlc.AdaptiveWindowRemovalPlan(
            bounds=(0, 100),
            estimate=estimate,
            targets_hz=(27.6,),
            notch_widths_hz=(0.1,),
            narrow_targets_hz=(),
            channel_residual_targets_hz=((27.7,),),
            channel_residual_widths_hz=((0.05,),),
        ),
        rlc.AdaptiveWindowRemovalPlan(
            bounds=(50, 150),
            estimate=estimate,
            targets_hz=(27.6,),
            notch_widths_hz=(0.1,),
            narrow_targets_hz=(),
            channel_residual_targets_hz=((),),
            channel_residual_widths_hz=((),),
        ),
        rlc.AdaptiveWindowRemovalPlan(
            bounds=(150, 250),
            estimate=estimate,
            targets_hz=(27.6,),
            notch_widths_hz=(0.1,),
            narrow_targets_hz=(),
            channel_residual_targets_hz=((),),
            channel_residual_widths_hz=((),),
        ),
    )

    routed = rlc._route_continuous_residual_support(rlc.RunRemovalPlan(model=None, windows=windows))

    assert routed.windows[1].channel_residual_targets_hz[0] == pytest.approx((27.7,))
    assert routed.windows[2].channel_residual_targets_hz[0] == ()


def test_study_residual_f_test_reports_a_remaining_authorised_line(monkeypatch):
    import mne

    raw = mne.io.RawArray(
        np.zeros((4, 200)),
        mne.create_info(["Cz", "Pz", "Fz", "Oz"], 100.0, "eeg"),
        verbose="ERROR",
    )
    estimate = _estimate_for_study_test()
    plan = rlc.RunRemovalPlan(
        model=None,
        windows=(
            rlc.AdaptiveWindowRemovalPlan(
                bounds=(0, 200),
                estimate=estimate,
                targets_hz=(27.6,),
                notch_widths_hz=(0.1,),
                narrow_targets_hz=(),
            ),
        ),
        study_windows=(
            rlc.StudyWindowRemovalPlan(
                bounds=(80, 120),
                targets_hz=(27.6,),
                notch_widths_hz=(0.1,),
            ),
        ),
    )

    def fake_statistics(data, **kwargs):
        freqs = np.arange(0.0, 50.1, 0.1)
        statistic = np.zeros((2, freqs.size))
        statistic[1, np.argmin(np.abs(freqs - 27.7))] = 20.0
        return freqs, statistic, 10.0, np.where(statistic > 10.0, 1e-9, 0.5)

    monkeypatch.setattr(rlc.lr, "thomson_f_statistics", fake_statistics)

    metrics = rlc.study_residual_f_test_metrics(raw, plan, rlc.RemovalSettings())

    assert metrics["study_significant_focal_residual_count"] == 1
    assert metrics["study_significant_focal_residual_hz"] == "0:Pz:27.700000"


def test_study_refinement_metrics_preserve_epoch_channel_provenance():
    estimate = _estimate_for_study_test()
    plan = rlc.RunRemovalPlan(
        model=None,
        windows=(
            rlc.AdaptiveWindowRemovalPlan(
                bounds=(0, 200),
                estimate=estimate,
                targets_hz=(27.6,),
                notch_widths_hz=(0.1,),
                narrow_targets_hz=(),
            ),
        ),
        study_windows=(
            rlc.StudyWindowRemovalPlan(
                bounds=(80, 120),
                targets_hz=(27.6,),
                notch_widths_hz=(0.1,),
                aggregate_residual_targets_hz=(27.7,),
                aggregate_residual_widths_hz=(0.1,),
                channel_residual_targets_hz=((), (27.8,)),
                channel_residual_widths_hz=((), (0.1,)),
            ),
        ),
    )

    metrics = rlc.study_refinement_metrics(plan, ("Cz", "Pz"))

    assert metrics["n_study_common_targets"] == 1
    assert metrics["n_study_aggregate_refinement_targets"] == 1
    assert metrics["n_study_focal_refinement_targets"] == 1
    assert metrics["n_study_focal_refinement_channel_windows"] == 1
    assert metrics["study_aggregate_refinement_hz"] == "0:27.700000"
    assert metrics["study_focal_refinement_hz"] == "0:Pz:27.800000"


@pytest.mark.parametrize(
    "kwargs",
    (
        {
            "aggregate_residual_targets_hz": (np.nan,),
            "aggregate_residual_widths_hz": (0.1,),
        },
        {
            "aggregate_residual_targets_hz": (27.7,),
            "aggregate_residual_widths_hz": (0.0,),
        },
        {
            "channel_residual_targets_hz": ((27.7,),),
            "channel_residual_widths_hz": ((np.inf,),),
        },
    ),
)
def test_study_window_rejects_invalid_residual_targets_and_widths(kwargs):
    with pytest.raises(ValueError, match="residual targets and widths"):
        rlc.StudyWindowRemovalPlan(
            bounds=(80, 120),
            targets_hz=(27.6,),
            notch_widths_hz=(0.1,),
            **kwargs,
        )


def test_study_signal_metrics_measure_only_the_supplied_epoch_bounds():
    import mne

    sfreq = 100.0
    n_times = 4000
    times = np.arange(n_times) / sfreq
    probe = np.sin(2 * np.pi * 17.35 * times)[None, :]
    rng = np.random.default_rng(2)
    data = rng.normal(scale=1e-6, size=(2, n_times))
    probe_info = mne.create_info(["probe"], sfreq, "eeg")
    data_info = mne.create_info(["Cz", "Pz"], sfreq, "eeg")
    probe_before = mne.io.RawArray(probe, probe_info, verbose="ERROR")
    probe_after = mne.io.RawArray(0.5 * probe, probe_info, verbose="ERROR")
    data_before = mne.io.RawArray(data, data_info, verbose="ERROR")
    data_after = mne.io.RawArray(data.copy(), data_info, verbose="ERROR")

    metrics = rlc.study_signal_metrics(
        probe_before,
        probe_after,
        data_before,
        data_after,
        bounds=((500, 2500),),
        targets=(27.6,),
        guard_hz=0.2,
    )

    assert metrics["study_max_probe_deviation_db"] == pytest.approx(6.0206, abs=0.01)
    assert metrics["study_max_nonline_change_db"] == pytest.approx(0.0, abs=1e-12)


def _estimate_for_study_test():
    return lr.CombEstimate(
        fundamental_hz=1.2,
        harmonics_used=tuple(range(24, 80)),
        harmonic_positions_hz=tuple(1.2 * harmonic for harmonic in range(24, 80)),
        residual_rms_hz=0.0,
        max_abs_residual_hz=0.0,
        fundamental_jackknife_se_hz=1e-4,
        isolated_hz=(),
        isolated_prominence_db=(),
    )


def test_production_stages_never_bypass_bids_channel_metadata():
    for function in (
        rlc.build_run_plans,
        rlc.benchmark_run,
        rlc.apply_run,
        rlc.verify_cohort,
    ):
        source = inspect.getsource(function)
        assert "read_raw_brainvision" not in source
        assert "read_bids_raw" in source


@pytest.fixture
def brainvision_run(tmp_path):
    """A small BrainVision file written the way the BIDS dataset writes them."""
    import mne

    mne.set_log_level("ERROR")
    sfreq, n_times = 1000.0, 4000
    rng = np.random.default_rng(0)
    data = rng.normal(scale=2e-5, size=(4, n_times))
    info = mne.create_info(["Fp1", "Cz", "Oz", "ECG"], sfreq, ["eeg", "eeg", "eeg", "ecg"])
    raw = mne.io.RawArray(data, info)
    path = tmp_path / "sub-0001_task-thermalactive_run-1_eeg.vhdr"
    mne.export.export_raw(path, raw, fmt="brainvision", overwrite=True)
    return path, raw


class TestRemovalSettings:
    def test_workflow_config_declares_the_exact_study_interval(self):
        settings = rlc.RemovalSettings.from_config(load_workflow_config(rlc.WORKFLOW, None))

        assert settings.study_event_name == "Trig_therm/T  1"
        assert settings.study_epoch_s == (-5.0, 15.0)
        assert settings.expected_study_events_per_run == 11

    def test_reads_the_packaged_config(self):
        settings = rlc.RemovalSettings.from_config(load_config())
        assert settings.nominal_fundamental_hz == pytest.approx(1.2)
        assert settings.harmonic_range == (24, 79)
        assert settings.removal_harmonic_range == (22, 82)
        assert settings.filter_length == "27s"
        assert settings.mt_bandwidth == pytest.approx(0.6)

    def test_static_isolated_line_settings_are_not_supported(self):
        settings = rlc.RemovalSettings.from_config(load_config())
        assert not hasattr(settings, "isolated_hz")
        assert not hasattr(settings, "detect_isolated")

    @pytest.mark.parametrize(
        "retired",
        ("isolated_hz", "isolated_search_hz", "detect_isolated", "max_isolated_lines"),
    )
    def test_retired_static_target_settings_fail_fast(self, retired):
        class Legacy:
            def get(self, key, default=None):
                return {retired: []}

        with pytest.raises(ValueError, match="Static isolated-line targeting"):
            rlc.RemovalSettings.from_config(Legacy())

    def test_notch_width_is_configured_rather_than_left_to_mne(self):
        settings = rlc.RemovalSettings.from_config(load_config())
        assert settings.notch_width_ratio == pytest.approx(450.0)
        assert settings.notch_width_min_hz == pytest.approx(0.05)
        # MNE's own default would empty a quarter of the band.
        assert settings.notch_width_ratio > 200.0

    def test_removal_reaches_below_the_fit_but_spares_harmonic_11(self):
        settings = rlc.RemovalSettings.from_config(load_config())
        assert settings.removal_harmonic_range[0] < settings.harmonic_range[0]
        assert settings.removal_harmonic_range[0] == 22  # 26.40 Hz
        assert settings.removal_harmonic_range[0] > 11  # 13.23 Hz stays

    def test_the_configured_width_keeps_the_band_inside_the_gate(self):
        settings = rlc.RemovalSettings.from_config(load_config())
        low, high = settings.removal_harmonic_range
        targets = [
            settings.nominal_fundamental_hz * k
            for k in range(low, high + 1)
            if not 59.5 <= settings.nominal_fundamental_hz * k <= 60.5
        ]
        widths = lr.notch_widths_for(
            targets, ratio=settings.notch_width_ratio, minimum_hz=settings.notch_width_min_hz
        )
        fraction = lr.removed_band_fraction(np.arange(0, 120, 0.05), targets, widths)
        assert fraction <= lr.PreservationGate().max_band_fraction_removed

    def test_the_comb_window_cannot_reach_the_next_harmonic(self):
        settings = rlc.RemovalSettings.from_config(load_config())
        assert settings.search_hz < settings.nominal_fundamental_hz / 2

    def test_falls_back_when_the_block_is_absent(self):
        class Empty:
            def get(self, key, default=None):
                return default

        assert rlc.RemovalSettings.from_config(Empty()) == rlc.RemovalSettings()

    def test_config_values_override_the_defaults(self):
        class Fake:
            def get(self, key, default=None):
                return {"harmonic_range": [10, 20], "mt_bandwidth": 0.9, "filter_length": "8s"}

        settings = rlc.RemovalSettings.from_config(Fake())
        assert settings.harmonic_range == (10, 20)
        assert settings.mt_bandwidth == pytest.approx(0.9)
        assert settings.filter_length == "8s"

    def test_candidate_floor_cannot_exceed_the_confirmation_floor(self):
        with pytest.raises(ValueError, match="candidate prominence"):
            rlc.RemovalSettings(
                detection_candidate_prominence_db=11.0,
                detection_min_prominence_db=10.0,
            )

    def test_detected_nominal_search_must_stay_below_one_line_width(self):
        with pytest.raises(ValueError, match="detection_search_hz"):
            rlc.RemovalSettings(detection_search_hz=lr._LINE_CLAIM_HZ)

    def test_uncertainty_multiplier_must_be_positive(self):
        with pytest.raises(ValueError, match="uncertainty_confidence_z"):
            rlc.RemovalSettings(uncertainty_confidence_z=0.0)
        with pytest.raises(ValueError, match="filter_jobs"):
            rlc.RemovalSettings(filter_jobs=0)

    def test_invalid_study_window_contract_fails_fast(self):
        with pytest.raises(ValueError, match="increasing bounds"):
            rlc.RemovalSettings(study_epoch_s=(15.0, -5.0))
        with pytest.raises(ValueError, match="expected_study_events_per_run"):
            rlc.RemovalSettings(expected_study_events_per_run=0)


class TestChannelScaling:
    def test_reads_names_and_resolutions(self, brainvision_run):
        path, raw = brainvision_run
        names, resolutions = rlc.parse_channel_scaling(path)
        assert names == raw.ch_names
        assert np.all(resolutions > 0)

    def test_rejects_a_format_it_cannot_write(self, tmp_path):
        path = tmp_path / "x.vhdr"
        path.write_text(
            "BinaryFormat=INT_16\nDataOrientation=MULTIPLEXED\nCh1=Fp1,,0.5,µV\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="IEEE_FLOAT_32"):
            rlc.parse_channel_scaling(path)

    def test_rejects_a_vectorised_layout(self, tmp_path):
        path = tmp_path / "x.vhdr"
        path.write_text(
            "BinaryFormat=IEEE_FLOAT_32\nDataOrientation=VECTORIZED\nCh1=Fp1,,0.5,µV\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="MULTIPLEXED"):
            rlc.parse_channel_scaling(path)

    def test_ignores_a_coordinates_section(self, tmp_path):
        """Headers written by Analyzer carry a second block of ``Ch<N>=`` lines.

        Those hold three comma-separated numbers rather than the four fields of
        ``[Channel Infos]``. A pattern whose character classes admit newlines runs one
        coordinate line into the next and parses a resolution of ``"-72\\nCh2=1"``.
        """
        path = tmp_path / "x.vhdr"
        path.write_text(
            "BinaryFormat=IEEE_FLOAT_32\n"
            "DataOrientation=MULTIPLEXED\n"
            "[Channel Infos]\n"
            "Ch1=Fp1,,0.5,µV\n"
            "Ch2=Cz,,0.5,µV\n"
            "Ch3=Oz,,0.5,µV\n"
            "[Coordinates]\n"
            "Ch1=1,-90,-72\n"
            "Ch2=1,45,90\n"
            "Ch3=1,0,0\n",
            encoding="utf-8",
        )

        names, resolutions = rlc.parse_channel_scaling(path)

        assert names == ["Fp1", "Cz", "Oz"]
        assert np.allclose(resolutions, 0.5)


class TestWriteEegBinary:
    def test_round_trips_through_the_original_header(self, brainvision_run, tmp_path):
        import mne

        path, raw = brainvision_run
        modified = raw.get_data() * 0.5
        destination = tmp_path / "out" / path.with_suffix(".eeg").name
        destination.parent.mkdir()
        rlc.write_eeg_binary(path, destination, modified)

        # Reuse the original header, which is exactly what the runner does.
        for suffix in (".vhdr", ".vmrk"):
            (destination.parent / path.with_suffix(suffix).name).write_bytes(
                path.with_suffix(suffix).read_bytes()
            )
        back = mne.io.read_raw_brainvision(
            destination.parent / path.name, preload=True, verbose="ERROR"
        )
        # float32 storage, so the comparison has to be relative to full scale.
        deviation = np.max(np.abs(back.get_data() - modified))
        assert deviation < rlc.ROUNDTRIP_RELATIVE_TOLERANCE * np.max(np.abs(modified))
        assert deviation > 0  # it really did go through float32

    def test_rejects_a_channel_count_mismatch(self, brainvision_run, tmp_path):
        path, raw = brainvision_run
        with pytest.raises(ValueError, match="header describes"):
            rlc.write_eeg_binary(path, tmp_path / "o.eeg", raw.get_data()[:2])


class TestMirrorSidecars:
    def test_copies_everything_except_binaries(self, tmp_path):
        source = tmp_path / "src"
        (source / "sub-01" / "eeg").mkdir(parents=True)
        (source / "dataset_description.json").write_text("{}", encoding="utf-8")
        (source / "sub-01" / "eeg" / "a_eeg.vhdr").write_text("h", encoding="utf-8")
        (source / "sub-01" / "eeg" / "a_eeg.eeg").write_bytes(b"\x00" * 16)
        (source / "sub-01" / "eeg" / "a_eeg.vhdr.lock").write_text("", encoding="utf-8")

        destination = tmp_path / "dst"
        assert rlc.mirror_sidecars(source, destination) == 2
        assert (destination / "dataset_description.json").exists()
        assert (destination / "sub-01" / "eeg" / "a_eeg.vhdr").exists()
        assert not (destination / "sub-01" / "eeg" / "a_eeg.eeg").exists()
        assert not (destination / "sub-01" / "eeg" / "a_eeg.vhdr.lock").exists()


class TestDiscoverRuns:
    def _make(self, root, subject, run):
        directory = root / subject / "eeg"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{subject}_task-thermalactive_run-{run}_eeg.vhdr"
        path.write_text("", encoding="utf-8")
        return path

    def test_finds_every_task_run(self, tmp_path):
        for subject in ("sub-0001", "sub-0002"):
            for run in (1, 2):
                self._make(tmp_path, subject, run)
        assert len(rlc.discover_runs(tmp_path, None)) == 4

    def test_filters_by_subject(self, tmp_path):
        for subject in ("sub-0001", "sub-0002"):
            self._make(tmp_path, subject, 1)
        found = rlc.discover_runs(tmp_path, ["sub-0002"])
        assert len(found) == 1
        assert found[0].parent.parent.name == "sub-0002"

    def test_ignores_macos_appledouble_files(self, tmp_path):
        real = self._make(tmp_path, "sub-0001", 1)
        (real.parent / f"._{real.name}").write_bytes(b"AppleDouble")

        assert rlc.discover_runs(tmp_path, None) == [real]

    def test_raises_when_nothing_matches(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No task runs"):
            rlc.discover_runs(tmp_path, None)


class TestRunSpectrum:
    def test_returns_a_tr_commensurate_grid(self):
        import mne

        mne.set_log_level("ERROR")
        sfreq = 1000.0
        n_times = int(sfreq * 0.9 * rlc.ESTIMATION_TR_COUNT * 2)
        rng = np.random.default_rng(1)
        info = mne.create_info(["Fp1", "Cz", "Oz"], sfreq, "eeg")
        raw = mne.io.RawArray(rng.normal(scale=1e-5, size=(3, n_times)), info)
        freqs, spectrum_db, prominence = rlc.run_spectrum(raw)
        assert freqs[1] == pytest.approx(1.0 / (0.9 * rlc.ESTIMATION_TR_COUNT))
        assert spectrum_db.shape == freqs.shape == prominence.shape
        # a comb line lands on a bin centre
        assert np.min(np.abs(freqs - 54.0)) < 1e-9

    def test_estimation_windows_overlap_by_half_and_cover_the_tail(self):
        import mne

        sfreq = 100.0
        window_samples = int(round(sfreq * 0.9)) * rlc.ESTIMATION_TR_COUNT
        info = mne.create_info(["Cz"], sfreq, "eeg")
        raw = mne.io.RawArray(np.zeros((1, 2 * window_samples + 137)), info, verbose="ERROR")

        _, windows, bounds = rlc.run_spectra(raw)

        assert len(windows) == len(bounds)
        assert bounds[0] == (0, window_samples)
        assert bounds[1][0] == window_samples // 2
        assert bounds[-1][1] == raw.n_times

    def test_rejects_a_recording_shorter_than_one_block(self):
        import mne

        mne.set_log_level("ERROR")
        info = mne.create_info(["Fp1"], 1000.0, "eeg")
        raw = mne.io.RawArray(np.zeros((1, 1000)), info)
        with pytest.raises(ValueError, match="shorter than one adaptive estimation window"):
            rlc.run_spectrum(raw)


def test_removal_settings_reads_the_mains_exclusion_flag():
    """Whether mains is left to the pipeline's FIR notch is a setting, not a constant."""

    class _Config:
        def __init__(self, block):
            self._block = block

        def get(self, key):
            return self._block if key == "line_comb_removal" else None

    assert rlc.RemovalSettings.from_config(_Config({})).exclude_mains is True
    assert rlc.RemovalSettings.from_config(_Config({"exclude_mains": False})).exclude_mains is False


def _synthetic_spectrum(peaks=(), *, f0=1.2, harmonics=(24, 79), df=0.002):
    """A spectrum carrying a full comb plus the given isolated peaks."""
    freqs = np.arange(1.0, 100.0, df)
    spectrum = np.zeros_like(freqs)
    sigma = 0.109 / 2.355

    def add(centre, height):
        spectrum[:] = np.maximum(spectrum, height * np.exp(-0.5 * ((freqs - centre) / sigma) ** 2))

    for k in range(harmonics[0], harmonics[1] + 1):
        add(k * f0, 14.0)
    for centre, height in peaks:
        add(centre, height)
    return freqs, spectrum, spectrum.copy()


def _adjacent_lines(spectrum):
    settings = rlc.RemovalSettings()
    estimate = rlc._detection_scaffold(*spectrum, settings)
    return rlc.detect_comb_adjacent_lines(*spectrum, estimate=estimate, settings=settings)


def test_a_distinct_narrow_comb_adjacent_line_is_detected():
    spectrum = _synthetic_spectrum(peaks=[(27.72, 14.0)])

    assert _adjacent_lines(spectrum) == pytest.approx((27.72,), abs=0.003)


@pytest.mark.parametrize(
    "frequency,prominence",
    (
        (27.72, 9.9),  # below the existing spatially robust prominence floor
        (27.78, 14.0),  # outside the residual-responsibility region
        (27.62, 14.0),  # already covered by the parent comb target
        (60.10, 14.0),  # mains belongs to the downstream mains notch
    ),
)
def test_an_unsupported_comb_adjacent_candidate_is_rejected(frequency, prominence):
    spectrum = _synthetic_spectrum(peaks=[(frequency, prominence)])

    assert _adjacent_lines(spectrum) == ()


def test_a_broad_comb_adjacent_peak_is_not_called_an_electrical_line():
    freqs, spectrum_db, prominence = _synthetic_spectrum()
    prominence = prominence.copy()
    prominence += 14.0 * np.exp(-0.5 * ((freqs - 27.72) / 0.2) ** 2)
    spectrum_db = np.maximum(spectrum_db, prominence)

    assert _adjacent_lines((freqs, spectrum_db, prominence)) == ()


def _session_run(whole, windows=(), *, study_windows=(), study_bounds=()):
    window_spectra = tuple(windows) or (whole,)
    bounds = tuple((index * 100, (index + 1) * 100) for index in range(len(window_spectra)))
    return rlc.SessionRunSpectra(
        whole=whole,
        windows=window_spectra,
        bounds=bounds,
        study_windows=tuple(study_windows),
        study_bounds=tuple(study_bounds),
    )


def test_a_study_epoch_line_is_routed_to_every_overlapping_filter_window():
    settings = rlc.RemovalSettings()
    empty = _synthetic_spectrum()
    adjacent = _synthetic_spectrum(peaks=[(27.72, 14.0)])
    run = rlc.SessionRunSpectra(
        whole=empty,
        windows=(empty, empty, empty),
        bounds=((0, 100), (50, 150), (100, 200)),
        study_windows=(adjacent,),
        study_bounds=((75, 125),),
    )

    plans = rlc.automatic_line_plans([run, _session_run(empty), _session_run(empty)], settings)

    assert all(
        any(abs(value - 27.72) < 0.003 for value in targets)
        for targets in plans[0].narrow_window_hz
    )
    assert any(abs(value - 27.72) < 0.003 for value in plans[0].narrow_study_hz[0])
    assert not any(targets for plan in plans[1:] for targets in plan.narrow_window_hz)


def test_a_routed_target_expands_to_the_observed_adjacent_line_support():
    settings = rlc.RemovalSettings()
    empty = _synthetic_spectrum()
    first_line = _synthetic_spectrum(peaks=[(81.72, 16.0)])
    second_line = _synthetic_spectrum(peaks=[(81.84, 16.0)])
    run = rlc.SessionRunSpectra(
        whole=empty,
        windows=(first_line, second_line, empty),
        bounds=((0, 100), (50, 150), (100, 200)),
        study_windows=(),
        study_bounds=(),
    )

    isolated_plans = rlc.automatic_line_plans(
        [run, _session_run(empty), _session_run(empty)],
        settings,
    )
    plan = rlc.build_run_plan_from_spectra(run, settings, isolated_plans[0])
    target_widths = [
        (target, width)
        for target, width in zip(
            plan.windows[1].targets_hz,
            plan.windows[1].notch_widths_hz,
        )
        if abs(target - 81.72) < 0.003
    ]

    assert len(target_widths) == 1
    target, width = target_widths[0]
    assert target + width / 2.0 >= 81.84


def test_an_overlapping_neighbor_cannot_widen_a_locally_absent_target():
    settings = rlc.RemovalSettings()
    empty = _synthetic_spectrum()
    first_line = _synthetic_spectrum(peaks=[(81.72, 16.0)])
    run = rlc.SessionRunSpectra(
        whole=empty,
        windows=(first_line, empty, empty),
        bounds=((0, 100), (50, 150), (100, 200)),
        study_windows=(),
        study_bounds=(),
    )
    isolated_plans = rlc.automatic_line_plans(
        [run, _session_run(empty), _session_run(empty)],
        settings,
    )

    plan = rlc.build_run_plan_from_spectra(run, settings, isolated_plans[0])
    target_widths = [
        (target, width)
        for target, width in zip(
            plan.windows[1].targets_hz,
            plan.windows[1].notch_widths_hz,
        )
        if abs(target - 81.72) < 0.003
    ]

    assert len(target_widths) == 1
    target, width = target_widths[0]
    assert target == pytest.approx(81.72, abs=0.003)
    assert width == pytest.approx(
        max(81.72 / settings.notch_width_ratio, settings.notch_width_min_hz)
    )


def test_a_study_epoch_uses_its_validated_continuous_comb_scaffold():
    settings = rlc.RemovalSettings()
    continuous = _synthetic_spectrum()
    short_study = _synthetic_spectrum(
        peaks=[(27.72, 14.0)],
        harmonics=(24, 42),
    )
    run = rlc.SessionRunSpectra(
        whole=continuous,
        windows=(continuous, continuous, continuous),
        bounds=((0, 100), (50, 150), (100, 200)),
        study_windows=(short_study,),
        study_bounds=((75, 125),),
    )

    plans = rlc.automatic_line_plans(
        [run, _session_run(continuous), _session_run(continuous)],
        settings,
    )

    assert all(
        any(abs(value - 27.72) < 0.003 for value in targets)
        for targets in plans[0].narrow_window_hz
    )


def test_a_study_epoch_can_supply_session_replicated_isolated_line_evidence():
    settings = rlc.RemovalSettings()
    continuous = _synthetic_spectrum()
    study_line = _synthetic_spectrum(peaks=[(63.0, 18.0)])
    runs = [
        _session_run(
            continuous,
            windows=(continuous, continuous),
            study_windows=(study_line,),
            study_bounds=((25, 75),),
        )
        for _ in range(3)
    ]

    plans = rlc.automatic_line_plans(runs, settings)

    assert all(
        any(abs(value - 63.0) < 0.003 for targets in plan.window_hz for value in targets)
        for plan in plans
    )
    assert all(any(abs(value - 63.0) < 0.003 for value in plan.study_hz[0]) for plan in plans)

    fitted = rlc.build_run_plan_from_spectra(runs[0], settings, plans[0])
    assert any(
        abs(value - 63.0) < 0.003 for window in fitted.windows for value in window.targets_hz
    )
    assert any(abs(value - 63.0) < 0.003 for value in fitted.study_windows[0].targets_hz)


def test_neighboring_evidence_cannot_authorise_an_exact_study_notch():
    settings = rlc.RemovalSettings()
    empty = _synthetic_spectrum()
    line = _synthetic_spectrum(peaks=[(63.0, 18.0)])
    runs = [
        _session_run(
            line,
            windows=(line, line),
            study_windows=(empty,),
            study_bounds=((25, 75),),
        )
        for _ in range(3)
    ]

    isolated_plans = rlc.automatic_line_plans(runs, settings)

    assert all(
        not any(abs(value - 63.0) < 0.003 for value in plan.study_hz[0]) for plan in isolated_plans
    )
    fitted = rlc.build_run_plan_from_spectra(runs[0], settings, isolated_plans[0])
    assert not any(abs(value - 63.0) < 0.003 for value in fitted.study_windows[0].targets_hz)


def test_whole_run_evidence_cannot_authorise_a_time_local_notch():
    settings = rlc.RemovalSettings()
    empty = _synthetic_spectrum()
    whole_line = _synthetic_spectrum(peaks=[(63.0, 18.0)])
    runs = [_session_run(whole_line, windows=(empty, empty)) for _ in range(3)]

    plans = rlc.automatic_line_plans(runs, settings)

    assert all(any(abs(value - 63.0) < 0.003 for value in plan.whole_hz) for plan in plans)
    assert not any(
        abs(value - 63.0) < 0.003
        for plan in plans
        for targets in plan.window_hz
        for value in targets
    )


def test_whole_run_shoulders_cannot_widen_a_time_local_notch():
    settings = rlc.RemovalSettings()
    empty = _synthetic_spectrum()
    whole = _synthetic_spectrum(peaks=[(81.68, 18.0)])
    run = _session_run(whole, windows=(empty, empty))
    isolated = rlc.RunIsolatedLinePlan(
        whole_hz=(),
        window_hz=((), ()),
        narrow_window_hz=((), ()),
        source_count=0,
    )

    plan = rlc.build_run_plan_from_spectra(run, settings, isolated)
    window = plan.windows[0]
    expected = lr.uncertainty_aware_notch_widths(
        window.estimate,
        window.targets_hz,
        ratio=settings.notch_width_ratio,
        minimum_hz=settings.notch_width_min_hz,
        confidence_z=settings.uncertainty_confidence_z,
        isolated_minimum_hz=(
            2.0 * lr.RESIDUAL_SEARCH_HZ
            + rlc.spectrum_fit_nominal_resolution_hz(settings.filter_length)
        ),
    )
    index = int(np.argmin(np.abs(np.asarray(window.targets_hz) - 81.6)))

    assert window.notch_widths_hz[index] == pytest.approx(expected[index])


def test_whole_run_adjacent_evidence_cannot_create_time_local_targets():
    settings = rlc.RemovalSettings()
    empty = _synthetic_spectrum()
    whole_adjacent = _synthetic_spectrum(peaks=[(27.72, 16.0)])
    run = _session_run(whole_adjacent, windows=(empty, empty))

    plans = rlc.automatic_line_plans(
        [
            run,
            _session_run(empty, windows=(empty, empty)),
            _session_run(empty, windows=(empty, empty)),
        ],
        settings,
    )

    assert not any(
        abs(value - 27.72) < 0.003 for targets in plans[0].narrow_window_hz for value in targets
    )


def test_simultaneous_resolvable_lines_form_distinct_source_clusters():
    observations = (
        rlc._LineObservation(0, 57.40, 18.0, (0, 100)),
        rlc._LineObservation(0, 57.50, 17.0, (0, 100)),
        rlc._LineObservation(0, 57.41, 19.0, (100, 200)),
        rlc._LineObservation(0, 57.51, 16.0, (100, 200)),
    )

    clusters = rlc._cluster_line_observations(observations)

    assert len(clusters) == 2
    assert sorted(cluster.centre_hz for cluster in clusters) == pytest.approx((57.405, 57.505))


def test_a_supported_isolated_line_is_not_discarded_beside_an_adjacent_line():
    settings = rlc.RemovalSettings()
    two_lines = _synthetic_spectrum(peaks=[(57.40, 24.0), (57.50, 23.0)])
    spectra = [_session_run(two_lines) for _ in range(3)]

    plans = rlc.automatic_line_plans(spectra, settings)

    assert all(
        any(abs(value - 57.40) < 0.003 for value in targets)
        for plan in plans
        for targets in plan.window_hz
    )


def test_a_peak_inside_comb_responsibility_never_enters_the_wide_isolated_route():
    settings = rlc.RemovalSettings()
    covered_shoulder = _synthetic_spectrum(peaks=[(73.27, 15.1)])

    plans = rlc.automatic_line_plans(
        [_session_run(covered_shoulder) for _ in range(3)],
        settings,
    )

    assert not any(
        abs(value - 73.27) < 0.02 for plan in plans for window in plan.window_hz for value in window
    )


def test_a_session_agrees_on_one_nominal_list_across_its_runs():
    """Replication supplies nominals without pooling any run's final estimate."""
    settings = rlc.RemovalSettings()
    # The line recurs in three runs, so it clears the recurrence rule, and the fourth run
    # is the one that has to be given the nominal anyway for pooling to line up.
    strong = _synthetic_spectrum(peaks=[(47.04, 24.0), (94.34, 26.0)])
    also = _synthetic_spectrum(peaks=[(47.04, 24.0), (94.34, 22.0)])
    third = _synthetic_spectrum(peaks=[(47.04, 24.0), (94.34, 18.0)])
    without = _synthetic_spectrum(peaks=[(47.04, 24.0)])

    nominals = rlc.session_nominals(
        [_session_run(spectrum) for spectrum in (strong, also, third, without)],
        settings,
    )
    assert any(
        abs(f - 94.34) < 0.02 for f in nominals
    ), "a line the session carries must be offered to every run in it"

    estimates = [
        lr.estimate_comb(
            freqs,
            spec,
            prom,
            nominal_hz=settings.nominal_fundamental_hz,
            harmonic_range=settings.harmonic_range,
            isolated_nominal_hz=nominals,
            search_hz=settings.search_hz,
            isolated_search_hz=settings.detection_search_hz,
            min_prominence_db=settings.min_prominence_db,
        )
        for freqs, spec, prom in (strong, also, third, without)
    ]
    assert {len(estimate.isolated_hz) for estimate in estimates} == {len(nominals)}


def test_the_session_returns_every_supported_artifact():
    settings = rlc.RemovalSettings()
    positions = tuple(harmonic * 1.2 + 0.3 for harmonic in range(24, 41))
    spectrum = _synthetic_spectrum(peaks=[(position, 20.0) for position in positions])
    spectra = [_session_run(spectrum) for _ in range(3)]

    assert rlc.session_nominals(spectra, settings) == pytest.approx(positions, abs=0.01)


def test_the_session_list_is_returned_in_frequency_order():
    """Ranking happens on strength; the result is still ordered for the manifest."""
    settings = rlc.RemovalSettings()
    spectrum = _synthetic_spectrum(peaks=[(94.344, 27.0), (47.043, 23.0), (28.1, 19.0)])
    spectra = [_session_run(spectrum) for _ in range(3)]
    kept = rlc.session_nominals(spectra, settings)
    assert list(kept) == sorted(kept)


def test_a_line_seen_in_only_one_run_of_a_session_is_not_taken():
    """The runs of a session are replication separating a line from a fluctuation."""
    settings = rlc.RemovalSettings()
    # 63.0 Hz is 0.6 Hz from the nearest comb position and 2.35 Hz from the nearest probe
    # tone, so if it is rejected it is the recurrence rule doing it and nothing else.
    real = (47.043, 23.0)
    once = (63.0, 18.0)
    spectra = [
        _session_run(_synthetic_spectrum(peaks=[real, once])),
        _session_run(_synthetic_spectrum(peaks=[real])),
        _session_run(_synthetic_spectrum(peaks=[real])),
    ]
    kept = rlc.session_nominals(spectra, settings)

    assert any(abs(f - real[0]) < 0.02 for f in kept), kept
    assert not any(
        abs(f - once[0]) < 0.02 for f in kept
    ), f"a peak present in one run of three was taken for the session: {kept}"


def test_repeated_subthreshold_candidates_are_authorised_by_an_independent_strong_run():
    """Replication must recover a stable line that a per-run 10 dB cutoff misses.

    This is the measured sub-0006 pattern at 58.426 Hz: one run is clearly above the
    conservative line threshold and four other runs carry the same narrow summit at
    6--10 dB.  Hard-thresholding every run independently discarded the real source.
    """
    settings = rlc.RemovalSettings()
    spectra = [
        _session_run(_synthetic_spectrum(peaks=[(58.426, strength)]))
        for strength in (13.4, 9.8, 9.7, 6.3, 2.0, 1.0)
    ]

    kept = rlc.session_nominals(spectra, settings)

    assert any(abs(frequency - 58.426) < 0.02 for frequency in kept), kept


def test_a_session_line_uses_the_median_position_not_the_strongest_runs_position():
    settings = rlc.RemovalSettings()
    spectra = [
        _session_run(_synthetic_spectrum(peaks=[(position, strength)]))
        for position, strength in ((58.40, 14.0), (58.42, 9.0), (58.44, 8.0))
    ]

    kept = rlc.session_nominals(spectra, settings)

    assert any(abs(frequency - 58.42) < 0.005 for frequency in kept), kept


def test_a_single_run_session_cannot_bypass_the_replication_requirement():
    settings = rlc.RemovalSettings()
    with pytest.raises(ValueError, match="at least 3 runs"):
        rlc.session_nominals(
            [_session_run(_synthetic_spectrum(peaks=[(47.043, 23.0)]))],
            settings,
        )


def test_a_block_replicated_line_in_two_runs_is_authorised():
    """A non-stationary line must not disappear merely by dilution in whole-run spectra."""
    settings = rlc.RemovalSettings()
    empty = _synthetic_spectrum()
    strong = _synthetic_spectrum(peaks=[(27.52, 16.5)])
    moderate = _synthetic_spectrum(peaks=[(27.52, 15.5)])
    spectra = [
        _session_run(empty, windows=(strong, strong)),
        _session_run(empty, windows=(moderate,)),
        _session_run(empty),
    ]

    plans = rlc.automatic_line_plans(spectra, settings)

    assert all(
        any(
            abs(frequency - 27.52) < 0.02
            for window in plan.narrow_window_hz
            for frequency in window
        )
        for plan in plans[:2]
    )


def test_weak_incidental_candidates_do_not_dilute_two_strong_supporting_runs():
    settings = rlc.RemovalSettings()
    empty = _synthetic_spectrum()
    strong = _synthetic_spectrum(peaks=[(27.52, 16.5)])
    moderate = _synthetic_spectrum(peaks=[(27.52, 15.5)])
    weak = _synthetic_spectrum(peaks=[(27.48, 7.0)])
    spectra = [
        _session_run(empty, windows=(strong, strong)),
        _session_run(empty, windows=(moderate,)),
        *[_session_run(empty, windows=(weak,)) for _ in range(4)],
    ]

    plans = rlc.automatic_line_plans(spectra, settings)

    assert all(
        any(
            abs(frequency - 27.52) < 0.02
            for window in plan.narrow_window_hz
            for frequency in window
        )
        for plan in plans[:2]
    )
    assert not any(
        abs(frequency - 27.52) < 0.02
        for plan in plans[2:]
        for window in plan.narrow_window_hz
        for frequency in window
    )


def test_overlapping_blocks_in_one_run_cannot_authorise_a_line():
    settings = rlc.RemovalSettings()
    empty = _synthetic_spectrum()
    line = _synthetic_spectrum(peaks=[(63.0, 18.0)])
    overlapping = rlc.SessionRunSpectra(
        whole=empty,
        windows=(line, line, line),
        bounds=((0, 100), (50, 150), (75, 175)),
        study_windows=(),
        study_bounds=(),
    )

    kept = rlc.session_nominals(
        [overlapping, _session_run(empty), _session_run(empty)],
        settings,
    )

    assert not any(abs(frequency - 63.0) < 0.02 for frequency in kept)


def test_temporally_replicated_line_is_targeted_only_in_its_supporting_run():
    """A future run's strong line must not need to recur in another acquisition."""
    settings = rlc.RemovalSettings()
    empty = _synthetic_spectrum()
    session_line = _synthetic_spectrum(peaks=[(47.04, 18.0)])
    run_line = _synthetic_spectrum(peaks=[(63.0, 18.0)])
    spectra = [
        _session_run(session_line, windows=(run_line, run_line, run_line)),
        _session_run(session_line, windows=(empty, empty, empty)),
        _session_run(session_line, windows=(empty, empty, empty)),
    ]

    plans = rlc.automatic_line_plans(spectra, settings)

    assert all(any(abs(value - 47.04) < 0.02 for value in plan.whole_hz) for plan in plans)
    assert all(
        any(abs(value - 63.0) < 0.02 for value in window_hz) for window_hz in plans[0].window_hz
    )
    assert not any(
        abs(value - 63.0) < 0.02
        for plan in plans[1:]
        for window_hz in plan.window_hz
        for value in window_hz
    )


def test_overlapping_blocks_do_not_create_a_run_specific_target():
    settings = rlc.RemovalSettings()
    empty = _synthetic_spectrum()
    line = _synthetic_spectrum(peaks=[(63.0, 18.0)])
    overlapping = rlc.SessionRunSpectra(
        whole=empty,
        windows=(line, line, line),
        bounds=((0, 100), (50, 150), (75, 175)),
        study_windows=(),
        study_bounds=(),
    )

    plans = rlc.automatic_line_plans(
        [overlapping, _session_run(empty), _session_run(empty)],
        settings,
    )

    assert not any(
        abs(value - 63.0) < 0.02
        for plan in plans
        for window_hz in plan.window_hz
        for value in window_hz
    )


def test_run_specific_support_is_kept_beside_session_support():
    settings = rlc.RemovalSettings()
    session_line = _synthetic_spectrum(peaks=[(47.04, 18.0)])
    run_line = _synthetic_spectrum(peaks=[(63.0, 18.0)])
    spectra = [
        _session_run(session_line, windows=(run_line, run_line, run_line)),
        _session_run(session_line, windows=(session_line, session_line, session_line)),
        _session_run(session_line, windows=(session_line, session_line, session_line)),
    ]

    plans = rlc.automatic_line_plans(spectra, settings)

    assert any(abs(value - 47.04) < 0.02 for value in plans[0].all_hz)
    assert any(abs(value - 63.0) < 0.02 for value in plans[0].all_hz)


def test_session_line_must_clear_every_adaptive_comb_grid():
    settings = rlc.RemovalSettings()
    cluster = rlc._LineCluster(
        [
            rlc._LineObservation(0, 63.0, 18.0, (0, 100)),
            rlc._LineObservation(1, 63.0, 18.0, (0, 100)),
        ]
    )
    near_harmonic_fundamental = (63.0 - 0.04) / 52

    assert not rlc._clears_every_comb_grid(
        cluster.centre_hz,
        fundamentals_hz=(1.2, near_harmonic_fundamental),
        settings=settings,
    )


def test_detected_lines_always_clear_the_estimator_guard():
    """The detector and the estimator must not disagree about what is too close to the comb.

    estimate_comb refuses a nominal within isolated_search_hz of a comb position, because a
    search that wide reaches across and refines onto the harmonic instead. Detection admits
    a peak once it is more than one line width away and outranks the harmonic beside it --
    and those two thresholds were 0.109 Hz against 0.15 Hz, so the detector offered
    sub-0001's 93.759 Hz line and the estimator raised on it, stopping the benchmark.

    Detected nominals sit on the summit already, so they need only a narrow refinement.
    Keeping that search below the detector's own floor makes the disagreement impossible
    rather than unlikely.
    """
    settings = rlc.RemovalSettings()
    assert (
        settings.detection_search_hz < lr._LINE_CLAIM_HZ
    ), "a line the detector admits could still be refused by estimate_comb"
