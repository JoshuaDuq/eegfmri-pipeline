"""The converter must transcribe measurements, never re-derive them.

A cohort figure is only trustworthy because it shows the same numbers as the subject
figure beneath it, and that holds exactly as long as this module copies rather than
computes. These tests pin the transcription, the two classifications it makes from the
evidence, and the places where a missing measurement must stay missing.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from eeg_pipeline.preprocessing.report.analyzer_qc import (
    MarkerAgreement,
    RrIntervals,
    compute_marker_agreement,
)
from eeg_pipeline.preprocessing.report.cohort.record import (
    acquisition_context_of,
    alpha_measurements,
    build_subject_sidecar,
    comb_curves,
    paradigm_of,
    run_table,
    spectrum_curves,
)
from eeg_pipeline.preprocessing.report.cohort.sidecar import (
    AcquisitionContext,
    Paradigm,
    run_columns_for,
)
from eeg_pipeline.preprocessing.report.continuity import RunContinuity
from eeg_pipeline.preprocessing.report.preservation import PosteriorAlpha
from eeg_pipeline.preprocessing.report.scanner import (
    CombResidual,
    VolumeLockedAverage,
    VolumeTiming,
)
from eeg_pipeline.preprocessing.report.spectra import RunSpectra, StageSpectrum

FREQUENCIES = np.asarray([1.0, 2.0, 3.0])
RUN = "sub-0014_task-x_run-1"


def _stage(level: float) -> StageSpectrum:
    return StageSpectrum(
        median_db=np.full(3, level),
        spread_low_db=np.full(3, level - 2),
        spread_high_db=np.full(3, level + 2),
        max_db=np.full(3, level + 5),
        aperiodic=None,
    )


def _spectra(recording_id: str = RUN) -> RunSpectra:
    return RunSpectra(
        recording_id=recording_id,
        frequencies=FREQUENCIES,
        before=_stage(-10.0),
        after=_stage(-14.0),
        n_channels=63,
        fmax_reason="low-pass",
    )


def _continuity(recording_id: str = RUN, *, events=(1.0, 2.0)) -> RunContinuity:
    return RunContinuity(
        recording_id=recording_id,
        window_seconds=1.0,
        times_s=np.arange(10.0),
        channel_names=("Cz",),
        relative_db=np.zeros((1, 10)) + 3.0,
        bad_spans=((0.0, 30.0),),
        duration_s=600.0,
        event_onsets=tuple(events),
    )


def _timing() -> VolumeTiming:
    return VolumeTiming(n_volumes=300, repetition_time_s=2.0, interval_jitter_s=0.004)


def _locked(recording_id: str = RUN) -> VolumeLockedAverage:
    return VolumeLockedAverage(
        recording_id=recording_id,
        times_s=np.arange(5.0),
        before_rms_uv=np.full(5, 3.0),
        after_rms_uv=np.full(5, 1.0),
        n_volumes=300,
        before_locked_rms_uv=3.0,
        after_locked_rms_uv=1.0,
        before_noise_floor_uv=0.5,
        after_noise_floor_uv=0.4,
        before_excess_power_uv2=8.75,
        after_excess_power_uv2=-0.2,
    )


def _comb(recording_id: str = RUN) -> CombResidual:
    return CombResidual(
        recording_id=recording_id,
        timing=_timing(),
        harmonic_frequencies_hz=np.asarray([16.0, 24.0]),
        channel_names=("Cz", "Pz"),
        before_excess_db=np.asarray([[12.0, 8.0], [20.0, 10.0]]),
        after_excess_db=np.asarray([[2.0, 1.0], [4.0, 3.0]]),
    )


def _alpha(*, prominence: float, residual: float, peak: float = 10.0) -> PosteriorAlpha:
    """One alpha measurement, scored over a search wide enough to need correcting for.

    ``residual`` is the aperiodic fit residual, which is the scale resolvability is judged
    against, over a search wide enough that the multiplicity correction bites.
    """
    return PosteriorAlpha(
        channel_names=("Oz",),
        frequencies_hz=FREQUENCIES,
        power_db=np.asarray([-5.0, -6.0, -7.0]),
        peak_frequency_hz=peak,
        prominence_db=prominence,
        background_residual_db=residual,
        n_search_bins=72,
    )


# --------------------------------------------------------------------------------------
# Classification from the evidence
# --------------------------------------------------------------------------------------


def test_observed_volume_markers_mean_the_recording_was_in_a_scanner() -> None:
    observed = replace(_continuity(), has_volume_markers=True)

    assert acquisition_context_of([observed]) is AcquisitionContext.IN_SCANNER
    assert acquisition_context_of([_continuity()]) is AcquisitionContext.OUT_OF_SCANNER


def test_too_few_volume_markers_for_timing_still_mean_in_scanner() -> None:
    observed = replace(_continuity(), has_volume_markers=True)

    assert acquisition_context_of([observed]) is AcquisitionContext.IN_SCANNER


def test_task_events_decide_the_paradigm() -> None:
    assert paradigm_of([_continuity()]) is Paradigm.TASK
    assert paradigm_of([_continuity(events=())]) is Paradigm.REST


# --------------------------------------------------------------------------------------
# Run table
# --------------------------------------------------------------------------------------


def test_the_run_table_carries_every_column_its_context_requires() -> None:
    frame = run_table(
        spectra=[_spectra()],
        continuity=[replace(_continuity(), has_volume_markers=True)],
        timings={RUN: _timing()},
        locked_averages=[_locked()],
        context=AcquisitionContext.IN_SCANNER,
    )

    for name in run_columns_for(AcquisitionContext.IN_SCANNER):
        assert name in frame.columns


def test_the_residual_column_is_accompanied_by_the_coverage_it_was_measured_over() -> None:
    """A residual averaged over a quarter of a run describes a quarter of that run.

    The sidecar is what cross-run analyses read, so the qualifier has to travel in it. Left
    to the residual alone, sub-0009 r3 records 1.33 uV -- measured on the 44 beats the
    correction found and none of the 443 it missed -- and reads as one of the cleanest runs
    in the cohort.
    """
    from eeg_pipeline.preprocessing.report.analyzer_qc import CardiacResidual

    frame = run_table(
        spectra=[_spectra()],
        continuity=[replace(_continuity(), has_volume_markers=True)],
        timings={RUN: _timing()},
        locked_averages=[_locked()],
        cardiac_residuals=[
            CardiacResidual(
                recording_id=RUN,
                marker_count=44,
                beat_source="analyzer-markers",
                residual_uv=1.33,
                beat_train_coverage=0.098,
            )
        ],
        context=AcquisitionContext.IN_SCANNER,
    )

    assert frame.loc[0, "bcg_residual_uv"] == pytest.approx(1.33)
    assert frame.loc[0, "bcg_beat_train_coverage"] == pytest.approx(0.098)


def test_continuity_reductions_are_transcribed_not_recomputed() -> None:
    """The flagged fraction is the property the subject panel already prints."""
    quality = _continuity()
    frame = run_table(
        spectra=[_spectra()],
        continuity=[quality],
        timings={},
        context=AcquisitionContext.OUT_OF_SCANNER,
    )

    assert frame.loc[0, "flagged_fraction"] == pytest.approx(quality.bad_fraction)
    assert frame.loc[0, "continuity_median_db"] == pytest.approx(3.0)
    assert frame.loc[0, "duration_s"] == pytest.approx(600.0)


def test_the_gradient_contract_preserves_observed_floor_and_signed_excess() -> None:
    frame = run_table(
        spectra=[_spectra()],
        continuity=[_continuity()],
        timings={RUN: _timing()},
        locked_averages=[_locked()],
        context=AcquisitionContext.IN_SCANNER,
    )

    assert frame.loc[0, "volume_locked_rms_before_uv"] == pytest.approx(3.0)
    assert frame.loc[0, "volume_locked_floor_before_uv"] == pytest.approx(0.5)
    assert frame.loc[0, "volume_locked_excess_power_before_uv2"] == pytest.approx(8.75)
    assert frame.loc[0, "volume_locked_resolved_before"]
    assert frame.loc[0, "volume_locked_rms_after_uv"] == pytest.approx(1.0)
    assert frame.loc[0, "volume_locked_floor_after_uv"] == pytest.approx(0.4)
    assert frame.loc[0, "volume_locked_excess_power_after_uv2"] == pytest.approx(-0.2)
    assert not frame.loc[0, "volume_locked_resolved_after"]
    assert frame.loc[0, "volume_jitter_s"] == pytest.approx(0.004)


def test_an_eeg_only_run_has_no_gradient_columns_at_all() -> None:
    frame = run_table(
        spectra=[_spectra()],
        continuity=[_continuity()],
        timings={},
        context=AcquisitionContext.OUT_OF_SCANNER,
    )

    assert "volume_locked_rms_after_uv" not in frame.columns
    assert "n_volumes" not in frame.columns


def test_a_run_without_beat_detection_holds_a_blank_rather_than_a_zero() -> None:
    """A zero heart rate is a measurement, and an impossible one."""
    frame = run_table(
        spectra=[_spectra()],
        continuity=[_continuity()],
        timings={},
        context=AcquisitionContext.OUT_OF_SCANNER,
    )

    assert np.isnan(frame.loc[0, "median_bpm"])
    assert np.isnan(frame.loc[0, "marker_matched_fraction"])


def test_beat_and_marker_measurements_are_transcribed_when_present() -> None:
    intervals = np.full(60, 1.0)
    beats = RrIntervals(
        recording_id=RUN,
        beat_times_s=np.arange(61.0),
        intervals_s=intervals,
    )
    agreement = MarkerAgreement(
        recording_id=RUN,
        marker_onsets_s=np.arange(60.0),
        detected_onsets_s=np.arange(60.0),
        tolerance_s=0.1,
        n_matched=57,
    )

    frame = run_table(
        spectra=[_spectra()],
        continuity=[_continuity()],
        timings={},
        rr_intervals=[beats],
        marker_agreements=[agreement],
        context=AcquisitionContext.OUT_OF_SCANNER,
    )

    assert frame.loc[0, "median_bpm"] == pytest.approx(60.0)
    assert frame.loc[0, "n_beats"] == pytest.approx(61)
    assert frame.loc[0, "marker_matched_fraction"] == pytest.approx(57 / 60)
    assert frame.loc[0, "n_matched_beats"] == pytest.approx(57)


def test_marker_lag_travels_with_the_matched_fraction() -> None:
    """A share of zero means two different things, and only the lag separates them.

    On sub-0012 runs 5 and 6 the markers describe the heartbeat exactly and sit a fixed
    ~300 ms ahead of the ECG peak the detector settles on, which the share alone reports
    as total disagreement. The subject report already prints the lag; without it here the
    cohort table sends a sound run to manual review as a physiology outlier.
    """
    agreement = compute_marker_agreement(
        recording_id=RUN,
        marker_onsets_s=np.arange(60.0),
        detected_onsets_s=np.arange(60.0) + 0.30,
    )

    frame = run_table(
        spectra=[_spectra()],
        continuity=[_continuity()],
        timings={},
        marker_agreements=[agreement],
        context=AcquisitionContext.OUT_OF_SCANNER,
    )

    assert frame.loc[0, "marker_matched_fraction"] == pytest.approx(0.0)
    assert frame.loc[0, "marker_median_lag_s"] == pytest.approx(0.30, abs=1e-6)
    assert frame.loc[0, "marker_lag_iqr_s"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# Curves
# --------------------------------------------------------------------------------------


def test_spectrum_curves_carry_both_stages_at_full_resolution() -> None:
    frame = spectrum_curves([_spectra()])

    assert set(frame["stage"]) == {"before", "after"}
    assert len(frame) == 2 * FREQUENCIES.size
    after = frame[frame["stage"] == "after"]
    assert after["median_db"].tolist() == [-14.0] * 3
    assert after["max_db"].tolist() == [-9.0] * 3


def test_comb_curves_carry_the_harmonic_index_as_well_as_its_frequency() -> None:
    """Participants at different repetition times can only be pooled on the index."""
    frame = comb_curves([_comb()])

    # Fundamental is 0.5 Hz at TR = 2 s, so 16 Hz is the 32nd harmonic.
    assert frame["harmonic_index"].tolist() == [32, 48]
    assert frame["harmonic_hz"].tolist() == [16.0, 24.0]


def test_comb_excess_is_reduced_across_channels_keeping_the_worst() -> None:
    """A montage median can sit near zero while individual sensors are unusable."""
    frame = comb_curves([_comb()])

    assert frame["before_excess_db_median"].tolist() == [16.0, 9.0]
    assert frame["before_excess_db_max"].tolist() == [20.0, 10.0]


def test_an_eeg_only_participant_produces_an_empty_comb_table() -> None:
    frame = comb_curves([])

    assert frame.empty
    assert "harmonic_index" in frame.columns


# --------------------------------------------------------------------------------------
# Alpha
# --------------------------------------------------------------------------------------


def test_a_resolvable_peak_contributes_its_frequency() -> None:
    measurements = alpha_measurements({"after": _alpha(prominence=6.0, residual=1.0)})

    assert measurements["alpha_peak_resolvable_after"] is True
    assert measurements["alpha_peak_frequency_hz_after"] == pytest.approx(10.0)


def test_an_unresolvable_peak_contributes_no_frequency_at_all() -> None:
    """The failure this guards: the argmax of noise is still an argmax."""
    measurements = alpha_measurements({"after": _alpha(prominence=1.2, residual=1.0)})

    assert measurements["alpha_peak_resolvable_after"] is False
    assert "alpha_peak_frequency_hz_after" not in measurements
    # The prominence is still recorded, and so is the count of participants without a
    # resolvable rhythm, which is a cohort measurement in its own right.
    assert measurements["alpha_prominence_db_after"] == pytest.approx(1.2)




# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------


def test_a_scanner_participant_assembles_a_complete_sidecar() -> None:
    sidecar = build_subject_sidecar(
        subject="0014",
        task="thermalactive",
        spectra=[_spectra()],
        continuity=[replace(_continuity(), has_volume_markers=True)],
        timings={RUN: _timing()},
        locked_averages=[_locked()],
        combs=[_comb()],
        alpha={"after": _alpha(prominence=6.0, residual=1.0)},
        measurements={"variance_removed": 0.86},
        versions={"mne": "1.12.1"},
    )

    assert sidecar.context is AcquisitionContext.IN_SCANNER
    assert sidecar.paradigm is Paradigm.TASK
    assert sidecar.n_runs == 1
    assert sidecar.has_comb_evidence
    assert sidecar.measurements["variance_removed"] == pytest.approx(0.86)
    assert sidecar.measurements["alpha_peak_frequency_hz_after"] == pytest.approx(10.0)


def test_a_resting_eeg_only_participant_assembles_a_narrower_one() -> None:
    sidecar = build_subject_sidecar(
        subject="0020",
        task="rest",
        spectra=[_spectra()],
        continuity=[_continuity(events=())],
        timings={},
    )

    assert sidecar.context is AcquisitionContext.OUT_OF_SCANNER
    assert sidecar.paradigm is Paradigm.REST
    assert not sidecar.has_comb_evidence
    assert sidecar.comb_curves.empty


def test_a_participant_with_no_measured_runs_is_an_error() -> None:
    with pytest.raises(ValueError, match="nothing for a cohort to read"):
        build_subject_sidecar(subject="0014", task="x", spectra=[], continuity=[], timings={})
