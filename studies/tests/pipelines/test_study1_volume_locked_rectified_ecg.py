from __future__ import annotations

import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


def test_summarize_run_rectifies_each_volume_epoch_before_averaging() -> None:
    from studies.pain_study.study1.figures.volume_locked_rectified_ecg import (
        VolumeLockedEcgSpecification,
        summarize_volume_locked_ecg_run,
    )

    sampling_frequency_hz = 10.0
    ecg_uv = np.array(
        [
            -1.0,
            2.0,
            -3.0,
            4.0,
            2.0,
            -4.0,
            6.0,
            -8.0,
            99.0,
        ]
    )
    raw = mne.io.RawArray(
        ecg_uv[np.newaxis, :] * 1e-6,
        mne.create_info(["ECG"], sampling_frequency_hz, ["ecg"]),
        verbose="ERROR",
    )
    raw.set_annotations(
        mne.Annotations(
            onset=[0.0, 0.4, 0.8],
            duration=[0.0, 0.0, 0.0],
            description=["Volume/V  1"] * 3,
        )
    )
    specification = VolumeLockedEcgSpecification(
        marker_description="Volume/V  1",
        epoch_duration_s=0.4,
        minimum_complete_epochs=2,
        peak_prominence_fraction=0.05,
        peak_minimum_distance_ms=30.0,
        peak_search_radius_ms=5.0,
        peak_maximum_latency_sd_ms=5.0,
    )

    trace = summarize_volume_locked_ecg_run(
        raw,
        subject_id="sub-0014",
        run_id="1",
        stage="raw",
        source_file="run.vhdr",
        expected_sampling_frequency_hz=sampling_frequency_hz,
        specification=specification,
    )

    np.testing.assert_allclose(trace.times_ms, [0.0, 100.0, 200.0, 300.0])
    np.testing.assert_allclose(trace.mean_rectified_ecg_uv, [1.5, 3.0, 4.5, 6.0])
    assert trace.n_complete_epochs == 2


def test_average_run_traces_weights_each_run_equally() -> None:
    from studies.pain_study.study1.figures.volume_locked_rectified_ecg import (
        VolumeLockedEcgRun,
        average_volume_locked_ecg_runs,
    )

    traces = tuple(
        VolumeLockedEcgRun(
            subject_id="sub-0014",
            run_id=str(run_id),
            stage="processed",
            source_file=f"run-{run_id}.vhdr",
            sampling_frequency_hz=1_000.0,
            n_complete_epochs=n_epochs,
            times_ms=np.array([0.0, 1.0]),
            mean_rectified_ecg_uv=np.array(values),
        )
        for run_id, n_epochs, values in (
            (1, 100, (1.0, 3.0)),
            (2, 900, (5.0, 7.0)),
        )
    )

    participant = average_volume_locked_ecg_runs(traces)

    np.testing.assert_allclose(participant.mean_rectified_ecg_uv, [3.0, 5.0])
    assert participant.n_runs == 2
    assert participant.n_complete_epochs == 1_000


def test_detect_stable_peaks_reports_run_latency_variability() -> None:
    from studies.pain_study.study1.figures.volume_locked_rectified_ecg import (
        VolumeLockedEcgRun,
        VolumeLockedEcgSpecification,
        average_volume_locked_ecg_runs,
        detect_stable_artifact_peaks,
    )

    times_ms = np.arange(900, dtype=float)
    runs = []
    for run_id, shift_ms in enumerate((-1.0, 0.0, 1.0, -1.0, 0.0, 1.0), start=1):
        signal = np.ones(times_ms.size)
        for center_ms, amplitude_uv in ((100.0, 8.0), (400.0, 5.0)):
            signal += amplitude_uv * np.exp(-0.5 * ((times_ms - center_ms - shift_ms) / 2.0) ** 2)
        runs.append(
            VolumeLockedEcgRun(
                subject_id="sub-0014",
                run_id=str(run_id),
                stage="processed",
                source_file=f"run-{run_id}.vhdr",
                sampling_frequency_hz=1_000.0,
                n_complete_epochs=500,
                times_ms=times_ms,
                mean_rectified_ecg_uv=signal,
            )
        )
    participant = average_volume_locked_ecg_runs(tuple(runs))
    specification = VolumeLockedEcgSpecification(
        marker_description="Volume/V  1",
        epoch_duration_s=0.9,
        minimum_complete_epochs=50,
        peak_prominence_fraction=0.1,
        peak_minimum_distance_ms=100.0,
        peak_search_radius_ms=5.0,
        peak_maximum_latency_sd_ms=2.0,
    )

    peaks = detect_stable_artifact_peaks(participant, specification)

    assert peaks["peak_id"].tolist() == ["P01", "P02"]
    assert peaks["latency_ms"].tolist() == pytest.approx([100.0, 400.0])
    assert peaks["contributing_runs"].tolist() == [6, 6]
    assert peaks["run_latency_sd_ms"].tolist() == pytest.approx(
        [np.std([-1.0, 0.0, 1.0] * 2, ddof=1)] * 2
    )


def test_detect_stable_troughs_reports_analyzer_display_minima() -> None:
    from studies.pain_study.study1.figures.volume_locked_rectified_ecg import (
        VolumeLockedEcgRun,
        VolumeLockedEcgSpecification,
        average_volume_locked_ecg_runs,
        detect_stable_artifact_troughs,
    )

    times_ms = np.arange(900, dtype=float)
    runs = []
    for run_id, shift_ms in enumerate((-1.0, 0.0, 1.0, -1.0, 0.0, 1.0), start=1):
        signal = np.full(times_ms.size, 10.0)
        for center_ms, amplitude_uv in ((120.0, 8.0), (420.0, 5.0)):
            signal -= amplitude_uv * np.exp(-0.5 * ((times_ms - center_ms - shift_ms) / 2.0) ** 2)
        runs.append(
            VolumeLockedEcgRun(
                subject_id="sub-0014",
                run_id=str(run_id),
                stage="processed",
                source_file=f"run-{run_id}.vhdr",
                sampling_frequency_hz=1_000.0,
                n_complete_epochs=500,
                times_ms=times_ms,
                mean_rectified_ecg_uv=signal,
            )
        )
    participant = average_volume_locked_ecg_runs(tuple(runs))
    specification = VolumeLockedEcgSpecification(
        marker_description="Volume/V  1",
        epoch_duration_s=0.9,
        minimum_complete_epochs=50,
        peak_prominence_fraction=0.1,
        peak_minimum_distance_ms=100.0,
        peak_search_radius_ms=5.0,
        peak_maximum_latency_sd_ms=2.0,
    )

    troughs = detect_stable_artifact_troughs(participant, specification)

    assert troughs["trough_id"].tolist() == ["T01", "T02"]
    assert troughs["latency_ms"].tolist() == pytest.approx([120.0, 420.0])
    assert troughs["contributing_runs"].tolist() == [6, 6]


def test_volume_locked_ecg_specification_loads_study_defaults() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config
    from studies.pain_study.study1.figures.volume_locked_rectified_ecg import (
        volume_locked_ecg_specification,
    )

    specification = volume_locked_ecg_specification(load_study1_config())

    assert specification.marker_description == "Volume/V  1"
    assert specification.epoch_duration_s == 0.9
    assert specification.minimum_complete_epochs == 50
    assert specification.peak_minimum_distance_ms == 30.0


def test_volume_locked_ecg_figure_uses_separate_native_scale_panels() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config
    from studies.pain_study.study1.figures.volume_locked_rectified_ecg_plot import (
        build_volume_locked_ecg_figure,
    )

    participants = tuple(
        _participant_trace(subject_id, stage, sampling_frequency_hz)
        for subject_id in ("sub-0014", "sub-0015")
        for stage, sampling_frequency_hz in (("raw", 5_000.0), ("processed", 1_000.0))
    )
    peaks = pd.DataFrame(
        [
            {
                "subject_id": participant.subject_id,
                "stage": participant.stage,
                "peak_id": "P01",
                "latency_ms": 400.0,
                "mean_rectified_ecg_uv": 2.0,
            }
            for participant in participants
        ]
    )
    troughs = peaks.rename(columns={"peak_id": "trough_id"}).copy()
    troughs["mean_rectified_ecg_uv"] = 1.0

    figure = build_volume_locked_ecg_figure(
        participants,
        peaks,
        troughs,
        load_study1_config(),
    )

    try:
        assert len(figure.axes) == 4
        assert figure.axes[0].get_title() == "Original BrainVision · 5 kHz"
        assert figure.axes[1].get_title() == "BrainVision corrected · 1 kHz"
        assert not figure.axes[0].get_shared_y_axes().joined(figure.axes[0], figure.axes[1])
        assert figure.axes[0].get_ylabel() == "sub-0014\nMean |ECG| (µV)"
        assert figure.axes[2].get_xlabel() == "Time from Volume/V  1 (ms)"
        assert any(
            line.get_label() == "Equal-run participant mean" for line in figure.axes[0].lines
        )
        assert any(line.get_marker() == "v" for line in figure.axes[0].lines)
    finally:
        plt.close(figure)


def test_build_summary_requires_paired_stage_runs() -> None:
    from studies.pain_study.study1.figures.volume_locked_rectified_ecg import (
        VolumeLockedEcgSpecification,
        build_volume_locked_ecg_summary,
    )

    participants = tuple(
        _participant_trace(subject_id, stage, sampling_frequency_hz)
        for subject_id in ("sub-0014", "sub-0015")
        for stage, sampling_frequency_hz in (("raw", 5_000.0), ("processed", 1_000.0))
    )
    runs = tuple(run for participant in participants for run in participant.runs)
    specification = VolumeLockedEcgSpecification(
        marker_description="Volume/V  1",
        epoch_duration_s=0.9,
        minimum_complete_epochs=50,
        peak_prominence_fraction=0.05,
        peak_minimum_distance_ms=30.0,
        peak_search_radius_ms=5.0,
        peak_maximum_latency_sd_ms=5.0,
    )

    summary = build_volume_locked_ecg_summary(
        runs,
        specification,
        expected_runs_per_participant=2,
    )

    assert len(summary.participants) == 4
    assert set(summary.peaks["subject_id"]) == {"sub-0014", "sub-0015"}
    assert set(summary.peaks["stage"]) == {"raw", "processed"}
    assert set(summary.troughs["subject_id"]) == {"sub-0014", "sub-0015"}
    assert set(summary.troughs["stage"]) == {"raw", "processed"}

    with pytest.raises(ValueError, match="paired raw and processed run identities"):
        build_volume_locked_ecg_summary(
            runs[:-1],
            specification,
            expected_runs_per_participant=2,
        )


def test_volume_locked_ecg_writer_creates_figure_and_source_tables(tmp_path) -> None:
    from studies.pain_study.study1.config.loader import load_study1_config
    from studies.pain_study.study1.figures.plot_volume_locked_rectified_ecg import (
        write_volume_locked_ecg_summary,
    )
    from studies.pain_study.study1.figures.volume_locked_rectified_ecg import (
        VolumeLockedEcgSpecification,
        build_volume_locked_ecg_summary,
    )

    participants = tuple(
        _participant_trace(subject_id, stage, sampling_frequency_hz)
        for subject_id in ("sub-0014", "sub-0015")
        for stage, sampling_frequency_hz in (("raw", 5_000.0), ("processed", 1_000.0))
    )
    summary = build_volume_locked_ecg_summary(
        tuple(run for participant in participants for run in participant.runs),
        VolumeLockedEcgSpecification(
            marker_description="Volume/V  1",
            epoch_duration_s=0.9,
            minimum_complete_epochs=50,
            peak_prominence_fraction=0.05,
            peak_minimum_distance_ms=30.0,
            peak_search_radius_ms=5.0,
            peak_maximum_latency_sd_ms=5.0,
        ),
        expected_runs_per_participant=2,
    )

    paths = write_volume_locked_ecg_summary(
        summary,
        config=load_study1_config(),
        output_dir=tmp_path,
    )

    assert paths.svg.name == "volume_locked_rectified_ecg.svg"
    assert paths.png.name == "volume_locked_rectified_ecg.png"
    assert paths.peaks_tsv.name == "volume_locked_rectified_ecg_artifact_peaks.tsv"
    assert paths.troughs_tsv.name == "volume_locked_rectified_ecg_analyzer_troughs.tsv"
    assert all(path.is_file() for path in paths.__dict__.values())
    run_table = pd.read_csv(paths.run_traces_tsv, sep="\t")
    participant_table = pd.read_csv(paths.participant_traces_tsv, sep="\t")
    assert set(run_table["rectification_order"]) == {"abs_then_volume_mean"}
    assert set(participant_table["run_weighting"]) == {"equal_run_mean"}


def test_volume_locked_source_discovery_ignores_unrequested_psd_repairs(
    monkeypatch,
    tmp_path,
) -> None:
    import studies.pain_study.study1.figures.plot_volume_locked_rectified_ecg as module
    from studies.pain_study.study1.config.loader import load_study1_config

    calls = []

    def record_discovery(source_root, **kwargs):
        calls.append((source_root, kwargs))
        return ()

    monkeypatch.setattr(module, "discover_raw_brainvision_runs", record_discovery)
    monkeypatch.setattr(module, "discover_processed_brainvision_runs", record_discovery)

    sources = module._discover_volume_locked_sources(
        source_data_root=tmp_path,
        requested_subjects=("sub-0014", "sub-0015"),
        config=load_study1_config(),
    )

    assert sources == ()
    assert len(calls) == 2
    for source_root, arguments in calls:
        assert source_root == tmp_path
        assert arguments["requested_subjects"] == ("sub-0014", "sub-0015")
        assert arguments["source_corrections"] == ()
        assert arguments["source_exclusions"] == ()


def _participant_trace(subject_id: str, stage: str, sampling_frequency_hz: float):
    from studies.pain_study.study1.figures.volume_locked_rectified_ecg import (
        VolumeLockedEcgRun,
        average_volume_locked_ecg_runs,
    )

    sample_count = int(0.9 * sampling_frequency_hz)
    times_ms = np.arange(sample_count, dtype=float) * 1_000.0 / sampling_frequency_hz
    runs = tuple(
        VolumeLockedEcgRun(
            subject_id=subject_id,
            run_id=str(run_id),
            stage=stage,
            source_file=f"{subject_id}-{stage}-run-{run_id}.vhdr",
            sampling_frequency_hz=sampling_frequency_hz,
            n_complete_epochs=500,
            times_ms=times_ms,
            mean_rectified_ecg_uv=1.0
            + run_id * 0.1
            + np.exp(-0.5 * ((times_ms - 400.0) / 4.0) ** 2)
            - 0.8 * np.exp(-0.5 * ((times_ms - 600.0) / 4.0) ** 2),
        )
        for run_id in (1, 2)
    )
    return average_volume_locked_ecg_runs(runs)
