from __future__ import annotations

from types import SimpleNamespace

import mne
import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.preprocessing import cardiac_artifact_qc as cardiac_qc
from eeg_pipeline.preprocessing import ica_cardiac_review as cardiac_review
from eeg_pipeline.preprocessing import ica_cardiac_report as cardiac_report
from eeg_pipeline.preprocessing.cardiac_artifact_qc import (
    add_marker_ctps_columns,
    compute_cardiac_attenuation,
    compute_marker_ctps_scores,
    pulse_marker_events,
    run_cardiac_attenuation_qc,
    run_marker_ctps_qc,
    write_cardiac_attenuation_qc,
)

#: The label this study's fixtures write. Core no longer defaults to it: a beat label is
#: a search instruction, so every caller that wants the marker train must name it, and
#: these tests name it the way a study config would.
BEAT_MARKER = "Pulse Artifact/R"


def test_cardiac_review_settings_are_opt_in_and_validated() -> None:
    disabled = cardiac_review.CardiacReviewSettings.from_mapping({})
    enabled = cardiac_review.CardiacReviewSettings.from_mapping(
        {
            "enabled": True,
            "ecg_channel": "ECG",
            "epoch_window": [-0.4, 0.6],
            "baseline": [-0.4, -0.1],
            "measurement_window": [0.0, 0.4],
            "ctps_threshold": "auto",
        }
    )

    assert disabled.enabled is False
    assert enabled.enabled is True
    assert enabled.ecg_channel == "ECG"
    assert enabled.epoch_window == (-0.4, 0.6)
    assert enabled.baseline == (-0.4, -0.1)
    assert enabled.measurement_window == (0.0, 0.4)
    assert enabled.ctps_threshold == "auto"
    assert not hasattr(enabled, "accepted_questionable_runs")
    assert not hasattr(enabled, "plausible_heart_rate_bpm")

    with pytest.raises(ValueError, match="baseline"):
        cardiac_review.CardiacReviewSettings.from_mapping(
            {"epoch_window": [-0.4, 0.6], "baseline": [-0.5, -0.1]}
        )
    with pytest.raises(ValueError, match="ctps_threshold"):
        cardiac_review.CardiacReviewSettings.from_mapping({"ctps_threshold": "fixed"})
    with pytest.raises(ValueError, match="Unsupported"):
        cardiac_review.CardiacReviewSettings.from_mapping({"accepted_questionable_runs": ["run-1"]})


def _pulse_locked_raw(*, artifact_scale: float) -> mne.io.RawArray:
    sfreq = 100.0
    duration_seconds = 30.0
    times = np.arange(int(duration_seconds * sfreq)) / sfreq
    pulse_onsets = np.arange(2.0, 29.0, 1.0)
    artifact = np.zeros_like(times)
    for onset in pulse_onsets:
        artifact += artifact_scale * 100e-6 * np.exp(-0.5 * ((times - onset) / 0.03) ** 2)

    info = mne.create_info(["Cz", "Pz", "ECG"], sfreq, ["eeg", "eeg", "ecg"])
    raw = mne.io.RawArray(
        np.vstack([artifact, artifact * 0.5, np.zeros_like(artifact)]),
        info,
        verbose=False,
    )
    raw.set_annotations(
        mne.Annotations(
            pulse_onsets,
            np.zeros(len(pulse_onsets)),
            ["Pulse Artifact/R"] * len(pulse_onsets),
        )
    )
    return raw


def _signal_detectable_ecg_raw() -> mne.io.RawArray:
    sfreq = 200.0
    times = np.arange(int(30.0 * sfreq)) / sfreq
    peak_times = np.arange(1.0, 29.5, 1.0)
    ecg = 0.02e-3 * np.sin(2 * np.pi * 1.0 * times)
    eeg = np.zeros_like(times)
    for peak_time in peak_times:
        qrs = np.exp(-0.5 * ((times - peak_time) / 0.02) ** 2)
        ecg += 1.0e-3 * qrs
        eeg += 30e-6 * np.exp(-0.5 * ((times - peak_time - 0.08) / 0.04) ** 2)
    info = mne.create_info(["Cz", "Pz", "ECG"], sfreq, ["eeg", "eeg", "ecg"])
    return mne.io.RawArray(np.vstack([eeg, -0.5 * eeg, ecg]), info, verbose=False)


def test_direct_ecg_detection_does_not_require_analyzer_markers() -> None:
    settings = cardiac_review.CardiacReviewSettings.from_mapping({"enabled": True})

    detection = cardiac_review.detect_ecg_events(_signal_detectable_ecg_raw(), settings)

    assert 25 <= len(detection.events) <= 31
    assert detection.detected_beats_per_recording_minute == pytest.approx(60.0, abs=3.0)
    assert detection.events.shape[1] == 3


def test_standardize_source_epoch_runs_preserves_between_epoch_amplitude() -> None:
    times = np.array([-0.4, -0.2, 0.0, 0.2])
    run_1 = np.array([[[1.0, 3.0, 5.0, 7.0]], [[3.0, 1.0, 5.0, 7.0]]])
    run_2 = np.array([[[1.0, 3.0, 8.0, 12.0]], [[3.0, 1.0, 8.0, 12.0]]])

    standardized = cardiac_review.standardize_source_epoch_runs(
        [run_1, run_2],
        times=times,
        baseline=(-0.4, -0.2),
    )

    np.testing.assert_allclose(standardized[0][..., :2].mean(axis=-1), 0.0)
    np.testing.assert_allclose(standardized[1][..., :2].mean(axis=-1), 0.0)
    assert standardized[1][..., 2:].mean() > standardized[0][..., 2:].mean()


def test_component_cardiac_evidence_table_is_review_only() -> None:
    review = cardiac_review.ComponentCardiacReview(
        run_ids=("run-1", "run-2"),
        times=np.array([-0.1, 0.0, 0.1]),
        run_mean_z=np.ones((2, 2, 3)),
        correlation_scores=np.array([[-0.1, 0.8], [0.2, -0.6]]),
        ctps_scores=np.array([[0.08, 0.31], [0.07, 0.20]]),
        correlation_flags=np.array([[False, True], [False, True]]),
        ctps_flags=np.array([[False, True], [False, False]]),
        r_locked_epoch_counts=np.array([20, 20]),
        run_ecg_z=np.ones((2, 3)),
    )

    statuses = pd.DataFrame(
        {
            "component": [0, 1],
            "status": ["bad", "good"],
            "status_description": ["Auto-detected eye blink", ""],
        }
    )
    table = cardiac_review.component_cardiac_evidence_table(review, statuses=statuses)
    run_table = cardiac_review.component_run_cardiac_evidence_table(review)

    assert table.columns.tolist() == [
        "component",
        "current_ica_status",
        "current_status_description",
    ]
    assert table["current_ica_status"].tolist() == ["bad", "good"]
    assert "manual_review_recommended" not in table
    assert len(run_table) == 4
    assert run_table["recording_id"].tolist() == ["run-1", "run-1", "run-2", "run-2"]
    assert run_table.columns.tolist() == [
        "recording_id",
        "component",
        "mne_ecg_correlation_score",
        "mne_ctps_score",
        "mne_correlation_flag",
        "mne_ctps_flag",
        "r_locked_epoch_count",
    ]
    assert run_table["mne_ecg_correlation_score"].tolist() == [-0.1, 0.8, 0.2, -0.6]


def test_cardiac_report_entries_use_review_order() -> None:
    entries = [
        SimpleNamespace(name="How to review ECG artifacts", tags=("ica-cardiac-review",)),
        SimpleNamespace(
            name="ECG detection and provisional correction by run",
            tags=("ica-cardiac-review",),
        ),
        SimpleNamespace(
            name="ICA components: R-locked cardiac evidence",
            tags=("ica-cardiac-review",),
        ),
        SimpleNamespace(name="ECG detection summary", tags=("ica-cardiac-review",)),
    ]

    assert cardiac_report._ordered_cardiac_indices(entries) == [0, 3, 1, 2]


def test_cardiac_review_guide_has_no_custom_classification_language() -> None:
    html = cardiac_report._cardiac_review_guide_html(
        cardiac_review.CardiacReviewSettings.from_mapping({})
    )

    assert "MNE" in html
    assert "reliable" not in html.lower()
    assert "recommended" not in html.lower()
    assert "ranking" not in html.lower()


def test_the_guide_reports_which_detector_the_runs_actually_used() -> None:
    """The guide once asserted flatly that the review needed no annotated beat train.

    ``detect_ecg_events`` prefers that train wherever a run carries one, so on a dataset
    that has it the sentence was false for every run in the section. The guide has to read
    the runs rather than assert one of the two branches.
    """
    settings = cardiac_review.CardiacReviewSettings.from_mapping({})

    from_markers = cardiac_report._cardiac_review_guide_html(
        settings, beat_sources=(cardiac_review.MARKER_TRAIN_SOURCE,) * 3
    )
    from_channel = cardiac_report._cardiac_review_guide_html(
        settings, beat_sources=(cardiac_review.ECG_CHANNEL_SOURCE,) * 3
    )

    assert "the recording's markers" in from_markers
    assert "does not depend on an annotated beat train" not in from_markers
    assert "does not depend on an annotated beat train" in from_channel


def test_the_guide_names_a_split_between_the_two_detectors() -> None:
    """The two sources fail on different runs, so a section can hold both."""
    settings = cardiac_review.CardiacReviewSettings.from_mapping({})

    html = cardiac_report._cardiac_review_guide_html(
        settings,
        beat_sources=(
            cardiac_review.MARKER_TRAIN_SOURCE,
            cardiac_review.ECG_CHANNEL_SOURCE,
            cardiac_review.ECG_CHANNEL_SOURCE,
        ),
    )

    assert "1 of 3" in html or "2 of 3" in html


def test_pulse_marker_events_use_preserved_analyzer_annotations() -> None:
    raw = _pulse_locked_raw(artifact_scale=1.0)

    events, is_fallback = pulse_marker_events(raw, marker_description=BEAT_MARKER)

    assert events.shape == (27, 3)
    assert events[0].tolist() == [200, 0, 999]


def test_pulse_marker_events_preserve_sample_positions_after_crop() -> None:
    raw = _pulse_locked_raw(artifact_scale=1.0)
    raw.crop(tmin=1.0, tmax=29.0)

    events, is_fallback = pulse_marker_events(raw, marker_description=BEAT_MARKER)

    assert raw.first_samp == 100
    assert events[0].tolist() == [200, 0, 999]


def test_compute_cardiac_attenuation_measures_marker_locked_rms_reduction() -> None:
    metrics = compute_cardiac_attenuation(
        _pulse_locked_raw(artifact_scale=1.0),
        _pulse_locked_raw(artifact_scale=0.25),
        recording_id="sub-0001_run-1",
        baseline=(-0.25, -0.05),
        measurement_window=(-0.05, 0.4),
        marker_description=BEAT_MARKER,
    )

    assert metrics.recording_id == "sub-0001_run-1"
    assert metrics.marker_count == 27
    assert metrics.after_rms_uv == pytest.approx(metrics.before_rms_uv * 0.25)
    assert metrics.attenuation_percent == pytest.approx(75.0)


def test_compute_cardiac_attenuation_uses_matching_average_references() -> None:
    before = _pulse_locked_raw(artifact_scale=1.0)
    after = before.copy().set_eeg_reference("average", projection=False, verbose=False)

    metrics = compute_cardiac_attenuation(
        before,
        after,
        recording_id="sub-0001_run-1",
        baseline=(-0.25, -0.05),
        measurement_window=(-0.05, 0.4),
        marker_description=BEAT_MARKER,
    )

    assert metrics.attenuation_percent == pytest.approx(0.0, abs=1e-8)


def test_add_marker_ctps_columns_flags_without_changing_manual_status() -> None:
    components = pd.DataFrame(
        {
            "component": [0, 1, 2],
            "status": ["good", "bad", "good"],
            "status_description": ["", "ICLabel", ""],
        }
    )

    result = add_marker_ctps_columns(
        components,
        np.array([0.04, 0.12, 0.3]),
        threshold=0.1,
    )

    assert result["status"].tolist() == ["good", "bad", "good"]
    assert result["status_description"].tolist() == ["", "ICLabel", ""]
    assert result["analyzer_marker_ctps_score"].tolist() == [0.04, 0.12, 0.3]
    assert result["analyzer_marker_ctps_flag"].tolist() == [False, True, True]


def test_add_marker_ctps_columns_rejects_component_mismatch() -> None:
    components = pd.DataFrame({"component": [0, 2], "status": ["good", "good"]})

    with pytest.raises(ValueError, match="component table"):
        add_marker_ctps_columns(components, np.array([0.1, 0.2]), threshold=0.1)


def test_compute_marker_ctps_scores_uses_marker_locked_epochs() -> None:
    class FakeIca:
        def find_bads_ecg(self, epochs, **kwargs):
            assert len(epochs) == 54
            assert kwargs["method"] == "ctps"
            assert kwargs["threshold"] == 0.1
            return [1], np.array([0.04, 0.2])

    scores, any_fallback = compute_marker_ctps_scores(
        [_pulse_locked_raw(artifact_scale=1.0), _pulse_locked_raw(artifact_scale=0.5)],
        FakeIca(),
        threshold=0.1,
        epoch_window=(-0.25, 0.5),
        marker_description=BEAT_MARKER,
    )

    assert scores.tolist() == [0.04, 0.2]


def test_write_cardiac_attenuation_qc_writes_run_metrics(tmp_path) -> None:
    output_path = write_cardiac_attenuation_qc(
        [
            (
                "sub-0001_run-1",
                _pulse_locked_raw(artifact_scale=1.0),
                _pulse_locked_raw(artifact_scale=0.25),
            )
        ],
        output_path=tmp_path / "cardiac_review.tsv",
        baseline=(-0.25, -0.05),
        measurement_window=(-0.05, 0.4),
        marker_description=BEAT_MARKER,
    )

    table = pd.read_csv(output_path, sep="\t")
    assert table["recording_id"].tolist() == ["sub-0001_run-1"]
    assert table["marker_count"].tolist() == [27]
    assert table["attenuation_percent"].iloc[0] == pytest.approx(75.0)
    assert output_path.with_suffix(".png").is_file()


def test_run_marker_ctps_qc_updates_native_table_without_excluding(monkeypatch, tmp_path) -> None:
    eeg_dir = tmp_path / "sub-0001" / "eeg"
    eeg_dir.mkdir(parents=True)
    ica_path = eeg_dir / "sub-0001_proc-ica_ica.fif"
    raw_path = eeg_dir / "sub-0001_task-pain_run-1_proc-filt_raw.fif"
    components_path = eeg_dir / "sub-0001_proc-ica_components.tsv"
    ica_path.touch()
    raw_path.touch()
    pd.DataFrame(
        {
            "component": [0, 1],
            "status": ["bad", "good"],
            "status_description": ["ICLabel", ""],
        }
    ).to_csv(components_path, sep="\t", index=False)

    fake_ica = object()
    fake_raw = _pulse_locked_raw(artifact_scale=1.0)
    monkeypatch.setattr(mne.preprocessing, "read_ica", lambda *_args, **_kwargs: fake_ica)
    monkeypatch.setattr(mne.io, "read_raw_fif", lambda *_args, **_kwargs: fake_raw)
    monkeypatch.setattr(
        "eeg_pipeline.preprocessing.cardiac_artifact_qc.compute_marker_ctps_scores",
        lambda *_args, **_kwargs: (np.array([0.2, 0.04]), False),
    )

    output_path = run_marker_ctps_qc(
        pipeline_root=tmp_path,
        subjects=["0001"],
        task="pain",
        threshold=0.1,
        epoch_window=(-0.25, 0.5),
    )

    updated = pd.read_csv(components_path, sep="\t")
    assert updated["status"].tolist() == ["bad", "good"]
    assert updated["analyzer_marker_ctps_flag"].tolist() == [True, False]
    assert output_path.is_file()


def test_run_cardiac_attenuation_qc_pairs_filtered_and_clean_raws(
    monkeypatch,
    tmp_path,
) -> None:
    eeg_dir = tmp_path / "sub-0001" / "eeg"
    eeg_dir.mkdir(parents=True)
    filtered = eeg_dir / "sub-0001_task-pain_run-1_proc-filt_raw.fif"
    clean = eeg_dir / "sub-0001_task-pain_run-1_proc-clean_raw.fif"
    filtered.touch()
    clean.touch()

    def read_raw(path, **_kwargs):
        scale = 0.25 if "proc-clean" in str(path) else 1.0
        return _pulse_locked_raw(artifact_scale=scale)

    monkeypatch.setattr(mne.io, "read_raw_fif", read_raw)

    output_path = run_cardiac_attenuation_qc(
        pipeline_root=tmp_path,
        subjects=["0001"],
        task="pain",
        baseline=(-0.25, -0.05),
        measurement_window=(-0.05, 0.4),
        marker_description=BEAT_MARKER,
    )

    table = pd.read_csv(output_path, sep="\t")
    assert table["recording_id"].tolist() == ["sub-0001_task-pain_run-1"]
    assert table["attenuation_percent"].iloc[0] == pytest.approx(75.0)


def test_attenuation_is_reported_in_decibels_as_well_as_percent() -> None:
    """dB is the pipeline's convention for attenuation; percent is kept for continuity."""
    import numpy as np

    from eeg_pipeline.preprocessing.cardiac_artifact_qc import CardiacAttenuationMetrics

    metrics = CardiacAttenuationMetrics(
        recording_id="sub-0001_run-1",
        marker_count=100,
        before_rms_uv=10.0,
        after_rms_uv=5.0,
        attenuation_percent=50.0,
        attenuation_db=float(20.0 * np.log10(10.0 / 5.0)),
    )

    # Halving amplitude is 6.02 dB, which 50% describes far less transparently.
    assert metrics.attenuation_db == pytest.approx(6.0206, abs=1e-3)
    assert metrics.attenuation_percent == pytest.approx(50.0)


def _pulse_locked_raw_with_live_ecg(*, artifact_scale: float) -> mne.io.RawArray:
    """A pulse-locked recording whose ECG channel carries a real, large QRS trace."""
    sfreq = 100.0
    times = np.arange(int(30.0 * sfreq)) / sfreq
    pulse_onsets = np.arange(2.0, 29.0, 1.0)
    artifact = np.zeros_like(times)
    ecg = np.zeros_like(times)
    for onset in pulse_onsets:
        shape = np.exp(-0.5 * ((times - onset) / 0.03) ** 2)
        artifact += artifact_scale * 100e-6 * shape
        # Two orders of magnitude above the EEG artifact, as a real ECG lead is.
        ecg += 10e-3 * shape

    info = mne.create_info(["Cz", "Pz", "ECG"], sfreq, ["eeg", "eeg", "ecg"])
    raw = mne.io.RawArray(np.vstack([artifact, artifact * 0.5, ecg]), info, verbose=False)
    raw.set_annotations(
        mne.Annotations(
            pulse_onsets,
            np.zeros(len(pulse_onsets)),
            ["Pulse Artifact/R"] * len(pulse_onsets),
        )
    )
    return raw


def test_attenuation_ignores_the_ecg_channel_ica_never_touched() -> None:
    """ICA cleans EEG only, so a live ECG trace must not dilute the measured attenuation.

    The ECG channel is identical before and after and is perfectly R-locked, so including
    it in the average would dominate the RMS and report a well-cleaned run as barely
    cleaned at all.
    """
    metrics = compute_cardiac_attenuation(
        _pulse_locked_raw_with_live_ecg(artifact_scale=1.0),
        _pulse_locked_raw_with_live_ecg(artifact_scale=0.25),
        recording_id="sub-0001_run-1",
        baseline=(-0.25, -0.05),
        measurement_window=(-0.05, 0.4),
        marker_description=BEAT_MARKER,
    )

    assert metrics.attenuation_percent == pytest.approx(75.0)


def test_marker_ctps_qc_finds_session_organized_derivatives(monkeypatch, tmp_path) -> None:
    """A ses- directory and a missing run entity must not make the QC find nothing."""
    eeg_dir = tmp_path / "sub-0001" / "ses-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    ica_path = eeg_dir / "sub-0001_ses-01_proc-ica_ica.fif"
    raw_path = eeg_dir / "sub-0001_ses-01_task-pain_proc-filt_raw.fif"
    components_path = eeg_dir / "sub-0001_ses-01_proc-ica_components.tsv"
    ica_path.touch()
    raw_path.touch()
    pd.DataFrame(
        {
            "component": [0, 1],
            "status": ["bad", "good"],
            "status_description": ["ICLabel", ""],
        }
    ).to_csv(components_path, sep="\t", index=False)

    monkeypatch.setattr(mne.preprocessing, "read_ica", lambda *_a, **_k: object())
    monkeypatch.setattr(
        mne.io, "read_raw_fif", lambda *_a, **_k: _pulse_locked_raw(artifact_scale=1.0)
    )
    monkeypatch.setattr(
        "eeg_pipeline.preprocessing.cardiac_artifact_qc.compute_marker_ctps_scores",
        lambda *_a, **_k: (np.array([0.2, 0.04]), False),
    )

    output_path = run_marker_ctps_qc(
        pipeline_root=tmp_path,
        subjects=["0001"],
        task="pain",
        threshold=0.1,
        epoch_window=(-0.25, 0.5),
    )

    updated = pd.read_csv(components_path, sep="\t")
    assert updated["analyzer_marker_ctps_flag"].tolist() == [True, False]
    summary = pd.read_csv(output_path, sep="\t")
    assert summary["decomposition_id"].unique().tolist() == ["sub-0001_ses-01"]


def test_attenuation_qc_reads_one_run_pair_at_a_time(monkeypatch, tmp_path) -> None:
    """The cohort must never be held in memory all at once."""
    eeg_dir = tmp_path / "sub-0001" / "eeg"
    eeg_dir.mkdir(parents=True)
    for run in (1, 2):
        (eeg_dir / f"sub-0001_task-pain_run-{run}_proc-filt_raw.fif").touch()
        (eeg_dir / f"sub-0001_task-pain_run-{run}_proc-clean_raw.fif").touch()

    reads = 0
    reads_before_first_measurement = None

    def read_raw(path, **_kwargs):
        nonlocal reads
        reads += 1
        scale = 0.25 if "proc-clean" in str(path) else 1.0
        return _pulse_locked_raw(artifact_scale=scale)

    real_compute = cardiac_qc.compute_cardiac_attenuation

    def tracked_compute(*args, **kwargs):
        nonlocal reads_before_first_measurement
        if reads_before_first_measurement is None:
            reads_before_first_measurement = reads
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(mne.io, "read_raw_fif", read_raw)
    monkeypatch.setattr(cardiac_qc, "compute_cardiac_attenuation", tracked_compute)

    run_cardiac_attenuation_qc(
        pipeline_root=tmp_path,
        subjects=["0001"],
        task="pain",
        baseline=(-0.25, -0.05),
        measurement_window=(-0.05, 0.4),
        marker_description=BEAT_MARKER,
    )

    # Two runs exist. Streaming measures the first pair after reading exactly that pair;
    # the previous implementation read all four files before measuring anything.
    assert reads == 4
    assert reads_before_first_measurement == 2


def _dead_ecg_raw() -> mne.io.RawArray:
    """A recording whose ECG lead came off: the channel exists and carries nothing.

    Flat rather than noisy on purpose. MNE's detector finds spurious peaks in noise, so a
    noisy fixture would be testing the detector's tuning; a detached lead is the case where
    there is provably no cardiac signal to find.
    """
    sfreq = 200.0
    times = np.arange(int(30.0 * sfreq)) / sfreq
    info = mne.create_info(["Cz", "Pz", "ECG"], sfreq, ["eeg", "eeg", "ecg"])
    return mne.io.RawArray(np.zeros((3, times.size)), info, verbose=False)


def test_an_ecg_with_no_detectable_beats_raises_a_typed_signal() -> None:
    """A run whose ECG yields no R peaks is a measurement that did not resolve.

    The caller has to tell that apart from a programming error so it can keep the rest of
    the study running, and a bare ``ValueError`` cannot express the difference: catching
    ``ValueError`` around the detector would swallow every genuine bug inside it too.
    """
    settings = cardiac_review.CardiacReviewSettings.from_mapping({"enabled": True})

    with pytest.raises(cardiac_review.UnusableEcg):
        cardiac_review.detect_ecg_events(_dead_ecg_raw(), settings)


def test_an_unusable_ecg_is_still_a_value_error() -> None:
    """Existing callers catch ValueError; narrowing the type must not slip past them."""
    assert issubclass(cardiac_review.UnusableEcg, ValueError)


def test_a_detectable_ecg_is_unaffected_by_the_new_signal() -> None:
    """The guard must not change what happens to a run that detects normally."""
    settings = cardiac_review.CardiacReviewSettings.from_mapping({"enabled": True})

    detection = cardiac_review.detect_ecg_events(_signal_detectable_ecg_raw(), settings)

    assert 25 <= len(detection.events) <= 31


# --------------------------------------------------------------------------------------
# One unusable run must not abort the study
# --------------------------------------------------------------------------------------


def _review_run_raw(*, detectable: bool, seconds: float = 30.0) -> mne.io.RawArray:
    """One run for the end-to-end review: EEG plus an ECG that may or may not resolve."""
    sfreq = 200.0
    times = np.arange(int(seconds * sfreq)) / sfreq
    rng = np.random.default_rng(0 if detectable else 1)
    eeg = 5e-6 * rng.standard_normal((3, times.size))
    ecg = np.zeros_like(times)
    if detectable:
        for peak_time in np.arange(1.0, seconds - 0.5, 1.0):
            qrs = np.exp(-0.5 * ((times - peak_time) / 0.02) ** 2)
            ecg += 1.0e-3 * qrs
            eeg += 30e-6 * np.exp(-0.5 * ((times - peak_time - 0.08) / 0.04) ** 2)
    info = mne.create_info(["Cz", "Pz", "Oz", "ECG"], sfreq, ["eeg", "eeg", "eeg", "ecg"])
    raw = mne.io.RawArray(np.vstack([eeg, ecg]), info, verbose=False)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), verbose=False)
    return raw


def _write_review_inputs(directory, *, detectable_runs):
    """Write the filtered runs, the ICA and its component table, and an empty report."""
    from eeg_pipeline.preprocessing.ica_exclusions import components_path_for_ica

    prefix = "sub-0000"
    filtered_paths = []
    for index, detectable in enumerate(detectable_runs, start=1):
        raw = _review_run_raw(detectable=detectable)
        path = directory / f"{prefix}_task-t_run-{index}_proc-filt_raw.fif"
        raw.save(path, overwrite=True, verbose="ERROR")
        filtered_paths.append(path)

    fit = _review_run_raw(detectable=True)
    ica = mne.preprocessing.ICA(n_components=2, random_state=0, max_iter=200, verbose="ERROR")
    ica.fit(fit.copy().pick("eeg"), verbose="ERROR")
    ica_path = directory / f"{prefix}_proc-ica_ica.fif"
    ica.save(ica_path, overwrite=True, verbose="ERROR")

    pd.DataFrame(
        {
            "component": [0, 1],
            "status": ["bad", "good"],
            "status_description": ["ecg", "kept"],
        }
    ).to_csv(components_path_for_ica(ica_path), sep="\t", index=False)

    report_path = directory / f"{prefix}_report.h5"
    report = mne.Report(title="sub-0000", verbose="ERROR")
    # The cardiac review inserts itself before the ICA component section, which the real
    # report already carries from the decomposition step. Without it the review has no
    # anchor to order itself against.
    report.add_html(
        html="<p>components</p>",
        title="ICA components",
        section="ICA: components",
        tags=("ica", "ica-component-review"),
    )
    report.save(report_path, overwrite=True, open_browser=False, verbose="ERROR")
    return filtered_paths, ica_path, report_path


def test_one_unusable_run_does_not_abort_the_cardiac_review(tmp_path) -> None:
    """A dead ECG lead in run 2 must not cost the reviewer runs 1 and 3.

    The failure this pins: a single run whose ECG yields no R peaks raised out of the
    review, out of the ICA stage, and out of the pipeline -- ending a 15-participant run
    at the review step and leaving no report for anybody, including the fourteen
    participants whose recordings were fine.
    """
    filtered_paths, ica_path, report_path = _write_review_inputs(
        tmp_path, detectable_runs=(True, False, True)
    )
    settings = cardiac_review.CardiacReviewSettings.from_mapping({"enabled": True})

    output = cardiac_report.generate_ica_cardiac_review(
        filtered_raw_paths=filtered_paths,
        ica_path=ica_path,
        report_path=report_path,
        output_path=tmp_path / "sub-0000_desc-icaecg_components.tsv",
        settings=settings,
    )

    assert output.is_file()
    runs = pd.read_csv(output.with_name(output.name.replace("components", "runs")), sep="\t")
    reviewed = set(runs["recording_id"])
    assert any("run-1" in name for name in reviewed)
    assert any("run-3" in name for name in reviewed)
    # The unusable run is absent from the reviewed set rather than carrying a fabricated rate.
    assert not any("run-2" in name for name in reviewed)


def test_the_unusable_run_is_reported_rather_than_dropped_silently(tmp_path) -> None:
    """A run excluded from the evidence has to be named, or the denominator lies."""
    filtered_paths, ica_path, report_path = _write_review_inputs(
        tmp_path, detectable_runs=(True, False, True)
    )
    settings = cardiac_review.CardiacReviewSettings.from_mapping({"enabled": True})

    cardiac_report.generate_ica_cardiac_review(
        filtered_raw_paths=filtered_paths,
        ica_path=ica_path,
        report_path=report_path,
        output_path=tmp_path / "sub-0000_desc-icaecg_components.tsv",
        settings=settings,
    )

    html = mne.open_report(report_path).html
    document = " ".join(html) if isinstance(html, (list, tuple)) else str(html)
    assert "run-2" in document
    assert "R peaks" in document or "no usable" in document.lower()


def test_a_subject_with_no_usable_run_still_reports_that(tmp_path) -> None:
    """With every ECG dead there is no review to draw, and that is itself the finding."""
    filtered_paths, ica_path, report_path = _write_review_inputs(
        tmp_path, detectable_runs=(False, False)
    )
    settings = cardiac_review.CardiacReviewSettings.from_mapping({"enabled": True})

    output = cardiac_report.generate_ica_cardiac_review(
        filtered_raw_paths=filtered_paths,
        ica_path=ica_path,
        report_path=report_path,
        output_path=tmp_path / "sub-0000_desc-icaecg_components.tsv",
        settings=settings,
    )

    assert output.is_file()
    html = mne.open_report(report_path).html
    document = " ".join(html) if isinstance(html, (list, tuple)) else str(html)
    assert "no usable" in document.lower() or "no run" in document.lower()


# --------------------------------------------------------------------------------------
# Beat source: Analyzer markers in preference to channel detection
# --------------------------------------------------------------------------------------


def _raw_with_markers(*, marker_interval_s: float, qrs_interval_s: float) -> mne.io.RawArray:
    """A recording whose Analyzer markers and ECG QRS deliberately disagree.

    The two intervals differ so a test can tell which source a rate came from. Analyzer's
    marker train is the one that was validated against the recording, so it has to win.
    """
    sfreq = 200.0
    duration = 60.0
    times = np.arange(int(duration * sfreq)) / sfreq
    ecg = 0.02e-3 * np.sin(2 * np.pi * 1.0 * times)
    eeg = np.zeros_like(times)
    for peak_time in np.arange(1.0, duration - 0.5, qrs_interval_s):
        ecg += 1.0e-3 * np.exp(-0.5 * ((times - peak_time) / 0.02) ** 2)
        eeg += 30e-6 * np.exp(-0.5 * ((times - peak_time - 0.08) / 0.04) ** 2)
    info = mne.create_info(["Cz", "Pz", "ECG"], sfreq, ["eeg", "eeg", "ecg"])
    raw = mne.io.RawArray(np.vstack([eeg, -0.5 * eeg, ecg]), info, verbose=False)
    onsets = np.arange(1.0, duration - 0.5, marker_interval_s)
    raw.set_annotations(
        mne.Annotations(onsets, np.zeros(onsets.size), ["Pulse Artifact/R"] * onsets.size)
    )
    return raw


def test_the_analyzer_marker_train_is_preferred_over_channel_detection() -> None:
    """Analyzer's markers drove the correction, and its detection is the validated one.

    On this dataset ``find_ecg_events`` disagrees with the marker train on the same runs --
    reporting 8 bpm where the markers report 61 -- so the review must not take the channel
    detector's word where a marker train exists.
    """
    settings = cardiac_review.CardiacReviewSettings.from_mapping(
        {"enabled": True, "marker_description": "Pulse Artifact/R"}
    )
    raw = _raw_with_markers(marker_interval_s=1.2, qrs_interval_s=1.0)

    detection = cardiac_review.detect_ecg_events(raw, settings)

    assert detection.source == "annotation-markers"
    # 1.2 s between markers is 50 bpm; the QRS train would have given 60.
    assert detection.detected_beats_per_recording_minute == pytest.approx(50.0, abs=3.0)


def test_marker_rate_exposes_missing_beats_over_the_recording() -> None:
    """A long detection gap must lower completeness instead of being hidden by median RR."""
    sfreq = 100.0
    duration_s = 120.0
    info = mne.create_info(["Cz", "ECG"], sfreq, ["eeg", "ecg"])
    raw = mne.io.RawArray(np.zeros((2, int(sfreq * duration_s))), info, verbose=False)
    onsets = np.r_[np.arange(1.0, 31.0), np.arange(90.0, 120.0)]
    raw.set_annotations(
        mne.Annotations(onsets, np.zeros(onsets.size), ["Pulse Artifact/R"] * onsets.size)
    )
    settings = cardiac_review.CardiacReviewSettings.from_mapping(
        {"enabled": True, "marker_description": "Pulse Artifact/R"}
    )

    detection = cardiac_review.detect_ecg_events(raw, settings)

    assert detection.detected_beats_per_recording_minute == pytest.approx(30.0, abs=0.1)


def test_channel_detection_is_used_when_no_marker_train_exists() -> None:
    """33 of 90 runs in this dataset carry no markers; they still need a beat train."""
    settings = cardiac_review.CardiacReviewSettings.from_mapping({"enabled": True})

    detection = cardiac_review.detect_ecg_events(_signal_detectable_ecg_raw(), settings)

    assert detection.source == "ecg-channel"
    assert detection.detected_beats_per_recording_minute == pytest.approx(60.0, abs=3.0)


def test_neither_source_resolving_is_still_an_unusable_ecg() -> None:
    settings = cardiac_review.CardiacReviewSettings.from_mapping({"enabled": True})

    with pytest.raises(cardiac_review.UnusableEcg):
        cardiac_review.detect_ecg_events(_dead_ecg_raw(), settings)


def test_a_marker_train_too_short_to_use_falls_back_to_the_channel() -> None:
    """Two markers are not a beat train, and the channel may still carry one."""
    settings = cardiac_review.CardiacReviewSettings.from_mapping({"enabled": True})
    raw = _signal_detectable_ecg_raw()
    raw.set_annotations(mne.Annotations([1.0, 2.0], [0.0, 0.0], ["Pulse Artifact/R"] * 2))

    detection = cardiac_review.detect_ecg_events(raw, settings)

    assert detection.source == "ecg-channel"
