from __future__ import annotations

from types import SimpleNamespace

import mne
import numpy as np
import pandas as pd
import pytest

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
    assert detection.average_pulse_bpm == pytest.approx(60.0, abs=3.0)
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


def test_pulse_marker_events_use_preserved_analyzer_annotations() -> None:
    raw = _pulse_locked_raw(artifact_scale=1.0)

    events = pulse_marker_events(raw)

    assert events.shape == (27, 3)
    assert events[0].tolist() == [200, 0, 999]


def test_pulse_marker_events_preserve_sample_positions_after_crop() -> None:
    raw = _pulse_locked_raw(artifact_scale=1.0)
    raw.crop(tmin=1.0, tmax=29.0)

    events = pulse_marker_events(raw)

    assert raw.first_samp == 100
    assert events[0].tolist() == [200, 0, 999]


def test_compute_cardiac_attenuation_measures_marker_locked_rms_reduction() -> None:
    metrics = compute_cardiac_attenuation(
        _pulse_locked_raw(artifact_scale=1.0),
        _pulse_locked_raw(artifact_scale=0.25),
        recording_id="sub-0001_run-1",
        baseline=(-0.25, -0.05),
        measurement_window=(-0.05, 0.4),
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

    scores = compute_marker_ctps_scores(
        [_pulse_locked_raw(artifact_scale=1.0), _pulse_locked_raw(artifact_scale=0.5)],
        FakeIca(),
        threshold=0.1,
        epoch_window=(-0.25, 0.5),
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
    )

    table = pd.read_csv(output_path, sep="\t")
    assert table["recording_id"].tolist() == ["sub-0001_run-1"]
    assert table["marker_count"].tolist() == [27]
    assert table["attenuation_percent"].iloc[0] == pytest.approx(75.0)
    assert output_path.with_suffix(".png").is_file()


def test_run_marker_ctps_qc_updates_native_table_without_excluding(monkeypatch, tmp_path) -> None:
    eeg_dir = tmp_path / "sub-0001" / "eeg"
    eeg_dir.mkdir(parents=True)
    ica_path = eeg_dir / "sub-0001_proc-icafit_ica.fif"
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
        lambda *_args, **_kwargs: np.array([0.2, 0.04]),
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
    )

    table = pd.read_csv(output_path, sep="\t")
    assert table["recording_id"].tolist() == ["sub-0001_task-pain_run-1"]
    assert table["attenuation_percent"].iloc[0] == pytest.approx(75.0)
