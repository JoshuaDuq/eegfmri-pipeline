from __future__ import annotations

import mne
import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.preprocessing.cardiac_artifact_qc import (
    add_marker_ctps_columns,
    compute_cardiac_attenuation,
    compute_marker_ctps_scores,
    pulse_marker_events,
    run_cardiac_attenuation_qc,
    run_marker_ctps_qc,
    write_cardiac_attenuation_qc,
)


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
        output_path=tmp_path / "cardiac_qc.tsv",
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
