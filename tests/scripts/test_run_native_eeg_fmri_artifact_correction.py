from __future__ import annotations

import csv
import json
from pathlib import Path

import mne
import numpy as np
from PIL import Image
import pytest

from eeg_pipeline.preprocessing.eeg_fmri.cardiac import QrsDetection, QrsQuality
from eeg_pipeline.preprocessing.eeg_fmri.cohort_spectrum import (
    aggregate_cohort_scanner_spectra,
    extract_run_scanner_spectra,
)
from eeg_pipeline.preprocessing.eeg_fmri.neuxus_qrs import NeuXusQrsDetection
from eeg_pipeline.preprocessing.eeg_fmri.pipeline import NativeCorrectionResult
from eeg_pipeline.preprocessing.eeg_fmri.qc import (
    CardiacLockedComparison,
    CardiacLockedSummary,
    HarmonicSpectrum,
    HarmonicStageQc,
)
from eeg_pipeline.preprocessing.eeg_fmri import plotting as qc_plotting
from studies.pain_study.scripts import run_native_eeg_fmri_artifact_correction as runner
from tests import REPO_ROOT

CONFIG_PATH = (
    REPO_ROOT
    / "studies"
    / "pain_study"
    / "scripts"
    / "config"
    / "native_eeg_fmri_artifact_correction.yaml"
)


def _input_root(tmp_path: Path) -> Path:
    input_root = tmp_path / "marker-sanitized"
    header = input_root / "sub-0001" / "eeg" / "recording.vhdr"
    header.parent.mkdir(parents=True)
    header.write_text("Brain Vision Data Exchange Header File Version 1.0\n", encoding="utf-8")
    manifest = input_root / "marker_sanitization_manifest.tsv"
    row = {
        "subject": "sub-0001",
        "run": "1",
        "staged_vhdr": str(header),
        "source_vhdr_sha256": "a" * 64,
        "source_vmrk_sha256": "b" * 64,
        "source_eeg_size": "1000",
        "source_eeg_mtime_ns": "2000",
        "verified": "True",
    }
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)
    return input_root


def _native_result() -> NativeCorrectionResult:
    sampling_frequency = 1_000.0
    raw = mne.io.RawArray(
        np.zeros((2, 2_000)),
        mne.create_info(["C3", "ECG"], sampling_frequency, ["eeg", "ecg"]),
        verbose=False,
    )
    qrs_times = np.arange(0.1, 1.91, 0.2)
    peak_samples = np.rint(qrs_times * 250.0).astype(int)
    probabilities = np.zeros(500)
    probabilities[peak_samples] = 0.9
    qrs = QrsDetection(
        times=qrs_times,
        quality=QrsQuality(
            qrs_count=qrs_times.size,
            median_heart_rate_bpm=60.0,
            minimum_rr_seconds=0.2,
            median_rr_seconds=1.0,
            maximum_rr_seconds=1.8,
            abnormal_rr_count=2,
            abnormal_rr_fraction=2 / (qrs_times.size - 1),
            temporal_coverage=0.95,
            warnings=("two abnormal intervals",),
            correction_permitted=True,
        ),
        diagnostics=NeuXusQrsDetection(
            times=qrs_times,
            peak_samples=peak_samples,
            filtered_ecg=np.zeros(500),
            probabilities=probabilities,
            probability_support=np.full(500, 4),
            sampling_frequency_hz=250.0,
            model_sha256="d" * 64,
        ),
    )
    cardiac_times = np.array([-0.2, 0.0, 0.2, 0.4])
    before = CardiacLockedSummary(
        channel_names=("C3",),
        times=cardiac_times,
        median_evoked=np.array([[0.0, 4.0, -2.0, 0.0]]),
        valid_epoch_count=8,
        median_evoked_rms=2.0,
        median_evoked_peak_to_peak=6.0,
        channel_rms=np.array([2.0]),
        channel_peak_to_peak=np.array([6.0]),
    )
    after = CardiacLockedSummary(
        channel_names=("C3",),
        times=cardiac_times,
        median_evoked=np.array([[0.0, 1.0, -0.5, 0.0]]),
        valid_epoch_count=8,
        median_evoked_rms=0.5,
        median_evoked_peak_to_peak=1.5,
        channel_rms=np.array([0.5]),
        channel_peak_to_peak=np.array([1.5]),
    )
    harmonic_summaries = {stage: {} for stage in ("raw", "gradient_corrected", "final")}
    for summary in harmonic_summaries.values():
        summary["n_channels"] = 12
    frequencies = np.linspace(15.0, 90.0, 1_501)
    stage_spectra = {
        "raw": np.full(frequencies.shape, -105.0),
        "gradient_corrected": np.full(frequencies.shape, -110.0),
        "final": np.full(frequencies.shape, -112.0),
    }
    for index, (prefix, frequency) in enumerate(
        zip(
            ("harmonic_18_23", "harmonic_38_43", "harmonic_56_67", "harmonic_77_85"),
            (20.0, 41.0, 61.0, 82.0),
            strict=True,
        )
    ):
        harmonic_summaries["raw"][f"{prefix}_reference_hz"] = frequency
        harmonic_summaries["raw"][f"{prefix}_reference_power_db"] = 20.0 + index
        harmonic_summaries["gradient_corrected"][f"{prefix}_reference_power_db"] = 3.0 + index
        harmonic_summaries["final"][f"{prefix}_reference_power_db"] = 1.0 + index
        harmonic_summaries["final"][f"{prefix}_reference_local_prominence_db"] = 7.0 + index
        peak = np.exp(-0.5 * ((frequencies - frequency) / 0.12) ** 2)
        stage_spectra["raw"] += 32.0 * peak
        stage_spectra["gradient_corrected"] += 14.0 * peak
        stage_spectra["final"] += 8.0 * peak
    harmonic_stages = {
        stage: HarmonicStageQc(
            summary=harmonic_summaries[stage],
            spectrum=HarmonicSpectrum(
                frequencies_hz=frequencies,
                median_power_db=stage_spectra[stage],
            ),
        )
        for stage in ("raw", "gradient_corrected", "final")
    }
    return NativeCorrectionResult(
        raw=raw,
        qrs=qrs,
        volume_shifts_samples=np.array([0.0, 0.25]),
        marker_offsets_samples=np.array([0, 1]),
        complete_volume_count=20,
        discarded_terminal_samples=0,
        harmonic_stages=harmonic_stages,
        cardiac_qc=CardiacLockedComparison(
            before=before,
            after=after,
            rms_attenuation_db=20.0 * np.log10(4.0),
            peak_to_peak_attenuation_db=20.0 * np.log10(4.0),
        ),
    )


def test_run_qc_uses_separate_publication_scale_figures() -> None:
    physiology = qc_plotting.build_physiology_qc_figure(
        _native_result(),
        recording_label="sub-0001 run-1",
    )
    spectrum = qc_plotting.build_scanner_spectrum_qc_figure(
        _native_result(),
        recording_label="sub-0001 run-1",
    )

    assert len(physiology.axes) == 4
    assert [axis.get_title() for axis in physiology.axes[:3]] == [
        "Representative 20-second QRS detection window",
        "RR quality: 10 peaks, 2 warning intervals",
        "Cardiac-locked EEG: 12.0 dB RMS attenuation",
    ]
    assert len(spectrum.axes) == 5
    broad_axis, *local_axes = spectrum.axes
    assert broad_axis.get_xlim() == pytest.approx((15.0, 90.0))
    assert broad_axis.get_ylabel() == "PSD (dB V²/Hz)"
    assert [axis.get_title() for axis in local_axes] == [
        "20.0 Hz raw reference",
        "41.0 Hz raw reference",
        "61.0 Hz raw reference",
        "82.0 Hz raw reference",
    ]
    assert all(len(axis.lines) == 4 for axis in local_axes)


def test_cohort_scanner_spectrum_qc_matches_run_layout() -> None:
    result = _native_result()
    runs = tuple(
        extract_run_scanner_spectra(
            subject=f"sub-{subject:04d}",
            run=1,
            harmonic_stages=result.harmonic_stages,
        )
        for subject in (1, 2)
    )
    cohort = aggregate_cohort_scanner_spectra(
        runs,
        bootstrap_iterations=20,
        confidence_level=0.95,
        bootstrap_seed=42,
    )

    figure = qc_plotting.build_cohort_scanner_spectrum_qc_figure(cohort)

    assert len(figure.axes) == 5
    broad_axis, *local_axes = figure.axes
    assert broad_axis.get_xlim() == pytest.approx((15.0, 90.0))
    assert broad_axis.get_ylabel() == "PSD (dB V²/Hz)"
    assert len(broad_axis.collections) >= 3
    assert [axis.get_title() for axis in local_axes] == [
        "20.0 Hz raw reference",
        "41.0 Hz raw reference",
        "61.0 Hz raw reference",
        "82.0 Hz raw reference",
    ]
    assert all(len(axis.collections) >= 3 for axis in local_axes)
    assert "Participant-first median across 2 participants | 2 runs" in figure._suptitle.get_text()


def test_cohort_qc_prioritizes_residual_prominence_and_qrs_quality() -> None:
    row = {
        "harmonic_18_23_raw_to_final_db": 25.0,
        "harmonic_38_43_raw_to_final_db": 24.0,
        "harmonic_56_67_raw_to_final_db": 26.0,
        "harmonic_77_85_raw_to_final_db": 25.5,
        "harmonic_18_23_final_prominence_db": 6.0,
        "harmonic_38_43_final_prominence_db": 10.0,
        "harmonic_56_67_final_prominence_db": 14.0,
        "harmonic_77_85_final_prominence_db": 11.0,
        "cardiac_rms_attenuation_db": 12.0,
        "cardiac_peak_to_peak_attenuation_db": 11.0,
        "median_heart_rate_bpm": 70.0,
        "abnormal_rr_fraction": 0.02,
    }

    figure = qc_plotting.build_cohort_qc_figure([row, row])

    assert [axis.get_title() for axis in figure.axes] == [
        "Scanner-harmonic attenuation",
        "Residual scanner-line prominence",
        "Cardiac-locked EEG attenuation",
        "Automatic QRS quality",
    ]
    assert figure.axes[1].get_ylabel() == "Final local prominence (dB)"
    assert figure.axes[3].get_xlabel() == "Median heart rate (bpm)"
    assert figure.axes[3].get_ylabel() == "RR intervals outside limits (%)"


def test_read_input_recordings_requires_verified_manifest_inventory(tmp_path: Path) -> None:
    input_root = _input_root(tmp_path)

    recordings = runner.read_input_recordings(input_root, expected_count=1)

    assert len(recordings) == 1
    assert recordings[0].subject == "sub-0001"
    assert recordings[0].run == 1
    assert recordings[0].vhdr_path.name == "recording.vhdr"


def test_build_run_qc_serializes_complete_methods_and_cardiac_quality(tmp_path: Path) -> None:
    recording = runner.read_input_recordings(_input_root(tmp_path), expected_count=1)[0]
    qc = runner.build_run_qc(
        recording,
        _native_result(),
        parameters=runner.load_native_eeg_fmri_parameters(CONFIG_PATH),
        outputs=runner.RunOutputProvenance(
            fif=tmp_path / "output_raw.fif",
            fif_sha256="a" * 64,
            qrs=tmp_path / "qrs.tsv",
            qrs_sha256="b" * 64,
            physiology_qc=tmp_path / "physiology_qc.png",
            physiology_qc_sha256="c" * 64,
            scanner_spectrum_qc=tmp_path / "scanner_spectrum_qc.png",
            scanner_spectrum_qc_sha256="f" * 64,
            config_sha256="e" * 64,
        ),
    )

    assert qc["gradient"]["method"] == "synchronized_average_artifact_subtraction"
    cardiac = qc["cardiac"]
    assert cardiac["qrs_detector"]["method"] == "NeuXus_v0.0.4_bidirectional_LSTM"
    assert cardiac["qrs_detector"]["model_sha256"] == "d" * 64
    assert cardiac["pulse_correction"]["method"] == "MNE_PCA_OBS"
    assert cardiac["qrs_quality"]["warnings"] == ["two abnormal intervals"]
    assert cardiac["qrs_quality"]["abnormal_rr_count"] == 2
    assert cardiac["locked_eeg"]["rms_attenuation_db"] == 20.0 * np.log10(4.0)
    assert cardiac["locked_eeg"]["before"]["channel_rms"] == {"C3": 2.0}
    assert qc["outputs"]["fif"] == "sub-0001/eeg/output_raw.fif"
    assert qc["outputs"]["qrs_sha256"] == "b" * 64
    assert qc["outputs"]["physiology_qc_sha256"] == "c" * 64
    assert qc["outputs"]["scanner_spectrum_qc_sha256"] == "f" * 64


def test_process_recording_writes_atomic_qrs_and_separate_qc_figures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    recording = runner.read_input_recordings(_input_root(tmp_path), expected_count=1)[0]
    result = _native_result()
    detector = object()
    detector_arguments = []
    monkeypatch.setattr(
        runner.mne.io,
        "read_raw_brainvision",
        lambda *args, **kwargs: result.raw.copy(),
    )

    def fake_preprocess(raw, *, parameters, qrs_detector):
        detector_arguments.append(qrs_detector)
        return result

    monkeypatch.setattr(runner, "preprocess_raw_in_place", fake_preprocess)
    parameters = runner.load_native_eeg_fmri_parameters(CONFIG_PATH)

    row = runner.process_recording(
        recording,
        tmp_path / "derivative",
        parameters=parameters,
        config_sha256="e" * 64,
        qrs_detector=detector,
    )

    assert detector_arguments == [detector]
    qrs_path = Path(row["output_qrs"])
    physiology_path = Path(row["output_physiology_qc"])
    spectrum_path = Path(row["output_scanner_spectrum_qc"])
    qc = json.loads(Path(row["output_qc"]).read_text(encoding="utf-8"))
    assert qrs_path.read_text(encoding="utf-8").splitlines()[0] == (
        "onset_seconds\tdetector_sample\tprobability\twindow_support"
    )
    for path in (physiology_path, spectrum_path):
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        with Image.open(path) as image:
            assert image.width >= 3_600
            assert image.height >= 2_400
            assert image.info["dpi"] == pytest.approx((300.0, 300.0), abs=0.1)
    assert qc["outputs"]["physiology_qc_sha256"]
    assert qc["outputs"]["scanner_spectrum_qc_sha256"]
    assert qc["cardiac"]["qrs_detector"]["model_sha256"] == "d" * 64
    assert row["qrs_model_sha256"] == "d" * 64
    assert row["harmonic_18_23_final_prominence_db"] == 7.0


def test_run_cohort_publishes_organized_derivative_atomically(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_root = _input_root(tmp_path)
    output_root = tmp_path / "native-v2"
    detector = object()
    detector_ids = []

    monkeypatch.setattr(runner, "build_qrs_detector", lambda parameters: detector)

    def fake_process(recording, output_root, **kwargs):
        detector_ids.append(id(kwargs["qrs_detector"]))
        output_dir = output_root / recording.subject / "eeg"
        output_dir.mkdir(parents=True)
        output_fif = output_dir / "sub-0001_task-thermalactive_run-1_desc-mriartifactclean_raw.fif"
        output_qc = output_dir / "sub-0001_task-thermalactive_run-1_desc-mriartifactclean_qc.json"
        output_qrs = output_dir / "sub-0001_task-thermalactive_run-1_desc-mriartifactclean_qrs.tsv"
        output_physiology_qc = output_dir / (
            "sub-0001_task-thermalactive_run-1_desc-mriartifactclean_physiology_qc.png"
        )
        output_scanner_spectrum_qc = output_dir / (
            "sub-0001_task-thermalactive_run-1_desc-mriartifactclean_scanner_spectrum_qc.png"
        )
        output_fif.write_bytes(b"fif")
        output_qc.write_text("{}\n", encoding="utf-8")
        output_qrs.write_text("onset_seconds\n", encoding="utf-8")
        output_physiology_qc.write_bytes(b"png")
        output_scanner_spectrum_qc.write_bytes(b"png")
        return {
            "subject": recording.subject,
            "run": recording.run,
            "source_vhdr": str(recording.vhdr_path),
            "output_fif": str(output_fif),
            "output_qc": str(output_qc),
            "output_qrs": str(output_qrs),
            "output_physiology_qc": str(output_physiology_qc),
            "output_scanner_spectrum_qc": str(output_scanner_spectrum_qc),
            "output_fif_sha256": "c" * 64,
            "qrs_count": 10,
            "median_heart_rate_bpm": 60.0,
            "abnormal_rr_count": 0,
            "qrs_warning_count": 0,
            "qrs_model_sha256": "d" * 64,
            "cardiac_rms_attenuation_db": 12.0,
            "cardiac_peak_to_peak_attenuation_db": 11.0,
            "abnormal_rr_fraction": 0.0,
            "harmonic_18_23_raw_to_final_db": 25.0,
            "harmonic_38_43_raw_to_final_db": 24.0,
            "harmonic_56_67_raw_to_final_db": 26.0,
            "harmonic_77_85_raw_to_final_db": 25.5,
            "harmonic_18_23_final_prominence_db": 6.0,
            "harmonic_38_43_final_prominence_db": 10.0,
            "harmonic_56_67_final_prominence_db": 14.0,
            "harmonic_77_85_final_prominence_db": 11.0,
            "complete_volume_count": 20,
        }

    monkeypatch.setattr(runner, "process_recording", fake_process)

    published = runner.run_cohort(
        input_root,
        output_root,
        CONFIG_PATH,
        expected_count=1,
    )

    assert published == output_root
    assert (output_root / "dataset_description.json").is_file()
    cohort_figure = output_root / "cohort_mriartifact_qc.png"
    with Image.open(cohort_figure) as image:
        assert image.width >= 3_600
        assert image.info["dpi"] == pytest.approx((300.0, 300.0), abs=0.1)
    assert (output_root / "native_eeg_fmri_artifact_correction.yaml").is_file()
    manifest_text = (output_root / "native_correction_manifest.tsv").read_text(encoding="utf-8")
    assert str(output_root / "sub-0001" / "eeg") in manifest_text
    assert "qrs_model_sha256" in manifest_text
    assert "cardiac_rms_attenuation_db" in manifest_text
    description = json.loads(
        (output_root / "dataset_description.json").read_text(encoding="utf-8")
    )["GeneratedBy"][0]["Description"]
    assert "21-volume" in description
    assert "NeuXus" in description
    assert "MNE PCA-OBS" in description
    assert detector_ids == [id(detector)]
    assert not (tmp_path / ".native-v2.incomplete").exists()


def test_run_cohort_refuses_to_overwrite_a_derivative_root(tmp_path: Path) -> None:
    input_root = _input_root(tmp_path)
    output_root = tmp_path / "native-v1"
    output_root.mkdir()

    try:
        runner.run_cohort(input_root, output_root, CONFIG_PATH, expected_count=1)
    except FileExistsError as error:
        assert str(output_root) in str(error)
    else:
        raise AssertionError("run_cohort must refuse existing derivative roots")


def test_fixed_cohort_boundary_and_default_output_are_versioned() -> None:
    assert runner.EXPECTED_RUN_COUNT == 83
    assert runner.DEFAULT_OUTPUT_ROOT.name == "native_eeg_fmri_correction-v2"
