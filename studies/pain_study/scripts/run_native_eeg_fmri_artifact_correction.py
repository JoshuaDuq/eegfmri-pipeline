"""Run native scanner-gradient and pulse correction on the fixed 5 kHz cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from matplotlib.figure import Figure
import mne
import numpy as np
import numba
import scipy

from eeg_pipeline.preprocessing.eeg_fmri.config import (
    NativeEegFmriParameters,
    load_native_eeg_fmri_parameters,
)
from eeg_pipeline.preprocessing.eeg_fmri.neuxus_qrs import (
    NeuXusQrsDetector,
    NeuXusQrsPredictor,
    QrsDetector,
    load_packaged_neuxus_qrs_model,
)
from eeg_pipeline.preprocessing.eeg_fmri.pipeline import (
    NativeCorrectionResult,
    preprocess_raw_in_place,
)
from eeg_pipeline.preprocessing.eeg_fmri.qc import CardiacLockedSummary

EXPECTED_RUN_COUNT = 83
DEFAULT_INPUT_ROOT = Path(
    "/Volumes/KINGSTON/EEG_fMRI_data/derivatives/brainvision_marker_sanitized-v1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/Volumes/KINGSTON/EEG_fMRI_data/derivatives/native_eeg_fmri_correction-v2"
)
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config/native_eeg_fmri_artifact_correction.yaml"


@dataclass(frozen=True)
class InputRecording:
    """One marker-sanitized original recording and its immutable provenance."""

    subject: str
    run: int
    vhdr_path: Path
    source_vhdr_sha256: str
    source_vmrk_sha256: str
    source_eeg_size: int
    source_eeg_mtime_ns: int


@dataclass(frozen=True)
class RunOutputProvenance:
    """Published run artifacts and hashes needed by the QC report."""

    fif: Path
    fif_sha256: str
    qrs: Path
    qrs_sha256: str
    diagnostic: Path
    diagnostic_sha256: str
    config_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_manifest_fields(row: dict[str, str], line_number: int) -> None:
    required = {
        "subject",
        "run",
        "staged_vhdr",
        "source_vhdr_sha256",
        "source_vmrk_sha256",
        "source_eeg_size",
        "source_eeg_mtime_ns",
        "verified",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"Marker manifest line {line_number} is missing fields: {missing}")


def read_input_recordings(
    input_root: Path,
    *,
    expected_count: int = EXPECTED_RUN_COUNT,
) -> list[InputRecording]:
    """Read and validate the marker-sanitized cohort manifest."""
    manifest_path = input_root / "marker_sanitization_manifest.tsv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Marker-sanitization manifest does not exist: {manifest_path}")
    recordings: list[InputRecording] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, row in enumerate(csv.DictReader(stream, delimiter="\t"), start=2):
            _require_manifest_fields(row, line_number)
            if row["verified"] != "True":
                raise ValueError(f"Marker manifest line {line_number} is not verified")
            vhdr_path = Path(row["staged_vhdr"]).resolve()
            if not vhdr_path.is_relative_to(input_root.resolve()):
                raise ValueError(f"Staged header escapes the input root: {vhdr_path}")
            if not vhdr_path.is_file():
                raise FileNotFoundError(f"Staged BrainVision header does not exist: {vhdr_path}")
            subject = row["subject"]
            if not subject.startswith("sub-"):
                raise ValueError(f"Invalid subject label on manifest line {line_number}: {subject}")
            recordings.append(
                InputRecording(
                    subject=subject,
                    run=int(row["run"]),
                    vhdr_path=vhdr_path,
                    source_vhdr_sha256=row["source_vhdr_sha256"],
                    source_vmrk_sha256=row["source_vmrk_sha256"],
                    source_eeg_size=int(row["source_eeg_size"]),
                    source_eeg_mtime_ns=int(row["source_eeg_mtime_ns"]),
                )
            )
    if len(recordings) != expected_count:
        raise ValueError(f"Expected {expected_count} input recordings, found {len(recordings)}")
    subject_runs = [(recording.subject, recording.run) for recording in recordings]
    if len(set(subject_runs)) != len(subject_runs):
        raise ValueError("Input manifest contains duplicate subject/run recordings")
    return sorted(recordings, key=lambda recording: (recording.subject, recording.run))


def _output_stem(recording: InputRecording) -> str:
    return f"{recording.subject}_task-thermalactive_run-{recording.run}" "_desc-mriartifactclean"


def _harmonic_attenuation(
    stages: dict[str, dict[str, object]],
) -> dict[str, float]:
    raw_stage = stages["raw"]
    gradient_stage = stages["gradient_corrected"]
    final_stage = stages["final"]
    reference_power_keys = sorted(
        key
        for key in raw_stage
        if key.startswith("harmonic_") and key.endswith("_reference_power_db")
    )
    attenuation: dict[str, float] = {}
    for reference_power_key in reference_power_keys:
        prefix = reference_power_key.removesuffix("_reference_power_db")
        attenuation[f"{prefix}_raw_to_gradient_db"] = float(raw_stage[reference_power_key]) - float(
            gradient_stage[reference_power_key]
        )
        attenuation[f"{prefix}_raw_to_final_db"] = float(raw_stage[reference_power_key]) - float(
            final_stage[reference_power_key]
        )
    return attenuation


def build_qrs_detector(parameters: NativeEegFmriParameters) -> NeuXusQrsDetector:
    """Load and validate the pinned model once for a complete cohort run."""
    model = load_packaged_neuxus_qrs_model()
    return NeuXusQrsDetector(
        predictor=NeuXusQrsPredictor(model),
        parameters=parameters.cardiac.detection,
    )


def _serialize_cardiac_summary(summary: CardiacLockedSummary) -> dict[str, object]:
    return {
        "valid_epoch_count": summary.valid_epoch_count,
        "median_evoked_rms": summary.median_evoked_rms,
        "median_evoked_peak_to_peak": summary.median_evoked_peak_to_peak,
        "channel_rms": {
            channel: float(value)
            for channel, value in zip(
                summary.channel_names,
                summary.channel_rms,
                strict=True,
            )
        },
        "channel_peak_to_peak": {
            channel: float(value)
            for channel, value in zip(
                summary.channel_names,
                summary.channel_peak_to_peak,
                strict=True,
            )
        },
    }


def _serialize_qrs_quality(result: NativeCorrectionResult) -> dict[str, object]:
    quality = result.qrs.quality
    return {
        "qrs_count": quality.qrs_count,
        "median_heart_rate_bpm": quality.median_heart_rate_bpm,
        "minimum_rr_seconds": quality.minimum_rr_seconds,
        "median_rr_seconds": quality.median_rr_seconds,
        "maximum_rr_seconds": quality.maximum_rr_seconds,
        "abnormal_rr_count": quality.abnormal_rr_count,
        "abnormal_rr_fraction": quality.abnormal_rr_fraction,
        "temporal_coverage": quality.temporal_coverage,
        "warnings": list(quality.warnings),
        "correction_permitted": quality.correction_permitted,
    }


def build_run_qc(
    recording: InputRecording,
    result: NativeCorrectionResult,
    *,
    parameters: NativeEegFmriParameters,
    outputs: RunOutputProvenance,
) -> dict[str, object]:
    """Build one JSON-serializable correction and provenance report."""
    shifts = result.volume_shifts_samples
    marker_offsets = result.marker_offsets_samples
    return {
        "subject": recording.subject,
        "run": recording.run,
        "source_vhdr": str(recording.vhdr_path),
        "source_vhdr_sha256": recording.source_vhdr_sha256,
        "source_vmrk_sha256": recording.source_vmrk_sha256,
        "source_eeg_size": recording.source_eeg_size,
        "source_eeg_mtime_ns": recording.source_eeg_mtime_ns,
        "outputs": {
            "fif": str(Path(recording.subject) / "eeg" / outputs.fif.name),
            "fif_sha256": outputs.fif_sha256,
            "qrs": str(Path(recording.subject) / "eeg" / outputs.qrs.name),
            "qrs_sha256": outputs.qrs_sha256,
            "diagnostic": str(Path(recording.subject) / "eeg" / outputs.diagnostic.name),
            "diagnostic_sha256": outputs.diagnostic_sha256,
        },
        "config_sha256": outputs.config_sha256,
        "software": {
            "mne": mne.__version__,
            "numba": numba.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "output": {
            "sampling_frequency_hz": float(result.raw.info["sfreq"]),
            "n_samples": int(result.raw.n_times),
            "duration_seconds": float(result.raw.n_times / result.raw.info["sfreq"]),
            "channel_count": len(result.raw.ch_names),
        },
        "gradient": {
            "method": "synchronized_average_artifact_subtraction",
            "moving_average_volumes": parameters.gradient.moving_average_volumes,
            "alignment_upsampling": parameters.gradient.alignment_upsampling,
            "maximum_alignment_shift_samples": (
                parameters.gradient.maximum_alignment_shift_samples
            ),
            "complete_volume_count": result.complete_volume_count,
            "discarded_terminal_samples_at_5khz": result.discarded_terminal_samples,
            "marker_offset_nonzero_count": int(np.count_nonzero(marker_offsets)),
            "marker_offset_max_abs_samples": int(np.max(np.abs(marker_offsets))),
            "alignment_shift_median_samples": float(np.median(shifts)),
            "alignment_shift_p95_abs_samples": float(np.percentile(np.abs(shifts), 95)),
            "alignment_shift_max_abs_samples": float(np.max(np.abs(shifts))),
        },
        "cardiac": {
            "qrs_detector": {
                "method": "NeuXus_v0.0.4_bidirectional_LSTM",
                "model_sha256": result.qrs.diagnostics.model_sha256,
                "sampling_frequency_hz": result.qrs.diagnostics.sampling_frequency_hz,
                "low_frequency_hz": parameters.cardiac.detection.low_frequency_hz,
                "high_frequency_hz": parameters.cardiac.detection.high_frequency_hz,
                "window_stride_samples": parameters.cardiac.detection.window_stride_samples,
                "probability_threshold": parameters.cardiac.detection.probability_threshold,
                "minimum_support_samples": parameters.cardiac.detection.minimum_support_samples,
                "refractory_period_seconds": (
                    parameters.cardiac.detection.refractory_period_seconds
                ),
            },
            "qrs_quality": _serialize_qrs_quality(result),
            "pulse_correction": {
                "method": "MNE_PCA_OBS",
                "components": parameters.cardiac.obs_components,
            },
            "locked_eeg": {
                "window_seconds": [-0.2, 0.6],
                "baseline_seconds": [-0.2, -0.05],
                "before": _serialize_cardiac_summary(result.cardiac_qc.before),
                "after": _serialize_cardiac_summary(result.cardiac_qc.after),
                "rms_attenuation_db": result.cardiac_qc.rms_attenuation_db,
                "peak_to_peak_attenuation_db": (result.cardiac_qc.peak_to_peak_attenuation_db),
            },
        },
        "scanner_harmonics": {
            "stages": result.harmonic_stages,
            "attenuation": _harmonic_attenuation(result.harmonic_stages),
        },
    }


def _write_qrs_table(result: NativeCorrectionResult, path: Path) -> None:
    diagnostics = result.qrs.diagnostics
    peak_samples = diagnostics.peak_samples
    if np.any((peak_samples < 0) | (peak_samples >= diagnostics.probabilities.size)):
        raise ValueError("QRS detector samples fall outside the probability timeline")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            [
                "onset_seconds",
                "detector_sample",
                "probability",
                "window_support",
            ]
        )
        for onset, sample in zip(result.qrs.times, peak_samples, strict=True):
            writer.writerow(
                [
                    float(onset),
                    int(sample),
                    float(diagnostics.probabilities[sample]),
                    int(diagnostics.probability_support[sample]),
                ]
            )


def _write_diagnostic_figure(result: NativeCorrectionResult, path: Path) -> None:
    diagnostics = result.qrs.diagnostics
    detection_times = np.arange(diagnostics.filtered_ecg.size) / diagnostics.sampling_frequency_hz
    figure = Figure(figsize=(12, 9), layout="constrained")
    ecg_axis, probability_axis, cardiac_axis = figure.subplots(3, 1)
    ecg_axis.plot(detection_times, diagnostics.filtered_ecg, color="black", linewidth=0.6)
    ecg_axis.scatter(
        result.qrs.times,
        diagnostics.filtered_ecg[diagnostics.peak_samples],
        color="#c43b32",
        s=8,
        label="Accepted R peaks",
    )
    ecg_axis.set(ylabel="Filtered ECG", title="NeuXus QRS detection")
    ecg_axis.legend(loc="upper right")

    probability_axis.plot(
        detection_times,
        diagnostics.probabilities,
        color="#315b8a",
        linewidth=0.7,
    )
    probability_axis.set(xlabel="Time (s)", ylabel="R-peak probability")

    before = result.cardiac_qc.before
    after = result.cardiac_qc.after
    before_rms = np.sqrt(np.mean(before.median_evoked**2, axis=0))
    after_rms = np.sqrt(np.mean(after.median_evoked**2, axis=0))
    cardiac_axis.plot(before.times, before_rms, label="Before OBS", color="#c43b32")
    cardiac_axis.plot(after.times, after_rms, label="After OBS", color="#315b8a")
    cardiac_axis.set(
        xlabel="Time from R peak (s)",
        ylabel="Across-channel RMS",
        title="Cardiac-locked EEG",
    )
    cardiac_axis.legend(loc="upper right")
    figure.savefig(path, dpi=140)
    figure.clear()


def process_recording(
    recording: InputRecording,
    output_root: Path,
    *,
    parameters: NativeEegFmriParameters,
    config_sha256: str,
    qrs_detector: QrsDetector,
) -> dict[str, object]:
    """Correct, save, hash, and report one recording."""
    output_dir = output_root / recording.subject / "eeg"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _output_stem(recording)
    output_fif = output_dir / f"{stem}_raw.fif"
    output_qc = output_dir / f"{stem}_qc.json"
    output_qrs = output_dir / f"{stem}_qrs.tsv"
    output_diagnostic = output_dir / f"{stem}_diagnostic.png"
    outputs = (output_fif, output_qc, output_qrs, output_diagnostic)
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"Native correction output already exists for {recording.subject}")

    raw = mne.io.read_raw_brainvision(recording.vhdr_path, preload=True, verbose=False)
    result = preprocess_raw_in_place(
        raw,
        parameters=parameters,
        qrs_detector=qrs_detector,
    )
    temporary_fif = output_dir / f".{stem}_raw.fif"
    temporary_qc = output_dir / f".{stem}_qc.json"
    temporary_qrs = output_dir / f".{stem}_qrs.tsv"
    temporary_diagnostic = output_dir / f".{stem}_diagnostic.png"
    temporary_outputs = (
        temporary_fif,
        temporary_qc,
        temporary_qrs,
        temporary_diagnostic,
    )
    if any(path.exists() for path in temporary_outputs):
        raise FileExistsError(f"Temporary native correction output already exists: {stem}")
    try:
        result.raw.save(temporary_fif, fmt="single", overwrite=False, verbose=False)
        _write_qrs_table(result, temporary_qrs)
        _write_diagnostic_figure(result, temporary_diagnostic)
        output_fif_hash = _sha256(temporary_fif)
        output_qrs_hash = _sha256(temporary_qrs)
        output_diagnostic_hash = _sha256(temporary_diagnostic)
        qc = build_run_qc(
            recording,
            result,
            parameters=parameters,
            outputs=RunOutputProvenance(
                fif=output_fif,
                fif_sha256=output_fif_hash,
                qrs=output_qrs,
                qrs_sha256=output_qrs_hash,
                diagnostic=output_diagnostic,
                diagnostic_sha256=output_diagnostic_hash,
                config_sha256=config_sha256,
            ),
        )
        temporary_qc.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
        temporary_fif.replace(output_fif)
        temporary_qrs.replace(output_qrs)
        temporary_diagnostic.replace(output_diagnostic)
        temporary_qc.replace(output_qc)
    finally:
        for path in temporary_outputs:
            path.unlink(missing_ok=True)

    quality = result.qrs.quality
    return {
        "subject": recording.subject,
        "run": recording.run,
        "source_vhdr": str(recording.vhdr_path),
        "output_fif": str(output_fif),
        "output_qc": str(output_qc),
        "output_qrs": str(output_qrs),
        "output_diagnostic": str(output_diagnostic),
        "output_fif_sha256": output_fif_hash,
        "qrs_count": quality.qrs_count,
        "median_heart_rate_bpm": quality.median_heart_rate_bpm,
        "abnormal_rr_count": quality.abnormal_rr_count,
        "qrs_warning_count": len(quality.warnings),
        "qrs_model_sha256": result.qrs.diagnostics.model_sha256,
        "cardiac_rms_attenuation_db": result.cardiac_qc.rms_attenuation_db,
        "cardiac_peak_to_peak_attenuation_db": (result.cardiac_qc.peak_to_peak_attenuation_db),
        "complete_volume_count": result.complete_volume_count,
    }


def _write_manifest(output_root: Path, rows: list[dict[str, object]]) -> None:
    manifest_path = output_root / "native_correction_manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def run_cohort(
    input_root: Path,
    output_root: Path,
    config_path: Path,
    *,
    expected_count: int = EXPECTED_RUN_COUNT,
) -> Path:
    """Run the fixed cohort into an atomically published derivative root."""
    if output_root.exists():
        raise FileExistsError(f"Output root already exists: {output_root}")
    incomplete_root = output_root.parent / f".{output_root.name}.incomplete"
    if incomplete_root.exists():
        raise FileExistsError(f"Incomplete output root already exists: {incomplete_root}")

    parameters = load_native_eeg_fmri_parameters(config_path)
    recordings = read_input_recordings(input_root, expected_count=expected_count)
    qrs_detector = build_qrs_detector(parameters)
    incomplete_root.mkdir(parents=True)
    config_copy = incomplete_root / "native_eeg_fmri_artifact_correction.yaml"
    shutil.copy2(config_path, config_copy)
    config_sha256 = _sha256(config_copy)
    rows = []
    for index, recording in enumerate(recordings, start=1):
        print(
            f"[{index}/{len(recordings)}] Correcting {recording.subject} run-{recording.run}",
            flush=True,
        )
        row = process_recording(
            recording,
            incomplete_root,
            parameters=parameters,
            config_sha256=config_sha256,
            qrs_detector=qrs_detector,
        )
        rows.append(row)
        print(
            f"[{index}/{len(recordings)}] Completed {recording.subject} run-{recording.run}",
            flush=True,
        )
    published_rows = []
    for row in rows:
        published_row = dict(row)
        for key in ("output_fif", "output_qc", "output_qrs", "output_diagnostic"):
            staged_path = Path(str(row[key]))
            published_row[key] = str(output_root / staged_path.relative_to(incomplete_root))
        published_rows.append(published_row)
    _write_manifest(incomplete_root, published_rows)
    (incomplete_root / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": "Pain study native EEG-fMRI MRI-artifact correction",
                "BIDSVersion": "1.10.0",
                "DatasetType": "derivative",
                "GeneratedBy": [
                    {
                        "Name": "EEG_fMRI_Pipeline native correction",
                        "Version": "2",
                        "Description": (
                            "Synchronized 21-volume average artifact subtraction with sub-sample "
                            "alignment, NeuXus v0.0.4-derived LSTM QRS detection, and MNE PCA-OBS "
                            "pulse correction"
                        ),
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    incomplete_root.replace(output_root)
    return output_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace BrainVision Analyzer MRI-artifact preprocessing natively."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_root = run_cohort(args.input_root, args.output_root, args.config)
    print(f"Published native EEG-fMRI correction: {output_root}")


if __name__ == "__main__":
    main()
