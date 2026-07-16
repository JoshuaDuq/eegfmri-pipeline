"""Run native scanner-gradient and pulse correction on the fixed 5 kHz cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import numba
import scipy

from eeg_pipeline.preprocessing.eeg_fmri.config import (
    NativeEegFmriParameters,
    load_native_eeg_fmri_parameters,
)
from eeg_pipeline.preprocessing.eeg_fmri.cohort_spectrum import (
    RunScannerSpectra,
    aggregate_cohort_scanner_spectra,
    extract_run_scanner_spectra,
    write_cohort_scanner_spectra_tsv,
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
from eeg_pipeline.preprocessing.eeg_fmri.plotting import (
    save_cohort_qc_figure,
    save_cohort_scanner_spectrum_qc_figure,
    save_physiology_qc_figure,
    save_scanner_spectrum_qc_figure,
)
from eeg_pipeline.preprocessing.eeg_fmri.qc import (
    CardiacLockedSummary,
    HarmonicStageQc,
)
from eeg_pipeline.preprocessing.eeg_fmri.sequence import load_multiband_slice_schedule

EXPECTED_RUN_COUNT = 83
DEFAULT_INPUT_ROOT = Path(
    "/Volumes/KINGSTON/EEG_fMRI_data/derivatives/brainvision_marker_sanitized-v1"
)
DEFAULT_BOLD_ROOT = Path("/Volumes/KINGSTON/EEG_fMRI_data/bids_output/fmri")
DEFAULT_OUTPUT_ROOT = Path(
    "/Volumes/KINGSTON/EEG_fMRI_data/source_data/native_eeg_fmri_processed_1khz"
)
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config/native_eeg_fmri_artifact_correction.yaml"


@dataclass(frozen=True)
class InputRecording:
    """One marker-sanitized original recording and its immutable provenance."""

    subject: str
    run: int
    vhdr_path: Path
    bold_json_path: Path
    bold_json_sha256: str
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
    physiology_qc: Path
    physiology_qc_sha256: str
    scanner_spectrum_qc: Path
    scanner_spectrum_qc_sha256: str
    config_sha256: str


@dataclass(frozen=True)
class CompletedRun:
    """Manifest values and cropped spectra retained from one completed correction."""

    manifest_row: dict[str, object]
    scanner_spectra: RunScannerSpectra


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
    bold_root: Path,
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
            run = int(row["run"])
            bold_json_path = (
                bold_root
                / subject
                / "func"
                / f"{subject}_task-thermalactive_run-{run:02d}_bold.json"
            )
            if not bold_json_path.is_file():
                raise FileNotFoundError(f"Matching BOLD metadata does not exist: {bold_json_path}")
            recordings.append(
                InputRecording(
                    subject=subject,
                    run=run,
                    vhdr_path=vhdr_path,
                    bold_json_path=bold_json_path.resolve(),
                    bold_json_sha256=_sha256(bold_json_path),
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
    stages: dict[str, HarmonicStageQc],
) -> dict[str, float]:
    raw_stage = stages["raw"].summary
    gradient_stage = stages["gradient_corrected"].summary
    final_stage = stages["final"].summary
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


def _final_harmonic_prominence(
    stages: dict[str, HarmonicStageQc],
) -> dict[str, float]:
    final_stage = stages["final"].summary
    prominence_keys = sorted(
        key
        for key in final_stage
        if key.startswith("harmonic_") and key.endswith("_reference_local_prominence_db")
    )
    return {
        f"{key.removesuffix('_reference_local_prominence_db')}_final_prominence_db": float(
            final_stage[key]
        )
        for key in prominence_keys
    }


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
    group_shifts = result.group_shifts_samples
    marker_offsets = result.marker_offsets_samples
    slice_schedule = load_multiband_slice_schedule(recording.bold_json_path)
    gradient_method = "whole_volume_phase_aligned_adaptive_AAS"
    if parameters.gradient.residual_obs_components:
        gradient_method += "_cross_fitted_residual_OBS"
    return {
        "subject": recording.subject,
        "run": recording.run,
        "source_vhdr": str(recording.vhdr_path),
        "source_vhdr_sha256": recording.source_vhdr_sha256,
        "source_bold_json": str(recording.bold_json_path),
        "source_bold_json_sha256": recording.bold_json_sha256,
        "source_vmrk_sha256": recording.source_vmrk_sha256,
        "source_eeg_size": recording.source_eeg_size,
        "source_eeg_mtime_ns": recording.source_eeg_mtime_ns,
        "outputs": {
            "fif": str(Path(recording.subject) / "eeg" / outputs.fif.name),
            "fif_sha256": outputs.fif_sha256,
            "qrs": str(Path(recording.subject) / "eeg" / outputs.qrs.name),
            "qrs_sha256": outputs.qrs_sha256,
            "physiology_qc": str(Path(recording.subject) / "eeg" / outputs.physiology_qc.name),
            "physiology_qc_sha256": outputs.physiology_qc_sha256,
            "scanner_spectrum_qc": str(
                Path(recording.subject) / "eeg" / outputs.scanner_spectrum_qc.name
            ),
            "scanner_spectrum_qc_sha256": outputs.scanner_spectrum_qc_sha256,
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
            "method": gradient_method,
            "bold_json": str(recording.bold_json_path),
            "bold_json_sha256": recording.bold_json_sha256,
            "multiband_factor": slice_schedule.multiband_factor,
            "slice_count": slice_schedule.slice_count,
            "slice_group_count": slice_schedule.group_count,
            "slice_group_times_seconds": slice_schedule.group_times_seconds.tolist(),
            "moving_average_volumes": parameters.gradient.moving_average_volumes,
            "alignment_upsampling": parameters.gradient.alignment_upsampling,
            "maximum_alignment_shift_samples": (
                parameters.gradient.maximum_alignment_shift_samples
            ),
            "residual_obs_components": parameters.gradient.residual_obs_components,
            "residual_obs_folds": parameters.gradient.residual_obs_folds,
            "residual_obs_seed": parameters.gradient.residual_obs_seed,
            "residual_obs_removed_rms_v": result.residual_obs_removed_rms,
            "complete_volume_count": result.complete_volume_count,
            "discarded_terminal_samples_at_5khz": result.discarded_terminal_samples,
            "marker_offset_nonzero_count": int(np.count_nonzero(marker_offsets)),
            "marker_offset_max_abs_samples": int(np.max(np.abs(marker_offsets))),
            "alignment_shift_median_samples": float(np.median(shifts)),
            "alignment_shift_p95_abs_samples": float(np.percentile(np.abs(shifts), 95)),
            "alignment_shift_max_abs_samples": float(np.max(np.abs(shifts))),
            "group_alignment_shift_median_samples": float(np.median(group_shifts)),
            "group_alignment_shift_p95_abs_samples": float(np.percentile(np.abs(group_shifts), 95)),
            "group_alignment_shift_max_abs_samples": float(np.max(np.abs(group_shifts))),
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
            "stages": {
                stage: stage_qc.summary for stage, stage_qc in result.harmonic_stages.items()
            },
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


def process_recording(
    recording: InputRecording,
    output_root: Path,
    *,
    parameters: NativeEegFmriParameters,
    config_sha256: str,
    qrs_detector: QrsDetector,
) -> CompletedRun:
    """Correct, save, hash, and report one recording."""
    output_dir = output_root / recording.subject / "eeg"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _output_stem(recording)
    output_fif = output_dir / f"{stem}_raw.fif"
    output_qc = output_dir / f"{stem}_qc.json"
    output_qrs = output_dir / f"{stem}_qrs.tsv"
    output_physiology_qc = output_dir / f"{stem}_physiology_qc.png"
    output_scanner_spectrum_qc = output_dir / f"{stem}_scanner_spectrum_qc.png"
    outputs = (
        output_fif,
        output_qc,
        output_qrs,
        output_physiology_qc,
        output_scanner_spectrum_qc,
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"Native correction output already exists for {recording.subject}")

    raw = mne.io.read_raw_brainvision(recording.vhdr_path, preload=True, verbose=False)
    slice_schedule = load_multiband_slice_schedule(recording.bold_json_path)
    result = preprocess_raw_in_place(
        raw,
        parameters=parameters,
        qrs_detector=qrs_detector,
        slice_schedule=slice_schedule,
    )
    temporary_fif = output_dir / f".{stem}_raw.fif"
    temporary_qc = output_dir / f".{stem}_qc.json"
    temporary_qrs = output_dir / f".{stem}_qrs.tsv"
    temporary_physiology_qc = output_dir / f".{stem}_physiology_qc.png"
    temporary_scanner_spectrum_qc = output_dir / f".{stem}_scanner_spectrum_qc.png"
    temporary_outputs = (
        temporary_fif,
        temporary_qc,
        temporary_qrs,
        temporary_physiology_qc,
        temporary_scanner_spectrum_qc,
    )
    if any(path.exists() for path in temporary_outputs):
        raise FileExistsError(f"Temporary native correction output already exists: {stem}")
    try:
        result.raw.save(temporary_fif, fmt="single", overwrite=False, verbose=False)
        _write_qrs_table(result, temporary_qrs)
        save_physiology_qc_figure(
            result,
            temporary_physiology_qc,
            recording_label=f"{recording.subject} run-{recording.run}",
        )
        save_scanner_spectrum_qc_figure(
            result,
            temporary_scanner_spectrum_qc,
            recording_label=f"{recording.subject} run-{recording.run}",
        )
        output_fif_hash = _sha256(temporary_fif)
        output_qrs_hash = _sha256(temporary_qrs)
        output_physiology_qc_hash = _sha256(temporary_physiology_qc)
        output_scanner_spectrum_qc_hash = _sha256(temporary_scanner_spectrum_qc)
        qc = build_run_qc(
            recording,
            result,
            parameters=parameters,
            outputs=RunOutputProvenance(
                fif=output_fif,
                fif_sha256=output_fif_hash,
                qrs=output_qrs,
                qrs_sha256=output_qrs_hash,
                physiology_qc=output_physiology_qc,
                physiology_qc_sha256=output_physiology_qc_hash,
                scanner_spectrum_qc=output_scanner_spectrum_qc,
                scanner_spectrum_qc_sha256=output_scanner_spectrum_qc_hash,
                config_sha256=config_sha256,
            ),
        )
        temporary_qc.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
        temporary_fif.replace(output_fif)
        temporary_qrs.replace(output_qrs)
        temporary_physiology_qc.replace(output_physiology_qc)
        temporary_scanner_spectrum_qc.replace(output_scanner_spectrum_qc)
        temporary_qc.replace(output_qc)
    finally:
        for path in temporary_outputs:
            path.unlink(missing_ok=True)

    quality = result.qrs.quality
    row = {
        "subject": recording.subject,
        "run": recording.run,
        "source_vhdr": str(recording.vhdr_path),
        "source_bold_json": str(recording.bold_json_path),
        "source_bold_json_sha256": recording.bold_json_sha256,
        "output_fif": str(output_fif),
        "output_qc": str(output_qc),
        "output_qrs": str(output_qrs),
        "output_physiology_qc": str(output_physiology_qc),
        "output_scanner_spectrum_qc": str(output_scanner_spectrum_qc),
        "output_fif_sha256": output_fif_hash,
        "qrs_count": quality.qrs_count,
        "median_heart_rate_bpm": quality.median_heart_rate_bpm,
        "abnormal_rr_count": quality.abnormal_rr_count,
        "qrs_warning_count": len(quality.warnings),
        "qrs_model_sha256": result.qrs.diagnostics.model_sha256,
        "cardiac_rms_attenuation_db": result.cardiac_qc.rms_attenuation_db,
        "cardiac_peak_to_peak_attenuation_db": (result.cardiac_qc.peak_to_peak_attenuation_db),
        "abnormal_rr_fraction": quality.abnormal_rr_fraction,
        "complete_volume_count": result.complete_volume_count,
        "residual_obs_removed_rms_v": result.residual_obs_removed_rms,
    }
    attenuation = _harmonic_attenuation(result.harmonic_stages)
    row.update(
        {key: value for key, value in attenuation.items() if key.endswith("_raw_to_final_db")}
    )
    row.update(_final_harmonic_prominence(result.harmonic_stages))
    return CompletedRun(
        manifest_row=row,
        scanner_spectra=extract_run_scanner_spectra(
            subject=recording.subject,
            run=recording.run,
            harmonic_stages=result.harmonic_stages,
        ),
    )


def _write_manifest(output_root: Path, rows: list[dict[str, object]]) -> None:
    manifest_path = output_root / "native_correction_manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def run_cohort(
    input_root: Path,
    bold_root: Path,
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
    recordings = read_input_recordings(
        input_root,
        bold_root,
        expected_count=expected_count,
    )
    qrs_detector = build_qrs_detector(parameters)
    incomplete_root.mkdir(parents=True)
    config_copy = incomplete_root / "native_eeg_fmri_artifact_correction.yaml"
    shutil.copy2(config_path, config_copy)
    config_sha256 = _sha256(config_copy)
    completed_runs = []
    for index, recording in enumerate(recordings, start=1):
        print(
            f"[{index}/{len(recordings)}] Correcting {recording.subject} run-{recording.run}",
            flush=True,
        )
        completed = process_recording(
            recording,
            incomplete_root,
            parameters=parameters,
            config_sha256=config_sha256,
            qrs_detector=qrs_detector,
        )
        completed_runs.append(completed)
        print(
            f"[{index}/{len(recordings)}] Completed {recording.subject} run-{recording.run}",
            flush=True,
        )
    rows = [completed.manifest_row for completed in completed_runs]
    published_rows = []
    for row in rows:
        published_row = dict(row)
        for key in (
            "output_fif",
            "output_qc",
            "output_qrs",
            "output_physiology_qc",
            "output_scanner_spectrum_qc",
        ):
            staged_path = Path(str(row[key]))
            published_row[key] = str(output_root / staged_path.relative_to(incomplete_root))
        published_rows.append(published_row)
    _write_manifest(incomplete_root, published_rows)
    save_cohort_qc_figure(rows, incomplete_root / "cohort_mriartifact_qc.png")
    cohort_spectra = aggregate_cohort_scanner_spectra(
        [completed.scanner_spectra for completed in completed_runs],
        bootstrap_iterations=parameters.qc_bootstrap_iterations,
        confidence_level=parameters.qc_bootstrap_confidence_level,
        bootstrap_seed=parameters.qc_bootstrap_seed,
    )
    write_cohort_scanner_spectra_tsv(
        cohort_spectra,
        incomplete_root / "cohort_scanner_spectrum_qc.tsv",
    )
    save_cohort_scanner_spectrum_qc_figure(
        cohort_spectra,
        incomplete_root / "cohort_scanner_spectrum_qc.png",
    )
    (incomplete_root / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": "Pain study native EEG-fMRI MRI-artifact correction",
                "BIDSVersion": "1.10.0",
                "DatasetType": "derivative",
                "GeneratedBy": [
                    {
                        "Name": "EEG_fMRI_Pipeline native correction",
                        "Version": "4",
                        "Description": (
                            "Run-specific whole-volume adaptive average artifact subtraction "
                            "with sub-sample alignment and exact BIDS multiband-sequence "
                            "validation, followed by NeuXus v0.0.4-derived LSTM QRS detection "
                            "and MNE PCA-OBS pulse correction"
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
    parser.add_argument("--bold-root", type=Path, default=DEFAULT_BOLD_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_root = run_cohort(
        args.input_root,
        args.bold_root,
        args.output_root,
        args.config,
    )
    print(f"Published native EEG-fMRI correction: {output_root}")


if __name__ == "__main__":
    main()
