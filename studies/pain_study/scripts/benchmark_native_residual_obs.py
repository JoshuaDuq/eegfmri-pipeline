"""Qualify residual gradient OBS at the native post-AAS 5 kHz boundary."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import mne
import numpy as np
import pandas as pd
import scipy
import yaml
from scipy.signal import resample_poly

from eeg_pipeline.analysis.qc.native_residual_obs import (
    NativeObsSelectionThresholds,
    select_native_obs_component_count,
    summarize_fixed_frequency_lines,
)
from eeg_pipeline.analysis.qc.residual_gradient import (
    InjectionSettings,
    build_validation_injection,
    compute_outside_harmonic_psd_change,
    compute_volume_locked_rms,
    evaluate_injection_recovery,
)
from eeg_pipeline.preprocessing.eeg_fmri.config import (
    NativeEegFmriParameters,
    load_native_eeg_fmri_parameters,
)
from eeg_pipeline.preprocessing.eeg_fmri.gradient import (
    FittedResidualObs,
    apply_fitted_residual_obs,
    correct_gradient_average,
    fit_cross_fitted_residual_obs,
    resolve_volume_segments,
)
from eeg_pipeline.preprocessing.eeg_fmri.mne_io import (
    extract_volume_samples,
    validate_acquisition,
)
from eeg_pipeline.preprocessing.eeg_fmri.sequence import (
    MultibandSliceSchedule,
    load_multiband_slice_schedule,
)
from studies.pain_study.scripts.run_native_eeg_fmri_artifact_correction import (
    DEFAULT_BOLD_ROOT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_INPUT_ROOT,
    InputRecording,
    read_input_recordings,
)

DEFAULT_BENCHMARK_CONFIG_PATH = Path(__file__).parent / "config/native_residual_obs_benchmark.yaml"
DEFAULT_OUTPUT_ROOT = Path(
    "/Volumes/KINGSTON/EEG_fMRI_data/derivatives/qc/native_residual_obs_benchmark"
)


class BasisScope(str, Enum):
    """Temporal support used to constrain residual OBS bases."""

    SLICE_GROUP = "slice_group"
    WHOLE_VOLUME = "whole_volume"


@dataclass(frozen=True)
class WelchSettings:
    duration_seconds: float
    overlap_fraction: float
    background_inner_hz: float
    background_outer_hz: float
    line_exclusion_half_width_hz: float

    def __post_init__(self) -> None:
        positive = (
            self.duration_seconds,
            self.background_inner_hz,
            self.background_outer_hz,
            self.line_exclusion_half_width_hz,
        )
        if not all(np.isfinite(value) and value > 0 for value in positive):
            raise ValueError("Welch values must be finite and positive")
        if not 0 <= self.overlap_fraction < 1:
            raise ValueError("welch.overlap_fraction must be in [0, 1)")
        if self.background_inner_hz >= self.background_outer_hz:
            raise ValueError("Welch background inner width must be below outer width")


@dataclass(frozen=True)
class NativeObsBenchmarkConfig:
    run_indices: tuple[int, ...]
    minimum_participants: int
    component_counts: tuple[int, ...]
    residual_line_frequencies_hz: tuple[float, ...]
    welch: WelchSettings
    injection: InjectionSettings
    injection_evaluation_sampling_frequency_hz: float
    acceptance: NativeObsSelectionThresholds

    def __post_init__(self) -> None:
        if not self.run_indices or any(run < 1 for run in self.run_indices):
            raise ValueError("run_indices must contain positive run numbers")
        if len(set(self.run_indices)) != len(self.run_indices):
            raise ValueError("run_indices must be unique")
        if self.minimum_participants < 3:
            raise ValueError("minimum_participants must be at least 3")
        if self.component_counts != (0, 1, 2, 3, 4):
            raise ValueError("component_counts must be exactly [0, 1, 2, 3, 4]")
        frequencies = np.asarray(self.residual_line_frequencies_hz)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("residual_line_frequencies_hz must be non-empty")
        if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
            raise ValueError("Residual-line frequencies must be finite and positive")
        if np.unique(frequencies).size != frequencies.size:
            raise ValueError("Residual-line frequencies must be unique")
        evaluation_frequency = self.injection_evaluation_sampling_frequency_hz
        if not np.isfinite(evaluation_frequency) or evaluation_frequency <= 0:
            raise ValueError("Injection evaluation sampling frequency must be positive")
        if max(self.injection.frequencies_hz) >= evaluation_frequency / 2:
            raise ValueError("Injection frequencies must be below evaluation Nyquist")


@dataclass(frozen=True)
class ReportPaths:
    line_audit: Path
    run_audit: Path
    preservation_audit: Path
    cohort_summary: Path
    provenance: Path
    decision: Path

    def all_paths(self) -> tuple[Path, ...]:
        return (
            self.line_audit,
            self.run_audit,
            self.preservation_audit,
            self.cohort_summary,
            self.provenance,
            self.decision,
        )


@dataclass(frozen=True)
class _PostAasRecording:
    data_v: np.ndarray
    volume_segments: tuple[np.ndarray, ...]
    group_boundaries: np.ndarray
    volume_shift_median_samples: float
    volume_shift_p95_absolute_samples: float


def _require_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _require_exact_keys(
    mapping: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    missing = sorted(expected - set(mapping))
    unexpected = sorted(set(mapping) - expected)
    if missing or unexpected:
        raise ValueError(f"Invalid {context} keys; missing={missing}, unexpected={unexpected}")


def _number_tuple(value: object, context: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    return tuple(float(item) for item in value)


def _integer_tuple(value: object, context: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise TypeError(f"{context} must be a list of integers")
    return tuple(value)


def load_benchmark_config(path: str | Path) -> NativeObsBenchmarkConfig:
    """Load the exact native residual-OBS benchmark schema."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Benchmark configuration does not exist: {config_path}")
    payload = _require_mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
        "benchmark configuration",
    )
    _require_exact_keys(
        payload,
        {
            "version",
            "run_indices",
            "minimum_participants",
            "component_counts",
            "residual_line_frequencies_hz",
            "welch",
            "injection",
            "acceptance",
        },
        "benchmark configuration",
    )
    if payload["version"] != 1:
        raise ValueError(f"Unsupported benchmark config version: {payload['version']!r}")

    welch = _require_mapping(payload["welch"], "welch")
    _require_exact_keys(
        welch,
        {
            "duration_seconds",
            "overlap_fraction",
            "background_inner_hz",
            "background_outer_hz",
            "line_exclusion_half_width_hz",
        },
        "welch",
    )
    injection = _require_mapping(payload["injection"], "injection")
    _require_exact_keys(
        injection,
        {
            "frequencies_hz",
            "sinusoid_amplitude_uv",
            "transient_amplitude_uv",
            "transient_spacing_s",
            "transient_width_s",
            "evaluation_sampling_frequency_hz",
        },
        "injection",
    )
    acceptance = _require_mapping(payload["acceptance"], "acceptance")
    acceptance_fields = set(NativeObsSelectionThresholds.__dataclass_fields__)
    _require_exact_keys(acceptance, acceptance_fields, "acceptance")

    return NativeObsBenchmarkConfig(
        run_indices=_integer_tuple(payload["run_indices"], "run_indices"),
        minimum_participants=int(payload["minimum_participants"]),
        component_counts=_integer_tuple(
            payload["component_counts"],
            "component_counts",
        ),
        residual_line_frequencies_hz=_number_tuple(
            payload["residual_line_frequencies_hz"],
            "residual_line_frequencies_hz",
        ),
        welch=WelchSettings(**{key: float(value) for key, value in welch.items()}),
        injection=InjectionSettings(
            frequencies_hz=_number_tuple(
                injection["frequencies_hz"],
                "injection.frequencies_hz",
            ),
            sinusoid_amplitude_uv=float(injection["sinusoid_amplitude_uv"]),
            transient_amplitude_uv=float(injection["transient_amplitude_uv"]),
            transient_spacing_s=float(injection["transient_spacing_s"]),
            transient_width_s=float(injection["transient_width_s"]),
        ),
        injection_evaluation_sampling_frequency_hz=float(
            injection["evaluation_sampling_frequency_hz"]
        ),
        acceptance=NativeObsSelectionThresholds(
            **{key: float(value) for key, value in acceptance.items()}
        ),
    )


def select_benchmark_recordings(
    recordings: Sequence[InputRecording],
    *,
    run_indices: Sequence[int],
    minimum_participants: int,
) -> list[InputRecording]:
    """Select configured runs from every discovered participant."""
    selected_runs = tuple(int(run) for run in run_indices)
    if not selected_runs or len(set(selected_runs)) != len(selected_runs):
        raise ValueError("run_indices must be non-empty and unique")
    available: dict[str, set[int]] = {}
    for recording in recordings:
        available.setdefault(recording.subject, set()).add(recording.run)
    missing = sorted(
        subject for subject, runs in available.items() if not set(selected_runs).issubset(runs)
    )
    if missing:
        raise ValueError(
            "Participants are missing configured benchmark runs: " + ", ".join(missing)
        )
    if len(available) < minimum_participants:
        raise ValueError(
            f"Benchmark requires at least {minimum_participants} participants, "
            f"found {len(available)}"
        )
    selected = [recording for recording in recordings if recording.run in selected_runs]
    return sorted(selected, key=lambda recording: (recording.subject, recording.run))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _crop_incomplete_terminal_volume(raw: mne.io.BaseRaw, stop_sample: int) -> None:
    sampling_frequency = float(raw.info["sfreq"])
    annotation_samples = np.rint(raw.annotations.onset * sampling_frequency).astype(int)
    forbidden = sorted(
        {
            description
            for sample, description in zip(
                annotation_samples,
                raw.annotations.description,
                strict=True,
            )
            if sample >= stop_sample
            and description != "Volume/V  1"
            and not description.startswith("SyncStatus/")
        }
    )
    if forbidden:
        raise ValueError(
            "Cannot crop terminal scanner interval containing task annotations: "
            + ", ".join(forbidden)
        )
    raw.crop(
        tmin=0.0,
        tmax=(stop_sample - 1) / sampling_frequency,
        include_tmax=True,
        verbose=False,
    )


def _prepare_post_aas_recording(
    recording: InputRecording,
    *,
    native_parameters: NativeEegFmriParameters,
    slice_schedule: MultibandSliceSchedule,
) -> _PostAasRecording:
    raw = mne.io.read_raw_brainvision(recording.vhdr_path, preload=True, verbose=False)
    validate_acquisition(
        raw,
        expected_sampling_frequency=native_parameters.acquisition_sampling_frequency_hz,
        expected_channel_count=native_parameters.expected_channel_count,
        ecg_channel=native_parameters.ecg_channel,
    )
    volume_samples = extract_volume_samples(
        raw,
        annotation_description=native_parameters.volume_annotation,
    )
    boundaries = resolve_volume_segments(
        volume_samples,
        n_samples=raw.n_times,
        sampling_frequency=float(raw.info["sfreq"]),
        repetition_time_seconds=native_parameters.repetition_time_seconds,
        maximum_marker_deviation_samples=(native_parameters.maximum_marker_deviation_samples),
    )
    crop_boundaries = [boundary for boundary in boundaries if boundary.crop_stop_sample]
    if len(crop_boundaries) > 1 or (crop_boundaries and crop_boundaries[0] is not boundaries[-1]):
        raise RuntimeError("Only the final scanner block may require terminal cropping")
    if crop_boundaries:
        _crop_incomplete_terminal_volume(raw, crop_boundaries[0].crop_stop_sample)

    eeg_picks = mne.pick_types(raw.info, eeg=True, ecg=False, exclude=[])
    corrected_data = raw._data
    volume_shifts = []
    group_boundaries = None
    for boundary in boundaries:
        average = correct_gradient_average(
            corrected_data,
            boundary.complete_volume_samples,
            sampling_frequency=float(raw.info["sfreq"]),
            alignment_picks=eeg_picks,
            slice_schedule=slice_schedule,
            parameters=native_parameters.gradient,
        )
        corrected_data = average.data
        volume_shifts.append(average.volume_shifts_samples)
        if group_boundaries is None:
            group_boundaries = average.group_boundaries_samples
        elif not np.array_equal(group_boundaries, average.group_boundaries_samples):
            raise RuntimeError("BOLD slice-group boundaries changed between scanner blocks")
    if group_boundaries is None:
        raise RuntimeError("No scanner blocks were corrected")

    missing_channels = sorted(set(native_parameters.qc_channels) - set(raw.ch_names))
    if missing_channels:
        raise ValueError(f"Scanner-QC channels are missing: {missing_channels}")
    qc_picks = [raw.ch_names.index(channel) for channel in native_parameters.qc_channels]
    post_aas = corrected_data[qc_picks].copy()
    shifts = np.concatenate(volume_shifts)
    return _PostAasRecording(
        data_v=post_aas,
        volume_segments=tuple(boundary.complete_volume_samples for boundary in boundaries),
        group_boundaries=np.asarray(group_boundaries, dtype=int),
        volume_shift_median_samples=float(np.median(shifts)),
        volume_shift_p95_absolute_samples=float(np.percentile(np.abs(shifts), 95)),
    )


def _fit_models(
    data_v: np.ndarray,
    *,
    volume_segments: Sequence[np.ndarray],
    group_boundaries: np.ndarray,
    maximum_components: int,
    n_folds: int,
    seed: int,
) -> tuple[FittedResidualObs, ...]:
    picks = np.arange(data_v.shape[0], dtype=int)
    return tuple(
        fit_cross_fitted_residual_obs(
            data_v,
            volume_samples,
            group_boundaries=group_boundaries,
            picks=picks,
            maximum_components=maximum_components,
            n_folds=n_folds,
            seed=seed,
        )
        for volume_samples in volume_segments
    )


def _apply_models(
    data_v: np.ndarray,
    *,
    models: Sequence[FittedResidualObs],
    n_components: int,
) -> tuple[np.ndarray, float]:
    corrected = data_v
    removed_rms = []
    weights = []
    for model in models:
        corrected, segment_removed_rms = apply_fitted_residual_obs(
            corrected,
            model=model,
            n_components=n_components,
        )
        removed_rms.append(segment_removed_rms)
        weights.append(model.volume_samples.size)
    combined_rms = float(np.sqrt(np.average(np.square(removed_rms), weights=weights)))
    return corrected, combined_rms


def _line_rows(
    data_v: np.ndarray,
    *,
    recording: InputRecording,
    n_components: int,
    sampling_frequency: float,
    config: NativeObsBenchmarkConfig,
) -> list[dict[str, Any]]:
    rows = summarize_fixed_frequency_lines(
        data_v,
        sampling_frequency=sampling_frequency,
        reference_frequencies_hz=config.residual_line_frequencies_hz,
        welch_duration_seconds=config.welch.duration_seconds,
        overlap_fraction=config.welch.overlap_fraction,
        background_inner_hz=config.welch.background_inner_hz,
        background_outer_hz=config.welch.background_outer_hz,
    )
    recording_id = f"{recording.subject}_run-{recording.run}"
    return [
        {
            "recording_id": recording_id,
            "subject": recording.subject,
            "run": recording.run,
            "n_components": n_components,
            **row,
        }
        for row in rows
    ]


def _preservation_row(
    *,
    baseline_v: np.ndarray,
    candidate_v: np.ndarray,
    injected_candidate_v: np.ndarray,
    recording: InputRecording,
    n_components: int,
    sampling_frequency: float,
    config: NativeObsBenchmarkConfig,
) -> dict[str, Any]:
    evaluation_frequency = config.injection_evaluation_sampling_frequency_hz
    decimation = sampling_frequency / evaluation_frequency
    rounded_decimation = round(decimation)
    if not np.isclose(decimation, rounded_decimation, rtol=0.0, atol=1e-12):
        raise ValueError("Injection evaluation rate must divide acquisition rate exactly")
    recovered = np.stack(
        [
            resample_poly(
                injected_candidate_v[channel] - candidate_v[channel],
                up=1,
                down=int(rounded_decimation),
            )
            for channel in range(candidate_v.shape[0])
        ]
    )
    expected = build_validation_injection(
        recovered.shape[1],
        evaluation_frequency,
        config.injection,
    )
    recovery = [
        evaluate_injection_recovery(
            expected=expected,
            recovered_v=channel,
            sfreq=evaluation_frequency,
        )
        for channel in recovered
    ]
    segment_samples = int(round(config.welch.duration_seconds * sampling_frequency))
    exclusion_windows = tuple(
        (
            frequency - config.welch.line_exclusion_half_width_hz,
            frequency + config.welch.line_exclusion_half_width_hz,
        )
        for frequency in config.residual_line_frequencies_hz
    )
    psd_change = compute_outside_harmonic_psd_change(
        baseline_v,
        candidate_v,
        sfreq=sampling_frequency,
        nperseg=segment_samples,
        overlap_fraction=config.welch.overlap_fraction,
        harmonic_windows_hz=exclusion_windows,
    )
    return {
        "recording_id": f"{recording.subject}_run-{recording.run}",
        "subject": recording.subject,
        "run": recording.run,
        "n_components": n_components,
        "minimum_sinusoid_amplitude_ratio": min(
            metrics.minimum_sinusoid_amplitude_ratio for metrics in recovery
        ),
        "maximum_phase_error_deg": max(metrics.maximum_phase_error_deg for metrics in recovery),
        "transient_peak_ratio": min(metrics.transient_peak_ratio for metrics in recovery),
        "outside_line_psd_change_db": psd_change.median_change_db,
        "maximum_channel_outside_line_psd_change_db": (
            psd_change.maximum_channel_absolute_change_db
        ),
    }


def _benchmark_recording(
    recording: InputRecording,
    *,
    native_parameters: NativeEegFmriParameters,
    config: NativeObsBenchmarkConfig,
    basis_scope: BasisScope,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sampling_frequency = native_parameters.acquisition_sampling_frequency_hz
    schedule = load_multiband_slice_schedule(recording.bold_json_path)
    post_aas = _prepare_post_aas_recording(
        recording,
        native_parameters=native_parameters,
        slice_schedule=schedule,
    )
    baseline = post_aas.data_v
    volume_samples = np.concatenate(post_aas.volume_segments)
    volume_samples_per_epoch = int(
        round(native_parameters.repetition_time_seconds * sampling_frequency)
    )
    if basis_scope is BasisScope.SLICE_GROUP:
        obs_boundaries = post_aas.group_boundaries
    else:
        obs_boundaries = np.array([0, volume_samples_per_epoch], dtype=int)
    injection = build_validation_injection(baseline.shape[1], sampling_frequency, config.injection)
    injected_baseline = baseline + injection.signal_v[np.newaxis, :]
    maximum_components = max(config.component_counts)
    baseline_models = _fit_models(
        baseline,
        volume_segments=post_aas.volume_segments,
        group_boundaries=obs_boundaries,
        maximum_components=maximum_components,
        n_folds=native_parameters.gradient.residual_obs_folds,
        seed=native_parameters.gradient.residual_obs_seed,
    )
    injected_models = _fit_models(
        injected_baseline,
        volume_segments=post_aas.volume_segments,
        group_boundaries=obs_boundaries,
        maximum_components=maximum_components,
        n_folds=native_parameters.gradient.residual_obs_folds,
        seed=native_parameters.gradient.residual_obs_seed,
    )

    line_rows = _line_rows(
        baseline,
        recording=recording,
        n_components=0,
        sampling_frequency=sampling_frequency,
        config=config,
    )
    run_rows = [
        {
            "recording_id": f"{recording.subject}_run-{recording.run}",
            "subject": recording.subject,
            "run": recording.run,
            "n_components": 0,
            "volume_locked_rms_v": compute_volume_locked_rms(
                baseline,
                volume_samples,
                volume_samples_per_epoch,
            ),
            "residual_obs_removed_rms_v": 0.0,
            "complete_volume_count": volume_samples.size,
            "scanner_block_count": len(post_aas.volume_segments),
            "volume_shift_median_samples": post_aas.volume_shift_median_samples,
            "volume_shift_p95_absolute_samples": (post_aas.volume_shift_p95_absolute_samples),
        }
    ]
    preservation_rows = []
    for n_components in config.component_counts[1:]:
        candidate, removed_rms = _apply_models(
            baseline,
            models=baseline_models,
            n_components=n_components,
        )
        injected_candidate, _ = _apply_models(
            injected_baseline,
            models=injected_models,
            n_components=n_components,
        )
        line_rows.extend(
            _line_rows(
                candidate,
                recording=recording,
                n_components=n_components,
                sampling_frequency=sampling_frequency,
                config=config,
            )
        )
        run_rows.append(
            {
                **run_rows[0],
                "n_components": n_components,
                "volume_locked_rms_v": compute_volume_locked_rms(
                    candidate,
                    volume_samples,
                    volume_samples_per_epoch,
                ),
                "residual_obs_removed_rms_v": removed_rms,
            }
        )
        preservation_rows.append(
            _preservation_row(
                baseline_v=baseline,
                candidate_v=candidate,
                injected_candidate_v=injected_candidate,
                recording=recording,
                n_components=n_components,
                sampling_frequency=sampling_frequency,
                config=config,
            )
        )
        del candidate, injected_candidate
        gc.collect()
    return line_rows, run_rows, preservation_rows


def build_cohort_summary(
    line_rows: Sequence[dict[str, Any]],
    *,
    component_counts: Sequence[int],
) -> list[dict[str, Any]]:
    """Aggregate paired line changes across benchmark recordings."""
    frame = pd.DataFrame(line_rows)
    references = frame.loc[frame["n_components"] == 0].set_index(
        ["recording_id", "reference_frequency_hz"]
    )
    rows = []
    for components in component_counts:
        if components == 0:
            continue
        candidates = frame.loc[frame["n_components"] == components].set_index(
            ["recording_id", "reference_frequency_hz"]
        )
        if not candidates.index.equals(references.index):
            raise ValueError(f"Cohort line index mismatch for components={components}")
        paired = references[["power_db", "local_prominence_db"]].join(
            candidates[["power_db", "local_prominence_db"]],
            lsuffix="_reference",
            rsuffix="_candidate",
        )
        for frequency, frequency_rows in paired.groupby(
            level="reference_frequency_hz",
            sort=True,
        ):
            power_reduction = (
                frequency_rows["power_db_reference"] - frequency_rows["power_db_candidate"]
            )
            prominence_reduction = (
                frequency_rows["local_prominence_db_reference"]
                - frequency_rows["local_prominence_db_candidate"]
            )
            rows.append(
                {
                    "n_components": components,
                    "reference_frequency_hz": float(frequency),
                    "recording_count": len(frequency_rows),
                    "median_reference_power_db": float(
                        frequency_rows["power_db_reference"].median()
                    ),
                    "median_candidate_power_db": float(
                        frequency_rows["power_db_candidate"].median()
                    ),
                    "median_power_reduction_db": float(power_reduction.median()),
                    "median_reference_prominence_db": float(
                        frequency_rows["local_prominence_db_reference"].median()
                    ),
                    "median_candidate_prominence_db": float(
                        frequency_rows["local_prominence_db_candidate"].median()
                    ),
                    "median_prominence_reduction_db": float(prominence_reduction.median()),
                    "maximum_recording_prominence_increase_db": float(
                        (-prominence_reduction).max()
                    ),
                }
            )
    return rows


def write_benchmark_reports(
    *,
    output_root: Path,
    line_rows: Sequence[dict[str, Any]],
    run_rows: Sequence[dict[str, Any]],
    preservation_rows: Sequence[dict[str, Any]],
    cohort_rows: Sequence[dict[str, Any]],
    provenance: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> ReportPaths:
    """Publish all benchmark audit artifacts into a new directory."""
    if output_root.exists():
        raise FileExistsError(f"Benchmark output root already exists: {output_root}")
    json.dumps(provenance)
    json.dumps(decision)
    output_root.mkdir(parents=True)
    paths = ReportPaths(
        line_audit=output_root / "native_residual_obs_line_audit.tsv",
        run_audit=output_root / "native_residual_obs_run_audit.tsv",
        preservation_audit=output_root / "native_residual_obs_preservation.tsv",
        cohort_summary=output_root / "native_residual_obs_cohort_summary.tsv",
        provenance=output_root / "native_residual_obs_provenance.json",
        decision=output_root / "native_residual_obs_decision.json",
    )
    pd.DataFrame(line_rows).to_csv(paths.line_audit, sep="\t", index=False)
    pd.DataFrame(run_rows).to_csv(paths.run_audit, sep="\t", index=False)
    pd.DataFrame(preservation_rows).to_csv(
        paths.preservation_audit,
        sep="\t",
        index=False,
    )
    pd.DataFrame(cohort_rows).to_csv(paths.cohort_summary, sep="\t", index=False)
    paths.provenance.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    paths.decision.write_text(
        json.dumps(decision, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths


def run_benchmark(
    *,
    input_root: Path,
    bold_root: Path,
    output_root: Path,
    native_config_path: Path,
    benchmark_config_path: Path,
    basis_scope: BasisScope = BasisScope.SLICE_GROUP,
) -> ReportPaths:
    """Run the multi-participant native residual-OBS qualification."""
    if not isinstance(basis_scope, BasisScope):
        raise TypeError("basis_scope must be a BasisScope")
    if output_root.exists():
        raise FileExistsError(f"Benchmark output root already exists: {output_root}")
    native_parameters = load_native_eeg_fmri_parameters(native_config_path)
    config = load_benchmark_config(benchmark_config_path)
    recordings = select_benchmark_recordings(
        read_input_recordings(input_root, bold_root),
        run_indices=config.run_indices,
        minimum_participants=config.minimum_participants,
    )
    line_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    preservation_rows: list[dict[str, Any]] = []
    for index, recording in enumerate(recordings, start=1):
        print(
            f"[{index}/{len(recordings)}] Benchmarking {recording.subject} " f"run-{recording.run}",
            flush=True,
        )
        recording_lines, recording_runs, recording_preservation = _benchmark_recording(
            recording,
            native_parameters=native_parameters,
            config=config,
            basis_scope=basis_scope,
        )
        line_rows.extend(recording_lines)
        run_rows.extend(recording_runs)
        preservation_rows.extend(recording_preservation)
        gc.collect()

    decision = select_native_obs_component_count(
        line_rows=line_rows,
        run_rows=run_rows,
        preservation_rows=preservation_rows,
        component_counts=config.component_counts,
        thresholds=config.acceptance,
    )
    cohort_rows = build_cohort_summary(
        line_rows,
        component_counts=config.component_counts,
    )
    provenance = {
        "method": ("native_post_AAS_cross_fitted_" f"{basis_scope.value}_locked_residual_OBS"),
        "basis_scope": basis_scope.value,
        "stage": "5_kHz_after_native_gradient_AAS_before_resampling_and_cardiac_OBS",
        "input_root": str(input_root.resolve()),
        "bold_root": str(bold_root.resolve()),
        "native_config": str(native_config_path.resolve()),
        "native_config_sha256": _sha256(native_config_path),
        "benchmark_config": str(benchmark_config_path.resolve()),
        "benchmark_config_sha256": _sha256(benchmark_config_path),
        "qc_channels": list(native_parameters.qc_channels),
        "residual_obs_folds": native_parameters.gradient.residual_obs_folds,
        "residual_obs_seed": native_parameters.gradient.residual_obs_seed,
        "software": {
            "mne": mne.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "recordings": [
            {
                "recording_id": f"{recording.subject}_run-{recording.run}",
                "source_vhdr": str(recording.vhdr_path),
                "source_vhdr_sha256": recording.source_vhdr_sha256,
                "source_vmrk_sha256": recording.source_vmrk_sha256,
                "bold_json": str(recording.bold_json_path),
                "bold_json_sha256": recording.bold_json_sha256,
            }
            for recording in recordings
        ],
    }
    decision_payload = {
        **decision.to_dict(),
        "thresholds": asdict(config.acceptance),
        "recording_count": len(recordings),
        "participant_count": len({recording.subject for recording in recordings}),
    }
    return write_benchmark_reports(
        output_root=output_root,
        line_rows=line_rows,
        run_rows=run_rows,
        preservation_rows=preservation_rows,
        cohort_rows=cohort_rows,
        provenance=provenance,
        decision=decision_payload,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--bold-root", type=Path, default=DEFAULT_BOLD_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--native-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=DEFAULT_BENCHMARK_CONFIG_PATH,
    )
    parser.add_argument(
        "--basis-scope",
        type=BasisScope,
        choices=tuple(BasisScope),
        default=BasisScope.SLICE_GROUP,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = run_benchmark(
        input_root=args.input_root,
        bold_root=args.bold_root,
        output_root=args.output_root,
        native_config_path=args.native_config,
        benchmark_config_path=args.benchmark_config,
        basis_scope=args.basis_scope,
    )
    print(f"Decision: {paths.decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
