"""Ordered native scanner-gradient and pulse-artifact preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np

from eeg_pipeline.preprocessing.eeg_fmri.cardiac import (
    QrsDetection,
    apply_cardiac_obs_in_place,
    detect_qrs,
)
from eeg_pipeline.preprocessing.eeg_fmri.config import NativeEegFmriParameters
from eeg_pipeline.preprocessing.eeg_fmri.gradient import (
    correct_gradient_artifact,
    resolve_volume_segments,
)
from eeg_pipeline.preprocessing.eeg_fmri.mne_io import (
    extract_volume_samples,
    validate_acquisition,
)
from eeg_pipeline.preprocessing.eeg_fmri.neuxus_qrs import (
    MneQrsDetector,
    PanTompkinsQrsDetector,
    QrsDetector,
)
from eeg_pipeline.preprocessing.eeg_fmri.qc import (
    CardiacLockedComparison,
    HarmonicStageQc,
    compare_cardiac_locked_summaries,
    reference_harmonic_frequencies,
    summarize_cardiac_locked_eeg,
    summarize_raw_harmonics,
)
from eeg_pipeline.preprocessing.eeg_fmri.sequence import MultibandSliceSchedule


@dataclass(frozen=True)
class NativeCorrectionResult:
    """Corrected recording and run-level artifact-correction diagnostics."""

    raw: mne.io.BaseRaw
    qrs: QrsDetection
    volume_shifts_samples: np.ndarray
    group_shifts_samples: np.ndarray
    residual_obs_removed_rms: float
    marker_offsets_samples: np.ndarray
    complete_volume_count: int
    discarded_terminal_samples: int
    harmonic_stages: dict[str, HarmonicStageQc]
    cardiac_qc: CardiacLockedComparison


def _validate_terminal_annotations(
    raw: mne.io.BaseRaw,
    crop_stop_sample: int,
) -> None:
    sampling_frequency = float(raw.info["sfreq"])
    onset_samples = np.rint(raw.annotations.onset * sampling_frequency).astype(int)
    forbidden = sorted(
        {
            description
            for onset, description in zip(
                onset_samples,
                raw.annotations.description,
                strict=True,
            )
            if onset >= crop_stop_sample
            and description != "Volume/V  1"
            and not description.startswith("SyncStatus/")
        }
    )
    if forbidden:
        raise ValueError(
            "Cannot crop an incomplete terminal scanner interval containing task annotations: "
            + ", ".join(forbidden)
        )


def _crop_incomplete_terminal_volume(raw: mne.io.BaseRaw, crop_stop_sample: int) -> None:
    _validate_terminal_annotations(raw, crop_stop_sample)
    sampling_frequency = float(raw.info["sfreq"])
    final_retained_time = (crop_stop_sample - 1) / sampling_frequency
    raw.crop(tmin=0.0, tmax=final_retained_time, include_tmax=True, verbose=False)


def _low_pass_and_resample(
    raw: mne.io.BaseRaw,
    parameters: NativeEegFmriParameters,
) -> None:
    raw.filter(
        l_freq=None,
        h_freq=parameters.low_pass_frequency_hz,
        picks=raw.ch_names,
        method="fir",
        phase="zero",
        fir_window="hamming",
        fir_design="firwin",
        verbose=False,
    )
    raw.resample(
        parameters.output_sampling_frequency_hz,
        npad="auto",
        method="polyphase",
        verbose=False,
    )
    if not np.isclose(
        raw.info["sfreq"],
        parameters.output_sampling_frequency_hz,
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError("MNE resampling did not produce the configured sampling frequency")


def preprocess_raw_in_place(
    raw: mne.io.BaseRaw,
    *,
    parameters: NativeEegFmriParameters,
    qrs_detector: QrsDetector,
    slice_schedule: MultibandSliceSchedule,
) -> NativeCorrectionResult:
    """Apply the complete native EEG-fMRI artifact pipeline in place."""
    if not raw.preload:
        raise ValueError("Native EEG-fMRI preprocessing requires preloaded data")
    validate_acquisition(
        raw,
        expected_sampling_frequency=parameters.acquisition_sampling_frequency_hz,
        expected_channel_count=parameters.expected_channel_count,
        ecg_channel=parameters.ecg_channel,
    )
    raw_harmonics = summarize_raw_harmonics(
        raw,
        stage="raw",
        channels=parameters.qc_channels,
        welch_duration_seconds=parameters.qc_welch_duration_seconds,
        minimum_duration_seconds=parameters.qc_minimum_duration_seconds,
    )
    harmonic_stages = {"raw": raw_harmonics}
    harmonic_references = reference_harmonic_frequencies(raw_harmonics.summary)
    observed_volume_samples = extract_volume_samples(
        raw,
        annotation_description=parameters.volume_annotation,
    )
    boundaries = resolve_volume_segments(
        observed_volume_samples,
        n_samples=raw.n_times,
        sampling_frequency=float(raw.info["sfreq"]),
        repetition_time_seconds=parameters.repetition_time_seconds,
        maximum_marker_deviation_samples=parameters.maximum_marker_deviation_samples,
    )
    crop_boundaries = [boundary for boundary in boundaries if boundary.crop_stop_sample is not None]
    if len(crop_boundaries) > 1 or (crop_boundaries and crop_boundaries[0] is not boundaries[-1]):
        raise RuntimeError("Only the final scanner block may require terminal cropping")
    if crop_boundaries:
        _crop_incomplete_terminal_volume(raw, crop_boundaries[0].crop_stop_sample)

    eeg_picks = mne.pick_types(raw.info, eeg=True, ecg=False, exclude=[])
    corrected_data = raw._data
    volume_shifts = []
    group_shifts = []
    residual_obs_rms = []
    residual_obs_weights = []
    for boundary in boundaries:
        gradient = correct_gradient_artifact(
            corrected_data,
            boundary.complete_volume_samples,
            sampling_frequency=float(raw.info["sfreq"]),
            alignment_picks=eeg_picks,
            residual_obs_picks=eeg_picks,
            slice_schedule=slice_schedule,
            parameters=parameters.gradient,
        )
        corrected_data = gradient.data
        volume_shifts.append(gradient.volume_shifts_samples)
        group_shifts.append(gradient.group_shifts_samples)
        residual_obs_rms.append(gradient.residual_obs_removed_rms)
        residual_obs_weights.append(boundary.complete_volume_samples.size)
    raw._data = corrected_data
    harmonic_stages["gradient_corrected"] = summarize_raw_harmonics(
        raw,
        stage="gradient_corrected",
        channels=parameters.qc_channels,
        welch_duration_seconds=parameters.qc_welch_duration_seconds,
        minimum_duration_seconds=parameters.qc_minimum_duration_seconds,
        reference_frequencies=harmonic_references,
    )
    _low_pass_and_resample(raw, parameters)
    qrs = detect_qrs(
        raw,
        ecg_channel=parameters.ecg_channel,
        detector=qrs_detector,
        fallback_detectors=(
            MneQrsDetector(parameters.cardiac.detection),
            PanTompkinsQrsDetector(parameters.cardiac.detection),
        ),
        parameters=parameters.cardiac,
    )
    cardiac_before = summarize_cardiac_locked_eeg(raw, qrs_times=qrs.times)
    apply_cardiac_obs_in_place(
        raw,
        qrs=qrs,
        parameters=parameters.cardiac,
    )
    cardiac_after = summarize_cardiac_locked_eeg(raw, qrs_times=qrs.times)
    cardiac_qc = compare_cardiac_locked_summaries(cardiac_before, cardiac_after)
    harmonic_stages["final"] = summarize_raw_harmonics(
        raw,
        stage="final",
        channels=parameters.qc_channels,
        welch_duration_seconds=parameters.qc_welch_duration_seconds,
        minimum_duration_seconds=parameters.qc_minimum_duration_seconds,
        reference_frequencies=harmonic_references,
    )
    return NativeCorrectionResult(
        raw=raw,
        qrs=qrs,
        volume_shifts_samples=np.concatenate(volume_shifts),
        group_shifts_samples=np.concatenate(group_shifts, axis=0),
        residual_obs_removed_rms=float(
            np.sqrt(
                np.average(
                    np.square(residual_obs_rms),
                    weights=residual_obs_weights,
                )
            )
        ),
        marker_offsets_samples=np.concatenate(
            [boundary.marker_offsets_samples for boundary in boundaries]
        ),
        complete_volume_count=sum(boundary.complete_volume_samples.size for boundary in boundaries),
        discarded_terminal_samples=sum(
            boundary.discarded_terminal_samples for boundary in boundaries
        ),
        harmonic_stages=harmonic_stages,
        cardiac_qc=cardiac_qc,
    )
