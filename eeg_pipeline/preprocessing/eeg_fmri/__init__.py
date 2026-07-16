"""Native EEG-fMRI artifact correction."""

from eeg_pipeline.preprocessing.eeg_fmri.cardiac import (
    CardiacArtifactParameters,
    QrsDetection,
    apply_cardiac_obs_in_place,
    detect_qrs,
)
from eeg_pipeline.preprocessing.eeg_fmri.gradient import (
    GradientArtifactParameters,
    GradientCorrectionResult,
    VolumeBoundary,
    correct_gradient_artifact,
    resolve_volume_boundary,
    resolve_volume_segments,
    validate_volume_samples,
)
from eeg_pipeline.preprocessing.eeg_fmri.pipeline import (
    NativeCorrectionResult,
    preprocess_raw_in_place,
)
from eeg_pipeline.preprocessing.eeg_fmri.qc import (
    CardiacLockedComparison,
    CardiacLockedSummary,
    compare_cardiac_locked_summaries,
    summarize_cardiac_locked_eeg,
)
from eeg_pipeline.preprocessing.eeg_fmri.sequence import (
    MultibandSliceSchedule,
    load_multiband_slice_schedule,
)

__all__ = [
    "CardiacArtifactParameters",
    "CardiacLockedComparison",
    "CardiacLockedSummary",
    "GradientArtifactParameters",
    "GradientCorrectionResult",
    "MultibandSliceSchedule",
    "NativeCorrectionResult",
    "QrsDetection",
    "VolumeBoundary",
    "apply_cardiac_obs_in_place",
    "compare_cardiac_locked_summaries",
    "correct_gradient_artifact",
    "detect_qrs",
    "load_multiband_slice_schedule",
    "preprocess_raw_in_place",
    "resolve_volume_boundary",
    "resolve_volume_segments",
    "summarize_cardiac_locked_eeg",
    "validate_volume_samples",
]
