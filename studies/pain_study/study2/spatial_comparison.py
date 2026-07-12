"""Spatial correspondence tests for Study 2 EEG and fMRI maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from studies.pain_study.study2.statistics import pearson_r, plus_one_p_value
from studies.pain_study.study2.validation import require_config_float, require_config_int


@dataclass(frozen=True)
class SpatialCorrespondenceResult:
    spatial_r: float
    p_value: float
    meaningful: bool
    surrogate_r: np.ndarray


def compute_spatial_correspondence(
    *,
    eeg_map: np.ndarray,
    fmri_map: np.ndarray,
    surrogate_maps: np.ndarray,
    config: Any,
    mask: np.ndarray | None,
) -> SpatialCorrespondenceResult:
    eeg = np.asarray(eeg_map, dtype=float)
    fmri = np.asarray(fmri_map, dtype=float)
    surrogates = np.asarray(surrogate_maps, dtype=float)
    analysis_mask = _analysis_mask(eeg, fmri, mask)
    _validate_inputs(eeg, fmri, surrogates, analysis_mask)
    expected_surrogates = require_config_int(
        config,
        "study2.spatial_comparison.brainsmash_surrogates",
    )
    if expected_surrogates < 1:
        raise ValueError("Study 2 BrainSMASH surrogate count must be positive.")
    if surrogates.shape[0] != expected_surrogates:
        raise ValueError(
            "Study 2 spatial comparison requires exactly "
            f"{expected_surrogates} surrogate maps, got {surrogates.shape[0]}."
        )

    spatial_r = pearson_r(eeg[analysis_mask], fmri[analysis_mask], name="spatial comparison")
    surrogate_r = np.asarray(
        [
            pearson_r(draw[analysis_mask], fmri[analysis_mask], name="spatial surrogate")
            for draw in surrogates
        ],
        dtype=float,
    )
    p_value = plus_one_p_value(abs(spatial_r), np.abs(surrogate_r))
    min_meaningful = require_config_float(
        config,
        "study2.spatial_comparison.min_meaningful_eeg_fmri_abs_r",
    )
    return SpatialCorrespondenceResult(
        spatial_r=spatial_r,
        p_value=p_value,
        meaningful=abs(spatial_r) >= min_meaningful,
        surrogate_r=surrogate_r,
    )


def _analysis_mask(eeg: np.ndarray, fmri: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if eeg.ndim != 1 or fmri.ndim != 1:
        raise ValueError("Study 2 spatial maps must be 1D.")
    if eeg.shape != fmri.shape:
        raise ValueError("Study 2 EEG and fMRI maps must have the same shape.")
    if mask is None:
        raise ValueError("Study 2 spatial mask must be provided.")
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape != eeg.shape:
        raise ValueError("Study 2 spatial mask must match map shape.")
    if not np.any(mask_arr):
        raise ValueError("Study 2 spatial mask must contain at least one vertex.")
    return mask_arr


def _validate_inputs(
    eeg: np.ndarray,
    fmri: np.ndarray,
    surrogates: np.ndarray,
    mask: np.ndarray,
) -> None:
    if surrogates.ndim != 2:
        raise ValueError("Study 2 surrogate_maps must be 2D.")
    if surrogates.shape[1] != eeg.shape[0]:
        raise ValueError("Study 2 surrogate_maps must have one column per map vertex.")
    if surrogates.shape[0] < 1:
        raise ValueError("Study 2 spatial comparison requires at least one surrogate map.")
    if not np.all(np.isfinite(eeg)):
        raise ValueError("Study 2 eeg_map contains non-finite values.")
    if not np.all(np.isfinite(fmri)):
        raise ValueError("Study 2 fmri_map contains non-finite values.")
    if not np.all(np.isfinite(surrogates)):
        raise ValueError("Study 2 surrogate_maps contains non-finite values.")
    if int(np.sum(mask)) < 2:
        raise ValueError("Study 2 spatial comparison requires at least two masked vertices.")


__all__ = ["SpatialCorrespondenceResult", "compute_spatial_correspondence"]
