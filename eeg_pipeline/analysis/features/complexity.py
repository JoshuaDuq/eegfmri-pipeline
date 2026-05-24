"""
Complexity Feature Extraction
============================

Computes nonlinear complexity metrics per trial/channel (optionally ROI/global):
- Lempel–Ziv complexity (LZC)
- Permutation entropy (PE)
- Sample entropy (SampEn)
- Multiscale entropy (MSE; coarse-graining scales)

Scientific notes
---------------
- Complexity is computed on a configurable signal basis:
  - "filtered": band-passed time series (default; interpretable as oscillatory complexity)
  - "envelope": amplitude envelope (interpretable as amplitude-dynamics complexity)
- No resampling is performed by default. Resampling can distort ordinal patterns and LZC.
- Strict minimum segment duration and sample-count gates prevent unstable estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from eeg_pipeline.analysis.features.rest import (
    is_resting_state_feature_mode,
    select_single_rest_analysis_segment,
    valid_rest_analysis_segment_masks,
)
from eeg_pipeline.domain.features.constants import validate_precomputed
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.types import PrecomputedData
from eeg_pipeline.utils.analysis.spatial import build_roi_map_if_needed
from eeg_pipeline.utils.analysis.signal_metrics import (
    compute_lempel_ziv_complexity as _lempel_ziv_complexity,
    compute_multiscale_entropy as _multiscale_entropy,
    compute_permutation_entropy as _permutation_entropy,
    compute_sample_entropy as _sample_entropy,
)
from eeg_pipeline.utils.analysis.windowing import get_segment_masks
from eeg_pipeline.utils.config.loader import get_config_value


@dataclass(frozen=True)
class ComplexityParams:
    signal_basis: str
    pe_order: int
    pe_delay: int
    sampen_order: int
    sampen_r: float
    mse_scale_min: int
    mse_scale_max: int
    zscore: bool
    min_segment_sec: float
    min_samples: int


def _extract_params(config: Any) -> ComplexityParams:
    signal_basis = (
        str(get_config_value(config, "feature_engineering.complexity.signal_basis", "filtered"))
        .strip()
        .lower()
    )
    if signal_basis not in {"filtered", "envelope"}:
        raise ValueError(
            "feature_engineering.complexity.signal_basis must be 'filtered' or 'envelope'"
        )

    pe_order = int(get_config_value(config, "feature_engineering.complexity.pe_order", 3))
    pe_delay = int(get_config_value(config, "feature_engineering.complexity.pe_delay", 1))
    sampen_order = int(get_config_value(config, "feature_engineering.complexity.sampen_order", 2))
    sampen_r = float(get_config_value(config, "feature_engineering.complexity.sampen_r", 0.2))
    mse_scale_min = int(get_config_value(config, "feature_engineering.complexity.mse_scale_min", 1))
    mse_scale_max = int(
        get_config_value(config, "feature_engineering.complexity.mse_scale_max", 20)
    )
    zscore = bool(get_config_value(config, "feature_engineering.complexity.zscore", True))
    min_segment_sec = float(
        get_config_value(config, "feature_engineering.complexity.min_segment_sec", 2.0)
    )
    min_samples = int(get_config_value(config, "feature_engineering.complexity.min_samples", 200))

    if pe_order < 2:
        raise ValueError("feature_engineering.complexity.pe_order must be >= 2")
    if pe_delay < 1:
        raise ValueError("feature_engineering.complexity.pe_delay must be >= 1")
    if sampen_order < 1:
        raise ValueError("feature_engineering.complexity.sampen_order must be >= 1")
    if not np.isfinite(sampen_r) or sampen_r <= 0:
        raise ValueError("feature_engineering.complexity.sampen_r must be a finite value > 0")
    if mse_scale_min < 1:
        raise ValueError("feature_engineering.complexity.mse_scale_min must be >= 1")
    if mse_scale_max < mse_scale_min:
        raise ValueError("feature_engineering.complexity.mse_scale_max must be >= mse_scale_min")
    if min_segment_sec <= 0:
        raise ValueError("feature_engineering.complexity.min_segment_sec must be > 0")
    if min_samples < 1:
        raise ValueError("feature_engineering.complexity.min_samples must be >= 1")

    # PE needs enough samples for ordinal patterns
    min_needed_for_pe = max(1, (pe_order - 1) * pe_delay + 2)
    # MSE at max scale needs enough points for SampEn templates.
    min_needed_for_mse = max(1, mse_scale_max * (sampen_order + 2))
    required_min_samples = max(min_needed_for_pe, min_needed_for_mse)
    if min_samples < required_min_samples:
        raise ValueError(
            "feature_engineering.complexity.min_samples must be >= "
            f"{required_min_samples} for the configured PE/MSE parameters"
        )

    return ComplexityParams(
        signal_basis=signal_basis,
        pe_order=pe_order,
        pe_delay=pe_delay,
        sampen_order=sampen_order,
        sampen_r=sampen_r,
        mse_scale_min=mse_scale_min,
        mse_scale_max=mse_scale_max,
        zscore=zscore,
        min_segment_sec=min_segment_sec,
        min_samples=min_samples,
    )


def _standardize_trace(trace: np.ndarray, *, zscore: bool) -> np.ndarray:
    trace = np.asarray(trace, dtype=float)
    if not zscore:
        return trace
    finite = trace[np.isfinite(trace)]
    if finite.size < 2:
        return trace
    mu = float(np.mean(finite))
    sd = float(np.std(finite, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return trace - mu
    return (trace - mu) / sd


def _pick_basis_array(band_data: Any, basis: str) -> np.ndarray:
    if basis == "envelope":
        return np.asarray(band_data.envelope, dtype=float)
    return np.asarray(band_data.filtered, dtype=float)


def _mse_scales(params: ComplexityParams) -> List[int]:
    return list(range(int(params.mse_scale_min), int(params.mse_scale_max) + 1))


def _mse_stat_name(scale: int) -> str:
    return f"mse{int(scale):02d}"


def _mean_finite_complexity_values(values: np.ndarray, *, label: str) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError(f"Complexity: {label} has no contributing channels.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"Complexity: non-finite complexity values for {label}; "
            "ROI/global summaries require fixed finite channel support."
        )
    return float(np.mean(arr))


def _valid_analysis_segments(
    segments: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    return valid_rest_analysis_segment_masks(segments)


def _resolve_complexity_segments(
    precomputed: PrecomputedData,
    cfg: Any,
    logger: Any,
) -> Dict[str, np.ndarray]:
    windows = precomputed.windows
    target_name = getattr(windows, "name", None) if windows else None
    task_is_rest = is_resting_state_feature_mode(cfg)

    if target_name and windows is not None:
        mask = windows.get_mask(target_name)
        if mask is not None and np.any(mask):
            return {target_name: np.asarray(mask, dtype=bool)}

        if task_is_rest:
            segment_name, segment_mask = select_single_rest_analysis_segment(
                get_segment_masks(precomputed.times, windows, cfg),
                feature_name="Complexity",
                target_name=str(target_name),
            )
            if logger:
                logger.info(
                    "Complexity: resting-state mode found no valid target window '%s'; "
                    "using available analysis segment '%s' instead.",
                    target_name,
                    segment_name,
                )
            return {segment_name: segment_mask}

        if logger:
            logger.warning(
                "Complexity: targeted window '%s' has no valid mask; skipping.",
                target_name,
            )
        return {}

    segments = get_segment_masks(precomputed.times, windows, cfg)
    if task_is_rest:
        return _valid_analysis_segments(segments)
    return {
        name: np.asarray(mask, dtype=bool)
        for name, mask in segments.items()
        if mask is not None and np.any(mask)
    }


def _compute_epoch_complexity(
    ep_idx: int,
    precomputed: PrecomputedData,
    segment_mask: np.ndarray,
    segment_name: str,
    params: ComplexityParams,
    spatial_modes: List[str],
    roi_map: Dict[str, List[int]],
) -> Dict[str, float]:
    record: Dict[str, float] = {}
    mse_scales = _mse_scales(params)

    sfreq = float(getattr(precomputed, "sfreq", np.nan))
    n_samples = int(np.sum(segment_mask))
    if not np.isfinite(sfreq) or sfreq <= 0:
        return record
    duration_sec = float(n_samples) / float(sfreq) if n_samples > 0 else 0.0
    if duration_sec < params.min_segment_sec or n_samples < params.min_samples:
        return record

    if not precomputed.band_data:
        return record

    for band, band_data in precomputed.band_data.items():
        basis_data = _pick_basis_array(band_data, params.signal_basis)
        if basis_data.ndim != 3:
            continue

        if len(segment_mask) != basis_data.shape[2]:
            continue

        # Extract epoch data first, then apply mask (avoids NumPy advanced indexing quirk)
        epoch_data = basis_data[ep_idx]  # (channels, times)
        trace_matrix = epoch_data[:, segment_mask]  # (channels, masked_times)

        if trace_matrix.shape[1] < params.min_samples:
            continue

        n_channels = len(precomputed.ch_names)
        lzc_per_channel = np.full((n_channels,), np.nan)
        pe_per_channel = np.full((n_channels,), np.nan)
        sampen_per_channel = np.full((n_channels,), np.nan)
        mse_per_channel = np.full((n_channels, len(mse_scales)), np.nan)

        for ch_idx, ch_name in enumerate(precomputed.ch_names):
            trace = _standardize_trace(trace_matrix[ch_idx], zscore=params.zscore)
            if np.isfinite(trace).sum() < params.min_samples:
                continue
            lzc_per_channel[ch_idx] = float(_lempel_ziv_complexity(trace))
            pe_per_channel[ch_idx] = float(
                _permutation_entropy(
                    trace,
                    order=params.pe_order,
                    delay=params.pe_delay,
                )
            )
            sampen_per_channel[ch_idx] = float(
                _sample_entropy(
                    trace,
                    order=params.sampen_order,
                    r=params.sampen_r,
                )
            )
            mse_values = _multiscale_entropy(
                trace,
                scales=mse_scales,
                order=params.sampen_order,
                r=params.sampen_r,
            )
            for scale_idx, scale in enumerate(mse_scales):
                mse_per_channel[ch_idx, scale_idx] = float(mse_values.get(scale, np.nan))

            if "channels" in spatial_modes:
                record[
                    NamingSchema.build("comp", segment_name, band, "ch", "lzc", channel=ch_name)
                ] = float(lzc_per_channel[ch_idx])
                record[
                    NamingSchema.build("comp", segment_name, band, "ch", "pe", channel=ch_name)
                ] = float(pe_per_channel[ch_idx])
                record[
                    NamingSchema.build("comp", segment_name, band, "ch", "sampen", channel=ch_name)
                ] = float(sampen_per_channel[ch_idx])
                for scale_idx, scale in enumerate(mse_scales):
                    record[
                        NamingSchema.build(
                            "comp",
                            segment_name,
                            band,
                            "ch",
                            _mse_stat_name(scale),
                            channel=ch_name,
                        )
                    ] = float(mse_per_channel[ch_idx, scale_idx])

        if "roi" in spatial_modes and roi_map:
            for roi_name, idxs in roi_map.items():
                if not idxs:
                    continue
                record[
                    NamingSchema.build("comp", segment_name, band, "roi", "lzc", channel=roi_name)
                ] = float(
                    _mean_finite_complexity_values(
                        lzc_per_channel[idxs],
                        label=f"segment={segment_name}, band={band}, roi={roi_name}, metric=lzc",
                    )
                )
                record[
                    NamingSchema.build("comp", segment_name, band, "roi", "pe", channel=roi_name)
                ] = float(
                    _mean_finite_complexity_values(
                        pe_per_channel[idxs],
                        label=f"segment={segment_name}, band={band}, roi={roi_name}, metric=pe",
                    )
                )
                record[
                    NamingSchema.build(
                        "comp", segment_name, band, "roi", "sampen", channel=roi_name
                    )
                ] = float(
                    _mean_finite_complexity_values(
                        sampen_per_channel[idxs],
                        label=f"segment={segment_name}, band={band}, roi={roi_name}, metric=sampen",
                    )
                )
                for scale_idx, scale in enumerate(mse_scales):
                    record[
                        NamingSchema.build(
                            "comp",
                            segment_name,
                            band,
                            "roi",
                            _mse_stat_name(scale),
                            channel=roi_name,
                        )
                    ] = _mean_finite_complexity_values(
                        mse_per_channel[idxs, scale_idx],
                        label=(
                            f"segment={segment_name}, band={band}, roi={roi_name}, "
                            f"metric={_mse_stat_name(scale)}"
                        ),
                    )

        if "global" in spatial_modes:
            record[NamingSchema.build("comp", segment_name, band, "global", "lzc")] = (
                _mean_finite_complexity_values(
                    lzc_per_channel,
                    label=f"segment={segment_name}, band={band}, global metric=lzc",
                )
            )
            record[NamingSchema.build("comp", segment_name, band, "global", "pe")] = (
                _mean_finite_complexity_values(
                    pe_per_channel,
                    label=f"segment={segment_name}, band={band}, global metric=pe",
                )
            )
            record[NamingSchema.build("comp", segment_name, band, "global", "sampen")] = float(
                _mean_finite_complexity_values(
                    sampen_per_channel,
                    label=f"segment={segment_name}, band={band}, global metric=sampen",
                )
            )
            for scale_idx, scale in enumerate(mse_scales):
                record[
                    NamingSchema.build("comp", segment_name, band, "global", _mse_stat_name(scale))
                ] = float(
                    _mean_finite_complexity_values(
                        mse_per_channel[:, scale_idx],
                        label=(
                            f"segment={segment_name}, band={band}, "
                            f"global metric={_mse_stat_name(scale)}"
                        ),
                    )
                )

    return record


def extract_complexity_from_precomputed(
    precomputed: PrecomputedData,
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    is_valid, err_msg = validate_precomputed(precomputed, require_windows=True, require_bands=True)
    if not is_valid:
        logger = getattr(precomputed, "logger", None)
        if logger is not None:
            logger.warning("Complexity: %s; skipping extraction.", err_msg)
        return pd.DataFrame(), []

    cfg = getattr(precomputed, "config", None) or {}
    logger = getattr(precomputed, "logger", None)
    params = _extract_params(cfg)
    segments = _resolve_complexity_segments(precomputed, cfg, logger)

    if not segments:
        return pd.DataFrame(), []

    spatial_modes = getattr(precomputed, "spatial_modes", None) or ["roi", "global"]
    roi_map = build_roi_map_if_needed(spatial_modes, precomputed.ch_names, cfg)

    n_epochs = int(precomputed.data.shape[0])
    n_jobs = int(get_config_value(cfg, "feature_engineering.parallel.n_jobs_complexity", n_jobs))
    n_jobs = max(1, n_jobs)

    # One record per epoch; merge segments in-place.
    records: List[Dict[str, float]] = [dict() for _ in range(n_epochs)]
    requested_segment = str(getattr(precomputed.windows, "name", "") or "").strip().lower()
    for segment_name, segment_mask in segments.items():
        if segment_name == "baseline" and requested_segment != "baseline":
            continue

        # Validate mask length matches data
        n_times = precomputed.data.shape[2]
        if len(segment_mask) != n_times:
            raise ValueError(
                "Complexity: requested segment "
                f"'{segment_name}' mask length ({len(segment_mask)}) does not "
                f"match data times ({n_times})."
            )

        n_masked = int(np.sum(segment_mask))
        if n_masked < params.min_samples:
            raise ValueError(
                "Complexity: requested segment "
                f"'{segment_name}' is too short "
                f"({n_masked} samples < {params.min_samples} required)."
            )

        per_epoch = Parallel(n_jobs=n_jobs)(
            delayed(_compute_epoch_complexity)(
                ep_idx,
                precomputed,
                np.asarray(segment_mask, dtype=bool),
                str(segment_name),
                params,
                spatial_modes,
                roi_map,
            )
            for ep_idx in range(n_epochs)
        )

        for i, rec in enumerate(per_epoch):
            records[i].update(rec)

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []

    df = pd.DataFrame(records)
    df.attrs["signal_basis"] = params.signal_basis
    df.attrs["pe_order"] = int(params.pe_order)
    df.attrs["pe_delay"] = int(params.pe_delay)
    df.attrs["sampen_order"] = int(params.sampen_order)
    df.attrs["sampen_r"] = float(params.sampen_r)
    df.attrs["mse_scale_min"] = int(params.mse_scale_min)
    df.attrs["mse_scale_max"] = int(params.mse_scale_max)
    df.attrs["zscore"] = bool(params.zscore)
    df.attrs["min_segment_sec"] = float(params.min_segment_sec)
    df.attrs["min_samples"] = int(params.min_samples)
    return df, list(df.columns)


__all__ = ["extract_complexity_from_precomputed"]
