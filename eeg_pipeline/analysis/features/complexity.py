"""
Complexity Feature Extraction
============================

Computes nonlinear complexity metrics per trial/channel (optionally ROI/global):
- Lempel–Ziv complexity (LZC)
- Permutation entropy (PE)

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

from eeg_pipeline.domain.features.constants import validate_precomputed
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.types import PrecomputedData
from eeg_pipeline.utils.analysis.signal_metrics import (
    compute_lempel_ziv_complexity as _lempel_ziv_complexity,
    compute_permutation_entropy as _permutation_entropy,
)
from eeg_pipeline.utils.analysis.windowing import get_segment_masks
from eeg_pipeline.utils.config.loader import get_config_value


@dataclass(frozen=True)
class ComplexityParams:
    signal_basis: str
    pe_order: int
    pe_delay: int
    zscore: bool
    min_segment_sec: float
    min_samples: int


def _extract_params(config: Any) -> ComplexityParams:
    signal_basis = str(get_config_value(config, "feature_engineering.complexity.signal_basis", "filtered")).strip().lower()
    if signal_basis not in {"filtered", "envelope"}:
        signal_basis = "filtered"

    pe_order = int(get_config_value(config, "feature_engineering.complexity.pe_order", 3))
    pe_delay = int(get_config_value(config, "feature_engineering.complexity.pe_delay", 1))
    zscore = bool(get_config_value(config, "feature_engineering.complexity.zscore", True))
    min_segment_sec = float(get_config_value(config, "feature_engineering.complexity.min_segment_sec", 2.0))
    min_samples = int(get_config_value(config, "feature_engineering.complexity.min_samples", 200))

    # PE needs enough samples for ordinal patterns
    min_needed_for_pe = max(1, (pe_order - 1) * pe_delay + 2)
    min_samples = max(min_samples, min_needed_for_pe)

    return ComplexityParams(
        signal_basis=signal_basis,
        pe_order=pe_order,
        pe_delay=pe_delay,
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

        for ch_idx, ch_name in enumerate(precomputed.ch_names):
            trace = _standardize_trace(trace_matrix[ch_idx], zscore=params.zscore)
            if np.isfinite(trace).sum() < params.min_samples:
                continue
            try:
                lzc_per_channel[ch_idx] = float(_lempel_ziv_complexity(trace))
            except Exception:
                lzc_per_channel[ch_idx] = np.nan
            try:
                pe_per_channel[ch_idx] = float(
                    _permutation_entropy(
                        trace,
                        order=params.pe_order,
                        delay=params.pe_delay,
                    )
                )
            except Exception:
                pe_per_channel[ch_idx] = np.nan

            if "channels" in spatial_modes:
                record[NamingSchema.build("comp", segment_name, band, "ch", "lzc", channel=ch_name)] = float(
                    lzc_per_channel[ch_idx]
                )
                record[NamingSchema.build("comp", segment_name, band, "ch", "pe", channel=ch_name)] = float(
                    pe_per_channel[ch_idx]
                )

        if "roi" in spatial_modes and roi_map:
            for roi_name, idxs in roi_map.items():
                if not idxs:
                    continue
                record[NamingSchema.build("comp", segment_name, band, "roi", "lzc", channel=roi_name)] = float(
                    np.nanmean(lzc_per_channel[idxs])
                )
                record[NamingSchema.build("comp", segment_name, band, "roi", "pe", channel=roi_name)] = float(
                    np.nanmean(pe_per_channel[idxs])
                )

        if "global" in spatial_modes:
            record[NamingSchema.build("comp", segment_name, band, "global", "lzc")] = float(np.nanmean(lzc_per_channel))
            record[NamingSchema.build("comp", segment_name, band, "global", "pe")] = float(np.nanmean(pe_per_channel))

    return record


def _build_roi_map(config: Any, ch_names: List[str], spatial_modes: List[str]) -> Dict[str, List[int]]:
    if "roi" not in spatial_modes:
        return {}
    from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
    from eeg_pipeline.utils.analysis.channels import build_roi_map

    roi_defs = get_roi_definitions(config)
    if not roi_defs:
        return {}
    return build_roi_map(ch_names, roi_defs)


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

    windows = precomputed.windows
    target_name = getattr(windows, "name", None) if windows else None
    allow_full_epoch_fallback = bool(
        get_config_value(cfg, "feature_engineering.windows.allow_full_epoch_fallback", False)
    )

    # Get segment masks with proper targeted window handling
    if target_name and windows is not None:
        mask = windows.get_mask(target_name)
        if mask is not None and np.any(mask):
            segments = {target_name: mask}
        else:
            if logger:
                if allow_full_epoch_fallback:
                    logger.warning(
                        "Complexity: targeted window '%s' has no valid mask; using full epoch.",
                        target_name,
                    )
                else:
                    logger.warning(
                        "Complexity: targeted window '%s' has no valid mask; skipping.",
                        target_name,
                    )
            if allow_full_epoch_fallback:
                segments = {target_name: np.ones(len(precomputed.times), dtype=bool)}
            else:
                return pd.DataFrame(), []
    else:
        segments = get_segment_masks(precomputed.times, windows, cfg)
        segments = {k: v for k, v in segments.items() if v is not None and np.any(v)}

    if not segments:
        return pd.DataFrame(), []

    spatial_modes = getattr(precomputed, "spatial_modes", None) or ["roi", "global"]
    roi_map = _build_roi_map(cfg, precomputed.ch_names, spatial_modes)

    n_epochs = int(precomputed.data.shape[0])
    n_jobs = int(get_config_value(cfg, "feature_engineering.parallel.n_jobs_complexity", n_jobs))
    n_jobs = max(1, n_jobs)

    # One record per epoch; merge segments in-place.
    records: List[Dict[str, float]] = [dict() for _ in range(n_epochs)]
    for segment_name, segment_mask in segments.items():
        if segment_name == "baseline":
            continue

        # Validate mask length matches data
        n_times = precomputed.data.shape[2]
        if len(segment_mask) != n_times:
            if logger:
                logger.warning(
                    "Complexity: segment '%s' mask length (%d) != data times (%d); skipping.",
                    segment_name, len(segment_mask), n_times,
                )
            continue

        n_masked = int(np.sum(segment_mask))
        if n_masked < params.min_samples:
            continue

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
    df.attrs["zscore"] = bool(params.zscore)
    df.attrs["min_segment_sec"] = float(params.min_segment_sec)
    df.attrs["min_samples"] = int(params.min_samples)
    return df, list(df.columns)


__all__ = ["extract_complexity_from_precomputed"]

