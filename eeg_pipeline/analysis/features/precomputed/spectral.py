from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from eeg_pipeline.types import PrecomputedData
from eeg_pipeline.utils.analysis.arrays import nanmean_with_fraction
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.domain.features.constants import validate_precomputed
from eeg_pipeline.utils.config.loader import get_feature_constant, get_frequency_bands

from .extras import validate_window_masks


def extract_power_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    is_valid, err_msg = validate_precomputed(precomputed, require_windows=True, require_bands=True)
    if not is_valid:
        logger = getattr(precomputed, "logger", None)
        if logger is not None:
            logger.warning("Power (precomputed): %s; skipping extraction.", err_msg)
        return pd.DataFrame(), [], {}

    n_epochs = precomputed.data.shape[0]
    min_epochs = get_feature_constant(precomputed.config, "MIN_EPOCHS_FOR_FEATURES", 10)
    if n_epochs < min_epochs:
        if precomputed.logger:
            precomputed.logger.warning(
                "Power extraction skipped: only %d epochs available (min=%d). "
                "Insufficient trials for stable power estimation.",
                n_epochs,
                min_epochs,
            )
        return pd.DataFrame(), [], {"skipped_reason": "insufficient_epochs", "n_epochs": n_epochs}

    if not validate_window_masks(precomputed, precomputed.logger):
        return pd.DataFrame(), [], {}

    epsilon = float(get_feature_constant(precomputed.config, "EPSILON_STD", 1e-12))
    min_valid_fraction = float(get_feature_constant(precomputed.config, "MIN_VALID_FRACTION", 0.5))
    windows = precomputed.windows

    # Determine which segments to process
    target_name = getattr(windows, "name", None)
    if target_name and target_name in windows.masks:
        segments_to_process = [(target_name, windows.get_mask(target_name))]
    else:
        # Process all non-baseline segments
        segments_to_process = [
            (name, mask) for name, mask in windows.masks.items() 
            if name.lower() != "baseline" and np.any(mask)
        ]
    
    spatial_modes = getattr(precomputed, "spatial_modes", ["roi", "global"])
    roi_map = {}
    if "roi" in spatial_modes:
        from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
        from eeg_pipeline.utils.analysis.channels import build_roi_map
        roi_defs = get_roi_definitions(precomputed.config)
        if roi_defs:
            roi_map = build_roi_map(precomputed.ch_names, roi_defs)

    times = precomputed.times
    baseline_mask = windows.baseline_mask
    if baseline_mask is None:
        baseline_mask = np.zeros_like(times, dtype=bool)
    records: List[Dict[str, Any]] = [{} for _ in range(n_epochs)]

    if not segments_to_process:
        if precomputed.logger:
            precomputed.logger.warning("Power extraction: No user-defined segments to process.")
        return pd.DataFrame(), [], {}

    qc_payload: Dict[str, Any] = {
        "baseline_valid_fractions": [[] for _ in range(n_epochs)],
        "min_valid_fraction": min_valid_fraction,
    }

    for seg_label, active_mask in segments_to_process:
        active_times = times[active_mask] if np.any(active_mask) else np.array([])
        
        for ep_idx in range(n_epochs):
            record = records[ep_idx]

            for band in bands:
                if band not in precomputed.band_data:
                    continue

                power = precomputed.band_data[band].power[ep_idx]
                baseline_valid_count = 0
                total_channels = len(precomputed.ch_names)
                all_power_full: List[float] = []

                for ch_idx, ch_name in enumerate(precomputed.ch_names):
                    baseline_power, baseline_frac, _, baseline_total = nanmean_with_fraction(
                        power[ch_idx], baseline_mask
                    )
                    active_power, _, _, _ = nanmean_with_fraction(power[ch_idx], active_mask)
                    baseline_valid = (
                        baseline_power > epsilon
                        and baseline_frac >= min_valid_fraction
                        and baseline_total > 0
                    )
                    if baseline_valid:
                        baseline_valid_count += 1

                    if baseline_valid and active_power > 0:
                        logratio = np.log10(active_power / baseline_power)
                    else:
                        logratio = np.nan
                    
                    record[
                        NamingSchema.build("spectral", seg_label, band, "ch", "logratio", channel=ch_name)
                    ] = float(logratio)
                    all_power_full.append(float(logratio) if np.isfinite(logratio) else np.nan)

                    if len(active_times) > 2:
                        active_power_trace = power[ch_idx, active_mask]
                        if baseline_valid:
                            logratio_trace = np.log10(np.maximum(active_power_trace / baseline_power, epsilon))
                            valid_mask = np.isfinite(logratio_trace)
                            if np.sum(valid_mask) > 2:
                                slope, _ = np.polyfit(
                                    active_times[valid_mask], logratio_trace[valid_mask], 1
                                )
                                record[
                                    NamingSchema.build("spectral", seg_label, band, "ch", "slope", channel=ch_name)
                                ] = float(slope)
                            else:
                                record[
                                    NamingSchema.build("spectral", seg_label, band, "ch", "slope", channel=ch_name)
                                ] = np.nan
                        else:
                            record[
                                NamingSchema.build("spectral", seg_label, band, "ch", "slope", channel=ch_name)
                            ] = np.nan
                    else:
                        record[
                            NamingSchema.build("spectral", seg_label, band, "ch", "slope", channel=ch_name)
                        ] = np.nan

                baseline_valid_fraction = (
                    baseline_valid_count / total_channels if total_channels > 0 else 0.0
                )
                
                # Store valid fraction in QC instead of columns
                qc_payload["baseline_valid_fractions"][ep_idx].append(float(baseline_valid_fraction))

                if "global" in spatial_modes:
                    if baseline_valid_fraction < min_valid_fraction:
                        record[
                            NamingSchema.build("spectral", seg_label, band, "global", "logratio_mean")
                        ] = np.nan
                        record[
                            NamingSchema.build("spectral", seg_label, band, "global", "logratio_std")
                        ] = np.nan
                    else:
                        valid_powers = [p for p in all_power_full if np.isfinite(p)]
                        if valid_powers:
                            record[
                                NamingSchema.build("spectral", seg_label, band, "global", "logratio_mean")
                            ] = float(np.mean(valid_powers))
                            record[
                                NamingSchema.build("spectral", seg_label, band, "global", "logratio_std")
                            ] = float(np.std(valid_powers))
                        else:
                            record[
                                NamingSchema.build("spectral", seg_label, band, "global", "logratio_mean")
                            ] = np.nan
                            record[
                                NamingSchema.build("spectral", seg_label, band, "global", "logratio_std")
                            ] = np.nan
                
                if "roi" in spatial_modes and roi_map:
                    for roi_name, roi_indices in roi_map.items():
                        roi_powers = [all_power_full[idx] for idx in roi_indices if np.isfinite(all_power_full[idx])]
                        if roi_powers:
                            record[NamingSchema.build("spectral", seg_label, band, "roi", "logratio_mean", channel=roi_name)] = float(np.mean(roi_powers))
                        else:
                            record[NamingSchema.build("spectral", seg_label, band, "roi", "logratio_mean", channel=roi_name)] = np.nan

                if "channels" not in spatial_modes:
                    for ch_name in precomputed.ch_names:
                        record.pop(NamingSchema.build("spectral", seg_label, band, "ch", "logratio", channel=ch_name), None)
                        record.pop(NamingSchema.build("spectral", seg_label, band, "ch", "slope", channel=ch_name), None)

    if not records or all(not r for r in records):
        return pd.DataFrame(), [], {}

    # Summarize QC
    all_fractions = [f for ep_list in qc_payload["baseline_valid_fractions"] for f in ep_list]
    if all_fractions:
        qc_payload["mean_baseline_valid_fraction"] = float(np.mean(all_fractions))
        qc_payload["min_baseline_valid_fraction"] = float(np.min(all_fractions))
    
    # Remove large per-trial list from final QC to keep it small
    qc_payload.pop("baseline_valid_fractions", None)

    df = pd.DataFrame(records)
    return df, list(df.columns), qc_payload


__all__ = [
    "extract_power_from_precomputed",
]
