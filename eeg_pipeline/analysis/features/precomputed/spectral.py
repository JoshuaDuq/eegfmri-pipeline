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

    records: List[Dict[str, float]] = []
    times = precomputed.times
    active_times = times[windows.active_mask]

    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}

        for band in bands:
            if band not in precomputed.band_data:
                continue

            power = precomputed.band_data[band].power[ep_idx]
            baseline_valid_count = 0
            total_channels = len(precomputed.ch_names)
            all_power_full: List[float] = []

            seg_label = getattr(windows, "name", "plateau") or "plateau"

            for ch_idx, ch_name in enumerate(precomputed.ch_names):
                baseline_power, baseline_frac, _, baseline_total = nanmean_with_fraction(
                    power[ch_idx], windows.baseline_mask
                )
                active_power, _, _, _ = nanmean_with_fraction(power[ch_idx], windows.active_mask)
                baseline_valid = (
                    baseline_power > epsilon
                    and baseline_frac >= min_valid_fraction
                    and baseline_total > 0
                )
                if baseline_valid:
                    baseline_valid_count += 1

                if baseline_valid:
                    logratio = np.log10(active_power / baseline_power)
                else:
                    logratio = np.nan
                
                record[
                    NamingSchema.build("spectral", seg_label, band, "ch", "logratio", channel=ch_name)
                ] = float(logratio)
                all_power_full.append(float(logratio) if np.isfinite(logratio) else np.nan)

                if len(active_times) > 2:
                    active_power_trace = power[ch_idx, windows.active_mask]
                    if baseline_valid:
                        logratio_trace = np.log10(active_power_trace / baseline_power)
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

            valid_power = [p for p in all_power_full if np.isfinite(p)]
            baseline_valid_fraction = (
                baseline_valid_count / total_channels if total_channels > 0 else 0.0
            )
            record[
                NamingSchema.build("spectral", "baseline", band, "global", "valid_fraction")
            ] = float(baseline_valid_fraction)

            if baseline_valid_fraction < min_valid_fraction:
                record[
                    NamingSchema.build("spectral", seg_label, band, "global", "logratio_mean")
                ] = np.nan
                record[
                    NamingSchema.build("spectral", seg_label, band, "global", "logratio_std")
                ] = np.nan
            elif valid_power:
                record[
                    NamingSchema.build("spectral", seg_label, band, "global", "logratio_mean")
                ] = float(np.mean(valid_power))
                record[
                    NamingSchema.build("spectral", seg_label, band, "global", "logratio_std")
                ] = float(np.std(valid_power))

        records.append(record)

    columns = list(records[0].keys()) if records else []
    return pd.DataFrame(records), columns, {}


__all__ = [
    "extract_power_from_precomputed",
]
