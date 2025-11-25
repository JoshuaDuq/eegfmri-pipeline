from __future__ import annotations

from typing import List, Tuple, Any

import pandas as pd
import mne


###################################################################
# Plateau Feature Extraction
###################################################################


def build_plateau_features(
    pow_df: pd.DataFrame,
    pow_cols: List[str],
    baseline_df: pd.DataFrame,
    baseline_cols: List[str],
    tfr: mne.time_frequency.EpochsTFR,
    power_bands: List[str],
    logger: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """Construct plateau-averaged band power DataFrame."""
    ch_names = tfr.info["ch_names"]
    col_name_to_series = {}
    plateau_cols = []

    for band in power_bands:
        for ch in ch_names:
            plateau_col_direct = f"pow_{band}_{ch}_plateau"
            if plateau_col_direct in pow_cols:
                name = f"pow_{band}_{ch}"
                col_name_to_series[name] = pow_df[plateau_col_direct]
                plateau_cols.append(name)
                continue

            early_col = f"pow_{band}_{ch}_early"
            mid_col = f"pow_{band}_{ch}_mid"
            late_col = f"pow_{band}_{ch}_late"

            if early_col in pow_cols and mid_col in pow_cols and late_col in pow_cols:
                plateau_val = pow_df[[early_col, mid_col, late_col]].mean(axis=1)
                name = f"pow_{band}_{ch}"
                col_name_to_series[name] = plateau_val
                plateau_cols.append(name)

        if not baseline_df.empty:
            for ch in ch_names:
                baseline_col = f"baseline_{band}_{ch}"
                if baseline_col in baseline_cols:
                    col_name_to_series[baseline_col] = baseline_df[baseline_col]
                    plateau_cols.append(baseline_col)

    plateau_df = pd.DataFrame(col_name_to_series)
    plateau_df = plateau_df.reindex(columns=plateau_cols)
    return plateau_df, plateau_cols

