from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


_DEFAULT_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


def infer_power_band(column_name: str, *, bands: Optional[List[str]] = None) -> Optional[str]:
    name = str(column_name).lower()
    if name.startswith("qc_"):
        return None

    bands_use = [str(b).lower() for b in (bands or _DEFAULT_BANDS)]
    bands_set = set(bands_use)

    if name.startswith("power_"):
        parts = name.split("_")
        # Support both legacy power_{band}_... and v2 power_{segment}_{band}_...
        for idx in (1, 2):
            if len(parts) > idx and parts[idx] in bands_set:
                return parts[idx]

        # Fallback: scan tokens
        for token in parts:
            if token in bands_set:
                return token
        return None
    if name.startswith("pow_"):
        parts = name.split("_")
        if len(parts) >= 2:
            if parts[1] in bands_set:
                return parts[1]

        for token in parts:
            if token in bands_set:
                return token
        return None

    # Substring-based legacy patterns (kept for backward compatibility)
    for band_l in bands_use:
        if (
            f"power_baseline_{band_l}" in name
            or f"power_stim_{band_l}" in name
            or f"_{band_l}_pow" in name
            or f"_{band_l}_power" in name
            or f"pow_{band_l}" in name
            or f"power_{band_l}" in name
        ):
            return band_l
    return None


def get_power_columns_by_band(df: pd.DataFrame, *, bands: Optional[List[str]] = None) -> Dict[str, List[str]]:
    bands = bands or _DEFAULT_BANDS
    power_cols: Dict[str, List[str]] = {}

    for band in bands:
        band_cols: List[str] = []
        for c in df.columns:
            c_band = infer_power_band(c, bands=bands)
            if c_band == band:
                band_cols.append(c)
        if band_cols:
            power_cols[band] = band_cols

    return power_cols


def get_connectivity_columns_by_band(df: pd.DataFrame, *, bands: Optional[List[str]] = None) -> Dict[str, List[str]]:
    bands = bands or _DEFAULT_BANDS
    conn_cols: Dict[str, List[str]] = {}

    for band in bands:
        band_cols = [
            c
            for c in df.columns
            if (
                f"conn_plateau_{band}_" in c.lower()
                or f"conn_{band}_" in c.lower()
                or f"conn_legacy_plateau_{band}_" in c.lower()
                or f"conn_legacy_{band}_" in c.lower()
            )
        ]
        if band_cols:
            conn_cols[band] = band_cols

    return conn_cols


def get_itpc_columns_by_band(df: pd.DataFrame, *, bands: Optional[List[str]] = None) -> Dict[str, List[str]]:
    bands = bands or _DEFAULT_BANDS
    itpc_cols: Dict[str, List[str]] = {}

    for band in bands:
        band_cols = [
            c
            for c in df.columns
            if f"itpc_plateau_{band}_" in c.lower() or f"itpc_{band}_" in c.lower()
        ]
        if band_cols:
            itpc_cols[band] = band_cols

    return itpc_cols


def get_aperiodic_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    aper_cols: Dict[str, List[str]] = {}

    for metric in ["slope", "offset", "exponent"]:
        cols = [
            c
            for c in df.columns
            if f"aper_{metric}" in c.lower() or f"aperiodic_{metric}" in c.lower()
        ]
        if metric == "slope":
            cols.extend([
                c
                for c in df.columns
                if str(c).lower().startswith("aperiodic_") and str(c).lower().endswith("_slope")
            ])
        if metric == "offset":
            cols.extend([
                c
                for c in df.columns
                if str(c).lower().startswith("aperiodic_") and str(c).lower().endswith("_offset")
            ])
        if cols:
            aper_cols[metric] = list(dict.fromkeys(cols))

    return aper_cols


def get_microstate_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    ms_cols: Dict[str, List[str]] = {}

    for metric in ["coverage", "duration", "occurrence", "gev"]:
        cols = [
            c
            for c in df.columns
            if f"ms_{metric}" in c.lower() or f"microstate_{metric}" in c.lower()
        ]
        cols.extend(
            [
                c
                for c in df.columns
                if str(c).lower().startswith("microstates_") and f"_{metric}" in str(c).lower()
            ]
        )
        if cols:
            ms_cols[metric] = list(dict.fromkeys(cols))

    return ms_cols
