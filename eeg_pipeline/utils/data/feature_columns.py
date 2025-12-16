from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from eeg_pipeline.domain.features.naming import NamingSchema, parse_legacy_power_feature_name


_DEFAULT_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


def infer_power_band(column_name: str, *, bands: Optional[List[str]] = None) -> Optional[str]:
    name = str(column_name).lower()
    if name.startswith("qc_"):
        return None

    bands_use = [str(b).lower() for b in (bands or _DEFAULT_BANDS)]
    bands_set = set(bands_use)

    parsed = NamingSchema.parse(str(column_name))
    if parsed.get("valid") and parsed.get("group") == "power":
        band = str(parsed.get("band") or "").lower()
        return band if band in bands_set else None

    legacy = parse_legacy_power_feature_name(str(column_name))
    if legacy is None:
        return None
    legacy_band, _legacy_ch = legacy
    legacy_band = str(legacy_band).lower()
    return legacy_band if legacy_band in bands_set else None


def get_power_columns_by_band(df: pd.DataFrame, *, bands: Optional[List[str]] = None) -> Dict[str, List[str]]:
    bands = bands or _DEFAULT_BANDS
    power_cols: Dict[str, List[str]] = {}

    for band in bands:
        band_cols: List[str] = []
        for c in df.columns:
            parsed = NamingSchema.parse(str(c))
            if parsed.get("valid") and parsed.get("group") == "power":
                if str(parsed.get("band") or "").lower() == str(band).lower():
                    band_cols.append(str(c))
                continue

            legacy = parse_legacy_power_feature_name(str(c))
            if legacy is None:
                continue
            legacy_band, _legacy_ch = legacy
            if str(legacy_band).lower() == str(band).lower():
                band_cols.append(str(c))
        if band_cols:
            power_cols[band] = band_cols

    return power_cols


def get_connectivity_columns_by_band(df: pd.DataFrame, *, bands: Optional[List[str]] = None) -> Dict[str, List[str]]:
    bands = bands or _DEFAULT_BANDS
    conn_cols: Dict[str, List[str]] = {}

    bands_set = {str(b).lower() for b in bands}
    for band in bands:
        band_l = str(band).lower()
        if band_l not in bands_set:
            continue
        band_cols: List[str] = []
        for c in df.columns:
            parsed = NamingSchema.parse(str(c))
            if not parsed.get("valid"):
                continue
            if parsed.get("group") != "connectivity":
                continue
            if str(parsed.get("band") or "").lower() != band_l:
                continue
            band_cols.append(str(c))
        if band_cols:
            conn_cols[band] = band_cols

    return conn_cols


def get_itpc_columns_by_band(df: pd.DataFrame, *, bands: Optional[List[str]] = None) -> Dict[str, List[str]]:
    bands = bands or _DEFAULT_BANDS
    itpc_cols: Dict[str, List[str]] = {}

    bands_set = {str(b).lower() for b in bands}
    for band in bands:
        band_l = str(band).lower()
        if band_l not in bands_set:
            continue
        band_cols: List[str] = []
        for c in df.columns:
            parsed = NamingSchema.parse(str(c))
            if not parsed.get("valid"):
                continue
            if parsed.get("group") != "itpc":
                continue
            if str(parsed.get("band") or "").lower() != band_l:
                continue
            band_cols.append(str(c))
        if band_cols:
            itpc_cols[band] = band_cols

    return itpc_cols


def get_aperiodic_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    aper_cols: Dict[str, List[str]] = {}

    for metric in ["slope", "offset", "exponent"]:
        cols: List[str] = []
        for c in df.columns:
            parsed = NamingSchema.parse(str(c))
            if not parsed.get("valid"):
                continue
            if parsed.get("group") != "aperiodic":
                continue
            stat = str(parsed.get("stat") or "")
            if stat == metric or stat.endswith(f"_{metric}"):
                cols.append(str(c))
        if cols:
            aper_cols[metric] = list(dict.fromkeys(cols))

    return aper_cols


def get_microstate_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    ms_cols: Dict[str, List[str]] = {}

    for metric in ["coverage", "duration", "occurrence", "gev"]:
        cols: List[str] = []
        for c in df.columns:
            parsed = NamingSchema.parse(str(c))
            if not parsed.get("valid"):
                continue
            if parsed.get("group") != "microstates":
                continue
            stat = str(parsed.get("stat") or "")
            if stat == metric or stat.endswith(f"_{metric}"):
                cols.append(str(c))
        if cols:
            ms_cols[metric] = list(dict.fromkeys(cols))

    return ms_cols
