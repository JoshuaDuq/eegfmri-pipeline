from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.types import PrecomputedData
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.utils.config.loader import get_config_value, get_feature_constant


def extract_band_ratios_from_precomputed(
    precomputed: PrecomputedData,
    config: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    if not precomputed.band_data or precomputed.windows is None:
        return pd.DataFrame(), []

    ratio_pairs = get_config_value(config, "feature_engineering.spectral.ratio_pairs", [])
    pairs: List[Tuple[str, str]] = []
    for entry in ratio_pairs:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            pairs.append((str(entry[0]), str(entry[1])))
    if not pairs:
        return pd.DataFrame(), []

    plateau_mask = getattr(precomputed.windows, "active_mask", None)
    if plateau_mask is None or not np.any(plateau_mask):
        return pd.DataFrame(), []

    eps = float(get_feature_constant(config, "EPSILON_STD", 1e-12))
    records: List[Dict[str, float]] = []
    for ep_idx in range(precomputed.data.shape[0]):
        rec: Dict[str, float] = {}
        band_means: Dict[str, float] = {}
        for band, bd in precomputed.band_data.items():
            p = bd.power[ep_idx]
            band_means[band] = float(np.nanmean(p[:, plateau_mask]))
        for num, den in pairs:
            if num not in band_means or den not in band_means:
                continue
            denom = band_means[den]
            if not np.isfinite(denom) or denom <= eps:
                ratio = np.nan
            else:
                ratio = float(band_means[num] / denom)
            rec[
                NamingSchema.build(
                    "ratios",
                    "plateau",
                    f"{num}_{den}",
                    "global",
                    "power_ratio",
                )
            ] = ratio
        records.append(rec)

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []
    df = pd.DataFrame(records)
    return df, list(df.columns)


__all__ = ["extract_band_ratios_from_precomputed"]
