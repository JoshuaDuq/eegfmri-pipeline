from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.types import PrecomputedData
from eeg_pipeline.domain.features.naming import NamingSchema


def extract_gfp_from_precomputed(
    precomputed: PrecomputedData,
    config: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    if precomputed.data is None or precomputed.windows is None:
        return pd.DataFrame(), []

    from eeg_pipeline.utils.analysis.windowing import get_segment_masks

    windows = precomputed.windows
    masks = get_segment_masks(precomputed.times, windows, config)

    records: List[Dict[str, float]] = []
    for ep_idx in range(precomputed.data.shape[0]):
        x = precomputed.data[ep_idx]
        gfp_t = np.nanstd(x, axis=0)
        rec: Dict[str, float] = {}
        for seg, mask in masks.items():
            if mask is None or not np.any(mask):
                continue
            vals = gfp_t[mask]
            rec[NamingSchema.build("gfp", seg, "broadband", "global", "mean")] = float(
                np.nanmean(vals)
            )
            rec[NamingSchema.build("gfp", seg, "broadband", "global", "max")] = float(
                np.nanmax(vals)
            )
        records.append(rec)

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []
    df = pd.DataFrame(records)
    return df, list(df.columns)


__all__ = ["extract_gfp_from_precomputed"]
