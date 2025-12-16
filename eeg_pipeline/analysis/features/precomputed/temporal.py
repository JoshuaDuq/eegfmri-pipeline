from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.types import PrecomputedData
from eeg_pipeline.domain.features.naming import NamingSchema


def extract_temporal_features_from_precomputed(
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
        rec: Dict[str, float] = {}
        for seg, mask in masks.items():
            if mask is None or not np.any(mask):
                continue
            seg_x = x[:, mask]
            var_mean = float(np.nanmean(np.nanvar(seg_x, axis=1)))
            rms_mean = float(np.nanmean(np.sqrt(np.nanmean(seg_x**2, axis=1))))
            ll_mean = (
                float(np.nanmean(np.nanmean(np.abs(np.diff(seg_x, axis=1)), axis=1)))
                if seg_x.shape[1] > 1
                else np.nan
            )
            rec[NamingSchema.build("temporal", seg, "broadband", "global", "var_mean")] = var_mean
            rec[NamingSchema.build("temporal", seg, "broadband", "global", "rms_mean")] = rms_mean
            rec[
                NamingSchema.build("temporal", seg, "broadband", "global", "line_length_mean")
            ] = ll_mean
        records.append(rec)

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []
    df = pd.DataFrame(records)
    return df, list(df.columns)


__all__ = ["extract_temporal_features_from_precomputed"]
