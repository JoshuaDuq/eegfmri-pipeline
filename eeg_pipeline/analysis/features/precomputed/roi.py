from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import re

import numpy as np
import pandas as pd

from eeg_pipeline.types import PrecomputedData
from eeg_pipeline.utils.analysis.features.metadata import NamingSchema
from eeg_pipeline.utils.config.loader import get_config_value, get_feature_constant


def _compile_roi_indices(ch_names: List[str], roi_defs: Dict[str, List[str]]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for roi_name, patterns in (roi_defs or {}).items():
        indices: List[int] = []
        compiled = [re.compile(p) for p in patterns] if isinstance(patterns, list) else []
        for idx, ch in enumerate(ch_names):
            if any(rgx.match(ch) for rgx in compiled):
                indices.append(idx)
        if indices:
            out[str(roi_name)] = indices
    return out


def extract_roi_features_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
    config: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    if not precomputed.band_data or precomputed.windows is None:
        return pd.DataFrame(), []

    roi_defs = get_config_value(config, "time_frequency_analysis.rois", {})
    roi_to_idx = _compile_roi_indices(precomputed.ch_names, roi_defs)
    if not roi_to_idx:
        return pd.DataFrame(), []

    from eeg_pipeline.utils.analysis.windowing import get_segment_masks

    windows = precomputed.windows
    masks = get_segment_masks(precomputed.times, windows, config)
    baseline_mask = masks.get("baseline")
    segments: Dict[str, Optional[np.ndarray]] = {
        "ramp": masks.get("ramp"),
        "plateau": masks.get("plateau"),
    }
    epsilon = float(get_feature_constant(config, "EPSILON_STD", 1e-12))

    if baseline_mask is None or not np.any(baseline_mask):
        return pd.DataFrame(), []

    records: List[Dict[str, float]] = []
    for ep_idx in range(precomputed.data.shape[0]):
        rec: Dict[str, float] = {}
        for band in bands:
            if band not in precomputed.band_data:
                continue
            power = precomputed.band_data[band].power[ep_idx]
            base_ch = np.nanmean(power[:, baseline_mask], axis=1)
            for roi_name, idxs in roi_to_idx.items():
                base = float(np.nanmean(base_ch[idxs]))
                base = base if np.isfinite(base) and base > epsilon else np.nan
                for seg, mask in segments.items():
                    if mask is None or not np.any(mask):
                        continue
                    seg_ch = np.nanmean(power[:, mask], axis=1)
                    seg_val = float(np.nanmean(seg_ch[idxs]))
                    if np.isfinite(base) and base > epsilon and np.isfinite(seg_val):
                        logratio = float(np.log10(seg_val / base))
                    else:
                        logratio = np.nan
                    rec[
                        NamingSchema.build(
                            "roi",
                            seg,
                            band,
                            "global",
                            f"{roi_name}_logratio_mean",
                        )
                    ] = logratio
        records.append(rec)

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []
    df = pd.DataFrame(records)
    return df, list(df.columns)


__all__ = ["extract_roi_features_from_precomputed"]
