from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from eeg_pipeline.utils.analysis.features.metadata import NamingSchema
from eeg_pipeline.utils.config.loader import get_config_value


def _process_asymmetry_epoch(
    ep_idx: int,
    band_data: Dict[str, Any],
    valid_pairs: List[Tuple[str, str, int, int]],
    active_mask: np.ndarray,
    segment_label: str,
) -> Dict[str, float]:
    record: Dict[str, float] = {}
    for band, bd in band_data.items():
        power = bd.power[ep_idx]
        p_active = power[:, active_mask] if not isinstance(active_mask, slice) else power
        p_mean = np.nanmean(p_active, axis=1)

        for l_name, r_name, l_idx, r_idx in valid_pairs:
            pl, pr = p_mean[l_idx], p_mean[r_idx]
            denom = pr + pl
            asym = (pr - pl) / denom if denom > 1e-12 else 0.0
            pair = f"{l_name}-{r_name}"
            record[
                NamingSchema.build(
                    "asymmetry",
                    segment_label,
                    band,
                    "chpair",
                    "index",
                    channel_pair=pair,
                )
            ] = float(asym)
            if pr > 0 and pl > 0:
                record[
                    NamingSchema.build(
                        "asymmetry",
                        segment_label,
                        band,
                        "chpair",
                        "logdiff",
                        channel_pair=pair,
                    )
                ] = float(np.log(pr) - np.log(pl))
    return record


def extract_asymmetry_from_precomputed(
    precomputed: Any,
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    if not precomputed.band_data:
        return pd.DataFrame(), []

    default_pairs = [("F3", "F4"), ("F7", "F8"), ("C3", "C4"), ("P3", "P4"), ("O1", "O2")]
    pairs_cfg = get_config_value(precomputed.config, "feature_engineering.asymmetry.channel_pairs", None)
    pairs: List[Tuple[str, str]] = []
    if isinstance(pairs_cfg, list):
        for entry in pairs_cfg:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                left, right = entry[0], entry[1]
                if isinstance(left, str) and isinstance(right, str):
                    pairs.append((left, right))
    if not pairs:
        pairs = default_pairs

    segment_label = str(
        get_config_value(precomputed.config, "feature_engineering.asymmetry.segment_label", "plateau")
    )

    ch_map = {name: i for i, name in enumerate(precomputed.ch_names)}
    valid_pairs = [(l, r, ch_map[l], ch_map[r]) for l, r in pairs if l in ch_map and r in ch_map]
    if not valid_pairs:
        return pd.DataFrame(), []

    mask = None
    if precomputed.windows is not None:
        plateau_mask = getattr(precomputed.windows, "plateau_mask", None)
        if isinstance(plateau_mask, np.ndarray) and np.any(plateau_mask):
            mask = plateau_mask
        else:
            active_mask = getattr(precomputed.windows, "active_mask", None)
            if isinstance(active_mask, np.ndarray) and np.any(active_mask):
                mask = active_mask
    if mask is None:
        mask = slice(None)

    n_epochs = precomputed.data.shape[0]
    if n_jobs != 1:
        records = Parallel(n_jobs=n_jobs)(
            delayed(_process_asymmetry_epoch)(
                ep_idx,
                precomputed.band_data,
                valid_pairs,
                mask,
                segment_label,
            )
            for ep_idx in range(n_epochs)
        )
    else:
        records = [
            _process_asymmetry_epoch(ep_idx, precomputed.band_data, valid_pairs, mask, segment_label)
            for ep_idx in range(n_epochs)
        ]

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []

    df = pd.DataFrame(records)
    return df, list(df.columns)
