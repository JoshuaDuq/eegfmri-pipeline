from pathlib import Path
from typing import List, Sequence
import logging

import numpy as np
import pandas as pd

###################################################################
# Feature Selection Utilities
###################################################################

SUPPORTED_BANDS = ("delta", "theta", "alpha", "beta", "gamma")
DEFAULT_BANDS = ("delta", "theta", "alpha", "beta", "gamma")


def select_direct_power_columns(columns: Sequence[str], bands: Sequence[str]) -> List[str]:
    pow_prefixes = tuple(f"pow_{band}_" for band in bands)
    baseline_prefixes = tuple(f"baseline_{band}_" for band in bands)
    selected = [col for col in columns if col.startswith(pow_prefixes) or col.startswith(baseline_prefixes)]
    return sorted(selected)


def select_roi_power_columns(columns: Sequence[str], bands: Sequence[str]) -> List[str]:
    keep: List[str] = []
    for col in columns:
        for band in bands:
            prefix = f"{band}_power_"
            if col.startswith(prefix):
                keep.append(col)
                break
    return sorted(keep)


def filter_zero_variance_features(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    logger: logging.Logger,
    tol: float = 1e-12,
) -> List[str]:
    if not feature_columns:
        return []

    variance = data.loc[:, feature_columns].var(axis=0, skipna=True, ddof=0)
    variance = variance.fillna(0.0)
    zero_cols = [col for col, var in variance.items() if float(var) <= tol]

    if zero_cols:
        preview = ", ".join(zero_cols[:20])
        if len(zero_cols) > 20:
            preview += f", ... (+{len(zero_cols) - 20})"
        logger.info(
            "Detected %d zero-variance feature(s); relying on per-fold transformers to drop them: %s",
            len(zero_cols),
            preview,
        )

    if zero_cols and len(zero_cols) == len(feature_columns):
        raise ValueError("All candidate feature columns exhibit zero variance; aborting.")

    return list(feature_columns)

