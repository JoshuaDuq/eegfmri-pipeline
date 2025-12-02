"""
Condition Comparison Analysis
==============================

Compare EEG features between pain and non-pain conditions.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from scipy import stats

from eeg_pipeline.utils.io.general import get_pain_column_from_config
from eeg_pipeline.utils.analysis.stats import hedges_g, fdr_bh
from eeg_pipeline.analysis.behavior.correlations import interpret_effect_size
from eeg_pipeline.analysis.behavior.parallel import parallel_condition_effects, get_n_jobs


def split_by_condition(
    events_df: pd.DataFrame,
    config: Any,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Split trials into pain and non-pain conditions.
    
    Returns:
        pain_mask, nonpain_mask, n_pain, n_nonpain
    """
    pain_col = get_pain_column_from_config(config, events_df)
    
    if pain_col is None or pain_col not in events_df.columns:
        logger.error("Pain column not found in events")
        return np.array([]), np.array([]), 0, 0
    
    pain_series = pd.to_numeric(events_df[pain_col], errors="coerce")
    pain_mask = (pain_series == 1).values
    nonpain_mask = (pain_series == 0).values
    
    n_pain = int(pain_mask.sum())
    n_nonpain = int(nonpain_mask.sum())
    
    logger.info(f"Condition split: {n_pain} pain, {n_nonpain} non-pain trials")
    
    return pain_mask, nonpain_mask, n_pain, n_nonpain


def compute_condition_effects(
    features_df: pd.DataFrame,
    pain_mask: np.ndarray,
    nonpain_mask: np.ndarray,
    min_samples: int = 5,
    fdr_alpha: float = 0.05,
    logger: Optional[logging.Logger] = None,
    n_jobs: int = -1,
    config: Optional[Any] = None,
) -> pd.DataFrame:
    """Compute effect sizes for pain vs non-pain comparison.
    
    Returns DataFrame with:
    - mean_pain, mean_nonpain
    - hedges_g (effect size)
    - t_statistic, p_value
    - q_value (FDR corrected)
    """
    n_jobs_actual = get_n_jobs(config, n_jobs)
    
    if logger:
        logger.debug(f"Computing condition effects for {len(features_df.columns)} features (n_jobs={n_jobs_actual})")
    
    # Parallel computation of condition effects
    feature_columns = list(features_df.columns)
    records = parallel_condition_effects(
        feature_columns=feature_columns,
        features_df=features_df,
        pain_mask=pain_mask,
        nonpain_mask=nonpain_mask,
        min_samples=min_samples,
        n_jobs=n_jobs_actual,
    )
    
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    # FDR correction
    df["q_value"] = fdr_bh(df["p_value"].values)
    df["significant_fdr"] = df["q_value"] < fdr_alpha
    
    # Sort by effect size
    df = df.sort_values("hedges_g", key=abs, ascending=False)
    
    if logger:
        n_sig = df["significant_fdr"].sum()
        n_large = (df["hedges_g"].abs() >= 0.8).sum()
        logger.info(f"Condition effects: {n_sig}/{len(df)} FDR significant, {n_large} large effects")
    
    return df
