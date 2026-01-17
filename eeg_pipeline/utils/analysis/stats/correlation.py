"""
Correlation Statistics
======================

Correlation computation, partial correlations, and Fisher aggregation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import hyp2f1

from .base import (
    _safe_float,
    ensure_config,
    get_ci_level,
    get_config_value,
    get_min_samples_for_correlation,
)
from .fdr import fdr_bh
from .reliability import _apply_spearman_brown as _spearman_brown
from eeg_pipeline.utils.config.loader import get_fisher_z_clip_values


# Constants
_VALID_CORR_METHODS = {"spearman", "pearson"}
_MIN_SAMPLES_CORRELATION = 3
_MIN_SAMPLES_PSI = 5
_MIN_SAMPLES_BAYES = 4
_MIN_SAMPLES_RELIABILITY = 10
_EPSILON_STD = 1e-12
_EPSILON_STD_STRICT = 1e-10
_EPSILON_CORRELATION = 1e-12
_DEFAULT_PRIOR_WIDTH = 0.707
_DEFAULT_WINSORIZE_TRIM = 0.2
_DEFAULT_SHEPHERD_ALPHA = 0.05
_DEFAULT_PERCENTAGE_BEND_BETA = 0.2


def normalize_correlation_method(method: Optional[str], default: str = "spearman") -> str:
    """Normalize correlation method names to supported values."""
    if method is None:
        return default
    try:
        cleaned = str(method).strip().lower()
    except (AttributeError, TypeError):
        return default
    return cleaned if cleaned in _VALID_CORR_METHODS else default


def format_correlation_method_label(method: Optional[str], robust_method: Optional[str] = None) -> str:
    """Format the exact correlation method label for outputs."""
    base = normalize_correlation_method(method, default="spearman")
    label = f"{base}_{robust_method}" if robust_method else base
    return str(label).strip().lower().replace(" ", "_") or "unknown"


def compute_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "spearman",
) -> Tuple[float, float]:
    """
    Compute correlation coefficient and p-value.
    
    Returns (r, p).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < _MIN_SAMPLES_CORRELATION:
        return np.nan, np.nan

    x_v, y_v = x[valid], y[valid]

    if np.std(x_v) < _EPSILON_STD or np.std(y_v) < _EPSILON_STD:
        return np.nan, np.nan

    method = normalize_correlation_method(method, default="spearman")

    if method == "spearman":
        r, p = stats.spearmanr(x_v, y_v)
    else:
        r, p = stats.pearsonr(x_v, y_v)

    return float(r), float(p)


@dataclass
class CorrelationRecord:
    """Standard record for a single correlation result."""
    identifier: str
    band: str
    correlation: float
    p_value: float
    n_valid: int
    method: str
    ci_low: float = np.nan
    ci_high: float = np.nan
    p_perm: float = np.nan
    q_value: float = np.nan
    r_partial: float = np.nan
    p_partial: float = np.nan
    n_partial: int = 0
    p_partial_perm: float = np.nan
    r_partial_temp: float = np.nan
    p_partial_temp: float = np.nan
    n_partial_temp: int = 0
    p_partial_temp_perm: float = np.nan
    identifier_type: str = "channel"
    analysis_type: str = "power"
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_stats(cls, identifier: str, band: str, stats: Any, n_valid: int,
                   method: str, identifier_type: str = "channel",
                   analysis_type: str = "power", **extra) -> "CorrelationRecord":
        """Create from stats object with correlation attributes."""
        return cls(
            identifier=identifier, band=band,
            correlation=_safe_float(stats.correlation),
            p_value=_safe_float(stats.p_value), n_valid=n_valid, method=method,
            ci_low=_safe_float(stats.ci_low), ci_high=_safe_float(stats.ci_high),
            p_perm=_safe_float(stats.p_perm),
            r_partial=_safe_float(stats.r_partial),
            p_partial=_safe_float(stats.p_partial),
            n_partial=int(getattr(stats, 'n_partial', 0)),
            p_partial_perm=_safe_float(stats.p_partial_perm),
            r_partial_temp=_safe_float(stats.r_partial_temp),
            p_partial_temp=_safe_float(stats.p_partial_temp),
            n_partial_temp=int(getattr(stats, 'n_partial_temp', 0)),
            p_partial_temp_perm=_safe_float(stats.p_partial_temp_perm),
            identifier_type=identifier_type, analysis_type=analysis_type,
            extra_fields=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for DataFrame."""
        effect_size = self._interpret_effect_size()
        d = {
            self.identifier_type: self.identifier,
            "band": self.band,
            "r": self.correlation,
            "p": self.p_value,
            "n": self.n_valid,
            "method": self.method,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_perm": self.p_perm,
            "q": self.q_value,
            "analysis": self.analysis_type,
            "effect_size": effect_size,
        }
        
        if np.isfinite(self.r_partial):
            d["r_partial"] = self.r_partial
        if np.isfinite(self.p_partial):
            d["p_partial"] = self.p_partial
        if self.n_partial > 0:
            d["n_partial"] = self.n_partial
        if np.isfinite(self.r_partial_temp):
            d["r_partial_temp"] = self.r_partial_temp
        if np.isfinite(self.p_partial_temp):
            d["p_partial_temp"] = self.p_partial_temp
        
        d.update(self.extra_fields)
        return d

    def _interpret_effect_size(self) -> str:
        """Interpret correlation effect size using Cohen's conventions."""
        return interpret_correlation(self.correlation)

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05


def build_correlation_record(identifier: str, band: str, r: float, p: float, n: int,
                              method: str = "spearman", *, ci_low: float = np.nan,
                              ci_high: float = np.nan, p_perm: float = np.nan,
                              r_partial: float = np.nan, p_partial: float = np.nan,
                              n_partial: int = 0, p_partial_perm: float = np.nan,
                              r_partial_temp: float = np.nan, p_partial_temp: float = np.nan,
                              n_partial_temp: int = 0, p_partial_temp_perm: float = np.nan,
                              identifier_type: str = "channel", analysis_type: str = "power",
                              **extra) -> CorrelationRecord:
    """Build standardized correlation record."""
    return CorrelationRecord(
        identifier=identifier, band=band, correlation=_safe_float(r), p_value=_safe_float(p),
        n_valid=int(n), method=method, ci_low=_safe_float(ci_low), ci_high=_safe_float(ci_high),
        p_perm=_safe_float(p_perm), r_partial=_safe_float(r_partial), p_partial=_safe_float(p_partial),
        n_partial=int(n_partial), p_partial_perm=_safe_float(p_partial_perm),
        r_partial_temp=_safe_float(r_partial_temp), p_partial_temp=_safe_float(p_partial_temp),
        n_partial_temp=int(n_partial_temp), p_partial_temp_perm=_safe_float(p_partial_temp_perm),
        identifier_type=identifier_type, analysis_type=analysis_type, extra_fields=extra,
    )


def safe_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "spearman",
    min_samples: Optional[int] = None,
    robust_method: Optional[str] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float, int]:
    """
    Compute correlation with validation. Returns (r, p, n_valid).
    
    If robust_method is specified, uses robust correlation.
    Options: "percentage_bend", "winsorized", "shepherd"
    """
    if min_samples is None:
        config = ensure_config(config)
        min_samples = get_min_samples_for_correlation(config)
    
    method = normalize_correlation_method(method, default="spearman")
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size != y.size:
        return np.nan, np.nan, 0

    mask = np.isfinite(x) & np.isfinite(y)
    n_valid = int(mask.sum())
    
    if n_valid < min_samples:
        return np.nan, np.nan, n_valid

    x_clean, y_clean = x[mask], y[mask]
    
    if np.std(x_clean) < _EPSILON_STD or np.std(y_clean) < _EPSILON_STD:
        return np.nan, np.nan, n_valid

    try:
        if robust_method:
            r, p = compute_robust_correlation(x_clean, y_clean, method=robust_method)
        elif method == "spearman":
            r, p = stats.spearmanr(x_clean, y_clean, nan_policy="omit")
        else:
            r, p = stats.pearsonr(x_clean, y_clean)
            
        r_float = float(r) if np.isfinite(r) else np.nan
        p_float = float(p) if np.isfinite(p) else np.nan
        return r_float, p_float, n_valid
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return np.nan, np.nan, n_valid


def compute_pain_sensitivity_index(
    ratings: pd.Series,
    temperatures: pd.Series,
) -> pd.Series:
    """Compute pain sensitivity as residual from temperature-rating regression."""
    valid = ratings.notna() & temperatures.notna()
    psi = pd.Series(np.nan, index=ratings.index)

    if valid.sum() < _MIN_SAMPLES_PSI:
        return psi

    ratings_valid = ratings[valid].values
    temps_valid = temperatures[valid].values

    design_matrix = np.column_stack([np.ones(len(temps_valid)), temps_valid])
    try:
        beta = np.linalg.lstsq(design_matrix, ratings_valid, rcond=None)[0]
        predicted = design_matrix @ beta
        psi.loc[valid] = ratings_valid - predicted
    except np.linalg.LinAlgError:
        psi.loc[valid] = ratings_valid

    return psi


def _align_psi_inputs(
    features_df: pd.DataFrame,
    ratings: pd.Series,
    temperatures: pd.Series,
    min_samples: int,
    logger: Optional[logging.Logger],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]:
    """Align features, ratings, and temperatures for PSI computation."""
    if features_df is None or features_df.empty or ratings is None or temperatures is None:
        return None, None, None

    common_index = features_df.index.intersection(ratings.index).intersection(temperatures.index)

    if common_index.empty:
        if logger:
            logger.error("No overlapping samples across features, ratings, and temperatures for PSI")
        return None, None, None

    features_aligned = features_df.loc[common_index]
    ratings_aligned = ratings.loc[common_index]
    temps_aligned = temperatures.loc[common_index]

    valid_mask = ratings_aligned.notna() & temps_aligned.notna()
    if valid_mask.sum() < min_samples:
        if logger:
            logger.warning(
                "Insufficient valid samples for PSI after alignment "
                f"(found {valid_mask.sum()}, need >= {min_samples})"
            )
        return None, None, None

    return (
        features_aligned.loc[valid_mask],
        ratings_aligned.loc[valid_mask],
        temps_aligned.loc[valid_mask],
    )


def run_pain_sensitivity_correlations(
    features_df: pd.DataFrame,
    ratings: pd.Series,
    temperatures: pd.Series,
    method: str = "spearman",
    robust_method: Optional[str] = None,
    min_samples: int = 10,
    logger: Optional[logging.Logger] = None,
    config: Optional[Any] = None,
) -> pd.DataFrame:
    """Correlate features with pain sensitivity index."""
    method_label = format_correlation_method_label(method, robust_method)
    
    feat_aligned, ratings_aligned, temps_aligned = _align_psi_inputs(
        features_df, ratings, temperatures, min_samples, logger
    )
    if feat_aligned is None or ratings_aligned is None or temps_aligned is None:
        return pd.DataFrame()

    psi = compute_pain_sensitivity_index(ratings_aligned, temps_aligned)

    if psi.isna().all():
        if logger:
            logger.warning("Could not compute pain sensitivity index")
        return pd.DataFrame()

    valid_psi = psi.notna()
    if valid_psi.sum() < min_samples:
        if logger:
            logger.warning(
                "Insufficient PSI samples after masking "
                f"(found {valid_psi.sum()}, need >= {min_samples})"
            )
        return pd.DataFrame()

    psi_valid = psi.loc[valid_psi]
    feat_valid = feat_aligned.loc[valid_psi]

    records = []
    for col in feat_valid.columns:
        vals = pd.to_numeric(feat_valid[col], errors="coerce").values
        r, p, n = safe_correlation(
            vals,
            psi_valid.values,
            method,
            min_samples,
            robust_method=robust_method,
        )

        if np.isfinite(r):
            records.append({
                "feature": col,
                "r_psi": float(r),
                "p_psi": float(p),
                "n": n,
                "effect_interpretation": interpret_correlation(r),
                "method": method,
                "robust_method": robust_method,
                "method_label": method_label,
                "target": "pain_sensitivity",
            })

    if logger:
        n_sig = sum(1 for r in records if r["p_psi"] < 0.05)
        logger.info(f"Pain sensitivity: {len(records)} features, {n_sig} significant")

    out = pd.DataFrame(records)
    if out.empty:
        return out

    out["p_raw"] = pd.to_numeric(out.get("p_psi", np.nan), errors="coerce")
    out["p_primary"] = out["p_raw"]
    out["p_kind_primary"] = "p_psi"
    out["p_primary_source"] = "psi"
    
    alpha = float(get_config_value(config, "behavior_analysis.statistics.fdr_alpha", 0.05)) if config else 0.05
    try:
        out["p_fdr"] = fdr_bh(
            pd.to_numeric(out["p_primary"], errors="coerce").to_numpy(),
            alpha=alpha,
            config=config
        )
    except (ValueError, RuntimeError):
        pass
    
    return out


def align_groups_to_series(
    series: pd.Series,
    groups: Optional[Union[pd.Series, np.ndarray]],
) -> Optional[np.ndarray]:
    """Align group labels to a pandas Series index."""
    if groups is None:
        return None
    if isinstance(groups, pd.Series):
        missing = series.index.difference(groups.index)
        if not missing.empty:
            raise ValueError(f"Group labels missing for {len(missing)} samples")
        return groups.loc[series.index].to_numpy()
    arr = np.asarray(groups)
    if arr.size != len(series):
        raise ValueError("Group labels length does not match series length")
    return arr


def align_features_and_targets(
    df: pd.DataFrame,
    targets: pd.Series,
    min_samples: int,
    logger: logging.Logger,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Align feature dataframe and targets on shared index and drop missing targets."""
    if df is None or df.empty or targets is None or targets.empty:
        return None, None

    if not df.index.equals(targets.index):
        common_index = df.index.intersection(targets.index)
        if common_index.empty:
            logger.error("No overlapping samples between features and targets")
            return None, None
        df = df.loc[common_index]
        targets = targets.loc[common_index]

    valid_mask = targets.notna()
    if valid_mask.sum() < min_samples:
        logger.warning(
            "Insufficient valid samples after alignment "
            f"(found {valid_mask.sum()}, need >= {min_samples})"
        )
        return None, None

    return df.loc[valid_mask], targets.loc[valid_mask]


def fisher_z(
    r: Union[float, np.ndarray], 
    config: Optional[Any] = None, 
    logger: Optional[Any] = None
) -> Union[float, np.ndarray]:
    """Fisher z-transform of correlation coefficient(s).
    
    Supports both scalar and array inputs.
    
    Args:
        r: Correlation coefficient(s) to transform (scalar or array)
        config: Optional config object for clipping bounds (defaults to config values)
        logger: Optional logger for clipping warnings
    """
    clip_min, clip_max = get_fisher_z_clip_values(config)
    r_array = np.asarray(r)
    r_orig = r_array.copy()
    r_clipped = np.clip(r_array, clip_min, clip_max)
    
    if logger is not None:
        if np.any(r_clipped != r_orig):
            logger.debug(
                f"Fisher z: clipped r values from range [{r_orig.min():.6f}, {r_orig.max():.6f}] "
                f"to [{r_clipped.min():.6f}, {r_clipped.max():.6f}]"
            )
    
    result = np.arctanh(r_clipped)
    return result.item() if np.isscalar(r) else result


def inverse_fisher_z(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Inverse Fisher z-transform.
    
    Supports both scalar and array inputs.
    """
    z_array = np.asarray(z)
    result = np.tanh(z_array)
    return result.item() if np.isscalar(z) else result


def fisher_ci(
    r: float,
    n: int,
    config: Optional[Any] = None,
    ci_level: Optional[float] = None,
) -> Tuple[float, float]:
    """Compute Fisher-based CI for correlation.
    
    Parameters
    ----------
    r : float
        Correlation coefficient
    n : int
        Sample size
    config : Optional[Any]
        Configuration object (used if ci_level is None)
    ci_level : Optional[float]
        Explicit confidence level (e.g., 0.95 for 95% CI). 
        If None, uses config or defaults to 0.95.
        
    Returns
    -------
    Tuple[float, float]
        (ci_low, ci_high)
    """
    if ci_level is None:
        ci_level = get_ci_level(config)
    else:
        ci_level = float(ci_level)

    if n < 4 or not np.isfinite(r):
        return np.nan, np.nan

    from .base import get_z_critical_value
    
    z = fisher_z(r, config)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = get_z_critical_value(ci_level)

    z_lo = z - z_crit * se
    z_hi = z + z_crit * se

    return float(inverse_fisher_z(z_lo)), float(inverse_fisher_z(z_hi))


def fisher_aggregate(
    rs: List[float],
    config: Optional[Any] = None,
) -> Tuple[float, float, float, int]:
    """
    Aggregate correlations via Fisher z-transform.
    
    Returns (r_mean, ci_low, ci_high, n_valid).
    """
    ci_level = get_ci_level(config)
    rs_arr = np.asarray(rs, dtype=float)
    valid = np.isfinite(rs_arr)

    if np.sum(valid) == 0:
        return np.nan, np.nan, np.nan, 0

    rs_v = rs_arr[valid]
    zs = np.array([fisher_z(r, config) for r in rs_v])

    z_mean = np.mean(zs)
    r_mean = inverse_fisher_z(z_mean)

    if len(zs) > 1:
        from .base import get_z_critical_value
        se = np.std(zs, ddof=1) / np.sqrt(len(zs))
        z_crit = get_z_critical_value(ci_level)
        z_lo = z_mean - z_crit * se
        z_hi = z_mean + z_crit * se
        ci_lo = inverse_fisher_z(z_lo)
        ci_hi = inverse_fisher_z(z_hi)
    else:
        ci_lo = ci_hi = r_mean

    return float(r_mean), float(ci_lo), float(ci_hi), int(np.sum(valid))


def weighted_fisher_aggregate(
    rs: List[float],
    weights: List[float],
    config: Optional[Any] = None,
) -> Tuple[float, float, float, int]:
    """Weighted aggregation of correlations."""
    ci_level = get_ci_level(config)

    rs_arr = np.asarray(rs, dtype=float)
    ws_arr = np.asarray(weights, dtype=float)

    valid = np.isfinite(rs_arr) & np.isfinite(ws_arr) & (ws_arr > 0)
    if np.sum(valid) == 0:
        return np.nan, np.nan, np.nan, 0

    rs_valid = rs_arr[valid]
    ws_valid = ws_arr[valid]
    ws_normalized = ws_valid / ws_valid.sum()

    zs = np.array([fisher_z(r, config) for r in rs_valid])
    z_mean = np.sum(zs * ws_normalized)
    r_mean = inverse_fisher_z(z_mean)

    if len(zs) > 1:
        from .base import get_z_critical_value
        var_z = np.sum(ws_normalized * (zs - z_mean) ** 2)
        se = np.sqrt(var_z / len(zs))
        z_crit = get_z_critical_value(ci_level)
        ci_low = inverse_fisher_z(z_mean - z_crit * se)
        ci_high = inverse_fisher_z(z_mean + z_crit * se)
    else:
        ci_low = ci_high = r_mean

    return float(r_mean), float(ci_low), float(ci_high), int(np.sum(valid))


def fisher_z_transform_mean(r_values: np.ndarray, config: Optional[Any] = None) -> float:
    """Compute Fisher z-transformed mean of correlations.
    
    This is a convenience function that computes the mean correlation
    using Fisher z-transform. For full statistics including CIs, use
    fisher_aggregate instead.
    
    Parameters
    ----------
    r_values : np.ndarray
        Array of correlation coefficients
    config : Optional[Any]
        Configuration object for clipping bounds
        
    Returns
    -------
    float
        Fisher-aggregated mean correlation
    """
    clip_min, clip_max = get_fisher_z_clip_values(config)
    r_clipped = np.clip(r_values, clip_min, clip_max)
    z_scores = np.arctanh(r_clipped)
    z_mean = np.mean(z_scores)
    return float(np.tanh(z_mean))


def compute_correlation_ci_fisher(
    z_mean: float,
    se: float,
    ci_multiplier: float = 1.96,
) -> Tuple[float, float]:
    """Compute confidence interval from Fisher z mean and standard error.
    
    This is a general utility for computing CIs when you already have
    the Fisher z-transformed mean and its standard error.
    
    Parameters
    ----------
    z_mean : float
        Fisher z-transformed mean
    se : float
        Standard error of the Fisher z mean
    ci_multiplier : float
        Multiplier for CI (default 1.96 for 95% CI)
        
    Returns
    -------
    Tuple[float, float]
        (ci_low, ci_high) in correlation space
    """
    if not np.isfinite(se):
        return np.nan, np.nan
    
    z_low = z_mean - ci_multiplier * se
    z_high = z_mean + ci_multiplier * se
    ci_low = float(np.tanh(z_low))
    ci_high = float(np.tanh(z_high))
    
    return ci_low, ci_high


def joint_valid_mask(*arrays: Sequence, require_all: bool = True) -> np.ndarray:
    """Create joint validity mask across multiple arrays."""
    if not arrays:
        return np.array([], dtype=bool)

    masks = [np.isfinite(np.asarray(a)) for a in arrays]
    if require_all:
        return np.all(np.stack(masks), axis=0)
    return np.any(np.stack(masks), axis=0)


def compute_correlation_pvalue(r_values: List[float], config: Optional[Any] = None) -> float:
    """Compute p-value for aggregated correlation (one-sample t-test on z)."""
    rs = np.asarray(r_values, dtype=float)
    valid = np.isfinite(rs) & (np.abs(rs) < 1)

    if np.sum(valid) < 2:
        return np.nan

    zs = np.array([fisher_z(r, config) for r in rs[valid]])
    t_stat = np.mean(zs) / (np.std(zs, ddof=1) / np.sqrt(len(zs)))
    p = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=len(zs) - 1))
    return float(p)


def compute_bayes_factor_correlation(
    x: np.ndarray,
    y: np.ndarray,
    prior_width: float = _DEFAULT_PRIOR_WIDTH,
    method: str = "spearman",
) -> Tuple[float, str]:
    """
    Compute Bayes Factor for H1: r≠0 vs H0: r=0.
    
    Uses the Jeffreys-Zellner-Siow (JZS) prior approximation.
    
    Parameters
    ----------
    x, y : array-like
        Data arrays
    prior_width : float
        Width of the Cauchy prior on r (default: sqrt(2)/2 ≈ 0.707)
    method : str
        Correlation method ("spearman" or "pearson")
    
    Returns
    -------
    Tuple[float, str]
        (BF10, interpretation)
        BF10 > 1: evidence for H1 (correlation exists)
        BF10 < 1: evidence for H0 (no correlation)
        
    Interpretation thresholds (Jeffreys):
        BF < 1: Evidence for H0
        1-3: Anecdotal
        3-10: Moderate
        10-30: Strong
        30-100: Very strong
        >100: Extreme
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    valid = np.isfinite(x) & np.isfinite(y)
    n = int(np.sum(valid))
    
    if n < _MIN_SAMPLES_BAYES:
        return np.nan, "insufficient_data"
    
    x_valid, y_valid = x[valid], y[valid]
    
    method = normalize_correlation_method(method, default="spearman")
    if method == "spearman":
        r, _ = stats.spearmanr(x_valid, y_valid)
    else:
        r, _ = stats.pearsonr(x_valid, y_valid)
    
    if not np.isfinite(r) or np.abs(r) >= 1:
        return np.nan, "invalid_r"
    
    r_squared = r ** 2
    
    try:
        log_bf = (
            np.log(np.sqrt(2) / prior_width)
            + 0.5 * np.log(n - 1)
            + ((n - 1) / 2) * np.log(1 - r_squared)
            + np.log(hyp2f1(0.5, 0.5, (n + 1) / 2, r_squared))
        )
        bf10 = np.exp(log_bf)
    except (ValueError, OverflowError, RuntimeWarning):
        t_stat = r * np.sqrt((n - 2) / (1 - r_squared))
        bf10 = np.sqrt((n + 1) / (2 * np.pi)) * (1 + t_stat**2 / n) ** (-(n + 1) / 2)
    
    if bf10 < 1/100:
        interpretation = "extreme_H0"
    elif bf10 < 1/30:
        interpretation = "very_strong_H0"
    elif bf10 < 1/10:
        interpretation = "strong_H0"
    elif bf10 < 1/3:
        interpretation = "moderate_H0"
    elif bf10 < 1:
        interpretation = "anecdotal_H0"
    elif bf10 < 3:
        interpretation = "anecdotal_H1"
    elif bf10 < 10:
        interpretation = "moderate_H1"
    elif bf10 < 30:
        interpretation = "strong_H1"
    elif bf10 < 100:
        interpretation = "very_strong_H1"
    else:
        interpretation = "extreme_H1"
    
    return float(bf10), interpretation


def compute_robust_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "percentage_bend",
) -> Tuple[float, float]:
    """Compute robust correlation resistant to outliers.
    
    Parameters
    ----------
    x, y : array-like
        Data arrays
    method : str
        Robust method:
        - "percentage_bend": Percentage bend correlation (default)
        - "winsorized": Winsorized correlation (20% trimming)
        - "shepherd": Shepherd's pi correlation (removes bivariate outliers)
    
    Returns
    -------
    Tuple[float, float]
        (r, p_value)
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    valid = np.isfinite(x) & np.isfinite(y)
    n = int(np.sum(valid))
    
    if n < 4:
        return np.nan, np.nan
    
    x_v, y_v = x[valid], y[valid]
    
    if method == "percentage_bend":
        return _percentage_bend_correlation(x_v, y_v)
    elif method == "winsorized":
        return _winsorized_correlation(x_v, y_v)
    elif method == "shepherd":
        return _shepherd_correlation(x_v, y_v)
    else:
        return stats.spearmanr(x_v, y_v)


def _percentage_bend_correlation(
    x: np.ndarray,
    y: np.ndarray,
    beta: float = _DEFAULT_PERCENTAGE_BEND_BETA,
) -> Tuple[float, float]:
    """
    Percentage bend correlation (Wilcox, 1994).
    
    Downweights observations far from the median.
    """
    n = len(x)
    
    median_x, median_y = np.median(x), np.median(y)
    mad_x = np.median(np.abs(x - median_x))
    mad_y = np.median(np.abs(y - median_y))
    
    if mad_x < _EPSILON_STD or mad_y < _EPSILON_STD:
        return stats.spearmanr(x, y)
    
    omega_x = (beta * (n - 1) + 0.5) / n
    omega_y = (beta * (n - 1) + 0.5) / n
    
    crit_x = np.percentile(np.abs(x - median_x) / mad_x, 100 * (1 - beta))
    crit_y = np.percentile(np.abs(y - median_y) / mad_y, 100 * (1 - beta))
    
    x_bent = np.clip((x - median_x) / mad_x, -crit_x, crit_x)
    y_bent = np.clip((y - median_y) / mad_y, -crit_y, crit_y)
    
    if np.std(x_bent) < _EPSILON_STD or np.std(y_bent) < _EPSILON_STD:
        return np.nan, np.nan
    
    r, _ = stats.pearsonr(x_bent, y_bent)
    
    t_stat = r * np.sqrt((n - 2) / (1 - r**2 + _EPSILON_CORRELATION))
    p = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - 2))
    
    return float(r), float(p)


def _winsorized_correlation(
    x: np.ndarray,
    y: np.ndarray,
    trim: float = _DEFAULT_WINSORIZE_TRIM,
) -> Tuple[float, float]:
    """
    Winsorized correlation (replaces extreme values with percentiles).
    """
    n = len(x)
    k = int(trim * n)
    
    if k < 1:
        if np.std(x) < _EPSILON_STD or np.std(y) < _EPSILON_STD:
            return np.nan, np.nan
        return stats.pearsonr(x, y)
    
    def winsorize(arr):
        sorted_arr = np.sort(arr)
        lower, upper = sorted_arr[k], sorted_arr[-(k+1)]
        return np.clip(arr, lower, upper)
    
    x_winsorized = winsorize(x)
    y_winsorized = winsorize(y)
    
    if np.std(x_winsorized) < _EPSILON_STD or np.std(y_winsorized) < _EPSILON_STD:
        return np.nan, np.nan
    
    r, _ = stats.pearsonr(x_winsorized, y_winsorized)
    
    n_effective = n - 2 * k
    if n_effective < _MIN_SAMPLES_CORRELATION:
        return float(r), np.nan
    
    t_stat = r * np.sqrt((n_effective - 2) / (1 - r**2 + _EPSILON_CORRELATION))
    p = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n_effective - 2))
    
    return float(r), float(p)


def _shepherd_correlation(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = _DEFAULT_SHEPHERD_ALPHA,
) -> Tuple[float, float]:
    """
    Shepherd's pi correlation (removes bivariate outliers via bootstrap MAD).
    """
    n = len(x)
    
    median_x, median_y = np.median(x), np.median(y)
    mad_scale = 1.4826
    mad_x = np.median(np.abs(x - median_x)) * mad_scale
    mad_y = np.median(np.abs(y - median_y)) * mad_scale
    
    if mad_x < _EPSILON_STD or mad_y < _EPSILON_STD:
        return stats.spearmanr(x, y)
    
    x_standardized = (x - median_x) / mad_x
    y_standardized = (y - median_y) / mad_y
    
    distance = np.sqrt(x_standardized**2 + y_standardized**2)
    
    threshold = np.percentile(distance, 100 * (1 - alpha))
    inliers = distance <= threshold
    
    if np.sum(inliers) < _MIN_SAMPLES_BAYES:
        return stats.spearmanr(x, y)
    
    r, p = stats.spearmanr(x[inliers], y[inliers])
    return float(r), float(p)


def compute_loso_correlation_stability(
    feature_values: np.ndarray,
    target_values: np.ndarray,
    subject_ids: np.ndarray,
    method: str = "spearman",
) -> Tuple[float, float, float, List[float]]:
    """
    Compute leave-one-subject-out correlation stability.
    
    Checks if the correlation holds when each subject is left out.
    Low std = stable finding across subjects.
    
    Parameters
    ----------
    feature_values : array-like
        Feature values (one per trial)
    target_values : array-like
        Target values (one per trial)
    subject_ids : array-like
        Subject ID for each trial
    method : str
        Correlation method
    
    Returns
    -------
    Tuple[float, float, float, List[float]]
        (r_mean, r_std, stability_index, per_subject_r)
        stability_index = 1 - (r_std / abs(r_mean)) bounded [0, 1]
    """
    feature_values = np.asarray(feature_values)
    target_values = np.asarray(target_values)
    subject_ids = np.asarray(subject_ids)
    
    unique_subjects = np.unique(subject_ids)
    
    if len(unique_subjects) < 3:
        return np.nan, np.nan, np.nan, []
    
    r_values = []
    
    for subj in unique_subjects:
        mask = subject_ids != subj
        x_loo = feature_values[mask]
        y_loo = target_values[mask]
        
        r, _ = compute_correlation(x_loo, y_loo, method)
        if np.isfinite(r):
            r_values.append(r)
    
    if len(r_values) < 2:
        return np.nan, np.nan, np.nan, r_values
    
    r_mean = float(np.mean(r_values))
    r_std = float(np.std(r_values, ddof=1))
    
    if abs(r_mean) > 1e-6:
        stability = max(0.0, 1.0 - (r_std / abs(r_mean)))
    else:
        stability = 0.0 if r_std > 0.1 else 1.0
    
    return r_mean, r_std, float(stability), r_values


def compute_correlation_reliability(
    feature_values: np.ndarray,
    target_values: np.ndarray,
    method: str = "split_half",
    n_iterations: int = 100,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute reliability of feature-target correlation.
    
    Parameters
    ----------
    feature_values, target_values : array-like
        Data arrays
    method : str
        "split_half": Random split-half with Spearman-Brown correction
        "odd_even": Odd/even split
    n_iterations : int
        Number of random splits (for split_half method)
    seed : int
        Random seed
    
    Returns
    -------
    Tuple[float, float, float]
        (reliability, ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    feature_values = np.asarray(feature_values)
    target_values = np.asarray(target_values)
    
    valid = np.isfinite(feature_values) & np.isfinite(target_values)
    n = int(np.sum(valid))
    
    if n < _MIN_SAMPLES_RELIABILITY:
        return np.nan, np.nan, np.nan
    
    x = feature_values[valid]
    y = target_values[valid]
    
    if method == "odd_even":
        r1, _ = stats.spearmanr(x[::2], y[::2])
        r2, _ = stats.spearmanr(x[1::2], y[1::2])
        r_half = np.corrcoef([r1, r2])[0, 1] if np.isfinite(r1) and np.isfinite(r2) else np.nan
        reliability = _spearman_brown(r_half)
        return reliability, np.nan, np.nan
    
    reliabilities = []
    indices = np.arange(n)
    
    for _ in range(n_iterations):
        rng.shuffle(indices)
        half = n // 2
        idx1, idx2 = indices[:half], indices[half:2*half]
        
        r1, _ = stats.spearmanr(x[idx1], y[idx1])
        r2, _ = stats.spearmanr(x[idx2], y[idx2])
        
        if np.isfinite(r1) and np.isfinite(r2):
            r_half = np.corrcoef([r1, r2])[0, 1]
            if np.isfinite(r_half):
                reliabilities.append(_spearman_brown(r_half))
    
    if not reliabilities:
        return np.nan, np.nan, np.nan
    
    reliability = float(np.mean(reliabilities))
    ci_low = float(np.percentile(reliabilities, 2.5))
    ci_high = float(np.percentile(reliabilities, 97.5))
    
    return reliability, ci_low, ci_high


def save_correlation_results(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    sep: str = "\t",
    index: bool = False,
) -> None:
    """Save correlation results to file."""
    if df.empty:
        return
    
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(path, sep=sep, index=index, float_format="%.6f")


EFFECT_SIZE_BENCHMARKS = {
    "negligible": (0.0, 0.2),
    "small": (0.2, 0.5),
    "medium": (0.5, 0.8),
    "large": (0.8, float("inf")),
}

CORRELATION_BENCHMARKS = {
    "negligible": (0.0, 0.1),
    "small": (0.1, 0.3),
    "medium": (0.3, 0.5),
    "large": (0.5, float("inf")),
}


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    for label, (lo, hi) in EFFECT_SIZE_BENCHMARKS.items():
        if lo <= d_abs < hi:
            return label
    return "unknown"


def interpret_correlation(r: float) -> str:
    """Interpret correlation magnitude."""
    r_abs = abs(r)
    for label, (lo, hi) in CORRELATION_BENCHMARKS.items():
        if lo <= r_abs < hi:
            return label
    return "unknown"


def correlate_single_feature(
    feature_values: np.ndarray,
    target_values: np.ndarray,
    temperature: Optional[np.ndarray],
    trial_order: Optional[np.ndarray],
    method: str,
    min_samples: int,
) -> Tuple[float, float, float, float, float, float, float, float, int]:
    """Compute all correlations for a single feature.
    
    Returns:
        (r_raw, p_raw, r_partial_temp, p_partial_temp, r_partial_order, 
         p_partial_order, r_partial_full, p_partial_full, n_valid)
    """
    valid = np.isfinite(feature_values) & np.isfinite(target_values)
    if temperature is not None:
        valid &= np.isfinite(temperature)
    if trial_order is not None:
        valid &= np.isfinite(trial_order)
    
    n_valid = int(valid.sum())
    if n_valid < min_samples:
        return (np.nan,) * 8 + (0,)
    
    x, y = feature_values[valid], target_values[valid]
    
    if np.std(x) < _EPSILON_STD_STRICT or np.std(y) < _EPSILON_STD_STRICT:
        return (np.nan,) * 8 + (n_valid,)
    
    r_raw, p_raw = compute_correlation(x, y, method)
    
    r_pt = p_pt = r_po = p_po = r_pf = p_pf = np.nan
    
    from .partial import partial_corr_xy_given_Z
    
    if temperature is not None:
        temp_df = pd.DataFrame({"temp": temperature[valid]})
        r_pt, p_pt, _ = partial_corr_xy_given_Z(pd.Series(x), pd.Series(y), temp_df, method)
    
    if trial_order is not None:
        order_df = pd.DataFrame({"order": trial_order[valid]})
        r_po, p_po, _ = partial_corr_xy_given_Z(pd.Series(x), pd.Series(y), order_df, method)
    
    if temperature is not None and trial_order is not None:
        full_df = pd.DataFrame({"temp": temperature[valid], "order": trial_order[valid]})
        r_pf, p_pf, _ = partial_corr_xy_given_Z(pd.Series(x), pd.Series(y), full_df, method)
    elif temperature is not None:
        r_pf, p_pf = r_pt, p_pt
    elif trial_order is not None:
        r_pf, p_pf = r_po, p_po
    
    return r_raw, p_raw, r_pt, p_pt, r_po, p_po, r_pf, p_pf, n_valid


def compute_batch_correlations(
    feature_columns: List[str],
    feature_df: pd.DataFrame,
    target_arr: np.ndarray,
    method: str = "spearman",
    min_samples: int = 3,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """Compute correlations for all features using vectorized operations.
    
    Much faster than per-feature loops for large feature sets.
    Computes raw correlations only (no partial correlations or reliability).
    """
    import warnings
    
    n_features = len(feature_columns)
    if n_features == 0:
        return []
    
    if logger:
        logger.info(f"Correlations: computing {n_features} features (vectorized batch mode)")
    
    data_matrix = feature_df[feature_columns].to_numpy(dtype=np.float64, na_value=np.nan)
    target = np.asarray(target_arr, dtype=np.float64).ravel()
    
    n_samples = len(target)
    if data_matrix.shape[0] != n_samples:
        raise ValueError(f"Feature matrix rows ({data_matrix.shape[0]}) != target length ({n_samples})")
    
    target_finite = np.isfinite(target)
    feature_finite = np.isfinite(data_matrix)
    valid_mask = feature_finite & target_finite[:, np.newaxis]
    
    n_valid_per_feature = valid_mask.sum(axis=0)
    
    method_normalized = normalize_correlation_method(method, default="spearman")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        if method_normalized == "spearman":
            r_values, p_values = _compute_batch_spearman(
                data_matrix, target, valid_mask, n_valid_per_feature, min_samples
            )
        else:
            r_values, p_values = _compute_batch_pearson(
                data_matrix, target, valid_mask, n_valid_per_feature, min_samples
            )
    
    results: List[Dict[str, Any]] = []
    
    for i, col in enumerate(feature_columns):
        r = r_values[i]
        p = p_values[i]
        n = int(n_valid_per_feature[i])
        
        if not np.isfinite(r):
            continue
        
        band = _extract_band_from_column(col)
        is_change = "_change_" in col
        effect_interp = interpret_correlation(r)
        
        results.append({
            "feature": col,
            "band": band,
            "r_raw": float(r),
            "p_raw": float(p),
            "n": n,
            "r_partial_temp": np.nan,
            "p_partial_temp": np.nan,
            "r_partial_order": np.nan,
            "p_partial_order": np.nan,
            "r_partial_full": np.nan,
            "p_partial_full": np.nan,
            "effect_interpretation": effect_interp,
            "reliability": np.nan,
            "is_change_score": is_change,
            "method": method_normalized,
            "r_primary": float(r),
            "p_primary": float(p),
        })
    
    if logger:
        logger.info(f"Correlations batch complete: {len(results)} valid features")
    
    return results


def _extract_band_from_column(column_name: str) -> str:
    """Extract frequency band name from column name."""
    frequency_bands = ["delta", "theta", "alpha", "beta", "gamma"]
    column_lower = column_name.lower()
    for band in frequency_bands:
        if band in column_lower:
            return band
    return "broadband"


def _compute_batch_spearman(
    data_matrix: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    n_valid: np.ndarray,
    min_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Spearman correlations for all features vectorized."""
    n_features = data_matrix.shape[1]
    r_values = np.full(n_features, np.nan)
    p_values = np.full(n_features, np.nan)
    
    all_valid = valid_mask.all(axis=0)
    
    if all_valid.any():
        valid_indices = np.where(all_valid)[0]
        valid_features = data_matrix[:, valid_indices]
        
        target_ranks = stats.rankdata(target)
        feature_ranks = np.apply_along_axis(stats.rankdata, 0, valid_features)
        
        r_batch, p_batch = _pearson_matrix_vector(feature_ranks, target_ranks)
        
        r_values[valid_indices] = r_batch
        p_values[valid_indices] = p_batch
    
    partial_valid = ~all_valid & (n_valid >= min_samples)
    if partial_valid.any():
        partial_indices = np.where(partial_valid)[0]
        for idx in partial_indices:
            mask = valid_mask[:, idx]
            x = data_matrix[mask, idx]
            y = target[mask]
            
            if np.std(x) < _EPSILON_STD or np.std(y) < _EPSILON_STD:
                continue
            
            r, p = stats.spearmanr(x, y)
            r_values[idx] = r
            p_values[idx] = p
    
    return r_values, p_values


def _compute_batch_pearson(
    data_matrix: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    n_valid: np.ndarray,
    min_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Pearson correlations for all features vectorized."""
    n_features = data_matrix.shape[1]
    r_values = np.full(n_features, np.nan)
    p_values = np.full(n_features, np.nan)
    
    all_valid = valid_mask.all(axis=0)
    
    if all_valid.any():
        valid_indices = np.where(all_valid)[0]
        valid_features = data_matrix[:, valid_indices]
        
        r_batch, p_batch = _pearson_matrix_vector(valid_features, target)
        
        r_values[valid_indices] = r_batch
        p_values[valid_indices] = p_batch
    
    partial_valid = ~all_valid & (n_valid >= min_samples)
    if partial_valid.any():
        partial_indices = np.where(partial_valid)[0]
        for idx in partial_indices:
            mask = valid_mask[:, idx]
            x = data_matrix[mask, idx]
            y = target[mask]
            
            if np.std(x) < _EPSILON_STD or np.std(y) < _EPSILON_STD:
                continue
            
            r, p = stats.pearsonr(x, y)
            r_values[idx] = r
            p_values[idx] = p
    
    return r_values, p_values


def _pearson_matrix_vector(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Pearson correlation between each column of X and vector y.
    
    Returns (r_values, p_values) arrays.
    """
    n = X.shape[0]
    
    X_centered = X - X.mean(axis=0)
    y_centered = y - y.mean()
    
    X_std = X.std(axis=0, ddof=0)
    y_std = y.std(ddof=0)
    
    valid_std = (X_std > _EPSILON_STD) & (y_std > _EPSILON_STD)
    
    r_values = np.full(X.shape[1], np.nan)
    
    if valid_std.any():
        cov_xy = (X_centered[:, valid_std] * y_centered[:, np.newaxis]).sum(axis=0) / n
        r_values[valid_std] = cov_xy / (X_std[valid_std] * y_std)
    
    df = n - 2
    with np.errstate(divide='ignore', invalid='ignore'):
        t_values = r_values * np.sqrt(df) / np.sqrt(1 - r_values**2)
    
    p_values = 2 * stats.t.sf(np.abs(t_values), df)
    
    return r_values, p_values


def compute_correlation_stats(
    x: pd.Series,
    y: pd.Series,
    method_code: str,
    bootstrap_ci: int,
    rng: Optional[np.random.Generator],
    min_samples: int = 3,
) -> Tuple[float, float, int, Tuple[float, float]]:
    """Compute correlation with optional bootstrap CI.
    
    This is a convenience function that computes correlation statistics
    including optional bootstrap confidence intervals.
    
    Parameters
    ----------
    x, y : pd.Series
        Input series to correlate
    method_code : str
        Correlation method ('pearson' or 'spearman')
    bootstrap_ci : int
        Number of bootstrap iterations (0 to skip)
    rng : Optional[np.random.Generator]
        Random number generator
    min_samples : int
        Minimum number of samples required
        
    Returns
    -------
    Tuple[float, float, int, Tuple[float, float]]
        (correlation, p_value, n_effective, (ci_low, ci_high))
    """
    from .bootstrap import bootstrap_corr_ci
    
    valid_mask = np.isfinite(x) & np.isfinite(y)
    n_effective = int(valid_mask.sum())
    
    if n_effective < min_samples:
        return np.nan, np.nan, n_effective, (np.nan, np.nan)
    
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    correlation, p_value = compute_correlation(x_valid, y_valid, method_code)
    
    if bootstrap_ci > 0:
        confidence_interval = bootstrap_corr_ci(
            x_valid,
            y_valid,
            method_code,
            n_boot=bootstrap_ci,
            rng=rng
        )
    else:
        confidence_interval = (np.nan, np.nan)
    
    return float(correlation), float(p_value), n_effective, confidence_interval