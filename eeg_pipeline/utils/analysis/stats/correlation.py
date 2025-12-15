"""
Correlation Statistics
======================

Correlation computation, partial correlations, and Fisher aggregation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from .base import get_ci_level, get_config_value
from eeg_pipeline.utils.config.loader import get_fisher_z_clip_values

from pathlib import Path


def get_correlation_method(use_spearman: bool) -> str:
    """Return correlation method name."""
    return "spearman" if use_spearman else "pearson"


def compute_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: Union[str, bool] = "spearman",
) -> Tuple[float, float]:
    """
    Compute correlation coefficient and p-value.
    
    Returns (r, p).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
        return np.nan, np.nan

    x_v, y_v = x[valid], y[valid]

    if np.std(x_v) < 1e-12 or np.std(y_v) < 1e-12:
        return np.nan, np.nan

    if isinstance(method, bool):
        method = "spearman" if method else "pearson"
    if not isinstance(method, str):
        raise TypeError("method must be a string ('spearman'/'pearson')")

    method = method.lower().strip()
    if method not in {"spearman", "pearson"}:
        raise ValueError("method must be 'spearman' or 'pearson'")

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
        """Create from CorrelationStats object."""
        def safe_float(v): return float(v) if np.isfinite(v) else np.nan
        return cls(
            identifier=identifier, band=band,
            correlation=safe_float(stats.correlation),
            p_value=safe_float(stats.p_value), n_valid=n_valid, method=method,
            ci_low=safe_float(stats.ci_low), ci_high=safe_float(stats.ci_high),
            p_perm=safe_float(stats.p_perm),
            r_partial=safe_float(stats.r_partial),
            p_partial=safe_float(stats.p_partial),
            n_partial=int(getattr(stats, 'n_partial', 0)),
            p_partial_perm=safe_float(stats.p_partial_perm),
            r_partial_temp=safe_float(stats.r_partial_temp),
            p_partial_temp=safe_float(stats.p_partial_temp),
            n_partial_temp=int(getattr(stats, 'n_partial_temp', 0)),
            p_partial_temp_perm=safe_float(stats.p_partial_temp_perm),
            identifier_type=identifier_type, analysis_type=analysis_type,
            extra_fields=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for DataFrame."""
        d = {
            self.identifier_type: self.identifier, "band": self.band,
            "r": self.correlation, "p": self.p_value, "n": self.n_valid,
            "method": self.method, "ci_low": self.ci_low, "ci_high": self.ci_high,
            "p_perm": self.p_perm, "q": self.q_value, "analysis": self.analysis_type,
        }
        if np.isfinite(self.r_partial): d["r_partial"] = self.r_partial
        if np.isfinite(self.p_partial): d["p_partial"] = self.p_partial
        if self.n_partial > 0: d["n_partial"] = self.n_partial
        if np.isfinite(self.r_partial_temp):
            d["r_partial_given_temp"] = self.r_partial_temp
            d["r_partial_temp"] = self.r_partial_temp
        if np.isfinite(self.p_partial_temp):
            d["p_partial_given_temp"] = self.p_partial_temp
            d["p_partial_temp"] = self.p_partial_temp
        d.update(self.extra_fields)
        return d

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
    def sf(v): return float(v) if np.isfinite(v) else np.nan
    return CorrelationRecord(
        identifier=identifier, band=band, correlation=sf(r), p_value=sf(p),
        n_valid=int(n), method=method, ci_low=sf(ci_low), ci_high=sf(ci_high),
        p_perm=sf(p_perm), r_partial=sf(r_partial), p_partial=sf(p_partial),
        n_partial=int(n_partial), p_partial_perm=sf(p_partial_perm),
        r_partial_temp=sf(r_partial_temp), p_partial_temp=sf(p_partial_temp),
        n_partial_temp=int(n_partial_temp), p_partial_temp_perm=sf(p_partial_temp_perm),
        identifier_type=identifier_type, analysis_type=analysis_type, extra_fields=extra,
    )


def correlate_features_loop(
    feature_df: pd.DataFrame,
    target_values: Union[pd.Series, np.ndarray],
    method: str = "spearman",
    min_samples: Optional[int] = None,
    logger: Optional[Any] = None,
    condition_mask: Optional[np.ndarray] = None,
    identifier_type: str = "feature",
    analysis_type: str = "unknown",
    feature_classifier: Optional[Any] = None,
    robust_method: Optional[str] = None,
    config: Optional[Any] = None,
    n_bootstrap: int = 0,
    n_permutations: int = 0,
    rng: Optional[np.random.Generator] = None,
    groups: Optional[np.ndarray] = None,
) -> Tuple[List[CorrelationRecord], pd.DataFrame]:
    """Correlate all features with target values."""
    if min_samples is None:
        from .base import ensure_config
        config = ensure_config(config)
        min_samples = int(get_config_value(config, "statistics.constants.min_samples_for_correlation", 5))
    
    if feature_df.empty:
        return [], pd.DataFrame()

    target_arr = target_values.values if isinstance(target_values, pd.Series) else np.asarray(target_values)
    if condition_mask is not None:
        if hasattr(condition_mask, 'dtype') and condition_mask.dtype == bool:
            idx = np.where(condition_mask)[0]
        else:
            idx = condition_mask
        feature_df = feature_df.iloc[idx]
        target_arr = target_arr[idx]

    n_f, n_t = len(feature_df), len(target_arr)
    if n_f != n_t:
        n_use = min(n_f, n_t)
        feature_df, target_arr = feature_df.iloc[:n_use], target_arr[:n_use]

    rng = rng or np.random.default_rng()
    ci_level = get_ci_level(config)

    records = []
    for col in feature_df.columns:
        vals = pd.to_numeric(feature_df[col], errors="coerce").to_numpy()
        if feature_classifier:
            ft, subtype, meta = feature_classifier(col)
            ident = meta.get("identifier", col)
            band = meta.get("band", "N/A")
        else:
            ft, ident, band = analysis_type, col, "N/A"

        valid_mask = np.isfinite(vals) & np.isfinite(target_arr)
        n_valid = int(valid_mask.sum())

        r, p, n = safe_correlation(vals, target_arr, method, min_samples, robust_method=robust_method)
        ci_low = ci_high = p_perm = np.nan

        if np.isfinite(r) and n_valid >= (min_samples or 0):
            if n_bootstrap and n_bootstrap > 0:
                from .eeg_stats import compute_bootstrap_ci

                ci_low, ci_high = compute_bootstrap_ci(
                    vals[valid_mask],
                    target_arr[valid_mask],
                    n_bootstrap=n_bootstrap,
                    ci_level=ci_level,
                    method=method,
                    rng=rng,
                )

            if n_permutations and n_permutations > 0:
                from .permutation import compute_permutation_pvalues

                x_series = pd.Series(vals[valid_mask])
                y_series = pd.Series(target_arr[valid_mask])
                p_perm, _, _ = compute_permutation_pvalues(
                    x_series,
                    y_series,
                    covariates_df=None,
                    temp_series=None,
                    method=method,
                    n_perm=n_permutations,
                    n_eff=n_valid,
                    rng=rng,
                    groups=groups,
                )

        if np.isfinite(r):
            records.append(build_correlation_record(
                ident,
                band,
                r,
                p,
                n,
                method,
                identifier_type=identifier_type,
                analysis_type=ft,
                ci_low=ci_low,
                ci_high=ci_high,
                p_perm=p_perm,
            ))

    if logger:
        logger.info(f"  {len(records)} features, {sum(1 for r in records if r.is_significant)} sig")
    return records, pd.DataFrame([r.to_dict() for r in records]) if records else pd.DataFrame()


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
        from .base import ensure_config
        config = ensure_config(config)
        min_samples = int(get_config_value(config, "statistics.constants.min_samples_for_correlation", 5))
    
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size != y.size:
        return np.nan, np.nan, 0

    mask = np.isfinite(x) & np.isfinite(y)
    n_valid = int(mask.sum())
    
    if n_valid < min_samples:
        return np.nan, np.nan, n_valid

    x_c, y_c = x[mask], y[mask]
    
    if np.std(x_c) == 0 or np.std(y_c) == 0:
        return np.nan, np.nan, n_valid

    try:
        if robust_method:
            r, p = compute_robust_correlation(x_c, y_c, method=robust_method)
        elif method == "spearman":
            r, p = stats.spearmanr(x_c, y_c, nan_policy="omit")
        else:
            r, p = stats.pearsonr(x_c, y_c)
            
        return (float(r) if np.isfinite(r) else np.nan,
                float(p) if np.isfinite(p) else np.nan, n_valid)
    except Exception:
        return np.nan, np.nan, n_valid


###################################################################
# Pain Sensitivity Helpers
###################################################################


def compute_pain_sensitivity_index(
    ratings: pd.Series,
    temperatures: pd.Series,
) -> pd.Series:
    """Compute pain sensitivity as residual from temperature-rating regression."""
    valid = ratings.notna() & temperatures.notna()
    psi = pd.Series(np.nan, index=ratings.index)

    if valid.sum() < 5:
        return psi

    r_valid = ratings[valid].values
    t_valid = temperatures[valid].values

    X = np.column_stack([np.ones(len(t_valid)), t_valid])
    try:
        beta = np.linalg.lstsq(X, r_valid, rcond=None)[0]
        predicted = X @ beta
        psi.loc[valid] = r_valid - predicted
    except np.linalg.LinAlgError:
        psi.loc[valid] = r_valid

    return psi


def compute_change_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Compute plateau - baseline change for matching feature pairs."""
    baseline_cols = [c for c in features_df.columns if "_baseline_" in c]

    change_data: Dict[str, np.ndarray] = {}
    for bl_col in baseline_cols:
        pl_col = bl_col.replace("_baseline_", "_plateau_")
        if pl_col in features_df.columns:
            bl_vals = features_df[bl_col].values
            pl_vals = features_df[pl_col].values

            if bl_vals.ndim != 1 or pl_vals.ndim != 1:
                continue

            change_col = bl_col.replace("_baseline_", "_change_")
            change_data[change_col] = pl_vals - bl_vals

    if not change_data:
        return pd.DataFrame(index=features_df.index)

    return pd.DataFrame(change_data, index=features_df.index)


@dataclass
class CorrelationResult:
    """Single feature correlation result (legacy structure)."""

    feature: str
    band: str
    r_raw: float
    p_raw: float
    n: int
    r_partial_temp: float
    p_partial_temp: float
    r_partial_order: float
    p_partial_order: float
    r_partial_full: float
    p_partial_full: float
    effect_interpretation: str
    reliability: float
    is_change_score: bool
    method: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "band": self.band,
            "r_raw": self.r_raw,
            "p_raw": self.p_raw,
            "n": self.n,
            "r_partial_temp": self.r_partial_temp,
            "p_partial_temp": self.p_partial_temp,
            "r_partial_order": self.r_partial_order,
            "p_partial_order": self.p_partial_order,
            "r_partial_full": self.r_partial_full,
            "p_partial_full": self.p_partial_full,
            "effect_interpretation": self.effect_interpretation,
            "reliability": self.reliability,
            "is_change_score": self.is_change_score,
            "method": self.method,
            "r_primary": self.r_partial_temp if np.isfinite(self.r_partial_temp) else self.r_raw,
            "p_primary": self.p_partial_temp if np.isfinite(self.p_partial_temp) else self.p_raw,
        }


def run_pain_sensitivity_correlations(
    features_df: pd.DataFrame,
    ratings: pd.Series,
    temperatures: pd.Series,
    method: str = "spearman",
    min_samples: int = 10,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Correlate features with pain sensitivity index."""
    def _align_psi_inputs() -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]:
        if features_df is None or features_df.empty or ratings is None or temperatures is None:
            return None, None, None

        common_index = features_df.index
        common_index = common_index.intersection(ratings.index)
        common_index = common_index.intersection(temperatures.index)

        if common_index.empty:
            if logger:
                logger.error("No overlapping samples across features, ratings, and temperatures for PSI")
            return None, None, None

        feat_aligned = features_df.loc[common_index]
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
            feat_aligned.loc[valid_mask],
            ratings_aligned.loc[valid_mask],
            temps_aligned.loc[valid_mask],
        )

    feat_aligned, ratings_aligned, temps_aligned = _align_psi_inputs()
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
        r, p, n = safe_correlation(vals, psi_valid.values, method, min_samples)

        if np.isfinite(r):
            records.append({
                "feature": col,
                "r_psi": float(r),
                "p_psi": float(p),
                "n": n,
                "effect_interpretation": interpret_correlation(r),
            })

    if logger:
        n_sig = sum(1 for r in records if r["p_psi"] < 0.05)
        logger.info(f"Pain sensitivity: {len(records)} features, {n_sig} significant")

    return pd.DataFrame(records)


###################################################################
# Alignment and ROI Helpers
###################################################################


def _align_groups_to_series(
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


def _align_features_and_targets(
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


def _build_temp_record_unified(
    x: pd.Series,
    temp: Optional[pd.Series],
    cov_no_temp: Optional[pd.DataFrame],
    identifier: str,
    id_key: str,
    band: str,
    cfg: Any,
    groups: Optional[np.ndarray] = None,
    **extra: Any,
) -> Optional[Dict[str, Any]]:
    """Build a temperature correlation record with optional bootstrap/permutation."""
    if temp is None or (hasattr(temp, "empty") and temp.empty):
        return None

    from . import compute_bootstrap_ci, compute_temp_permutation_pvalues, prepare_aligned_data

    x_a, temp_a, cov_a, _, _ = prepare_aligned_data(x, temp, cov_no_temp)
    if len(x_a) == 0 or len(temp_a) == 0:
        return None

    try:
        grp = _align_groups_to_series(x_a, groups if groups is not None else getattr(cfg, "groups", None))
    except ValueError:
        grp = None

    min_s = getattr(cfg, "min_samples_channel", None) or getattr(cfg, "min_samples_roi", 0)
    r, p, _ = safe_correlation(x_a, temp_a, cfg.method, min_s)

    ci_lo, ci_hi = np.nan, np.nan
    p_perm = np.nan

    if getattr(cfg, "bootstrap", 0) > 0:
        ci_lo, ci_hi = compute_bootstrap_ci(
            x_a, temp_a, cfg.bootstrap, 0.95,
            "spearman" if getattr(cfg, "use_spearman", False) else "pearson", cfg.rng
        )

    if getattr(cfg, "n_perm", 0) > 0:
        p_perm, _ = compute_temp_permutation_pvalues(
            x_a, temp_a, cov_a, cfg.method, cfg.n_perm, cfg.rng,
            band, identifier, getattr(cfg, "logger", None), groups=grp
        )

    return build_correlation_record(
        identifier, band, r, p, len(x_a), cfg.method,
        ci_low=ci_lo, ci_high=ci_hi, p_perm=p_perm,
        identifier_type=id_key, **extra,
    ).to_dict()


def _compute_roi_correlation_stats(
    x: pd.Series,
    y: pd.Series,
    x_a: np.ndarray,
    y_a: np.ndarray,
    cov: Optional[pd.DataFrame],
    temp: Optional[pd.Series],
    n_eff: int,
    band: str,
    roi: str,
    context: str,
    cfg: Any,
    groups: Optional[np.ndarray] = None,
    me_records: Optional[List[Dict]] = None,
) -> Any:
    """Compute comprehensive correlation statistics for an ROI."""
    from . import compute_bootstrap_ci, compute_partial_correlations, compute_permutation_pvalues
    from .base import CorrelationStats

    method = getattr(cfg, "method", "spearman")
    r, p = compute_correlation(x_a, y_a, method)

    r_part, p_part, n_part, r_part_temp, p_part_temp, n_part_temp = compute_partial_correlations(
        x, y, cov, temp, cfg.method, context, getattr(cfg, "logger", None), cfg.min_samples_roi
    )

    ci_lo, ci_hi = compute_bootstrap_ci(
        x_a, y_a, cfg.bootstrap, 0.95,
        "spearman" if getattr(cfg, "use_spearman", False) else "pearson", cfg.rng
    )

    x_series = pd.Series(x_a) if not isinstance(x_a, pd.Series) else x_a
    y_series = pd.Series(y_a) if not isinstance(y_a, pd.Series) else y_a

    p_perm, p_part_perm, p_part_temp_perm = compute_permutation_pvalues(
        x_series, y_series, cov, temp, cfg.method, cfg.n_perm, n_eff, cfg.rng,
        band, roi, groups=groups
    )

    return CorrelationStats(
        correlation=r,
        p_value=p,
        ci_low=ci_lo,
        ci_high=ci_hi,
        r_partial=r_part,
        p_partial=p_part,
        n_partial=n_part,
        r_partial_temp=r_part_temp,
        p_partial_temp=p_part_temp,
        n_partial_temp=n_part_temp,
        p_perm=p_perm,
        p_partial_perm=p_part_perm,
        p_partial_temp_perm=p_part_temp_perm,
    )


def fisher_z(r: float, config: Optional[Any] = None) -> float:
    """Fisher z-transform of correlation coefficient.
    
    Args:
        r: Correlation coefficient to transform
        config: Optional config object for clipping bounds (defaults to config values)
    """
    clip_min, clip_max = get_fisher_z_clip_values(config)
    r = np.clip(r, clip_min, clip_max)
    return 0.5 * np.log((1 + r) / (1 - r))


def inverse_fisher_z(z: float) -> float:
    """Inverse Fisher z-transform."""
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


def fisher_ci(
    r: float,
    n: int,
    config: Optional[Any] = None,
) -> Tuple[float, float]:
    """Compute Fisher-based CI for correlation."""
    ci_level = get_ci_level(config)

    if n < 4 or not np.isfinite(r):
        return np.nan, np.nan

    z = fisher_z(r, config)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf((1 + ci_level) / 2)

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
        se = np.std(zs, ddof=1) / np.sqrt(len(zs))
        z_crit = stats.norm.ppf((1 + ci_level) / 2)
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

    rs_v = rs_arr[valid]
    ws_v = ws_arr[valid]
    ws_v = ws_v / ws_v.sum()

    zs = np.array([fisher_z(r) for r in rs_v])
    z_mean = np.sum(zs * ws_v)
    r_mean = inverse_fisher_z(z_mean)

    if len(zs) > 1:
        var_z = np.sum(ws_v * (zs - z_mean) ** 2)
        se = np.sqrt(var_z / len(zs))
        z_crit = stats.norm.ppf((1 + ci_level) / 2)
        ci_lo = inverse_fisher_z(z_mean - z_crit * se)
        ci_hi = inverse_fisher_z(z_mean + z_crit * se)
    else:
        ci_lo = ci_hi = r_mean

    return float(r_mean), float(ci_lo), float(ci_hi), int(np.sum(valid))


def joint_valid_mask(*arrays: Sequence, require_all: bool = True) -> np.ndarray:
    """Create joint validity mask across multiple arrays."""
    if not arrays:
        return np.array([], dtype=bool)

    masks = [np.isfinite(np.asarray(a)) for a in arrays]
    if require_all:
        return np.all(np.stack(masks), axis=0)
    return np.any(np.stack(masks), axis=0)


def partial_corr_xy_given_Z(
    x: pd.Series,
    y: pd.Series,
    Z: pd.DataFrame,
    method: str,
    config: Optional[Any] = None,
) -> Tuple[float, float, int]:
    """
    Partial correlation of x,y controlling for Z.
    
    Returns (r_partial, p_value, n).
    """
    from scipy.linalg import lstsq

    df = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1).dropna()
    if len(df) < Z.shape[1] + 3:
        return np.nan, np.nan, 0

    X_mat = df[Z.columns].values
    X_mat = np.column_stack([np.ones(len(X_mat)), X_mat])

    x_vals = df["x"].values
    y_vals = df["y"].values

    # Residualize
    try:
        beta_x, *_ = lstsq(X_mat, x_vals)
        beta_y, *_ = lstsq(X_mat, y_vals)
    except Exception:
        return np.nan, np.nan, 0

    res_x = x_vals - X_mat @ beta_x
    res_y = y_vals - X_mat @ beta_y

    r, p = compute_correlation(res_x, res_y, method)
    return float(r), float(p), len(df)


def compute_partial_corr(
    x: pd.Series,
    y: pd.Series,
    Z: Optional[pd.DataFrame],
    method: str,
    *,
    logger: Optional[logging.Logger] = None,
    context: str = "",
    config: Optional[Any] = None,
) -> Tuple[float, float, int]:
    """
    Compute partial correlation, handling edge cases.
    
    If Z is None or empty, returns simple correlation.
    """
    if Z is None or Z.empty:
        valid = np.isfinite(x.values) & np.isfinite(y.values)
        if np.sum(valid) < 3:
            return np.nan, np.nan, 0
        r, p = compute_correlation(x.values[valid], y.values[valid], method)
        return r, p, int(np.sum(valid))

    return partial_corr_xy_given_Z(x, y, Z, method, config)


def normalize_series(s: pd.Series, epsilon: float = 1e-12) -> pd.Series:
    """Z-score normalize a series."""
    std = s.std()
    if std < epsilon:
        return pd.Series(np.zeros_like(s.values), index=s.index)
    return (s - s.mean()) / std


def compute_correlation_pvalue(r_values: List[float], config: Optional[Any] = None) -> float:
    """Compute p-value for aggregated correlation (one-sample t-test on z)."""
    rs = np.asarray(r_values, dtype=float)
    valid = np.isfinite(rs) & (np.abs(rs) < 1)

    if np.sum(valid) < 2:
        return np.nan

    zs = np.array([fisher_z(r) for r in rs[valid]])
    t_stat = np.mean(zs) / (np.std(zs, ddof=1) / np.sqrt(len(zs)))
    p = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=len(zs) - 1))
    return float(p)


###################################################################
# Bayes Factor for Correlations
###################################################################


def compute_bayes_factor_correlation(
    x: np.ndarray,
    y: np.ndarray,
    prior_width: float = 0.707,
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
    from scipy.special import hyp2f1
    
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    valid = np.isfinite(x) & np.isfinite(y)
    n = int(np.sum(valid))
    
    if n < 4:
        return np.nan, "insufficient_data"
    
    x_v, y_v = x[valid], y[valid]
    
    # Compute correlation
    if method == "spearman":
        r, _ = stats.spearmanr(x_v, y_v)
    else:
        r, _ = stats.pearsonr(x_v, y_v)
    
    if not np.isfinite(r) or np.abs(r) >= 1:
        return np.nan, "invalid_r"
    
    # JZS Bayes Factor approximation (Wetzels & Wagenmakers, 2012)
    # BF10 ≈ ((1 + t²/ν)^(-(ν+1)/2)) / Beta((1/2), (ν/2)) * integral term
    # Simplified approximation using hypergeometric function
    r2 = r ** 2
    
    # Compute BF10 using the exact formula for correlation
    # Based on Ly et al. (2016) "Harold Jeffreys's default Bayes factor hypothesis tests explained"
    try:
        # Log BF10 for numerical stability
        log_bf = (
            np.log(np.sqrt(2) / prior_width)
            + 0.5 * np.log(n - 1)
            + ((n - 1) / 2) * np.log(1 - r2)
            + np.log(hyp2f1(0.5, 0.5, (n + 1) / 2, r2))
        )
        bf10 = np.exp(log_bf)
    except (ValueError, OverflowError, RuntimeWarning):
        # Fallback: simpler approximation
        t = r * np.sqrt((n - 2) / (1 - r2))
        bf10 = np.sqrt((n + 1) / (2 * np.pi)) * (1 + t**2 / n) ** (-(n + 1) / 2)
    
    # Interpretation
    if bf10 < 1/100:
        interp = "extreme_H0"
    elif bf10 < 1/30:
        interp = "very_strong_H0"
    elif bf10 < 1/10:
        interp = "strong_H0"
    elif bf10 < 1/3:
        interp = "moderate_H0"
    elif bf10 < 1:
        interp = "anecdotal_H0"
    elif bf10 < 3:
        interp = "anecdotal_H1"
    elif bf10 < 10:
        interp = "moderate_H1"
    elif bf10 < 30:
        interp = "strong_H1"
    elif bf10 < 100:
        interp = "very_strong_H1"
    else:
        interp = "extreme_H1"
    
    return float(bf10), interp


###################################################################
# Robust Correlations (Outlier-Resistant)
###################################################################


def compute_robust_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "percentage_bend",
) -> Tuple[float, float]:
    """
    Compute robust correlation resistant to outliers.
    
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
        # Fallback to Spearman (already robust to monotonic outliers)
        return stats.spearmanr(x_v, y_v)


def _percentage_bend_correlation(x: np.ndarray, y: np.ndarray, beta: float = 0.2) -> Tuple[float, float]:
    """
    Percentage bend correlation (Wilcox, 1994).
    
    Downweights observations far from the median.
    """
    n = len(x)
    
    # Compute median and MAD
    mx, my = np.median(x), np.median(y)
    mad_x = np.median(np.abs(x - mx))
    mad_y = np.median(np.abs(y - my))
    
    if mad_x < 1e-12 or mad_y < 1e-12:
        return stats.spearmanr(x, y)
    
    # Bend parameter
    omega_x = (beta * (n - 1) + 0.5) / n
    omega_y = (beta * (n - 1) + 0.5) / n
    
    # Critical values
    crit_x = np.percentile(np.abs(x - mx) / mad_x, 100 * (1 - beta))
    crit_y = np.percentile(np.abs(y - my) / mad_y, 100 * (1 - beta))
    
    # Winsorize
    x_pb = np.clip((x - mx) / mad_x, -crit_x, crit_x)
    y_pb = np.clip((y - my) / mad_y, -crit_y, crit_y)
    
    # Compute correlation on bent data
    r, _ = stats.pearsonr(x_pb, y_pb)
    
    # Approximate p-value using t-distribution
    from .base import ensure_config
    config = ensure_config(None)
    epsilon = float(get_config_value(config, "statistics.constants.correlation_epsilon", 1e-12))
    t = r * np.sqrt((n - 2) / (1 - r**2 + epsilon))
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n - 2))
    
    return float(r), float(p)


def _winsorized_correlation(x: np.ndarray, y: np.ndarray, trim: float = 0.2) -> Tuple[float, float]:
    """
    Winsorized correlation (replaces extreme values with percentiles).
    """
    n = len(x)
    k = int(trim * n)
    
    if k < 1:
        return stats.pearsonr(x, y)
    
    # Winsorize both arrays
    def winsorize(arr):
        sorted_arr = np.sort(arr)
        lower, upper = sorted_arr[k], sorted_arr[-(k+1)]
        return np.clip(arr, lower, upper)
    
    x_w = winsorize(x)
    y_w = winsorize(y)
    
    r, _ = stats.pearsonr(x_w, y_w)
    
    # Approximate p-value
    n_eff = n - 2 * k
    if n_eff < 3:
        return float(r), np.nan
    
    t = r * np.sqrt((n_eff - 2) / (1 - r**2 + 1e-12))
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n_eff - 2))
    
    return float(r), float(p)


def _shepherd_correlation(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Shepherd's pi correlation (removes bivariate outliers via bootstrap MAD).
    """
    n = len(x)
    
    # Compute Mahalanobis-like distance using MAD
    mx, my = np.median(x), np.median(y)
    mad_x = np.median(np.abs(x - mx)) * 1.4826  # Scale to match std
    mad_y = np.median(np.abs(y - my)) * 1.4826
    
    if mad_x < 1e-12 or mad_y < 1e-12:
        return stats.spearmanr(x, y)
    
    # Standardize
    x_std = (x - mx) / mad_x
    y_std = (y - my) / mad_y
    
    # Distance from center
    dist = np.sqrt(x_std**2 + y_std**2)
    
    # Remove outliers (top alpha fraction by distance)
    threshold = np.percentile(dist, 100 * (1 - alpha))
    inliers = dist <= threshold
    
    if np.sum(inliers) < 4:
        return stats.spearmanr(x, y)
    
    r, p = stats.spearmanr(x[inliers], y[inliers])
    return float(r), float(p)


###################################################################
# LOSO Correlation Stability
###################################################################


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
    
    # Stability index: high = stable, low = unstable
    if abs(r_mean) > 1e-6:
        stability = max(0.0, 1.0 - (r_std / abs(r_mean)))
    else:
        stability = 0.0 if r_std > 0.1 else 1.0
    
    return r_mean, r_std, float(stability), r_values


###################################################################
# Split-Half Reliability
###################################################################


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
    
    if n < 10:
        return np.nan, np.nan, np.nan
    
    x = feature_values[valid]
    y = target_values[valid]
    
    if method == "odd_even":
        r1, _ = stats.spearmanr(x[::2], y[::2])
        r2, _ = stats.spearmanr(x[1::2], y[1::2])
        r_half = np.corrcoef([r1, r2])[0, 1] if np.isfinite(r1) and np.isfinite(r2) else np.nan
        reliability = _spearman_brown(r_half)
        return reliability, np.nan, np.nan
    
    # Random split-half
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
    ci_lo = float(np.percentile(reliabilities, 2.5))
    ci_hi = float(np.percentile(reliabilities, 97.5))
    
    return reliability, ci_lo, ci_hi


def _spearman_brown(r: float) -> float:
    """Spearman-Brown prophecy formula for split-half reliability."""
    if not np.isfinite(r) or abs(r) >= 1:
        return np.nan
    return (2 * r) / (1 + abs(r))


def save_correlation_results(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    sep: str = "\t",
    index: bool = False,
) -> None:
    """Save correlation results to file."""
    if df.empty:
        return
    
    # Ensure directory exists
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Format floats
    float_format = "%.6f"
    
    df.to_csv(path, sep=sep, index=index, float_format=float_format)


###################################################################
# Effect Size Benchmarks
###################################################################


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
    
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return (np.nan,) * 8 + (n_valid,)
    
    r_raw, p_raw = compute_correlation(x, y, method)
    
    r_pt, p_pt, r_po, p_po, r_pf, p_pf = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
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

