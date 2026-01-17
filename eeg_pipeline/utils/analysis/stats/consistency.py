"""
Effect Direction Consistency (Subject-Level)
===========================================

Builds a per-feature summary that compares effect direction across:
- correlations (feature vs rating/temperature)
- trialwise regression (beta_feature)
- model families (beta_feature by family)

This is non-gating and intended to surface sign flips and contradictions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _sign(x: Any) -> float:
    """Extract sign of a value, returning NaN for invalid inputs."""
    try:
        value = float(x)
    except (ValueError, TypeError):
        return np.nan
    
    if not np.isfinite(value):
        return np.nan
    if value == 0:
        return 0.0
    return float(np.sign(value))


def _extract_column(df: pd.DataFrame, primary: str, fallback: str) -> pd.Series:
    """Extract column from dataframe with primary and fallback names."""
    if primary in df.columns:
        return pd.to_numeric(df[primary], errors="coerce")
    if fallback in df.columns:
        return pd.to_numeric(df[fallback], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _extract_unique_features(*dataframes: Optional[pd.DataFrame]) -> List[str]:
    """Collect unique feature names from all dataframes."""
    features: List[str] = []
    for df in dataframes:
        if df is None or df.empty:
            continue
        if "feature" not in df.columns:
            continue
        unique_features = df["feature"].dropna().unique()
        features.extend([str(f) for f in unique_features])
    return sorted(set(features))


def _select_best_per_feature(
    df: pd.DataFrame,
    sort_column: str,
) -> pd.DataFrame:
    """Select row with minimum sort_column value per feature."""
    sorted_df = df.sort_values(["feature", sort_column], ascending=[True, True])
    best_per_feature = sorted_df.groupby("feature", sort=False).head(1)
    return best_per_feature.set_index("feature")


def _add_correlation_columns(
    out: pd.DataFrame,
    corr_df: pd.DataFrame,
    target: str,
    prefix: str,
) -> None:
    """Add correlation columns for a specific target."""
    filtered = corr_df[corr_df["target"] == target]
    if filtered.empty:
        return
    
    best = _select_best_per_feature(filtered, "p_val")
    out[f"{prefix}_r"] = out["feature"].map(best["r_val"]).astype(float)
    out[f"{prefix}_p"] = out["feature"].map(best["p_val"]).astype(float)
    out[f"{prefix}_sign"] = out[f"{prefix}_r"].map(_sign)


def _add_regression_columns(
    out: pd.DataFrame,
    regression_df: pd.DataFrame,
    target: str,
    prefix: str,
) -> None:
    """Add regression beta columns for a specific target."""
    filtered = regression_df[regression_df["target"] == target]
    if filtered.empty:
        return
    
    sort_col = "p_primary" if "p_primary" in filtered.columns else "p_feature"
    if sort_col in filtered.columns:
        sorted_df = filtered.sort_values(["feature", sort_col], ascending=[True, True])
    else:
        sorted_df = filtered
    
    best = sorted_df.groupby("feature", sort=False).head(1).set_index("feature")
    out[f"{prefix}_beta"] = out["feature"].map(best["beta"]).astype(float)
    
    p_col_name = "p_primary" if "p_primary" in best.columns else "p_feature"
    if p_col_name in best.columns:
        p_values = pd.to_numeric(best[p_col_name], errors="coerce")
    else:
        p_values = pd.Series(np.nan, index=best.index)
    out[f"{prefix}_p"] = out["feature"].map(p_values).astype(float)
    out[f"{prefix}_sign"] = out[f"{prefix}_beta"].map(_sign)


def _get_p_value_column(models_df: pd.DataFrame) -> Optional[str]:
    """Determine which p-value column to use from models dataframe."""
    if "p_primary" in models_df.columns:
        return "p_primary"
    if "p_feature" in models_df.columns:
        return "p_feature"
    return None


def _add_model_columns(
    out: pd.DataFrame,
    models_df: pd.DataFrame,
    family: str,
    target: str,
    prefix: str,
) -> None:
    """Add model beta columns for a specific family and target."""
    required_cols = ["model_family", "target"]
    if not all(col in models_df.columns for col in required_cols):
        return
    
    filtered = models_df[
        (models_df["model_family"] == family) & (models_df["target"] == target)
    ]
    if filtered.empty:
        return
    
    sorted_df = filtered.sort_values(["feature", "p_val"], ascending=[True, True])
    best = sorted_df.groupby("feature", sort=False).head(1).set_index("feature")
    out[f"{prefix}_beta"] = out["feature"].map(best["beta"]).astype(float)
    out[f"{prefix}_p"] = out["feature"].map(best["p_val"]).astype(float)
    out[f"{prefix}_sign"] = out[f"{prefix}_beta"].map(_sign)


def _detect_sign_flip(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """Detect sign flips between two series."""
    sign_a = series_a.map(_sign)
    sign_b = series_b.map(_sign)
    
    both_finite = np.isfinite(sign_a) & np.isfinite(sign_b)
    both_nonzero = (sign_a != 0) & (sign_b != 0)
    valid = both_finite & both_nonzero
    
    opposite_signs = sign_a * sign_b < 0
    return opposite_signs.where(valid, np.nan)


def _add_sign_flip_flags(out: pd.DataFrame) -> None:
    """Add sign flip detection columns."""
    if "reg_rating_beta" in out.columns and "reg_pain_residual_beta" in out.columns:
        out["flip_reg_rating_vs_pain_residual"] = _detect_sign_flip(
            out["reg_rating_beta"], out["reg_pain_residual_beta"]
        )
    
    if "corr_rating_r" in out.columns and "reg_rating_beta" in out.columns:
        out["flip_corr_vs_reg_rating"] = _detect_sign_flip(
            out["corr_rating_r"], out["reg_rating_beta"]
        )
    
    if "corr_rating_r" in out.columns and "model_ols_hc3_rating_beta" in out.columns:
        out["flip_corr_vs_model_ols_rating"] = _detect_sign_flip(
            out["corr_rating_r"], out["model_ols_hc3_rating_beta"]
        )
    
    if "reg_rating_beta" in out.columns and "model_ols_hc3_rating_beta" in out.columns:
        out["flip_reg_vs_model_ols_rating"] = _detect_sign_flip(
            out["reg_rating_beta"], out["model_ols_hc3_rating_beta"]
        )


def build_effect_direction_consistency_summary(
    *,
    corr_df: Optional[pd.DataFrame],
    regression_df: Optional[pd.DataFrame],
    models_df: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Merge per-feature evidence across analyses and flag sign flips.

    Output is one row per feature with columns like:
      corr_rating_r, reg_rating_beta, model_ols_rating_beta, ...
      flip_* booleans for contradictions.
    """
    features = _extract_unique_features(corr_df, regression_df, models_df)
    if not features:
        return pd.DataFrame(), {"status": "empty"}

    out = pd.DataFrame({"feature": features})

    if _has_valid_correlation_data(corr_df):
        _process_correlations(out, corr_df)

    if _has_valid_regression_data(regression_df):
        _process_regressions(out, regression_df)

    if _has_valid_models_data(models_df):
        _process_models(out, models_df)

    _add_sign_flip_flags(out)

    meta = _build_metadata(out, corr_df, regression_df, models_df)
    return out, meta


def _has_valid_correlation_data(corr_df: Optional[pd.DataFrame]) -> bool:
    """Check if correlation dataframe is valid and has required columns."""
    if corr_df is None or corr_df.empty:
        return False
    return "target" in corr_df.columns


def _has_valid_regression_data(regression_df: Optional[pd.DataFrame]) -> bool:
    """Check if regression dataframe is valid and has required columns."""
    if regression_df is None or regression_df.empty:
        return False
    return "target" in regression_df.columns


def _has_valid_models_data(models_df: Optional[pd.DataFrame]) -> bool:
    """Check if models dataframe is valid."""
    return models_df is not None and not models_df.empty


def _process_correlations(out: pd.DataFrame, corr_df: pd.DataFrame) -> None:
    """Process correlation data and add columns to output dataframe."""
    corr_processed = corr_df.copy()
    corr_processed["target"] = corr_processed["target"].astype(str)
    corr_processed["r_val"] = _extract_column(corr_processed, "r_primary", "r")
    corr_processed["p_val"] = _extract_column(corr_processed, "p_primary", "p")

    _add_correlation_columns(out, corr_processed, "rating", "corr_rating")
    _add_correlation_columns(out, corr_processed, "temperature", "corr_temperature")


def _process_regressions(out: pd.DataFrame, regression_df: pd.DataFrame) -> None:
    """Process regression data and add columns to output dataframe."""
    reg_processed = regression_df.copy()
    reg_processed["target"] = reg_processed["target"].astype(str)
    reg_processed["beta"] = _extract_column(reg_processed, "beta_feature", "beta")

    regression_targets = [
        ("rating", "reg_rating"),
        ("pain_residual", "reg_pain_residual"),
        ("temperature", "reg_temperature"),
    ]
    for target, prefix in regression_targets:
        _add_regression_columns(out, reg_processed, target, prefix)


def _process_models(out: pd.DataFrame, models_df: pd.DataFrame) -> None:
    """Process model data and add columns to output dataframe."""
    models_processed = models_df.copy()
    for col in ["target", "model_family"]:
        if col in models_processed.columns:
            models_processed[col] = models_processed[col].astype(str)
    
    models_processed["beta"] = _extract_column(models_processed, "beta_feature", "beta")
    
    p_column_name = _get_p_value_column(models_processed)
    if p_column_name is not None:
        models_processed["p_val"] = pd.to_numeric(
            models_processed[p_column_name], errors="coerce"
        )
    else:
        models_processed["p_val"] = np.nan

    model_families = ["ols_hc3", "robust_rlm", "quantile_50", "logit"]
    model_targets = ["rating", "pain_residual"]
    
    for family in model_families:
        for target in model_targets:
            prefix = f"model_{family}_{target}"
            _add_model_columns(out, models_processed, family, target, prefix)


def _build_metadata(
    out: pd.DataFrame,
    corr_df: Optional[pd.DataFrame],
    regression_df: Optional[pd.DataFrame],
    models_df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """Build metadata dictionary with summary information."""
    return {
        "status": "ok",
        "n_features": int(len(out)),
        "has_correlations": bool(corr_df is not None and not corr_df.empty),
        "has_regression": bool(regression_df is not None and not regression_df.empty),
        "has_models": bool(models_df is not None and not models_df.empty),
    }


__all__ = ["build_effect_direction_consistency_summary"]

