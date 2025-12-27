"""
Temperature→Rating Model Comparisons (Subject-Level)
====================================================

Pain studies often exhibit non-linear dose-response curves. This module provides
lightweight, subject-level comparisons of model families for `rating ~ f(temperature)`,
plus an optional single-breakpoint ("hinge") test.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _get(config: Any, key: str, default: Any) -> Any:
    try:
        if hasattr(config, "get"):
            return config.get(key, default)
    except Exception:
        pass
    return default


def _rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    return float(np.sqrt(np.mean((y - y_hat) ** 2))) if y.size else np.nan


def compare_temperature_rating_models(
    temperature: pd.Series,
    rating: pd.Series,
    *,
    config: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compare linear / polynomial / spline models for rating ~ f(temperature)."""
    temp = pd.to_numeric(temperature, errors="coerce")
    y = pd.to_numeric(rating, errors="coerce")
    idx = temp.index.intersection(y.index)
    temp = temp.loc[idx]
    y = y.loc[idx]

    valid = temp.notna() & y.notna()
    n = int(valid.sum())
    min_samples = int(_get(config, "behavior_analysis.pain_residual.model_comparison.min_samples", 10))
    meta: Dict[str, Any] = {"n_valid": n, "min_samples": min_samples}
    if n < min_samples:
        return pd.DataFrame(), {**meta, "status": "skipped_insufficient_samples"}

    data = pd.DataFrame({"temp": temp[valid].to_numpy(dtype=float), "rating": y[valid].to_numpy(dtype=float)})

    try:
        import statsmodels.formula.api as smf
        _has_statsmodels = True
    except Exception:
        smf = None  # type: ignore[assignment]
        _has_statsmodels = False

    meta["has_statsmodels"] = _has_statsmodels
    if not _has_statsmodels:
        return pd.DataFrame(), {**meta, "status": "missing_statsmodels"}

    recs: List[Dict[str, Any]] = []

    def _add(name: str, formula: str) -> None:
        try:
            model = smf.ols(formula, data=data).fit()
        except Exception:
            return
        try:
            yhat = np.asarray(model.predict(data), dtype=float)
        except Exception:
            yhat = np.full(n, np.nan)
        recs.append(
            {
                "model": name,
                "formula": formula,
                "n": n,
                "k_params": int(getattr(model, "df_model", np.nan)) + 1 if hasattr(model, "df_model") else np.nan,
                "aic": float(model.aic) if np.isfinite(model.aic) else np.nan,
                "bic": float(model.bic) if np.isfinite(model.bic) else np.nan,
                "r2": float(model.rsquared) if hasattr(model, "rsquared") else np.nan,
                "rmse": _rmse(data["rating"].to_numpy(), yhat) if np.isfinite(yhat).all() else np.nan,
            }
        )

    _add("linear", "rating ~ temp")

    degrees = _get(config, "behavior_analysis.pain_residual.model_comparison.poly_degrees", [2, 3])
    if not isinstance(degrees, (list, tuple)) or not degrees:
        degrees = [2, 3]
    for d in degrees:
        try:
            deg = int(d)
        except Exception:
            continue
        if deg < 2 or deg > 5:
            continue
        terms = " + ".join([f"I(temp**{k})" for k in range(2, deg + 1)])
        _add(f"poly{deg}", f"rating ~ temp + {terms}")

    df_candidates = _get(config, "behavior_analysis.pain_residual.spline_df_candidates", [3, 4, 5])
    if not isinstance(df_candidates, (list, tuple)) or not df_candidates:
        df_candidates = [3, 4, 5]
    for df_k in df_candidates:
        try:
            k = int(df_k)
        except Exception:
            continue
        if k < 3:
            continue
        _add(f"spline_df{k}", f"rating ~ bs(temp, df={k}, degree=3)")

    if not recs:
        return pd.DataFrame(), {**meta, "status": "empty"}

    out = pd.DataFrame(recs).sort_values("aic", ascending=True)
    best = out.iloc[0].to_dict() if len(out) else {}
    meta.update({"status": "ok", "best_model": best.get("model"), "best_aic": best.get("aic")})
    out["delta_aic"] = out["aic"] - float(best.get("aic")) if best.get("aic") is not None else np.nan
    return out, meta


def fit_temperature_breakpoint_test(
    temperature: pd.Series,
    rating: pd.Series,
    *,
    config: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fit a single-breakpoint hinge model and compare to linear baseline.

    Model: rating ~ temp + max(0, temp - c)
    """
    temp = pd.to_numeric(temperature, errors="coerce")
    y = pd.to_numeric(rating, errors="coerce")
    idx = temp.index.intersection(y.index)
    temp = temp.loc[idx]
    y = y.loc[idx]

    valid = temp.notna() & y.notna()
    n = int(valid.sum())
    min_samples = int(_get(config, "behavior_analysis.pain_residual.breakpoint_test.min_samples", 12))
    meta: Dict[str, Any] = {"n_valid": n, "min_samples": min_samples}
    if n < min_samples:
        return pd.DataFrame(), {**meta, "status": "skipped_insufficient_samples"}

    data = pd.DataFrame({"temp": temp[valid].to_numpy(dtype=float), "rating": y[valid].to_numpy(dtype=float)})

    try:
        import statsmodels.api as sm
        _has_statsmodels = True
    except Exception:
        sm = None  # type: ignore[assignment]
        _has_statsmodels = False
    meta["has_statsmodels"] = _has_statsmodels
    if not _has_statsmodels:
        return pd.DataFrame(), {**meta, "status": "missing_statsmodels"}

    # Baseline linear model
    X0 = sm.add_constant(data["temp"].to_numpy(dtype=float), has_constant="add")
    try:
        m0 = sm.OLS(data["rating"].to_numpy(dtype=float), X0).fit()
    except Exception as exc:
        return pd.DataFrame(), {**meta, "status": "failed_linear", "error": str(exc)}

    q_lo = float(_get(config, "behavior_analysis.pain_residual.breakpoint_test.quantile_low", 0.15))
    q_hi = float(_get(config, "behavior_analysis.pain_residual.breakpoint_test.quantile_high", 0.85))
    n_candidates = int(_get(config, "behavior_analysis.pain_residual.breakpoint_test.n_candidates", 15))

    temps = data["temp"].to_numpy(dtype=float)
    lo = float(np.quantile(temps, q_lo))
    hi = float(np.quantile(temps, q_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.DataFrame(), {**meta, "status": "skipped_invalid_temperature_range"}

    candidates = np.linspace(lo, hi, num=max(n_candidates, 5))

    recs: List[Dict[str, Any]] = []
    best = None
    for c in candidates:
        hinge = np.maximum(0.0, temps - float(c))
        X = np.column_stack([np.ones(len(temps)), temps, hinge])
        try:
            m = sm.OLS(data["rating"].to_numpy(dtype=float), X).fit()
        except Exception:
            continue
        rec = {
            "breakpoint_c": float(c),
            "aic": float(m.aic) if np.isfinite(m.aic) else np.nan,
            "bic": float(m.bic) if np.isfinite(m.bic) else np.nan,
            "r2": float(m.rsquared) if hasattr(m, "rsquared") else np.nan,
            "beta_temp": float(m.params[1]) if len(m.params) > 1 else np.nan,
            "beta_hinge": float(m.params[2]) if len(m.params) > 2 else np.nan,
            "p_hinge": float(m.pvalues[2]) if hasattr(m, "pvalues") and len(m.pvalues) > 2 else np.nan,
        }
        recs.append(rec)
        if best is None or (np.isfinite(rec["aic"]) and rec["aic"] < best["aic"]):
            best = rec

    if not recs or best is None:
        return pd.DataFrame(), {**meta, "status": "empty"}

    out = pd.DataFrame(recs).sort_values("aic", ascending=True)

    # Nested-model F test: linear (2 params) vs hinge (3 params) at best breakpoint.
    c_best = float(best["breakpoint_c"])
    hinge_best = np.maximum(0.0, temps - c_best)
    X1 = np.column_stack([np.ones(len(temps)), temps, hinge_best])
    try:
        m1 = sm.OLS(data["rating"].to_numpy(dtype=float), X1).fit()
    except Exception as exc:
        return out, {**meta, "status": "ok", "best_breakpoint": c_best, "f_test_error": str(exc)}

    rss0 = float(np.sum(m0.resid**2))
    rss1 = float(np.sum(m1.resid**2))
    df0 = int(m0.df_model) + 1 if hasattr(m0, "df_model") else 2
    df1 = int(m1.df_model) + 1 if hasattr(m1, "df_model") else 3
    n_obs = int(len(temps))
    if n_obs > df1 and rss1 > 0 and rss0 >= rss1:
        num = (rss0 - rss1) / max(df1 - df0, 1)
        den = rss1 / max(n_obs - df1, 1)
        f_stat = num / den if den > 0 else np.nan
        from scipy.stats import f as f_dist

        p_f = float(f_dist.sf(f_stat, dfn=max(df1 - df0, 1), dfd=max(n_obs - df1, 1))) if np.isfinite(f_stat) else np.nan
    else:
        f_stat = np.nan
        p_f = np.nan

    meta.update(
        {
            "status": "ok",
            "best_breakpoint": c_best,
            "best_aic": float(best["aic"]),
            "delta_aic_vs_linear": float(best["aic"] - float(m0.aic)) if np.isfinite(m0.aic) and np.isfinite(best["aic"]) else np.nan,
            "f_test_stat": f_stat,
            "f_test_p": p_f,
        }
    )
    return out, meta


__all__ = ["compare_temperature_rating_models", "fit_temperature_breakpoint_test"]

