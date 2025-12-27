"""
Pain Residual (Subject-Level)
=============================

Defines a subject-level pain residual:

    pain_residual = rating - f(temperature)

where f(·) is a flexible (but stable) dose-response curve. This targets
"pain beyond stimulus intensity" for downstream feature associations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _get_config_value(config: Any, key: str, default: Any) -> Any:
    try:
        if hasattr(config, "get"):
            return config.get(key, default)
    except Exception:
        pass
    return default


def fit_temperature_rating_curve(
    temperature: pd.Series,
    rating: pd.Series,
    *,
    config: Optional[Any] = None,
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """
    Fit rating ~ f(temperature) and return (predicted, residual, metadata).

    Uses a spline model when statsmodels is available; otherwise falls back
    to a low-order polynomial.
    """
    temp = pd.to_numeric(temperature, errors="coerce")
    y = pd.to_numeric(rating, errors="coerce")
    idx = temp.index.intersection(y.index)
    temp = temp.loc[idx]
    y = y.loc[idx]

    meta: Dict[str, Any] = {
        "model": None,
        "status": "init",
        "n_total": int(len(idx)),
    }

    valid = temp.notna() & y.notna()
    n_valid = int(valid.sum())
    meta["n_valid"] = n_valid

    min_samples = int(_get_config_value(config, "behavior_analysis.pain_residual.min_samples", 10))
    if n_valid < min_samples:
        meta["status"] = "skipped_insufficient_samples"
        pred = pd.Series(np.nan, index=idx, dtype=float)
        resid = pd.Series(np.nan, index=idx, dtype=float)
        return pred, resid, meta

    temp_v = temp[valid]
    y_v = y[valid]

    method = str(_get_config_value(config, "behavior_analysis.pain_residual.method", "spline")).strip().lower()

    # Default outputs
    pred_full = pd.Series(np.nan, index=idx, dtype=float)
    resid_full = pd.Series(np.nan, index=idx, dtype=float)

    if method == "spline":
        try:
            import statsmodels.formula.api as smf
            import pandas as _pd
        except Exception:
            method = "poly"
        else:
            df_candidates = _get_config_value(config, "behavior_analysis.pain_residual.spline_df_candidates", [3, 4, 5])
            if not isinstance(df_candidates, (list, tuple)) or not df_candidates:
                df_candidates = [3, 4, 5]

            data = _pd.DataFrame({"temp": temp_v.to_numpy(dtype=float), "rating": y_v.to_numpy(dtype=float)})
            best = None
            best_meta = None
            for df_k in df_candidates:
                try:
                    k = int(df_k)
                except Exception:
                    continue
                if k < 3:
                    continue
                # patsy bs() is available through statsmodels' patsy dependency
                formula = f"rating ~ bs(temp, df={k}, degree=3)"
                try:
                    model = smf.ols(formula, data=data).fit()
                except Exception:
                    continue
                if best is None or (np.isfinite(model.aic) and model.aic < best.aic):
                    best = model
                    best_meta = {"df": k, "aic": float(model.aic), "bic": float(model.bic), "formula": formula}

            if best is None:
                method = "poly"
            else:
                # Predict for all temps (including NaNs -> NaN)
                try:
                    pred_v = best.predict(_pd.DataFrame({"temp": temp_v.to_numpy(dtype=float)}))
                except Exception:
                    method = "poly"
                else:
                    pred_full.loc[temp_v.index] = np.asarray(pred_v, dtype=float)
                    resid_full.loc[y_v.index] = y_v.to_numpy(dtype=float) - pred_full.loc[y_v.index].to_numpy(dtype=float)
                    meta["status"] = "ok"
                    meta["model"] = "spline"
                    if best_meta:
                        meta.update(best_meta)
                    # R^2 of the curve fit
                    try:
                        meta["r2"] = float(best.rsquared)
                    except Exception:
                        pass
                    return pred_full, resid_full, meta

    # Polynomial fallback
    degree = int(_get_config_value(config, "behavior_analysis.pain_residual.poly_degree", 2))
    degree = max(1, min(degree, 5))
    try:
        coef = np.polyfit(temp_v.to_numpy(dtype=float), y_v.to_numpy(dtype=float), deg=degree)
        poly = np.poly1d(coef)
        pred_full.loc[temp_v.index] = poly(temp_v.to_numpy(dtype=float))
        resid_full.loc[y_v.index] = y_v.to_numpy(dtype=float) - pred_full.loc[y_v.index].to_numpy(dtype=float)
        meta["status"] = "ok"
        meta["model"] = "poly"
        meta["poly_degree"] = int(degree)
    except Exception as exc:
        meta["status"] = "failed"
        meta["error"] = str(exc)

    return pred_full, resid_full, meta


__all__ = ["fit_temperature_rating_curve"]

