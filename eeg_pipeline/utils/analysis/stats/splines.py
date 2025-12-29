"""
Spline Helpers (Lightweight, Subject-Level)
==========================================

Implements a minimal restricted cubic spline (RCS) basis generator for
nonlinear temperature control without depending on statsmodels/patsy.

This is intended for covariate control (not interpretability of spline terms).
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


def _tp3(x: np.ndarray) -> np.ndarray:
    """Truncated power function (cube)."""
    x = np.asarray(x, dtype=float)
    return np.maximum(x, 0.0) ** 3


def build_temperature_rcs_design(
    temperature: pd.Series,
    *,
    config: Optional[Any] = None,
    key_prefix: str = "behavior_analysis.regression.temperature_spline",
    name_prefix: str = "temperature_rcs",
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Build restricted cubic spline covariate columns for temperature.

    Returns
    -------
    df_cols : pd.DataFrame
        Columns to join into the design matrix. Includes a linear `temperature`
        term plus K-2 nonlinear spline columns when possible.
    covariate_names : list[str]
        Names of columns in `df_cols` in order.
    meta : dict
        Diagnostics about knots and fallbacks.
    """
    temp = pd.to_numeric(temperature, errors="coerce")
    x = temp.to_numpy(dtype=float)
    finite = np.isfinite(x)

    n_valid = int(finite.sum())
    min_samples = int(_get(config, f"{key_prefix}.min_samples", 12))
    n_knots = int(_get(config, f"{key_prefix}.n_knots", 4))
    q_low = float(_get(config, f"{key_prefix}.quantile_low", 0.05))
    q_high = float(_get(config, f"{key_prefix}.quantile_high", 0.95))

    meta: Dict[str, Any] = {
        "n_valid": n_valid,
        "min_samples": min_samples,
        "n_knots": n_knots,
        "quantile_low": q_low,
        "quantile_high": q_high,
    }

    out = pd.DataFrame(index=temp.index)
    out["temperature"] = temp

    if n_valid < min_samples:
        meta["status"] = "skipped_insufficient_samples"
        return out, ["temperature"], meta

    if n_knots < 4:
        # RCS requires >=4 knots for at least one nonlinear term.
        meta["status"] = "skipped_n_knots_lt_4"
        return out, ["temperature"], meta

    x_v = x[finite]
    if np.nanstd(x_v, ddof=1) <= 1e-12:
        meta["status"] = "skipped_constant_temperature"
        return out, ["temperature"], meta

    # Choose knots using quantiles, bounded away from extremes.
    q_low = min(max(q_low, 0.0), 0.49)
    q_high = max(min(q_high, 1.0), 0.51)
    if q_high <= q_low:
        q_low, q_high = 0.05, 0.95

    # Evenly space internal quantiles between q_low and q_high.
    qs = np.linspace(q_low, q_high, num=max(n_knots, 4))
    knots = np.quantile(x_v, qs)
    knots = np.unique(np.asarray(knots, dtype=float))

    if knots.size < 4:
        meta["status"] = "skipped_insufficient_unique_knots"
        meta["knots_unique"] = int(knots.size)
        return out, ["temperature"], meta

    knots = np.sort(knots)
    k0, k1, k_lastm1, k_last = float(knots[0]), float(knots[1]), float(knots[-2]), float(knots[-1])
    denom = (k_last - k_lastm1)
    if not np.isfinite(denom) or abs(denom) <= 1e-12:
        meta["status"] = "skipped_degenerate_boundary_knots"
        return out, ["temperature"], meta

    # Nonlinear terms: for each interior knot excluding the first and last.
    # Harrell restricted cubic spline basis:
    # h_j(x) = tp(x-k_j) - tp(x-k_{K-1})*(k_K-k_j)/(k_K-k_{K-1}) + tp(x-k_K)*(k_{K-1}-k_j)/(k_K-k_{K-1})
    interior = knots[1:-1]
    n_terms = int(max(len(interior) - 1, 0))  # exclude the last interior knot (k_{K-1})
    if n_terms <= 0:
        meta["status"] = "ok_linear_only"
        meta["knots"] = [float(k) for k in knots.tolist()]
        return out, ["temperature"], meta

    # Prepare arrays for all rows; missing -> NaN
    x_all = x.astype(float)
    tp_lastm1 = _tp3(x_all - k_lastm1)
    tp_last = _tp3(x_all - k_last)

    cov_names: List[str] = ["temperature"]
    for j, k_j in enumerate(interior[:-1]):
        k_jf = float(k_j)
        term = _tp3(x_all - k_jf)
        term = term - tp_lastm1 * (k_last - k_jf) / denom + tp_last * (k_lastm1 - k_jf) / denom
        term[~finite] = np.nan
        col_name = f"{name_prefix}_{j+1}"
        out[col_name] = term
        cov_names.append(col_name)

    meta["status"] = "ok"
    meta["knots"] = [float(k) for k in knots.tolist()]
    meta["n_spline_terms"] = int(len(cov_names) - 1)
    return out, cov_names, meta


__all__ = ["build_temperature_rcs_design"]

