"""
Influence Diagnostics (Subject-Level, Non-Gating)
=================================================

Computes leverage and Cook's distance for per-feature subject-level models:

  outcome ~ (temperature control) + trial order + run/block dummies + feature (+ optional interaction)

Used to detect when single trials dominate an effect.
"""

from __future__ import annotations

from dataclasses import dataclass
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


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return np.full_like(x, np.nan)
    return (x - mu) / sd


def _build_covariate_design(
    df: pd.DataFrame,
    covariate_cols: List[str],
    *,
    add_intercept: bool = True,
    max_dummies: int = 20,
) -> Tuple[np.ndarray, List[str]]:
    parts = []
    names: List[str] = []
    if add_intercept:
        parts.append(pd.Series(1.0, index=df.index, name="intercept"))
        names.append("intercept")

    for col in covariate_cols:
        s = df[col]
        if s.dtype == object or str(s.dtype).startswith("category"):
            n_levels = int(pd.Series(s).nunique(dropna=True))
            if n_levels <= 1 or n_levels > max_dummies:
                continue
            dummies = pd.get_dummies(s.astype("category"), prefix=str(col), drop_first=True)
            for c in dummies.columns:
                parts.append(pd.to_numeric(dummies[c], errors="coerce").fillna(0.0))
                names.append(str(c))
        else:
            parts.append(pd.to_numeric(s, errors="coerce"))
            names.append(str(col))

    design_df = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    X = design_df.to_numpy(dtype=float)
    return X, names


def _ols_fit(X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    return beta


def _hat_diag(X: np.ndarray) -> Optional[np.ndarray]:
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    return np.einsum("ij,jk,ik->i", X, XtX_inv, X)


def _cooks_distance(
    resid: np.ndarray,
    h: np.ndarray,
    *,
    p: int,
) -> np.ndarray:
    resid = np.asarray(resid, dtype=float)
    h = np.asarray(h, dtype=float)
    n = int(len(resid))
    dof = max(n - p, 1)
    mse = float(np.nansum(resid**2) / dof) if dof > 0 else np.nan
    denom = (1.0 - h) ** 2
    denom = np.where(np.isfinite(denom) & (denom > 1e-12), denom, np.nan)
    cd = (resid**2 / (p * mse + 1e-12)) * (h / denom)
    return cd


@dataclass
class InfluenceConfig:
    enabled: bool = True
    outcomes: List[str] = None  # type: ignore[assignment]
    max_features: int = 20
    include_trial_order: bool = True
    include_run_block: bool = True
    include_temperature: bool = True
    temperature_control: str = "linear"  # "linear" | "rating_hat" | "spline"
    include_interaction: bool = False
    standardize: bool = True

    @classmethod
    def from_config(cls, config: Any) -> "InfluenceConfig":
        outcomes = _get(config, "behavior_analysis.influence.outcomes", ["rating", "pain_residual"])
        if not isinstance(outcomes, (list, tuple)) or not outcomes:
            outcomes = ["rating", "pain_residual"]
        return cls(
            enabled=bool(_get(config, "behavior_analysis.influence.enabled", True)),
            outcomes=[str(x) for x in outcomes],
            max_features=int(_get(config, "behavior_analysis.influence.max_features", 20)),
            include_trial_order=bool(_get(config, "behavior_analysis.influence.include_trial_order", True)),
            include_run_block=bool(_get(config, "behavior_analysis.influence.include_run_block", True)),
            include_temperature=bool(_get(config, "behavior_analysis.influence.include_temperature", True)),
            temperature_control=str(_get(config, "behavior_analysis.influence.temperature_control", "linear")).strip().lower(),
            include_interaction=bool(_get(config, "behavior_analysis.influence.include_interaction", False)),
            standardize=bool(_get(config, "behavior_analysis.influence.standardize", True)),
        )


def _select_top_features(
    *,
    corr_df: Optional[pd.DataFrame],
    regression_df: Optional[pd.DataFrame],
    models_df: Optional[pd.DataFrame],
    max_features: int,
) -> List[str]:
    candidates: List[Tuple[float, str]] = []

    # Prefer smallest p for rating effects.
    if regression_df is not None and not regression_df.empty:
        df = regression_df.copy()
        if "target" in df.columns:
            df = df[df["target"].astype(str) == "rating"]
        if "p_primary" in df.columns:
            df["score"] = pd.to_numeric(df["p_primary"], errors="coerce")
            for _, row in df.dropna(subset=["feature"]).iterrows():
                candidates.append((float(row.get("score", np.nan)), str(row["feature"])))

    if models_df is not None and not models_df.empty:
        df = models_df.copy()
        if "model_family" in df.columns:
            df = df[df["model_family"].astype(str) == "ols_hc3"]
        if "target" in df.columns:
            df = df[df["target"].astype(str) == "rating"]
        if "p_primary" in df.columns:
            df["score"] = pd.to_numeric(df["p_primary"], errors="coerce")
            for _, row in df.dropna(subset=["feature"]).iterrows():
                candidates.append((float(row.get("score", np.nan)), str(row["feature"])))

    if corr_df is not None and not corr_df.empty:
        df = corr_df.copy()
        if "target" in df.columns:
            df = df[df["target"].astype(str) == "rating"]
        pcol = "p_primary" if "p_primary" in df.columns else ("p_raw" if "p_raw" in df.columns else ("p_value" if "p_value" in df.columns else None))
        if pcol is not None:
            df["score"] = pd.to_numeric(df[pcol], errors="coerce")
            for _, row in df.dropna(subset=["feature"]).iterrows():
                candidates.append((float(row.get("score", np.nan)), str(row["feature"])))

    # If p missing, fall back to abs effects.
    if not candidates and regression_df is not None and not regression_df.empty:
        df = regression_df.copy()
        if "beta_feature" in df.columns:
            df["score"] = -pd.to_numeric(df["beta_feature"], errors="coerce").abs()
            for _, row in df.dropna(subset=["feature"]).iterrows():
                candidates.append((float(row.get("score", np.nan)), str(row["feature"])))

    # Deduplicate by best score (smaller is better).
    best: Dict[str, float] = {}
    for score, feat in candidates:
        if not np.isfinite(score):
            continue
        if feat not in best or score < best[feat]:
            best[feat] = score
    ranked = sorted(best.items(), key=lambda kv: kv[1])
    return [feat for feat, _ in ranked[: max(1, int(max_features))]]


def compute_influence_diagnostics(
    trial_df: pd.DataFrame,
    *,
    feature_cols: List[str],
    corr_df: Optional[pd.DataFrame] = None,
    regression_df: Optional[pd.DataFrame] = None,
    models_df: Optional[pd.DataFrame] = None,
    config: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cfg = InfluenceConfig.from_config(config)
    meta: Dict[str, Any] = {"enabled": cfg.enabled}
    if not cfg.enabled:
        return pd.DataFrame(), {**meta, "status": "disabled"}

    selected = _select_top_features(
        corr_df=corr_df, regression_df=regression_df, models_df=models_df, max_features=cfg.max_features
    )
    if not selected:
        return pd.DataFrame(), {**meta, "status": "empty"}
    meta["selected_features"] = selected

    records: List[Dict[str, Any]] = []

    for outcome in cfg.outcomes:
        if outcome not in trial_df.columns:
            continue
        y_all = pd.to_numeric(trial_df[outcome], errors="coerce").to_numpy(dtype=float)

        # Outcome-specific covariates (avoid leaking temperature into temperature outcome).
        covariates: List[str] = []
        temp_ctrl = str(cfg.temperature_control or "linear").strip().lower()
        temp_design_df = None
        temp_meta: Dict[str, Any] = {"temperature_control_requested": temp_ctrl}
        if cfg.include_temperature and outcome != "temperature":
            if temp_ctrl in ("rating_hat", "rating_hat_from_temp", "nonlinear") and "rating_hat_from_temp" in trial_df.columns:
                covariates.append("rating_hat_from_temp")
                temp_meta.update({"temperature_control_used": "rating_hat", "temperature_control_column": "rating_hat_from_temp"})
            elif temp_ctrl in ("spline", "rcs", "restricted_cubic") and "temperature" in trial_df.columns:
                from eeg_pipeline.utils.analysis.stats.splines import build_temperature_rcs_design

                temp_design_df, spline_cols, spline_meta = build_temperature_rcs_design(
                    trial_df["temperature"],
                    config=config,
                    key_prefix="behavior_analysis.influence.temperature_spline",
                    name_prefix="temperature_rcs",
                )
                for c in spline_cols:
                    if c not in covariates:
                        covariates.append(c)
                temp_meta.update({"temperature_control_used": "spline", "temperature_control_column": "temperature", "temperature_spline": spline_meta})
            elif "temperature" in trial_df.columns:
                covariates.append("temperature")
                temp_meta.update({"temperature_control_used": "linear", "temperature_control_column": "temperature"})

        if cfg.include_trial_order:
            for c in ["trial_index_within_group", "trial_index"]:
                if c in trial_df.columns:
                    covariates.append(c)
                    break
        if cfg.include_run_block:
            for c in ["run", "block"]:
                if c in trial_df.columns:
                    covariates.append(c)

        base_present = [c for c in covariates if c in trial_df.columns]
        base = trial_df[base_present].copy() if base_present else pd.DataFrame(index=trial_df.index)
        if temp_design_df is not None:
            for c in covariates:
                if c in base.columns:
                    continue
                if c in temp_design_df.columns:
                    base[c] = temp_design_df[c]

        Xb, Xb_names = _build_covariate_design(base, covariates, add_intercept=True)
        meta.setdefault("temperature_control_by_outcome", {})[str(outcome)] = temp_meta

        for feat in selected:
            if feat not in trial_df.columns:
                continue
            x_raw = pd.to_numeric(trial_df[feat], errors="coerce").to_numpy(dtype=float)
            x = _zscore(x_raw) if cfg.standardize else x_raw

            valid = np.isfinite(y_all) & np.isfinite(x) & np.all(np.isfinite(Xb), axis=1)
            if int(valid.sum()) < max(12, Xb.shape[1] + 5):
                continue

            y = y_all[valid]
            x_v = x[valid]
            X_parts = [Xb[valid], x_v[:, None]]
            names = [*Xb_names, "feature"]

            if cfg.include_interaction and "temperature" in trial_df.columns:
                t = pd.to_numeric(trial_df["temperature"], errors="coerce").to_numpy(dtype=float)
                t_v = t[valid]
                t_use = _zscore(t_v) if cfg.standardize else t_v
                if np.isfinite(t_use).any():
                    X_parts.append((x_v * t_use)[:, None])
                    names.append("feature_x_temperature")

            X = np.column_stack(X_parts)
            beta = _ols_fit(X, y)
            if beta is None:
                continue

            y_hat = X @ beta
            resid = y - y_hat
            h = _hat_diag(X)
            if h is None:
                continue

            p = int(X.shape[1])
            cooks = _cooks_distance(resid, h, p=p)

            # Common heuristics
            cooks_cfg = _get(config, "behavior_analysis.influence.cooks_threshold", None)
            lev_cfg = _get(config, "behavior_analysis.influence.leverage_threshold", None)
            cooks_thr = float(cooks_cfg) if cooks_cfg is not None else float(4.0 / max(int(len(y)), 1))
            lev_thr = float(lev_cfg) if lev_cfg is not None else float(2.0 * p / max(int(len(y)), 1))

            worst_cooks_i = int(np.nanargmax(cooks)) if np.isfinite(cooks).any() else -1
            worst_lev_i = int(np.nanargmax(h)) if np.isfinite(h).any() else -1

            valid_idx = np.where(valid)[0]
            worst_cooks_trial = int(valid_idx[worst_cooks_i]) if worst_cooks_i >= 0 else -1
            worst_lev_trial = int(valid_idx[worst_lev_i]) if worst_lev_i >= 0 else -1

            epoch_col = "epoch" if "epoch" in trial_df.columns else None
            epoch_worst = int(trial_df.iloc[worst_cooks_trial][epoch_col]) if epoch_col and worst_cooks_trial >= 0 else np.nan

            records.append(
                {
                    "feature": feat,
                    "target": str(outcome),
                    "n": int(len(y)),
                    "p": int(p),
                    "temperature_control": cfg.temperature_control,
                    "temperature_control_column": meta.get("temperature_control_column", None),
                    "cooks_threshold": cooks_thr,
                    "leverage_threshold": lev_thr,
                    "max_cooks": float(np.nanmax(cooks)) if np.isfinite(cooks).any() else np.nan,
                    "n_cooks_gt_threshold": int((cooks > cooks_thr).sum()) if np.isfinite(cooks).any() else 0,
                    "max_leverage": float(np.nanmax(h)) if np.isfinite(h).any() else np.nan,
                    "n_leverage_gt_threshold": int((h > lev_thr).sum()) if np.isfinite(h).any() else 0,
                    "worst_cooks_trial_index": worst_cooks_trial,
                    "worst_cooks_epoch": epoch_worst,
                    "worst_leverage_trial_index": worst_lev_trial,
                }
            )

    if not records:
        return pd.DataFrame(), {**meta, "status": "empty_after_fit"}

    out = pd.DataFrame(records)
    return out, {**meta, "status": "ok", "n_rows": int(len(out))}


__all__ = ["InfluenceConfig", "compute_influence_diagnostics"]
