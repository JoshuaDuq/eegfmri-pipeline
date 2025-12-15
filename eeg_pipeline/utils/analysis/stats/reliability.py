"""
Reliability and Validity Statistics
====================================

Functions for assessing measurement reliability and predictive validity:
- ICC (Intraclass Correlation Coefficient)
- Split-half reliability
- Hierarchical FDR correction
- Cross-validated predictive modeling
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
from scipy import stats


###################################################################
# Intraclass Correlation Coefficient (ICC)
###################################################################


def compute_icc(
    data: np.ndarray,
    icc_type: str = "ICC(2,1)",
) -> Tuple[float, float, float]:
    """Compute Intraclass Correlation Coefficient.
    
    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_subjects, n_raters/n_sessions).
        Each row is a subject, each column is a rater/session.
    icc_type : str
        Type of ICC to compute:
        - "ICC(1,1)": One-way random, single rater
        - "ICC(2,1)": Two-way random, single rater (default)
        - "ICC(3,1)": Two-way mixed, single rater
        - "ICC(1,k)": One-way random, average of k raters
        - "ICC(2,k)": Two-way random, average of k raters
        - "ICC(3,k)": Two-way mixed, average of k raters
    
    Returns
    -------
    icc : float
        ICC value
    ci_low : float
        Lower 95% CI bound
    ci_high : float
        Upper 95% CI bound
    """
    data = np.asarray(data)
    if data.ndim != 2:
        return np.nan, np.nan, np.nan
    
    n, k = data.shape
    if n < 2 or k < 2:
        return np.nan, np.nan, np.nan
    
    # Grand mean
    grand_mean = np.mean(data)
    
    # Row means (subjects)
    row_means = np.mean(data, axis=1)
    
    # Column means (raters/sessions)
    col_means = np.mean(data, axis=0)
    
    # Sum of squares
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)  # Between subjects
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)  # Between raters
    ss_error = ss_total - ss_rows - ss_cols  # Residual
    
    # Mean squares
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    
    # Compute ICC based on type
    if icc_type in ["ICC(1,1)", "ICC1"]:
        # One-way random, single rater
        ms_within = (ss_cols + ss_error) / (n * (k - 1))
        icc = (ms_rows - ms_within) / (ms_rows + (k - 1) * ms_within)
        
    elif icc_type in ["ICC(2,1)", "ICC2"]:
        # Two-way random, single rater (most common)
        icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n)
        
    elif icc_type in ["ICC(3,1)", "ICC3"]:
        # Two-way mixed, single rater
        icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error)
        
    elif icc_type in ["ICC(1,k)", "ICC1k"]:
        # One-way random, average of k raters
        ms_within = (ss_cols + ss_error) / (n * (k - 1))
        icc = (ms_rows - ms_within) / ms_rows
        
    elif icc_type in ["ICC(2,k)", "ICC2k"]:
        # Two-way random, average of k raters
        icc = (ms_rows - ms_error) / (ms_rows + (ms_cols - ms_error) / n)
        
    elif icc_type in ["ICC(3,k)", "ICC3k"]:
        # Two-way mixed, average of k raters
        icc = (ms_rows - ms_error) / ms_rows
        
    else:
        raise ValueError(f"Unknown ICC type: {icc_type}")
    
    # Confidence intervals using F-distribution approximation
    # Simplified CI calculation
    f_value = ms_rows / ms_error if ms_error > 0 else np.inf
    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    
    if np.isfinite(f_value) and f_value > 0:
        f_low = f_value / stats.f.ppf(0.975, df1, df2)
        f_high = f_value * stats.f.ppf(0.975, df2, df1)
        
        ci_low = (f_low - 1) / (f_low + k - 1)
        ci_high = (f_high - 1) / (f_high + k - 1)
    else:
        ci_low, ci_high = np.nan, np.nan
    
    return float(np.clip(icc, -1, 1)), float(ci_low), float(ci_high)


def compute_split_half_reliability(
    data: np.ndarray,
    n_splits: int = 100,
    method: str = "spearman",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Compute split-half reliability with Spearman-Brown correction.
    
    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_trials, n_features) or 1D array of values.
    n_splits : int
        Number of random splits to average over.
    method : str
        Correlation method ('spearman' or 'pearson').
    rng : np.random.Generator, optional
        Random number generator.
    
    Returns
    -------
    reliability : float
        Spearman-Brown corrected reliability coefficient.
    ci_low : float
        Lower 95% CI bound.
    ci_high : float
        Upper 95% CI bound.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    n_trials = data.shape[0]
    if n_trials < 4:
        return np.nan, np.nan, np.nan
    
    correlations = []
    
    for _ in range(n_splits):
        # Random split
        indices = rng.permutation(n_trials)
        half = n_trials // 2
        
        half1 = data[indices[:half]].mean(axis=0)
        half2 = data[indices[half:half + half]].mean(axis=0)
        
        if len(half1) < 2:
            # Single feature case
            half1_vals = data[indices[:half], 0]
            half2_vals = data[indices[half:half + half], 0]
            
            if method == "spearman":
                r, _ = stats.spearmanr(half1_vals, half2_vals)
            else:
                r, _ = stats.pearsonr(half1_vals, half2_vals)
        else:
            if method == "spearman":
                r, _ = stats.spearmanr(half1, half2)
            else:
                r, _ = stats.pearsonr(half1, half2)
        
        if np.isfinite(r):
            correlations.append(r)
    
    if not correlations:
        return np.nan, np.nan, np.nan
    
    # Mean split-half correlation
    mean_r = np.mean(correlations)
    
    # Spearman-Brown prophecy formula
    reliability = (2 * mean_r) / (1 + mean_r) if mean_r > -1 else np.nan
    
    # Bootstrap CI
    boot_reliabilities = []
    for r in correlations:
        sb = (2 * r) / (1 + r) if r > -1 else np.nan
        if np.isfinite(sb):
            boot_reliabilities.append(sb)
    
    if len(boot_reliabilities) > 10:
        ci_low = np.percentile(boot_reliabilities, 2.5)
        ci_high = np.percentile(boot_reliabilities, 97.5)
    else:
        ci_low, ci_high = np.nan, np.nan
    
    return float(reliability), float(ci_low), float(ci_high)


def compute_feature_reliability(
    df: pd.DataFrame,
    feature_col: str,
    value_col: str,
    groupby_col: str = "trial",
    session_col: Optional[str] = None,
    min_observations: int = 10,
) -> pd.DataFrame:
    """Compute reliability metrics for each feature.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with features and values.
    feature_col : str
        Column containing feature names.
    value_col : str
        Column containing feature values.
    groupby_col : str
        Column to group by (e.g., 'trial', 'block').
    session_col : str, optional
        Column for session/run (for ICC).
    min_observations : int
        Minimum observations required.
    
    Returns
    -------
    pd.DataFrame
        Reliability metrics per feature.
    """
    results = []
    
    for feature in df[feature_col].unique():
        feat_df = df[df[feature_col] == feature]
        
        if len(feat_df) < min_observations:
            continue
        
        values = feat_df[value_col].values
        
        # Split-half reliability
        sh_rel, sh_low, sh_high = compute_split_half_reliability(values)
        
        result = {
            "feature": feature,
            "n": len(values),
            "split_half_reliability": sh_rel,
            "sh_ci_low": sh_low,
            "sh_ci_high": sh_high,
        }
        
        # ICC if session info available
        if session_col and session_col in feat_df.columns:
            sessions = feat_df[session_col].unique()
            if len(sessions) >= 2:
                # Reshape for ICC
                pivot = feat_df.pivot_table(
                    index=groupby_col, 
                    columns=session_col, 
                    values=value_col,
                    aggfunc="mean"
                ).dropna()
                
                if pivot.shape[0] >= 3 and pivot.shape[1] >= 2:
                    icc, icc_low, icc_high = compute_icc(pivot.values)
                    result["icc"] = icc
                    result["icc_ci_low"] = icc_low
                    result["icc_ci_high"] = icc_high
        
        results.append(result)
    
    return pd.DataFrame(results)


###################################################################
# Hierarchical FDR Correction
###################################################################


def hierarchical_fdr(
    p_values: Dict[str, np.ndarray],
    alpha: float = 0.05,
    method: str = "bh",
) -> Dict[str, Dict[str, Any]]:
    """Apply hierarchical FDR correction across multiple families.
    
    Two-stage procedure:
    1. Apply FDR within each family
    2. Apply FDR across family-level summary statistics
    
    Parameters
    ----------
    p_values : Dict[str, np.ndarray]
        Dictionary mapping family names to arrays of p-values.
    alpha : float
        FDR threshold.
    method : str
        FDR method ('bh' for Benjamini-Hochberg).
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Results per family with:
        - q_values: FDR-corrected q-values within family
        - reject: Boolean rejection mask within family
        - q_global: Global q-value for the family
        - reject_global: Whether family passes global FDR
        - n_tests: Number of tests in family
        - n_reject: Number of rejections within family
    """
    from eeg_pipeline.utils.analysis.stats import fdr_bh
    
    results = {}
    family_min_p = []
    family_names = []
    
    # Stage 1: Within-family FDR
    for family_name, p_arr in p_values.items():
        p_arr = np.asarray(p_arr)
        valid_mask = np.isfinite(p_arr)
        
        if not valid_mask.any():
            results[family_name] = {
                "q_values": np.full_like(p_arr, np.nan),
                "reject": np.zeros_like(p_arr, dtype=bool),
                "q_global": np.nan,
                "reject_global": False,
                "n_tests": 0,
                "n_reject": 0,
            }
            continue
        
        # Apply BH-FDR within family
        q_arr = np.full_like(p_arr, np.nan)
        q_arr[valid_mask] = fdr_bh(p_arr[valid_mask], alpha=alpha)
        reject = q_arr < alpha
        
        results[family_name] = {
            "q_values": q_arr,
            "reject": reject,
            "n_tests": int(valid_mask.sum()),
            "n_reject": int(reject.sum()),
        }
        
        # Collect minimum p-value per family for global correction
        min_p = np.nanmin(p_arr[valid_mask])
        family_min_p.append(min_p)
        family_names.append(family_name)
    
    # Stage 2: Global FDR across families
    if family_min_p:
        family_min_p = np.array(family_min_p)
        global_q = fdr_bh(family_min_p, alpha=alpha)
        global_reject = global_q < alpha
        
        for i, family_name in enumerate(family_names):
            results[family_name]["q_global"] = float(global_q[i])
            results[family_name]["reject_global"] = bool(global_reject[i])
    
    return results


def compute_hierarchical_fdr_summary(
    stats_dir,
    alpha: float = 0.05,
    config: Optional[Any] = None,
) -> pd.DataFrame:
    """Compute hierarchical FDR summary from stats directory.
    
    Groups tests by analysis type and applies two-stage FDR.
    
    Parameters
    ----------
    stats_dir : Path
        Directory containing stats TSV files.
    alpha : float
        FDR threshold.
    config : Any, optional
        Pipeline configuration.
    
    Returns
    -------
    pd.DataFrame
        Summary with hierarchical FDR results.
    """
    from pathlib import Path
    from eeg_pipeline.io.tsv import read_tsv
    
    stats_dir = Path(stats_dir)
    
    # Group files by analysis type
    analysis_groups = {
        "power": [],
        "connectivity": [],
        "microstates": [],
        "aperiodic": [],
        "pac": [],
        "other": [],
    }
    
    for f in stats_dir.glob("corr_stats_*.tsv"):
        fname = f.name.lower()
        if "pow" in fname or "power" in fname:
            analysis_groups["power"].append(f)
        elif "conn" in fname or "edge" in fname or "graph" in fname:
            analysis_groups["connectivity"].append(f)
        elif "microstate" in fname or "ms_" in fname:
            analysis_groups["microstates"].append(f)
        elif "aperiodic" in fname:
            analysis_groups["aperiodic"].append(f)
        elif "pac" in fname:
            analysis_groups["pac"].append(f)
        else:
            analysis_groups["other"].append(f)
    
    # Collect p-values by group
    p_by_group = {}
    file_refs = {}
    
    for group_name, files in analysis_groups.items():
        all_p = []
        refs = []
        
        for fpath in files:
            df = read_tsv(fpath)
            if df is None or df.empty:
                continue
            
            p_col = None
            for col in ["p", "p_value", "p_perm", "pvalue"]:
                if col in df.columns:
                    p_col = col
                    break
            
            if p_col is None:
                continue
            
            p_vals = pd.to_numeric(df[p_col], errors="coerce").values
            for i, p in enumerate(p_vals):
                if np.isfinite(p):
                    all_p.append(p)
                    refs.append((fpath, i))
        
        if all_p:
            p_by_group[group_name] = np.array(all_p)
            file_refs[group_name] = refs
    
    # Apply hierarchical FDR
    if not p_by_group:
        return pd.DataFrame()
    
    hier_results = hierarchical_fdr(p_by_group, alpha=alpha)
    
    # Build summary dataframe
    summary_rows = []
    for group_name, results in hier_results.items():
        summary_rows.append({
            "analysis_type": group_name,
            "n_tests": results["n_tests"],
            "n_reject_within": results["n_reject"],
            "pct_reject_within": 100 * results["n_reject"] / max(results["n_tests"], 1),
            "q_global": results.get("q_global", np.nan),
            "reject_global": results.get("reject_global", False),
        })
    
    return pd.DataFrame(summary_rows)


###################################################################
# Predictive Validity (Cross-Validated Modeling)
###################################################################


def cross_validated_prediction(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "ridge",
    n_folds: int = 5,
    n_permutations: int = 100,
    alpha_values: Optional[List[float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Cross-validated prediction with permutation null.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Target vector (n_samples,).
    model_type : str
        Model type: 'ridge', 'elasticnet', 'rf' (random forest).
    n_folds : int
        Number of CV folds.
    n_permutations : int
        Number of permutations for null distribution.
    alpha_values : List[float], optional
        Regularization values to try (for ridge/elasticnet).
    rng : np.random.Generator, optional
        Random number generator.
    
    Returns
    -------
    Dict[str, Any]
        Results including:
        - cv_r2: Cross-validated R²
        - cv_mae: Cross-validated MAE
        - cv_predictions: Out-of-fold predictions
        - feature_weights: Model coefficients
        - null_r2: Permutation null R² distribution
        - p_value: Permutation p-value
    """
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import r2_score, mean_absolute_error
    
    if rng is None:
        rng = np.random.default_rng()
    
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    
    # Handle NaN
    valid_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    
    n_samples = len(y_clean)
    if n_samples < 10:
        return {
            "cv_r2": np.nan,
            "cv_mae": np.nan,
            "cv_predictions": np.full_like(y, np.nan),
            "feature_weights": np.full(X.shape[1], np.nan),
            "null_r2": np.array([]),
            "p_value": np.nan,
        }
    
    # Adjust folds if needed
    n_folds = min(n_folds, n_samples // 2)
    if n_folds < 2:
        n_folds = 2
    
    # Select model
    if alpha_values is None:
        alpha_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    if model_type == "ridge":
        from sklearn.linear_model import RidgeCV
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", RidgeCV(alphas=alpha_values))
        ])
    elif model_type == "elasticnet":
        from sklearn.linear_model import ElasticNetCV
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", ElasticNetCV(alphas=alpha_values, l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=5000))
        ])
    elif model_type == "rf":
        from sklearn.ensemble import RandomForestRegressor
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42))
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Cross-validated predictions
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    try:
        cv_predictions = cross_val_predict(model, X_clean, y_clean, cv=cv)
        cv_r2 = r2_score(y_clean, cv_predictions)
        cv_mae = mean_absolute_error(y_clean, cv_predictions)
    except Exception:
        cv_predictions = np.full_like(y_clean, np.nan)
        cv_r2 = np.nan
        cv_mae = np.nan
    
    # Fit final model for feature weights
    try:
        model.fit(X_clean, y_clean)
        if hasattr(model.named_steps["regressor"], "coef_"):
            feature_weights = model.named_steps["regressor"].coef_
        elif hasattr(model.named_steps["regressor"], "feature_importances_"):
            feature_weights = model.named_steps["regressor"].feature_importances_
        else:
            feature_weights = np.full(X.shape[1], np.nan)
    except Exception:
        feature_weights = np.full(X.shape[1], np.nan)
    
    # Permutation null
    null_r2 = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y_clean)
        try:
            cv_pred_perm = cross_val_predict(model, X_clean, y_perm, cv=cv)
            null_r2.append(r2_score(y_perm, cv_pred_perm))
        except Exception:
            pass
    
    null_r2 = np.array(null_r2)
    
    # Permutation p-value
    if len(null_r2) > 0 and np.isfinite(cv_r2):
        p_value = (np.sum(null_r2 >= cv_r2) + 1) / (len(null_r2) + 1)
    else:
        p_value = np.nan
    
    # Expand predictions to original size
    full_predictions = np.full_like(y, np.nan)
    full_predictions[valid_mask] = cv_predictions
    
    return {
        "cv_r2": float(cv_r2),
        "cv_mae": float(cv_mae),
        "cv_predictions": full_predictions,
        "feature_weights": feature_weights,
        "null_r2": null_r2,
        "p_value": float(p_value),
        "n_samples": n_samples,
        "n_features": X.shape[1],
        "model_type": model_type,
    }


def compute_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute calibration curve for regression predictions.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    n_bins : int
        Number of bins.
    
    Returns
    -------
    bin_centers : np.ndarray
        Center of each prediction bin.
    mean_true : np.ndarray
        Mean true value in each bin.
    bin_counts : np.ndarray
        Number of samples in each bin.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    
    if len(y_true) < n_bins:
        return np.array([]), np.array([]), np.array([])
    
    # Bin by predicted values
    bin_edges = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-10  # Include max value
    
    bin_centers = []
    mean_true = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append(np.mean(y_pred[mask]))
            mean_true.append(np.mean(y_true[mask]))
            bin_counts.append(mask.sum())
    
    return np.array(bin_centers), np.array(mean_true), np.array(bin_counts)


###################################################################
# Power Analysis Utilities
###################################################################


def compute_required_n_for_correlation(
    r: float,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> int:
    """Compute required sample size to detect a correlation.
    
    Parameters
    ----------
    r : float
        Expected correlation coefficient.
    power : float
        Desired statistical power.
    alpha : float
        Significance level.
    alternative : str
        'two-sided' or 'one-sided'.
    
    Returns
    -------
    int
        Required sample size.
    """
    if abs(r) < 0.01:
        return 999999  # Effectively infinite
    
    # Fisher z transformation
    z_r = np.arctanh(np.clip(r, -0.999, 0.999))
    
    # Z-scores for alpha and power
    if alternative == "two-sided":
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    
    z_beta = stats.norm.ppf(power)
    
    # Required n
    n = ((z_alpha + z_beta) / z_r) ** 2 + 3
    
    return int(np.ceil(n))


def assess_statistical_power(
    n: int,
    r: float,
    alpha: float = 0.05,
) -> float:
    """Assess statistical power for a correlation test.
    
    Parameters
    ----------
    n : int
        Sample size.
    r : float
        Observed or expected correlation.
    alpha : float
        Significance level.
    
    Returns
    -------
    float
        Statistical power (0-1).
    """
    if n < 4 or abs(r) < 0.001:
        return 0.0
    
    # Fisher z transformation
    z_r = np.arctanh(np.clip(r, -0.999, 0.999))
    
    # Standard error
    se = 1 / np.sqrt(n - 3)
    
    # Non-centrality parameter
    ncp = z_r / se
    
    # Critical value
    z_crit = stats.norm.ppf(1 - alpha / 2)
    
    # Power
    power = 1 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)
    
    return float(np.clip(power, 0, 1))


def is_underpowered(
    n: int,
    r: float,
    min_power: float = 0.5,
    alpha: float = 0.05,
) -> bool:
    """Check if a correlation test is underpowered.
    
    Parameters
    ----------
    n : int
        Sample size.
    r : float
        Observed correlation.
    min_power : float
        Minimum acceptable power.
    alpha : float
        Significance level.
    
    Returns
    -------
    bool
        True if underpowered.
    """
    power = assess_statistical_power(n, r, alpha)
    return power < min_power


###################################################################
# Feature-Extractor-Based Split-Half Reliability
###################################################################


def compute_feature_split_half_reliability(
    feature_matrix: np.ndarray,
    ratings: np.ndarray,
    n_boot: int,
    use_spearman: bool,
    rng: np.random.Generator,
    min_samples_per_split: int = 10,
) -> Tuple[float, float, float]:
    """Split-half reliability across feature-level correlations.
    
    Parameters
    ----------
    feature_matrix : np.ndarray
        Feature matrix of shape (n_trials, n_features)
    ratings : np.ndarray
        Target values of shape (n_trials,)
    n_boot : int
        Number of bootstrap iterations
    use_spearman : bool
        Whether to use Spearman correlation
    rng : np.random.Generator
        Random number generator
    min_samples_per_split : int
        Minimum samples required per split for valid correlation estimation.
        Default is 10 to ensure stable correlation estimates.
    
    Returns
    -------
    Tuple[float, float, float]
        (median reliability, 2.5th percentile, 97.5th percentile)
    """
    if feature_matrix.size == 0 or ratings.size == 0:
        return np.nan, np.nan, np.nan
    n_trials = feature_matrix.shape[0]
    
    min_total_samples = 2 * min_samples_per_split
    if n_trials < min_total_samples:
        return np.nan, np.nan, np.nan

    rel_values = []
    for _ in range(int(n_boot)):
        idx = rng.permutation(n_trials)
        half = n_trials // 2
        
        if half < min_samples_per_split:
            continue
            
        idx_a, idx_b = idx[:half], idx[half:]
        Xa, Xb = feature_matrix[idx_a], feature_matrix[idx_b]
        ya, yb = ratings[idx_a], ratings[idx_b]

        ra = []
        rb = []
        for col in range(feature_matrix.shape[1]):
            fa = Xa[:, col]
            fb = Xb[:, col]
            if use_spearman:
                r_a, _ = stats.spearmanr(fa, ya)
                r_b, _ = stats.spearmanr(fb, yb)
            else:
                r_a, _ = stats.pearsonr(fa, ya)
                r_b, _ = stats.pearsonr(fb, yb)
            ra.append(r_a if np.isfinite(r_a) else np.nan)
            rb.append(r_b if np.isfinite(r_b) else np.nan)

        ra = np.asarray(ra, dtype=float)
        rb = np.asarray(rb, dtype=float)
        mask = np.isfinite(ra) & np.isfinite(rb)
        if not np.any(mask):
            continue
        r_half, _ = stats.spearmanr(ra[mask], rb[mask]) if use_spearman else stats.pearsonr(ra[mask], rb[mask])
        if np.isfinite(r_half):
            rel_values.append((2 * r_half) / (1 + r_half))  # Spearman-Brown

    if not rel_values:
        return np.nan, np.nan, np.nan
    rel_values = np.asarray(rel_values, dtype=float)
    return (
        float(np.nanmedian(rel_values)),
        float(np.nanpercentile(rel_values, 2.5)),
        float(np.nanpercentile(rel_values, 97.5)),
    )


def compute_correlation_split_half_reliability(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "spearman",
    n_splits: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Compute split-half reliability for a single correlation with Spearman-Brown correction.
    
    Parameters
    ----------
    x : np.ndarray
        First variable
    y : np.ndarray
        Second variable (e.g., ratings)
    method : str
        Correlation method ('spearman' or 'pearson')
    n_splits : int
        Number of random splits
    rng : np.random.Generator, optional
        Random number generator
        
    Returns
    -------
    float
        Spearman-Brown corrected reliability
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    valid = np.isfinite(x) & np.isfinite(y)
    n_valid = int(valid.sum())
    
    if n_valid < 20:
        return np.nan
    
    x_v, y_v = x[valid], y[valid]
    indices = np.arange(n_valid)
    
    correlations = []
    for _ in range(n_splits):
        rng.shuffle(indices)
        half = n_valid // 2
        idx1, idx2 = indices[:half], indices[half:2*half]
        
        if method == "spearman":
            r1, _ = stats.spearmanr(x_v[idx1], y_v[idx1])
            r2, _ = stats.spearmanr(x_v[idx2], y_v[idx2])
        else:
            r1, _ = stats.pearsonr(x_v[idx1], y_v[idx1])
            r2, _ = stats.pearsonr(x_v[idx2], y_v[idx2])
        
        if np.isfinite(r1) and np.isfinite(r2):
            correlations.append((r1 + r2) / 2)
    
    if not correlations:
        return np.nan
    
    r_half = np.mean(correlations)
    if r_half <= -1 or not np.isfinite(r_half):
        return np.nan
    return float((2 * r_half) / (1 + r_half))


def get_subject_seed(base_seed: int, subject: str) -> int:
    """Generate a reproducible seed for a subject."""
    import hashlib
    digest = hashlib.sha256(f"{base_seed}:{subject}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**31)


###################################################################
# DataFrame-Based Reliability (Wide Format)
###################################################################

from dataclasses import dataclass
from typing import Callable

try:
    from eeg_pipeline.types import ProgressCallback, null_progress
except ImportError:
    ProgressCallback = Callable[[str, float], None]
    def null_progress(msg: str, frac: float) -> None:
        pass


@dataclass
class ReliabilityResult:
    """Result of reliability computation for a single feature."""
    name: str
    reliability: float
    ci_lower: float
    ci_upper: float
    n_samples: int
    method: str
    
    def is_acceptable(self, threshold: float = 0.7) -> bool:
        return self.reliability >= threshold
    
    def is_good(self, threshold: float = 0.8) -> bool:
        return self.reliability >= threshold
    
    def is_excellent(self, threshold: float = 0.9) -> bool:
        return self.reliability >= threshold


def _pearson_with_ci(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Compute Pearson correlation with 95% CI via Fisher z-transform."""
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]
    n = len(x_clean)
    if n < 3:
        return np.nan, np.nan, np.nan
    r, _ = stats.pearsonr(x_clean, y_clean)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    return float(r), float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def _spearman_brown(r: float) -> float:
    """Apply Spearman-Brown prophecy formula."""
    if np.isnan(r) or r <= -1:
        return np.nan
    return (2 * r) / (1 + r)


def compute_odd_even_reliability(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    *,
    exclude_columns: Optional[List[str]] = None,
) -> Dict[str, ReliabilityResult]:
    """Compute odd-even split reliability for DataFrame features.
    
    Splits trials into odd and even numbered, then correlates values between halves.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with one row per trial/epoch
    feature_columns : Optional[List[str]]
        Columns to compute reliability for (if None, uses all numeric)
    exclude_columns : Optional[List[str]]
        Columns to exclude
    
    Returns
    -------
    Dict[str, ReliabilityResult]
        Reliability results per feature
    """
    if exclude_columns is None:
        exclude_columns = ["condition", "epoch", "trial", "subject", "run"]
    
    if feature_columns is None:
        feature_columns = [c for c in df.columns 
                         if c not in exclude_columns
                         and pd.api.types.is_numeric_dtype(df[c])]
    
    n_rows = len(df)
    if n_rows < 4:
        return {}
    
    odd_mask = np.arange(n_rows) % 2 == 1
    even_mask = ~odd_mask
    results = {}
    
    for col in feature_columns:
        values = df[col].to_numpy(dtype=float)
        odd_vals = values[odd_mask]
        even_vals = values[even_mask]
        min_len = min(len(odd_vals), len(even_vals))
        odd_vals, even_vals = odd_vals[:min_len], even_vals[:min_len]
        
        r, ci_lower, ci_upper = _pearson_with_ci(odd_vals, even_vals)
        reliability = _spearman_brown(r)
        ci_lower_sb = _spearman_brown(ci_lower) if np.isfinite(ci_lower) else np.nan
        ci_upper_sb = _spearman_brown(ci_upper) if np.isfinite(ci_upper) else np.nan
        
        results[col] = ReliabilityResult(
            name=col, reliability=reliability,
            ci_lower=ci_lower_sb, ci_upper=ci_upper_sb,
            n_samples=min_len, method="odd_even",
        )
    return results


def compute_bootstrap_reliability(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    *,
    n_iterations: int = 1000,
    sample_fraction: float = 0.5,
    random_state: int = 42,
    exclude_columns: Optional[List[str]] = None,
    progress: ProgressCallback = None,
) -> Dict[str, ReliabilityResult]:
    """Compute bootstrap reliability estimates.
    
    Repeatedly samples subsets and computes mean features, then correlates between samples.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame
    feature_columns : Optional[List[str]]
        Columns to analyze
    n_iterations : int
        Number of bootstrap iterations
    sample_fraction : float
        Fraction of data to sample each iteration
    random_state : int
        Random seed
    exclude_columns : Optional[List[str]]
        Columns to exclude
    progress : ProgressCallback
        Progress callback
    
    Returns
    -------
    Dict[str, ReliabilityResult]
        Reliability results per feature
    """
    if progress is None:
        progress = null_progress
    if exclude_columns is None:
        exclude_columns = ["condition", "epoch", "trial", "subject", "run"]
    
    if feature_columns is None:
        feature_columns = [c for c in df.columns 
                         if c not in exclude_columns
                         and pd.api.types.is_numeric_dtype(df[c])]
    
    n_rows = len(df)
    sample_size = int(n_rows * sample_fraction)
    if sample_size < 2:
        return {}
    
    rng = np.random.default_rng(random_state)
    bootstrap_means: Dict[str, List[float]] = {col: [] for col in feature_columns}
    
    for i in range(n_iterations):
        if i % 100 == 0:
            progress("Bootstrap", i / n_iterations)
        idx = rng.choice(n_rows, size=sample_size, replace=True)
        sample = df.iloc[idx]
        for col in feature_columns:
            bootstrap_means[col].append(np.nanmean(sample[col].to_numpy(dtype=float)))
    
    progress("Computing reliability", 0.9)
    results = {}
    
    for col in feature_columns:
        means = np.array(bootstrap_means[col])
        finite_means = means[np.isfinite(means)]
        
        if len(finite_means) < 10:
            results[col] = ReliabilityResult(
                name=col, reliability=np.nan, ci_lower=np.nan, ci_upper=np.nan,
                n_samples=len(finite_means), method="bootstrap",
            )
            continue
        
        half = len(finite_means) // 2
        r, ci_lower, ci_upper = _pearson_with_ci(finite_means[:half], finite_means[half:2*half])
        reliability = _spearman_brown(r)
        ci_lower_sb = _spearman_brown(ci_lower) if np.isfinite(ci_lower) else np.nan
        ci_upper_sb = _spearman_brown(ci_upper) if np.isfinite(ci_upper) else np.nan
        
        results[col] = ReliabilityResult(
            name=col, reliability=reliability,
            ci_lower=ci_lower_sb, ci_upper=ci_upper_sb,
            n_samples=len(finite_means), method="bootstrap",
        )
    
    progress("Complete", 1.0)
    return results


def compute_dataframe_reliability(
    df: pd.DataFrame,
    method: str = "odd_even",
    *,
    feature_columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None,
    n_iterations: int = 100,
    random_state: int = 42,
    progress: ProgressCallback = None,
) -> pd.DataFrame:
    """Compute reliability for all features in a wide-format DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame (one row per trial, columns are features)
    method : str
        Method: "odd_even", "bootstrap"
    feature_columns : Optional[List[str]]
        Columns to analyze
    exclude_columns : Optional[List[str]]
        Columns to exclude
    n_iterations : int
        Number of iterations (for bootstrap)
    random_state : int
        Random seed
    progress : ProgressCallback
        Progress callback
    
    Returns
    -------
    pd.DataFrame
        Reliability results with columns:
        [name, reliability, ci_lower, ci_upper, n_samples, method, is_acceptable, is_good]
    """
    if progress is None:
        progress = null_progress
        
    if method == "odd_even":
        results = compute_odd_even_reliability(df, feature_columns, exclude_columns=exclude_columns)
    elif method == "bootstrap":
        results = compute_bootstrap_reliability(
            df, feature_columns, n_iterations=n_iterations,
            random_state=random_state, exclude_columns=exclude_columns, progress=progress,
        )
    else:
        raise ValueError(f"Unknown reliability method: {method}")
    
    records = []
    for name, result in results.items():
        records.append({
            "name": result.name,
            "reliability": result.reliability,
            "ci_lower": result.ci_lower,
            "ci_upper": result.ci_upper,
            "n_samples": result.n_samples,
            "method": result.method,
            "is_acceptable": result.is_acceptable(),
            "is_good": result.is_good(),
            "is_excellent": result.is_excellent(),
        })
    return pd.DataFrame(records)


def filter_reliable_features(
    df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    threshold: float = 0.7,
    *,
    keep_metadata_cols: bool = True,
    metadata_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Filter features to keep only those meeting reliability threshold.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original feature DataFrame
    reliability_df : pd.DataFrame
        Reliability results from compute_dataframe_reliability
    threshold : float
        Minimum reliability to keep
    keep_metadata_cols : bool
        Keep metadata columns
    metadata_cols : Optional[List[str]]
        Metadata columns to keep
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    if metadata_cols is None:
        metadata_cols = ["condition", "epoch", "trial", "subject", "run"]
    
    reliable = reliability_df[reliability_df["reliability"] >= threshold]["name"].tolist()
    
    if keep_metadata_cols:
        cols_to_keep = [c for c in metadata_cols if c in df.columns] + reliable
    else:
        cols_to_keep = reliable
    
    return df[cols_to_keep].copy()


###################################################################
# Exports
###################################################################

__all__ = [
    # ICC and reliability
    "compute_icc",
    "compute_split_half_reliability",
    "compute_feature_reliability",
    "compute_feature_split_half_reliability",
    "compute_correlation_split_half_reliability",
    "get_subject_seed",
    # DataFrame-based reliability (wide format)
    "ReliabilityResult",
    "compute_odd_even_reliability",
    "compute_bootstrap_reliability",
    "compute_dataframe_reliability",
    "filter_reliable_features",
    # Hierarchical FDR
    "hierarchical_fdr",
    "compute_hierarchical_fdr_summary",
    # Predictive validity
    "cross_validated_prediction",
    "compute_calibration_curve",
    # Power analysis
    "compute_required_n_for_correlation",
    "assess_statistical_power",
    "is_underpowered",
]
