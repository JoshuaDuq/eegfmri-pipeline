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

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
from scipy import stats


###################################################################
# Constants
###################################################################

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8
DEFAULT_RANDOM_STATE = 42
MIN_SAMPLES_FOR_CORRELATION = 3
MIN_SAMPLES_FOR_SPLIT_HALF = 4
MIN_SAMPLES_FOR_RELIABILITY = 10
MIN_SAMPLES_PER_SPLIT = 10
MIN_SAMPLES_FOR_CALIBRATION = 10
MIN_SAMPLES_FOR_POWER = 4
RELIABILITY_THRESHOLD_ACCEPTABLE = 0.7
RELIABILITY_THRESHOLD_GOOD = 0.8
RELIABILITY_THRESHOLD_EXCELLENT = 0.9
DEFAULT_N_SPLITS = 100
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_N_FOLDS = 5
DEFAULT_N_PERMUTATIONS = 100
DEFAULT_N_BINS = 10
DEFAULT_SAMPLE_FRACTION = 0.5
MIN_CORRELATION_FOR_POWER = 0.001
EFFECTIVELY_INFINITE_N = 999999
DEFAULT_ALPHA_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
DEFAULT_ELASTICNET_L1_RATIOS = [0.1, 0.5, 0.9]
DEFAULT_RF_N_ESTIMATORS = 100
DEFAULT_RF_MAX_DEPTH = 5
MIN_SAMPLES_FOR_CV = 10
MIN_FOLDS = 2
MIN_SAMPLES_FOR_CORRELATION_SPLIT_HALF = 20


###################################################################
# Intraclass Correlation Coefficient (ICC)
###################################################################


def _compute_icc_one_way_single(ms_rows: float, ms_within: float, k: int) -> float:
    """Compute ICC(1,1): One-way random, single rater."""
    return (ms_rows - ms_within) / (ms_rows + (k - 1) * ms_within)


def _compute_icc_two_way_random_single(ms_rows: float, ms_error: float, ms_cols: float, n: int, k: int) -> float:
    """Compute ICC(2,1): Two-way random, single rater."""
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    return (ms_rows - ms_error) / denominator


def _compute_icc_two_way_mixed_single(ms_rows: float, ms_error: float, k: int) -> float:
    """Compute ICC(3,1): Two-way mixed, single rater."""
    return (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error)


def _compute_icc_one_way_average(ms_rows: float, ms_within: float) -> float:
    """Compute ICC(1,k): One-way random, average of k raters."""
    return (ms_rows - ms_within) / ms_rows


def _compute_icc_two_way_random_average(ms_rows: float, ms_error: float, ms_cols: float, n: int) -> float:
    """Compute ICC(2,k): Two-way random, average of k raters."""
    return (ms_rows - ms_error) / (ms_rows + (ms_cols - ms_error) / n)


def _compute_icc_two_way_mixed_average(ms_rows: float, ms_error: float) -> float:
    """Compute ICC(3,k): Two-way mixed, average of k raters."""
    return (ms_rows - ms_error) / ms_rows


def _compute_icc_confidence_intervals(
    ms_rows: float,
    ms_error: float,
    n: int,
    k: int,
) -> Tuple[float, float]:
    """Compute 95% confidence intervals for ICC using F-distribution."""
    if ms_error <= 0:
        return np.nan, np.nan
    
    f_value = ms_rows / ms_error
    if not (np.isfinite(f_value) and f_value > 0):
        return np.nan, np.nan
    
    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    f_critical_upper = stats.f.ppf(0.975, df1, df2)
    f_critical_lower = stats.f.ppf(0.975, df2, df1)
    
    f_low = f_value / f_critical_upper
    f_high = f_value * f_critical_lower
    
    ci_low = (f_low - 1) / (f_low + k - 1)
    ci_high = (f_high - 1) / (f_high + k - 1)
    
    return float(ci_low), float(ci_high)


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
    
    n_subjects, n_raters = data.shape
    if n_subjects < 2 or n_raters < 2:
        return np.nan, np.nan, np.nan
    
    grand_mean = np.mean(data)
    row_means = np.mean(data, axis=1)
    col_means = np.mean(data, axis=0)
    
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_rows = n_raters * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n_subjects * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols
    
    ms_rows = ss_rows / (n_subjects - 1)
    ms_cols = ss_cols / (n_raters - 1)
    ms_error = ss_error / ((n_subjects - 1) * (n_raters - 1))
    
    icc_type_upper = icc_type.upper()
    if icc_type_upper in ["ICC(1,1)", "ICC1"]:
        ms_within = (ss_cols + ss_error) / (n_subjects * (n_raters - 1))
        icc = _compute_icc_one_way_single(ms_rows, ms_within, n_raters)
    elif icc_type_upper in ["ICC(2,1)", "ICC2"]:
        icc = _compute_icc_two_way_random_single(ms_rows, ms_error, ms_cols, n_subjects, n_raters)
    elif icc_type_upper in ["ICC(3,1)", "ICC3"]:
        icc = _compute_icc_two_way_mixed_single(ms_rows, ms_error, n_raters)
    elif icc_type_upper in ["ICC(1,K)", "ICC1K"]:
        ms_within = (ss_cols + ss_error) / (n_subjects * (n_raters - 1))
        icc = _compute_icc_one_way_average(ms_rows, ms_within)
    elif icc_type_upper in ["ICC(2,K)", "ICC2K"]:
        icc = _compute_icc_two_way_random_average(ms_rows, ms_error, ms_cols, n_subjects)
    elif icc_type_upper in ["ICC(3,K)", "ICC3K"]:
        icc = _compute_icc_two_way_mixed_average(ms_rows, ms_error)
    else:
        raise ValueError(f"Unknown ICC type: {icc_type}")
    
    ci_low, ci_high = _compute_icc_confidence_intervals(ms_rows, ms_error, n_subjects, n_raters)
    icc_clipped = float(np.clip(icc, -1, 1))
    
    return icc_clipped, ci_low, ci_high


def compute_icc_from_dataframe(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    icc_type: str = "ICC(1,1)",
) -> Tuple[float, Tuple[float, float]]:
    """Compute Intraclass Correlation Coefficient from DataFrame.
    
    This is a convenience wrapper that converts a DataFrame to the array format
    required by compute_icc.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with value and group columns
    value_col : str
        Column name containing values
    group_col : str
        Column name containing group identifiers
    icc_type : str
        Type of ICC to compute (default: "ICC(1,1)")
    
    Returns
    -------
    icc : float
        ICC value
    ci : Tuple[float, float]
        (ci_low, ci_high) confidence interval bounds
    """
    if value_col not in df.columns or group_col not in df.columns:
        return np.nan, (np.nan, np.nan)

    df_clean = df[[value_col, group_col]].dropna()

    groups = df_clean[group_col].unique()
    n_groups = len(groups)

    MIN_GROUPS_FOR_ICC = 2
    MIN_PIVOT_ROWS_FOR_ICC = 2
    MIN_PIVOT_COLS_FOR_ICC = 2

    if n_groups < MIN_GROUPS_FOR_ICC:
        return np.nan, (np.nan, np.nan)

    try:
        pivot = df_clean.pivot_table(
            index=group_col,
            columns=df_clean.groupby(group_col).cumcount(),
            values=value_col,
            aggfunc="first"
        ).dropna()

        n_rows, n_cols = pivot.shape
        if n_rows < MIN_PIVOT_ROWS_FOR_ICC or n_cols < MIN_PIVOT_COLS_FOR_ICC:
            return np.nan, (np.nan, np.nan)

        icc, ci_low, ci_high = compute_icc(pivot.values, icc_type=icc_type)
        return float(icc), (float(ci_low), float(ci_high))
    except (ValueError, KeyError, IndexError):
        return np.nan, (np.nan, np.nan)


def _apply_spearman_brown(r: float) -> float:
    """Apply Spearman-Brown prophecy formula."""
    if r <= -1 or not np.isfinite(r):
        return np.nan
    return (2 * r) / (1 + r)


def compute_split_half_reliability(
    data: np.ndarray,
    n_splits: int = DEFAULT_N_SPLITS,
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
    if n_trials < MIN_SAMPLES_FOR_SPLIT_HALF:
        return np.nan, np.nan, np.nan
    
    correlations = []
    half_size = n_trials // 2
    
    for _ in range(n_splits):
        indices = rng.permutation(n_trials)
        half1_indices = indices[:half_size]
        half2_indices = indices[half_size:2 * half_size]
        
        half1_means = data[half1_indices].mean(axis=0)
        half2_means = data[half2_indices].mean(axis=0)
        
        from .correlation import compute_correlation
        if len(half1_means) == 1:
            half1_values = data[half1_indices, 0]
            half2_values = data[half2_indices, 0]
            r, _ = compute_correlation(half1_values, half2_values, method)
        else:
            r, _ = compute_correlation(half1_means, half2_means, method)
        r = r if np.isfinite(r) else np.nan
        
        if np.isfinite(r):
            correlations.append(r)
    
    if not correlations:
        return np.nan, np.nan, np.nan
    
    mean_correlation = np.mean(correlations)
    reliability = _apply_spearman_brown(mean_correlation)
    
    boot_reliabilities = [
        _apply_spearman_brown(r) for r in correlations
        if np.isfinite(_apply_spearman_brown(r))
    ]
    
    min_samples_for_ci = 10
    if len(boot_reliabilities) > min_samples_for_ci:
        ci_low = float(np.percentile(boot_reliabilities, 2.5))
        ci_high = float(np.percentile(boot_reliabilities, 97.5))
    else:
        ci_low, ci_high = np.nan, np.nan
    
    return float(reliability), ci_low, ci_high


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


def hierarchical_fdr_dict(
    p_values: Dict[str, np.ndarray],
    alpha: float = DEFAULT_ALPHA,
    method: str = "bh",
) -> Dict[str, Dict[str, Any]]:
    """Apply hierarchical FDR correction across multiple families (dict API).
    
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
    alpha: float = DEFAULT_ALPHA,
    config: Optional[Any] = None,
    include_glob: Union[str, Iterable[str]] = "corr_stats_*.tsv",
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
    include_glob : str or Iterable[str]
        Glob pattern(s) for files to include.
    
    Returns
    -------
    pd.DataFrame
        Summary with hierarchical FDR results.
    """
    from pathlib import Path
    from eeg_pipeline.infra.tsv import read_tsv
    from eeg_pipeline.utils.analysis.stats.fdr import infer_fdr_family, select_p_column_for_fdr
    
    stats_dir = Path(stats_dir)
    
    if isinstance(include_glob, str):
        files = list(stats_dir.glob(include_glob))
    else:
        files = []
        for pat in include_glob:
            files.extend(list(stats_dir.glob(pat)))
        seen = set()
        files = [f for f in files if not (f in seen or seen.add(f))]

    def _extract_feature_family(family: str) -> str:
        if "|features:" in family:
            return str(family.split("|features:", 1)[1]).strip()
        return str(family)

    # Group files by feature family, using the same inference as apply_global_fdr
    analysis_groups: Dict[str, List[Path]] = {}
    for fpath in files:
        df = read_tsv(fpath)
        if df is None or df.empty:
            continue

        family = infer_fdr_family(fpath, df)
        group_name = _extract_feature_family(family)
        analysis_groups.setdefault(group_name, []).append(fpath)
    
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
            
            p_col = select_p_column_for_fdr(df)
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
    
    hier_results = hierarchical_fdr_dict(p_by_group, alpha=alpha)
    
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


def _create_ridge_model(alpha_values: List[float]) -> Any:
    """Create Ridge regression model with cross-validation."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RidgeCV(alphas=alpha_values))
    ])


def _create_elasticnet_model(alpha_values: List[float]) -> Any:
    """Create ElasticNet regression model with cross-validation."""
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", ElasticNetCV(
            alphas=alpha_values,
            l1_ratio=DEFAULT_ELASTICNET_L1_RATIOS,
            cv=3,
            max_iter=5000
        ))
    ])


def _create_random_forest_model() -> Any:
    """Create Random Forest regression model."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor(
            n_estimators=DEFAULT_RF_N_ESTIMATORS,
            max_depth=DEFAULT_RF_MAX_DEPTH,
            random_state=DEFAULT_RANDOM_STATE
        ))
    ])


def _create_model(model_type: str, alpha_values: Optional[List[float]]) -> Any:
    """Create model based on type."""
    if alpha_values is None:
        alpha_values = DEFAULT_ALPHA_VALUES
    
    if model_type == "ridge":
        return _create_ridge_model(alpha_values)
    elif model_type == "elasticnet":
        return _create_elasticnet_model(alpha_values)
    elif model_type == "rf":
        return _create_random_forest_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _extract_feature_weights(model: Any, n_features: int) -> np.ndarray:
    """Extract feature weights from fitted model."""
    regressor = model.named_steps["regressor"]
    if hasattr(regressor, "coef_"):
        return regressor.coef_
    elif hasattr(regressor, "feature_importances_"):
        return regressor.feature_importances_
    else:
        return np.full(n_features, np.nan)


def _compute_cv_predictions(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
) -> Tuple[np.ndarray, float, float]:
    """Compute cross-validated predictions and metrics."""
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.metrics import r2_score, mean_absolute_error
    
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
    
    try:
        predictions = cross_val_predict(model, X, y, cv=cv)
        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)
        return predictions, r2, mae
    except (ValueError, RuntimeError, AttributeError) as e:
        logging.warning(f"CV prediction failed: {e}")
        return np.full_like(y, np.nan), np.nan, np.nan


def cross_validated_prediction(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "ridge",
    n_folds: int = DEFAULT_N_FOLDS,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
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
    from sklearn.metrics import r2_score
    
    if rng is None:
        rng = np.random.default_rng()
    
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    
    valid_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    
    n_samples = len(y_clean)
    if n_samples < MIN_SAMPLES_FOR_CV:
        return {
            "cv_r2": np.nan,
            "cv_mae": np.nan,
            "cv_predictions": np.full_like(y, np.nan),
            "feature_weights": np.full(X.shape[1], np.nan),
            "null_r2": np.array([]),
            "p_value": np.nan,
        }
    
    n_folds_adjusted = max(MIN_FOLDS, min(n_folds, n_samples // 2))
    model = _create_model(model_type, alpha_values)
    
    cv_predictions, cv_r2, cv_mae = _compute_cv_predictions(
        model, X_clean, y_clean, n_folds_adjusted
    )
    
    try:
        model.fit(X_clean, y_clean)
        feature_weights = _extract_feature_weights(model, X.shape[1])
    except (ValueError, RuntimeError, AttributeError) as e:
        logging.warning(f"Model fitting failed: {e}")
        feature_weights = np.full(X.shape[1], np.nan)
    
    cv = KFold(n_splits=n_folds_adjusted, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
    null_r2 = []
    for _ in range(n_permutations):
        y_permuted = rng.permutation(y_clean)
        try:
            predictions_perm = cross_val_predict(model, X_clean, y_permuted, cv=cv)
            null_r2.append(r2_score(y_permuted, predictions_perm))
        except (ValueError, RuntimeError, AttributeError):
            continue
    
    null_r2_array = np.array(null_r2)
    
    if len(null_r2_array) > 0 and np.isfinite(cv_r2):
        n_exceeding = np.sum(null_r2_array >= cv_r2)
        p_value = (n_exceeding + 1) / (len(null_r2_array) + 1)
    else:
        p_value = np.nan
    
    full_predictions = np.full_like(y, np.nan)
    full_predictions[valid_mask] = cv_predictions
    
    return {
        "cv_r2": float(cv_r2),
        "cv_mae": float(cv_mae),
        "cv_predictions": full_predictions,
        "feature_weights": feature_weights,
        "null_r2": null_r2_array,
        "p_value": float(p_value),
        "n_samples": n_samples,
        "n_features": X.shape[1],
        "model_type": model_type,
    }


def compute_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
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
    
    percentile_levels = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(y_pred, percentile_levels)
    bin_edges[-1] += 1e-10
    
    bin_centers = []
    mean_true_values = []
    bin_counts = []
    
    for i in range(n_bins):
        in_bin_mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        n_in_bin = in_bin_mask.sum()
        if n_in_bin > 0:
            bin_centers.append(np.mean(y_pred[in_bin_mask]))
            mean_true_values.append(np.mean(y_true[in_bin_mask]))
            bin_counts.append(n_in_bin)
    
    return (
        np.array(bin_centers),
        np.array(mean_true_values),
        np.array(bin_counts),
    )


###################################################################
# Power Analysis Utilities
###################################################################


def compute_required_n_for_correlation(
    r: float,
    power: float = DEFAULT_POWER,
    alpha: float = DEFAULT_ALPHA,
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
    min_correlation_for_power = 0.01
    if abs(r) < min_correlation_for_power:
        return EFFECTIVELY_INFINITE_N
    
    max_correlation_for_transform = 0.999
    z_r = np.arctanh(np.clip(r, -max_correlation_for_transform, max_correlation_for_transform))
    
    if alternative == "two-sided":
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    
    z_beta = stats.norm.ppf(power)
    n_adjustment = 3
    n = ((z_alpha + z_beta) / z_r) ** 2 + n_adjustment
    
    return int(np.ceil(n))


def assess_statistical_power(
    n: int,
    r: float,
    alpha: float = DEFAULT_ALPHA,
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
    if n < MIN_SAMPLES_FOR_POWER or abs(r) < MIN_CORRELATION_FOR_POWER:
        return 0.0
    
    max_correlation_for_transform = 0.999
    z_r = np.arctanh(np.clip(r, -max_correlation_for_transform, max_correlation_for_transform))
    
    df_adjustment = 3
    standard_error = 1 / np.sqrt(n - df_adjustment)
    
    non_centrality_parameter = z_r / standard_error
    
    z_critical = stats.norm.ppf(1 - alpha / 2)
    
    power = (
        1 - stats.norm.cdf(z_critical - non_centrality_parameter) +
        stats.norm.cdf(-z_critical - non_centrality_parameter)
    )
    
    return float(np.clip(power, 0, 1))


def is_underpowered(
    n: int,
    r: float,
    min_power: float = 0.5,
    alpha: float = DEFAULT_ALPHA,
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
    min_samples_per_split: int = MIN_SAMPLES_PER_SPLIT,
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

        correlations_a = []
        correlations_b = []
        method = "spearman" if use_spearman else "pearson"
        
        for col in range(feature_matrix.shape[1]):
            feature_a = Xa[:, col]
            feature_b = Xb[:, col]
            
            from .correlation import compute_correlation
            r_a, _ = compute_correlation(feature_a, ya, method)
            r_b, _ = compute_correlation(feature_b, yb, method)
            r_a = r_a if np.isfinite(r_a) else np.nan
            r_b = r_b if np.isfinite(r_b) else np.nan
            
            correlations_a.append(r_a)
            correlations_b.append(r_b)

        correlations_a = np.asarray(correlations_a, dtype=float)
        correlations_b = np.asarray(correlations_b, dtype=float)
        valid_mask = np.isfinite(correlations_a) & np.isfinite(correlations_b)
        
        if not np.any(valid_mask):
            continue
        
        from .correlation import compute_correlation
        r_half, _ = compute_correlation(
            correlations_a[valid_mask],
            correlations_b[valid_mask],
            method
        )
        r_half = r_half if np.isfinite(r_half) else np.nan
        
        if np.isfinite(r_half):
            reliability = _apply_spearman_brown(r_half)
            rel_values.append(reliability)

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
    n_splits: int = DEFAULT_N_SPLITS,
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
        rng = np.random.default_rng(DEFAULT_RANDOM_STATE)
    
    valid_mask = np.isfinite(x) & np.isfinite(y)
    n_valid = int(valid_mask.sum())
    
    if n_valid < MIN_SAMPLES_FOR_CORRELATION_SPLIT_HALF:
        return np.nan
    
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    indices = np.arange(n_valid)
    
    correlations = []
    half_size = n_valid // 2
    
    for _ in range(n_splits):
        rng.shuffle(indices)
        idx1 = indices[:half_size]
        idx2 = indices[half_size:2 * half_size]
        
        from .correlation import compute_correlation
        r1, _ = compute_correlation(x_valid[idx1], y_valid[idx1], method)
        r2, _ = compute_correlation(x_valid[idx2], y_valid[idx2], method)
        r1 = r1 if np.isfinite(r1) else np.nan
        r2 = r2 if np.isfinite(r2) else np.nan
        
        if np.isfinite(r1) and np.isfinite(r2):
            mean_correlation = (r1 + r2) / 2
            correlations.append(mean_correlation)
    
    if not correlations:
        return np.nan
    
    mean_half_correlation = np.mean(correlations)
    return float(_apply_spearman_brown(mean_half_correlation))


# Re-export from base for backwards compatibility
from .base import get_subject_seed


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
    
    def is_acceptable(self, threshold: float = RELIABILITY_THRESHOLD_ACCEPTABLE) -> bool:
        return self.reliability >= threshold
    
    def is_good(self, threshold: float = RELIABILITY_THRESHOLD_GOOD) -> bool:
        return self.reliability >= threshold
    
    def is_excellent(self, threshold: float = RELIABILITY_THRESHOLD_EXCELLENT) -> bool:
        return self.reliability >= threshold


def _pearson_with_ci(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Compute Pearson correlation with 95% CI via Fisher z-transform."""
    from .correlation import fisher_ci
    
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    n = len(x_clean)
    
    if n < MIN_SAMPLES_FOR_CORRELATION:
        return np.nan, np.nan, np.nan
    
    r, _ = stats.pearsonr(x_clean, y_clean)
    
    # Use consolidated fisher_ci function (defaults to 95% CI)
    ci_lower, ci_upper = fisher_ci(r, n, config=None)
    
    return float(r), ci_lower, ci_upper


# Alias for backwards compatibility
_spearman_brown = _apply_spearman_brown


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
    n_iterations: int = DEFAULT_N_BOOTSTRAP,
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
    random_state: int = DEFAULT_RANDOM_STATE,
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
    n_iterations: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
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
    threshold: float = RELIABILITY_THRESHOLD_ACCEPTABLE,
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
    "compute_icc_from_dataframe",
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
