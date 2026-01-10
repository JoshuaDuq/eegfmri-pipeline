"""
Regression and Model Diagnostics
================================

Statistical diagnostics for linear models including:
- Multicollinearity (Variance Inflation Factor)
- Influence (Cook's Distance, Leverage)
- Distributional checks (Normality summary)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Any, Dict, Tuple

from eeg_pipeline.utils.analysis.stats.validation import (
    check_normality_dagostino,
    check_normality_shapiro,
)
from eeg_pipeline.utils.analysis.stats._regression_utils import _ols_fit, _r2


_MIN_SAMPLE_SIZE_FOR_REGRESSION = 2
_MIN_VARIANCE_THRESHOLD = 1e-10
_MAX_R_SQUARED_FOR_VIF = 0.9999
_MIN_SAMPLE_SIZE_FOR_LEVERAGE = 4
_MIN_SAMPLE_SIZE_FOR_MOMENTS = 8
_MIN_SAMPLE_SIZE_FOR_DAGOSTINO = 20
_COOKS_THRESHOLD_FACTOR = 4.0
_NUMERICAL_STABILITY_EPSILON = 1e-12


def _compute_r_squared(
    design_matrix: np.ndarray,
    response: np.ndarray,
) -> float:
    """Compute R-squared from design matrix and response.
    
    Uses consolidated regression utilities for consistency.
    
    Parameters
    ----------
    design_matrix : np.ndarray
        Design matrix with intercept column
    response : np.ndarray
        Response variable
        
    Returns
    -------
    float
        R-squared value, clipped to [0, 0.9999]
    """
    beta = _ols_fit(design_matrix, response)
    if beta is None:
        return 0.0
    
    predicted = design_matrix @ beta
    r_squared = _r2(response, predicted)
    
    if not np.isfinite(r_squared):
        return 0.0
    
    return np.clip(r_squared, 0.0, _MAX_R_SQUARED_FOR_VIF)


def _compute_vif_for_predictor(
    predictor_values: np.ndarray,
    other_predictors: np.ndarray,
) -> float:
    """Compute VIF for a single predictor.
    
    Parameters
    ----------
    predictor_values : np.ndarray
        Values of the predictor being tested
    other_predictors : np.ndarray
        Values of all other predictors
        
    Returns
    -------
    float
        VIF value, or np.nan if computation fails
    """
    min_required_samples = other_predictors.shape[1] + _MIN_SAMPLE_SIZE_FOR_REGRESSION
    valid_mask = np.isfinite(predictor_values) & np.all(
        np.isfinite(other_predictors), axis=1
    )
    
    if valid_mask.sum() < min_required_samples:
        return np.nan
    
    valid_predictor = predictor_values[valid_mask]
    valid_other_predictors = other_predictors[valid_mask]
    
    intercept_column = np.ones(len(valid_predictor))
    design_matrix = np.column_stack([intercept_column, valid_other_predictors])
    
    try:
        r_squared = _compute_r_squared(design_matrix, valid_predictor)
        vif = 1.0 / (1.0 - r_squared)
        return float(vif)
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


def compute_vif(X: pd.DataFrame) -> pd.Series:
    """Compute Variance Inflation Factor for each predictor.
    
    VIF = 1 / (1 - R²) where R² is from regressing each predictor on others.
    VIF > 5: Moderate multicollinearity
    VIF > 10: High multicollinearity
    
    Parameters
    ----------
    X : pd.DataFrame
        Covariate matrix (each column is a predictor)
        
    Returns
    -------
    pd.Series
        VIF for each column
    """
    if X.empty:
        return pd.Series(dtype=float)
    
    num_predictors = X.shape[1]
    if num_predictors == 1:
        return pd.Series({X.columns[0]: 1.0})
    
    vif_values = {}
    
    for column_name in X.columns:
        predictor_values = pd.to_numeric(X[column_name], errors="coerce").values
        other_predictors_df = X.drop(columns=[column_name])
        other_predictors = other_predictors_df.apply(
            pd.to_numeric, errors="coerce"
        ).values
        
        vif_values[column_name] = _compute_vif_for_predictor(
            predictor_values, other_predictors
        )
    
    return pd.Series(vif_values)


def _compute_hat_matrix_diagonal(
    design_matrix: np.ndarray,
    covariance_matrix_inverse: np.ndarray,
) -> np.ndarray:
    """Compute diagonal of hat matrix.
    
    Parameters
    ----------
    design_matrix : np.ndarray
        Design matrix with intercept column
    covariance_matrix_inverse : np.ndarray
        Inverse of X.T @ X
        
    Returns
    -------
    np.ndarray
        Diagonal of hat matrix (leverage values)
    """
    hat_matrix = design_matrix @ covariance_matrix_inverse @ design_matrix.T
    return np.diag(hat_matrix)


def _compute_studentized_residuals(
    residuals: np.ndarray,
    mean_squared_error: float,
    leverage_values: np.ndarray,
) -> np.ndarray:
    """Compute studentized residuals.
    
    Parameters
    ----------
    residuals : np.ndarray
        Raw residuals
    mean_squared_error : float
        MSE from regression
    leverage_values : np.ndarray
        Hat values (leverage)
        
    Returns
    -------
    np.ndarray
        Studentized residuals
    """
    variance_estimate = mean_squared_error * (1.0 - leverage_values)
    denominator = np.sqrt(variance_estimate + _NUMERICAL_STABILITY_EPSILON)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        return residuals / denominator


def _compute_cooks_distance(
    studentized_residuals: np.ndarray,
    leverage_values: np.ndarray,
    num_parameters: int,
) -> np.ndarray:
    """Compute Cook's distance for each observation.
    
    Parameters
    ----------
    studentized_residuals : np.ndarray
        Studentized residuals
    leverage_values : np.ndarray
        Hat values (leverage)
    num_parameters : int
        Number of parameters in model (intercept + predictors)
        
    Returns
    -------
    np.ndarray
        Cook's distance for each observation
    """
    leverage_complement = 1.0 - leverage_values + _NUMERICAL_STABILITY_EPSILON
    
    with np.errstate(divide="ignore", invalid="ignore"):
        cooks_distance = (
            (studentized_residuals ** 2 / num_parameters)
            * (leverage_values / leverage_complement)
        )
    
    return cooks_distance


def compute_leverage_and_cooks(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute leverage (hat values) and Cook's distance for simple regression.
    
    Parameters
    ----------
    x : np.ndarray
        Predictor variable
    y : np.ndarray
        Response variable
        
    Returns
    -------
    leverage : np.ndarray
        Hat values (diagonal of hat matrix)
    cooks_d : np.ndarray
        Cook's distance for each observation
    residuals : np.ndarray
        Studentized residuals
    cooks_threshold : float
        Threshold for influential points (4/n)
    """
    predictor = np.asarray(x, dtype=float)
    response = np.asarray(y, dtype=float)
    valid_mask = np.isfinite(predictor) & np.isfinite(response)
    num_valid_samples = int(valid_mask.sum())
    
    if num_valid_samples < _MIN_SAMPLE_SIZE_FOR_LEVERAGE:
        nan_array = np.full(len(x), np.nan)
        return nan_array, nan_array, nan_array, np.nan
    
    valid_predictor = predictor[valid_mask]
    valid_response = response[valid_mask]
    
    intercept_column = np.ones(num_valid_samples)
    design_matrix = np.column_stack([intercept_column, valid_predictor])
    
    try:
        covariance_matrix_inverse = np.linalg.inv(design_matrix.T @ design_matrix)
        leverage_values = _compute_hat_matrix_diagonal(
            design_matrix, covariance_matrix_inverse
        )
    except np.linalg.LinAlgError:
        nan_array = np.full(len(x), np.nan)
        return nan_array, nan_array, nan_array, np.nan
    
    coefficients = covariance_matrix_inverse @ design_matrix.T @ valid_response
    fitted_values = design_matrix @ coefficients
    residuals = valid_response - fitted_values
    
    num_parameters = 2  # intercept + slope
    degrees_of_freedom = num_valid_samples - num_parameters
    mean_squared_error = (
        np.sum(residuals ** 2) / degrees_of_freedom if degrees_of_freedom > 0 else 0.0
    )
    
    studentized_residuals = _compute_studentized_residuals(
        residuals, mean_squared_error, leverage_values
    )
    
    cooks_distances = _compute_cooks_distance(
        studentized_residuals, leverage_values, num_parameters
    )
    
    cooks_threshold = _COOKS_THRESHOLD_FACTOR / num_valid_samples
    
    leverage = np.full(len(x), np.nan)
    cooks_d = np.full(len(x), np.nan)
    studentized_residuals_full = np.full(len(x), np.nan)
    
    leverage[valid_mask] = leverage_values
    cooks_d[valid_mask] = cooks_distances
    studentized_residuals_full[valid_mask] = studentized_residuals
    
    return leverage, cooks_d, studentized_residuals_full, cooks_threshold


def compute_normality_summary(data: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """Compute multiple normality tests and provide a summary.
    
    Parameters
    ----------
    data : np.ndarray
        Data to test for normality
    alpha : float, default=0.05
        Significance level for tests
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - shapiro_stat: Shapiro-Wilk test statistic
        - shapiro_p: Shapiro-Wilk p-value
        - dagostino_stat: D'Agostino K² test statistic
        - dagostino_p: D'Agostino K² p-value
        - skewness: Distribution skewness
        - kurtosis: Distribution kurtosis
        - is_normal: Boolean (p > alpha for both tests)
        - n: Sample size
    """
    clean_data = np.asarray(data)
    clean_data = clean_data[np.isfinite(clean_data)]
    sample_size = len(clean_data)
    
    shapiro_result = check_normality_shapiro(clean_data, alpha=alpha)
    dagostino_result = check_normality_dagostino(clean_data, alpha=alpha)
    
    has_sufficient_samples_for_moments = sample_size > _MIN_SAMPLE_SIZE_FOR_MOMENTS
    skewness = (
        float(stats.skew(clean_data)) if has_sufficient_samples_for_moments else np.nan
    )
    kurtosis = (
        float(stats.kurtosis(clean_data))
        if has_sufficient_samples_for_moments
        else np.nan
    )
    
    dagostino_applicable = sample_size >= _MIN_SAMPLE_SIZE_FOR_DAGOSTINO
    is_normal = shapiro_result.passed and (
        dagostino_result.passed or not dagostino_applicable
    )
    
    return {
        "shapiro_stat": float(shapiro_result.statistic),
        "shapiro_p": float(shapiro_result.p_value),
        "dagostino_stat": float(dagostino_result.statistic),
        "dagostino_p": float(dagostino_result.p_value),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "is_normal": is_normal,
        "n": sample_size,
    }


__all__ = [
    "compute_vif",
    "compute_leverage_and_cooks",
    "compute_normality_summary",
]
