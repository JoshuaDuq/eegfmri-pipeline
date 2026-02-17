"""
SHAP Feature Importance
=======================

Compute SHAP (SHapley Additive exPlanations) values for model interpretation.

SHAP provides:
- Feature importance that accounts for feature interactions
- Per-sample feature contributions
- Consistent, theoretically-grounded importance measures

Usage:
    from eeg_pipeline.analysis.machine_learning.shap_importance import (
        compute_shap_importance,
        compute_shap_values,
    )
    
    # Get importance
    importance_df = compute_shap_importance(model, X, feature_names)
    
    # Get detailed SHAP values
    shap_values, explainer = compute_shap_values(model, X)
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, GridSearchCV

from eeg_pipeline.analysis.machine_learning.preprocessing import transform_feature_names_through_steps
from eeg_pipeline.analysis.machine_learning.cv import apply_fold_feature_harmonization

###################################################################
# Helper Functions
###################################################################


def _check_shap_available() -> bool:
    """Check if SHAP is installed."""
    return importlib.util.find_spec("shap") is not None


def _generate_feature_names(n_features: int) -> List[str]:
    """Generate default feature names."""
    return [f"feature_{i}" for i in range(n_features)]


def _extract_estimator_transform_and_feature_names(
    model: Any,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Tuple[Any, np.ndarray, Optional[List[str]]]:
    """Extract estimator from pipeline and transform X (and feature names) through preprocessing."""
    estimator = model
    names = feature_names

    if hasattr(model, "named_steps"):
        step_names = list(model.named_steps.keys())
        estimator = model.named_steps[step_names[-1]]

        steps = [(n, model.named_steps[n]) for n in step_names[:-1]]
        if names is not None:
            # Apply selector-aware name mapping first (so it follows the fitted support masks).
            names = transform_feature_names_through_steps(steps, names)
            # PCA does not expose stable feature names; use component labels.
            for _, step in steps:
                if isinstance(step, PCA):
                    try:
                        n_out = int(getattr(step, "n_components_", 0) or 0)
                    except Exception:
                        n_out = 0
                    if n_out > 0:
                        names = [f"PC{i+1}" for i in range(n_out)]

        X_transformed = X
        for _, step in steps:
            if hasattr(step, "transform"):
                X_transformed = step.transform(X_transformed)
        X = X_transformed

    return estimator, X, names


def _create_predict_fn(model: Any):
    """Create prediction function for KernelExplainer."""
    if hasattr(model, "predict_proba"):
        def predict_fn(x):
            return model.predict_proba(x)[:, 1]
        return predict_fn
    return model.predict


def _handle_binary_classification_output(shap_values: Union[np.ndarray, List], expected_value: Optional[Union[float, np.ndarray]] = None) -> Tuple[np.ndarray, Optional[Union[float, np.ndarray]]]:
    """Handle binary classification output format."""
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
    
    if expected_value is not None and isinstance(expected_value, np.ndarray) and len(expected_value) == 2:
        expected_value = expected_value[1]
    
    return shap_values, expected_value


###################################################################
# SHAP Value Computation
###################################################################


@dataclass
class SHAPResult:
    """Container for SHAP analysis results."""
    
    shap_values: np.ndarray  # (n_samples, n_features) or list for multi-output
    expected_value: Union[float, np.ndarray]  # Base value(s)
    feature_names: List[str]
    X: np.ndarray  # Feature matrix
    
    # Computed importance
    importance_df: Optional[pd.DataFrame] = None
    
    # Per-feature statistics
    mean_abs_shap: Optional[np.ndarray] = None
    std_shap: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Compute importance statistics."""
        if self.shap_values is not None:
            shap_values_array = self.shap_values
            if isinstance(shap_values_array, list):
                # Multi-output: average across outputs
                shap_values_array = np.mean(
                    np.abs(np.stack(shap_values_array)),
                    axis=0,
                )
            
            self.mean_abs_shap = np.mean(np.abs(shap_values_array), axis=0)
            self.std_shap = np.std(shap_values_array, axis=0)
            
            self.importance_df = pd.DataFrame({
                "feature": self.feature_names,
                "shap_importance": self.mean_abs_shap,
                "shap_std": self.std_shap,
            }).sort_values("shap_importance", ascending=False).reset_index(drop=True)
    
    def get_top_features(self, n: int = 20) -> pd.DataFrame:
        """Get top N most important features."""
        if self.importance_df is None:
            return pd.DataFrame()
        return self.importance_df.head(n)
    
    def get_feature_shap(self, feature: str) -> np.ndarray:
        """Get SHAP values for a specific feature."""
        if feature not in self.feature_names:
            raise ValueError(f"Feature '{feature}' not found")
        idx = self.feature_names.index(feature)
        vals = self.shap_values
        if isinstance(vals, list):
            vals = vals[0]  # Use first output
        return vals[:, idx]


def compute_shap_values(
    model: Any,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    *,
    background_samples: int = 100,
    check_additivity: bool = False,
    seed: int = 42,
) -> SHAPResult:
    """
    Compute SHAP values for model predictions.
    
    Automatically selects appropriate SHAP explainer based on model type.
    
    Parameters
    ----------
    model : Any
        Fitted sklearn model or pipeline
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    feature_names : List[str], optional
        Feature names. If None, generates "feature_0", "feature_1", etc.
    background_samples : int
        Number of background samples for KernelExplainer
    check_additivity : bool
        Whether to check SHAP additivity (slower)
    seed : int
        Random seed for background sampling
    
    Returns
    -------
    SHAPResult
        Container with SHAP values and importance
    """
    if not _check_shap_available():
        raise ImportError("SHAP not installed. Install with: pip install shap")
    
    import shap
    
    X = np.asarray(X)
    
    if feature_names is None:
        feature_names = _generate_feature_names(X.shape[1])

    estimator, X, feature_names = _extract_estimator_transform_and_feature_names(
        model, X, feature_names
    )
    
    rng = np.random.default_rng(seed)
    
    try:
        if hasattr(estimator, "feature_importances_"):
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X, check_additivity=check_additivity)
        elif hasattr(estimator, "coef_"):
            explainer = shap.LinearExplainer(estimator, X)
            shap_values = explainer.shap_values(X)
        else:
            n_bg = min(background_samples, len(X))
            bg_idx = rng.choice(len(X), n_bg, replace=False)
            background = X[bg_idx]
            
            # KernelExplainer must consume the same feature space used for background/X.
            # At this point X has already been transformed through pipeline preprocessing,
            # so predict_fn must target the extracted estimator (not the full pipeline).
            predict_fn = _create_predict_fn(estimator)
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X, nsamples=100)
    except Exception as e:
        raise RuntimeError(f"SHAP computation failed: {e}")
    
    expected_value = explainer.expected_value
    shap_values, expected_value = _handle_binary_classification_output(shap_values, expected_value)
    
    if feature_names is None:
        feature_names = _generate_feature_names(X.shape[1])
    if shap_values.shape[1] != len(feature_names):
        feature_names = _generate_feature_names(shap_values.shape[1])

    return SHAPResult(
        shap_values=shap_values,
        expected_value=expected_value,
        feature_names=feature_names,
        X=X,
    )


def compute_shap_importance(
    model: Any,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Compute SHAP-based feature importance.
    
    Convenience function that returns just the importance DataFrame.
    
    Parameters
    ----------
    model : Any
        Fitted model
    X : np.ndarray
        Feature matrix
    feature_names : List[str], optional
        Feature names
    **kwargs
        Additional arguments passed to compute_shap_values
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: feature, shap_importance, shap_std
    """
    result = compute_shap_values(model, X, feature_names, **kwargs)
    return result.importance_df


###################################################################
# Aggregated SHAP Analysis
###################################################################


def compute_shap_for_cv_folds(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    feature_names: Optional[List[str]] = None,
    seed: int = 42,
    groups: Optional[np.ndarray] = None,
    harmonization_mode: Optional[str] = None,
    param_grid: Optional[Dict[str, Any]] = None,
    inner_cv_splits: int = 3,
) -> pd.DataFrame:
    """
    Compute SHAP importance aggregated across CV folds.
    
    This provides robust importance estimates by averaging
    across different train/test splits.
    
    Parameters
    ----------
    model_factory : callable
        Function that creates a new model instance
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target values
    cv_splits : List[Tuple]
        List of (train_idx, test_idx) tuples
    feature_names : List[str], optional
        Feature names
    seed : int
        Random seed
    
    Returns
    -------
    pd.DataFrame
        Aggregated importance with columns: feature, shap_importance, shap_std, n_folds
    """
    if not _check_shap_available():
        raise ImportError("SHAP not installed")
    
    if feature_names is None:
        feature_names = _generate_feature_names(X.shape[1])
    
    # Collect SHAP importances from each fold (feature-name keyed for robustness)
    fold_importances: List[pd.DataFrame] = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        groups_train = groups[train_idx] if groups is not None else None

        # Train model
        model = model_factory()
        keep_mask = np.ones(X_train.shape[1], dtype=bool)
        if groups_train is not None:
            X_train, X_test, keep_mask = apply_fold_feature_harmonization(
                X_train,
                X_test,
                groups_train,
                harmonization_mode,
            )

        if param_grid and groups_train is not None and len(np.unique(groups_train)) >= 2:
            n_splits = min(int(inner_cv_splits), len(np.unique(groups_train)))
            if n_splits >= 2:
                try:
                    inner_cv = GroupKFold(n_splits=n_splits)
                    gs = GridSearchCV(
                        estimator=model,
                        param_grid=param_grid,
                        scoring="r2",
                        cv=inner_cv,
                        n_jobs=1,
                        refit=True,
                        error_score="raise",
                    )
                    gs.fit(X_train, y_train, groups=groups_train)
                    model = gs.best_estimator_
                except Exception:
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        try:
            result = compute_shap_values(
                model,
                X_test,
                [f for f, keep in zip(feature_names or _generate_feature_names(X.shape[1]), keep_mask) if keep],
                seed=seed + fold_idx,
            )
            if result.importance_df is not None and not result.importance_df.empty:
                df = result.importance_df[["feature", "shap_importance"]].copy()
                df["fold"] = int(fold_idx)
                fold_importances.append(df)
        except (RuntimeError, ImportError):
            continue
    
    if not fold_importances:
        return pd.DataFrame()

    merged = pd.concat(fold_importances, axis=0, ignore_index=True)
    n_folds_used = len({int(v) for v in merged["fold"].unique().tolist()})
    n_folds_attempted = int(len(cv_splits))

    agg = (
        merged.groupby("feature", as_index=False)["shap_importance"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "shap_importance",
                "std": "shap_std_across_folds",
                "count": "n_folds_present",
            }
        )
    )
    agg["n_folds_used"] = int(n_folds_used)
    agg["n_folds_attempted"] = int(n_folds_attempted)
    return agg.sort_values("shap_importance", ascending=False).reset_index(drop=True)
