"""
SHAP Feature Importance
=======================

Compute SHAP (SHapley Additive exPlanations) values for model interpretation.

SHAP provides:
- Feature importance that accounts for feature interactions
- Per-sample feature contributions
- Consistent, theoretically-grounded importance measures

Usage:
    from eeg_pipeline.analysis.decoding.shap_importance import (
        compute_shap_importance,
        compute_shap_values,
        plot_shap_summary,
    )
    
    # Get importance
    importance_df = compute_shap_importance(model, X, feature_names)
    
    # Get detailed SHAP values
    shap_values, explainer = compute_shap_values(model, X)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


###################################################################
# SHAP Availability Check
###################################################################


def _check_shap_available() -> bool:
    """Check if SHAP is installed."""
    try:
        import shap
        return True
    except ImportError:
        return False


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
            vals = self.shap_values
            if isinstance(vals, list):
                # Multi-output: average across outputs
                vals = np.mean(np.abs(np.stack(vals)), axis=0)
            
            self.mean_abs_shap = np.mean(np.abs(vals), axis=0)
            self.std_shap = np.std(vals, axis=0)
            
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
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Get the actual estimator from pipeline if needed
    estimator = model
    if hasattr(model, "named_steps"):
        # It's a pipeline - get the last step
        step_names = list(model.named_steps.keys())
        estimator = model.named_steps[step_names[-1]]
        
        # Transform X through preprocessing steps
        X_transformed = X
        for name in step_names[:-1]:
            step = model.named_steps[name]
            if hasattr(step, "transform"):
                X_transformed = step.transform(X_transformed)
        X = X_transformed
    
    # Select explainer based on model type
    rng = np.random.default_rng(seed)
    
    try:
        if hasattr(estimator, "feature_importances_"):
            # Tree-based model (RF, XGBoost, etc.)
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X, check_additivity=check_additivity)
        elif hasattr(estimator, "coef_"):
            # Linear model
            explainer = shap.LinearExplainer(estimator, X)
            shap_values = explainer.shap_values(X)
        else:
            # Fall back to KernelExplainer (model-agnostic but slower)
            n_bg = min(background_samples, len(X))
            bg_idx = rng.choice(len(X), n_bg, replace=False)
            background = X[bg_idx]
            
            # Create prediction function
            if hasattr(model, "predict_proba"):
                predict_fn = lambda x: model.predict_proba(x)[:, 1]
            else:
                predict_fn = model.predict
            
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X, nsamples=100)
    except Exception as e:
        raise RuntimeError(f"SHAP computation failed: {e}")
    
    # Handle multi-class output
    if isinstance(shap_values, list):
        # For binary classification, use class 1
        if len(shap_values) == 2:
            shap_values = shap_values[1]
    
    expected_value = explainer.expected_value
    if isinstance(expected_value, np.ndarray) and len(expected_value) == 2:
        expected_value = expected_value[1]
    
    return SHAPResult(
        shap_values=shap_values,
        expected_value=expected_value,
        feature_names=feature_names[:shap_values.shape[1]],  # Adjust for preprocessing
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
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Collect SHAP values from each fold
    all_shap_importance = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        
        # Train model
        model = model_factory()
        model.fit(X_train, y_train)
        
        # Compute SHAP on test set
        try:
            result = compute_shap_values(
                model, X_test, feature_names, seed=seed + fold_idx
            )
            all_shap_importance.append(result.mean_abs_shap)
        except Exception:
            continue
    
    if not all_shap_importance:
        return pd.DataFrame()
    
    # Aggregate across folds
    stacked = np.stack(all_shap_importance)
    mean_importance = np.mean(stacked, axis=0)
    std_importance = np.std(stacked, axis=0)
    
    # Adjust feature names if preprocessing changed count
    n_features = len(mean_importance)
    if n_features != len(feature_names):
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    return pd.DataFrame({
        "feature": feature_names,
        "shap_importance": mean_importance,
        "shap_std": std_importance,
        "n_folds": len(all_shap_importance),
    }).sort_values("shap_importance", ascending=False).reset_index(drop=True)


###################################################################
# SHAP Interaction Analysis
###################################################################


def compute_shap_interactions(
    model: Any,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_features: int = 20,
) -> pd.DataFrame:
    """
    Compute SHAP interaction values for top features.
    
    Only works with tree-based models.
    
    Parameters
    ----------
    model : Any
        Fitted tree-based model
    X : np.ndarray
        Feature matrix
    feature_names : List[str], optional
        Feature names
    max_features : int
        Maximum features to include in interaction analysis
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: feature_1, feature_2, interaction_strength
    """
    if not _check_shap_available():
        raise ImportError("SHAP not installed")
    
    import shap
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Get estimator from pipeline
    estimator = model
    if hasattr(model, "named_steps"):
        step_names = list(model.named_steps.keys())
        estimator = model.named_steps[step_names[-1]]
    
    if not hasattr(estimator, "feature_importances_"):
        raise ValueError("Interaction analysis requires tree-based model")
    
    # Compute interactions
    explainer = shap.TreeExplainer(estimator)
    shap_interaction = explainer.shap_interaction_values(X[:min(500, len(X))])
    
    if isinstance(shap_interaction, list):
        shap_interaction = shap_interaction[1]  # Binary classification
    
    # Mean absolute interaction strength
    mean_interaction = np.mean(np.abs(shap_interaction), axis=0)
    
    # Extract top interactions
    n_features = min(max_features, len(feature_names))
    
    # Get indices of top features by main effect
    main_effects = np.diag(mean_interaction)
    top_idx = np.argsort(main_effects)[::-1][:n_features]
    
    # Build interaction matrix
    records = []
    for i in range(len(top_idx)):
        for j in range(i + 1, len(top_idx)):
            idx_i, idx_j = top_idx[i], top_idx[j]
            interaction = mean_interaction[idx_i, idx_j]
            records.append({
                "feature_1": feature_names[idx_i] if idx_i < len(feature_names) else f"f_{idx_i}",
                "feature_2": feature_names[idx_j] if idx_j < len(feature_names) else f"f_{idx_j}",
                "interaction_strength": float(interaction),
            })
    
    return pd.DataFrame(records).sort_values(
        "interaction_strength", ascending=False
    ).reset_index(drop=True)


###################################################################
# SHAP Plotting
###################################################################


def plot_shap_summary(
    result: SHAPResult,
    max_features: int = 20,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> Any:
    """
    Create SHAP summary plot.
    
    Parameters
    ----------
    result : SHAPResult
        SHAP computation result
    max_features : int
        Maximum features to display
    save_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    
    Returns
    -------
    matplotlib.Figure
        The generated figure
    """
    if not _check_shap_available():
        raise ImportError("SHAP not installed")
    
    import shap
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    shap.summary_plot(
        result.shap_values,
        result.X,
        feature_names=result.feature_names,
        max_display=max_features,
        show=False,
    )
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_shap_bar(
    result: SHAPResult,
    max_features: int = 20,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> Any:
    """
    Create SHAP bar plot (mean absolute SHAP values).
    
    Parameters
    ----------
    result : SHAPResult
        SHAP computation result
    max_features : int
        Maximum features to display
    save_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    
    Returns
    -------
    matplotlib.Figure
        The generated figure
    """
    import matplotlib.pyplot as plt
    
    df = result.get_top_features(max_features)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.barh(
        range(len(df)),
        df["shap_importance"],
        xerr=df["shap_std"],
        color="#1f77b4",
        alpha=0.8,
        capsize=3,
    )
    
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance")
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.close(fig)
    return fig


def plot_shap_dependence(
    result: SHAPResult,
    feature: str,
    interaction_feature: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> Any:
    """
    Create SHAP dependence plot for a specific feature.
    
    Parameters
    ----------
    result : SHAPResult
        SHAP computation result
    feature : str
        Feature to plot
    interaction_feature : str, optional
        Feature to use for coloring (interaction effect)
    save_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    
    Returns
    -------
    matplotlib.Figure
        The generated figure
    """
    if not _check_shap_available():
        raise ImportError("SHAP not installed")
    
    import shap
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    interaction_idx = None
    if interaction_feature and interaction_feature in result.feature_names:
        interaction_idx = result.feature_names.index(interaction_feature)
    
    shap.dependence_plot(
        feature,
        result.shap_values,
        result.X,
        feature_names=result.feature_names,
        interaction_index=interaction_idx,
        ax=ax,
        show=False,
    )
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


###################################################################
# Save/Load
###################################################################


def save_shap_results(
    result: SHAPResult,
    output_path: Path,
    prefix: str = "shap",
) -> Dict[str, Path]:
    """
    Save SHAP results to files.
    
    Parameters
    ----------
    result : SHAPResult
        SHAP computation result
    output_path : Path
        Output directory
    prefix : str
        Filename prefix
    
    Returns
    -------
    Dict[str, Path]
        Mapping of result type to saved path
    """
    from eeg_pipeline.infra.tsv import write_tsv
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    saved = {}
    
    # Importance
    if result.importance_df is not None:
        path = output_path / f"{prefix}_importance.tsv"
        write_tsv(result.importance_df, path)
        saved["importance"] = path
    
    # Raw SHAP values (as compressed numpy)
    shap_path = output_path / f"{prefix}_values.npz"
    np.savez_compressed(
        shap_path,
        shap_values=result.shap_values,
        expected_value=result.expected_value,
        feature_names=result.feature_names,
    )
    saved["values"] = shap_path
    
    return saved

















