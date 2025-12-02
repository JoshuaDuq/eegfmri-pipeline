"""
ML pipeline factories for decoding.

Provides pre-configured sklearn pipelines for regression decoding.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import VarianceThreshold


###################################################################
# Configuration Loading
###################################################################


def _get_decoding_config(config: Any) -> Dict[str, Any]:
    """Extract decoding configuration with defaults."""
    if config is None:
        from eeg_pipeline.utils.config.loader import load_settings
        config = load_settings()

    cv = config.get("decoding.cv", {})
    constants = config.get("decoding.constants", {})
    preprocessing = config.get("decoding.preprocessing", {})
    models = config.get("decoding.models", {})
    elasticnet = models.get("elasticnet", {})
    rf = models.get("random_forest", {})

    return {
        # CV settings
        "min_trials_for_within_subject": cv.get("min_trials_for_within_subject", 10),
        "default_n_splits": cv.get("default_n_splits", 5),
        "default_n_bins": cv.get("default_n_bins", 5),
        # Constants
        "variance_threshold": constants.get("variance_threshold", 0.0),
        "min_variance_for_correlation": constants.get("min_variance_for_correlation", 1e-10),
        # Preprocessing
        "imputer_strategy": preprocessing.get("imputer_strategy", "median"),
        "power_transformer_method": preprocessing.get("power_transformer_method", "yeo-johnson"),
        "power_transformer_standardize": preprocessing.get("power_transformer_standardize", True),
        "variance_threshold_grid": preprocessing.get("variance_threshold_grid", [0.0, 0.01, 0.1]),
        # ElasticNet
        "elasticnet_max_iter": elasticnet.get("max_iter", 10000),
        "elasticnet_tol": elasticnet.get("tol", 1e-4),
        "elasticnet_selection": elasticnet.get("selection", "cyclic"),
        "elasticnet_alpha_grid": elasticnet.get("alpha_grid", [0.01, 0.1, 1.0, 10.0]),
        "elasticnet_l1_ratio_grid": elasticnet.get("l1_ratio_grid", [0.1, 0.5, 0.9]),
        # Random Forest
        "rf_n_estimators": rf.get("n_estimators", 100),
        "rf_bootstrap": rf.get("bootstrap", True),
        "rf_max_depth_grid": rf.get("max_depth_grid", [5, 10, 20, None]),
        "rf_min_samples_split_grid": rf.get("min_samples_split_grid", [2, 5, 10]),
        "rf_min_samples_leaf_grid": rf.get("min_samples_leaf_grid", [1, 2, 4]),
    }


###################################################################
# Pipeline Factories
###################################################################


def create_base_preprocessing_pipeline(
    include_scaling: bool = True,
    config: Any = None,
) -> Pipeline:
    """Create base preprocessing pipeline (impute, variance filter, scale)."""
    cfg = _get_decoding_config(config)

    steps = [
        ("impute", SimpleImputer(strategy=cfg["imputer_strategy"])),
        ("var", VarianceThreshold(threshold=cfg["variance_threshold"])),
    ]

    if include_scaling:
        steps.append(("scale", StandardScaler()))

    return Pipeline(steps)


def create_elasticnet_pipeline(
    seed: int = 42,
    config: Any = None,
) -> Pipeline:
    """Create ElasticNet regression pipeline with target transformation."""
    cfg = _get_decoding_config(config)

    base_steps = create_base_preprocessing_pipeline(include_scaling=True, config=config).steps

    regressor = TransformedTargetRegressor(
        regressor=ElasticNet(
            random_state=seed,
            max_iter=cfg["elasticnet_max_iter"],
            tol=cfg["elasticnet_tol"],
            selection=cfg["elasticnet_selection"],
        ),
        transformer=PowerTransformer(
            method=cfg["power_transformer_method"],
            standardize=cfg["power_transformer_standardize"],
        ),
    )

    base_steps.append(("regressor", regressor))
    return Pipeline(base_steps)


def create_rf_pipeline(
    n_estimators: Optional[int] = None,
    n_jobs: int = 1,
    seed: int = 42,
    config: Any = None,
) -> Pipeline:
    """Create Random Forest regression pipeline."""
    cfg = _get_decoding_config(config)

    if n_estimators is None:
        n_estimators = cfg["rf_n_estimators"]

    steps = [
        ("impute", SimpleImputer(strategy=cfg["imputer_strategy"])),
        ("var", VarianceThreshold(threshold=cfg["variance_threshold"])),
        (
            "rf",
            RandomForestRegressor(
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                random_state=seed,
                bootstrap=cfg["rf_bootstrap"],
            ),
        ),
    ]

    return Pipeline(steps)


###################################################################
# Parameter Grids
###################################################################


def build_elasticnet_param_grid(config: Any = None) -> dict:
    """Build hyperparameter grid for ElasticNet including variance threshold."""
    cfg = _get_decoding_config(config)

    param_grid = {
        "regressor__regressor__alpha": cfg["elasticnet_alpha_grid"],
        "regressor__regressor__l1_ratio": cfg["elasticnet_l1_ratio_grid"],
    }
    
    # Add variance threshold grid if available
    if "variance_threshold_grid" in cfg:
        param_grid["var__threshold"] = cfg["variance_threshold_grid"]
    
    return param_grid


def build_rf_param_grid(config: Any = None) -> dict:
    """Build hyperparameter grid for Random Forest."""
    cfg = _get_decoding_config(config)
    
    return {
        "rf__max_depth": cfg["rf_max_depth_grid"],
        "rf__min_samples_split": cfg["rf_min_samples_split_grid"],
        "rf__min_samples_leaf": cfg["rf_min_samples_leaf_grid"],
    }
