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


from eeg_pipeline.analysis.decoding.config import get_decoding_config


###################################################################
# Pipeline Factories
###################################################################


def create_base_preprocessing_pipeline(
    include_scaling: bool = True,
    config: Any = None,
) -> Pipeline:
    """Create base preprocessing pipeline (impute, variance filter, scale)."""
    cfg = get_decoding_config(config)

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
    cfg = get_decoding_config(config)

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
    cfg = get_decoding_config(config)

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
    cfg = get_decoding_config(config)

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
    cfg = get_decoding_config(config)
    
    return {
        "rf__max_depth": cfg["rf_max_depth_grid"],
        "rf__min_samples_split": cfg["rf_min_samples_split_grid"],
        "rf__min_samples_leaf": cfg["rf_min_samples_leaf_grid"],
    }
