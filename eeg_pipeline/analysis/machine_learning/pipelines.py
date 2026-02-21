"""
ML pipeline factories.

Provides pre-configured sklearn pipelines for regression.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor

from eeg_pipeline.analysis.machine_learning.config import get_ml_config
from eeg_pipeline.analysis.machine_learning.preprocessing import (
    build_base_preprocessing_steps,
)



def create_elasticnet_pipeline(
    seed: int = 42,
    config: Any = None,
    n_covariates: int = 0,
) -> Pipeline:
    """Create ElasticNet regression pipeline with target transformation."""
    cfg = get_ml_config(config)

    steps = build_base_preprocessing_steps(
        cfg=cfg,
        include_scaling=True,
        n_covariates=n_covariates,
        config=config,
        score_func=f_regression,
    )

    steps.append((
        "regressor",
        ElasticNet(
            random_state=seed,
            max_iter=cfg["elasticnet_max_iter"],
            tol=cfg["elasticnet_tol"],
            selection=cfg["elasticnet_selection"],
        ),
    ))
    pipeline = Pipeline(steps)

    return TransformedTargetRegressor(
        regressor=pipeline,
        transformer=PowerTransformer(
            method=cfg["power_transformer_method"],
            standardize=cfg["power_transformer_standardize"],
        ),
    )


def create_ridge_pipeline(
    seed: int = 42,
    config: Any = None,
    n_covariates: int = 0,
) -> Pipeline:
    """Create Ridge regression pipeline with target transformation."""
    cfg = get_ml_config(config)

    steps = build_base_preprocessing_steps(
        cfg=cfg,
        include_scaling=True,
        n_covariates=n_covariates,
        config=config,
        score_func=f_regression,
    )

    steps.append(("regressor", Ridge(random_state=seed)))
    pipeline = Pipeline(steps)

    return TransformedTargetRegressor(
        regressor=pipeline,
        transformer=PowerTransformer(
            method=cfg["power_transformer_method"],
            standardize=cfg["power_transformer_standardize"],
        ),
    )


def create_rf_pipeline(
    n_estimators: Optional[int] = None,
    n_jobs: int = 1,
    seed: int = 42,
    config: Any = None,
    n_covariates: int = 0,
) -> Pipeline:
    """Create Random Forest regression pipeline."""
    cfg = get_ml_config(config)

    if n_estimators is None:
        n_estimators = cfg["rf_n_estimators"]

    steps = build_base_preprocessing_steps(
        cfg=cfg,
        include_scaling=False,
        n_covariates=n_covariates,
        config=config,
        score_func=f_regression,
    )
    steps.append(
        (
            "rf",
            RandomForestRegressor(
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                random_state=seed,
                bootstrap=cfg["rf_bootstrap"],
            ),
        )
    )
    pipeline = Pipeline(steps)

    return TransformedTargetRegressor(
        regressor=pipeline,
        transformer=PowerTransformer(
            method=cfg["power_transformer_method"],
            standardize=cfg["power_transformer_standardize"],
        ),
    )


def build_elasticnet_param_grid(config: Any = None, n_covariates: int = 0) -> Dict[str, Any]:
    """Build hyperparameter grid for ElasticNet including variance threshold."""
    cfg = get_ml_config(config)

    var_prefix = "regressor__preprocessing__eeg__var" if n_covariates > 0 else "regressor__var"
    
    return {
        "regressor__regressor__alpha": cfg["elasticnet_alpha_grid"],
        "regressor__regressor__l1_ratio": cfg["elasticnet_l1_ratio_grid"],
        f"{var_prefix}__threshold": cfg["variance_threshold_grid"],
    }


def build_ridge_param_grid(config: Any = None, n_covariates: int = 0) -> Dict[str, Any]:
    """Build hyperparameter grid for Ridge regression."""
    cfg = get_ml_config(config)
    
    var_prefix = "regressor__preprocessing__eeg__var" if n_covariates > 0 else "regressor__var"
    
    grid = {"regressor__regressor__alpha": cfg["ridge_alpha_grid"]}
    if "variance_threshold_grid" in cfg:
        grid[f"{var_prefix}__threshold"] = cfg["variance_threshold_grid"]
        
    return grid


def build_rf_param_grid(config: Any = None, n_covariates: int = 0) -> Dict[str, Any]:
    """Build hyperparameter grid for Random Forest."""
    cfg = get_ml_config(config)
    
    var_prefix = "preprocessing__eeg__var" if n_covariates > 0 else "var"
    
    grid = {
        # Keep legacy key names for external callers/tests; orchestration resolves
        # them to nested TransformedTargetRegressor params before GridSearchCV.
        "rf__max_depth": cfg["rf_max_depth_grid"],
        "rf__min_samples_split": cfg["rf_min_samples_split_grid"],
        "rf__min_samples_leaf": cfg["rf_min_samples_leaf_grid"],
    }
    
    if "variance_threshold_grid" in cfg:
        grid[f"{var_prefix}__threshold"] = cfg["variance_threshold_grid"]
        
    return grid
