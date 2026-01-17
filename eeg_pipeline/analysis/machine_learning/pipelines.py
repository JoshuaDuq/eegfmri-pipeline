"""
ML pipeline factories.

Provides pre-configured sklearn pipelines for regression.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import VarianceThreshold

from eeg_pipeline.analysis.machine_learning.config import get_ml_config


def _build_base_preprocessing_steps(cfg: Dict[str, Any], include_scaling: bool) -> list[tuple[str, Any]]:
    """Build the common preprocessing steps shared by all regression pipelines."""
    steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy=cfg["imputer_strategy"])),
        ("var", VarianceThreshold(threshold=cfg["variance_threshold"])),
    ]

    if include_scaling:
        steps.append(("scale", StandardScaler()))

    return steps


def create_elasticnet_pipeline(
    seed: int = 42,
    config: Any = None,
) -> Pipeline:
    """Create ElasticNet regression pipeline with target transformation."""
    cfg = get_ml_config(config)

    steps = _build_base_preprocessing_steps(cfg=cfg, include_scaling=True)

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

    steps.append(("regressor", regressor))
    return Pipeline(steps)


def create_ridge_pipeline(
    seed: int = 42,
    config: Any = None,
) -> Pipeline:
    """Create Ridge regression pipeline with target transformation."""
    cfg = get_ml_config(config)

    steps = _build_base_preprocessing_steps(cfg=cfg, include_scaling=True)

    regressor = TransformedTargetRegressor(
        regressor=Ridge(random_state=seed),
        transformer=PowerTransformer(
            method=cfg["power_transformer_method"],
            standardize=cfg["power_transformer_standardize"],
        ),
    )

    steps.append(("regressor", regressor))
    return Pipeline(steps)


def create_rf_pipeline(
    n_estimators: Optional[int] = None,
    n_jobs: int = 1,
    seed: int = 42,
    config: Any = None,
) -> Pipeline:
    """Create Random Forest regression pipeline."""
    cfg = get_ml_config(config)

    if n_estimators is None:
        n_estimators = cfg["rf_n_estimators"]

    steps = _build_base_preprocessing_steps(cfg=cfg, include_scaling=False)
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

    return Pipeline(steps)


def build_elasticnet_param_grid(config: Any = None) -> Dict[str, Any]:
    """Build hyperparameter grid for ElasticNet including variance threshold."""
    cfg = get_ml_config(config)

    return {
        "regressor__regressor__alpha": cfg["elasticnet_alpha_grid"],
        "regressor__regressor__l1_ratio": cfg["elasticnet_l1_ratio_grid"],
        "var__threshold": cfg["variance_threshold_grid"],
    }


def build_ridge_param_grid(config: Any = None) -> Dict[str, Any]:
    """Build hyperparameter grid for Ridge regression."""
    cfg = get_ml_config(config)
    return {"regressor__regressor__alpha": cfg["ridge_alpha_grid"]}


def build_rf_param_grid(config: Any = None) -> Dict[str, Any]:
    """Build hyperparameter grid for Random Forest."""
    cfg = get_ml_config(config)
    return {
        "rf__max_depth": cfg["rf_max_depth_grid"],
        "rf__min_samples_split": cfg["rf_min_samples_split_grid"],
        "rf__min_samples_leaf": cfg["rf_min_samples_leaf_grid"],
    }
