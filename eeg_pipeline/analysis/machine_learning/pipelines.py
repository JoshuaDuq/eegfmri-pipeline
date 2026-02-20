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
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion

from eeg_pipeline.analysis.machine_learning.config import get_ml_config
from eeg_pipeline.analysis.machine_learning.preprocessing import (
    DropAllNaNColumns,
    ReplaceInfWithNaN,
    VarianceThreshold,
)


def _build_base_preprocessing_steps(cfg: Dict[str, Any], include_scaling: bool, n_covariates: int = 0) -> list[tuple[str, Any]]:
    """Build the common preprocessing steps shared by all regression pipelines.
    
    Covariates (the last `n_covariates` columns) bypass VarianceThreshold and PCA,
    but still receive imputation and (optionally) scaling.
    """
    steps: list[tuple[str, Any]] = [
        ("finite", ReplaceInfWithNaN()),
        ("drop_all_nan", DropAllNaNColumns()),
    ]
    
    # Feature specific steps
    feature_steps = [
        ("impute", SimpleImputer(strategy=cfg["imputer_strategy"])),
        ("var", VarianceThreshold(threshold=cfg["variance_threshold"]))
    ]
    if include_scaling:
        feature_steps.append(("scale", StandardScaler()))
    if cfg.get("pca_enabled", False):
        feature_steps.append(
            (
                "pca",
                PCA(
                    n_components=cfg.get("pca_n_components", 0.95),
                    whiten=bool(cfg.get("pca_whiten", False)),
                    random_state=cfg.get("pca_random_state", None),
                    svd_solver=str(cfg.get("pca_svd_solver", "auto")),
                ),
            )
        )
    feature_pipe = Pipeline(feature_steps)

    if n_covariates > 0:
        cov_steps = [
            ("impute", SimpleImputer(strategy="most_frequent"))
        ]
        if include_scaling:
            cov_steps.append(("scale", StandardScaler()))
            covariate_pipe = Pipeline(cov_steps)
        else:
            covariate_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
        
        def feature_idx(X):
            return list(range(X.shape[1] - n_covariates))
            
        def cov_idx(X):
            return list(range(X.shape[1] - n_covariates, X.shape[1]))
            
        preprocessor = ColumnTransformer(
            transformers=[
                ("eeg", feature_pipe, feature_idx),
                ("cov", covariate_pipe, cov_idx),
            ],
            remainder="drop"
        )
        steps.append(("preprocessing", preprocessor))
    else:
        steps.extend(feature_steps)

    return steps



def create_elasticnet_pipeline(
    seed: int = 42,
    config: Any = None,
    n_covariates: int = 0,
) -> Pipeline:
    """Create ElasticNet regression pipeline with target transformation."""
    cfg = get_ml_config(config)

    steps = _build_base_preprocessing_steps(cfg=cfg, include_scaling=True, n_covariates=n_covariates)

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
    n_covariates: int = 0,
) -> Pipeline:
    """Create Ridge regression pipeline with target transformation."""
    cfg = get_ml_config(config)

    steps = _build_base_preprocessing_steps(cfg=cfg, include_scaling=True, n_covariates=n_covariates)

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
    n_covariates: int = 0,
) -> Pipeline:
    """Create Random Forest regression pipeline."""
    cfg = get_ml_config(config)

    if n_estimators is None:
        n_estimators = cfg["rf_n_estimators"]

    steps = _build_base_preprocessing_steps(cfg=cfg, include_scaling=False, n_covariates=n_covariates)
    steps.append(
        (
            "rf",
            TransformedTargetRegressor(
                regressor=RandomForestRegressor(
                    n_estimators=n_estimators,
                    n_jobs=n_jobs,
                    random_state=seed,
                    bootstrap=cfg["rf_bootstrap"],
                ),
                transformer=PowerTransformer(
                    method=cfg["power_transformer_method"],
                    standardize=cfg["power_transformer_standardize"],
                ),
            ),
        )
    )

    return Pipeline(steps)


def build_elasticnet_param_grid(config: Any = None, n_covariates: int = 0) -> Dict[str, Any]:
    """Build hyperparameter grid for ElasticNet including variance threshold."""
    cfg = get_ml_config(config)

    var_prefix = "preprocessing__eeg__var" if n_covariates > 0 else "var"
    
    return {
        "regressor__regressor__alpha": cfg["elasticnet_alpha_grid"],
        "regressor__regressor__l1_ratio": cfg["elasticnet_l1_ratio_grid"],
        f"{var_prefix}__threshold": cfg["variance_threshold_grid"],
    }


def build_ridge_param_grid(config: Any = None, n_covariates: int = 0) -> Dict[str, Any]:
    """Build hyperparameter grid for Ridge regression."""
    cfg = get_ml_config(config)
    
    var_prefix = "preprocessing__eeg__var" if n_covariates > 0 else "var"
    
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
