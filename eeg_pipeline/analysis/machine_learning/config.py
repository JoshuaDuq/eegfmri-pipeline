"""
Shared ML configuration loader to keep defaults consistent.
"""

from __future__ import annotations

from typing import Any, Dict

from eeg_pipeline.utils.config.loader import load_config


def get_ml_config(config: Any = None) -> Dict[str, Any]:
    """
    Extract ML configuration with unified defaults for regression and classification.
    """
    resolved_config = config if config is not None else load_config()

    cv_config = resolved_config.get("machine_learning.cv", {})
    constant_config = resolved_config.get("machine_learning.constants", {})
    preprocessing_config = resolved_config.get("machine_learning.preprocessing", {})
    model_config = resolved_config.get("machine_learning.models", {})

    elasticnet_config = model_config.get("elasticnet", {})
    ridge_config = model_config.get("ridge", {})
    random_forest_config = model_config.get("random_forest", {})
    svm_config = model_config.get("svm", {})
    lr_config = model_config.get("logistic_regression", {})

    return {
        # CV settings
        "default_n_splits": cv_config.get("default_n_splits", 5),
        "default_n_bins": cv_config.get("default_n_bins", 5),
        # Constants
        "variance_threshold": constant_config.get("variance_threshold", 0.0),
        "min_variance_for_correlation": constant_config.get("min_variance_for_correlation", 1e-10),
        # Preprocessing
        "imputer_strategy": preprocessing_config.get("imputer_strategy", "median"),
        "power_transformer_method": preprocessing_config.get("power_transformer_method", "yeo-johnson"),
        "power_transformer_standardize": preprocessing_config.get("power_transformer_standardize", True),
        "variance_threshold_grid": preprocessing_config.get("variance_threshold_grid", [0.0, 0.01, 0.1]),
        # ElasticNet regression
        "elasticnet_max_iter": elasticnet_config.get("max_iter", 10000),
        "elasticnet_tol": elasticnet_config.get("tol", 1e-4),
        "elasticnet_selection": elasticnet_config.get("selection", "cyclic"),
        "elasticnet_alpha_grid": elasticnet_config.get("alpha_grid", [0.01, 0.1, 1.0, 10.0]),
        "elasticnet_l1_ratio_grid": elasticnet_config.get("l1_ratio_grid", [0.1, 0.5, 0.9]),
        # Ridge regression
        "ridge_alpha_grid": ridge_config.get("alpha_grid", [0.01, 0.1, 1.0, 10.0, 100.0]),
        # Random Forest (shared)
        "rf_n_estimators": random_forest_config.get("n_estimators", 100),
        "rf_bootstrap": random_forest_config.get("bootstrap", True),
        "rf_max_depth_grid": random_forest_config.get("max_depth_grid", [5, 10, 20, None]),
        "rf_min_samples_split_grid": random_forest_config.get("min_samples_split_grid", [2, 5, 10]),
        "rf_min_samples_leaf_grid": random_forest_config.get("min_samples_leaf_grid", [1, 2, 4]),
        "rf_class_weight": random_forest_config.get("class_weight", "balanced"),
        # SVM classification
        "svm_kernel": svm_config.get("kernel", "rbf"),
        "svm_C_grid": svm_config.get("C_grid", [0.1, 1.0, 10.0]),
        "svm_gamma_grid": svm_config.get("gamma_grid", ["scale", "auto"]),
        "svm_class_weight": svm_config.get("class_weight", "balanced"),
        # Logistic Regression classification
        "lr_penalty": lr_config.get("penalty", "l2"),
        "lr_C_grid": lr_config.get("C_grid", [0.01, 0.1, 1.0, 10.0]),
        "lr_max_iter": lr_config.get("max_iter", 1000),
        "lr_class_weight": lr_config.get("class_weight", "balanced"),
    }


__all__ = ["get_ml_config"]
