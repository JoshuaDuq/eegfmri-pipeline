"""
Shared decoding configuration loader to keep defaults consistent.
"""

from __future__ import annotations

from typing import Any, Dict

from eeg_pipeline.utils.config.loader import load_config


def _ensure_config(config: Any) -> Any:
    return config if config is not None else load_config()


def get_decoding_config(config: Any = None) -> Dict[str, Any]:
    """
    Extract decoding configuration with unified defaults for regression and classification.
    """
    cfg = _ensure_config(config)

    cv = cfg.get("decoding.cv", {})
    constants = cfg.get("decoding.constants", {})
    preprocessing = cfg.get("decoding.preprocessing", {})
    models = cfg.get("decoding.models", {})

    elasticnet = models.get("elasticnet", {})
    rf = models.get("random_forest", {})
    svm_cfg = models.get("svm", {})
    lr_cfg = models.get("logistic_regression", {})

    return {
        # CV settings
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
        # ElasticNet regression
        "elasticnet_max_iter": elasticnet.get("max_iter", 10000),
        "elasticnet_tol": elasticnet.get("tol", 1e-4),
        "elasticnet_selection": elasticnet.get("selection", "cyclic"),
        "elasticnet_alpha_grid": elasticnet.get("alpha_grid", [0.01, 0.1, 1.0, 10.0]),
        "elasticnet_l1_ratio_grid": elasticnet.get("l1_ratio_grid", [0.1, 0.5, 0.9]),
        # Random Forest (shared)
        "rf_n_estimators": rf.get("n_estimators", 100),
        "rf_bootstrap": rf.get("bootstrap", True),
        "rf_max_depth_grid": rf.get("max_depth_grid", [5, 10, 20, None]),
        "rf_min_samples_split_grid": rf.get("min_samples_split_grid", [2, 5, 10]),
        "rf_min_samples_leaf_grid": rf.get("min_samples_leaf_grid", [1, 2, 4]),
        "rf_class_weight": rf.get("class_weight", "balanced"),
        # SVM classification
        "svm_kernel": svm_cfg.get("kernel", "rbf"),
        "svm_C_grid": svm_cfg.get("C_grid", [0.1, 1.0, 10.0]),
        "svm_gamma_grid": svm_cfg.get("gamma_grid", ["scale", "auto"]),
        "svm_class_weight": svm_cfg.get("class_weight", "balanced"),
        # Logistic Regression classification
        "lr_penalty": lr_cfg.get("penalty", "l2"),
        "lr_C_grid": lr_cfg.get("C_grid", [0.01, 0.1, 1.0, 10.0]),
        "lr_max_iter": lr_cfg.get("max_iter", 1000),
        "lr_class_weight": lr_cfg.get("class_weight", "balanced"),
    }


__all__ = ["get_decoding_config"]





