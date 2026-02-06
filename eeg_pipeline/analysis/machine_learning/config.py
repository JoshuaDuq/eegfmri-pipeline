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

    constant_config = resolved_config.get("machine_learning.constants", {})
    preprocessing_config = resolved_config.get("machine_learning.preprocessing", {})
    pca_config = preprocessing_config.get("pca", {}) if isinstance(preprocessing_config, dict) else {}
    model_config = resolved_config.get("machine_learning.models", {})

    elasticnet_config = model_config.get("elasticnet", {})
    ridge_config = model_config.get("ridge", {})
    random_forest_config = model_config.get("random_forest", {})
    svm_config = model_config.get("svm", {})
    lr_config = model_config.get("logistic_regression", {})
    cnn_config = model_config.get("cnn", {})

    return {
        # Constants
        "variance_threshold": constant_config.get("variance_threshold", 0.0),
        # Preprocessing
        "imputer_strategy": preprocessing_config.get("imputer_strategy", "median"),
        "power_transformer_method": preprocessing_config.get("power_transformer_method", "yeo-johnson"),
        "power_transformer_standardize": preprocessing_config.get("power_transformer_standardize", True),
        "variance_threshold_grid": preprocessing_config.get("variance_threshold_grid", [0.0, 0.01, 0.1]),
        "pca_enabled": bool(pca_config.get("enabled", False)),
        "pca_n_components": pca_config.get("n_components", 0.95),
        "pca_whiten": bool(pca_config.get("whiten", False)),
        "pca_svd_solver": pca_config.get("svd_solver", "auto"),
        "pca_random_state": pca_config.get("random_state", None),
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
        # CNN classification (EEGNet-style)
        "cnn_temporal_filters": int(cnn_config.get("temporal_filters", 8)),
        "cnn_depth_multiplier": int(cnn_config.get("depth_multiplier", 2)),
        "cnn_pointwise_filters": int(cnn_config.get("pointwise_filters", 16)),
        "cnn_kernel_length": int(cnn_config.get("kernel_length", 64)),
        "cnn_separable_kernel_length": int(cnn_config.get("separable_kernel_length", 16)),
        "cnn_dropout": float(cnn_config.get("dropout", 0.5)),
        "cnn_batch_size": int(cnn_config.get("batch_size", 64)),
        "cnn_max_epochs": int(cnn_config.get("max_epochs", 75)),
        "cnn_patience": int(cnn_config.get("patience", 10)),
        "cnn_learning_rate": float(cnn_config.get("learning_rate", 1e-3)),
        "cnn_weight_decay": float(cnn_config.get("weight_decay", 1e-3)),
        "cnn_val_fraction": float(cnn_config.get("val_fraction", 0.2)),
        "cnn_gradient_clip_norm": float(cnn_config.get("gradient_clip_norm", 1.0)),
        "cnn_use_cuda": bool(cnn_config.get("use_cuda", False)),
    }


__all__ = ["get_ml_config"]
