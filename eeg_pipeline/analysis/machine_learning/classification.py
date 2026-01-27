"""
Pain Classification Pipelines
=============================

Classification models for pain vs no-pain prediction from EEG features.
Complements the regression pipelines for predicting pain intensity.

Models:
- SVM with RBF kernel (robust to outliers)
- Logistic Regression with L1/L2 (sparse, interpretable)
- Random Forest Classifier (non-linear, feature importance)

Usage:
    from eeg_pipeline.analysis.machine_learning.classification import (
        create_svm_pipeline,
        decode_pain_binary,
        nested_loso_classification,
    )
    
    # Quick classification
    results = decode_pain_binary(X, y_binary, cv="loso", groups=subject_ids)
    print(f"AUC: {results['auc']:.3f}, Balanced Acc: {results['balanced_acc']:.3f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from eeg_pipeline.analysis.machine_learning.config import get_ml_config
from eeg_pipeline.analysis.machine_learning.preprocessing import (
    DropAllNaNColumns,
    ReplaceInfWithNaN,
    VarianceThreshold,
)


###################################################################
# Pipeline Factories
###################################################################

def _build_base_preprocessing_steps(cfg: Dict[str, Any], include_scaling: bool) -> List[Tuple[str, Any]]:
    steps: List[Tuple[str, Any]] = [
        ("finite", ReplaceInfWithNaN()),
        ("drop_all_nan", DropAllNaNColumns()),
        ("impute", SimpleImputer(strategy=cfg["imputer_strategy"])),
        ("var", VarianceThreshold(threshold=cfg["variance_threshold"])),
    ]
    if include_scaling:
        steps.append(("scale", StandardScaler()))
        if cfg.get("pca_enabled", False):
            steps.append(
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
    return steps


def create_svm_pipeline(
    kernel: str = "rbf",
    seed: int = 42,
    config: Any = None,
) -> Pipeline:
    """
    Create SVM classification pipeline.
    
    SVM with RBF kernel is robust to outliers and works well
    for moderate-dimensional EEG feature spaces.
    
    Parameters
    ----------
    kernel : str
        SVM kernel: "rbf", "linear", "poly"
    seed : int
        Random seed
    config : Any
        Configuration object
    
    Returns
    -------
    Pipeline
        sklearn Pipeline with imputation, scaling, and SVM
    """
    cfg = get_ml_config(config)

    steps = _build_base_preprocessing_steps(cfg=cfg, include_scaling=True)
    steps.append(
        (
            "svm",
            SVC(
                kernel=kernel,
                probability=True,
                random_state=seed,
                class_weight=cfg["svm_class_weight"],
            ),
        )
    )
    return Pipeline(steps)


def create_logistic_pipeline(
    penalty: str = "l2",
    seed: int = 42,
    config: Any = None,
) -> Pipeline:
    """
    Create Logistic Regression classification pipeline.
    
    L1 penalty gives sparse solutions (feature selection).
    L2 penalty is more stable for correlated features.
    
    Parameters
    ----------
    penalty : str
        Regularization: "l1", "l2", "elasticnet"
    seed : int
        Random seed
    config : Any
        Configuration object
    
    Returns
    -------
    Pipeline
        sklearn Pipeline with imputation, scaling, and LogisticRegression
    """
    cfg = get_ml_config(config)

    solver = "saga" if penalty in ("l1", "elasticnet") else "lbfgs"

    steps = _build_base_preprocessing_steps(cfg=cfg, include_scaling=True)
    steps.append(
        (
            "lr",
            LogisticRegression(
                penalty=penalty,
                solver=solver,
                max_iter=cfg["lr_max_iter"],
                random_state=seed,
                class_weight=cfg["lr_class_weight"],
            ),
        )
    )
    return Pipeline(steps)


def create_rf_classification_pipeline(
    seed: int = 42,
    config: Any = None,
) -> Pipeline:
    """
    Create Random Forest classification pipeline.
    
    RF captures non-linear relationships and provides
    built-in feature importance.
    
    Parameters
    ----------
    seed : int
        Random seed
    config : Any
        Configuration object
    
    Returns
    -------
    Pipeline
        sklearn Pipeline with imputation and RandomForestClassifier
    """
    cfg = get_ml_config(config)

    steps = _build_base_preprocessing_steps(cfg=cfg, include_scaling=False)
    steps.append(
        (
            "rf",
            RandomForestClassifier(
                n_estimators=cfg["rf_n_estimators"],
                random_state=seed,
                class_weight=cfg["rf_class_weight"],
                n_jobs=-1,
            ),
        )
    )
    return Pipeline(steps)


def create_ensemble_pipeline(
    seed: int = 42,
    config: Any = None,
) -> Pipeline:
    """
    Create ensemble classifier combining SVM, LR, and RF.
    
    Soft voting uses probability predictions for better calibration.
    """
    cfg = get_ml_config(config)

    svm = SVC(
        kernel=cfg["svm_kernel"],
        probability=True,
        random_state=seed,
        class_weight=cfg["svm_class_weight"],
    )
    solver = "saga" if cfg["lr_penalty"] in ("l1", "elasticnet") else "lbfgs"
    lr = LogisticRegression(
        penalty=cfg["lr_penalty"],
        solver=solver,
        max_iter=cfg["lr_max_iter"],
        random_state=seed,
        class_weight=cfg["lr_class_weight"],
    )
    rf = RandomForestClassifier(
        n_estimators=cfg["rf_n_estimators"],
        random_state=seed,
        class_weight=cfg["rf_class_weight"],
        n_jobs=-1,
    )

    ensemble = VotingClassifier(
        estimators=[("svm", svm), ("lr", lr), ("rf", rf)],
        voting="soft",
    )

    steps = _build_base_preprocessing_steps(cfg=cfg, include_scaling=True)
    steps.append(("ensemble", ensemble))
    return Pipeline(steps)


###################################################################
# Parameter Grids
###################################################################


def build_svm_param_grid(config: Any = None) -> Dict[str, List]:
    """Build parameter grid for SVM hyperparameter tuning."""
    cfg = get_ml_config(config)
    return {
        "svm__C": cfg["svm_C_grid"],
        "svm__gamma": cfg["svm_gamma_grid"],
        "var__threshold": cfg["variance_threshold_grid"],
    }


def build_logistic_param_grid(config: Any = None) -> Dict[str, List]:
    """Build parameter grid for Logistic Regression."""
    cfg = get_ml_config(config)
    return {
        "lr__C": cfg["lr_C_grid"],
        "var__threshold": cfg["variance_threshold_grid"],
    }


def build_rf_classification_param_grid(config: Any = None) -> Dict[str, List]:
    """Build parameter grid for Random Forest classifier."""
    cfg = get_ml_config(config)
    return {
        "rf__max_depth": cfg["rf_max_depth_grid"],
        "rf__min_samples_leaf": [1, 3, 5],
        "var__threshold": cfg["variance_threshold_grid"],
    }


###################################################################
# Classification Metrics
###################################################################


@dataclass
class ClassificationResult:
    """Container for classification results."""
    
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: Optional[np.ndarray] = None
    groups: Optional[np.ndarray] = None
    
    # Computed metrics
    accuracy: float = np.nan
    balanced_accuracy: float = np.nan
    auc: float = np.nan
    average_precision: float = np.nan
    f1: float = np.nan
    precision: float = np.nan
    recall: float = np.nan
    specificity: float = np.nan
    
    # Confusion matrix
    confusion: Optional[np.ndarray] = None
    
    # ROC curve data
    fpr: Optional[np.ndarray] = None
    tpr: Optional[np.ndarray] = None
    thresholds: Optional[np.ndarray] = None
    
    # Per-subject metrics (for LOSO)
    per_subject_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Compute all metrics."""
        self._compute_metrics()
    
    def _compute_metrics(self):
        """Compute classification metrics."""
        if len(self.y_true) == 0:
            return
        
        # Basic metrics
        self.accuracy = float(accuracy_score(self.y_true, self.y_pred))
        self.balanced_accuracy = float(balanced_accuracy_score(self.y_true, self.y_pred))
        self.f1 = float(f1_score(self.y_true, self.y_pred, zero_division=0))
        self.precision = float(precision_score(self.y_true, self.y_pred, zero_division=0))
        self.recall = float(recall_score(self.y_true, self.y_pred, zero_division=0))
        
        # Confusion matrix
        self.confusion = confusion_matrix(self.y_true, self.y_pred)
        tn, fp, fn, tp = self.confusion.ravel()
        self.specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
        
        # AUC (requires probabilities)
        if self.y_prob is not None and len(np.unique(self.y_true)) == 2:
            try:
                self.auc = float(roc_auc_score(self.y_true, self.y_prob))
                self.average_precision = float(average_precision_score(self.y_true, self.y_prob))
                self.fpr, self.tpr, self.thresholds = roc_curve(self.y_true, self.y_prob)
            except ValueError:
                pass
        
        # Per-subject metrics
        if self.groups is not None:
            for subj in np.unique(self.groups):
                mask = self.groups == subj
                if mask.sum() < 2:
                    continue
                y_t, y_p = self.y_true[mask], self.y_pred[mask]
                rec: Dict[str, float] = {
                    "accuracy": float(accuracy_score(y_t, y_p)),
                    "n_trials": int(mask.sum()),
                }
                if len(np.unique(y_t)) == 2:
                    rec["balanced_accuracy"] = float(balanced_accuracy_score(y_t, y_p))
                if self.y_prob is not None and len(np.unique(y_t)) == 2:
                    try:
                        rec["auc"] = float(roc_auc_score(y_t, self.y_prob[mask]))
                        rec["average_precision"] = float(average_precision_score(y_t, self.y_prob[mask]))
                    except Exception:
                        pass
                self.per_subject_metrics[str(subj)] = rec
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "auc": self.auc,
            "average_precision": self.average_precision,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "n_samples": len(self.y_true),
            "n_positive": int(self.y_true.sum()),
            "n_negative": int(len(self.y_true) - self.y_true.sum()),
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Classification Results:\n"
            f"  AUC: {self.auc:.3f}\n"
            f"  Balanced Accuracy: {self.balanced_accuracy:.3f}\n"
            f"  F1: {self.f1:.3f}\n"
            f"  Sensitivity: {self.recall:.3f}\n"
            f"  Specificity: {self.specificity:.3f}\n"
            f"  N samples: {len(self.y_true)} ({int(self.y_true.sum())} pain, "
            f"{int(len(self.y_true) - self.y_true.sum())} no-pain)"
        )


###################################################################
# Quick Classification
###################################################################


def decode_pain_binary(
    X: np.ndarray,
    y: np.ndarray,
    cv: Union[str, int] = "loso",
    groups: Optional[np.ndarray] = None,
    model: str = "svm",
    seed: int = 42,
    config: Any = None,
) -> ClassificationResult:
    """
    Quick pain classification from features.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Binary labels (0=no-pain, 1=pain)
    cv : str or int
        Cross-validation strategy:
        - "loso": Leave-one-subject-out (requires groups)
        - int: Number of stratified K-folds
    groups : np.ndarray, optional
        Subject IDs for LOSO
    model : str
        Model type: "svm", "lr", "rf", "ensemble"
    seed : int
        Random seed
    config : Any
        Configuration object
    
    Returns
    -------
    ClassificationResult
        Classification results with metrics
    """
    # Validate
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    
    if X.shape[0] != len(y):
        raise ValueError(f"X and y length mismatch: {X.shape[0]} vs {len(y)}")
    
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must be binary (0 or 1)")
    
    # Create pipeline
    if model == "svm":
        pipe = create_svm_pipeline(seed=seed, config=config)
    elif model == "lr":
        pipe = create_logistic_pipeline(seed=seed, config=config)
    elif model == "rf":
        pipe = create_rf_classification_pipeline(seed=seed, config=config)
    elif model == "ensemble":
        pipe = create_ensemble_pipeline(seed=seed, config=config)
    else:
        raise ValueError(f"Unknown model: {model}")
    
    # Set up CV
    if cv == "loso":
        if groups is None:
            raise ValueError("groups required for LOSO")
        cv_obj = LeaveOneGroupOut()
        cv_splits = list(cv_obj.split(X, y, groups))
    else:
        n_splits = int(cv) if isinstance(cv, int) else 5
        cv_obj = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        cv_splits = list(cv_obj.split(X, y))
    
    # Cross-validation predictions
    y_pred = np.zeros(len(y), dtype=int)
    y_prob = np.zeros(len(y), dtype=float)
    
    for train_idx, test_idx in cv_splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        
        pipe_clone = clone(pipe)
        pipe_clone.fit(X_train, y_train)
        
        y_pred[test_idx] = pipe_clone.predict(X_test)
        if hasattr(pipe_clone, "predict_proba"):
            y_prob[test_idx] = pipe_clone.predict_proba(X_test)[:, 1]
    
    return ClassificationResult(
        y_true=y,
        y_pred=y_pred,
        y_prob=y_prob if np.any(y_prob) else None,
        groups=groups,
    )


###################################################################
# Nested LOSO Classification
###################################################################


def nested_loso_classification(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model: str = "svm",
    inner_splits: int = 3,
    seed: int = 42,
    config: Any = None,
    logger: Any = None,
) -> Tuple[ClassificationResult, pd.DataFrame]:
    """
    Nested leave-one-subject-out classification with hyperparameter tuning.
    
    Outer loop: LOSO for unbiased evaluation
    Inner loop: Stratified K-fold for hyperparameter tuning
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Binary labels
    groups : np.ndarray
        Subject IDs
    model : str
        Model type: "svm", "lr", "rf"
    inner_splits : int
        Number of inner CV splits
    seed : int
        Random seed
    config : Any
        Configuration
    logger : Any
        Logger instance
    
    Returns
    -------
    Tuple[ClassificationResult, pd.DataFrame]
        (results, best_params_df)
    """
    import logging
    log = logger or logging.getLogger(__name__)
    
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    
    # Create pipeline and param grid
    if model == "svm":
        pipe = create_svm_pipeline(seed=seed, config=config)
        param_grid = build_svm_param_grid(config)
    elif model == "lr":
        pipe = create_logistic_pipeline(seed=seed, config=config)
        param_grid = build_logistic_param_grid(config)
    elif model == "rf":
        pipe = create_rf_classification_pipeline(seed=seed, config=config)
        param_grid = build_rf_classification_param_grid(config)
    else:
        raise ValueError(f"Unknown model: {model}")
    
    outer_cv = LeaveOneGroupOut()
    
    y_pred = np.zeros(len(y), dtype=int)
    y_prob = np.zeros(len(y), dtype=float)
    best_params_records = []
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
        test_subject = groups[test_idx[0]]
        log.info(f"Fold {fold}: testing on subject {test_subject}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        train_groups = groups[train_idx]
        
        # Skip if only one class in training
        if len(np.unique(y_train)) < 2:
            log.warning(f"Fold {fold}: only one class in training, skipping")
            y_pred[test_idx] = int(np.median(y_train))
            continue
        
        # Inner CV: group-aware stratified CV to prevent within-subject mixing
        # This ensures hyperparameter tuning generalizes across subjects
        n_unique_train_groups = len(np.unique(train_groups))
        effective_splits = min(inner_splits, n_unique_train_groups)
        
        if effective_splits < 2:
            log.warning(f"Fold {fold}: <2 groups in training, fitting without inner CV")
            pipe_clone = clone(pipe)
            pipe_clone.fit(X_train, y_train)
            y_pred[test_idx] = pipe_clone.predict(X_test)
            if hasattr(pipe_clone, "predict_proba"):
                y_prob[test_idx] = pipe_clone.predict_proba(X_test)[:, 1]
            continue
        
        inner_cv = StratifiedGroupKFold(n_splits=effective_splits, shuffle=True, random_state=seed + fold)
        
        # GridSearch with group-aware inner CV
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        
        try:
            gs.fit(X_train, y_train, groups=train_groups)
            best_params_records.append({
                "fold": fold,
                "test_subject": test_subject,
                **gs.best_params_,
                "best_score": gs.best_score_,
            })
            
            y_pred[test_idx] = gs.predict(X_test)
            if hasattr(gs.best_estimator_, "predict_proba"):
                y_prob[test_idx] = gs.best_estimator_.predict_proba(X_test)[:, 1]
        except Exception as e:
            log.error(f"Fold {fold} failed: {e}")
            y_pred[test_idx] = int(np.median(y_train))
    
    result = ClassificationResult(
        y_true=y,
        y_pred=y_pred,
        y_prob=y_prob if np.any(y_prob) else None,
        groups=groups,
    )
    
    best_params_df = pd.DataFrame(best_params_records) if best_params_records else pd.DataFrame()
    
    log.info(result.summary())
    return result, best_params_df


###################################################################
# Utilities
###################################################################


def save_classification_results(
    result: ClassificationResult,
    output_path: Path,
    prefix: str = "classification",
) -> Dict[str, Path]:
    """Save classification results to files."""
    from eeg_pipeline.infra.tsv import write_tsv
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    saved = {}
    
    # Metrics
    metrics_df = pd.DataFrame([result.to_dict()])
    metrics_path = output_path / f"{prefix}_metrics.tsv"
    write_tsv(metrics_df, metrics_path)
    saved["metrics"] = metrics_path
    
    # Predictions
    pred_df = pd.DataFrame({
        "y_true": result.y_true,
        "y_pred": result.y_pred,
    })
    if result.y_prob is not None:
        pred_df["y_prob"] = result.y_prob
    if result.groups is not None:
        pred_df["group"] = result.groups
    
    pred_path = output_path / f"{prefix}_predictions.tsv"
    write_tsv(pred_df, pred_path)
    saved["predictions"] = pred_path
    
    # ROC curve data
    if result.fpr is not None:
        roc_df = pd.DataFrame({
            "fpr": result.fpr,
            "tpr": result.tpr,
            "threshold": result.thresholds,
        })
        roc_path = output_path / f"{prefix}_roc.tsv"
        write_tsv(roc_df, roc_path)
        saved["roc"] = roc_path
    
    return saved
