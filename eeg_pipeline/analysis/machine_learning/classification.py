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

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import f_classif
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
    GroupKFold,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from eeg_pipeline.analysis.machine_learning.config import get_ml_config
from eeg_pipeline.analysis.machine_learning.cv import apply_fold_feature_harmonization
from eeg_pipeline.analysis.machine_learning.preprocessing import (
    build_base_preprocessing_steps,
)
from eeg_pipeline.utils.config.loader import get_config_value

logger = logging.getLogger(__name__)


###################################################################
# Pipeline Factories
###################################################################

def _append_classification_resampler(steps: List[Tuple[str, Any]], cfg: Dict[str, Any]) -> None:
    resampler = str(cfg.get("classification_resampler", "none")).strip().lower()
    if resampler == "undersample":
        steps.append(
            (
                "resampler",
                RandomUnderSampler(random_state=int(cfg.get("classification_resampler_seed", 42))),
            )
        )
    elif resampler == "smote":
        steps.append(("resampler", SMOTE(random_state=int(cfg.get("classification_resampler_seed", 42)))))



def create_svm_pipeline(
    kernel: str = "rbf",
    seed: int = 42,
    config: Any = None,
    n_covariates: int = 0,
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

    steps = build_base_preprocessing_steps(
        cfg=cfg,
        include_scaling=True,
        n_covariates=n_covariates,
        config=config,
        score_func=f_classif,
    )
    _append_classification_resampler(steps, cfg)
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
    return ImbPipeline(steps)


def create_logistic_pipeline(
    penalty: str = "l2",
    seed: int = 42,
    config: Any = None,
    n_covariates: int = 0,
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

    steps = build_base_preprocessing_steps(
        cfg=cfg,
        include_scaling=True,
        n_covariates=n_covariates,
        config=config,
        score_func=f_classif,
    )
    _append_classification_resampler(steps, cfg)
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
    return ImbPipeline(steps)


def create_rf_classification_pipeline(
    seed: int = 42,
    config: Any = None,
    n_covariates: int = 0,
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

    steps = build_base_preprocessing_steps(
        cfg=cfg,
        include_scaling=False,
        n_covariates=n_covariates,
        config=config,
        score_func=f_classif,
    )
    _append_classification_resampler(steps, cfg)
    steps.append(
        (
            "rf",
            RandomForestClassifier(
                n_estimators=cfg["rf_n_estimators"],
                random_state=seed,
                class_weight=cfg["rf_class_weight"],
                n_jobs=1,
            ),
        )
    )
    return ImbPipeline(steps)


def create_ensemble_pipeline(
    seed: int = 42,
    config: Any = None,
    n_covariates: int = 0,
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
        n_jobs=1,
    )

    if cfg.get("calibrate_ensemble", False):
        from sklearn.calibration import CalibratedClassifierCV
        ensemble = VotingClassifier(
            estimators=[
                ("svm", CalibratedClassifierCV(svm, method="sigmoid", cv=2)),
                ("lr", lr), # L2 logistic regression is natively well calibrated
                ("rf", CalibratedClassifierCV(rf, method="sigmoid", cv=2)),
            ],
            voting="soft",
        )
    else:
        ensemble = VotingClassifier(
            estimators=[("svm", svm), ("lr", lr), ("rf", rf)],
            voting="soft",
        )

    steps = build_base_preprocessing_steps(
        cfg=cfg,
        include_scaling=True,
        n_covariates=n_covariates,
        config=config,
        score_func=f_classif,
    )
    _append_classification_resampler(steps, cfg)
    steps.append(("ensemble", ensemble))
    return ImbPipeline(steps)


###################################################################
# Parameter Grids
###################################################################


def build_svm_param_grid(config: Any = None, n_covariates: int = 0) -> Dict[str, List]:
    """Build parameter grid for SVM hyperparameter tuning."""
    cfg = get_ml_config(config)
    var_prefix = "preprocessing__eeg__var" if n_covariates > 0 else "var"
    return {
        "svm__C": cfg["svm_C_grid"],
        "svm__gamma": cfg["svm_gamma_grid"],
        f"{var_prefix}__threshold": cfg["variance_threshold_grid"],
    }


def build_logistic_param_grid(config: Any = None, n_covariates: int = 0) -> Dict[str, List]:
    """Build parameter grid for Logistic Regression."""
    cfg = get_ml_config(config)
    var_prefix = "preprocessing__eeg__var" if n_covariates > 0 else "var"
    return {
        "lr__C": cfg["lr_C_grid"],
        f"{var_prefix}__threshold": cfg["variance_threshold_grid"],
    }


def build_rf_classification_param_grid(config: Any = None, n_covariates: int = 0) -> Dict[str, List]:
    """Build parameter grid for Random Forest classifier."""
    cfg = get_ml_config(config)
    var_prefix = "preprocessing__eeg__var" if n_covariates > 0 else "var"
    return {
        "rf__max_depth": cfg["rf_max_depth_grid"],
        "rf__min_samples_leaf": [1, 3, 5],
        f"{var_prefix}__threshold": cfg["variance_threshold_grid"],
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
    mean_subject_auc: float = np.nan
    fold_ids: Optional[np.ndarray] = None
    test_indices: Optional[np.ndarray] = None
    failed_fold_count: int = 0
    n_folds_total: int = 0
    
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
        self.confusion = confusion_matrix(self.y_true, self.y_pred, labels=[0, 1])
        tn, fp, fn, tp = self.confusion.ravel()
        self.specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
        
        # AUC (requires probabilities)
        if self.y_prob is not None:
            try:
                prob_mask = np.isfinite(self.y_prob) & np.isfinite(self.y_true)
                if np.sum(prob_mask) >= 2 and len(np.unique(self.y_true[prob_mask])) == 2:
                    self.auc = float(roc_auc_score(self.y_true[prob_mask], self.y_prob[prob_mask]))
                    self.average_precision = float(
                        average_precision_score(self.y_true[prob_mask], self.y_prob[prob_mask])
                    )
                    self.fpr, self.tpr, self.thresholds = roc_curve(
                        self.y_true[prob_mask], self.y_prob[prob_mask]
                    )
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
                    "precision": float(precision_score(y_t, y_p, zero_division=0)),
                    "recall": float(recall_score(y_t, y_p, zero_division=0)),
                    "f1": float(f1_score(y_t, y_p, zero_division=0)),
                    "n_trials": int(mask.sum()),
                }
                cm_subj = confusion_matrix(y_t, y_p, labels=[0, 1]).astype(float)
                tn, fp, fn, tp = cm_subj.ravel()
                rec["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
                support = cm_subj.sum(axis=1)
                if np.sum(support > 0) < 2:
                    rec["balanced_accuracy"] = np.nan
                else:
                    recall = np.full(2, np.nan, dtype=float)
                    valid_support = support > 0
                    recall[valid_support] = np.diag(cm_subj)[valid_support] / support[valid_support]
                    rec["balanced_accuracy"] = float(np.nanmean(recall))
                if self.y_prob is not None and len(np.unique(y_t)) == 2:
                    try:
                        subj_prob = self.y_prob[mask]
                        prob_mask = np.isfinite(subj_prob) & np.isfinite(y_t)
                        if np.sum(prob_mask) >= 2 and len(np.unique(y_t[prob_mask])) == 2:
                            rec["auc"] = float(roc_auc_score(y_t[prob_mask], subj_prob[prob_mask]))
                            rec["average_precision"] = float(
                                average_precision_score(y_t[prob_mask], subj_prob[prob_mask])
                            )
                    except Exception as exc:
                        logger.debug(
                            "Skipping per-subject probability metrics for subject=%s: %s",
                            subj,
                            exc,
                        )
                self.per_subject_metrics[str(subj)] = rec
                
        # Compute mean subject AUC if available
        if self.per_subject_metrics:
            aucs = [m["auc"] for m in self.per_subject_metrics.values() if "auc" in m and np.isfinite(m["auc"])]
            if aucs:
                self.mean_subject_auc = float(np.mean(aucs))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "auc": self.auc,
            "mean_subject_auc": self.mean_subject_auc,
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
        auc_str = f"{self.mean_subject_auc:.3f} (mean per-subject) / {self.auc:.3f} (pooled)" if np.isfinite(self.mean_subject_auc) else f"{self.auc:.3f} (pooled)"
        return (
            f"Classification Results:\n"
            f"  AUC: {auc_str}\n"
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
        pipe = create_svm_pipeline(seed=seed, config=config, n_covariates=0)
    elif model == "lr":
        pipe = create_logistic_pipeline(seed=seed, config=config, n_covariates=0)
    elif model == "rf":
        pipe = create_rf_classification_pipeline(seed=seed, config=config, n_covariates=0)
    elif model == "ensemble":
        pipe = create_ensemble_pipeline(seed=seed, config=config, n_covariates=0)
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
        if groups is not None:
            groups_arr = np.asarray(groups)
            n_unique_groups = len(np.unique(groups_arr))
            n_splits = min(n_splits, n_unique_groups)
            if n_splits < 2:
                raise ValueError("Grouped classification CV requires at least 2 unique groups")
            cv_obj = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            cv_splits = list(cv_obj.split(X, y, groups_arr))
        else:
            cv_obj = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            cv_splits = list(cv_obj.split(X, y))
    
    # Cross-validation predictions
    y_pred = np.zeros(len(y), dtype=int)
    y_prob = np.full(len(y), np.nan, dtype=float)
    fold_ids = np.zeros(len(y), dtype=int)
    test_indices = np.arange(len(y), dtype=int)

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits, start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        groups_train = groups[train_idx] if groups is not None else None

        X_train, X_test, _ = apply_fold_feature_harmonization(
            X_train,
            X_test,
            groups_train if groups_train is not None else np.array(["all"] * len(X_train), dtype=object),
            "intersection" if groups_train is not None else "union_impute",
            n_covariates=0,
        )
        
        pipe_clone = clone(pipe)
        pipe_clone.fit(X_train, y_train)

        y_pred[test_idx] = pipe_clone.predict(X_test)
        fold_ids[test_idx] = int(fold_idx)
        if hasattr(pipe_clone, "predict_proba"):
            y_prob[test_idx] = pipe_clone.predict_proba(X_test)[:, 1]
    
    return ClassificationResult(
        y_true=y,
        y_pred=y_pred,
        y_prob=y_prob if np.any(np.isfinite(y_prob)) else None,
        groups=groups,
        fold_ids=fold_ids,
        test_indices=test_indices,
        failed_fold_count=0,
        n_folds_total=int(len(cv_splits)),
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
    harmonization_mode: Optional[str] = None,
    n_covariates: int = 0,
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
    n_covariates : int
        Number of covariate columns appended to X
    
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
        pipe = create_svm_pipeline(seed=seed, config=config, n_covariates=n_covariates)
        param_grid = build_svm_param_grid(config, n_covariates=n_covariates)
    elif model == "lr":
        pipe = create_logistic_pipeline(seed=seed, config=config, n_covariates=n_covariates)
        param_grid = build_logistic_param_grid(config, n_covariates=n_covariates)
    elif model == "rf":
        pipe = create_rf_classification_pipeline(seed=seed, config=config, n_covariates=n_covariates)
        param_grid = build_rf_classification_param_grid(config, n_covariates=n_covariates)
    else:
        raise ValueError(f"Unknown model: {model}")
    
    outer_cv = LeaveOneGroupOut()
    outer_splits = list(outer_cv.split(X, y, groups))
    
    y_pred = np.zeros(len(y), dtype=int)
    y_prob = np.full(len(y), np.nan, dtype=float)
    fold_ids = np.zeros(len(y), dtype=int)
    test_indices = np.arange(len(y), dtype=int)
    best_params_records = []
    failed_fold_count = 0
    n_folds_total = len(outer_splits)
    
    scoring_metric = str(get_config_value(config, "machine_learning.classification.scoring", "average_precision")).strip()

    for fold, (train_idx, test_idx) in enumerate(outer_splits):
        fold_number = int(fold + 1)
        test_subject = groups[test_idx[0]]
        log.info(f"Fold {fold_number}: testing on subject {test_subject}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        train_groups = groups[train_idx]
        X_train, X_test, _ = apply_fold_feature_harmonization(
            X_train,
            X_test,
            train_groups,
            harmonization_mode,
            n_covariates=n_covariates,
        )
        
        # Skip if only one class in training
        if len(np.unique(y_train)) < 2:
            log.warning(f"Fold {fold_number}: only one class in training, skipping")
            y_pred[test_idx] = int(np.median(y_train))
            fold_ids[test_idx] = fold_number
            failed_fold_count += 1
            continue
        
        # Inner CV: group-aware stratified CV to prevent within-subject mixing
        # This ensures hyperparameter tuning generalizes across subjects
        n_unique_train_groups = len(np.unique(train_groups))
        effective_splits = min(inner_splits, n_unique_train_groups)
        
        # Check if minority class has enough samples for StratifiedGroupKFold
        counts = np.bincount(y_train)
        min_class_count = np.min(counts) if len(counts) > 0 else 0
        
        if effective_splits < 2:
            log.warning(f"Fold {fold_number}: Not enough groups for inner CV. Skipping tuning.")
            try:
                pipe_clone = clone(pipe)
                pipe_clone.fit(X_train, y_train)
                y_pred[test_idx] = pipe_clone.predict(X_test)
                fold_ids[test_idx] = fold_number
                if hasattr(pipe_clone, "predict_proba"):
                    y_prob[test_idx] = pipe_clone.predict_proba(X_test)[:, 1]
            except Exception as e:
                log.error(f"Fold {fold_number} fallback fit failed: {e}")
                y_pred[test_idx] = int(np.median(y_train))
                fold_ids[test_idx] = fold_number
                failed_fold_count += 1
            continue
            
        if min_class_count < effective_splits:
            log.warning(
                f"Fold {fold_number}: Minority class count ({min_class_count}) < effective_splits ({effective_splits}). "
                "Falling back to GroupKFold to maintain subject isolation."
            )
            inner_cv = GroupKFold(n_splits=effective_splits)
            cv_groups = train_groups
        else:
            inner_cv = StratifiedGroupKFold(n_splits=effective_splits, shuffle=True, random_state=seed + fold)
            cv_groups = train_groups
        
        # GridSearch with group-aware inner CV
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=scoring_metric,
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
            error_score=np.nan,
        )
        
        try:
            gs.fit(X_train, y_train, groups=cv_groups)
            best_params_records.append({
                "fold": fold_number,
                "test_subject": test_subject,
                **gs.best_params_,
                "best_score": gs.best_score_,
            })
            
            y_pred[test_idx] = gs.predict(X_test)
            fold_ids[test_idx] = fold_number
            if hasattr(gs.best_estimator_, "predict_proba"):
                y_prob[test_idx] = gs.best_estimator_.predict_proba(X_test)[:, 1]
        except Exception as e:
            log.error(f"Fold {fold_number} failed: {e}")
            y_pred[test_idx] = int(np.median(y_train))
            fold_ids[test_idx] = fold_number
            failed_fold_count += 1
    
    result = ClassificationResult(
        y_true=y,
        y_pred=y_pred,
        y_prob=y_prob if np.any(np.isfinite(y_prob)) else None,
        groups=groups,
        fold_ids=fold_ids,
        test_indices=test_indices,
        failed_fold_count=int(failed_fold_count),
        n_folds_total=int(n_folds_total),
    )
    
    best_params_df = pd.DataFrame(best_params_records) if best_params_records else pd.DataFrame()
    
    log.info(result.summary())
    return result, best_params_df
