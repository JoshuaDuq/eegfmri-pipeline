"""
Decoding Pipeline (Canonical)
=============================

Pipeline class for ML-based EEG decoding analysis.
This module provides the PipelineBase subclass for decoding.

The actual algorithm implementations remain in analysis/decoding/:
- cv.py: Cross-validation utilities and fold management
- pipelines.py: ML pipeline factories (ElasticNet, RF)
- time_generalization.py: Temporal generalization analysis
- classification.py: Binary classification
- shap_importance.py: SHAP feature importance
- uncertainty.py: Prediction intervals and calibration

Usage:
    pipeline = DecodingPipeline(config=config)
    pipeline.run_batch(["0001", "0002"])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.pipelines.base import PipelineBase
from eeg_pipeline.utils.config.loader import load_settings
from eeg_pipeline.utils.io.general import get_logger, ensure_dir


###################################################################
# Pipeline Class
###################################################################


class DecodingPipeline(PipelineBase):
    """Pipeline for ML-based EEG decoding analysis.
    
    Unlike other pipelines, decoding requires multiple subjects for LOSO CV.
    The process_subject method is not used; instead use run_decoding directly.
    """
    
    def __init__(self, config: Optional[Any] = None):
        super().__init__(name="decoding", config=config)
        self.results_root = self.deriv_root / "decoding"

    def process_subject(self, subject: str, task: Optional[str] = None, **kwargs) -> None:
        raise NotImplementedError(
            "DecodingPipeline requires multiple subjects for LOSO CV. "
            "Use run_decoding() or run_batch() instead."
        )

    def run_batch(self, subjects: List[str], task: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        task = task or self.config.get("project.task")
        if task is None:
            raise ValueError("Missing required config value: project.task")
        
        min_subjects = self.config.get("analysis.min_subjects_for_group", 2)
        if len(subjects) < min_subjects:
            raise ValueError(f"Decoding requires at least {min_subjects} subjects, got {len(subjects)}")
        
        n_perm = kwargs.get("n_perm", 0)
        inner_splits = kwargs.get("inner_splits", 3)
        outer_jobs = kwargs.get("outer_jobs", 1)
        rng_seed = kwargs.get("rng_seed") or self.config.get("project.random_state", 42)
        skip_time_gen = kwargs.get("skip_time_gen", False)
        
        self.logger.info(
            f"Starting decoding: {len(subjects)} subjects, task={task}, n_perm={n_perm}, "
            f"inner_splits={inner_splits}, outer_jobs={outer_jobs}"
        )
        
        results_dir = self.run_regression_decoding(
            subjects=subjects,
            task=task,
            n_perm=n_perm,
            inner_splits=inner_splits,
            outer_jobs=outer_jobs,
            rng_seed=rng_seed,
        )
        
        if not skip_time_gen:
            self.run_time_generalization(
                subjects=subjects,
                task=task,
                n_perm=n_perm,
                rng_seed=rng_seed,
            )
        
        self.logger.info("Decoding complete.")
        
        return [{"subjects": subjects, "status": "success", "results_dir": str(results_dir)}]

    def run_regression_decoding(
        self,
        subjects: List[str],
        task: str,
        n_perm: int = 0,
        inner_splits: int = 3,
        outer_jobs: int = 1,
        rng_seed: int = 42,
    ) -> Path:
        from eeg_pipeline.utils.io.decoding import export_predictions, export_indices, prepare_best_params_path
        from eeg_pipeline.plotting.decoding import (
            plot_prediction_scatter,
            plot_per_subject_performance,
            plot_residual_diagnostics,
            plot_permutation_null,
        )
        from eeg_pipeline.utils.data.loading import load_plateau_matrix
        from eeg_pipeline.analysis.decoding.pipelines import create_elasticnet_pipeline, build_elasticnet_param_grid
        from eeg_pipeline.analysis.decoding.cv import nested_loso_predictions_matrix, compute_subject_level_r

        X, y, groups, feature_names, meta = load_plateau_matrix(subjects, task, self.deriv_root, self.config, self.logger)

        results_dir = self.results_root / "regression"
        plots_dir = results_dir / "plots"
        ensure_dir(results_dir)
        ensure_dir(plots_dir)

        pipe = create_elasticnet_pipeline(seed=rng_seed, config=self.config)
        param_grid = build_elasticnet_param_grid(self.config)
        best_params_path = prepare_best_params_path(results_dir / "best_params_elasticnet.jsonl", mode="truncate")
        null_path = results_dir / "loso_null_elasticnet.npz" if n_perm > 0 else None

        y_true, y_pred, groups_ordered, test_indices, fold_ids = nested_loso_predictions_matrix(
            X=X, y=y, groups=groups, pipe=pipe, param_grid=param_grid,
            inner_cv_splits=inner_splits, n_jobs=-1, seed=rng_seed,
            best_params_log_path=best_params_path, model_name="elasticnet",
            outer_n_jobs=outer_jobs, null_n_perm=n_perm, null_output_path=null_path,
        )

        pred_path = results_dir / "loso_predictions.tsv"
        pred_df = export_predictions(
            y_true, y_pred, groups_ordered, test_indices, fold_ids,
            "elasticnet", meta.reset_index(drop=True), pred_path,
        )
        export_indices(groups_ordered, test_indices, fold_ids, meta.reset_index(drop=True),
                       results_dir / "loso_indices.tsv", add_heldout_subject_id=True)

        r_subj, per_subj_r, ci_low, ci_high = compute_subject_level_r(pred_df, self.config)
        p_val = np.nan

        if null_path and null_path.exists():
            data = np.load(null_path)
            null_rs = data.get("null_r")
            if null_rs is not None and null_rs.size > 0 and np.isfinite(r_subj):
                finite = null_rs[np.isfinite(null_rs)]
                if finite.size > 0:
                    p_val = float(((np.abs(finite) >= abs(r_subj)).sum() + 1) / (finite.size + 1))

        try:
            from sklearn.metrics import r2_score
            r2_val = float(r2_score(y_true, y_pred))
        except Exception:
            r2_val = np.nan

        metrics = {
            "pearson_r": r_subj,
            "r_subject_ci_low": ci_low,
            "r_subject_ci_high": ci_high,
            "r2": r2_val,
            "p_value": p_val,
        }

        plot_prediction_scatter(pred_df, "elasticnet", metrics, plots_dir / "loso_prediction_scatter.png", config=self.config)
        plot_per_subject_performance(pred_df, "elasticnet", plots_dir / "loso_per_subject_performance.png", config=self.config)
        plot_residual_diagnostics(pred_df, "elasticnet", plots_dir / "loso_residual_diagnostics.png", config=self.config)

        if null_path and null_path.exists():
            data = np.load(null_path)
            null_rs = data.get("null_r")
            if null_rs is not None and null_rs.size > 0 and np.isfinite(r_subj):
                plot_permutation_null(null_rs, r_subj, p_val, plots_dir / "loso_null_distribution.png", config=self.config)

        self.logger.info(f"Saved decoding results to {results_dir}")
        return results_dir

    def run_time_generalization(
        self,
        subjects: List[str],
        task: str,
        n_perm: int = 0,
        rng_seed: int = 42,
    ) -> None:
        from eeg_pipeline.plotting.decoding import plot_time_generalization_with_null
        from eeg_pipeline.analysis.decoding.time_generalization import time_generalization_regression

        results_dir = self.results_root / "time_generalization"
        ensure_dir(results_dir)

        try:
            tg_r, tg_r2, window_centers = time_generalization_regression(
                deriv_root=self.deriv_root,
                subjects=subjects,
                task=task,
                results_dir=results_dir,
                config_dict=self.config,
                n_perm=n_perm,
                seed=rng_seed,
            )
        except Exception as exc:
            self.logger.warning(f"Time-generalization failed: {exc}")
            return

        tg_path = results_dir / "time_generalization_regression.npz"
        tg_npz = np.load(tg_path) if tg_path.exists() else None
        null_mat = tg_npz["null_r"] if tg_npz and "null_r" in tg_npz else None

        plot_time_generalization_with_null(
            tg_matrix=tg_r, null_matrix=null_mat, window_centers=window_centers,
            save_path=results_dir / "time_generalization_r.png", metric="r", config=self.config,
        )
        plot_time_generalization_with_null(
            tg_matrix=tg_r2, null_matrix=null_mat, window_centers=window_centers,
            save_path=results_dir / "time_generalization_r2.png", metric="r2", config=self.config,
        )

        self.logger.info(f"Saved time-generalization outputs to {results_dir}")


###################################################################
# Module-Level Entry Points (delegating to orchestration)
###################################################################


def run_regression_decoding(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config: Any,
    n_perm: int,
    inner_splits: int,
    outer_jobs: int,
    rng_seed: int,
    results_root: Path,
    logger: logging.Logger,
) -> Path:
    pipeline = DecodingPipeline(config=config)
    pipeline.logger = logger
    pipeline.results_root = results_root
    return pipeline.run_regression_decoding(
        subjects=subjects,
        task=task,
        n_perm=n_perm,
        inner_splits=inner_splits,
        outer_jobs=outer_jobs,
        rng_seed=rng_seed,
    )


def run_time_generalization(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config: Any,
    n_perm: int,
    rng_seed: int,
    results_root: Path,
    logger: logging.Logger,
) -> None:
    pipeline = DecodingPipeline(config=config)
    pipeline.logger = logger
    pipeline.results_root = results_root
    pipeline.run_time_generalization(
        subjects=subjects,
        task=task,
        n_perm=n_perm,
        rng_seed=rng_seed,
    )


__all__ = [
    "DecodingPipeline",
    "run_regression_decoding",
    "run_time_generalization",
]
