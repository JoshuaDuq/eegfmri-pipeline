"""Machine Learning Pipeline (Canonical)
======================================

Thin pipeline wrapper that delegates ML orchestration to
eeg_pipeline.analysis.machine_learning.orchestration.

Compute and visualization are separated (SRP):
- run_batch(): compute only, writes stable results contract
- run_batch_with_plots(): compute + visualization (convenience)
- visualize(): plot from existing results on disk

Modes:
- regression: LOSO regression predicting pain intensity
- timegen: Time-generalization analysis
- classify: Binary pain classification

Usage:
    pipeline = MLPipeline(config=config)
    
    # Compute only (SRP)
    pipeline.run_batch(["0001", "0002"], mode="regression")
    
    # Compute + visualize
    pipeline.run_batch_with_plots(["0001", "0002"], mode="regression")
    
    # Visualize from disk
    pipeline.visualize(results_dir, mode="regression")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from eeg_pipeline.analysis.machine_learning.orchestration import (
    run_regression_ml,
    run_within_subject_regression_ml,
    run_time_generalization,
    run_classification_ml,
    run_model_comparison_ml,
    run_incremental_validity_ml,
)
from eeg_pipeline.pipelines.base import PipelineBase
from eeg_pipeline.plotting.orchestration.machine_learning import (
    visualize_regression_from_disk,
    visualize_time_generalization_from_disk,
    visualize_classification_from_disk,
)


MLMode = Literal[
    "regression", "timegen", "classify",
    "model_comparison", "incremental_validity",
    "uncertainty", "shap", "permutation",
]


###################################################################
# Pipeline Class
###################################################################


class MLPipeline(PipelineBase):
    """Pipeline for ML-based EEG analysis.
    
    Unlike other pipelines, ML requires multiple subjects for LOSO CV.
    The process_subject method is not used; instead use run_batch directly.
    
    Modes:
        - regression: LOSO regression predicting pain intensity
        - timegen: Time-generalization analysis
        - classify: Binary pain classification
    """
    
    def __init__(self, config: Optional[Any] = None):
        super().__init__(name="machine_learning", config=config)
        self.results_root = self.deriv_root / "machine_learning"

    def process_subject(self, subject: str, task: Optional[str] = None, **kwargs) -> None:
        raise NotImplementedError(
            "MLPipeline requires multiple subjects for LOSO CV. "
            "Use run_batch() instead."
        )

    def run_batch(
        self,
        subjects: List[str],
        task: Optional[str] = None,
        mode: MLMode = "regression",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Run ML pipeline (compute only, no visualization).
        
        Single responsibility: Compute and write results contract.
        
        Args:
            subjects: List of subject IDs
            task: Task name
            mode: "regression", "timegen", or "classify"
            **kwargs: Additional options (n_perm, inner_splits, etc.)
        
        Returns:
            List of result dicts with results_dir paths
        """
        from eeg_pipeline.cli.common import ProgressReporter
        
        task = task or self.config.get("project.task")
        if task is None:
            raise ValueError("Missing required config value: project.task")

        cv_scope = kwargs.get("cv_scope", "group")
        if cv_scope not in {"group", "subject"}:
            raise ValueError(f"Invalid cv_scope: {cv_scope} (expected 'group' or 'subject')")

        if cv_scope == "group":
            min_subjects = self.config.get("analysis.min_subjects_for_group", 2)
            if len(subjects) < min_subjects:
                raise ValueError(f"ML pipeline requires at least {min_subjects} subjects, got {len(subjects)}")
        
        progress = kwargs.get("progress") or ProgressReporter(enabled=False)
        n_perm = kwargs.get("n_perm", 0)
        inner_splits = kwargs.get("inner_splits", 3)
        outer_jobs = kwargs.get("outer_jobs", 1)
        rng_seed = kwargs.get("rng_seed") or self.config.get("project.random_state", 42)
        model = kwargs.get("model", "elasticnet")

        self.logger.info(
            f"Starting ML pipeline (mode={mode}, cv_scope={cv_scope}, model={model}): {len(subjects)} subjects, "
            f"task={task}, n_perm={n_perm}, inner_splits={inner_splits}"
        )
        
        progress.start("machine_learning", subjects)
        results_dirs: List[Path] = []

        if mode == "regression":
            progress.step("Regression ML", current=1, total=1)
            if cv_scope == "subject":
                results_dir = run_within_subject_regression_ml(
                    subjects=subjects,
                    task=task,
                    deriv_root=self.deriv_root,
                    config=self.config,
                    n_perm=n_perm,
                    inner_splits=inner_splits,
                    outer_jobs=outer_jobs,
                    rng_seed=rng_seed,
                    results_root=self.results_root,
                    logger=self.logger,
                    model=model,
                )
            else:
                results_dir = run_regression_ml(
                    subjects=subjects,
                    task=task,
                    deriv_root=self.deriv_root,
                    config=self.config,
                    n_perm=n_perm,
                    inner_splits=inner_splits,
                    outer_jobs=outer_jobs,
                    rng_seed=rng_seed,
                    results_root=self.results_root,
                    logger=self.logger,
                    model=model,
                )
            results_dirs.append(results_dir)

        elif mode == "timegen":
            if cv_scope == "subject":
                self.logger.warning("Time-generalization requires group scope; skipping.")
            else:
                progress.step("Time generalization", current=1, total=1)
                run_time_generalization(
                    subjects=subjects,
                    task=task,
                    deriv_root=self.deriv_root,
                    config=self.config,
                    n_perm=n_perm,
                    rng_seed=rng_seed,
                    results_root=self.results_root,
                    logger=self.logger,
                )
                results_dirs.append(self.results_root / "time_generalization")

        elif mode == "classify":
            progress.step("Classification ML", current=1, total=1)
            results_dir = run_classification_ml(
                subjects=subjects,
                task=task,
                deriv_root=self.deriv_root,
                config=self.config,
                n_perm=n_perm,
                inner_splits=inner_splits,
                outer_jobs=outer_jobs,
                rng_seed=rng_seed,
                results_root=self.results_root,
                logger=self.logger,
            )
            results_dirs.append(results_dir)

        elif mode == "model_comparison":
            progress.step("Model Comparison", current=1, total=1)
            results_dir = run_model_comparison_ml(
                subjects=subjects,
                task=task,
                deriv_root=self.deriv_root,
                config=self.config,
                inner_splits=inner_splits,
                rng_seed=rng_seed,
                results_root=self.results_root,
                logger=self.logger,
            )
            results_dirs.append(results_dir)

        elif mode == "incremental_validity":
            progress.step("Incremental Validity", current=1, total=1)
            results_dir = run_incremental_validity_ml(
                subjects=subjects,
                task=task,
                deriv_root=self.deriv_root,
                config=self.config,
                n_perm=n_perm,
                inner_splits=inner_splits,
                rng_seed=rng_seed,
                results_root=self.results_root,
                logger=self.logger,
            )
            results_dirs.append(results_dir)

        elif mode == "uncertainty":
            progress.step("Uncertainty Quantification", current=1, total=1)
            alpha = kwargs.get("uncertainty_alpha", 0.1)
            results_dir = self._run_uncertainty(
                subjects=subjects,
                task=task,
                rng_seed=rng_seed,
                alpha=alpha,
            )
            if results_dir:
                results_dirs.append(results_dir)

        elif mode == "shap":
            progress.step("SHAP Importance", current=1, total=1)
            results_dir = self._run_shap(
                subjects=subjects,
                task=task,
                rng_seed=rng_seed,
            )
            if results_dir:
                results_dirs.append(results_dir)

        elif mode == "permutation":
            progress.step("Permutation Importance", current=1, total=1)
            n_repeats = kwargs.get("perm_n_repeats", 10)
            results_dir = self._run_permutation_importance(
                subjects=subjects,
                task=task,
                rng_seed=rng_seed,
                n_repeats=n_repeats,
            )
            if results_dir:
                results_dirs.append(results_dir)

        else:
            raise ValueError(
                f"Unknown mode: {mode} (expected one of: regression, timegen, classify, "
                f"model_comparison, incremental_validity, uncertainty, shap, permutation)"
            )
        
        self.logger.info(f"ML pipeline ({mode}) complete.")
        progress.complete(success=True)
        
        return [{"subjects": subjects, "status": "success", "mode": mode, "results_dir": str(d)} for d in results_dirs]

    def visualize(
        self,
        results_dir: Path,
        mode: MLMode = "regression",
    ) -> None:
        """Visualize ML results from disk.
        
        Single responsibility: Read results contract and create plots.
        """
        if mode == "regression":
            visualize_regression_from_disk(
                results_dir=results_dir,
                config=self.config,
                logger=self.logger,
            )
        elif mode == "timegen":
            visualize_time_generalization_from_disk(
                results_dir=results_dir,
                config=self.config,
                logger=self.logger,
            )
        elif mode == "classify":
            visualize_classification_from_disk(
                results_dir=results_dir,
                config=self.config,
                logger=self.logger,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _run_uncertainty(
        self,
        subjects: List[str],
        task: str,
        rng_seed: int,
        alpha: float = 0.1,
    ) -> Optional[Path]:
        """Run uncertainty quantification via conformal prediction."""
        from eeg_pipeline.analysis.machine_learning.orchestration import _run_uncertainty_stage
        from eeg_pipeline.utils.data.machine_learning import load_active_matrix
        
        X, y, groups, feature_names, meta = load_active_matrix(
            subjects, task, self.deriv_root, self.config, self.logger
        )
        
        results_dir = self.results_root / "uncertainty"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        return _run_uncertainty_stage(
            X=X, y=y, groups=groups,
            config=self.config, seed=rng_seed, alpha=alpha,
            results_dir=results_dir, logger=self.logger,
        )

    def _run_shap(
        self,
        subjects: List[str],
        task: str,
        rng_seed: int,
    ) -> Optional[Path]:
        """Run SHAP-based feature importance."""
        from eeg_pipeline.analysis.machine_learning.orchestration import _run_shap_importance_stage
        from eeg_pipeline.utils.data.machine_learning import load_active_matrix
        
        X, y, groups, feature_names, meta = load_active_matrix(
            subjects, task, self.deriv_root, self.config, self.logger
        )
        
        results_dir = self.results_root / "shap"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        return _run_shap_importance_stage(
            X=X, y=y, groups=groups, feature_names=feature_names,
            config=self.config, seed=rng_seed,
            results_dir=results_dir, logger=self.logger,
        )

    def _run_permutation_importance(
        self,
        subjects: List[str],
        task: str,
        rng_seed: int,
        n_repeats: int = 10,
    ) -> Optional[Path]:
        """Run permutation-based feature importance."""
        from eeg_pipeline.analysis.machine_learning.orchestration import _run_permutation_importance_stage
        from eeg_pipeline.utils.data.machine_learning import load_active_matrix
        
        X, y, groups, feature_names, meta = load_active_matrix(
            subjects, task, self.deriv_root, self.config, self.logger
        )
        
        results_dir = self.results_root / "permutation_importance"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        return _run_permutation_importance_stage(
            X=X, y=y, groups=groups, feature_names=feature_names,
            config=self.config, seed=rng_seed, n_repeats=n_repeats,
            results_dir=results_dir, logger=self.logger,
        )

    def run_batch_with_plots(
        self,
        subjects: List[str],
        task: Optional[str] = None,
        mode: MLMode = "regression",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Run ML pipeline with visualization (convenience wrapper).
        
        Composes: run_batch + visualize
        """
        results = self.run_batch(subjects, task, mode, **kwargs)
        
        for r in results:
            if r.get("status") == "success" and r.get("results_dir"):
                self.visualize(Path(r["results_dir"]), mode)
        
        return results


__all__ = [
    "MLPipeline",
]
