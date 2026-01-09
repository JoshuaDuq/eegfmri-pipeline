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
from typing import Any, Callable, Dict, List, Literal, Optional

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


DEFAULT_N_PERM = 0
DEFAULT_INNER_SPLITS = 3
DEFAULT_OUTER_JOBS = 1
DEFAULT_RNG_SEED = 42
DEFAULT_MODEL = "elasticnet"
DEFAULT_UNCERTAINTY_ALPHA = 0.1
DEFAULT_PERM_N_REPEATS = 10
VALID_CV_SCOPES = {"group", "subject"}


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

    def _validate_inputs(
        self,
        subjects: List[str],
        task: Optional[str],
        cv_scope: str,
    ) -> str:
        """Validate inputs and return resolved task."""
        if not subjects:
            raise ValueError("No subjects specified")
        
        resolved_task = task or self.config.get("project.task")
        if resolved_task is None:
            raise ValueError("Missing required config value: project.task")
        
        if cv_scope not in VALID_CV_SCOPES:
            raise ValueError(
                f"Invalid cv_scope: {cv_scope} "
                f"(expected one of: {', '.join(sorted(VALID_CV_SCOPES))})"
            )
        
        if cv_scope == "group":
            min_subjects = self.config.get("analysis.min_subjects_for_group", 2)
            if len(subjects) < min_subjects:
                raise ValueError(
                    f"ML pipeline requires at least {min_subjects} subjects "
                    f"for group scope, got {len(subjects)}"
                )
        
        return resolved_task

    def _extract_ml_parameters(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate ML parameters from kwargs."""
        from eeg_pipeline.cli.common import ProgressReporter
        
        return {
            "cv_scope": kwargs.get("cv_scope", "group"),
            "progress": kwargs.get("progress") or ProgressReporter(enabled=False),
            "n_perm": kwargs.get("n_perm", DEFAULT_N_PERM),
            "inner_splits": kwargs.get("inner_splits", DEFAULT_INNER_SPLITS),
            "outer_jobs": kwargs.get("outer_jobs", DEFAULT_OUTER_JOBS),
            "rng_seed": kwargs.get("rng_seed") or self.config.get("project.random_state", DEFAULT_RNG_SEED),
            "model": kwargs.get("model", DEFAULT_MODEL),
            "uncertainty_alpha": kwargs.get("uncertainty_alpha", DEFAULT_UNCERTAINTY_ALPHA),
            "perm_n_repeats": kwargs.get("perm_n_repeats", DEFAULT_PERM_N_REPEATS),
        }

    def _run_uncertainty(
        self,
        subjects: List[str],
        task: str,
        rng_seed: int,
        alpha: float = DEFAULT_UNCERTAINTY_ALPHA,
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
        n_repeats: int = DEFAULT_PERM_N_REPEATS,
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

    def _execute_regression(
        self,
        subjects: List[str],
        task: str,
        cv_scope: str,
        params: Dict[str, Any],
        progress: Any,
    ) -> Path:
        """Execute regression ML analysis."""
        progress.step("Regression ML", current=1, total=1)
        
        if cv_scope == "subject":
            return run_within_subject_regression_ml(
                subjects=subjects,
                task=task,
                deriv_root=self.deriv_root,
                config=self.config,
                n_perm=params["n_perm"],
                inner_splits=params["inner_splits"],
                outer_jobs=params["outer_jobs"],
                rng_seed=params["rng_seed"],
                results_root=self.results_root,
                logger=self.logger,
                model=params["model"],
            )
        else:
            return run_regression_ml(
                subjects=subjects,
                task=task,
                deriv_root=self.deriv_root,
                config=self.config,
                n_perm=params["n_perm"],
                inner_splits=params["inner_splits"],
                outer_jobs=params["outer_jobs"],
                rng_seed=params["rng_seed"],
                results_root=self.results_root,
                logger=self.logger,
                model=params["model"],
            )

    def _execute_timegen(
        self,
        subjects: List[str],
        task: str,
        cv_scope: str,
        params: Dict[str, Any],
        progress: Any,
    ) -> Optional[Path]:
        """Execute time-generalization analysis."""
        if cv_scope == "subject":
            self.logger.warning("Time-generalization requires group scope; skipping.")
            return None
        
        progress.step("Time generalization", current=1, total=1)
        run_time_generalization(
            subjects=subjects,
            task=task,
            deriv_root=self.deriv_root,
            config=self.config,
            n_perm=params["n_perm"],
            rng_seed=params["rng_seed"],
            results_root=self.results_root,
            logger=self.logger,
        )
        return self.results_root / "time_generalization"

    def _execute_classify(
        self,
        subjects: List[str],
        task: str,
        cv_scope: str,
        params: Dict[str, Any],
        progress: Any,
    ) -> Path:
        """Execute classification ML analysis."""
        progress.step("Classification ML", current=1, total=1)
        return run_classification_ml(
            subjects=subjects,
            task=task,
            deriv_root=self.deriv_root,
            config=self.config,
            n_perm=params["n_perm"],
            inner_splits=params["inner_splits"],
            outer_jobs=params["outer_jobs"],
            rng_seed=params["rng_seed"],
            results_root=self.results_root,
            logger=self.logger,
        )

    def _execute_model_comparison(
        self,
        subjects: List[str],
        task: str,
        cv_scope: str,
        params: Dict[str, Any],
        progress: Any,
    ) -> Path:
        """Execute model comparison analysis."""
        progress.step("Model Comparison", current=1, total=1)
        return run_model_comparison_ml(
            subjects=subjects,
            task=task,
            deriv_root=self.deriv_root,
            config=self.config,
            inner_splits=params["inner_splits"],
            rng_seed=params["rng_seed"],
            results_root=self.results_root,
            logger=self.logger,
        )

    def _execute_incremental_validity(
        self,
        subjects: List[str],
        task: str,
        cv_scope: str,
        params: Dict[str, Any],
        progress: Any,
    ) -> Path:
        """Execute incremental validity analysis."""
        progress.step("Incremental Validity", current=1, total=1)
        return run_incremental_validity_ml(
            subjects=subjects,
            task=task,
            deriv_root=self.deriv_root,
            config=self.config,
            n_perm=params["n_perm"],
            inner_splits=params["inner_splits"],
            rng_seed=params["rng_seed"],
            results_root=self.results_root,
            logger=self.logger,
        )

    def _execute_uncertainty(
        self,
        subjects: List[str],
        task: str,
        cv_scope: str,
        params: Dict[str, Any],
        progress: Any,
    ) -> Optional[Path]:
        """Execute uncertainty quantification."""
        progress.step("Uncertainty Quantification", current=1, total=1)
        return self._run_uncertainty(
            subjects=subjects,
            task=task,
            rng_seed=params["rng_seed"],
            alpha=params["uncertainty_alpha"],
        )

    def _execute_shap(
        self,
        subjects: List[str],
        task: str,
        cv_scope: str,
        params: Dict[str, Any],
        progress: Any,
    ) -> Optional[Path]:
        """Execute SHAP importance analysis."""
        progress.step("SHAP Importance", current=1, total=1)
        return self._run_shap(
            subjects=subjects,
            task=task,
            rng_seed=params["rng_seed"],
        )

    def _execute_permutation(
        self,
        subjects: List[str],
        task: str,
        cv_scope: str,
        params: Dict[str, Any],
        progress: Any,
    ) -> Optional[Path]:
        """Execute permutation importance analysis."""
        progress.step("Permutation Importance", current=1, total=1)
        return self._run_permutation_importance(
            subjects=subjects,
            task=task,
            rng_seed=params["rng_seed"],
            n_repeats=params["perm_n_repeats"],
        )

    def _get_mode_dispatcher(self) -> Dict[str, Callable]:
        """Return mode dispatch dictionary."""
        return {
            "regression": self._execute_regression,
            "timegen": self._execute_timegen,
            "classify": self._execute_classify,
            "model_comparison": self._execute_model_comparison,
            "incremental_validity": self._execute_incremental_validity,
            "uncertainty": self._execute_uncertainty,
            "shap": self._execute_shap,
            "permutation": self._execute_permutation,
        }

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
            mode: ML analysis mode
            **kwargs: Additional options (n_perm, inner_splits, etc.)
        
        Returns:
            List of result dicts with results_dir paths
        """
        params = self._extract_ml_parameters(kwargs)
        resolved_task = self._validate_inputs(subjects, task, params["cv_scope"])
        
        self.logger.info(
            f"Starting ML pipeline (mode={mode}, cv_scope={params['cv_scope']}, "
            f"model={params['model']}): {len(subjects)} subjects, task={resolved_task}, "
            f"n_perm={params['n_perm']}, inner_splits={params['inner_splits']}"
        )
        
        params["progress"].start("machine_learning", subjects)
        dispatcher = self._get_mode_dispatcher()
        
        if mode not in dispatcher:
            valid_modes = ", ".join(sorted(dispatcher.keys()))
            raise ValueError(f"Unknown mode: {mode} (expected one of: {valid_modes})")
        
        executor = dispatcher[mode]
        results_dir = executor(
            subjects=subjects,
            task=resolved_task,
            cv_scope=params["cv_scope"],
            params=params,
            progress=params["progress"],
        )
        
        results_dirs = [results_dir] if results_dir is not None else []
        
        self.logger.info(f"ML pipeline ({mode}) complete.")
        params["progress"].complete(success=True)
        
        return [
            {
                "subjects": subjects,
                "status": "success",
                "mode": mode,
                "results_dir": str(d),
            }
            for d in results_dirs
        ]

    def _visualize_regression(self, results_dir: Path) -> None:
        """Visualize regression results."""
        visualize_regression_from_disk(
            results_dir=results_dir,
            config=self.config,
            logger=self.logger,
        )

    def _visualize_timegen(self, results_dir: Path) -> None:
        """Visualize time-generalization results."""
        visualize_time_generalization_from_disk(
            results_dir=results_dir,
            config=self.config,
            logger=self.logger,
        )

    def _visualize_classify(self, results_dir: Path) -> None:
        """Visualize classification results."""
        visualize_classification_from_disk(
            results_dir=results_dir,
            config=self.config,
            logger=self.logger,
        )

    def _get_visualization_dispatcher(self) -> Dict[str, Callable[[Path], None]]:
        """Return visualization dispatch dictionary."""
        return {
            "regression": self._visualize_regression,
            "timegen": self._visualize_timegen,
            "classify": self._visualize_classify,
        }

    def visualize(
        self,
        results_dir: Path,
        mode: MLMode = "regression",
    ) -> None:
        """Visualize ML results from disk.
        
        Single responsibility: Read results contract and create plots.
        """
        dispatcher = self._get_visualization_dispatcher()
        
        if mode not in dispatcher:
            valid_modes = ", ".join(sorted(dispatcher.keys()))
            raise ValueError(
                f"Visualization not supported for mode: {mode} "
                f"(expected one of: {valid_modes})"
            )
        
        dispatcher[mode](results_dir)

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
