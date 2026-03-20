"""Machine Learning Pipeline (Canonical)
======================================

Thin pipeline wrapper that delegates ML orchestration to
eeg_pipeline.analysis.machine_learning.orchestration.

Compute and visualization are separated (SRP):
- run_batch(): compute only, writes stable results contract

ML inputs
---------
By default, ML loads per-trial feature tables saved by the feature pipeline
(derivatives/*/eeg/features/*/features_*.parquet) and aligns them to the *clean*
events.tsv for targets/covariates. Configure via:
- config: machine_learning.data.*, machine_learning.targets.*
- CLI: --feature-families, --target, --binary-threshold, --feature-harmonization

Notes on target validity
------------------------
- Regression modes assume a continuous target. If the selected target is binary-like (e.g., {0,1}),
  the pipeline logs a warning (or errors if `machine_learning.targets.strict_regression_target_continuous=true`).
- For binary outcomes, prefer `mode="classify"`.

Modes:
- regression: LOSO regression predicting a continuous target
- timegen: Time-generalization analysis
- classify: Binary classification
- model_comparison: Compare multiple models (ElasticNet/Ridge/RF)
- incremental_validity: Δ performance from EEG over baseline
- uncertainty: Conformal prediction intervals
- shap: SHAP feature importance
- permutation: Permutation feature importance

Usage:
    pipeline = MLPipeline(config=config)
    
    # Compute only (SRP)
    pipeline.run_batch(["0001", "0002"], mode="regression")
    
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from eeg_pipeline.utils.config.loader import require_config_value
from eeg_pipeline.analysis.machine_learning.orchestration import (
    run_regression_ml,
    run_within_subject_regression_ml,
    run_time_generalization,
    run_classification_ml,
    run_within_subject_classification_ml,
    run_model_comparison_ml,
    run_incremental_validity_ml,
)
from eeg_pipeline.pipelines.base import PipelineBase
from eeg_pipeline.pipelines.progress import ensure_progress_reporter


MLMode = Literal[
    "regression", "timegen", "classify",
    "model_comparison", "incremental_validity",
    "uncertainty", "shap", "permutation",
]


DEFAULT_N_PERM = 0
DEFAULT_OUTER_JOBS = 1
DEFAULT_RNG_SEED = 42
DEFAULT_MODEL = "elasticnet"
DEFAULT_UNCERTAINTY_ALPHA = 0.1
DEFAULT_PERM_N_REPEATS = 5
VALID_CV_SCOPES = {"group", "subject"}


class MLPipeline(PipelineBase):
    """Pipeline for ML-based EEG analysis.
    
    Unlike other pipelines, ML requires multiple subjects for LOSO CV.
    The process_subject method is not used; instead use run_batch directly.
    
    Modes:
        - regression: LOSO regression predicting a continuous target
        - timegen: Time-generalization analysis
        - classify: Binary classification
        - model_comparison: Compare multiple models (ElasticNet/Ridge/RF)
        - incremental_validity: Δ performance from EEG over baseline
        - uncertainty: Conformal prediction intervals
        - shap: SHAP feature importance
        - permutation: Permutation feature importance
    """
    
    def __init__(self, config: Optional[Any] = None):
        super().__init__(name="machine_learning", config=config)
        self.results_root = self.deriv_root / "machine_learning"

    def process_subject(self, subject: str, task: Optional[str] = None, **kwargs) -> None:
        raise NotImplementedError(
            "MLPipeline requires multiple subjects for LOSO CV. "
            "Use run_batch() instead."
        )

    def visualize(self, _results_dir: Path) -> None:
        """Visualization is not implemented in this pipeline wrapper."""
        raise NotImplementedError(
            "Visualization is not implemented in MLPipeline. "
            "Use analysis outputs directly or plotting commands."
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
            min_subjects = int(
                require_config_value(self.config, "analysis.min_subjects_for_group")
            )
            if len(subjects) < min_subjects:
                raise ValueError(
                    f"ML pipeline requires at least {min_subjects} subjects "
                    f"for group scope, got {len(subjects)}"
                )
        
        return resolved_task

    def _extract_ml_parameters(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate ML parameters from kwargs."""
        return {
            "cv_scope": kwargs.get("cv_scope", "group"),
            "progress": ensure_progress_reporter(kwargs.get("progress")),
            "n_perm": kwargs.get("n_perm", DEFAULT_N_PERM),
            "inner_splits": kwargs.get("inner_splits", self.config.get("machine_learning.cv.inner_splits", 5)),
            "outer_jobs": kwargs.get("outer_jobs", DEFAULT_OUTER_JOBS),
            "rng_seed": kwargs.get("rng_seed") or self.config.get("project.random_state", DEFAULT_RNG_SEED),
            "model": kwargs.get("model", self.config.get("machine_learning.models.regression_default", DEFAULT_MODEL)),
            "uncertainty_alpha": kwargs.get("uncertainty_alpha", self.config.get("machine_learning.analysis.uncertainty.alpha", DEFAULT_UNCERTAINTY_ALPHA)),
            "perm_n_repeats": kwargs.get("perm_n_repeats", self.config.get("machine_learning.analysis.permutation_importance.n_repeats", DEFAULT_PERM_N_REPEATS)),
            "classification_model": kwargs.get("classification_model"),
            # Data/target controls (kept out of core config for CLI override friendliness)
            "feature_families": kwargs.get("feature_families"),
            "feature_bands": kwargs.get("feature_bands"),
            "feature_segments": kwargs.get("feature_segments"),
            "feature_scopes": kwargs.get("feature_scopes"),
            "feature_stats": kwargs.get("feature_stats"),
            "feature_harmonization": kwargs.get("feature_harmonization"),
            "target": kwargs.get("target"),
            "binary_threshold": kwargs.get("binary_threshold"),
            "baseline_predictors": kwargs.get("baseline_predictors"),
            "covariates": kwargs.get("covariates"),
        }

    def _run_uncertainty(
        self,
        subjects: List[str],
        task: str,
        rng_seed: int,
        alpha: float = DEFAULT_UNCERTAINTY_ALPHA,
        *,
        target: Optional[str] = None,
        feature_families: Optional[List[str]] = None,
        feature_harmonization: Optional[str] = None,
        covariates: Optional[List[str]] = None,
        feature_bands: Optional[List[str]] = None,
        feature_segments: Optional[List[str]] = None,
        feature_scopes: Optional[List[str]] = None,
        feature_stats: Optional[List[str]] = None,
        model: Optional[str] = None,
    ) -> Optional[Path]:
        """Run uncertainty quantification via conformal prediction."""
        from eeg_pipeline.analysis.machine_learning.orchestration import _run_uncertainty_stage
        from eeg_pipeline.utils.data.machine_learning import load_active_matrix
        
        X, y, groups, _, _ = load_active_matrix(
            subjects,
            task,
            self.deriv_root,
            self.config,
            self.logger,
            feature_families=feature_families,
            feature_harmonization=feature_harmonization,  # type: ignore[arg-type]
            target=target or self.config.get("machine_learning.targets.regression", None),
            target_kind="continuous",
            covariates=covariates,
            feature_bands=feature_bands,
            feature_segments=feature_segments,
            feature_scopes=feature_scopes,
            feature_stats=feature_stats,
        )
        
        results_dir = self.results_root / "uncertainty"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        return _run_uncertainty_stage(
            X=X, y=y, groups=groups,
            config=self.config, seed=rng_seed, alpha=alpha,
            results_dir=results_dir, logger=self.logger,
            model_name=model or DEFAULT_MODEL,
        )

    def _run_shap(
        self,
        subjects: List[str],
        task: str,
        rng_seed: int,
        *,
        target: Optional[str] = None,
        feature_families: Optional[List[str]] = None,
        feature_harmonization: Optional[str] = None,
        covariates: Optional[List[str]] = None,
        feature_bands: Optional[List[str]] = None,
        feature_segments: Optional[List[str]] = None,
        feature_scopes: Optional[List[str]] = None,
        feature_stats: Optional[List[str]] = None,
        model: Optional[str] = None,
    ) -> Optional[Path]:
        """Run SHAP-based feature importance."""
        from eeg_pipeline.analysis.machine_learning.orchestration import _run_shap_importance_stage
        from eeg_pipeline.utils.data.machine_learning import load_active_matrix
        
        X, y, groups, feature_names, _ = load_active_matrix(
            subjects,
            task,
            self.deriv_root,
            self.config,
            self.logger,
            feature_families=feature_families,
            feature_harmonization=feature_harmonization,  # type: ignore[arg-type]
            target=target or self.config.get("machine_learning.targets.regression", None),
            target_kind="continuous",
            covariates=covariates,
            feature_bands=feature_bands,
            feature_segments=feature_segments,
            feature_scopes=feature_scopes,
            feature_stats=feature_stats,
        )
        
        results_dir = self.results_root / "shap"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        return _run_shap_importance_stage(
            X=X, y=y, groups=groups, feature_names=feature_names,
            config=self.config, seed=rng_seed,
            results_dir=results_dir, logger=self.logger,
            model_name=model or DEFAULT_MODEL,
        )

    def _run_permutation_importance(
        self,
        subjects: List[str],
        task: str,
        rng_seed: int,
        n_repeats: int = DEFAULT_PERM_N_REPEATS,
        *,
        target: Optional[str] = None,
        feature_families: Optional[List[str]] = None,
        feature_harmonization: Optional[str] = None,
        covariates: Optional[List[str]] = None,
        feature_bands: Optional[List[str]] = None,
        feature_segments: Optional[List[str]] = None,
        feature_scopes: Optional[List[str]] = None,
        feature_stats: Optional[List[str]] = None,
        model: Optional[str] = None,
    ) -> Optional[Path]:
        """Run permutation-based feature importance."""
        from eeg_pipeline.analysis.machine_learning.orchestration import _run_permutation_importance_stage
        from eeg_pipeline.utils.data.machine_learning import load_active_matrix
        
        X, y, groups, feature_names, _ = load_active_matrix(
            subjects,
            task,
            self.deriv_root,
            self.config,
            self.logger,
            feature_families=feature_families,
            feature_harmonization=feature_harmonization,  # type: ignore[arg-type]
            target=target or self.config.get("machine_learning.targets.regression", None),
            target_kind="continuous",
            covariates=covariates,
            feature_bands=feature_bands,
            feature_segments=feature_segments,
            feature_scopes=feature_scopes,
            feature_stats=feature_stats,
        )
        
        results_dir = self.results_root / "permutation_importance"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        return _run_permutation_importance_stage(
            X=X, y=y, groups=groups, feature_names=feature_names,
            config=self.config, seed=rng_seed, n_repeats=n_repeats,
            results_dir=results_dir, logger=self.logger,
            model_name=model or DEFAULT_MODEL,
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
                target=params.get("target"),
                feature_families=params.get("feature_families"),
                feature_bands=params.get("feature_bands"),
                feature_segments=params.get("feature_segments"),
                feature_scopes=params.get("feature_scopes"),
                feature_stats=params.get("feature_stats"),
                feature_harmonization=params.get("feature_harmonization"),
                covariates=params.get("covariates"),
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
                target=params.get("target"),
                feature_families=params.get("feature_families"),
                feature_bands=params.get("feature_bands"),
                feature_segments=params.get("feature_segments"),
                feature_scopes=params.get("feature_scopes"),
                feature_stats=params.get("feature_stats"),
                feature_harmonization=params.get("feature_harmonization"),
                covariates=params.get("covariates"),
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
            raise ValueError(
                "Time-generalization uses leave-one-subject-out (group-level) CV and "
                "does not support --cv-scope subject. Use --cv-scope group."
            )

        progress.step("Time generalization", current=1, total=1)
        results_dir = run_time_generalization(
            subjects=subjects,
            task=task,
            deriv_root=self.deriv_root,
            config=self.config,
            n_perm=params["n_perm"],
            rng_seed=params["rng_seed"],
            results_root=self.results_root,
            logger=self.logger,
            target=params.get("target"),
        )
        return results_dir

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
        if cv_scope == "subject":
            return run_within_subject_classification_ml(
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
                classification_model=params.get("classification_model"),
                target=params.get("target"),
                binary_threshold=params.get("binary_threshold"),
                feature_families=params.get("feature_families"),
                feature_bands=params.get("feature_bands"),
                feature_segments=params.get("feature_segments"),
                feature_scopes=params.get("feature_scopes"),
                feature_stats=params.get("feature_stats"),
                feature_harmonization=params.get("feature_harmonization"),
                covariates=params.get("covariates"),
            )
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
            classification_model=params.get("classification_model"),
            target=params.get("target"),
            binary_threshold=params.get("binary_threshold"),
            feature_families=params.get("feature_families"),
            feature_bands=params.get("feature_bands"),
            feature_segments=params.get("feature_segments"),
            feature_scopes=params.get("feature_scopes"),
            feature_stats=params.get("feature_stats"),
            feature_harmonization=params.get("feature_harmonization"),
            covariates=params.get("covariates"),
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
            n_perm=params["n_perm"],
            inner_splits=params["inner_splits"],
            outer_jobs=params["outer_jobs"],
            rng_seed=params["rng_seed"],
            results_root=self.results_root,
            logger=self.logger,
            target=params.get("target"),
            feature_families=params.get("feature_families"),
            feature_bands=params.get("feature_bands"),
            feature_segments=params.get("feature_segments"),
            feature_scopes=params.get("feature_scopes"),
            feature_stats=params.get("feature_stats"),
            feature_harmonization=params.get("feature_harmonization"),
            covariates=params.get("covariates"),
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
            target=params.get("target"),
            baseline_predictors=params.get("baseline_predictors"),
            feature_families=params.get("feature_families"),
            feature_bands=params.get("feature_bands"),
            feature_segments=params.get("feature_segments"),
            feature_scopes=params.get("feature_scopes"),
            feature_stats=params.get("feature_stats"),
            feature_harmonization=params.get("feature_harmonization"),
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
            target=params.get("target"),
            feature_families=params.get("feature_families"),
            feature_bands=params.get("feature_bands"),
            feature_segments=params.get("feature_segments"),
            feature_scopes=params.get("feature_scopes"),
            feature_stats=params.get("feature_stats"),
            feature_harmonization=params.get("feature_harmonization"),
            covariates=params.get("covariates"),
            model=params.get("model"),
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
            target=params.get("target"),
            feature_families=params.get("feature_families"),
            feature_bands=params.get("feature_bands"),
            feature_segments=params.get("feature_segments"),
            feature_scopes=params.get("feature_scopes"),
            feature_stats=params.get("feature_stats"),
            feature_harmonization=params.get("feature_harmonization"),
            covariates=params.get("covariates"),
            model=params.get("model"),
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
            target=params.get("target"),
            feature_families=params.get("feature_families"),
            feature_bands=params.get("feature_bands"),
            feature_segments=params.get("feature_segments"),
            feature_scopes=params.get("feature_scopes"),
            feature_stats=params.get("feature_stats"),
            feature_harmonization=params.get("feature_harmonization"),
            covariates=params.get("covariates"),
            model=params.get("model"),
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
        run_context = self._create_run_metadata_context(
            subjects=subjects,
            task=resolved_task,
            kwargs={**kwargs, "mode": mode},
        )
        run_status = "failed"
        run_error: Optional[str] = None
        results_dir: Optional[Path] = None

        import time as _time

        try:
            self.logger.info(
                "=== ML pipeline: mode=%s, cv_scope=%s, model=%s ===",
                mode, params["cv_scope"], params["model"],
            )
            self.logger.info(
                "Subjects: %d, task: %s, permutations: %d, inner_splits: %d",
                len(subjects), resolved_task, params["n_perm"], params["inner_splits"],
            )

            params["progress"].start("machine_learning", subjects)
            dispatcher = self._get_mode_dispatcher()

            if mode not in dispatcher:
                valid_modes = ", ".join(sorted(dispatcher.keys()))
                raise ValueError(f"Unknown mode: {mode} (expected one of: {valid_modes})")

            t0 = _time.perf_counter()
            executor = dispatcher[mode]
            results_dir = executor(
                subjects=subjects,
                task=resolved_task,
                cv_scope=params["cv_scope"],
                params=params,
                progress=params["progress"],
            )
            if results_dir is None:
                params["progress"].complete(success=False)
                raise RuntimeError(
                    f"ML pipeline ({mode}) produced no output. "
                    "Treating this as a failed run to avoid false-success reporting."
                )
            elapsed = _time.perf_counter() - t0

            results_dirs = [results_dir]

            self.logger.info(
                "ML pipeline (%s) complete: %s (%.1fs)",
                mode, results_dir or "no output", elapsed,
            )
            params["progress"].complete(success=True)
            run_status = "success"

            return [
                {
                    "subjects": subjects,
                    "status": "success",
                    "mode": mode,
                    "results_dir": str(d),
                }
                for d in results_dirs
            ]
        except Exception as exc:
            run_error = str(exc)
            raise
        finally:
            self._write_run_metadata(
                run_context,
                status=run_status,
                error=run_error,
                outputs={
                    "results_dir": str(results_dir) if results_dir is not None else None,
                },
                summary={
                    "n_subjects": len(subjects),
                    "mode": mode,
                    "cv_scope": params.get("cv_scope"),
                },
            )



__all__ = [
    "MLPipeline",
]
