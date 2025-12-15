"""Decoding Pipeline (Canonical)
=============================

Thin pipeline wrapper that delegates decoding orchestration to
eeg_pipeline.analysis.decoding.orchestration.

Usage:
    pipeline = DecodingPipeline(config=config)
    pipeline.run_batch(["0001", "0002"])
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from eeg_pipeline.analysis.decoding.orchestration import (
    run_regression_decoding,
    run_time_generalization,
)
from eeg_pipeline.pipelines.base import PipelineBase
from eeg_pipeline.pipelines.viz.decoding import (
    visualize_regression_from_disk,
    visualize_time_generalization_from_disk,
)


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
        
        results_dir = run_regression_decoding(
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
        
        visualize_regression_from_disk(
            results_dir=results_dir,
            config=self.config,
            logger=self.logger,
        )
        
        if not skip_time_gen:
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
            
            tg_results_dir = self.results_root / "time_generalization"
            visualize_time_generalization_from_disk(
                results_dir=tg_results_dir,
                config=self.config,
                logger=self.logger,
            )
        
        self.logger.info("Decoding complete.")
        
        return [{"subjects": subjects, "status": "success", "results_dir": str(results_dir)}]


__all__ = [
    "DecodingPipeline",
]
