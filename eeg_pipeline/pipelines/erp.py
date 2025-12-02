"""
ERP Analysis Pipeline.

Extracts event-related potential statistics from preprocessed epochs.

Usage:
    # Single subject via pipeline class
    pipeline = ErpPipeline(config=config)
    pipeline.process_subject("0001", task="thermalactive")

    # Multiple subjects
    pipeline.run_batch(["0001", "0002"])

    # Or use module-level functions
    extract_erp_stats("0001", "thermalactive", config=config)
    extract_erp_stats_for_subjects(["0001", "0002"])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from eeg_pipeline.pipelines.base import PipelineBase


###################################################################
# Configuration
###################################################################


def get_erp_config(config=None) -> dict:
    """Return ERP configuration with defaults."""
    from eeg_pipeline.utils.config.loader import load_settings

    if config is None:
        config = load_settings()

    erp = config.get("erp_analysis.erp", {})

    return {
        "baseline_window": tuple(erp.get("baseline_window", [-0.2, 0.0])),
        "picks": erp.get("picks", "eeg"),
        "pain_color": erp.get("pain_color", "crimson"),
        "nonpain_color": erp.get("nonpain_color", "navy"),
        "combine": erp.get("combine", "gfp"),
        "include_tmax_in_crop": bool(erp.get("include_tmax_in_crop", False)),
        "default_crop_tmin": erp.get("default_crop_tmin"),
        "default_crop_tmax": erp.get("default_crop_tmax"),
        "output_files": erp.get(
            "output_files",
            {
                "pain_gfp": "erp_pain_binary_gfp.png",
                "pain_butterfly": "erp_pain_binary_butterfly.png",
                "temp_gfp": "erp_by_temperature_gfp.png",
                "temp_butterfly": "erp_by_temperature_butterfly.png",
            },
        ),
    }


###################################################################
# Data Loading
###################################################################


def load_and_prepare_epochs(
    subject: str,
    task: str,
    config,
    crop_tmin: Optional[float],
    crop_tmax: Optional[float],
    include_tmax_in_crop: bool,
    logger: logging.Logger,
):
    """Load epochs, attach metadata, and crop if requested."""
    from eeg_pipeline.utils.data.loading import load_epochs_for_analysis, crop_epochs

    epochs, aligned_events = load_epochs_for_analysis(
        subject, task,
        align="strict",
        preload=False,
        deriv_root=config.deriv_root,
        bids_root=config.bids_root,
        config=config,
        logger=logger,
    )

    if epochs is None or aligned_events is None:
        return None

    epochs.metadata = aligned_events

    if crop_tmin is not None or crop_tmax is not None:
        epochs = crop_epochs(epochs, crop_tmin, crop_tmax, include_tmax_in_crop, logger)

    return epochs


###################################################################
# Pipeline Class
###################################################################


class ErpPipeline(PipelineBase):
    """Pipeline for ERP analysis using shared PipelineBase pattern."""
    
    def __init__(self, config: Optional[Any] = None):
        super().__init__(
            name="erp_analysis",
            config=config,
        )
        self._erp_config = get_erp_config(self.config)
        self._crop_tmin: Optional[float] = None
        self._crop_tmax: Optional[float] = None
    
    def set_crop_params(
        self,
        crop_tmin: Optional[float] = None,
        crop_tmax: Optional[float] = None,
    ) -> "ErpPipeline":
        """Set crop parameters for epoch windowing."""
        self._crop_tmin = crop_tmin if crop_tmin is not None else self._erp_config.get("default_crop_tmin")
        self._crop_tmax = crop_tmax if crop_tmax is not None else self._erp_config.get("default_crop_tmax")
        return self

    def process_subject(self, subject: str, task: Optional[str] = None, **kwargs) -> None:
        """Process a single subject for ERP statistics extraction."""
        from eeg_pipeline.utils.io.general import (
            deriv_stats_path,
            ensure_dir,
            write_tsv,
            find_pain_column_in_metadata,
            find_temperature_column_in_metadata,
        )
        from eeg_pipeline.utils.analysis.stats import count_trials_by_condition
        
        task = task or self.config.get("project.task")
        if task is None:
            raise ValueError("Missing required config value: project.task")
        
        crop_tmin = kwargs.get("crop_tmin", self._crop_tmin)
        crop_tmax = kwargs.get("crop_tmax", self._crop_tmax)
        include_tmax = bool(self._erp_config.get("include_tmax_in_crop", False))
        
        self.logger.info(f"=== ERP analysis: sub-{subject}, task-{task} ===")
        
        epochs = load_and_prepare_epochs(
            subject, task, self.config, crop_tmin, crop_tmax, include_tmax, self.logger
        )
        
        if epochs is None:
            self.logger.error(f"Failed to load data for sub-{subject}, task-{task}")
            return
        
        self.logger.info(f"Loaded {len(epochs)} epochs")
        
        n_pain = 0
        n_nonpain = 0
        temperatures = []
        
        pain_col = find_pain_column_in_metadata(epochs, self.config)
        if pain_col:
            n_pain, n_nonpain = count_trials_by_condition(epochs, pain_col, logger=None)
        
        temp_col = find_temperature_column_in_metadata(epochs, self.config)
        if temp_col:
            temperatures = sorted(epochs.metadata[temp_col].unique().tolist())
        
        self.logger.info(f"ERP stats: pain={n_pain}, non-pain={n_nonpain}, temps={temperatures}")
        
        stats_dir = deriv_stats_path(self.deriv_root, subject)
        ensure_dir(stats_dir)
        counts_file_name = self.config.get("erp_analysis.erp.output_files.counts_file_name", "erp_trial_counts.tsv")
        stats_path = stats_dir / counts_file_name
        
        try:
            write_tsv(
                pd.DataFrame([{
                    "subject": subject,
                    "n_trials_pain": n_pain,
                    "n_trials_nonpain": n_nonpain,
                    "temperatures": ",".join(map(str, temperatures)),
                }]),
                stats_path,
            )
            self.logger.info(f"Saved ERP stats to {stats_path}")
        except Exception as e:
            self.logger.error(f"Failed to write ERP stats: {e}")


###################################################################
# Module-Level Entry Points (Backward Compatibility)
###################################################################


def extract_erp_stats(
    subject: str,
    task: str,
    crop_tmin: Optional[float] = None,
    crop_tmax: Optional[float] = None,
    include_tmax_in_crop: bool = False,
    config=None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Extract ERP statistics for a single subject."""
    from eeg_pipeline.utils.config.loader import load_settings
    
    if config is None:
        config = load_settings()
    
    pipeline = ErpPipeline(config=config)
    pipeline.set_crop_params(crop_tmin, crop_tmax)
    pipeline.process_subject(subject, task=task)


def extract_erp_stats_for_subjects(
    subjects: List[str],
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    config=None,
    crop_tmin: Optional[float] = None,
    crop_tmax: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Extract ERP statistics for multiple subjects."""
    from eeg_pipeline.utils.config.loader import load_settings

    if not subjects:
        raise ValueError("No subjects specified")

    if config is None:
        config = load_settings()

    pipeline = ErpPipeline(config=config)
    pipeline.set_crop_params(crop_tmin, crop_tmax)
    
    return pipeline.run_batch(subjects, task=task, crop_tmin=crop_tmin, crop_tmax=crop_tmax)


__all__ = [
    "ErpPipeline",
    "extract_erp_stats",
    "extract_erp_stats_for_subjects",
    "get_erp_config",
    "load_and_prepare_epochs",
]
