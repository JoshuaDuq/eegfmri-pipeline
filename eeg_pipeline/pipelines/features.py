"""
Feature Extraction Pipeline (Canonical)
========================================

Single source of truth for feature extraction orchestration.
This module consolidates all feature extraction entry points:
- FeaturePipeline: PipelineBase subclass for batch processing
- extract_all_features: TFR-based feature extraction
- extract_precomputed_features: Precomputed-based feature extraction
- extract_fmri_prediction_features: fMRI-optimized subset

The pipeline class selects TFR vs precomputed mode internally based on config.

Usage:
    # Single subject
    pipeline = FeaturePipeline(config=config)
    pipeline.process_subject("0001", "thermalactive")

    # Multiple subjects
    pipeline.run_batch(["0001", "0002"])

    # Direct function calls
    from eeg_pipeline.pipelines.features import extract_all_features
    result = extract_all_features(ctx)
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import mne

from eeg_pipeline.context.features import FeatureContext
from eeg_pipeline.pipelines.base import PipelineBase
from eeg_pipeline.types import PrecomputedData
from eeg_pipeline.infra.paths import deriv_features_path, deriv_plots_path, ensure_dir, _load_events_df
from eeg_pipeline.plotting.io.figures import setup_matplotlib
from eeg_pipeline.utils.data.columns import pick_target_column
from eeg_pipeline.utils.data.epochs import load_epochs_for_analysis
from eeg_pipeline.utils.data.features import align_feature_dataframes
from eeg_pipeline.utils.data.feature_io import (
    compute_group_microstate_templates,
    export_fmri_regressors,
    save_all_features,
    save_dropped_trials_log,
    save_microstate_templates,
    save_trial_alignment_manifest,
)
from eeg_pipeline.utils.config.loader import get_frequency_band_names

from eeg_pipeline.analysis.features.api import (
    extract_all_features as _extract_all_features,
    extract_precomputed_features as _extract_precomputed_features,
    extract_fmri_prediction_features as _extract_fmri_prediction_features,
)
from eeg_pipeline.analysis.features.selection import resolve_feature_categories
from eeg_pipeline.analysis.features.results import (
    FeatureSet,
    ExtractionResult,
    FeatureExtractionResult,
)


###################################################################
# TFR-Based Feature Extraction
###################################################################


def _resolve_feature_categories(config: Any, requested: Optional[List[str]]) -> List[str]:
    return resolve_feature_categories(config, requested)


def extract_all_features(
    ctx: FeatureContext,
    *,
    precomputed_groups_override: Optional[List[str]] = None,
) -> FeatureExtractionResult:
    return _extract_all_features(ctx, precomputed_groups_override=precomputed_groups_override)


###################################################################
# Precomputed-Based Feature Extraction
###################################################################


def extract_precomputed_features(
    epochs: "mne.Epochs",
    bands: List[str],
    config: Any,
    logger: Any,
    *,
    feature_groups: Optional[List[str]] = None,
    events_df: Optional[pd.DataFrame] = None,
    n_plateau_windows: int = 5,
    precomputed: Optional[PrecomputedData] = None,
) -> ExtractionResult:
    return _extract_precomputed_features(
        epochs,
        bands,
        config,
        logger,
        feature_groups=feature_groups,
        events_df=events_df,
        n_plateau_windows=n_plateau_windows,
        precomputed=precomputed,
    )


def extract_fmri_prediction_features(
    epochs: "mne.Epochs",
    config: Any,
    logger: Any,
    events_df: Optional[pd.DataFrame] = None,
) -> ExtractionResult:
    return _extract_fmri_prediction_features(epochs, config, logger, events_df=events_df)


###################################################################
# Pipeline Class
###################################################################


class FeaturePipeline(PipelineBase):
    """Pipeline for EEG feature extraction."""
    
    def __init__(self, config: Optional[Any] = None):
        super().__init__(
            name="feature_extraction",
            config=config,
        )

    def run_batch(self, subjects: List[str], task: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        ledger = super().run_batch(subjects, task=task, **kwargs)
        
        if self.config.get("feature_engineering.microstates.build_group_templates", False):
            n_microstates = int(self.config.get("feature_engineering.microstates.n_states", 4))
            self.logger.info("Building group microstate templates from %d subjects...", len(subjects))
            group_templates, _ = compute_group_microstate_templates(
                self.deriv_root, n_microstates, self.logger
            )
            if group_templates is not None:
                self.logger.info("Group microstate templates saved successfully")
        
        return ledger

    def process_subject(self, subject: str, task: Optional[str] = None, **kwargs) -> None:
        task = task or self.config.get("project.task")
        if task is None:
            raise ValueError("Missing required config value: project.task")
        fixed_templates_path = kwargs.get("fixed_templates_path")
        feature_categories = _resolve_feature_categories(self.config, kwargs.get("feature_categories"))
        precomputed_groups = kwargs.get("precomputed_groups")
        
        self.logger.info(f"=== Feature extraction: sub-{subject}, task-{task} ===")
        self.logger.info("Feature categories: %s", ", ".join(feature_categories))
        if precomputed_groups:
            self.logger.info("Precomputed groups override: %s", ", ".join(precomputed_groups))
        
        features_dir = deriv_features_path(self.deriv_root, subject)
        ensure_dir(features_dir)
        
        setup_matplotlib(self.config)

        epochs, aligned_events = load_epochs_for_analysis(
            subject, task, align="strict", preload=False,
            deriv_root=self.deriv_root, logger=self.logger, config=self.config,
        )

        if epochs is None:
            self.logger.error(f"No cleaned epochs for sub-{subject}; skipping")
            return

        if aligned_events is None:
            self.logger.warning("No events available; skipping")
            return

        original_events = _load_events_df(subject, task, bids_root=self.config.bids_root, config=self.config)
        if original_events is not None:
            save_dropped_trials_log(epochs, original_events, features_dir / "dropped_trials.tsv", self.logger)

        save_trial_alignment_manifest(aligned_events, epochs, features_dir / "trial_alignment.json", self.config, self.logger)

        target_columns = list(self.config.get("event_columns.rating", []) or [])
        target_col = pick_target_column(aligned_events, target_columns=target_columns)
        if target_col is None:
            self.logger.warning("No target column found; skipping")
            return

        y = pd.to_numeric(aligned_events[target_col], errors="coerce")

        fixed_templates = None
        fixed_template_ch_names = None
        if fixed_templates_path and fixed_templates_path.exists():
            data = np.load(fixed_templates_path)
            fixed_templates = data["templates"]
            fixed_template_ch_names = data.get("ch_names")
            self.logger.info(f"Loaded fixed templates from {fixed_templates_path}")

        ctx = FeatureContext(
            subject=subject,
            task=task,
            config=self.config,
            deriv_root=self.deriv_root,
            logger=self.logger,
            epochs=epochs,
            aligned_events=aligned_events,
            fixed_templates=fixed_templates,
            fixed_template_ch_names=fixed_template_ch_names,
            feature_categories=feature_categories,
        )

        features = extract_all_features(ctx, precomputed_groups_override=precomputed_groups)

        tfr = features.tfr
        pow_df = features.pow_df
        pow_cols = features.pow_cols
        baseline_df = features.baseline_df
        baseline_cols = features.baseline_cols
        conn_df = features.conn_df
        conn_cols = features.conn_cols
        ms_df = features.ms_df
        ms_cols = features.ms_cols
        ms_templates = features.ms_templates
        aper_df = features.aper_df
        aper_cols = features.aper_cols
        itpc_df = features.phase_df
        itpc_cols = features.phase_cols
        itpc_trial_df = features.itpc_trial_df
        itpc_trial_cols = features.itpc_trial_cols
        pac_df = features.pac_df
        pac_trials_df = features.pac_trials_df
        pac_time_df = features.pac_time_df
        precomputed_df = features.precomputed_df
        precomputed_cols = features.precomputed_cols
        
        comp_df = features.comp_df
        comp_cols = features.comp_cols
        dynamics_df = features.dynamics_df
        dynamics_cols = features.dynamics_cols
        cfc_df = features.cfc_df
        cfc_cols = features.cfc_cols

        if itpc_trial_df is not None and not itpc_trial_df.empty:
            pow_df = pd.concat([pow_df, itpc_trial_df], axis=1)
            pow_cols.extend(itpc_trial_cols)

        power_bands = get_frequency_band_names(self.config)
        n_microstates = int(self.config.get("feature_engineering.microstates.n_states", 4))
        save_microstate_templates(epochs, ms_templates, subject, n_microstates, self.deriv_root, self.logger)

        self.logger.info("Aligning features...")

        critical_features = ["target"]
        if "power" in ctx.feature_categories:
            critical_features.append("power")
            critical_features.append("baseline")

        extra_blocks = {
            "itpc": itpc_df,
            "itpc_trial": itpc_trial_df,
            "pac": pac_df,
            "pac_trials": pac_trials_df,
            "pac_time": pac_time_df,
            "precomputed": precomputed_df,
            "complexity": comp_df,
            "dynamics": dynamics_df,
            "cfc": cfc_df,
        }
        extra_blocks = {k: v for k, v in extra_blocks.items() if v is not None and not getattr(v, "empty", False)}

        (
            pow_df_aligned, baseline_df_aligned, conn_df_aligned,
            ms_df_aligned, aper_df_aligned, y_aligned, retention_stats
        ) = align_feature_dataframes(
            pow_df, baseline_df, conn_df, ms_df, aper_df, y, aligned_events, features_dir, self.logger, self.config,
            critical_features=critical_features,
            extra_blocks=extra_blocks,
        )

        if retention_stats is None:
            self.logger.error("Feature alignment failed. Skipping save.")
            return

        extra_aligned = retention_stats.get("extra_aligned", {})
        itpc_df = extra_aligned.get("itpc", itpc_df)
        itpc_trial_df = extra_aligned.get("itpc_trial", itpc_trial_df)
        pac_df = extra_aligned.get("pac", pac_df)
        pac_trials_df = extra_aligned.get("pac_trials", pac_trials_df)
        pac_time_df = extra_aligned.get("pac_time", pac_time_df)
        precomputed_df = extra_aligned.get("precomputed", precomputed_df)
        comp_df = extra_aligned.get("complexity", comp_df)
        dynamics_df = extra_aligned.get("dynamics", dynamics_df)
        cfc_df = extra_aligned.get("cfc", cfc_df)

        feature_qc: Dict[str, Any] = {}
        if features.aper_qc is not None:
            feature_qc["aperiodic"] = features.aper_qc
        if ctx.precomputed is not None and getattr(ctx.precomputed, "qc", None) is not None:
            try:
                feature_qc["precomputed_intermediates"] = asdict(ctx.precomputed.qc)
            except TypeError:
                # If qc is not a dataclass, use it directly
                feature_qc["precomputed_intermediates"] = getattr(ctx.precomputed, "qc", None)

        combined_df = save_all_features(
            pow_df_aligned, pow_cols, baseline_df_aligned, baseline_cols,
            conn_df_aligned, conn_cols, ms_df_aligned, ms_cols,
            aper_df_aligned, aper_cols, itpc_df, itpc_cols,
            pac_df, pac_trials_df, pac_time_df,
            features.aper_qc, None, None, y_aligned, features_dir, self.logger, self.config,
            comp_df=comp_df, comp_cols=comp_cols,
            dynamics_df=dynamics_df, dynamics_cols=dynamics_cols,
            cfc_df=cfc_df, cfc_cols=cfc_cols,
            precomputed_df=precomputed_df, precomputed_cols=precomputed_cols,
            feature_qc=feature_qc or None,
        )

        regressor_df = export_fmri_regressors(
            aligned_events, pow_df_aligned, pow_cols, ms_df_aligned,
            features.pac_trials_df, aper_df_aligned, y_aligned,
            power_bands, subject, task, features_dir, self.config, self.logger,
        )
        if regressor_df is not None:
            plots_dir = deriv_plots_path(self.deriv_root, subject, subdir="behavior")
            self.logger.warning("plot_regressor_distributions not yet implemented, skipping")

        n_trials = len(y_aligned)
        n_pow = pow_df_aligned.shape[1] if pow_df_aligned is not None else 0
        n_conn = conn_df_aligned.shape[1] if conn_df_aligned is not None and not conn_df_aligned.empty else 0
        n_ms = ms_df_aligned.shape[1] if ms_df_aligned is not None and not ms_df_aligned.empty else 0
        n_aper = aper_df_aligned.shape[1] if aper_df_aligned is not None and not aper_df_aligned.empty else 0
        n_precomp = precomputed_df.shape[1] if precomputed_df is not None and not precomputed_df.empty else 0
        if combined_df is None:
             n_total = 0
        else:
             n_total = combined_df.shape[1]

        self.logger.info(
            f"Done: sub-{subject}, trials={n_trials}, power={n_pow}, conn={n_conn}, "
            f"ms={n_ms}, aper={n_aper}, precomp={n_precomp}, total={n_total}"
        )


def process_subject(subject: str, task: Optional[str] = None, config: Optional[Any] = None, **kwargs: Any) -> None:
    pipeline = FeaturePipeline(config=config)
    pipeline.process_subject(subject, task=task, **kwargs)


def extract_features_for_subjects(
    subjects: List[str],
    task: Optional[str] = None,
    config: Optional[Any] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    pipeline = FeaturePipeline(config=config)
    return pipeline.run_batch(subjects, task=task, **kwargs)


__all__ = [
    "FeaturePipeline",
    "process_subject",
    "extract_features_for_subjects",
    "extract_all_features",
    "extract_precomputed_features",
    "extract_fmri_prediction_features",
    "FeatureExtractionResult",
    "ExtractionResult",
    "FeatureSet",
]
