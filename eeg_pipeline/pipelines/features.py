"""
Feature Extraction Pipeline (Canonical)
========================================

Single source of truth for feature extraction orchestration.
This module consolidates all feature extraction entry points:
- FeaturePipeline: PipelineBase subclass for batch processing
- extract_all_features: TFR-based feature extraction
- extract_precomputed_features: Precomputed-based feature extraction
- extract_precomputed_features: Precomputed-based feature extraction

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
    save_all_features,
    save_dropped_trials_log,
    save_trial_alignment_manifest,
)
from eeg_pipeline.utils.config.loader import get_frequency_band_names

from eeg_pipeline.analysis.features.api import (
    extract_all_features as _extract_all_features,
    extract_precomputed_features as _extract_precomputed_features,
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
) -> FeatureExtractionResult:
    return _extract_all_features(ctx)


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
    precomputed: Optional[PrecomputedData] = None,
) -> ExtractionResult:
    return _extract_precomputed_features(
        epochs,
        bands,
        config,
        logger,
        feature_groups=feature_groups,
        events_df=events_df,
        precomputed=precomputed,
    )




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
        
        return ledger

    def process_subject(self, subject: str, task: Optional[str] = None, **kwargs) -> None:
        from eeg_pipeline.cli.common import ProgressReporter
        
        task = task or self.config.get("project.task")
        if task is None:
            raise ValueError("Missing required config value: project.task")
        fixed_templates_path = kwargs.get("fixed_templates_path")
        feature_categories = _resolve_feature_categories(self.config, kwargs.get("feature_categories"))
        
        # Initialize progress reporter
        progress = kwargs.get("progress") or ProgressReporter(enabled=False)
        
        self.logger.info(f"=== Feature extraction: sub-{subject}, task-{task} ===")
        self.logger.info("Feature categories: %s", ", ".join(feature_categories))
        
        progress.subject_start(f"sub-{subject}")
        
        features_dir = deriv_features_path(self.deriv_root, subject)
        ensure_dir(features_dir)
        
        setup_matplotlib(self.config)

        # Determine time ranges early to calculate total steps
        explicit_windows = kwargs.get("time_ranges")
        time_ranges_raw = list(explicit_windows) if explicit_windows else None
        if not time_ranges_raw:
            time_ranges_raw = [
                {"name": None, "tmin": kwargs.get("tmin"), "tmax": kwargs.get("tmax")}
            ]
        
        # Calculate total steps dynamically: load + 3 per time range (extract, align, save)
        n_ranges = len(time_ranges_raw)
        total_steps = 1 + (n_ranges * 3)
        current_step = 0

        current_step += 1
        progress.step("Loading epochs", current=current_step, total=total_steps)
        epochs, aligned_events = load_epochs_for_analysis(
            subject, task, align="strict", preload=True,
            deriv_root=self.deriv_root, logger=self.logger, config=self.config,
        )

        if epochs is None:
            self.logger.error(f"No cleaned epochs for sub-{subject}; skipping")
            progress.error("no_epochs", f"No cleaned epochs for sub-{subject}")
            return

        if aligned_events is None:
            self.logger.warning("No events available; skipping")
            progress.error("no_events", "No aligned events")
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

        # Time ranges already determined above for step calculation
        time_ranges = time_ranges_raw
        ranges_to_process = time_ranges

        # Optimization: Pre-compute TFR on full epochs if we have multiple ranges
        # and TFR is needed for the requested categories.
        tfr_full = None
        tfr_categories = {"power", "itpc", "pac"}
        if len(time_ranges) > 1 and any(cat in feature_categories for cat in tfr_categories):
            from eeg_pipeline.utils.analysis.tfr import compute_tfr_morlet
            self.logger.info("Pre-computing TFR on full epochs for multi-range extraction...")
            tfr_full = compute_tfr_morlet(epochs, self.config, logger=self.logger)

        # Optimization: Pre-compute shared intermediates on full epochs
        precomputed_full = None
        precompute_categories = {
            "connectivity", "erds", "ratios", "asymmetry", "pac", "complexity", "bursts"
        }
        if len(time_ranges) > 1 and any(cat in feature_categories for cat in precompute_categories):
            from eeg_pipeline.analysis.features.preparation import precompute_data
            from eeg_pipeline.utils.analysis.windowing import TimeWindowSpec
            self.logger.info("Pre-computing shared intermediates on full epochs for multi-range extraction...")
            # Use all bands requested or in config
            bands = kwargs.get("bands") or get_frequency_band_names(self.config)
            # Create a TimeWindowSpec using the user-specified time ranges, not config
            user_windows_spec = TimeWindowSpec(
                times=epochs.times,
                config=self.config,
                sampling_rate=float(epochs.info["sfreq"]),
                logger=self.logger,
                explicit_windows=time_ranges,  # Pass user's time ranges
            )
            precomputed_full = precompute_data(
                epochs,
                bands,
                self.config,
                self.logger,
                compute_psd_data=True,
                windows_spec=user_windows_spec,
            )

        for tr_spec in ranges_to_process:
            name = tr_spec.get("name")
            tmin = tr_spec.get("tmin")
            tmax = tr_spec.get("tmax")

            # Validate and swap if needed
            if tmin is not None and tmax is not None and tmin > tmax:
                self.logger.warning(
                    f"Time range '{name}' has tmin ({tmin}) > tmax ({tmax}). Swapping values."
                )
                tmin, tmax = tmax, tmin
            
            # Treat "full" as default (no suffix)
            suffix = name if name and name.lower() != "full" else None
            
            range_info = f"range '{name}'" if name else "default range"
            self.logger.info(f"--- Processing {range_info} ({tmin} to {tmax}s) ---")

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
                bands=kwargs.get("bands"),  # Runtime band override
                spatial_modes=kwargs.get("spatial_modes") or self.config.get("feature_engineering.spatial_modes", ["roi", "channels", "global"]),
                tmin=tmin,
                tmax=tmax,
                name=name,
                aggregation_method=kwargs.get("aggregation_method", "mean"),
                tfr=tfr_full,
                precomputed=precomputed_full,
                explicit_windows=explicit_windows,
            )
            
            # Pass progress reporter to context
            ctx.progress = progress
            ctx.total_steps = total_steps
            ctx.current_step = current_step

            current_step += 1
            progress.step(f"Extracting features ({name or 'full'})", current=current_step, total=total_steps)
            features = extract_all_features(ctx)

            pow_df = features.pow_df
            pow_cols = features.pow_cols
            baseline_df = features.baseline_df
            baseline_cols = features.baseline_cols
            conn_df = features.conn_df
            conn_cols = features.conn_cols
            aper_df = features.aper_df
            aper_cols = features.aper_cols
            erp_df = features.erp_df
            erp_cols = features.erp_cols
            itpc_df = features.phase_df
            itpc_cols = features.phase_cols
            itpc_trial_df = features.itpc_trial_df
            itpc_trial_cols = features.itpc_trial_cols
            pac_df = features.pac_df
            pac_trials_df = features.pac_trials_df
            pac_time_df = features.pac_time_df
            comp_df = features.comp_df
            comp_cols = features.comp_cols
            bursts_df = features.bursts_df
            bursts_cols = features.bursts_cols
            spectral_df = features.spectral_df
            spectral_cols = features.spectral_cols
            erds_df = features.erds_df
            erds_cols = features.erds_cols

            power_bands = get_frequency_band_names(self.config)

            current_step += 1
            progress.step(f"Aligning features ({name or 'full'})", current=current_step, total=total_steps)
            self.logger.info(f"Aligning features for {range_info}...")

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
                "complexity": comp_df,
                "spectral": spectral_df,
                "erp": erp_df,
                "bursts": bursts_df,
                "erds": erds_df,
                "ratios": features.ratios_df,
                "asymmetry": features.asymmetry_df,
                "quality": features.quality_df,
            }
            extra_blocks = {k: v for k, v in extra_blocks.items() if v is not None and not getattr(v, "empty", False)}

            (
                pow_df_aligned, baseline_df_aligned, conn_df_aligned,
                aper_df_aligned, y_aligned, retention_stats
            ) = align_feature_dataframes(
                pow_df, baseline_df, conn_df, aper_df, y, aligned_events, features_dir, self.logger, self.config,
                critical_features=critical_features,
                extra_blocks=extra_blocks,
            )

            if retention_stats is None:
                self.logger.error(f"Feature alignment failed for {range_info}. Skipping save.")
                continue

            extra_aligned = retention_stats.get("extra_aligned", {})
            itpc_df = extra_aligned.get("itpc", itpc_df)
            itpc_trial_df = extra_aligned.get("itpc_trial", itpc_trial_df)
            pac_df = extra_aligned.get("pac", pac_df)
            pac_trials_df = extra_aligned.get("pac_trials", pac_trials_df)
            pac_time_df = extra_aligned.get("pac_time", pac_time_df)
            comp_df = extra_aligned.get("complexity", comp_df)
            spectral_df = extra_aligned.get("spectral", spectral_df)
            erp_df = extra_aligned.get("erp", erp_df)
            bursts_df = extra_aligned.get("bursts", bursts_df)
            erds_df = extra_aligned.get("erds", erds_df)
            features.ratios_df = extra_aligned.get("ratios", features.ratios_df)
            features.asymmetry_df = extra_aligned.get("asymmetry", features.asymmetry_df)
            features.quality_df = extra_aligned.get("quality", features.quality_df)

            feature_qc: Dict[str, Any] = {}
            if features.aper_qc is not None:
                feature_qc["aperiodic"] = features.aper_qc
            if ctx.precomputed is not None and getattr(ctx.precomputed, "qc", None) is not None:
                try:
                    feature_qc["precomputed_intermediates"] = asdict(ctx.precomputed.qc)
                except TypeError:
                    feature_qc["precomputed_intermediates"] = getattr(ctx.precomputed, "qc", None)

            current_step += 1
            progress.step(f"Saving features ({name or 'full'})", current=current_step, total=total_steps)
            
            should_combine = self.config.get("feature_engineering.create_combined_features", False)
            combined_df = save_all_features(
                pow_df=pow_df_aligned,
                pow_cols=pow_cols,
                baseline_df=baseline_df_aligned,
                baseline_cols=baseline_cols,
                conn_df=conn_df_aligned,
                conn_cols=conn_cols,
                aper_df=aper_df_aligned,
                aper_cols=aper_cols,
                erp_df=erp_df,
                erp_cols=erp_cols,
                itpc_df=itpc_df,
                itpc_cols=itpc_cols,
                pac_df=pac_df,
                pac_trials_df=pac_trials_df,
                pac_time_df=pac_time_df,
                aper_qc=features.aper_qc,
                y=y_aligned,
                features_dir=features_dir,
                logger=self.logger,
                config=self.config,
                comp_df=comp_df,
                comp_cols=comp_cols,
                bursts_df=bursts_df,
                bursts_cols=bursts_cols,
                spectral_df=spectral_df,
                spectral_cols=spectral_cols,
                erds_df=erds_df,
                erds_cols=erds_cols,
                ratios_df=features.ratios_df,
                ratios_cols=features.ratios_cols,
                asymmetry_df=features.asymmetry_df,
                asymmetry_cols=features.asymmetry_cols,
                quality_df=features.quality_df,
                quality_cols=features.quality_cols,
                feature_qc=feature_qc or None,
                export_all=should_combine,
                suffix=suffix,
            )


            n_trials = len(y_aligned)
            n_pow = pow_df_aligned.shape[1] if pow_df_aligned is not None else 0
            n_conn = conn_df_aligned.shape[1] if conn_df_aligned is not None and not conn_df_aligned.empty else 0
            n_aper = aper_df_aligned.shape[1] if aper_df_aligned is not None and not aper_df_aligned.empty else 0
            n_spectral = spectral_df.shape[1] if spectral_df is not None and not spectral_df.empty else 0
            n_total = combined_df.shape[1] if combined_df is not None else 0

            # Save extraction configuration metadata
            extraction_config = {
                "name": name,
                "spatial_modes": ctx.spatial_modes,
                "aggregation_method": ctx.aggregation_method,
                "tmin": ctx.tmin,
                "tmax": ctx.tmax,
                "bands": ctx.bands,
                "feature_categories": feature_categories,
                "n_trials": n_trials,
                "subject": subject,
                "task": task,
            }
            config_name = f"extraction_config_{suffix}.json" if suffix else "extraction_config.json"
            config_path = features_dir / config_name
            import json
            with open(config_path, "w") as f:
                json.dump(extraction_config, f, indent=2)
            self.logger.info(f"Saved extraction config: {config_path}")

            self.logger.info(
                f"Done {range_info}: sub-{subject}, trials={n_trials}, power={n_pow}, conn={n_conn}, "
                f"aper={n_aper}, spectral={n_spectral}, total={n_total}"
            )

        progress.subject_done(f"sub-{subject}", success=True)


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
    "FeatureExtractionResult",
    "ExtractionResult",
    "FeatureSet",
]
