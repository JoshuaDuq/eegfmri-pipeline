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

        # Accumulators for merging features from all time ranges into default files
        # This ensures all segment features (baseline, active, ramp, etc.) are available
        # in the default feature files for behavior analysis
        accumulated_features: Dict[str, List[pd.DataFrame]] = {
            "power": [],
            "baseline": [],
            "connectivity": [],
            "aperiodic": [],
            "erp": [],
            "itpc": [],
            "pac": [],
            "pac_trials": [],
            "pac_time": [],
            "complexity": [],
            "bursts": [],
            "spectral": [],
            "erds": [],
            "ratios": [],
            "asymmetry": [],
            "quality": [],
        }
        accumulated_y = None
        
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
                suffix=suffix,
            )
            
            # Accumulate features for merging when processing multiple time ranges
            if len(ranges_to_process) > 1:
                if pow_df_aligned is not None and not pow_df_aligned.empty:
                    accumulated_features["power"].append(pow_df_aligned)
                if baseline_df_aligned is not None and not baseline_df_aligned.empty:
                    accumulated_features["baseline"].append(baseline_df_aligned)
                if conn_df_aligned is not None and not conn_df_aligned.empty:
                    accumulated_features["connectivity"].append(conn_df_aligned)
                if aper_df_aligned is not None and not aper_df_aligned.empty:
                    accumulated_features["aperiodic"].append(aper_df_aligned)
                if erp_df is not None and not erp_df.empty:
                    accumulated_features["erp"].append(erp_df)
                if itpc_df is not None and not itpc_df.empty:
                    accumulated_features["itpc"].append(itpc_df)
                if pac_df is not None and not pac_df.empty:
                    accumulated_features["pac"].append(pac_df)
                if pac_trials_df is not None and not pac_trials_df.empty:
                    accumulated_features["pac_trials"].append(pac_trials_df)
                if pac_time_df is not None and not pac_time_df.empty:
                    accumulated_features["pac_time"].append(pac_time_df)
                if comp_df is not None and not comp_df.empty:
                    accumulated_features["complexity"].append(comp_df)
                if bursts_df is not None and not bursts_df.empty:
                    accumulated_features["bursts"].append(bursts_df)
                if spectral_df is not None and not spectral_df.empty:
                    accumulated_features["spectral"].append(spectral_df)
                if erds_df is not None and not erds_df.empty:
                    accumulated_features["erds"].append(erds_df)
                if features.ratios_df is not None and not features.ratios_df.empty:
                    accumulated_features["ratios"].append(features.ratios_df)
                if features.asymmetry_df is not None and not features.asymmetry_df.empty:
                    accumulated_features["asymmetry"].append(features.asymmetry_df)
                if features.quality_df is not None and not features.quality_df.empty:
                    accumulated_features["quality"].append(features.quality_df)
                if accumulated_y is None and y_aligned is not None:
                    accumulated_y = y_aligned


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

        # Merge accumulated features from multiple time ranges into default (unsuffixed) files
        # This ensures all segment features are available for behavior analysis correlations
        if len(ranges_to_process) > 1:
            self.logger.info("Merging features from all time ranges into consolidated default files...")
            
            from eeg_pipeline.infra.tsv import write_tsv
            
            def _merge_dfs(dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
                """Merge DataFrames by concatenating columns, avoiding duplicates."""
                if not dfs:
                    return None
                # Filter out empty DataFrames
                valid_dfs = [df for df in dfs if df is not None and not df.empty]
                if not valid_dfs:
                    return None
                if len(valid_dfs) == 1:
                    return valid_dfs[0]
                # Concatenate column-wise, keeping only unique columns
                merged = pd.concat(valid_dfs, axis=1)
                # Drop duplicate columns (keeping first occurrence)
                merged = merged.loc[:, ~merged.columns.duplicated(keep="first")]
                return merged
            
            # Merge and save each feature type
            merged_power = _merge_dfs(accumulated_features["power"] + accumulated_features["baseline"])
            if merged_power is not None:
                write_tsv(merged_power, features_dir / "features_power.tsv")
                self.logger.info(f"Saved merged power features: {merged_power.shape[1]} columns")
            
            merged_conn = _merge_dfs(accumulated_features["connectivity"])
            if merged_conn is not None:
                write_tsv(merged_conn, features_dir / "features_connectivity.tsv")
                self.logger.info(f"Saved merged connectivity features: {merged_conn.shape[1]} columns")
            
            merged_aper = _merge_dfs(accumulated_features["aperiodic"])
            if merged_aper is not None:
                write_tsv(merged_aper, features_dir / "features_aperiodic.tsv")
                self.logger.info(f"Saved merged aperiodic features: {merged_aper.shape[1]} columns")
            
            merged_erp = _merge_dfs(accumulated_features["erp"])
            if merged_erp is not None:
                write_tsv(merged_erp, features_dir / "features_erp.tsv")
                self.logger.info(f"Saved merged ERP features: {merged_erp.shape[1]} columns")
            
            merged_itpc = _merge_dfs(accumulated_features["itpc"])
            if merged_itpc is not None:
                write_tsv(merged_itpc, features_dir / "features_itpc.tsv")
                self.logger.info(f"Saved merged ITPC features: {merged_itpc.shape[1]} columns")
            
            merged_pac = _merge_dfs(accumulated_features["pac"])
            if merged_pac is not None:
                write_tsv(merged_pac, features_dir / "features_pac.tsv")
                self.logger.info(f"Saved merged PAC features: {merged_pac.shape[1]} columns")
            
            merged_pac_trials = _merge_dfs(accumulated_features["pac_trials"])
            if merged_pac_trials is not None:
                write_tsv(merged_pac_trials, features_dir / "features_pac_trials.tsv")
                self.logger.info(f"Saved merged PAC trials features: {merged_pac_trials.shape[1]} columns")
            
            merged_comp = _merge_dfs(accumulated_features["complexity"])
            if merged_comp is not None:
                write_tsv(merged_comp, features_dir / "features_complexity.tsv")
                self.logger.info(f"Saved merged complexity features: {merged_comp.shape[1]} columns")
            
            merged_bursts = _merge_dfs(accumulated_features["bursts"])
            if merged_bursts is not None:
                write_tsv(merged_bursts, features_dir / "features_bursts.tsv")
                self.logger.info(f"Saved merged bursts features: {merged_bursts.shape[1]} columns")
            
            merged_spectral = _merge_dfs(accumulated_features["spectral"])
            if merged_spectral is not None:
                write_tsv(merged_spectral, features_dir / "features_spectral.tsv")
                self.logger.info(f"Saved merged spectral features: {merged_spectral.shape[1]} columns")
            
            merged_erds = _merge_dfs(accumulated_features["erds"])
            if merged_erds is not None:
                write_tsv(merged_erds, features_dir / "features_erds.tsv")
                self.logger.info(f"Saved merged ERDS features: {merged_erds.shape[1]} columns")
            
            merged_ratios = _merge_dfs(accumulated_features["ratios"])
            if merged_ratios is not None:
                write_tsv(merged_ratios, features_dir / "features_ratios.tsv")
                self.logger.info(f"Saved merged ratios features: {merged_ratios.shape[1]} columns")
            
            merged_asymmetry = _merge_dfs(accumulated_features["asymmetry"])
            if merged_asymmetry is not None:
                write_tsv(merged_asymmetry, features_dir / "features_asymmetry.tsv")
                self.logger.info(f"Saved merged asymmetry features: {merged_asymmetry.shape[1]} columns")
            
            merged_quality = _merge_dfs(accumulated_features["quality"])
            if merged_quality is not None:
                write_tsv(merged_quality, features_dir / "features_quality.tsv")
                self.logger.info(f"Saved merged quality features: {merged_quality.shape[1]} columns")
            
            # Save targets (use the first valid one since they should all be the same)
            if accumulated_y is not None:
                rating_columns = self.config.get("event_columns.rating", ["vas_rating"])
                target_column_name = rating_columns[0] if rating_columns else "vas_rating"
                write_tsv(accumulated_y.to_frame(name=target_column_name), features_dir / "target_vas_ratings.tsv")
                self.logger.info(f"Saved merged targets: {len(accumulated_y)} trials")
            
            # Save merged extraction config summarizing all time ranges
            merged_extraction_config = {
                "merged": True,
                "time_ranges": [tr.get("name") for tr in ranges_to_process],
                "spatial_modes": kwargs.get("spatial_modes") or self.config.get("feature_engineering.spatial_modes", ["roi", "global"]),
                "aggregation_method": kwargs.get("aggregation_method", "mean"),
                "feature_categories": feature_categories,
                "n_trials": len(accumulated_y) if accumulated_y is not None else 0,
                "subject": subject,
                "task": task,
            }
            import json
            with open(features_dir / "extraction_config.json", "w") as f:
                json.dump(merged_extraction_config, f, indent=2)
            self.logger.info("Saved merged extraction config")

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
