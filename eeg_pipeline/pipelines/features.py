"""
Feature Extraction Pipeline (Canonical)
========================================

Single source of truth for feature extraction orchestration.
This module consolidates all feature extraction entry points:
- FeaturePipeline: PipelineBase subclass for batch processing
- extract_all_features: TFR-based feature extraction
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

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import mne

from eeg_pipeline.analysis.features.api import (
    extract_all_features,
    extract_precomputed_features,
)
from eeg_pipeline.analysis.features.preparation import precompute_data
from eeg_pipeline.analysis.features.results import (
    ExtractionResult,
    FeatureExtractionResult,
    FeatureSet,
)
from eeg_pipeline.analysis.features.selection import resolve_feature_categories
from eeg_pipeline.context.features import FeatureContext
from eeg_pipeline.infra.paths import (
    _load_events_df,
    deriv_features_path,
    ensure_dir,
)
from eeg_pipeline.infra.tsv import write_tsv
from eeg_pipeline.pipelines.base import PipelineBase
from eeg_pipeline.plotting.io.figures import setup_matplotlib
from eeg_pipeline.types import PrecomputedData
from eeg_pipeline.utils.analysis.tfr import compute_tfr_morlet
from eeg_pipeline.utils.analysis.windowing import TimeWindowSpec
from eeg_pipeline.utils.config.loader import get_frequency_band_names
from eeg_pipeline.utils.data.columns import pick_target_column
from eeg_pipeline.utils.data.epochs import load_epochs_for_analysis
from eeg_pipeline.utils.data.features import align_feature_dataframes
from eeg_pipeline.utils.data.feature_io import (
    save_all_features,
    save_dropped_trials_log,
    save_trial_alignment_manifest,
)


_FEATURE_ACCUMULATOR_KEYS = [
    "power",
    "baseline",
    "connectivity",
    "directed_connectivity",
    "source_localization",
    "aperiodic",
    "erp",
    "itpc",
    "pac",
    "pac_trials",
    "pac_time",
    "complexity",
    "bursts",
    "spectral",
    "erds",
    "ratios",
    "asymmetry",
    "quality",
]

_TFR_CATEGORIES = {"power", "itpc", "pac"}
_PRECOMPUTE_CATEGORIES = {
    "connectivity",
    "erds",
    "ratios",
    "asymmetry",
    "pac",
    "complexity",
    "bursts",
}


def _resolve_time_ranges(explicit_windows: Optional[List[Dict[str, Any]]], tmin: Optional[float], tmax: Optional[float]) -> List[Dict[str, Any]]:
    """Resolve time ranges from explicit windows or single range parameters."""
    if explicit_windows:
        return list(explicit_windows)
    return [{"name": None, "tmin": tmin, "tmax": tmax}]


def _calculate_total_steps(n_ranges: int) -> int:
    """Calculate total progress steps: load + 3 per time range (extract, align, save)."""
    return 1 + (n_ranges * 3)


def _load_fixed_templates(templates_path: Optional[Path], logger: Any) -> tuple[Optional[np.ndarray], Optional[List[str]]]:
    """Load fixed templates from file if path exists."""
    if templates_path is None or not templates_path.exists():
        return None, None
    
    data = np.load(templates_path)
    templates = data["templates"]
    ch_names = data.get("ch_names")
    logger.info(f"Loaded fixed templates from {templates_path}")
    return templates, ch_names


def _precompute_tfr_if_needed(
    epochs: "mne.Epochs",
    time_ranges: List[Dict[str, Any]],
    feature_categories: List[str],
    config: Any,
    logger: Any,
) -> Optional[Any]:
    """Pre-compute TFR on full epochs if multiple ranges and TFR categories requested."""
    needs_tfr = len(time_ranges) > 1 and any(cat in feature_categories for cat in _TFR_CATEGORIES)
    if not needs_tfr:
        return None
    
    logger.info("Pre-computing TFR on full epochs for multi-range extraction...")
    return compute_tfr_morlet(epochs, config, logger=logger)


def _precompute_intermediates_if_needed(
    epochs: "mne.Epochs",
    time_ranges: List[Dict[str, Any]],
    feature_categories: List[str],
    bands: Optional[List[str]],
    config: Any,
    logger: Any,
) -> Optional[PrecomputedData]:
    """Pre-compute shared intermediates if multiple ranges and precompute categories requested."""
    needs_precompute = len(time_ranges) > 1 and any(
        cat in feature_categories for cat in _PRECOMPUTE_CATEGORIES
    )
    if not needs_precompute:
        return None
    
    logger.info("Pre-computing shared intermediates on full epochs for multi-range extraction...")
    resolved_bands = bands or get_frequency_band_names(config)
    windows_spec = TimeWindowSpec(
        times=epochs.times,
        config=config,
        sampling_rate=float(epochs.info["sfreq"]),
        logger=logger,
        explicit_windows=time_ranges,
    )
    return precompute_data(
        epochs,
        resolved_bands,
        config,
        logger,
        compute_psd_data=True,
        windows_spec=windows_spec,
    )


def _create_feature_accumulator() -> Dict[str, List[pd.DataFrame]]:
    """Create accumulator dictionary for merging features from multiple time ranges."""
    return {key: [] for key in _FEATURE_ACCUMULATOR_KEYS}


def _unpack_feature_results(features: FeatureExtractionResult) -> Dict[str, Any]:
    """Extract all feature DataFrames and column lists from extraction result."""
    return {
        "pow_df": features.pow_df,
        "pow_cols": features.pow_cols,
        "baseline_df": features.baseline_df,
        "baseline_cols": features.baseline_cols,
        "conn_df": features.conn_df,
        "conn_cols": features.conn_cols,
        "dconn_df": features.dconn_df,
        "dconn_cols": features.dconn_cols,
        "source_df": features.source_df,
        "source_cols": features.source_cols,
        "aper_df": features.aper_df,
        "aper_cols": features.aper_cols,
        "erp_df": features.erp_df,
        "erp_cols": features.erp_cols,
        "itpc_df": features.phase_df,
        "itpc_cols": features.phase_cols,
        "itpc_trial_df": features.itpc_trial_df,
        "itpc_trial_cols": features.itpc_trial_cols,
        "pac_df": features.pac_df,
        "pac_trials_df": features.pac_trials_df,
        "pac_time_df": features.pac_time_df,
        "comp_df": features.comp_df,
        "comp_cols": features.comp_cols,
        "bursts_df": features.bursts_df,
        "bursts_cols": features.bursts_cols,
        "spectral_df": features.spectral_df,
        "spectral_cols": features.spectral_cols,
        "erds_df": features.erds_df,
        "erds_cols": features.erds_cols,
        "ratios_df": features.ratios_df,
        "ratios_cols": features.ratios_cols,
        "asymmetry_df": features.asymmetry_df,
        "asymmetry_cols": features.asymmetry_cols,
        "quality_df": features.quality_df,
        "quality_cols": features.quality_cols,
        "aper_qc": features.aper_qc,
    }


def _build_extra_blocks(unpacked: Dict[str, Any], features: FeatureExtractionResult) -> Dict[str, pd.DataFrame]:
    """Build extra blocks dictionary for alignment, filtering out None/empty DataFrames."""
    extra_blocks = {
        "itpc": unpacked["itpc_df"],
        "itpc_trial": unpacked["itpc_trial_df"],
        "pac": unpacked["pac_df"],
        "pac_trials": unpacked["pac_trials_df"],
        "pac_time": unpacked["pac_time_df"],
        "complexity": unpacked["comp_df"],
        "spectral": unpacked["spectral_df"],
        "erp": unpacked["erp_df"],
        "bursts": unpacked["bursts_df"],
        "erds": unpacked["erds_df"],
        "dconn": unpacked["dconn_df"],
        "source": unpacked["source_df"],
        "ratios": features.ratios_df,
        "asymmetry": features.asymmetry_df,
        "quality": features.quality_df,
    }
    return {
        k: v
        for k, v in extra_blocks.items()
        if v is not None and not getattr(v, "empty", False)
    }


def _update_from_aligned_extra(
    unpacked: Dict[str, Any],
    features: FeatureExtractionResult,
    extra_aligned: Dict[str, pd.DataFrame],
) -> None:
    """Update unpacked dict and features object with aligned extra blocks."""
    unpacked["itpc_df"] = extra_aligned.get("itpc", unpacked["itpc_df"])
    unpacked["itpc_trial_df"] = extra_aligned.get("itpc_trial", unpacked["itpc_trial_df"])
    unpacked["pac_df"] = extra_aligned.get("pac", unpacked["pac_df"])
    unpacked["pac_trials_df"] = extra_aligned.get("pac_trials", unpacked["pac_trials_df"])
    unpacked["pac_time_df"] = extra_aligned.get("pac_time", unpacked["pac_time_df"])
    unpacked["comp_df"] = extra_aligned.get("complexity", unpacked["comp_df"])
    unpacked["spectral_df"] = extra_aligned.get("spectral", unpacked["spectral_df"])
    unpacked["erp_df"] = extra_aligned.get("erp", unpacked["erp_df"])
    unpacked["bursts_df"] = extra_aligned.get("bursts", unpacked["bursts_df"])
    unpacked["erds_df"] = extra_aligned.get("erds", unpacked["erds_df"])
    unpacked["dconn_df"] = extra_aligned.get("dconn", unpacked["dconn_df"])
    unpacked["source_df"] = extra_aligned.get("source", unpacked["source_df"])
    features.ratios_df = extra_aligned.get("ratios", features.ratios_df)
    features.asymmetry_df = extra_aligned.get("asymmetry", features.asymmetry_df)
    features.quality_df = extra_aligned.get("quality", features.quality_df)


def _build_feature_qc(features: FeatureExtractionResult, ctx: FeatureContext) -> Dict[str, Any]:
    """Build feature QC dictionary from extraction result and context."""
    qc: Dict[str, Any] = {}
    if features.aper_qc is not None:
        qc["aperiodic"] = features.aper_qc
    if ctx.precomputed is not None and getattr(ctx.precomputed, "qc", None) is not None:
        try:
            qc["precomputed_intermediates"] = asdict(ctx.precomputed.qc)
        except TypeError:
            qc["precomputed_intermediates"] = getattr(ctx.precomputed, "qc", None)
    return qc


def _accumulate_features(
    accumulated: Dict[str, List[pd.DataFrame]],
    unpacked: Dict[str, Any],
    features: FeatureExtractionResult,
    aligned: Dict[str, pd.DataFrame],
) -> None:
    """Accumulate aligned features for later merging."""
    feature_mapping = {
        "power": aligned.get("pow_df_aligned"),
        "baseline": aligned.get("baseline_df_aligned"),
        "connectivity": aligned.get("conn_df_aligned"),
        "directed_connectivity": unpacked["dconn_df"],
        "source_localization": unpacked["source_df"],
        "aperiodic": aligned.get("aper_df_aligned"),
        "erp": unpacked["erp_df"],
        "itpc": unpacked["itpc_df"],
        "pac": unpacked["pac_df"],
        "pac_trials": unpacked["pac_trials_df"],
        "pac_time": unpacked["pac_time_df"],
        "complexity": unpacked["comp_df"],
        "bursts": unpacked["bursts_df"],
        "spectral": unpacked["spectral_df"],
        "erds": unpacked["erds_df"],
        "ratios": features.ratios_df,
        "asymmetry": features.asymmetry_df,
        "quality": features.quality_df,
    }
    
    for key, df in feature_mapping.items():
        if df is not None and not df.empty:
            accumulated[key].append(df)


def _merge_dataframes(dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Merge DataFrames by concatenating columns, avoiding duplicates."""
    valid_dfs = [df for df in dfs if df is not None and not df.empty]
    if not valid_dfs:
        return None
    if len(valid_dfs) == 1:
        return valid_dfs[0]
    
    merged = pd.concat(valid_dfs, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated(keep="first")]
    return merged


def _save_merged_features(
    accumulated: Dict[str, List[pd.DataFrame]],
    features_dir: Path,
    logger: Any,
) -> None:
    """Merge and save accumulated features from multiple time ranges."""
    feature_file_mapping = {
        "power": ("features_power.tsv", ["power", "baseline"]),
        "connectivity": ("features_connectivity.tsv", ["connectivity"]),
        "directed_connectivity": ("features_directed_connectivity.tsv", ["directed_connectivity"]),
        "source_localization": ("features_source_localization.tsv", ["source_localization"]),
        "aperiodic": ("features_aperiodic.tsv", ["aperiodic"]),
        "erp": ("features_erp.tsv", ["erp"]),
        "itpc": ("features_itpc.tsv", ["itpc"]),
        "pac": ("features_pac.tsv", ["pac"]),
        "pac_trials": ("features_pac_trials.tsv", ["pac_trials"]),
        "complexity": ("features_complexity.tsv", ["complexity"]),
        "bursts": ("features_bursts.tsv", ["bursts"]),
        "spectral": ("features_spectral.tsv", ["spectral"]),
        "erds": ("features_erds.tsv", ["erds"]),
        "ratios": ("features_ratios.tsv", ["ratios"]),
        "asymmetry": ("features_asymmetry.tsv", ["asymmetry"]),
        "quality": ("features_quality.tsv", ["quality"]),
    }
    
    for filename, keys in feature_file_mapping.values():
        dfs_to_merge = []
        for key in keys:
            dfs_to_merge.extend(accumulated.get(key, []))
        
        merged_df = _merge_dataframes(dfs_to_merge)
        if merged_df is not None:
            from eeg_pipeline.utils.data.feature_io import _get_folder_for_feature
            base_name = filename.replace(".tsv", "")
            folder = _get_folder_for_feature(base_name)
            save_path = features_dir / folder / filename
            write_tsv(merged_df, save_path)
            feature_name = filename.replace("features_", "").replace(".tsv", "")
            logger.info(f"Saved merged {feature_name} features: {merged_df.shape[1]} columns")


def _save_extraction_config(
    config: Dict[str, Any],
    features_dir: Path,
    suffix: Optional[str],
    logger: Any,
) -> None:
    """Save extraction configuration to JSON file."""
    config_name = f"extraction_config_{suffix}.json" if suffix else "extraction_config.json"
    save_path = features_dir / "metadata" / config_name
    ensure_dir(save_path.parent)
    with open(save_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved extraction config: {save_path}")


class FeaturePipeline(PipelineBase):
    """Pipeline for EEG feature extraction."""

    def __init__(self, config: Optional[Any] = None):
        super().__init__(name="feature_extraction", config=config)

    def process_subject(self, subject: str, task: Optional[str] = None, **kwargs) -> None:
        from eeg_pipeline.cli.common import ProgressReporter

        task = task or self.config.get("project.task")
        if task is None:
            raise ValueError("Missing required config value: project.task")

        feature_categories = resolve_feature_categories(
            self.config, kwargs.get("feature_categories")
        )
        progress = kwargs.get("progress") or ProgressReporter(enabled=False)

        self.logger.info(f"=== Feature extraction: sub-{subject}, task-{task} ===")
        self.logger.info("Feature categories: %s", ", ".join(feature_categories))
        progress.subject_start(f"sub-{subject}")

        features_dir = deriv_features_path(self.deriv_root, subject)
        ensure_dir(features_dir)
        setup_matplotlib(self.config)

        explicit_windows = kwargs.get("time_ranges")
        time_ranges = _resolve_time_ranges(explicit_windows, kwargs.get("tmin"), kwargs.get("tmax"))
        total_steps = _calculate_total_steps(len(time_ranges))
        current_step = 0

        current_step += 1
        progress.step("Loading epochs", current=current_step, total=total_steps)
        epochs, aligned_events = load_epochs_for_analysis(
            subject,
            task,
            align="strict",
            preload=True,
            deriv_root=self.deriv_root,
            logger=self.logger,
            config=self.config,
        )

        if epochs is None:
            self.logger.error(f"No cleaned epochs for sub-{subject}; skipping")
            progress.error("no_epochs", f"No cleaned epochs for sub-{subject}")
            return

        if aligned_events is None:
            self.logger.warning("No events available; skipping")
            progress.error("no_events", "No aligned events")
            return

        original_events = _load_events_df(
            subject, task, bids_root=self.config.bids_root, config=self.config
        )
        if original_events is not None:
            save_dropped_trials_log(
                epochs, original_events, features_dir / "metadata" / "dropped_trials.tsv", self.logger
            )

        save_trial_alignment_manifest(
            aligned_events,
            epochs,
            features_dir / "metadata" / "trial_alignment.json",
            self.config,
            self.logger,
        )

        target_columns = list(self.config.get("event_columns.rating", []) or [])
        target_col = pick_target_column(aligned_events, target_columns=target_columns)
        if target_col is None:
            self.logger.warning("No target column found; skipping")
            return

        y = pd.to_numeric(aligned_events[target_col], errors="coerce")

        fixed_templates_path = kwargs.get("fixed_templates_path")
        fixed_templates, fixed_template_ch_names = _load_fixed_templates(
            fixed_templates_path, self.logger
        )

        tfr_full = _precompute_tfr_if_needed(
            epochs, time_ranges, feature_categories, self.config, self.logger
        )

        precomputed_full = _precompute_intermediates_if_needed(
            epochs,
            time_ranges,
            feature_categories,
            kwargs.get("bands"),
            self.config,
            self.logger,
        )

        accumulated_features = _create_feature_accumulator()
        accumulated_y = None

        for tr_spec in time_ranges:
            name = tr_spec.get("name")
            tmin = tr_spec.get("tmin")
            tmax = tr_spec.get("tmax")

            if tmin is not None and tmax is not None and tmin > tmax:
                self.logger.warning(
                    f"Time range '{name}' has tmin ({tmin}) > tmax ({tmax}). Swapping values."
                )
                tmin, tmax = tmax, tmin

            suffix = name
            range_info = f"range '{name}'" if name else "default range"
            self.logger.info(f"--- Processing {range_info} ({tmin} to {tmax}s) ---")

            spatial_modes = kwargs.get("spatial_modes") or self.config.get(
                "feature_engineering.spatial_modes", ["roi", "channels", "global"]
            )
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
                bands=kwargs.get("bands"),
                spatial_modes=spatial_modes,
                tmin=tmin,
                tmax=tmax,
                name=name,
                aggregation_method=kwargs.get("aggregation_method", "mean"),
                tfr=tfr_full,
                precomputed=precomputed_full,
                explicit_windows=explicit_windows,
            )
            ctx.progress = progress
            ctx.total_steps = total_steps
            ctx.current_step = current_step

            current_step += 1
            progress.step(
                f"Extracting features ({name or 'full'})",
                current=current_step,
                total=total_steps,
            )
            features = extract_all_features(ctx)

            unpacked = _unpack_feature_results(features)

            current_step += 1
            progress.step(
                f"Aligning features ({name or 'full'})",
                current=current_step,
                total=total_steps,
            )
            self.logger.info(f"Aligning features for {range_info}...")

            critical_features = ["target"]
            if "power" in ctx.feature_categories:
                critical_features.extend(["power", "baseline"])

            extra_blocks = _build_extra_blocks(unpacked, features)

            (
                pow_df_aligned,
                baseline_df_aligned,
                conn_df_aligned,
                aper_df_aligned,
                y_aligned,
                retention_stats,
            ) = align_feature_dataframes(
                unpacked["pow_df"],
                unpacked["baseline_df"],
                unpacked["conn_df"],
                unpacked["aper_df"],
                y,
                aligned_events,
                features_dir,
                self.logger,
                self.config,
                critical_features=critical_features,
                extra_blocks=extra_blocks,
                requested_categories=ctx.feature_categories,
            )

            if retention_stats is None:
                self.logger.error(f"Feature alignment failed for {range_info}. Skipping save.")
                continue

            extra_aligned = retention_stats.get("extra_aligned", {})
            _update_from_aligned_extra(unpacked, features, extra_aligned)

            feature_qc = _build_feature_qc(features, ctx)

            current_step += 1
            progress.step(
                f"Saving features ({name or 'full'})",
                current=current_step,
                total=total_steps,
            )

            combined_df = save_all_features(
                pow_df=pow_df_aligned,
                pow_cols=unpacked["pow_cols"],
                baseline_df=baseline_df_aligned,
                baseline_cols=unpacked["baseline_cols"],
                conn_df=conn_df_aligned,
                conn_cols=unpacked["conn_cols"],
                aper_df=aper_df_aligned,
                aper_cols=unpacked["aper_cols"],
                erp_df=unpacked["erp_df"],
                erp_cols=unpacked["erp_cols"],
                itpc_df=unpacked["itpc_df"],
                itpc_cols=unpacked["itpc_cols"],
                pac_df=unpacked["pac_df"],
                pac_trials_df=unpacked["pac_trials_df"],
                pac_time_df=unpacked["pac_time_df"],
                aper_qc=features.aper_qc,
                y=y_aligned,
                features_dir=features_dir,
                logger=self.logger,
                config=self.config,
                comp_df=unpacked["comp_df"],
                comp_cols=unpacked["comp_cols"],
                bursts_df=unpacked["bursts_df"],
                bursts_cols=unpacked["bursts_cols"],
                spectral_df=unpacked["spectral_df"],
                spectral_cols=unpacked["spectral_cols"],
                erds_df=unpacked["erds_df"],
                erds_cols=unpacked["erds_cols"],
                ratios_df=features.ratios_df,
                ratios_cols=features.ratios_cols,
                asymmetry_df=features.asymmetry_df,
                asymmetry_cols=features.asymmetry_cols,
                quality_df=features.quality_df,
                quality_cols=features.quality_cols,
                dconn_df=unpacked["dconn_df"],
                dconn_cols=unpacked["dconn_cols"],
                source_df=unpacked["source_df"],
                source_cols=unpacked["source_cols"],
                feature_qc=feature_qc or None,
                suffix=suffix,
            )

            if len(time_ranges) > 1:
                aligned_dict = {
                    "pow_df_aligned": pow_df_aligned,
                    "baseline_df_aligned": baseline_df_aligned,
                    "conn_df_aligned": conn_df_aligned,
                    "aper_df_aligned": aper_df_aligned,
                }
                _accumulate_features(accumulated_features, unpacked, features, aligned_dict)
                if accumulated_y is None and y_aligned is not None:
                    accumulated_y = y_aligned

            n_trials = len(y_aligned)
            n_pow = pow_df_aligned.shape[1] if pow_df_aligned is not None else 0
            n_conn = (
                conn_df_aligned.shape[1]
                if conn_df_aligned is not None and not conn_df_aligned.empty
                else 0
            )
            n_dconn = (
                unpacked["dconn_df"].shape[1]
                if unpacked["dconn_df"] is not None and not unpacked["dconn_df"].empty
                else 0
            )
            n_source = (
                unpacked["source_df"].shape[1]
                if unpacked["source_df"] is not None and not unpacked["source_df"].empty
                else 0
            )
            n_aper = (
                aper_df_aligned.shape[1]
                if aper_df_aligned is not None and not aper_df_aligned.empty
                else 0
            )
            n_spectral = (
                unpacked["spectral_df"].shape[1]
                if unpacked["spectral_df"] is not None
                and not unpacked["spectral_df"].empty
                else 0
            )
            n_total = combined_df.shape[1] if combined_df is not None else 0

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
            _save_extraction_config(extraction_config, features_dir, suffix, self.logger)

            self.logger.info(
                f"Done {range_info}: sub-{subject}, trials={n_trials}, "
                f"power={n_pow}, conn={n_conn}, dconn={n_dconn}, source={n_source}, "
                f"aper={n_aper}, spectral={n_spectral}, total={n_total}"
            )

        if len(time_ranges) > 1:
            self.logger.info(
                "Merging features from all time ranges into consolidated default files..."
            )
            _save_merged_features(accumulated_features, features_dir, self.logger)

            if accumulated_y is not None:
                rating_columns = self.config.get("event_columns.rating", ["vas_rating"])
                target_column_name = rating_columns[0] if rating_columns else "vas_rating"
                write_tsv(
                    accumulated_y.to_frame(name=target_column_name),
                    features_dir / "behavior" / "target_vas_ratings.tsv",
                )
                self.logger.info(f"Saved merged targets: {len(accumulated_y)} trials")

            merged_extraction_config = {
                "merged": True,
                "time_ranges": [tr.get("name") for tr in time_ranges],
                "spatial_modes": kwargs.get("spatial_modes")
                or self.config.get("feature_engineering.spatial_modes", ["roi", "global"]),
                "aggregation_method": kwargs.get("aggregation_method", "mean"),
                "feature_categories": feature_categories,
                "n_trials": len(accumulated_y) if accumulated_y is not None else 0,
                "subject": subject,
                "task": task,
            }
            _save_extraction_config(
                merged_extraction_config, features_dir, None, self.logger
            )
            self.logger.info("Saved merged extraction config")

        progress.subject_done(f"sub-{subject}", success=True)


def process_subject(
    subject: str, task: Optional[str] = None, config: Optional[Any] = None, **kwargs: Any
) -> None:
    """Process a single subject through the feature extraction pipeline."""
    pipeline = FeaturePipeline(config=config)
    pipeline.process_subject(subject, task=task, **kwargs)


def extract_features_for_subjects(
    subjects: List[str],
    task: Optional[str] = None,
    config: Optional[Any] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Extract features for multiple subjects."""
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
