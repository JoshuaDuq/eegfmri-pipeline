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

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from itertools import combinations

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import mne

from eeg_pipeline.context.features import (
    FeatureContext,
    FEATURE_CATEGORIES,
    PRECOMPUTED_GROUP_CHOICES,
)
from eeg_pipeline.pipelines.base import PipelineBase
from eeg_pipeline.types import PrecomputedData
from eeg_pipeline.utils.io.paths import deriv_features_path, deriv_plots_path, ensure_dir, _load_events_df
from eeg_pipeline.utils.io.plotting import setup_matplotlib
from eeg_pipeline.utils.io.tsv import write_tsv
from eeg_pipeline.utils.io.columns import _pick_target_column
from eeg_pipeline.utils.data.loading import load_epochs_for_analysis
from eeg_pipeline.utils.data.features import (
    align_feature_dataframes,
    compute_group_microstate_templates,
    export_fmri_regressors,
    load_group_microstate_templates,
    save_all_features,
    save_dropped_trials_log,
    save_microstate_templates,
    save_trial_alignment_manifest,
)
from eeg_pipeline.utils.config.loader import get_frequency_band_names, get_frequency_bands
from eeg_pipeline.utils.validation import validate_epochs
from eeg_pipeline.utils.progress import PipelineProgress
from eeg_pipeline.utils.analysis.tfr import (
    compute_tfr_for_subject,
    get_tfr_config,
    compute_adaptive_n_cycles,
    save_tfr_with_sidecar,
    resolve_tfr_workers,
    compute_complex_tfr,
)

from eeg_pipeline.analysis.features.power import (
    extract_power_features,
    extract_spectral_extras_from_precomputed,
    extract_asymmetry_from_precomputed,
    extract_segment_power_from_precomputed,
)
from eeg_pipeline.analysis.features.connectivity import (
    extract_connectivity_features,
    extract_connectivity_from_precomputed,
)
from eeg_pipeline.analysis.features.microstates import (
    extract_microstate_features,
    extract_microstate_features_from_epochs,
)
from eeg_pipeline.analysis.features.aperiodic import (
    extract_aperiodic_features,
    extract_aperiodic_features_from_epochs,
)
from eeg_pipeline.analysis.features.phase import extract_phase_features, compute_pac_comodulograms
from eeg_pipeline.analysis.features.complexity import (
    extract_dynamics_features,
    extract_complexity_from_precomputed,
)
from eeg_pipeline.analysis.features.quality import extract_quality_features, compute_trial_quality_metrics
from eeg_pipeline.analysis.features.precompute import precompute_data
from eeg_pipeline.analysis.features.extractors import (
    extract_erds_from_precomputed,
    extract_power_from_precomputed,
    extract_gfp_from_precomputed,
    extract_roi_features_from_precomputed,
    extract_temporal_features_from_precomputed,
    extract_band_ratios_from_precomputed,
)
from eeg_pipeline.analysis.features.results import (
    FeatureSet,
    ExtractionResult,
    FeatureExtractionResult,
)
from eeg_pipeline.analysis.features.dynamics import extract_dynamics_from_precomputed
from eeg_pipeline.analysis.features.cfc import extract_pac_from_precomputed, extract_all_cfc_features


###################################################################
# TFR-Based Feature Extraction
###################################################################


def _get_precomputed_groups(config: Any, override: Optional[List[str]] = None) -> List[str]:
    """
    Resolve which precomputed feature groups to run.
    
    Priority:
    1. Runtime override (CLI/kwargs)
    2. Config entry feature_engineering.precomputed_groups
    """
    groups = override if override is not None else config.get("feature_engineering.precomputed_groups")
    if not groups:
        raise ValueError(
            "Precomputed groups not specified; set feature_engineering.precomputed_groups "
            "in eeg_config.yaml or provide via CLI."
        )
    
    groups = [str(g) for g in groups]
    groups = list(dict.fromkeys(groups))
    if not groups:
        raise ValueError("No precomputed groups specified; provide at least one.")
    
    unknown = [g for g in groups if g not in PRECOMPUTED_GROUP_CHOICES]
    if unknown:
        logging.getLogger(__name__).warning(
            "Skipping unsupported precomputed groups (not implemented/allowed): %s",
            ", ".join(unknown),
        )

    supported = [g for g in groups if g in PRECOMPUTED_GROUP_CHOICES]
    if not supported:
        raise ValueError(
            "No supported precomputed groups requested after filtering; requested="
            f"{groups}. Supported={PRECOMPUTED_GROUP_CHOICES}"
        )
    return supported




def _resolve_feature_categories(config: Any, requested: Optional[List[str]]) -> List[str]:
    """
    Decide which feature categories to compute.
    
    Priority:
    1. Runtime-provided list (CLI/kwargs)
    2. Config entry feature_engineering.feature_categories
    3. All available categories (default)
    """
    if requested is not None:
        categories = list(requested)
    else:
        from_config = config.get("feature_engineering.feature_categories")
        categories = list(from_config) if from_config else list(FEATURE_CATEGORIES)
    
    categories = list(dict.fromkeys(str(cat) for cat in categories))
    if not categories:
        raise ValueError("No feature categories specified; provide at least one.")
    
    invalid = [c for c in categories if c not in FEATURE_CATEGORIES]
    if invalid:
        raise ValueError(
            f"Invalid feature categories {invalid}. "
            f"Valid options: {', '.join(FEATURE_CATEGORIES)}"
        )
    return categories


def _combine_feature_groups(result: Any, groups: List[str]) -> tuple:
    dfs: List[pd.DataFrame] = []
    cols: List[str] = []
    for group in groups:
        fs = result.features.get(group)
        if fs is None or fs.df.empty:
            continue
        dfs.append(fs.df)
        cols.extend(fs.columns)
    if not dfs:
        return pd.DataFrame(), []
    combined = pd.concat(dfs, axis=1)
    if result.condition is not None:
        combined.insert(0, "condition", result.condition)
    fixed_cols = ["condition"] if "condition" in combined.columns else []
    other_cols = sorted([c for c in combined.columns if c not in fixed_cols])
    combined = combined[fixed_cols + other_cols] if fixed_cols else combined[other_cols]
    return combined, cols


def extract_all_features(
    ctx: FeatureContext,
    *,
    precomputed_groups_override: Optional[List[str]] = None,
) -> FeatureExtractionResult:
    """
    Extract all EEG features for a subject using TFR-based approach.
    
    This is the primary feature extraction function that computes TFR on-the-fly
    and extracts power, connectivity, microstates, aperiodic, phase, and PAC features.
    """
    power_bands = get_frequency_band_names(ctx.config)
    n_microstates = int(ctx.config.get("feature_engineering.microstates.n_states", 4))
    expected_n_trials = len(ctx.epochs)

    precomputed_data = None
    needs_precompute = any(c in ctx.feature_categories for c in ["precomputed", "cfc", "dynamics_advanced"])
    if needs_precompute and ctx.ensure_precomputed():
        precomputed_data = ctx.precomputed

    def _check_length(name: str, df: Optional[pd.DataFrame]) -> None:
        if df is None or getattr(df, "empty", False):
            return
        if len(df) != expected_n_trials:
            raise ValueError(f"{name} length mismatch: {len(df)} vs {expected_n_trials}")

    def _resolve_precomputed_groups() -> List[str]:
        try:
            return _get_precomputed_groups(ctx.config, precomputed_groups_override)
        except Exception as exc:
            ctx.logger.error(f"Precomputed groups unavailable: {exc}")
            return []

    validation = validate_epochs(ctx.epochs, ctx.config, logger=ctx.logger)
    if not validation.valid:
        ctx.logger.warning(f"Validation issues: {validation.issues}")
        if validation.critical:
            raise ValueError(f"Critical errors: {validation.critical}")

    results = FeatureExtractionResult()

    tfr_dependent_categories = {"power", "connectivity", "itpc", "pac"}
    needs_tfr = bool(tfr_dependent_categories & set(ctx.feature_categories))
    
    tfr_complex = None
    tfr_power = None
    tfr = None
    baseline_df = None
    baseline_cols = []
    b_start, b_end = None, None
    
    if needs_tfr:
        if any(c in ctx.feature_categories for c in ["itpc", "pac"]):
            tfr_complex = compute_complex_tfr(ctx.epochs, ctx.config, ctx.logger)
            if tfr_complex is not None:
                ctx.tfr_complex = tfr_complex
                ctx.logger.info("Deriving power TFR from complex TFR...")
                tfr_power = tfr_complex.copy()
                tfr_power.data = np.abs(tfr_complex.data) ** 2
                tfr_power.comment = "derived_from_complex"

        tfr, baseline_df, baseline_cols, b_start, b_end = compute_tfr_for_subject(
            ctx.epochs, ctx.aligned_events, ctx.subject, ctx.task, ctx.config, ctx.deriv_root, ctx.logger,
            tfr_computed=tfr_power
        )
        if tfr is None:
            raise ValueError("TFR computation failed; aborting feature extraction.")
        results.tfr = tfr
        results.baseline_df = baseline_df
        results.baseline_cols = baseline_cols
        
        ctx.results["tfr"] = tfr
        ctx.tfr = tfr
        ctx.baseline_df = baseline_df
        ctx.baseline_cols = baseline_cols

        # IMPORTANT: keep the in-memory TFR in raw power units for feature extraction.
        # Baseline-normalized power features are computed explicitly in extract_power_features().
        # This avoids ambiguous mixing of already-baselined vs logratio-derived features.
        if ctx.config.get("feature_engineering.save_tfr_with_sidecar", False):
            try:
                if tfr is not None and b_start is not None and b_end is not None:
                    baseline_mode = str(ctx.config.get("time_frequency_analysis.baseline_mode", "logratio"))
                    tfr_to_save = tfr.copy()
                    tfr_to_save.apply_baseline(baseline=(b_start, b_end), mode=baseline_mode)
                    tfr_to_save.comment = f"BASELINED:mode={baseline_mode};win=({b_start:.3f},{b_end:.3f})"
                    tfr_out = ctx.deriv_root / f"sub-{ctx.subject}" / "eeg" / f"sub-{ctx.subject}_task-{ctx.task}_power_epo-tfr.h5"
                    save_tfr_with_sidecar(tfr_to_save, tfr_out, (b_start, b_end), baseline_mode, ctx.logger, ctx.config)
            except Exception as exc:
                ctx.logger.warning("Failed to save baselined TFR with sidecar: %s", exc)
    else:
        ctx.logger.info("Skipping TFR computation (not needed for requested feature categories)")

    progress = PipelineProgress(total=len(ctx.feature_categories), logger=ctx.logger, desc="Features")
    progress.start()

    if "power" in ctx.feature_categories:
        progress.step(message="Extracting power features...")
        pow_df, pow_cols = extract_power_features(ctx, power_bands)
        if pow_df is not None and not pow_df.empty:
            if len(pow_df) != expected_n_trials:
                 raise ValueError(f"Power length mismatch: {len(pow_df)} vs {expected_n_trials}")
        results.pow_df = pow_df
        results.pow_cols = pow_cols

    if "connectivity" in ctx.feature_categories:
        progress.step(message="Extracting connectivity features...")
        conn_df, conn_cols = extract_connectivity_features(ctx, power_bands)
        if conn_df is not None and not conn_df.empty:
             if len(conn_df) != expected_n_trials:
                raise ValueError(f"Connectivity length mismatch: {len(conn_df)} vs {expected_n_trials}")
        results.conn_df = conn_df
        results.conn_cols = conn_cols

    if "microstates" in ctx.feature_categories:
        progress.step(message="Extracting microstate features...")
        use_fixed = bool(ctx.config.get("feature_engineering.microstates.use_fixed_templates", False))
        use_group = bool(ctx.config.get("feature_engineering.microstates.use_group_templates", False))

        if use_group and not use_fixed:
            group_templates, group_ch_names = load_group_microstate_templates(ctx.deriv_root, n_microstates, ctx.logger)
            if group_templates is not None:
                ctx.fixed_templates = group_templates
                ctx.fixed_template_ch_names = group_ch_names
                use_fixed = True
                ctx.logger.info("Using group-level microstate templates")

        ms_df, ms_cols, ms_templates = extract_microstate_features(ctx)

        _check_length("Microstates", ms_df)
        results.ms_df = ms_df
        results.ms_cols = ms_cols
        results.ms_templates = ms_templates

    if "aperiodic" in ctx.feature_categories:
        progress.step(message="Extracting aperiodic features...")
        aper_df, aper_cols, qc_payload = extract_aperiodic_features(ctx, power_bands)
        _check_length("Aperiodic", aper_df)
        results.aper_df = aper_df
        results.aper_cols = aper_cols
        results.aper_qc = qc_payload

    if "complexity" in ctx.feature_categories:
        progress.step(message="Extracting complexity features...")
        comp_df, comp_cols = extract_dynamics_features(ctx, power_bands)
        _check_length("Complexity", comp_df)
        results.comp_df = comp_df
        results.comp_cols = comp_cols

    if "itpc" in ctx.feature_categories or "phase" in ctx.feature_categories:
        progress.step(message="Extracting phase features...")
        phase_df, phase_cols = extract_phase_features(ctx, power_bands)
        _check_length("Phase", phase_df)
        results.phase_df = phase_df
        results.phase_cols = phase_cols

    if "pac" in ctx.feature_categories:
        progress.step(message="Computing PAC features...")
        if tfr_complex is None:
            tfr_complex = compute_complex_tfr(ctx.epochs, ctx.config, ctx.logger)
            if tfr_complex is not None:
                ctx.tfr_complex = tfr_complex

        if tfr_complex is not None:
            freq_min, freq_max, n_freqs, *_ = get_tfr_config(ctx.config)
            freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)

            plateau_meta = ctx.windows.metadata.get("plateau") if ctx.windows is not None else None
            plateau_window = None
            if plateau_meta is not None and getattr(plateau_meta, "valid", False):
                plateau_window = (float(plateau_meta.start), float(plateau_meta.end))

            pac_df, pac_phase_freqs, pac_amp_freqs, pac_trials_df, pac_time_df = compute_pac_comodulograms(
                tfr_complex,
                freqs,
                tfr_complex.times,
                ctx.epochs.info,
                ctx.config,
                ctx.logger,
                segment_name="plateau",
                segment_window=plateau_window,
            )
            _check_length("PAC trials", pac_trials_df)
            _check_length("PAC time-resolved", pac_time_df)
            results.pac_df = pac_df
            results.pac_phase_freqs = pac_phase_freqs
            results.pac_amp_freqs = pac_amp_freqs
            results.pac_trials_df = pac_trials_df
            results.pac_time_df = pac_time_df

    precomputed_result = None
    configured_groups = _resolve_precomputed_groups() if "precomputed" in ctx.feature_categories else []
    requested_precomputed_groups = configured_groups.copy()
    if "cfc" in ctx.feature_categories:
        requested_precomputed_groups.append("cfc")
    if "dynamics_advanced" in ctx.feature_categories:
        requested_precomputed_groups.append("dynamics_advanced")
    requested_precomputed_groups = list(dict.fromkeys(requested_precomputed_groups))

    def _ensure_precomputed_result() -> Optional[Any]:
        nonlocal precomputed_result
        if precomputed_result is not None or not requested_precomputed_groups:
            return precomputed_result
        precomputed_result = extract_precomputed_features(
            ctx.epochs,
            power_bands,
            ctx.config,
            ctx.logger,
            feature_groups=requested_precomputed_groups,
            n_plateau_windows=int(ctx.config.get("feature_engineering.erds.n_temporal_windows", 5)),
            events_df=ctx.aligned_events,
            precomputed=precomputed_data,
        )
        if ctx.precomputed is None and precomputed_result is not None:
            ctx.precomputed = precomputed_result.precomputed
        return precomputed_result

    if "precomputed" in ctx.feature_categories:
        progress.step(message="Extracting precomputed features...")
        precomputed_result = _ensure_precomputed_result()
        if precomputed_result is not None:
            precomputed_df, precomputed_cols = _combine_feature_groups(precomputed_result, configured_groups)
            results.precomputed_df = precomputed_df
            results.precomputed_cols = precomputed_cols
            _check_length("Precomputed", precomputed_df)
            if precomputed_df is not None and not precomputed_df.empty:
                condition_info = ""
                if precomputed_result.condition is not None:
                    condition_info = f" (pain={precomputed_result.n_pain}, nonpain={precomputed_result.n_nonpain})"
                ctx.logger.info(f"Extracted {len(precomputed_cols)} precomputed features{condition_info}")

    if "cfc" in ctx.feature_categories:
        progress.step(message="Extracting CFC features...")
        precomputed_result = _ensure_precomputed_result()
        if precomputed_result is not None:
            cfc_df = precomputed_result.get_feature_group_df("cfc")
            _check_length("CFC", cfc_df)
            if cfc_df is not None and not cfc_df.empty:
                cfc_fs = precomputed_result.features.get("cfc")
                if cfc_fs is None:
                    ctx.logger.warning("CFC features missing from shared precomputed result")
                else:
                    results.cfc_df = cfc_df
                    results.cfc_cols = list(cfc_fs.columns)
                    ctx.logger.info(f"Extracted {len(results.cfc_cols)} CFC features")

    if "dynamics_advanced" in ctx.feature_categories:
        progress.step(message="Extracting advanced dynamics features...")
        precomputed_result = _ensure_precomputed_result()
        if precomputed_result is not None:
            dyn_df = precomputed_result.get_feature_group_df("dynamics_advanced")
            _check_length("Advanced dynamics", dyn_df)
            if dyn_df is not None and not dyn_df.empty:
                dyn_fs = precomputed_result.features.get("dynamics_advanced")
                if dyn_fs is None:
                    ctx.logger.warning("Dynamics features missing from shared precomputed result")
                else:
                    results.dynamics_df = dyn_df
                    results.dynamics_cols = list(dyn_fs.columns)
                    ctx.logger.info(f"Extracted {len(results.dynamics_cols)} dynamics features")

    if "quality" in ctx.feature_categories:
        progress.step(message="Computing trial quality metrics...")
        qual_df, qual_cols = extract_quality_features(ctx)
        _check_length("Quality metrics", qual_df)
        results.quality_df = qual_df
        results.quality_cols = qual_cols

    progress.finish()
    return results


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
    """
    Orchestrate efficient feature extraction using precomputed intermediates.
    
    This is more efficient than TFR-on-the-fly extraction for most use cases.
    """
    from eeg_pipeline.utils.config.loader import get_config_value
    from eeg_pipeline.analysis.features.phase import extract_itpc_from_precomputed
    
    if feature_groups is None:
        feature_groups = ["erds", "spectral"]
    
    compute_bands = any(
        g in feature_groups
        for g in [
            "erds",
            "spectral",
            "connectivity",
            "pac",
            "dynamics_advanced",
            "roi",
            "ratios",
        ]
    )
    compute_psd_data = any(g in feature_groups for g in ["aperiodic", "spectral"])
    
    if precomputed is None:
        logger.info("Precomputing intermediate data (bands=%s, psd=%s)...", compute_bands, compute_psd_data)
        precomputed = precompute_data(
            epochs,
            bands,
            config,
            logger,
            compute_bands=compute_bands,
            compute_psd_data=compute_psd_data,
            n_plateau_windows=n_plateau_windows,
        )
    else:
        logger.info("Using provided precomputed intermediates")
    
    result = ExtractionResult(precomputed=precomputed)
    
    if events_df is not None and len(events_df) == precomputed.data.shape[0]:
        if "condition" in events_df.columns:
            result.condition = events_df["condition"].to_numpy()
        elif "trial_type" in events_df.columns:
            result.condition = events_df["trial_type"].to_numpy()
    
    if "erds" in feature_groups:
        logger.info("Extracting ERDS features...")
        df, cols, qc = extract_erds_from_precomputed(precomputed, bands)
        if not df.empty:
            result.features["erds"] = FeatureSet(df, cols, "erds")
            result.qc["erds"] = qc
        else:
            result.qc["erds"] = {"skipped_reason": "empty_result"}

    if "spectral" in feature_groups:
        logger.info("Extracting spectral power features...")
        n_jobs_spectral = int(config.get("feature_engineering.parallel.n_jobs_bands", -1))
        df, cols, qc = extract_power_from_precomputed(precomputed, bands)
        extra_df, extra_cols = extract_spectral_extras_from_precomputed(precomputed, bands, n_jobs=n_jobs_spectral)
        seg_df, seg_cols = extract_segment_power_from_precomputed(precomputed, bands, n_jobs=n_jobs_spectral)
        if not extra_df.empty:
            df = pd.concat([df, extra_df], axis=1) if not df.empty else extra_df
            cols.extend(extra_cols)
        if not seg_df.empty:
            df = pd.concat([df, seg_df], axis=1) if not df.empty else seg_df
            cols.extend(seg_cols)
        if not df.empty:
            result.features["spectral"] = FeatureSet(df, cols, "spectral")
            result.qc["spectral"] = qc
        else:
            result.qc["spectral"] = {"skipped_reason": "empty_result"}

    if "aperiodic" in feature_groups:
        logger.info("Extracting aperiodic features...")
        baseline_window = getattr(precomputed.windows, "baseline_range", None)
        if baseline_window is None or np.isnan(baseline_window).any():
             baseline_window = get_config_value(
                 config, "time_frequency_analysis.baseline_window", [-3.0, -0.5]
             )
             baseline_window = tuple(baseline_window)
        
        df, cols, qc = extract_aperiodic_features_from_epochs(
            epochs,
            baseline_window,
            bands,
            config,
            logger,
            events_df=events_df,
        )
        if not df.empty:
            result.features["aperiodic"] = FeatureSet(df, cols, "aperiodic")
            result.qc["aperiodic"] = qc
        else:
            result.qc["aperiodic"] = {"skipped_reason": "empty_result"}

    if "connectivity" in feature_groups:
        logger.info("Extracting connectivity features (PLV) from precomputed phases...")
        conn_df, conn_cols = extract_connectivity_from_precomputed(precomputed)
        if not conn_df.empty:
            result.features["connectivity"] = FeatureSet(conn_df, conn_cols, "connectivity")
        else:
            result.qc["connectivity"] = {"skipped_reason": "empty_result"}

    if "gfp" in feature_groups:
        logger.info("Extracting GFP features...")
        gfp_df, gfp_cols = extract_gfp_from_precomputed(precomputed, config)
        if not gfp_df.empty:
            result.features["gfp"] = FeatureSet(gfp_df, gfp_cols, "gfp")
        else:
            result.qc["gfp"] = {"skipped_reason": "empty_result"}

    if "roi" in feature_groups:
        logger.info("Extracting ROI-aggregated features...")
        roi_df, roi_cols = extract_roi_features_from_precomputed(precomputed, bands, config)
        if not roi_df.empty:
            result.features["roi"] = FeatureSet(roi_df, roi_cols, "roi")
        else:
            result.qc["roi"] = {"skipped_reason": "empty_result"}

    if "temporal" in feature_groups:
        logger.info("Extracting temporal (time-domain) features...")
        tmp_df, tmp_cols = extract_temporal_features_from_precomputed(precomputed, config)
        if not tmp_df.empty:
            result.features["temporal"] = FeatureSet(tmp_df, tmp_cols, "temporal")
        else:
            result.qc["temporal"] = {"skipped_reason": "empty_result"}

    if "ratios" in feature_groups:
        logger.info("Extracting band ratio features...")
        ratio_df, ratio_cols = extract_band_ratios_from_precomputed(precomputed, config)
        if not ratio_df.empty:
            result.features["ratios"] = FeatureSet(ratio_df, ratio_cols, "ratios")
        else:
            result.qc["ratios"] = {"skipped_reason": "empty_result"}

    if "microstates" in feature_groups:
        logger.info("Extracting microstate features...")
        n_states = int(config.get("feature_engineering.microstates.n_states", 4))
        ms_df, ms_cols, _ = extract_microstate_features_from_epochs(
            epochs,
            n_states,
            config,
            logger,
        )
        if not ms_df.empty:
            result.features["microstates"] = FeatureSet(ms_df, ms_cols, "microstates")
        else:
            result.qc["microstates"] = {"skipped_reason": "empty_result"}

    if "pac" in feature_groups:
        logger.info("Extracting PAC features from precomputed analytic signals...")
        pac_df, pac_cols = extract_pac_from_precomputed(precomputed, config)
        if not pac_df.empty:
            result.features["pac"] = FeatureSet(pac_df, pac_cols, "pac")
        else:
            result.qc["pac"] = {"skipped_reason": "empty_result"}

    if "cfc" in feature_groups:
        logger.info("Extracting CFC features...")
        cfc_df, cfc_cols = extract_all_cfc_features(
            epochs,
            config,
            logger,
            include_pac=True,
            include_aac=True,
            include_ppc=True,
        )
        if not cfc_df.empty:
            result.features["cfc"] = FeatureSet(cfc_df, cfc_cols, "cfc")
        else:
            result.qc["cfc"] = {"skipped_reason": "empty_result"}

    if "dynamics_advanced" in feature_groups:
        logger.info("Extracting advanced dynamics from precomputed data...")
        n_jobs_dynamics = int(config.get("feature_engineering.parallel.n_jobs_temporal", -1))
        dyn_df, dyn_cols = extract_dynamics_from_precomputed(precomputed, n_jobs=n_jobs_dynamics)
        if not dyn_df.empty:
            result.features["dynamics_advanced"] = FeatureSet(dyn_df, dyn_cols, "dynamics_advanced")
        else:
            result.qc["dynamics_advanced"] = {"skipped_reason": "empty_result"}

    if "itpc" in feature_groups:
        logger.info("Extracting ITPC from precomputed phases...")
        itpc_df, itpc_cols = extract_itpc_from_precomputed(precomputed)
        if not itpc_df.empty:
            result.features["itpc"] = FeatureSet(itpc_df, itpc_cols, "itpc")
        else:
            result.qc["itpc"] = {"skipped_reason": "empty_result"}

    if "asymmetry" in feature_groups:
        logger.info("Computing hemispheric asymmetry features...")
        n_jobs_asym = int(config.get("feature_engineering.parallel.n_jobs_bands", -1))
        asym_df, asym_cols = extract_asymmetry_from_precomputed(precomputed, n_jobs=n_jobs_asym)
        if not asym_df.empty:
            result.features["asymmetry"] = FeatureSet(asym_df, asym_cols, "asymmetry")
        else:
            result.qc["asymmetry"] = {"skipped_reason": "empty_result"}

    if "complexity" in feature_groups:
        logger.info("Computing complexity metrics (LZC, entropy, DFA-lite)...")
        n_jobs_complexity = int(config.get("feature_engineering.parallel.n_jobs_complexity", -1))
        comp_df, comp_cols = extract_complexity_from_precomputed(precomputed, n_jobs=n_jobs_complexity)
        if not comp_df.empty:
            result.features["complexity"] = FeatureSet(comp_df, comp_cols, "complexity")
        else:
            result.qc["complexity"] = {"skipped_reason": "empty_result"}

    if "quality" in feature_groups:
        logger.info("Computing trial-level quality metrics...")
        qual_df = compute_trial_quality_metrics(epochs, config, logger)
        qual_cols = list(qual_df.columns)
        if not qual_df.empty:
            result.features["quality"] = FeatureSet(qual_df, qual_cols, "quality")
        else:
            result.qc["quality"] = {"skipped_reason": "empty_result"}

    return result


def extract_fmri_prediction_features(
    epochs: "mne.Epochs",
    config: Any,
    logger: Any,
    events_df: Optional[pd.DataFrame] = None,
) -> ExtractionResult:
    """
    Extract a subset of features optimized for fMRI prediction.
    
    Typically includes:
    - Spectral power (alpha, beta, theta, gamma)
    - Aperiodic components (exponent, offset)
    - Global Field Power (GFP)
    """
    feature_groups = ["spectral", "aperiodic", "erds"]
    bands = get_frequency_band_names(config)
    
    return extract_precomputed_features(
        epochs,
        bands,
        config,
        logger,
        feature_groups=feature_groups,
        events_df=events_df,
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

        save_trial_alignment_manifest(aligned_events, epochs, features_dir / "trial_alignment.tsv", self.config, self.logger)

        constants = {"TARGET_COLUMNS": self.config.get("event_columns.rating", [])}
        target_col = _pick_target_column(aligned_events, constants=constants)
        if target_col is None:
            self.logger.warning("No target column found; skipping")
            return

        y = pd.to_numeric(aligned_events[target_col], errors="coerce")

        fixed_templates = None
        fixed_template_ch_names = None
        if fixed_templates_path and fixed_templates_path.exists():
            try:
                data = np.load(fixed_templates_path)
                fixed_templates = data["templates"]
                fixed_template_ch_names = data.get("ch_names")
                self.logger.info(f"Loaded fixed templates from {fixed_templates_path}")
            except Exception as e:
                self.logger.error(f"Failed to load templates: {e}")

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
        if precomputed_result is not None and getattr(precomputed_result, "qc", None):
            feature_qc["precomputed"] = precomputed_result.qc
        if ctx.precomputed is not None and getattr(ctx.precomputed, "qc", None) is not None:
            try:
                feature_qc["precomputed_intermediates"] = asdict(ctx.precomputed.qc)
            except Exception:
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


__all__ = [
    "FeaturePipeline",
    "extract_all_features",
    "extract_precomputed_features",
    "extract_fmri_prediction_features",
    "FeatureExtractionResult",
    "ExtractionResult",
    "FeatureSet",
]
