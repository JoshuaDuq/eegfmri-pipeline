from __future__ import annotations

from typing import Any, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import mne

from eeg_pipeline.context.features import FeatureContext
from eeg_pipeline.analysis.features.selection import resolve_feature_categories
from eeg_pipeline.types import PrecomputedData
from eeg_pipeline.utils.config.loader import get_frequency_band_names
from eeg_pipeline.utils.validation import validate_epochs
from eeg_pipeline.utils.progress import PipelineProgress
from eeg_pipeline.analysis.features.results import (
    FeatureSet,
    ExtractionResult,
    FeatureExtractionResult,
    combine_feature_groups,
)
from eeg_pipeline.analysis.features.precomputed.erds import extract_erds_from_precomputed
from eeg_pipeline.analysis.features.precomputed.spectral import extract_power_from_precomputed
from eeg_pipeline.analysis.features.precomputed.extras import (
    extract_band_ratios_from_precomputed,
    extract_asymmetry_from_precomputed,
)
from eeg_pipeline.analysis.features.spectral import extract_spectral_features
from eeg_pipeline.analysis.features.phase import (
    compute_pac_comodulograms,
    extract_itpc_from_precomputed,
    extract_phase_features,
    extract_pac_from_precomputed,
)
from eeg_pipeline.analysis.features.aperiodic import extract_aperiodic_features
from eeg_pipeline.analysis.features.complexity import extract_complexity_from_precomputed
from eeg_pipeline.analysis.features.erp import extract_erp_features
from eeg_pipeline.analysis.features.bursts import extract_burst_features
from eeg_pipeline.analysis.features.connectivity import extract_connectivity_features
from eeg_pipeline.analysis.features.power import extract_power_features
from eeg_pipeline.analysis.features.preparation import precompute_data
from eeg_pipeline.analysis.features.quality import (
    compute_trial_quality_metrics,
    extract_quality_features,
)
from eeg_pipeline.analysis.features.temporal import extract_temporal_features
from eeg_pipeline.utils.analysis.tfr import (
    compute_tfr_for_subject,
    get_tfr_config,
    compute_complex_tfr,
    save_tfr_with_sidecar,
)


def filter_features_by_spatial_modes(
    df: Optional[pd.DataFrame],
    spatial_modes: List[str],
    config: Any,
) -> Optional[pd.DataFrame]:
    """
    Filter feature DataFrame columns to only include those matching spatial_modes.
    
    Feature columns are identified by naming patterns:
    - '_ch_': per-channel features (include if 'channels' in spatial_modes)
    - '_global_' or '_global': global features (include if 'global' in spatial_modes)
    - '_roi_' or ROI name patterns: ROI features (include if 'roi' in spatial_modes)
    
    Non-spatial features are always included.
    """
    if not spatial_modes or df is None or getattr(df, 'empty', True):
        return df
    
    from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
    
    # Get ROI names for pattern matching
    roi_defs = get_roi_definitions(config)
    roi_names = list(roi_defs.keys()) if roi_defs else []
    
    cols_to_keep: List[str] = []
    
    for col in df.columns:
        col_str = str(col)
        
        # Check if column is spatial (per-channel, global, or ROI)
        is_per_channel = '_ch_' in col_str
        is_channel_pair = '_chpair_' in col_str
        is_global = '_global_' in col_str or col_str.endswith('_global')
        is_roi = '_roi_' in col_str or any(f'_{roi}_' in col_str or col_str.endswith(f'_{roi}') for roi in roi_names)
        
        # If not a spatial column, always keep it
        if not is_per_channel and not is_channel_pair and not is_global and not is_roi:
            cols_to_keep.append(col)
            continue
        
        # Check if this spatial type is in the selected modes
        if is_per_channel and 'channels' in spatial_modes:
            cols_to_keep.append(col)
        elif is_channel_pair and 'channels' in spatial_modes:
            cols_to_keep.append(col)
        elif is_global and 'global' in spatial_modes:
            cols_to_keep.append(col)
        elif is_roi and 'roi' in spatial_modes:
            cols_to_keep.append(col)
    
    return df[cols_to_keep] if cols_to_keep else pd.DataFrame()


def extract_all_features(
    ctx: FeatureContext,
) -> FeatureExtractionResult:
    from eeg_pipeline.utils.analysis.spatial import crop_epochs_to_time_range
    
    # Store time range but don't crop epochs immediately to avoid wavelet length issues in TFR
    tmin, tmax = ctx.tmin, ctx.tmax
    if tmin is not None or tmax is not None:
        ctx.logger.info(f"Feature extraction restricted to time range: tmin={tmin}, tmax={tmax}")
    
    # Log spatial modes being used
    if ctx.spatial_modes:
        ctx.logger.info(f"Spatial aggregation modes: {', '.join(ctx.spatial_modes)}")
    
    power_bands = ctx.bands if ctx.bands else get_frequency_band_names(ctx.config)
    
    # We will compute counts based on the intended range if possible
    # but some extractors use ctx.epochs directly
    if tmin is not None or tmax is not None:
        # For non-TFR features (Connectivity, Microstates, etc.), we use cropped epochs
        # to ensure they only see the relevant window.
        working_epochs = crop_epochs_to_time_range(ctx.epochs, tmin, tmax, ctx.logger)
    else:
        working_epochs = ctx.epochs
        
    expected_n_trials = len(working_epochs)

    # Categories that need precomputed intermediate data
    precompute_categories = {
        "connectivity", "erds", "ratios", "asymmetry", "complexity", "bursts",
    }
    # Categories that require baseline for normalization - do NOT crop precomputed data for these
    baseline_dependent_categories = {"erds", "bursts"}
    
    precomputed_data = None
    needs_precompute = bool(precompute_categories & set(ctx.feature_categories))
    if needs_precompute:
        if ctx.precomputed is not None:
            precomputed_data = ctx.precomputed
            # Only crop precomputed data for features that DON'T need baseline normalization
            # For ERDS/spectral, keep full data so baseline is always available
            needs_baseline = bool(baseline_dependent_categories & set(ctx.feature_categories))
            if (tmin is not None or tmax is not None) and not needs_baseline:
                ctx.logger.info(f"Cropping precomputed intermediates to range [{tmin}, {tmax}]")
                precomputed_data = precomputed_data.crop(tmin, tmax)
            elif needs_baseline and (tmin is not None or tmax is not None):
                ctx.logger.info(f"Keeping full precomputed data for baseline normalization (target range: [{tmin}, {tmax}])")
        else:
            if working_epochs is None:
                raise ValueError("Missing epochs; cannot precompute feature intermediates.")
            if not working_epochs.preload:
                ctx.logger.info("Preloading epochs data...")
                working_epochs.load_data()
            relevant_cats = sorted(list(precompute_categories & set(ctx.feature_categories)))
            ctx.logger.info(f"Computing shared intermediate data for: {', '.join(relevant_cats)}...")
            compute_psd_data = bool(
                any(c in ctx.feature_categories for c in ["erds", "aperiodic", "spectral"])
            )
            precomputed_data = precompute_data(
                working_epochs,
                power_bands,
                ctx.config,
                ctx.logger,
                windows_spec=ctx.windows,
                compute_psd_data=compute_psd_data,
            )
            ctx.set_precomputed(precomputed_data)

    if precomputed_data is not None and ctx.windows is not None:
        aligned_windows = None
        try:
            ctx_times = getattr(ctx.windows, "times", None)
            if ctx_times is not None and len(precomputed_data.times) == len(ctx_times) and np.allclose(
                precomputed_data.times,
                ctx_times,
                atol=0,
                rtol=0,
            ):
                aligned_windows = ctx.windows
            else:
                ranges = getattr(ctx.windows, "ranges", None)
                if ranges:
                    explicit = [
                        {"name": name, "tmin": rng[0], "tmax": rng[1]}
                        for name, rng in ranges.items()
                        if isinstance(rng, (list, tuple)) and len(rng) >= 2
                    ]
                else:
                    explicit = None

                if explicit:
                    from eeg_pipeline.utils.analysis.windowing import (
                        TimeWindowSpec,
                        time_windows_from_spec,
                    )

                    spec = TimeWindowSpec(
                        times=precomputed_data.times,
                        config=ctx.config,
                        sampling_rate=float(getattr(precomputed_data, "sfreq", 1.0)),
                        logger=ctx.logger,
                        name=ctx.name,
                        explicit_windows=explicit,
                    )
                    aligned_windows = time_windows_from_spec(spec, logger=ctx.logger, strict=False)
        except Exception as exc:
            ctx.logger.warning("Failed to align precomputed windows with context: %s", exc)

        if aligned_windows is not None:
            precomputed_data = precomputed_data.with_windows(aligned_windows)
            ctx.set_precomputed(precomputed_data)

    if precomputed_data is not None:
        precomputed_data.spatial_modes = list(ctx.spatial_modes) if ctx.spatial_modes else None

    def _check_length(name: str, df: Optional[pd.DataFrame]) -> None:
        if df is None or getattr(df, "empty", False):
            return
        if len(df) != expected_n_trials:
            raise ValueError(f"{name} length mismatch: {len(df)} vs {expected_n_trials}")


    validation = validate_epochs(working_epochs, ctx.config, logger=ctx.logger)
    if not validation.valid:
        ctx.logger.warning(f"Validation issues: {validation.issues}")
        if validation.critical:
            raise ValueError(f"Critical errors: {validation.critical}")

    results = FeatureExtractionResult()

    tfr_dependent_categories = {"power", "itpc", "pac", "temporal"}
    needs_tfr = bool(tfr_dependent_categories & set(ctx.feature_categories))

    tfr_complex = None
    tfr_power = None
    tfr = None
    baseline_df = None
    baseline_cols: List[str] = []
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
                
                # CRITICAL: We DO NOT crop here yet, because compute_tfr_for_subject 
                # needs the full time range to find and validate the baseline window.
                # We will crop the final TFR object after baseline extraction.

        if tfr_power is None and ctx.tfr is not None:
             ctx.logger.info("Using pre-computed TFR from context")
             tfr_power = ctx.tfr

        # CRITICAL: Compute TFR on FULL epochs to avoid "wavelet longer than signal" errors,
        # then we will crop the TFR result if a time range is requested.
        baseline_override = None
        if ctx.windows is not None:
            b_start_override, b_end_override = ctx.windows.baseline_range
            if np.isfinite(b_start_override) and np.isfinite(b_end_override):
                baseline_override = (float(b_start_override), float(b_end_override))

        tfr, baseline_df, baseline_cols, b_start, b_end = compute_tfr_for_subject(
            ctx.epochs, # Full epochs
            ctx.aligned_events,
            ctx.subject,
            ctx.task,
            ctx.config,
            ctx.deriv_root,
            ctx.logger,
            tfr_computed=tfr_power,
            baseline_window=baseline_override,
        )
        
        if tfr is None:
            raise ValueError("TFR computation failed; aborting feature extraction.")
            
        # Crop TFR to the requested range if needed
        if tmin is not None or tmax is not None:
             # Safety check: swap if needed (defensive against out-of-order parameters)
             ctmin, ctmax = tmin, tmax
             if ctmin is not None and ctmax is not None and ctmin > ctmax:
                 ctmin, ctmax = ctmax, ctmin
             
             # Safety check: clamp to available times in TFR to avoid MNE errors
             ctmin = max(ctmin, tfr.times[0]) if ctmin is not None else tfr.times[0]
             ctmax = min(ctmax, tfr.times[-1]) if ctmax is not None else tfr.times[-1]
             
             if ctmin < ctmax:
                 ctx.logger.info(f"Cropping TFR to range [{ctmin:.3f}, {ctmax:.3f}]")
                 tfr = tfr.copy().crop(ctmin, ctmax)
             else:
                 ctx.logger.warning(f"Requested TFR range [{tmin}, {tmax}] is invalid or outside available data; skipping crop.")
        results.tfr = tfr
        results.baseline_df = baseline_df
        results.baseline_cols = baseline_cols

        ctx.results["tfr"] = tfr
        ctx.tfr = tfr
        ctx.baseline_df = baseline_df
        ctx.baseline_cols = baseline_cols

        if ctx.config.get("feature_engineering.save_tfr_with_sidecar", False):
            try:
                if tfr is not None and b_start is not None and b_end is not None:
                    baseline_mode = str(ctx.config.get("time_frequency_analysis.baseline_mode", "logratio"))
                    tfr_to_save = tfr.copy()
                    tfr_to_save.apply_baseline(baseline=(b_start, b_end), mode=baseline_mode)
                    tfr_to_save.comment = (
                        f"BASELINED:mode={baseline_mode};win=({b_start:.3f},{b_end:.3f})"
                    )
                    tfr_out = (
                        ctx.deriv_root
                        / f"sub-{ctx.subject}"
                        / "eeg"
                        / f"sub-{ctx.subject}_task-{ctx.task}_power_epo-tfr.h5"
                    )
                    save_tfr_with_sidecar(
                        tfr_to_save,
                        tfr_out,
                        (b_start, b_end),
                        baseline_mode,
                        ctx.logger,
                        ctx.config,
                    )
            except Exception as exc:
                ctx.logger.warning("Failed to save baselined TFR with sidecar: %s", exc)
    else:
        ctx.logger.info("Skipping TFR computation (not needed for requested feature categories)")

    progress = PipelineProgress(total=len(ctx.feature_categories), logger=ctx.logger, desc="Features")
    progress.start()

    if "power" in ctx.feature_categories:
        progress.step(message="Extracting power features...")
        pow_df, pow_cols = extract_power_features(ctx, power_bands)
        if pow_df is not None and not pow_df.empty and len(pow_df) != expected_n_trials:
            raise ValueError(f"Power length mismatch: {len(pow_df)} vs {expected_n_trials}")
        results.pow_df = pow_df
        results.pow_cols = pow_cols

    if "connectivity" in ctx.feature_categories:
        progress.step(message="Extracting connectivity features...")
        conn_df, conn_cols = extract_connectivity_features(ctx, power_bands)
        if conn_df is not None and not conn_df.empty and len(conn_df) != expected_n_trials:
            raise ValueError(f"Connectivity length mismatch: {len(conn_df)} vs {expected_n_trials}")
        results.conn_df = conn_df
        results.conn_cols = conn_cols

    if "aperiodic" in ctx.feature_categories:
        progress.step(message="Extracting aperiodic features...")
        aper_df, aper_cols, qc_payload = extract_aperiodic_features(ctx, power_bands)
        _check_length("Aperiodic", aper_df)
        results.aper_df = aper_df
        results.aper_cols = aper_cols
        results.aper_qc = qc_payload

    if "erp" in ctx.feature_categories:
        progress.step(message="Extracting ERP/LEP features...")
        erp_df, erp_cols = extract_erp_features(ctx)
        _check_length("ERP", erp_df)
        results.erp_df = erp_df
        results.erp_cols = erp_cols

    if "complexity" in ctx.feature_categories:
        progress.step(message="Extracting complexity features...")
        if precomputed_data is not None:
            comp_df, comp_cols = extract_complexity_from_precomputed(precomputed_data)
            _check_length("Complexity", comp_df)
            results.comp_df = comp_df
            results.comp_cols = comp_cols

    if "bursts" in ctx.feature_categories:
        progress.step(message="Extracting burst features...")
        bursts_df, bursts_cols = extract_burst_features(ctx, power_bands)
        _check_length("Bursts", bursts_df)
        results.bursts_df = bursts_df
        results.bursts_cols = bursts_cols

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

            segment_window = None
            segment_label = ctx.name or getattr(ctx.windows, "name", None) or "active"
            if ctx.windows is not None:
                if ctx.name and ctx.windows.ranges.get(ctx.name) is not None:
                    seg_range = ctx.windows.ranges.get(ctx.name)
                else:
                    seg_range = ctx.windows.ranges.get("plateau") or ctx.windows.ranges.get("active")
                    if seg_range is None and ctx.windows.active_range is not None:
                        ar = ctx.windows.active_range
                        if np.isfinite(ar[0]) and np.isfinite(ar[1]):
                            seg_range = ar
                if seg_range is not None:
                    segment_window = (float(seg_range[0]), float(seg_range[1]))

            pac_df, pac_phase_freqs, pac_amp_freqs, pac_trials_df, pac_time_df = compute_pac_comodulograms(
                tfr_complex,
                freqs,
                tfr_complex.times,
                ctx.epochs.info,
                ctx.config,
                ctx.logger,
                segment_name=segment_label,
                segment_window=segment_window,
                spatial_modes=ctx.spatial_modes,
            )
            _check_length("PAC trials", pac_trials_df)
            _check_length("PAC time-resolved", pac_time_df)
            results.pac_df = pac_df
            results.pac_phase_freqs = pac_phase_freqs
            results.pac_amp_freqs = pac_amp_freqs
            results.pac_trials_df = pac_trials_df
            results.pac_time_df = pac_time_df

    # ERDS - Event-related (de)synchronization
    if "erds" in ctx.feature_categories:
        progress.step(message="Extracting ERDS features...")
        if precomputed_data is not None:
            erds_df, erds_cols, erds_qc = extract_erds_from_precomputed(precomputed_data, power_bands)
            _check_length("ERDS", erds_df)
            if not erds_df.empty:
                results.erds_df = erds_df
                results.erds_cols = erds_cols

    # Spectral - Peak frequency, IAF, spectral edge
    if "spectral" in ctx.feature_categories:
        progress.step(message="Extracting spectral features (IAF)...")
        spectral_df, spectral_cols = extract_spectral_features(ctx, power_bands)
        _check_length("Spectral", spectral_df)
        if not spectral_df.empty:
            results.spectral_df = spectral_df
            results.spectral_cols = spectral_cols

    # Ratios - band power ratios
    if "ratios" in ctx.feature_categories:
        progress.step(message="Extracting band ratio features...")
        if precomputed_data is not None:
            ratio_df, ratio_cols = extract_band_ratios_from_precomputed(precomputed_data, ctx.config)
            _check_length("Ratios", ratio_df)
            if not ratio_df.empty:
                results.ratios_df = ratio_df
                results.ratios_cols = ratio_cols

    # Asymmetry - hemispheric asymmetry
    if "asymmetry" in ctx.feature_categories:
        progress.step(message="Extracting asymmetry features...")
        if precomputed_data is not None:
            n_jobs = int(ctx.config.get("feature_engineering.parallel.n_jobs_bands", -1))
            asym_df, asym_cols = extract_asymmetry_from_precomputed(precomputed_data, n_jobs=n_jobs)
            _check_length("Asymmetry", asym_df)
            if not asym_df.empty:
                results.asymmetry_df = asym_df
                results.asymmetry_cols = asym_cols

    # Quality metrics
    if "quality" in ctx.feature_categories:
        progress.step(message="Computing trial quality metrics...")
        qual_df, qual_cols = extract_quality_features(ctx)
        _check_length("Quality metrics", qual_df)
        results.quality_df = qual_df
        results.quality_cols = qual_cols

    # Temporal features (binned)
    if "temporal" in ctx.feature_categories:
        progress.step(message="Extracting temporal binned features...")
        temp_df, temp_cols = extract_temporal_features(ctx, power_bands)
        _check_length("Temporal", temp_df)
        results.temp_df = temp_df
        results.temp_cols = temp_cols

    # Apply spatial mode filtering to all feature DataFrames
    if ctx.spatial_modes:
        ctx.logger.info(f"Filtering features by spatial modes: {ctx.spatial_modes}")
        if results.pow_df is not None:
            results.pow_df = filter_features_by_spatial_modes(results.pow_df, ctx.spatial_modes, ctx.config)
            results.pow_cols = list(results.pow_df.columns) if results.pow_df is not None else []
        if results.erp_df is not None:
            results.erp_df = filter_features_by_spatial_modes(results.erp_df, ctx.spatial_modes, ctx.config)
            results.erp_cols = list(results.erp_df.columns) if results.erp_df is not None else []
        if results.aper_df is not None:
            results.aper_df = filter_features_by_spatial_modes(results.aper_df, ctx.spatial_modes, ctx.config)
            results.aper_cols = list(results.aper_df.columns) if results.aper_df is not None else []
        if results.conn_df is not None:
            results.conn_df = filter_features_by_spatial_modes(results.conn_df, ctx.spatial_modes, ctx.config)
            results.conn_cols = list(results.conn_df.columns) if results.conn_df is not None else []
        if results.comp_df is not None:
            results.comp_df = filter_features_by_spatial_modes(results.comp_df, ctx.spatial_modes, ctx.config)
            results.comp_cols = list(results.comp_df.columns) if results.comp_df is not None else []
        if results.bursts_df is not None:
            results.bursts_df = filter_features_by_spatial_modes(results.bursts_df, ctx.spatial_modes, ctx.config)
            results.bursts_cols = list(results.bursts_df.columns) if results.bursts_df is not None else []
        if results.spectral_df is not None:
            results.spectral_df = filter_features_by_spatial_modes(results.spectral_df, ctx.spatial_modes, ctx.config)
            results.spectral_cols = list(results.spectral_df.columns) if results.spectral_df is not None else []
        if results.temp_df is not None:
            results.temp_df = filter_features_by_spatial_modes(results.temp_df, ctx.spatial_modes, ctx.config)
            results.temp_cols = list(results.temp_df.columns) if results.temp_df is not None else []

    progress.finish()
    return results


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
    from eeg_pipeline.utils.config.loader import get_config_value

    if feature_groups is None:
        feature_groups = ["erds", "spectral"]

    compute_bands = any(
        g in feature_groups
        for g in [
            "erds",
            "spectral",
            "connectivity",
            "pac",
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
        df, cols, qc = extract_power_from_precomputed(precomputed, bands)
        if not df.empty:
            result.features["spectral"] = FeatureSet(df, cols, "spectral")
            result.qc["spectral"] = qc
        else:
            result.qc["spectral"] = {"skipped_reason": "empty_result"}

    if "aperiodic" in feature_groups:
        logger.info("Extracting aperiodic features...")
        baseline_window = getattr(precomputed.windows, "baseline_range", None)
        if baseline_window is None or np.isnan(baseline_window).any():
            baseline_window = get_config_value(config, "time_frequency_analysis.baseline_window", [-3.0, -0.5])
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


    if "ratios" in feature_groups:
        logger.info("Extracting band ratio features...")
        ratio_df, ratio_cols = extract_band_ratios_from_precomputed(precomputed, config)
        if not ratio_df.empty:
            result.features["ratios"] = FeatureSet(ratio_df, ratio_cols, "ratios")
        else:
            result.qc["ratios"] = {"skipped_reason": "empty_result"}


    if "pac" in feature_groups:
        logger.info("Extracting PAC features from precomputed analytic signals...")
        pac_df, pac_cols = extract_pac_from_precomputed(precomputed, config)
        if not pac_df.empty:
            result.features["pac"] = FeatureSet(pac_df, pac_cols, "pac")
        else:
            result.qc["pac"] = {"skipped_reason": "empty_result"}



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
        logger.info("Computing complexity metrics (LZC, permutation entropy)...")
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


__all__ = [
    "extract_all_features",
    "extract_precomputed_features",
    "extract_fmri_prediction_features",
    "resolve_feature_categories",
]
