from __future__ import annotations

# Standard library
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import argparse
import logging

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Third-party
import mne
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.cluster import KMeans

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    matplotlib = None
    plt = None
    sns = None

# Local - config and data
from eeg_pipeline.utils.config_loader import load_settings
from eeg_pipeline.utils.data_loading import (
    get_available_subjects,
    load_epochs_for_analysis,
    parse_subject_args,
    flatten_lower_triangles,
    align_feature_blocks,
    validate_trial_alignment_manifest,
    validate_feature_block_lengths,
    register_feature_block,
)
from eeg_pipeline.utils.io_utils import (
    _find_clean_epochs_path,
    _load_events_df,
    _pick_target_column,
    deriv_features_path,
    deriv_plots_path,
    deriv_stats_path,
    ensure_dir,
    find_first,
    get_band_color,
    get_group_logger,
    get_subject_logger,
    save_fig,
    setup_matplotlib,
)
from eeg_pipeline.utils.tfr_utils import (
    compute_adaptive_n_cycles,
    find_tfr_path,
    log_tfr_resolution,
    read_tfr_average_with_logratio,
    run_tfr_morlet,
    save_tfr_with_sidecar,
    validate_baseline_indices,
    validate_tfr_parameters,
)
from eeg_pipeline.utils.stats_utils import (
    validate_baseline_window_pre_stimulus,
)

# Local - analysis
from eeg_pipeline.analysis.feature_extraction import (
    compute_gfp as _compute_gfp,
    compute_microstate_duration_stats,
    compute_microstate_metric_correlations,
    compute_microstate_transition_stats,
    corr_maps as _corr_maps,
    extract_band_power_features,
    extract_baseline_power_features,
    extract_connectivity_features,
    extract_microstate_features,
    extract_templates_from_trials as _extract_templates_from_trials,
    label_timecourse as _label_timecourse,
    zscore_maps as _zscore_maps,
)

# Local - plotting
from eeg_pipeline.plotting.plot_features import (
    plot_channel_power_heatmap,
    plot_group_band_power_time_courses,
    plot_group_power_plots,
    plot_inter_band_spatial_power_correlation,
    plot_microstate_coverage_by_pain,
    plot_microstate_duration_distributions as _plot_microstate_duration_distributions,
    plot_microstate_gfp_by_temporal_bins,
    plot_microstate_gfp_colored_by_state,
    plot_microstate_pain_correlation_heatmap as _plot_microstate_pain_correlation_heatmap,
    plot_microstate_templates,
    plot_microstate_templates_by_pain,
    plot_microstate_templates_by_temperature,
    plot_microstate_temporal_evolution,
    plot_microstate_transition_network as _plot_microstate_transition_network,
    plot_power_distributions,
    plot_power_spectral_density,
    plot_power_spectral_density_by_pain,
    plot_power_time_course_by_temperature,
    plot_power_time_courses,
    plot_trial_power_variability,
)

###################################################################
# Helper Functions
###################################################################

def _setup_logging(subject: str, config=None):
    if config is None:
        config = load_settings()
    log_file_name = config.get("logging.file_names.feature_engineering", "03_feature_extraction.log")
    return get_subject_logger("feature_engineering", subject, log_file_name, config=config)




def _find_connectivity_arrays(subject: str, task: str, band: str, deriv_root: Path):
    def _find_measure(measure: str) -> Optional[Path]:
        search_paths = [
            deriv_root / "features" / f"sub-{subject}" / "ses-1" / "eeg",
            deriv_root / "features" / f"sub-{subject}" / "eeg",
            deriv_root / f"sub-{subject}" / "eeg",
        ]
        
        patterns = [
            f"sub-{subject}_task-{task}_ses-*_connectivity_{measure}_{band}*_all_trials.npy",
            f"sub-{subject}_task-{task}_*connectivity_{measure}_{band}*_all_trials.npy",
            f"sub-{subject}_task-{task}_connectivity_{measure}_{band}*_all_trials.npy",
        ]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
            for pattern in patterns:
                full_pattern = str((search_path / pattern).as_posix())
                path = find_first(full_pattern)
                if path:
                    return Path(path)
        return None
    
    return _find_measure("aec"), _find_measure("wpli")


def _load_labels(subject: str, task: str, deriv_root: Path) -> Optional[np.ndarray]:
    search_paths = [
        deriv_root / f"sub-{subject}" / "eeg",
        deriv_root / "features" / f"sub-{subject}" / "ses-1" / "eeg",
        deriv_root / "features" / f"sub-{subject}" / "eeg",
    ]
    
    patterns = [
        f"sub-{subject}_task-{task}_*connectivity_labels*.npy",
        f"sub-{subject}_task-{task}_connectivity_labels*.npy",
        f"sub-{subject}_task-{task}_ses-*_connectivity_labels*.npy",
    ]
    
    for search_path in search_paths:
        if not search_path.exists():
            continue
        for pattern in patterns:
            path = find_first(str((search_path / pattern).as_posix()))
            if path:
                return np.load(path, allow_pickle=True)
    return None








###################################################################
# Feature Extraction Helpers
###################################################################



def _save_dropped_trials_log(epochs, events_df, drop_log_path, logger):
    selection = getattr(epochs, "selection", None)
    if selection is None:
        raise AttributeError("epochs.selection missing")
    
    selection_arr = np.asarray(selection, dtype=int)
    valid_mask = (selection_arr >= 0) & (selection_arr < len(events_df))
    
    if not np.all(valid_mask):
        logger.warning("Epoch selection contains indices outside events range; restricting to valid entries")
        selection_arr = selection_arr[valid_mask]
    
    kept_indices = set(int(idx) for idx in selection_arr.tolist())
    dropped_indices = [idx for idx in range(len(events_df)) if idx not in kept_indices]

    if not dropped_indices:
        empty_df = pd.DataFrame(columns=["original_index", "drop_reason"])
        empty_df.to_csv(drop_log_path, sep="\t", index=False)
        logger.info("No dropped trials detected; wrote empty drop log to %s", drop_log_path)
        return

    dropped_events = events_df.iloc[dropped_indices].copy()
    drop_log = getattr(epochs, "drop_log", None)
    
    if isinstance(drop_log, (list, tuple)) and len(drop_log) == len(events_df):
        drop_reasons = [
            ";".join(str(x) for x in entry if x) if isinstance(entry, (list, tuple)) else str(entry) if entry else ""
            for idx in dropped_indices
            for entry in [drop_log[idx]]
        ]
    else:
        drop_reasons = [""] * len(dropped_indices)

    dropped_events.insert(0, "original_index", dropped_indices)
    dropped_events["drop_reason"] = drop_reasons
    dropped_events.to_csv(drop_log_path, sep="\t", index=False)
    logger.info("Saved drop log with %d dropped trials to %s", len(dropped_events), drop_log_path)


def _save_trial_alignment_manifest(aligned_events, epochs, manifest_path, config, logger):
    if aligned_events is None or len(aligned_events) == 0:
        raise ValueError("Cannot save trial alignment manifest: aligned_events is None or empty")
    
    if len(aligned_events) != len(epochs):
        raise ValueError(
            f"Cannot save trial alignment manifest: length mismatch "
            f"(aligned_events={len(aligned_events)}, epochs={len(epochs)})"
        )
    
    manifest = pd.DataFrame({"trial_index": np.arange(len(aligned_events), dtype=int)})
    
    if "sample" in aligned_events.columns:
        manifest["sample"] = aligned_events["sample"].values
    if "onset" in aligned_events.columns:
        manifest["onset"] = aligned_events["onset"].values
    
    constants = {"TARGET_COLUMNS": config.get("event_columns.rating", [])}
    target_col = _pick_target_column(aligned_events, constants=constants)
    if target_col is not None:
        manifest["target_value"] = pd.to_numeric(aligned_events[target_col], errors="coerce").values
    
    manifest.to_csv(manifest_path, sep="\t", index=False)
    logger.info("Saved trial alignment manifest with %d trials to %s", len(manifest), manifest_path)


def _prepare_tfr_with_baseline(epochs, aligned_events, subject, task, config, deriv_root, logger):
    freq_min = config.get("time_frequency_analysis.tfr.freq_min", 1.0)
    freq_max = config.get("time_frequency_analysis.tfr.freq_max", 100.0)
    n_freqs = config.get("time_frequency_analysis.tfr.n_freqs", 40)
    n_cycles_factor = config.get("time_frequency_analysis.tfr.n_cycles_factor", 2.0)
    tfr_decim = config.get("time_frequency_analysis.tfr.decim", 4)
    tfr_picks = config.get("time_frequency_analysis.tfr.picks", "eeg")
    
    freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
    n_cycles = compute_adaptive_n_cycles(freqs, cycles_factor=n_cycles_factor, config=config)
    
    tfr = run_tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        decim=tfr_decim,
        picks=tfr_picks,
        config=config,
        logger=logger,
    )
    if len(aligned_events) != len(tfr):
        raise ValueError(
            f"Alignment mismatch in TFR preparation: aligned_events has {len(aligned_events)} rows "
            f"but TFR has {len(tfr)} epochs. This indicates a critical alignment failure. "
            f"Expected aligned_events length ({len(aligned_events)}) to match TFR length ({len(tfr)})."
        )
    tfr.metadata = aligned_events.copy()
    
    times = np.asarray(tfr.times)
    tfr_baseline_raw = tuple(config.get("time_frequency_analysis.baseline_window", [-2.0, 0.0]))
    tfr_baseline = validate_baseline_window_pre_stimulus(
        tfr_baseline_raw, logger=logger
    )
    min_baseline_samples = int(config.get("tfr_topography_pipeline.min_baseline_samples", 5))
    b_start, b_end, _ = validate_baseline_indices(times, tfr_baseline, min_samples=min_baseline_samples)
    
    power_bands = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    
    tfr_comment = getattr(tfr, "comment", None)
    if isinstance(tfr_comment, str) and "BASELINED:" in tfr_comment:
        raise ValueError(
            f"TFR already baseline-corrected (comment: '{tfr_comment}'). "
            f"Cannot extract baseline features from corrected data."
        )
    
    baseline_df, baseline_cols = extract_baseline_power_features(
        tfr, power_bands, (b_start, b_end), config, logger
    )
    
    # Apply baseline correction AFTER extracting baseline features
    # This ensures baseline features represent absolute power, while post-stimulus features
    # are baseline-corrected (log-ratio) for proper comparison.
    tfr.apply_baseline(baseline=(b_start, b_end), mode="logratio")
    tfr.comment = f"BASELINED:mode=logratio;win=({b_start:.3f},{b_end:.3f})"
    
    save_cfg = config.get("feature_engineering.save_tfr_with_sidecar", False)
    if save_cfg:
        tfr_out = deriv_root / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_power_epo-tfr.h5"
        save_tfr_with_sidecar(
            tfr,
            tfr_out,
            baseline_window=(b_start, b_end),
            mode="logratio",
            logger=logger,
            config=config,
        )
    
    return tfr, baseline_df, baseline_cols


def _save_microstate_templates(epochs, templates, subject, n_states, deriv_root, logger):
    if templates is None:
        return
    
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(stats_dir)
    template_path = stats_dir / f"microstates_templates_K{n_states}.npz"
    
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    ch_names = [epochs.info["ch_names"][i] for i in picks]
    
    np.savez_compressed(
        template_path,
        templates=templates,
        ch_names=np.array(ch_names),
        n_states=n_states
    )
    logger.info(f"Saved microstate templates to {template_path}")






def _align_feature_dataframes(pow_df, baseline_df, conn_df, ms_df, y, aligned_events, features_dir, logger, config, initial_trial_count: Optional[int] = None):
    block_registry: Dict[str, pd.DataFrame] = {}
    before_lengths: Dict[str, int] = {}
    
    register_feature_block("power", pow_df if pow_df is not None and not pow_df.empty else None, block_registry, before_lengths)
    register_feature_block("baseline", baseline_df if baseline_df is not None and not baseline_df.empty else None, block_registry, before_lengths)
    register_feature_block("connectivity", conn_df if conn_df is not None and not conn_df.empty else None, block_registry, before_lengths)
    register_feature_block("microstates", ms_df if ms_df is not None and not ms_df.empty else None, block_registry, before_lengths)
    register_feature_block("target", y, block_registry, before_lengths)
    
    if not block_registry:
        logger.warning("No features extracted; skipping save")
        return None, None, None, None, None, None
    
    validate_feature_block_lengths(before_lengths, logger)
    
    if aligned_events is not None and len(aligned_events) > 0:
        validate_trial_alignment_manifest(aligned_events, features_dir, logger)
    
    logger.info(f"Validated feature block lengths ({', '.join(f'{k}={v}' for k, v in before_lengths.items())})")
    
    pow_df_aligned = block_registry.get("power")
    if pow_df_aligned is not None:
        pow_df_aligned = pow_df_aligned.reset_index(drop=True)
    else:
        pow_df_aligned = pow_df
    
    baseline_df_aligned = block_registry.get("baseline")
    if baseline_df_aligned is not None:
        baseline_df_aligned = baseline_df_aligned.reset_index(drop=True)
    else:
        baseline_df_aligned = baseline_df if baseline_df is not None else pd.DataFrame()
    
    conn_df_aligned = block_registry.get("connectivity")
    if conn_df_aligned is not None:
        conn_df_aligned = conn_df_aligned.reset_index(drop=True)
    else:
        conn_df_aligned = conn_df
    
    ms_df_aligned = block_registry.get("microstates")
    if ms_df_aligned is not None:
        ms_df_aligned = ms_df_aligned.reset_index(drop=True)
    else:
        ms_df_aligned = ms_df
    target_block = block_registry.get("target")
    if target_block is not None:
        y_aligned = target_block.iloc[:, 0] if target_block.shape[1] == 1 else target_block
    else:
        y_aligned = y
    
    def _get_length(df):
        if df is None:
            return 0
        if isinstance(df, pd.Series):
            return len(df) if not df.empty else 0
        if isinstance(df, pd.DataFrame):
            return len(df) if not df.empty else 0
        return 0
    
    after_lengths = {
        "power": _get_length(pow_df_aligned),
        "baseline": _get_length(baseline_df_aligned),
        "connectivity": _get_length(conn_df_aligned),
        "microstates": _get_length(ms_df_aligned),
        "target": _get_length(y_aligned),
    }
    
    retention_stats = {
        "initial_trial_count": initial_trial_count,
        "before_alignment": before_lengths,
        "after_alignment": after_lengths,
    }
    
    return pow_df_aligned, baseline_df_aligned, conn_df_aligned, ms_df_aligned, y_aligned, retention_stats


def _build_plateau_features(pow_df, pow_cols, baseline_df, baseline_cols, tfr, power_bands, logger):
    ch_names = tfr.info["ch_names"]
    col_name_to_series = {}
    plateau_cols = []

    for band in power_bands:
        for ch in ch_names:
            early_col = f"pow_{band}_{ch}_early"
            mid_col = f"pow_{band}_{ch}_mid"
            late_col = f"pow_{band}_{ch}_late"

            if early_col in pow_cols and mid_col in pow_cols and late_col in pow_cols:
                plateau_val = pow_df[[early_col, mid_col, late_col]].mean(axis=1)
                name = f"pow_{band}_{ch}"
                col_name_to_series[name] = plateau_val
                plateau_cols.append(name)

        if not baseline_df.empty:
            for ch in ch_names:
                baseline_col = f"baseline_{band}_{ch}"
                if baseline_col in baseline_cols:
                    col_name_to_series[baseline_col] = baseline_df[baseline_col]
                    plateau_cols.append(baseline_col)

    plateau_df = pd.DataFrame(col_name_to_series)
    plateau_df = plateau_df.reindex(columns=plateau_cols)
    return plateau_df, plateau_cols


def _save_all_features(
    pow_df, pow_cols,
    baseline_df, baseline_cols,
    conn_df, conn_cols,
    ms_df, ms_cols,
    plateau_df, plateau_cols,
    y, target_col,
    features_dir, logger, config
):
    direct_blocks = [pow_df]
    direct_cols = list(pow_cols)
    
    if not baseline_df.empty:
        direct_blocks.append(baseline_df)
        direct_cols.extend(baseline_cols)
    
    if ms_df is not None and not ms_df.empty:
        direct_blocks.append(ms_df)
        direct_cols.extend(ms_cols)
    
    direct_df = pd.concat(direct_blocks, axis=1)
    direct_df.columns = direct_cols
    
    eeg_direct_path = features_dir / "features_eeg_direct.tsv"
    eeg_direct_cols_path = features_dir / "features_eeg_direct_columns.tsv"
    logger.info(f"Saving direct EEG features: {eeg_direct_path}")
    direct_df.to_csv(eeg_direct_path, sep="\t", index=False)
    pd.Series(direct_cols, name="feature").to_csv(eeg_direct_cols_path, sep="\t", index=False)
    
    if not plateau_df.empty:
        plateau_path = features_dir / "features_eeg_plateau.tsv"
        plateau_cols_path = features_dir / "features_eeg_plateau_columns.tsv"
        logger.info(f"Saving plateau-averaged EEG features: {plateau_path}")
        plateau_df.to_csv(plateau_path, sep="\t", index=False)
        pd.Series(plateau_cols, name="feature").to_csv(plateau_cols_path, sep="\t", index=False)
    
    if conn_df is not None and not conn_df.empty:
        conn_path = features_dir / "features_connectivity.tsv"
        logger.info(f"Saving connectivity features: {conn_path}")
        if conn_cols:
            conn_df.columns = conn_cols
        conn_df.to_csv(conn_path, sep="\t", index=False)
    
    if ms_df is not None and not ms_df.empty:
        ms_path = features_dir / "features_microstates.tsv"
        logger.info(f"Saving microstate features: {ms_path}")
        if ms_cols:
            ms_df.columns = ms_cols
        ms_df.to_csv(ms_path, sep="\t", index=False)
    
    blocks = [direct_df]
    cols_all = list(direct_cols)
    if conn_df is not None and not conn_df.empty:
        blocks.append(conn_df)
        cols_all.extend(conn_cols)
    
    combined_df = pd.concat(blocks, axis=1)
    combined_df.columns = cols_all
    combined_path = features_dir / "features_all.tsv"
    logger.info(f"Saving combined features: {combined_path}")
    combined_df.to_csv(combined_path, sep="\t", index=False)
    
    y_path = features_dir / "target_vas_ratings.tsv"
    rating_columns = config.get("event_columns.rating", ["vas_rating"])
    target_column_name = rating_columns[0] if rating_columns else "vas_rating"
    logger.info(f"Saving behavioral target vector: {y_path} (column: {target_column_name})")
    y.to_frame(name=target_column_name).to_csv(y_path, sep="\t", index=False)
    
    return combined_df


def _generate_all_plots(tfr, pow_df, epochs, ms_templates, ms_df, aligned_events, subject, task, plots_dir, logger, config):
    logger.info(f"Generating power visualizations in: {plots_dir}")
    
    power_bands = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    n_microstates = int(config.get("feature_engineering.microstates.n_states", 4))
    
    plot_power_distributions(pow_df, power_bands, subject, plots_dir, logger, config)
    plot_channel_power_heatmap(pow_df, power_bands, subject, plots_dir, logger, config)
    plot_power_time_courses(tfr, power_bands, subject, plots_dir, logger, config)
    plot_trial_power_variability(pow_df, power_bands, subject, plots_dir, logger, config)
    plot_inter_band_spatial_power_correlation(tfr, subject, plots_dir, logger, config)
    plot_power_spectral_density(tfr, subject, plots_dir, logger, aligned_events, config)
    plot_power_spectral_density_by_pain(tfr, subject, plots_dir, logger, aligned_events, config)
    
    for band in power_bands:
        plot_power_time_course_by_temperature(tfr, subject, plots_dir, logger, aligned_events, band, config)
    
    logger.info("Successfully generated all power visualizations")
    
    has_ms_templates = ms_templates is not None and len(ms_templates) > 0
    has_ms_features = ms_df is not None and not ms_df.empty
    
    if not (has_ms_templates and has_ms_features):
        logger.info("Skipping microstate visualizations: templates or features missing")
        return
    
    logger.info("Generating microstate visualizations")
    
    try:
        picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
        if len(picks) == 0:
            logger.warning("No EEG channels available for microstate plotting; skipping microstate visualizations")
            return
        
        info_eeg = mne.pick_info(epochs.info, picks)
        
        if ms_templates is not None and len(ms_templates) > 0:
            try:
                plot_microstate_templates(ms_templates, info_eeg, subject, plots_dir, n_microstates, logger, config)
            except (OSError, ValueError) as e:
                logger.warning(f"Failed to plot microstate templates: {e}; skipping this visualization")
            except Exception as e:
                if HAS_MATPLOTLIB and matplotlib is not None:
                    if isinstance(e, Exception):
                        logger.warning(f"Failed to plot microstate templates: {e}; skipping this visualization")
                    else:
                        raise
                else:
                    raise
        
        if aligned_events is not None and not aligned_events.empty:
            try:
                plot_microstate_templates_by_pain(epochs, aligned_events, subject, task, plots_dir, n_microstates, logger, config)
            except Exception as e:
                logger.warning(f"Failed to plot microstate templates by pain: {e}; skipping this visualization")
            
            try:
                plot_microstate_templates_by_temperature(epochs, aligned_events, subject, task, plots_dir, n_microstates, logger, config)
            except Exception as e:
                logger.warning(f"Failed to plot microstate templates by temperature: {e}; skipping this visualization")
            
            try:
                plot_microstate_gfp_colored_by_state(epochs, ms_templates, aligned_events, subject, plots_dir, n_microstates, logger, config)
            except Exception as e:
                logger.warning(f"Failed to plot microstate GFP colored by state: {e}; skipping this visualization")
            
            try:
                plot_microstate_gfp_by_temporal_bins(epochs, ms_templates, aligned_events, subject, task, plots_dir, n_microstates, logger, config)
            except Exception as e:
                logger.warning(f"Failed to plot microstate GFP by temporal bins: {e}; skipping this visualization")
            
            try:
                plot_microstate_coverage_by_pain(ms_df, aligned_events, subject, plots_dir, n_microstates, logger, config)
            except Exception as e:
                logger.warning(f"Failed to plot microstate coverage by pain: {e}; skipping this visualization")
            
            try:
                corr_df, pval_df = compute_microstate_metric_correlations(ms_df, aligned_events, config=config)
                _plot_microstate_pain_correlation_heatmap(corr_df, pval_df, subject, plots_dir, logger, config)
            except Exception as e:
                logger.warning(f"Failed to plot microstate pain correlation heatmap: {e}; skipping this visualization")
            
            try:
                plot_microstate_temporal_evolution(epochs, ms_templates, aligned_events, subject, task, plots_dir, n_microstates, logger, config)
            except Exception as e:
                logger.warning(f"Failed to plot microstate temporal evolution: {e}; skipping this visualization")
            
            try:
                stats = compute_microstate_transition_stats(ms_df, aligned_events, n_states=n_microstates, config=config)
                _plot_microstate_transition_network(stats, subject, plots_dir, logger, config)
            except Exception as e:
                logger.warning(f"Failed to plot microstate transition network: {e}; skipping this visualization")
            
            try:
                stats = compute_microstate_duration_stats(ms_df, aligned_events, n_states=n_microstates, config=config)
                _plot_microstate_duration_distributions(stats, subject, plots_dir, logger, config)
            except Exception as e:
                logger.warning(f"Failed to plot microstate duration distributions: {e}; skipping this visualization")
        
        logger.info("Successfully generated all microstate visualizations")
    except (OSError, ValueError) as e:
        logger.warning(f"Microstate visualization failed: {e}; skipping all microstate plots for this subject")
    except Exception as e:
        logger.warning(f"Microstate visualization failed: {e}; skipping all microstate plots for this subject")


###################################################################
# Main Subject Processing
###################################################################

def _extract_all_features(epochs, aligned_events, subject, task, config, deriv_root, logger):
    power_bands = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    n_microstates = int(config.get("feature_engineering.microstates.n_states", 4))
    expected_n_trials = len(epochs)
    
    tfr, baseline_df, baseline_cols = _prepare_tfr_with_baseline(epochs, aligned_events, subject, task, config, deriv_root, logger)
    
    pow_df, pow_cols = extract_band_power_features(tfr, power_bands, config, logger)
    if pow_df is not None and len(pow_df) != expected_n_trials:
        error_msg = (
            f"Power feature extraction length mismatch: expected {expected_n_trials} trials "
            f"but got {len(pow_df)}. This indicates a critical extraction error."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    conn_df, conn_cols = extract_connectivity_features(subject, task, power_bands, deriv_root, logger)
    if conn_df is not None and len(conn_df) != expected_n_trials:
        error_msg = (
            f"Connectivity feature extraction length mismatch: expected {expected_n_trials} trials "
            f"but got {len(conn_df)}. This indicates a critical extraction error."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("Extracting microstate features")
    ms_df, ms_cols, ms_templates = extract_microstate_features(epochs, n_microstates, config, logger)
    if ms_df is not None and len(ms_df) != expected_n_trials:
        error_msg = (
            f"Microstate feature extraction length mismatch: expected {expected_n_trials} trials "
            f"but got {len(ms_df)}. This indicates a critical extraction error."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if baseline_df is not None and len(baseline_df) != expected_n_trials:
        error_msg = (
            f"Baseline feature extraction length mismatch: expected {expected_n_trials} trials "
            f"but got {len(baseline_df)}. This indicates a critical extraction error."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return tfr, pow_df, pow_cols, baseline_df, baseline_cols, conn_df, conn_cols, ms_df, ms_cols, ms_templates


def _align_and_save_features(pow_df, pow_cols, baseline_df, baseline_cols, conn_df, conn_cols, ms_df, ms_cols, y, target_col, features_dir, tfr, power_bands, aligned_events, logger, config, initial_trial_count: Optional[int] = None):
    result = _align_feature_dataframes(pow_df, baseline_df, conn_df, ms_df, y, aligned_events, features_dir, logger, config, initial_trial_count=initial_trial_count)
    if result is None or result[0] is None:
        return None
    pow_df, baseline_df, conn_df, ms_df, y, retention_stats = result
    
    if pow_cols:
        pow_df.columns = pow_cols
    if not baseline_df.empty and baseline_cols:
        baseline_df.columns = baseline_cols
    
    plateau_df, plateau_cols = _build_plateau_features(pow_df, pow_cols, baseline_df, baseline_cols, tfr, power_bands, logger)
    combined_df = _save_all_features(
        pow_df, pow_cols,
        baseline_df, baseline_cols,
        conn_df, conn_cols,
        ms_df, ms_cols,
        plateau_df, plateau_cols,
        y, target_col,
        features_dir, logger, config
    )
    return combined_df, pow_df, baseline_df, conn_df, ms_df, y, retention_stats


def process_subject(subject, task, deriv_root, config, no_plots=False):
    if not subject or not isinstance(subject, str):
        raise ValueError(f"subject must be non-empty string, got: {subject}")
    if not task or not isinstance(task, str):
        raise ValueError(f"task must be non-empty string, got: {task}")
    
    setup_matplotlib(config)
    logger = _setup_logging(subject, config)
    logger.info(f"=== Feature engineering: sub-{subject}, task-{task} ===")
    
    features_dir = deriv_features_path(deriv_root, subject)
    ensure_dir(features_dir)

    epochs, aligned_events = load_epochs_for_analysis(
        subject,
        task,
        align="strict",
        preload=False,
        deriv_root=deriv_root,
        logger=logger,
        config=config,
    )
    
    if epochs is None:
        logger.error(f"No cleaned epochs found for sub-{subject}; skipping")
        return
    
    if aligned_events is None:
        logger.warning("No events available for targets; skipping subject")
        return

    original_events_df = _load_events_df(subject, task, bids_root=config.bids_root, config=config)
    if original_events_df is None:
        logger.warning("Could not load original events TSV for dropped trials log; skipping drop log")
    else:
        drop_log_path = features_dir / "dropped_trials.tsv"
        _save_dropped_trials_log(epochs, original_events_df, drop_log_path, logger)
    
    manifest_path = features_dir / "trial_alignment.tsv"
    _save_trial_alignment_manifest(aligned_events, epochs, manifest_path, config, logger)

    constants = {"TARGET_COLUMNS": config.get("event_columns.rating", [])}
    target_col = _pick_target_column(aligned_events, constants=constants)
    if target_col is None:
        logger.warning("No suitable target column found in events; skipping")
        return
    
    y = pd.to_numeric(aligned_events[target_col], errors="coerce")
    
    tfr, pow_df, pow_cols, baseline_df, baseline_cols, conn_df, conn_cols, ms_df, ms_cols, ms_templates = _extract_all_features(epochs, aligned_events, subject, task, config, deriv_root, logger)
    
    power_bands = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    n_microstates = int(config.get("feature_engineering.microstates.n_states", 4))
    _save_microstate_templates(epochs, ms_templates, subject, n_microstates, deriv_root, logger)
    
    initial_trial_count = len(epochs) if epochs is not None else len(aligned_events) if aligned_events is not None else None
    
    alignment = _align_and_save_features(
        pow_df,
        pow_cols,
        baseline_df,
        baseline_cols,
        conn_df,
        conn_cols,
        ms_df,
        ms_cols,
        y,
        target_col,
        features_dir,
        tfr,
        power_bands,
        aligned_events,
        logger,
        config,
        initial_trial_count=initial_trial_count,
    )
    if alignment is None:
        return
    combined_df, pow_df_aligned, baseline_df_aligned, conn_df_aligned, ms_df_aligned, y_aligned, retention_stats = alignment

    _write_qc_summary(features_dir, pow_df_aligned, baseline_df_aligned, conn_df_aligned, ms_df_aligned, y_aligned, logger)
    
    _save_trial_retention_tsv(retention_stats, subject, features_dir, logger)

    enable_plots = config.get("feature_engineering.enable_plots", True) and not no_plots
    if enable_plots and HAS_MATPLOTLIB:
        plots_dir = deriv_plots_path(deriv_root, subject, subdir="03_feature_extraction")
        ensure_dir(plots_dir)
        qc_plots_dir = deriv_plots_path(deriv_root, subject, subdir="qc")
        ensure_dir(qc_plots_dir)
        try:
            _plot_trial_retention_diagnostics(retention_stats, subject, qc_plots_dir, logger, config)
        except Exception as e:
            logger.warning(f"Trial retention diagnostics plot failed: {e}")
        try:
            _generate_all_plots(tfr, pow_df_aligned, epochs, ms_templates, ms_df_aligned, aligned_events, subject, task, plots_dir, logger, config)
        except Exception as e:
            logger.warning(f"Plotting failed: {e}; continuing without visualizations")
    elif enable_plots:
        logger.warning("Matplotlib not available; skipping visualization generation")
    else:
        logger.info("Plotting disabled (enable_plots=False); skipping visualization generation")

    n_trials = len(y_aligned)
    n_pow_features = pow_df_aligned.shape[1] if pow_df_aligned is not None else 0
    n_conn_features = conn_df_aligned.shape[1] if conn_df_aligned is not None and not conn_df_aligned.empty else 0
    n_ms_features = ms_df_aligned.shape[1] if ms_df_aligned is not None and not ms_df_aligned.empty else 0
    n_all_features = combined_df.shape[1]
    
    logger.info(
        f"Done: sub-{subject}, n_trials={n_trials}, n_direct_features={n_pow_features}, "
        f"n_conn_features={n_conn_features}, n_microstate_features={n_ms_features}, "
        f"n_all_features={n_all_features} (power = log10(power/baseline))"
    )


def _plot_trial_retention_diagnostics(
    retention_stats: Dict[str, Any],
    subject: str,
    plots_dir: Path,
    logger: logging.Logger,
    config,
) -> None:
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib not available; skipping trial retention plot")
        return
    
    initial_count = retention_stats.get("initial_trial_count")
    before = retention_stats.get("before_alignment", {})
    after = retention_stats.get("after_alignment", {})
    
    if initial_count is None or not before or not after:
        logger.debug("Insufficient retention stats for plotting")
        return
    
    feature_families = ["power", "baseline", "connectivity", "microstates", "target"]
    families_present = [f for f in feature_families if before.get(f, 0) > 0 or after.get(f, 0) > 0]
    
    if not families_present:
        logger.debug("No feature families to plot")
        return
    
    retained = [after.get(f, 0) for f in families_present]
    dropped_before = [initial_count - before.get(f, 0) for f in families_present]
    dropped_after = [before.get(f, 0) - after.get(f, 0) for f in families_present]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(families_present))
    width = 0.6
    
    bottom_retained = np.zeros(len(families_present))
    bottom_dropped_after = retained
    bottom_dropped_before = [retained[i] + dropped_after[i] for i in range(len(families_present))]
    
    colors = {
        "retained": "#2ecc71",
        "dropped_after": "#e74c3c",
        "dropped_before": "#95a5a6",
    }
    
    bars_retained = ax.bar(x_pos, retained, width, label="Retained", color=colors["retained"], bottom=bottom_retained)
    bars_dropped_after = ax.bar(x_pos, dropped_after, width, label="Dropped (alignment)", color=colors["dropped_after"], bottom=bottom_dropped_after)
    bars_dropped_before = ax.bar(x_pos, dropped_before, width, label="Dropped (extraction)", color=colors["dropped_before"], bottom=bottom_dropped_before)
    
    ax.set_xlabel("Feature Family", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Trials", fontsize=12, fontweight="bold")
    ax.set_title(f"Trial Retention by Feature Family: sub-{subject}", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f.capitalize() for f in families_present], fontsize=11)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    
    for i, (r, da, db) in enumerate(zip(retained, dropped_after, dropped_before)):
        total = r + da + db
        if total > 0:
            pct_retained = 100 * r / total
            ax.text(i, r / 2, f"{r}\n({pct_retained:.1f}%)", ha="center", va="center", fontsize=9, fontweight="bold", color="white")
            if da > 0:
                ax.text(i, r + da / 2, f"{da}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
            if db > 0:
                ax.text(i, r + da + db / 2, f"{db}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    
    ax.text(0.02, 0.98, f"Initial trials: {initial_count}", transform=ax.transAxes, 
            fontsize=10, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.tight_layout()
    
    output_path = plots_dir / "qc_trials.png"
    save_formats = config.get("output.save_formats", ["png"])
    save_fig(fig, output_path.with_suffix(""), formats=save_formats, bbox_inches="tight", logger=logger)
    plt.close(fig)
    
    logger.info(f"Saved trial retention diagnostics to {output_path}")


def _save_trial_retention_tsv(
    retention_stats: Dict[str, Any],
    subject: str,
    features_dir: Path,
    logger: logging.Logger,
) -> None:
    initial_count = retention_stats.get("initial_trial_count")
    before = retention_stats.get("before_alignment", {})
    after = retention_stats.get("after_alignment", {})
    
    if initial_count is None or not before or not after:
        logger.debug("Insufficient retention stats for TSV")
        return
    
    feature_families = ["power", "baseline", "connectivity", "microstates", "target"]
    records = []
    
    for family in feature_families:
        before_count = before.get(family, 0)
        after_count = after.get(family, 0)
        dropped_before = initial_count - before_count
        dropped_after = before_count - after_count
        retained = after_count
        
        records.append({
            "feature_family": family,
            "initial_trials": initial_count,
            "trials_before_alignment": before_count,
            "trials_after_alignment": after_count,
            "dropped_during_extraction": dropped_before,
            "dropped_during_alignment": dropped_after,
            "retained": retained,
            "retention_rate": retained / initial_count if initial_count > 0 else 0.0,
        })
    
    df = pd.DataFrame(records)
    tsv_path = features_dir / "trial_retention_stats.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)
    logger.info(f"Saved trial retention statistics to {tsv_path}")


def _write_qc_summary(features_dir: Path, pow_df, baseline_df, conn_df, ms_df, y, logger) -> None:
    def _block_stats(name: str, df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        if df is None or len(df) == 0:
            return {
                "component": name,
                "n_trials": 0,
                "n_features": 0,
                "frac_nan": np.nan,
            }
        frac_nan = float(df.isna().mean().mean()) if df.size else 0.0
        return {
            "component": name,
            "n_trials": int(len(df)),
            "n_features": int(df.shape[1]),
            "frac_nan": frac_nan,
        }

    records = [
        _block_stats("power", pow_df),
        _block_stats("baseline", baseline_df),
        _block_stats("connectivity", conn_df),
        _block_stats("microstates", ms_df),
    ]
    y_stats = {
        "component": "target",
        "n_trials": int(len(y)),
        "n_features": 1,
        "frac_nan": float(pd.isna(y).mean()),
    }
    records.append(y_stats)

    summary_df = pd.DataFrame(records)
    summary_path = features_dir / "qc_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    logger.info("Wrote QC summary to %s", summary_path)
def main(subjects=None, task=None, all_subjects=False, no_plots=False):
    config = load_settings()
    deriv_root = Path(config.deriv_root)
    
    if task is None:
        task = config.task
    
    if all_subjects:
        subjects = get_available_subjects(deriv_root=deriv_root, task=task, config=config)
        if not subjects:
            raise ValueError(f"No subjects with cleaned epochs found in {deriv_root}")
    elif not subjects:
        raise ValueError("No subjects specified. Use --group all|A,B,C, or --subject (repeatable), or --all-subjects.")
    
    for sub in subjects:
        process_subject(sub, task, deriv_root, config, no_plots=no_plots)
    
    if len(subjects) >= 2:
        from eeg_pipeline.analysis.group_aggregation import aggregate_feature_stats
        aggregate_feature_stats(subjects, task, deriv_root, config)
    else:
        logger = get_group_logger("feature_engineering", config=config)
        logger.info(f"Group-level aggregation requires at least 2 subjects. Found {len(subjects)} subject(s); skipping group-level aggregation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG feature engineering: power + connectivity (single or multiple subjects)")

    sel = parser.add_mutually_exclusive_group(required=False)
    sel.add_argument(
        "--group", type=str,
        help=(
            "Group to process: 'all' or comma/space-separated subject labels without 'sub-' "
            "(e.g., '0001,0002,0003')."
        ),
    )
    sel.add_argument(
        "--subject", "-s", type=str, action="append",
        help=(
            "BIDS subject label(s) without 'sub-' prefix (e.g., 0001). "
            "Can be specified multiple times."
        ),
    )
    sel.add_argument(
        "--all-subjects", action="store_true",
        help="Process all available subjects with cleaned epochs",
    )
    parser.add_argument("--subjects", nargs="*", default=None, help="[Deprecated] Subject IDs list. Prefer --subject or --group.")

    parser.add_argument("--task", default=None, help="Task label (default from config)")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip all plotting operations"
    )
    args = parser.parse_args()

    config = load_settings()
    deriv_root = Path(config.deriv_root)
    
    subjects = parse_subject_args(args, config, task=args.task, deriv_root=deriv_root)
    
    if not subjects:
        print("No subjects provided. Use --group all|A,B,C, or --subject (repeatable), or --all-subjects.")
        raise SystemExit(2)

    main(subjects=subjects, task=args.task, all_subjects=False, no_plots=args.no_plots)
