from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List
import logging

import numpy as np
import pandas as pd
import mne
from scipy import stats, signal
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

from eeg_pipeline.utils.config_loader import load_settings
from eeg_pipeline.utils.tfr_utils import (
    validate_baseline_indices,
    read_tfr_average_with_logratio,
    compute_adaptive_n_cycles,
    log_tfr_resolution,
    validate_tfr_parameters,
    save_tfr_with_sidecar,
    run_tfr_morlet,
    find_tfr_path,
)
from eeg_pipeline.analysis.feature_extraction import (
    extract_baseline_power_features,
    extract_band_power_features,
    extract_connectivity_features,
    extract_microstate_features,
)
from eeg_pipeline.utils.data_loading import (
    get_available_subjects,
    load_epochs_for_analysis,
    parse_subject_args,
)
from eeg_pipeline.utils.io_utils import get_subject_logger, get_group_logger
from eeg_pipeline.utils.io_utils import (
    _pick_target_column,
    save_fig,
    _find_clean_epochs_path,
    deriv_features_path,
    deriv_plots_path,
    deriv_stats_path,
    ensure_dir,
    find_first,
    setup_matplotlib,
)
from eeg_pipeline.utils.io_utils import get_band_color
from eeg_pipeline.analysis.feature_extraction import (
    compute_microstate_metric_correlations,
    compute_microstate_transition_stats,
    compute_microstate_duration_stats,
    zscore_maps as _zscore_maps,
    compute_gfp as _compute_gfp,
    corr_maps as _corr_maps,
    label_timecourse as _label_timecourse,
    extract_templates_from_trials as _extract_templates_from_trials,
)
from eeg_pipeline.plotting.plot_features import (
    plot_microstate_templates,
    plot_microstate_coverage_by_pain,
    plot_microstate_pain_correlation_heatmap as _plot_microstate_pain_correlation_heatmap,
    plot_microstate_temporal_evolution,
    plot_microstate_templates_by_pain,
    plot_microstate_templates_by_temperature,
    plot_microstate_gfp_colored_by_state,
    plot_microstate_gfp_by_temporal_bins,
    plot_microstate_transition_network as _plot_microstate_transition_network,
    plot_microstate_duration_distributions as _plot_microstate_duration_distributions,
    plot_power_distributions,
    plot_channel_power_heatmap,
    plot_power_time_courses,
    plot_power_spectral_density,
    plot_power_spectral_density_by_pain,
    plot_power_time_course_by_temperature,
    plot_trial_power_variability,
    plot_inter_band_spatial_power_correlation,
    plot_group_power_plots,
    plot_group_band_power_time_courses,
)
import argparse

###################################################################
# Configuration (loaded in main() and passed as parameters)
###################################################################


def _setup_logging(subject: str, config=None):
    if config is None:
        config = load_settings()
    log_file_name = config.get("logging.file_names.feature_engineering", "02_feature_extraction.log")
    return get_subject_logger("feature_engineering", subject, log_file_name, config=config)


def _extract_baseline_power_features(tfr, bands, baseline_window, config, logger):
    return extract_baseline_power_features(tfr, bands, baseline_window, config, logger)


def _extract_band_power_features(tfr, bands, config, logger):
    return extract_band_power_features(tfr, bands, config, logger)


def _find_connectivity_arrays(subject: str, task: str, band: str, deriv_root: Path):
    def _find_measure(measure: str) -> Optional[Path]:
        patterns = [
            f"sub-{subject}/eeg/sub-{subject}_task-{task}_*connectivity_{measure}_{band}*_all_trials.npy",
            f"sub-{subject}/eeg/sub-{subject}_task-{task}_connectivity_{measure}_{band}*_all_trials.npy",
        ]
        for pattern in patterns:
            path = find_first(str((deriv_root / pattern).as_posix()))
            if path:
                return path
        return None
    
    return _find_measure("aec"), _find_measure("wpli")


def _load_labels(subject: str, task: str, deriv_root: Path) -> Optional[np.ndarray]:
    patterns = [
        f"sub-{subject}/eeg/sub-{subject}_task-{task}_*connectivity_labels*.npy",
        f"sub-{subject}/eeg/sub-{subject}_task-{task}_connectivity_labels*.npy",
    ]
    for pattern in patterns:
        path = find_first(str((deriv_root / pattern).as_posix()))
        if path:
            return np.load(path, allow_pickle=True)
    return None


###################################################################
# Microstate Plotting Helpers
###################################################################

def plot_microstate_pain_correlation_heatmap(ms_df, events_df, subject, save_dir, n_states, logger, config):
    corr_df, pval_df = compute_microstate_metric_correlations(ms_df, events_df, config=config)
    return _plot_microstate_pain_correlation_heatmap(corr_df, pval_df, subject, save_dir, logger, config)


def plot_microstate_transition_network(ms_df, events_df, subject, save_dir, n_states, logger, config):
    stats = compute_microstate_transition_stats(ms_df, events_df, n_states=n_states, config=config)
    return _plot_microstate_transition_network(stats, subject, save_dir, logger, config)


def plot_microstate_duration_distributions(ms_df, events_df, subject, save_dir, n_states, logger, config):
    stats = compute_microstate_duration_stats(ms_df, events_df, n_states=n_states, config=config)
    return _plot_microstate_duration_distributions(stats, subject, save_dir, logger, config)


###################################################################
# Power Plotting Functions
###################################################################


def _flatten_lower_triangles(conn_trials, labels, prefix):
    if conn_trials.ndim != 3:
        raise ValueError("Connectivity array must be 3D (trials, nodes, nodes)")
    n_trials, n_nodes, _ = conn_trials.shape
    idx_i, idx_j = np.tril_indices(n_nodes, k=-1)
    out = conn_trials[:, idx_i, idx_j]

    if labels is not None and len(labels) == n_nodes:
        pair_names = [f"{labels[i]}__{labels[j]}" for i, j in zip(idx_i, idx_j)]
    else:
        pair_names = [f"n{i}_n{j}" for i, j in zip(idx_i, idx_j)]
    cols = [f"{prefix}_{p}" for p in pair_names]
    return pd.DataFrame(out), cols


def _extract_connectivity_features(subject: str, task: str, bands, deriv_root, logger):
    return extract_connectivity_features(subject, task, bands, deriv_root, logger)


###################################################################
# Feature Extraction Helpers
###################################################################

def align_feature_blocks(blocks: List[pd.DataFrame]) -> List[pd.DataFrame]:
    if not blocks:
        return []
    
    n_trials_ref = None
    aligned_blocks = []
    
    for block in blocks:
        if block is None or block.empty:
            continue
        
        if n_trials_ref is None:
            n_trials_ref = len(block)
            aligned_blocks.append(block)
        else:
            min_n = min(n_trials_ref, len(block))
            aligned_blocks.append(block.iloc[:min_n, :])
            aligned_blocks = [b.iloc[:min_n, :] for b in aligned_blocks]
            n_trials_ref = min_n
    
    return aligned_blocks


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
    n_cycles = compute_adaptive_n_cycles(freqs, cycles_factor=n_cycles_factor, min_cycles=3.0)
    
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
    tfr_baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-2.0, 0.0]))
    min_baseline_samples = int(config.get("tfr_topography_pipeline.min_baseline_samples", 5))
    b_start, b_end, _ = validate_baseline_indices(times, tfr_baseline, min_samples=min_baseline_samples)
    
    power_bands = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    baseline_df, baseline_cols = _extract_baseline_power_features(
        tfr, power_bands, (b_start, b_end), config, logger
    )
    
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


def _align_feature_dataframes(pow_df, baseline_df, conn_df, ms_df, y, logger):
    blocks_to_align = []
    block_map = []
    
    if pow_df is not None and not pow_df.empty:
        blocks_to_align.append(pow_df)
        block_map.append("power")
    
    if baseline_df is not None and not baseline_df.empty:
        blocks_to_align.append(baseline_df)
        block_map.append("baseline")
    
    if conn_df is not None and not conn_df.empty:
        blocks_to_align.append(conn_df)
        block_map.append("connectivity")
    
    if ms_df is not None and not ms_df.empty:
        blocks_to_align.append(ms_df)
        block_map.append("microstates")
    
    y_frame = None
    if y is not None:
        y_frame = y.to_frame() if isinstance(y, pd.Series) else y
        blocks_to_align.append(y_frame)
        block_map.append("target")
    
    if not blocks_to_align:
        logger.warning("No features extracted; skipping save")
        return None, None, None, None, None
    
    aligned_blocks = align_feature_blocks(blocks_to_align)
    
    if not aligned_blocks:
        logger.warning("Alignment failed; all blocks are empty")
        return None, None, None, None, None
    
    aligned_idx = 0
    aligned_map = {}
    for i, block_name in enumerate(block_map):
        block = blocks_to_align[i]
        if block is not None and not block.empty:
            aligned_map[block_name] = aligned_blocks[aligned_idx]
            aligned_idx += 1
    
    n_aligned = len(aligned_blocks[0]) if aligned_blocks else 0
    logger.info(f"Aligned {len(blocks_to_align)} feature blocks to {n_aligned} trials")
    
    pow_df_aligned = aligned_map.get("power", pow_df)
    baseline_df_aligned = aligned_map.get("baseline", baseline_df if baseline_df is not None else pd.DataFrame())
    conn_df_aligned = aligned_map.get("connectivity", conn_df if conn_df is not None else None)
    ms_df_aligned = aligned_map.get("microstates", ms_df if ms_df is not None else None)
    
    target_aligned = aligned_map.get("target")
    if target_aligned is not None:
        y_aligned = target_aligned.iloc[:, 0] if target_aligned.shape[1] == 1 else target_aligned
    else:
        y_aligned = y
    
    return pow_df_aligned, baseline_df_aligned, conn_df_aligned, ms_df_aligned, y_aligned


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
    features_dir, logger
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
    TARGET_COLUMN_NAME = "vas_rating"
    logger.info(f"Saving behavioral target vector: {y_path} (column: {TARGET_COLUMN_NAME})")
    y.to_frame(name=TARGET_COLUMN_NAME).to_csv(y_path, sep="\t", index=False)
    
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
            plot_functions = [
                (lambda: plot_microstate_templates_by_pain(epochs, aligned_events, subject, task, plots_dir, n_microstates, logger, config), "templates by pain"),
                (lambda: plot_microstate_templates_by_temperature(epochs, aligned_events, subject, task, plots_dir, n_microstates, logger, config), "templates by temperature"),
                (lambda: plot_microstate_gfp_colored_by_state(epochs, ms_templates, aligned_events, subject, plots_dir, n_microstates, logger, config), "GFP colored by state"),
                (lambda: plot_microstate_gfp_by_temporal_bins(epochs, ms_templates, aligned_events, subject, task, plots_dir, n_microstates, logger, config), "GFP by temporal bins"),
                (lambda: plot_microstate_coverage_by_pain(ms_df, aligned_events, subject, plots_dir, n_microstates, logger, config), "coverage by pain"),
                (lambda: plot_microstate_pain_correlation_heatmap(ms_df, aligned_events, subject, plots_dir, n_microstates, logger, config), "pain correlation heatmap"),
                (lambda: plot_microstate_temporal_evolution(epochs, ms_templates, aligned_events, subject, task, plots_dir, n_microstates, logger, config), "temporal evolution"),
                (lambda: plot_microstate_transition_network(ms_df, aligned_events, subject, plots_dir, n_microstates, logger, config), "transition network"),
                (lambda: plot_microstate_duration_distributions(ms_df, aligned_events, subject, plots_dir, n_microstates, logger, config), "duration distributions"),
            ]

            for plot_func, plot_name in plot_functions:
                try:
                    plot_func()
                except (OSError, ValueError) as e:
                    logger.warning(f"Failed to plot microstate {plot_name}: {e}; skipping this visualization")
                except Exception as e:
                    logger.warning(f"Failed to plot microstate {plot_name}: {e}; skipping this visualization")
        
        logger.info("Successfully generated all microstate visualizations")
    except (OSError, ValueError) as e:
        logger.warning(f"Microstate visualization failed: {e}; skipping all microstate plots for this subject")
    except Exception as e:
        logger.warning(f"Microstate visualization failed: {e}; skipping all microstate plots for this subject")


###################################################################
# Main Subject Processing
###################################################################

def process_subject(subject, task, deriv_root, config):
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

    drop_log_path = features_dir / "dropped_trials.tsv"
    _save_dropped_trials_log(epochs, aligned_events, drop_log_path, logger)
    
    manifest_path = features_dir / "trial_alignment.tsv"
    _save_trial_alignment_manifest(aligned_events, epochs, manifest_path, config, logger)

    constants = {"TARGET_COLUMNS": config.get("event_columns.rating", [])}
    target_col = _pick_target_column(aligned_events, constants=constants)
    if target_col is None:
        logger.warning("No suitable target column found in events; skipping")
        return
    
    y = pd.to_numeric(aligned_events[target_col], errors="coerce")
    tfr, baseline_df, baseline_cols = _prepare_tfr_with_baseline(epochs, aligned_events, subject, task, config, deriv_root, logger)

    power_bands = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    n_microstates = int(config.get("feature_engineering.microstates.n_states", 4))
    
    pow_df, pow_cols = _extract_band_power_features(tfr, power_bands, config, logger)
    conn_df, conn_cols = _extract_connectivity_features(subject, task, power_bands, deriv_root, logger)
    
    logger.info("Extracting microstate features")
    ms_df, ms_cols, ms_templates = extract_microstate_features(epochs, n_microstates, config, logger)
    _save_microstate_templates(epochs, ms_templates, subject, n_microstates, deriv_root, logger)

    pow_df, baseline_df, conn_df, ms_df, y = _align_feature_dataframes(
        pow_df, baseline_df, conn_df, ms_df, y, logger
    )
    if pow_df is None:
        return

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
        features_dir, logger
    )

    enable_plots = config.get("feature_engineering.enable_plots", True)
    if enable_plots:
        if not HAS_MATPLOTLIB:
            logger.warning("Matplotlib not available; skipping visualization generation")
        else:
            plots_dir = deriv_plots_path(deriv_root, subject, subdir="02_feature_engineering")
            ensure_dir(plots_dir)
            try:
                _generate_all_plots(tfr, pow_df, epochs, ms_templates, ms_df, aligned_events, subject, task, plots_dir, logger, config)
            except (OSError, ValueError) as e:
                logger.warning(f"Plotting failed: {e}; continuing without visualizations")
            except Exception as e:
                logger.warning(f"Plotting failed: {e}; continuing without visualizations")
    else:
        logger.info("Plotting disabled (enable_plots=False); skipping visualization generation")

    n_trials = len(y)
    n_pow_features = pow_df.shape[1]
    n_conn_features = conn_df.shape[1] if conn_df is not None and not conn_df.empty else 0
    n_ms_features = ms_df.shape[1] if ms_df is not None and not ms_df.empty else 0
    n_all_features = combined_df.shape[1]
    
    logger.info(
        f"Done: sub-{subject}, n_trials={n_trials}, n_direct_features={n_pow_features}, "
        f"n_conn_features={n_conn_features}, n_microstate_features={n_ms_features}, "
        f"n_all_features={n_all_features} (power = log10(power/baseline))"
    )
def main(subjects=None, task=None, all_subjects=False):
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
        process_subject(sub, task, deriv_root, config)
    
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
    args = parser.parse_args()

    config = load_settings()
    deriv_root = Path(config.deriv_root)
    
    subjects = parse_subject_args(args, config, task=args.task, deriv_root=deriv_root)
    
    if not subjects:
        print("No subjects provided. Use --group all|A,B,C, or --subject (repeatable), or --all-subjects.")
        raise SystemExit(2)

    main(subjects=subjects, task=args.task, all_subjects=False)
