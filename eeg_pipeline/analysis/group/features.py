from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging

import numpy as np
import pandas as pd
import mne

from eeg_pipeline.utils.config.loader import load_settings
from eeg_pipeline.utils.data.loading import (
    load_subject_features,
    load_feature_dfs_for_subjects,
    load_epochs_for_analysis,
)
from eeg_pipeline.utils.io.general import (
    deriv_group_stats_path,
    deriv_group_plots_path,
    deriv_features_path,
    ensure_dir,
    write_tsv,
    get_group_logger,
    _find_clean_epochs_path,
    save_fig,
    read_tsv,
)
from eeg_pipeline.utils.analysis.tfr import (
    validate_baseline_indices,
    read_tfr_average_with_logratio,
    find_tfr_path,
    compute_adaptive_n_cycles,
    extract_band_time_courses,
    get_tfr_config,
    resolve_tfr_workers,
)
from eeg_pipeline.utils.analysis.stats import (
    compute_band_summaries,
    fdr_bh,
)
from eeg_pipeline.plotting.features import (
    plot_group_power_plots,
    plot_group_band_power_time_courses,
    plot_group_microstate_transition_summary,
)
from eeg_pipeline.analysis.features.microstates import (
    compute_microstate_transition_stats,
    _state_labels,
    MicrostateTransitionStats,
)
from eeg_pipeline.analysis.group.statistics import (
    parse_pow_column,
    extract_bin_token,
    validate_subjects_for_group_analysis,
)


###################################################################
# Helper Functions
###################################################################


def _parse_pow_column(col: str) -> Optional[Tuple[str, str, str]]:
    """Alias for parse_pow_column from statistics module."""
    return parse_pow_column(col)


def _extract_bin_token(col_name: str) -> Optional[str]:
    """Alias for extract_bin_token from statistics module."""
    return extract_bin_token(col_name)


def _are_tfrs_compatible(tfr1: "mne.time_frequency.AverageTFR", tfr2: "mne.time_frequency.AverageTFR") -> bool:
    if len(tfr1.freqs) != len(tfr2.freqs):
        return False
    if not np.allclose(tfr1.freqs, tfr2.freqs, rtol=1e-5):
        return False
    if len(tfr1.times) != len(tfr2.times):
        return False
    if not np.allclose(tfr1.times, tfr2.times, rtol=1e-5):
        return False
    if tfr1.info["ch_names"] != tfr2.info["ch_names"]:
        return False
    return True




###################################################################
# Feature Aggregation Functions
###################################################################


def aggregate_power_features(
    subjects: List[str],
    deriv_root: Path,
    logger: logging.Logger,
    config,
) -> pd.DataFrame:
    logger.info("Aggregating power features across subjects...")
    
    combined = load_feature_dfs_for_subjects(
        subjects, deriv_root, "feature_aggregation.input_filenames.power", logger, config
    )
    
    if combined.empty:
        return pd.DataFrame()
    
    power_bands = config.get("features.frequency_bands")
    if not power_bands:
        logger.error("features.frequency_bands not found in config")
        return pd.DataFrame()
    
    summary_rows = []
    
    bin_label = config.get("feature_aggregation.power_bin", "plateau")
    subject_channel_sets = []
    for _, df_subj in combined.groupby("subject"):
        chans = set()
        for c in df_subj.columns:
            parsed = _parse_pow_column(c)
            if parsed:
                _, ch, _ = parsed
                chans.add(ch)
        subject_channel_sets.append(chans)
    common_channels = set.intersection(*subject_channel_sets) if subject_channel_sets else set()
    if not common_channels:
        logger.error("No common EEG channels across subjects for power aggregation; aborting.")
        return pd.DataFrame()
    
    for band in power_bands:
        band_cols_all = []
        for col in combined.columns:
            parsed = _parse_pow_column(col)
            if parsed and parsed[0] == band:
                band_cols_all.append(col)

        if not band_cols_all:
            continue

        if bin_label is None:
            bin_tokens = {_extract_bin_token(c) for c in band_cols_all}
            bin_tokens.discard(None)
            if len(bin_tokens) != 1:
                logger.error(
                    "Multiple temporal bins found for band %s (%s) but feature_aggregation.power_bin is unset. "
                    "Set feature_aggregation.power_bin to one of these to ensure consistent aggregation.",
                    band, ", ".join(sorted(bin_tokens)),
                )
                continue
            bin_label_use = bin_tokens.pop()
        else:
            bin_label_use = bin_label

            band_cols = []
            for c in band_cols_all:
                parsed = _parse_pow_column(c)
                if parsed is None:
                    continue
                _, ch, bin_token = parsed
                if bin_token == bin_label_use and ch in common_channels:
                    band_cols.append(c)
            if not band_cols:
                logger.warning(
                    "No power columns for band %s with bin '%s'; available bins: %s",
                    band, bin_label_use,
                    ", ".join(sorted({_extract_bin_token(c) or 'plateau' for c in band_cols_all})),
                )
                continue

        subject_band = combined.groupby("subject")[band_cols].mean()
        if subject_band.empty:
            continue
        band_data = subject_band

        summary_rows.append({
            "band": band,
            "n_subjects": band_data.shape[0],
            "n_channels": len(band_cols),
            "mean": float(band_data.mean().mean()),
            "std": float(band_data.std().mean()),
            "median": float(band_data.median().mean()),
            "min": float(band_data.min().min()),
            "max": float(band_data.max().max()),
        })
    
    return pd.DataFrame(summary_rows)


def aggregate_microstate_features(
    subjects: List[str],
    deriv_root: Path,
    logger: logging.Logger,
    config,
) -> pd.DataFrame:
    logger.info("Aggregating microstate features across subjects...")
    
    used_fixed_templates = config.get("feature_engineering.microstates.use_fixed_templates", False)
    
    if not used_fixed_templates:
        error_msg = (
            "SCIENTIFIC VALIDITY ERROR: Attempting to aggregate microstate features without fixed templates. "
            "If individual templates were used, 'State 0' for Subject X is not comparable to 'State 0' for Subject Y. "
            "To fix this:\n"
            "1. Use 'extract_microstates_group_templates' to generate group maps first.\n"
            "2. Set 'feature_engineering.microstates.use_fixed_templates = True' and provide the template path."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    combined = load_feature_dfs_for_subjects(
        subjects, deriv_root, "feature_aggregation.input_filenames.microstates", logger, config
    )
    
    if combined.empty:
        return pd.DataFrame()
    
    coverage_cols = [col for col in combined.columns if "coverage_" in col]
    duration_cols = [col for col in combined.columns if "duration_" in col]
    subject_means = combined.groupby("subject")[coverage_cols + duration_cols].mean()
    
    summary_rows = []
    n_subjects = subject_means.shape[0]
    
    for col in coverage_cols:
        state = col.split("_")[-1]
        if col not in subject_means.columns:
            continue
        summary_rows.append({
            "metric": "coverage",
            "state": state,
            "n_subjects": n_subjects,
            "mean": float(subject_means[col].mean()),
            "std": float(subject_means[col].std()),
            "median": float(subject_means[col].median()),
        })
    
    for col in duration_cols:
        state = col.split("_")[-1]
        if col not in subject_means.columns:
            continue
        summary_rows.append({
            "metric": "duration",
            "state": state,
            "n_subjects": n_subjects,
            "mean": float(subject_means[col].mean()),
            "std": float(subject_means[col].std()),
            "median": float(subject_means[col].median()),
        })
    
    return pd.DataFrame(summary_rows)


def aggregate_microstate_transition_stats(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    logger: logging.Logger,
    config,
) -> Tuple[Optional[MicrostateTransitionStats], pd.DataFrame]:
    """
    Aggregate microstate transition rates across subjects and compute pain vs non-pain effects.
    Returns a MicrostateTransitionStats with group means plus a long-form summary dataframe.
    """
    from scipy import stats
    
    used_fixed_templates = config.get("feature_engineering.microstates.use_fixed_templates", False)
    if not used_fixed_templates:
        logger.error(
            "Microstate transitions cannot be aggregated without fixed templates. "
            "Set feature_engineering.microstates.use_fixed_templates=True and rerun extraction."
        )
        return None, pd.DataFrame()

    n_states = int(config.get("feature_engineering.microstates.n_states", 4))
    state_labels = _state_labels(n_states, config)
    alpha = float(config.get("behavior_analysis.statistics.fdr_alpha", 0.05))
    min_subjects = int(config.get("analysis.min_subjects_for_group", 2))

    trans_nonpain_list: List[np.ndarray] = []
    trans_pain_list: List[np.ndarray] = []
    summary_rows: List[dict] = []

    for subj in subjects:
        ms_path = deriv_features_path(deriv_root, subj) / "features_microstates.tsv"
        if not ms_path.exists():
            logger.warning(f"Missing microstate features for sub-{subj}: {ms_path}")
            continue
        try:
            ms_df = read_tsv(ms_path)
        except Exception as exc:
            logger.warning(f"Failed to read microstate features for sub-{subj}: {exc}")
            continue
        if ms_df.empty:
            logger.warning(f"No microstate rows for sub-{subj}; skipping.")
            continue

        epochs, events_df = load_epochs_for_analysis(
            subj,
            task,
            align="strict",
            preload=False,
            deriv_root=deriv_root,
            bids_root=config.bids_root,
            config=config,
            logger=logger,
        )
        if events_df is None or len(events_df) == 0:
            logger.warning(f"No events available for sub-{subj}; skipping microstate transitions.")
            continue
        if len(events_df) != len(ms_df):
            logger.warning(
                "Mismatch between events (%d) and microstate rows (%d) for sub-%s; skipping.",
                len(events_df),
                len(ms_df),
                subj,
            )
            continue

        stats_subj = compute_microstate_transition_stats(
            ms_df,
            events_df,
            n_states=n_states,
            config=config,
        )
        if stats_subj is None:
            continue
        trans_nonpain_list.append(stats_subj.nonpain)
        trans_pain_list.append(stats_subj.pain)

    n_valid = len(trans_nonpain_list)
    if n_valid < min_subjects:
        logger.warning(
            "Microstate transition aggregation requires >=%d subjects; found %d.",
            min_subjects,
            n_valid,
        )
        return None, pd.DataFrame()

    nonpain_stack = np.stack(trans_nonpain_list, axis=0)
    pain_stack = np.stack(trans_pain_list, axis=0)
    mean_nonpain = np.nanmean(nonpain_stack, axis=0)
    mean_pain = np.nanmean(pain_stack, axis=0)

    p_mat = np.full((n_states, n_states), np.nan, dtype=float)
    q_mat = np.full((n_states, n_states), np.nan, dtype=float)
    p_values_flat: List[float] = []
    coords: List[Tuple[int, int]] = []

    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                continue
            diff_vals = pain_stack[:, i, j] - nonpain_stack[:, i, j]
            finite_mask = np.isfinite(diff_vals)
            if np.sum(finite_mask) < min_subjects:
                continue
            try:
                stat = stats.wilcoxon(diff_vals[finite_mask], zero_method="wilcox", alternative="two-sided")
                p_val = float(stat.pvalue) if np.isfinite(stat.pvalue) else np.nan
            except ValueError:
                p_val = np.nan
            p_mat[i, j] = p_val
            if np.isfinite(p_val):
                p_values_flat.append(p_val)
                coords.append((i, j))

    if p_values_flat:
        q_flat = fdr_bh(p_values_flat, config=config)
        for (i, j), q_val in zip(coords, q_flat):
            q_mat[i, j] = float(q_val)

    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                continue
            summary_rows.append({
                "from_state": state_labels[i],
                "to_state": state_labels[j],
                "mean_nonpain": float(mean_nonpain[i, j]),
                "mean_pain": float(mean_pain[i, j]),
                "difference": float(mean_pain[i, j] - mean_nonpain[i, j]),
                "p_value": float(p_mat[i, j]) if np.isfinite(p_mat[i, j]) else np.nan,
                "q_value": float(q_mat[i, j]) if np.isfinite(q_mat[i, j]) else np.nan,
                "n_subjects": n_valid,
            })

    transitions = MicrostateTransitionStats(
        nonpain=mean_nonpain,
        pain=mean_pain,
        state_labels=state_labels,
        p_values=p_mat,
        q_values=q_mat,
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(["from_state", "to_state"]).reset_index(drop=True)

    n_sig = int(np.sum(np.isfinite(q_mat) & (q_mat < alpha)))
    logger.info(
        "Aggregated microstate transitions across %d subjects (alpha=%.3f, %d significant edges after FDR).",
        n_valid,
        alpha,
        n_sig,
    )

    return transitions, summary_df


def aggregate_connectivity_features(
    subjects: List[str],
    deriv_root: Path,
    logger: logging.Logger,
    config,
) -> pd.DataFrame:
    logger.info("Aggregating connectivity features across subjects...")
    
    combined = load_feature_dfs_for_subjects(
        subjects, deriv_root, "feature_aggregation.input_filenames.connectivity", logger, config
    )
    
    if combined.empty:
        return pd.DataFrame()
    
    aec_cols = [col for col in combined.columns if "aec" in col.lower()]
    wpli_cols = [col for col in combined.columns if "wpli" in col.lower()]
    subject_cols = [set(combined[combined["subject"] == s].columns) - {"subject"} for s in combined["subject"].unique()]
    common_cols = set.intersection(*subject_cols) if subject_cols else set()
    if common_cols:
        aec_cols = [c for c in aec_cols if c in common_cols]
        wpli_cols = [c for c in wpli_cols if c in common_cols]
    else:
        logger.error("No common connectivity columns across subjects; aborting connectivity aggregation.")
        return pd.DataFrame()
    
    summary_rows = []
    n_subjects = combined["subject"].nunique()
    
    if aec_cols:
        aec_data = combined.groupby("subject")[aec_cols].mean()
        z_vals = np.arctanh(np.clip(aec_data.to_numpy(dtype=float), -0.999999, 0.999999))
        z_subject_means = np.nanmean(z_vals, axis=1)
        z_mean = np.nanmean(z_subject_means)
        z_median = np.nanmedian(z_subject_means)
        z_std = np.nanstd(z_subject_means, ddof=1)
        summary_rows.append({
            "measure": "AEC",
            "n_subjects": n_subjects,
            "n_features": len(aec_cols),
            "mean": float(np.tanh(z_mean)),
            "std_z": float(z_std),
            "median": float(np.tanh(z_median)),
        })
    
    if wpli_cols:
        wpli_data = combined.groupby("subject")[wpli_cols].mean()
        wpli_subject_means = np.nanmean(wpli_data.to_numpy(dtype=float), axis=1)
        summary_rows.append({
            "measure": "wPLI",
            "n_subjects": n_subjects,
            "n_features": len(wpli_cols),
            "mean": float(np.nanmean(wpli_subject_means)),
            "std": float(np.nanstd(wpli_subject_means, ddof=1)),
            "median": float(np.nanmedian(wpli_subject_means)),
        })
    
    return pd.DataFrame(summary_rows)


def compute_subject_band_means(subject_features: Dict[str, pd.DataFrame], bands: List[str], config=None) -> pd.DataFrame:
    if not subject_features or not bands:
        return pd.DataFrame()
    
    records = []
    if config is None:
        config = load_settings()
    bin_label = config.get("feature_aggregation.power_bin", "plateau")

    all_channel_sets = []
    for df in subject_features.values():
        chans = {c.split("_")[2] for c in df.columns if str(c).startswith("pow_") and len(str(c).split("_")) >= 3}
        all_channel_sets.append(chans)
    common_channels = set.intersection(*all_channel_sets) if all_channel_sets else set()
    if not common_channels:
        return pd.DataFrame()

    for band in bands:
        for subject, df in subject_features.items():
            band_cols_all = []
            for col in df.columns:
                parsed = _parse_pow_column(col)
                if parsed and parsed[0] == band:
                    band_cols_all.append(col)
            if not band_cols_all:
                continue
            if bin_label is None:
                bin_tokens = {_extract_bin_token(c) for c in band_cols_all}
                bin_tokens.discard(None)
                if len(bin_tokens) != 1:
                    continue
                bin_label_use = bin_tokens.pop()
            else:
                bin_label_use = bin_label

            power_cols = []
            for c in band_cols_all:
                parsed = _parse_pow_column(c)
                if parsed is None:
                    continue
                _, ch, bin_token = parsed
                if bin_token == bin_label_use and ch in common_channels:
                    power_cols.append(c)
            if not power_cols:
                continue
            
            values = pd.to_numeric(df[power_cols].stack(), errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            
            records.append({
                "subject": subject,
                "band": band,
                "mean_power": float(np.mean(values)),
            })
    
    return pd.DataFrame(records)


###################################################################
# Plotting and Processing Functions
###################################################################


def plot_band_power_distributions(means_df: pd.DataFrame, bands_present: List[str], 
                                  output_path: Path, config, logger: logging.Logger) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    
    summary_df = compute_band_summaries(means_df, bands_present)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    x_positions = np.arange(len(bands_present))
    
    ax.bar(x_positions, summary_df["group_mean"], color='steelblue', alpha=0.8)
    
    yerr_low = [
        mu - lo if np.isfinite(mu) and np.isfinite(lo) else 0 
        for lo, mu in zip(summary_df["ci_low"], summary_df["group_mean"])
    ]
    yerr_high = [
        hi - mu if np.isfinite(mu) and np.isfinite(hi) else 0 
        for hi, mu in zip(summary_df["ci_high"], summary_df["group_mean"])
    ]
    yerr = np.array([yerr_low, yerr_high])
    
    ax.errorbar(x_positions, summary_df["group_mean"], yerr=yerr, fmt='none', ecolor='k', capsize=3)
    
    rng_seed = config.get("random.seed", 42)
    rng = np.random.default_rng(rng_seed)
    
    for idx, band in enumerate(bands_present):
        band_values = means_df[means_df["band"] == band]["mean_power"].to_numpy(dtype=float)
        jitter_range = config.get("behavior_analysis.group_aggregation.jitter_range", 0.2)
        scatter_size = config.get("behavior_analysis.group_aggregation.scatter_size", 12)
        scatter_alpha = config.get("behavior_analysis.group_aggregation.scatter_alpha", 0.6)
        jitter = (rng.random(len(band_values)) - 0.5) * jitter_range
        ax.scatter(
            np.full_like(band_values, idx, dtype=float) + jitter, 
            band_values, 
            color='k', 
            s=scatter_size, 
            alpha=scatter_alpha
        )
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels([band.capitalize() for band in bands_present])
    ax.set_ylabel('Mean log10(power/baseline) across subjects')
    ax.set_title('Group Band Power Summary (subject means, 95% CI)')
    ax.axhline(0, color='k', linewidth=0.8)
    plt.tight_layout()
    
    formats = config.get("output.save_formats", ["png"])
    constants = {
        "FIG_DPI": int(config.get("output.fig_dpi", 300)),
        "SAVE_FORMATS": list(formats),
        "output.bbox_inches": config.get("output.bbox_inches", "tight"),
        "output.pad_inches": float(config.get("output.pad_inches", 0.02)),
    }
    save_fig(
        fig, output_path, formats=tuple(formats), dpi=constants["FIG_DPI"],
        bbox_inches=constants["output.bbox_inches"],
        pad_inches=constants["output.pad_inches"], constants=constants
    )
    
    return summary_df


def compute_avg_tfr_for_subject(subject: str, task: str, deriv_root: Path, 
                                  config) -> Optional["mne.time_frequency.AverageTFR"]:
    epochs_path = _find_clean_epochs_path(subject, task, deriv_root=deriv_root)
    if epochs_path is None or not epochs_path.exists():
        return None
    
    epochs = mne.read_epochs(epochs_path, preload=False, verbose=False)
    
    freq_min, freq_max, n_freqs, n_cycles_factor, decim, picks = get_tfr_config(config)
    custom_freqs = config.get("time_frequency_analysis.tfr.custom_freqs", None)
    
    if custom_freqs is not None:
        frequencies = np.array(custom_freqs)
    else:
        frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
    
    n_cycles = compute_adaptive_n_cycles(frequencies, cycles_factor=n_cycles_factor, config=config)
    workers_default = int(config.get("tfr_topography_pipeline.tfr.workers", -1))
    workers = resolve_tfr_workers(workers_default=workers_default)

    tfr_epochs = epochs.compute_tfr(
        method="morlet",
        freqs=frequencies,
        n_cycles=n_cycles,
        decim=decim,
        picks=picks,
        use_fft=True,
        return_itc=False,
        average=False,
        n_jobs=workers,
    )
    
    times = np.asarray(tfr_epochs.times)
    baseline_window = tuple(config.get("time_frequency_analysis.baseline_window", [-2.0, 0.0]))
    b_start, b_end, _ = validate_baseline_indices(times, baseline_window)
    tfr_epochs.apply_baseline(baseline=(b_start, b_end), mode="logratio")
    
    return tfr_epochs.average()


def load_tfr_data(subjects: List[str], task: str, deriv_root: Path, 
                   config, logger: logging.Logger) -> List["mne.time_frequency.AverageTFR"]:
    if not subjects:
        return []
    
    tfr_list = []
    missing_subjects = []
    reference_tfr = None
    
    for subject in subjects:
        tfr_path = find_tfr_path(subject, task, deriv_root)
        if tfr_path is None or not tfr_path.exists():
            missing_subjects.append(subject)
            continue
        
        baseline_window = tuple(config.get("time_frequency_analysis.baseline_window", [-2.0, 0.0]))
        tfr_avg = read_tfr_average_with_logratio(tfr_path, baseline_window, logger)
        if tfr_avg is None:
            missing_subjects.append(subject)
            continue
        
        if reference_tfr is None:
            reference_tfr = tfr_avg
            tfr_list.append(tfr_avg)
        else:
            if not _are_tfrs_compatible(reference_tfr, tfr_avg):
                logger.warning(
                    f"TFR for subject {subject} has incompatible frequency grid or time axis "
                    f"compared to reference (freqs: {len(reference_tfr.freqs)} vs {len(tfr_avg.freqs)}, "
                    f"times: {len(reference_tfr.times)} vs {len(tfr_avg.times)}); skipping."
                )
                missing_subjects.append(subject)
            else:
                tfr_list.append(tfr_avg)
    
    if tfr_list:
        channel_sets = [set(tfr.info["ch_names"]) for tfr in tfr_list]
        common_channels = set.intersection(*channel_sets) if channel_sets else set()
        min_channels = int(config.get("time_frequency_analysis.min_channels_for_group", 4))
        if len(common_channels) < min_channels:
            logger.warning(
                "Too few common channels across TFRs (%d found, need >=%d); skipping group TFR aggregation.",
                len(common_channels), min_channels,
            )
            return []
        
        common_order = [ch for ch in tfr_list[0].info["ch_names"] if ch in common_channels]
        aligned_list = []
        for tfr in tfr_list:
            picks = mne.pick_channels(tfr.info["ch_names"], include=common_order, ordered=True)
            tfr_aligned = tfr.copy()
            tfr_aligned.pick_channels(common_order, ordered=True)
            aligned_list.append(tfr_aligned)
        tfr_list = aligned_list

    min_subjects_for_group = config.get("analysis.min_subjects_for_group", 2)
    if len(tfr_list) < min_subjects_for_group and missing_subjects:
        computed_count = 0
        skipped_incompatible = []
        for subject in missing_subjects:
            tfr_avg = compute_avg_tfr_for_subject(subject, task, deriv_root, config)
            if tfr_avg is None:
                continue
            
            if reference_tfr is not None and not _are_tfrs_compatible(reference_tfr, tfr_avg):
                logger.warning(
                    f"On-the-fly computed TFR for subject {subject} has incompatible frequency grid "
                    f"or time axis compared to reference; skipping to avoid mixing heterogeneous data."
                )
                skipped_incompatible.append(subject)
                continue
            
            if reference_tfr is None:
                reference_tfr = tfr_avg
            tfr_list.append(tfr_avg)
            computed_count += 1
        
        if computed_count > 0:
            logger.info(f"Computed AverageTFR on the fly for {computed_count} subjects (no saved TFR found)")
        if skipped_incompatible:
            logger.warning(
                f"Skipped {len(skipped_incompatible)} subjects due to incompatible TFR parameters: "
                f"{', '.join(skipped_incompatible)}"
            )
    
    return tfr_list


def process_band_power_distributions(means_df: pd.DataFrame, bands: List[str],
                                     group_plots: Path, group_stats: Path,
                                     config, logger: logging.Logger) -> None:
    if means_df.empty:
        return
    
    bands_present = [band for band in bands if band in means_df["band"].values]
    if not bands_present:
        return
    
    output_path = group_plots / "group_power_distributions_per_band_across_subjects"
    summary_df = plot_band_power_distributions(means_df, bands_present, output_path, config, logger)
    write_tsv(summary_df, group_stats / "group_band_power_subject_means.tsv")
    logger.info("Saved group band power distributions and stats.")


def process_tfr_time_courses(tfr_list: List["mne.time_frequency.AverageTFR"],
                             bands: List[str], freq_bands_dict: Dict[str, Tuple[float, float]],
                             group_plots: Path, config, logger: logging.Logger) -> None:
    min_subjects_for_group = config.get("analysis.min_subjects_for_group", 2)
    if len(tfr_list) < min_subjects_for_group:
        logger.info(f"Skipping group band power time courses: need at least {min_subjects_for_group} subjects with TFR (saved or computed).")
        return
    
    tmin = float(min(tfr.times[0] for tfr in tfr_list))
    tmax = float(max(tfr.times[-1] for tfr in tfr_list))
    
    band_tc_logr, band_tc_pct, reference_times = extract_band_time_courses(
        tfr_list, bands, freq_bands_dict, tmin, tmax
    )
    valid_bands = [
        band for band in bands 
        if len(band_tc_logr.get(band, [])) >= min_subjects_for_group
    ]
    
    if not valid_bands:
        logger.info(f"No bands with >={min_subjects_for_group} subjects for time-course plotting; skipping.")
        return
    
    plot_group_band_power_time_courses(
        valid_bands, band_tc_logr, band_tc_pct, 
        reference_times, group_plots, config, logger
    )


def _save_summary(df: pd.DataFrame, filename_key: str, description: str, config, group_dir, logger) -> None:
    if df.empty:
        return
    filename = config.get(f"feature_aggregation.filenames.{filename_key}")
    if filename is None:
        filename = f"aggregated_{filename_key}.tsv"
    file_path = group_dir / filename
    write_tsv(df, file_path)
    logger.info(f"Saved {description} to {file_path}")


###################################################################
# Main Entry Point
###################################################################


def aggregate_feature_stats(subjects: List[str], task: str, deriv_root: Path, config) -> None:
    logger = get_group_logger("feature_engineering", config=config)
    
    if not validate_subjects_for_group_analysis(subjects, logger, config):
        return
    
    group_plots = deriv_group_plots_path(deriv_root, "02_feature_engineering")
    group_stats = deriv_group_stats_path(deriv_root)
    ensure_dir(group_plots)
    ensure_dir(group_stats)
    
    power_summary = aggregate_power_features(subjects, deriv_root, logger, config)
    _save_summary(power_summary, "power_summary", "power summary", config, group_stats, logger)
    
    microstate_summary = aggregate_microstate_features(subjects, deriv_root, logger, config)
    _save_summary(microstate_summary, "microstate_summary", "microstate summary", config, group_stats, logger)

    transitions, transition_summary = aggregate_microstate_transition_stats(
        subjects, task, deriv_root, logger, config
    )
    if transition_summary is not None and not transition_summary.empty:
        transition_filename = config.get(
            "feature_aggregation.filenames.microstate_transitions",
            "aggregated_microstate_transitions.tsv",
        )
        write_tsv(transition_summary, group_stats / transition_filename)
        logger.info("Saved microstate transition summary to %s", group_stats / transition_filename)
    if transitions is not None:
        plot_group_microstate_transition_summary(transitions, group_plots, logger, config)
    
    connectivity_summary = aggregate_connectivity_features(subjects, deriv_root, logger, config)
    _save_summary(connectivity_summary, "connectivity_summary", "connectivity summary", config, group_stats, logger)

    subject_features = load_subject_features(subjects, deriv_root, logger)
    min_subjects_for_group = config.get("analysis.min_subjects_for_group", 2)
    if len(subject_features) < min_subjects_for_group:
        logger.warning(
            f"Group-level aggregation requires at least {min_subjects_for_group} subjects with valid feature files. "
            f"Found {len(subject_features)} subject(s) with features; skipping group-level aggregation."
        )
        return
    
    power_bands = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    bands = list(power_bands)
    
    plot_group_power_plots(subject_features, bands, group_plots, group_stats, config, logger)
    
    means_df = compute_subject_band_means(subject_features, bands, config=config)
    process_band_power_distributions(means_df, bands, group_plots, group_stats, config, logger)
    
    freq_bands = config.get("time_frequency_analysis.bands", {
        "delta": [1.0, 3.9],
        "theta": [4.0, 7.9],
        "alpha": [8.0, 12.9],
        "beta": [13.0, 30.0],
        "gamma": [30.1, 80.0],
    })
    freq_bands_dict = {name: tuple(freqs) for name, freqs in freq_bands.items()}
    
    tfr_list = load_tfr_data(subjects, task, deriv_root, config, logger)
    process_tfr_time_courses(tfr_list, bands, freq_bands_dict, group_plots, config, logger)

