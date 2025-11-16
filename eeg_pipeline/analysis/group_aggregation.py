from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from scipy import stats

from eeg_pipeline.utils.config_loader import load_settings
from eeg_pipeline.utils.data_loading import (
    get_available_subjects,
    load_subject_features,
    load_subject_data_for_summary,
    load_channel_correlations,
    load_connectivity_files,
)
from eeg_pipeline.utils.io_utils import (
    deriv_stats_path,
    deriv_group_stats_path,
    deriv_group_plots_path,
    ensure_dir,
    fdr_bh_reject,
    get_group_logger,
    save_fig,
    _find_clean_epochs_path,
    get_band_color as _get_band_color,
    get_behavior_footer as _get_behavior_footer,
    sanitize_label,
    format_band_label,
    get_correlation_type_labels,
    read_tsv,
    write_tsv,
)
from eeg_pipeline.utils.tfr_utils import (
    validate_baseline_indices,
    read_tfr_average_with_logratio,
    run_tfr_morlet,
    find_tfr_path,
    compute_adaptive_n_cycles,
    extract_band_time_courses,
    prepare_bands_data_for_topomap,
)
from eeg_pipeline.utils.stats_utils import (
    _safe_float,
    _get_ttest_pvalue,
    fisher_aggregate as _fisher_aggregate,
    compute_correlation_pvalue,
    compute_group_channel_statistics,
    pool_data_by_strategy,
    compute_band_summaries,
)
from eeg_pipeline.plotting.plot_features import (
    plot_group_power_plots,
    plot_group_band_power_time_courses,
)
from eeg_pipeline.plotting.plot_behavioral import (
    plot_group_power_roi_scatter,
)
from eeg_pipeline.analysis.behavior import collect_group_power_roi_inputs


###################################################################
# Feature Statistics Aggregation
###################################################################


def compute_subject_band_means(subject_features: Dict[str, pd.DataFrame], bands: List[str]) -> pd.DataFrame:
    if not subject_features or not bands:
        return pd.DataFrame()
    
    records = []
    for band in bands:
        for subject, df in subject_features.items():
            power_cols = [col for col in df.columns if col.startswith(f"pow_{band}_")]
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


def plot_band_power_distributions(means_df: pd.DataFrame, bands_present: List[str], 
                                  output_path: Path, config, logger: logging.Logger) -> pd.DataFrame:
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
    
    custom_freqs = config.get("time_frequency_analysis.tfr.custom_freqs", None)
    if custom_freqs is not None:
        frequencies = np.array(custom_freqs)
    else:
        freq_min = config.get("time_frequency_analysis.tfr.freq_min", 1.0)
        freq_max = config.get("time_frequency_analysis.tfr.freq_max", 100.0)
        n_freqs = config.get("time_frequency_analysis.tfr.n_freqs", 40)
        frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
    
    cycles_factor = float(config.get("time_frequency_analysis.tfr.n_cycles_factor", 2.0))
    n_cycles = compute_adaptive_n_cycles(frequencies, cycles_factor=cycles_factor, config=config)
    decim = config.get("time_frequency_analysis.tfr.decim", 4)
    
    tfr_epochs = run_tfr_morlet(
        epochs, freqs=frequencies, n_cycles=n_cycles, decim=decim,
        picks="eeg", workers=-1, config=config,
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


def _are_tfrs_compatible(tfr1: "mne.time_frequency.AverageTFR", tfr2: "mne.time_frequency.AverageTFR") -> bool:
    if len(tfr1.freqs) != len(tfr2.freqs):
        return False
    if not np.allclose(tfr1.freqs, tfr2.freqs, rtol=1e-5):
        return False
    if len(tfr1.times) != len(tfr2.times):
        return False
    if not np.allclose(tfr1.times, tfr2.times, rtol=1e-5):
        return False
    return True


def validate_subjects_for_group_analysis(subjects: List[str], logger: logging.Logger, config=None) -> bool:
    if config is None:
        config = load_settings()
    min_subjects_for_group = config.get("analysis.min_subjects_for_group", 2)
    if subjects is None or len(subjects) < min_subjects_for_group:
        n_found = len(subjects) if subjects else 0
        logger.info(
            f"Group-level aggregation requires at least {min_subjects_for_group} subjects. "
            f"Found {n_found} subject(s); skipping group-level aggregation."
        )
        return False
    return True


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


def aggregate_feature_stats(subjects: List[str], task: str, deriv_root: Path, config) -> None:
    logger = get_group_logger("feature_engineering", config=config)
    
    if not validate_subjects_for_group_analysis(subjects, logger, config):
        return
    
    group_plots = deriv_group_plots_path(deriv_root, "02_feature_engineering")
    group_stats = deriv_group_stats_path(deriv_root)
    ensure_dir(group_plots)
    ensure_dir(group_stats)
    
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
    
    means_df = compute_subject_band_means(subject_features, bands)
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


###################################################################
# Behavior Correlation Aggregation
###################################################################


def load_roi_correlations(subjects: List[str], deriv_root: Path) -> Dict[Tuple[str, str], List[float]]:
    if not subjects:
        return {}
    
    correlations_by_key = {}
    for subject in subjects:
        corr_path = deriv_stats_path(deriv_root, subject) / "corr_stats_pow_roi_vs_rating.tsv"
        if not corr_path.exists():
            continue
        
        df = read_tsv(corr_path)
        for _, row in df.iterrows():
            key = (str(row.get("roi")), str(row.get("band")))
            r_value = row.get("r")
            try:
                r_value = _safe_float(r_value)
            except (ValueError, TypeError):
                r_value = np.nan
            correlations_by_key.setdefault(key, []).append(r_value)
    
    return correlations_by_key



def aggregate_roi_correlations(correlations_by_key: Dict[Tuple[str, str], List[float]], 
                                 config) -> pd.DataFrame:
    if not correlations_by_key:
        return pd.DataFrame()
    
    records = []
    for (roi, band), r_values in correlations_by_key.items():
        r_group, ci_low, ci_high, n = _fisher_aggregate(r_values)
        p_value = compute_correlation_pvalue(r_values)
        
        records.append({
            "roi": roi,
            "band": band,
            "r_group": r_group,
            "r_ci_low": ci_low,
            "r_ci_high": ci_high,
            "n_subjects": n,
            "p_group": p_value,
        })
    
    return pd.DataFrame(records)


def _apply_fdr_to_dataframe(
    df: pd.DataFrame,
    p_column: str,
    alpha: float,
    group_by_column: Optional[str] = None,
) -> pd.DataFrame:
    if df.empty or p_column not in df.columns:
        return df
    
    if group_by_column is None or group_by_column not in df.columns:
        reject, crit = fdr_bh_reject(df[p_column].to_numpy(), alpha=alpha)
        df = df.copy()
        df["fdr_reject"] = reject
        df["fdr_crit_p"] = crit
        return df
    
    output_rows = []
    for group_value in sorted(df[group_by_column].unique()):
        df_group = df[df[group_by_column] == group_value].copy()
        reject, crit = fdr_bh_reject(df_group[p_column].to_numpy(), alpha=alpha)
        df_group["fdr_reject"] = reject
        df_group["fdr_crit_p"] = crit
        output_rows.append(df_group)
    
    return pd.concat(output_rows, ignore_index=True)


def apply_fdr_correction_by_band(df: pd.DataFrame, config) -> pd.DataFrame:
    if df.empty or "band" not in df.columns or "p_group" not in df.columns:
        return df
    
    fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
    return _apply_fdr_to_dataframe(df, "p_group", fdr_alpha, group_by_column="band")



def plot_roi_correlations(df: pd.DataFrame, output_path: Path, config) -> None:
    if df.empty or "band" not in df.columns:
        return
    
    for band in sorted(df["band"].unique()):
        df_band = df[df["band"] == band]
        if df_band.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(6, 3.2))
        
        roi_order = sorted(df_band["roi"].unique())
        sns.barplot(data=df_band, x="roi", y="r_group", order=roi_order, color="steelblue", ax=ax)
        
        for idx, roi in enumerate(roi_order):
            row = df_band[df_band["roi"] == roi].iloc[0]
            y_value = row["r_group"]
            yerr_low = y_value - row["r_ci_low"] if np.isfinite(row["r_ci_low"]) else 0
            yerr_high = row["r_ci_high"] - y_value if np.isfinite(row["r_ci_high"]) else 0
            ax.errorbar(idx, y_value, yerr=[[yerr_low], [yerr_high]], fmt="none", ecolor="k", capsize=3)
        
        ax.set_ylabel("Group r (Fisher back-transformed)")
        ax.set_xlabel("ROI")
        ax.set_title(f"Group ROI power vs rating: {format_band_label(band, config)}")
        ax.axhline(0, color="k", linewidth=0.8)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")
            fig.tight_layout()
        
        save_formats = config.get("output.save_formats", ["svg"])
        save_fig(
            fig, output_path / f"group_roi_power_vs_rating_{sanitize_label(band)}",
            formats=save_formats, bbox_inches="tight", footer=_get_behavior_footer(config)
        )
        plt.close(fig)


def aggregate_connectivity_roi_group(roi_i, roi_j, group: pd.DataFrame, config) -> Dict:
    if group.empty or "r" not in group.columns:
        return {
            "roi_i": roi_i,
            "roi_j": roi_j,
            "summary_type": "within" if roi_i == roi_j else "between",
            "r_group": np.nan,
            "r_ci_low": np.nan,
            "r_ci_high": np.nan,
            "n_subjects": 0,
            "p_group": np.nan,
        }
    
    r_values = group["r"].to_numpy(dtype=float)
    r_group, ci_low, ci_high, n = _fisher_aggregate(r_values.tolist(), config=config)
    
    corr_clip_low = config.get("behavior_analysis.group_aggregation.correlation_clip_low", -0.999999)
    corr_clip_high = config.get("behavior_analysis.group_aggregation.correlation_clip_high", 0.999999)
    min_samples_for_stats = config.get("behavior_analysis.group_aggregation.min_samples_for_stats", 2)
    valid_values = np.clip(r_values[np.isfinite(r_values)], corr_clip_low, corr_clip_high)
    if valid_values.size >= min_samples_for_stats:
        z_scores = np.arctanh(valid_values)
        ttest_result = stats.ttest_1samp(z_scores, popmean=0.0)
        p_value = _get_ttest_pvalue(ttest_result)
    else:
        p_value = np.nan
    
    return {
        "roi_i": roi_i,
        "roi_j": roi_j,
        "summary_type": "within" if roi_i == roi_j else "between",
        "r_group": r_group,
        "r_ci_low": ci_low,
        "r_ci_high": ci_high,
        "n_subjects": n,
        "p_group": p_value,
    }


def aggregate_connectivity_correlations(subjects: List[str], deriv_root: Path, 
                                        group_stats: Path, config, logger: logging.Logger) -> None:
    connectivity_by_measure = load_connectivity_files(subjects, deriv_root)
    
    if not connectivity_by_measure:
        return
    
    for measure_band, dfs in connectivity_by_measure.items():
        concatenated = pd.concat(dfs, ignore_index=True)
        
        output_rows = []
        for (roi_i, roi_j), group in concatenated.groupby(["roi_i", "roi_j"], dropna=False):
            row = aggregate_connectivity_roi_group(roi_i, roi_j, group, config)
            row["measure_band"] = measure_band
            output_rows.append(row)
        
        output_df = pd.DataFrame(output_rows)
        fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
        output_df = _apply_fdr_to_dataframe(output_df, "p_group", fdr_alpha)
        
        output_path = group_stats / f"group_corr_conn_roi_summary_{sanitize_label(measure_band)}_vs_rating.tsv"
        write_tsv(output_df, output_path)
        logger.info(f"Wrote group connectivity ROI summary: {output_path}")


def process_roi_correlations(subjects: List[str], deriv_root: Path, group_stats: Path,
                             group_plots: Path, config, logger: logging.Logger) -> None:
    correlations_by_key = load_roi_correlations(subjects, deriv_root)
    if not correlations_by_key:
        return
    
    df_grouped = aggregate_roi_correlations(correlations_by_key, config)
    df_fdr = apply_fdr_correction_by_band(df_grouped, config)
    
    output_path = group_stats / "group_corr_pow_roi_vs_rating.tsv"
    write_tsv(df_fdr, output_path)
    logger.info(f"Wrote group ROI power vs rating summary: {output_path}")
    
    plot_roi_correlations(df_fdr, group_plots, config)


def aggregate_behavior_correlations(
    subjects: Optional[List[str]] = None,
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    *,
    pooling_strategy: str = "within_subject_centered",
    cluster_bootstrap: int = 0,
    subject_fixed_effects: bool = True,
    config=None,
) -> None:
    if config is None:
        config = load_settings()
    
    if task is None:
        task = config.task
    if deriv_root is None:
        deriv_root = Path(config.deriv_root)
    
    if subjects is None or subjects == ["all"]:
        subjects = get_available_subjects(task=task, config=config)
    
    group_stats = deriv_group_stats_path(deriv_root)
    plot_subdir = config.get("plotting.behavioral.plot_subdir", "04_behavior_correlations")
    group_plots = deriv_group_plots_path(deriv_root, subdir=plot_subdir)
    ensure_dir(group_stats)
    ensure_dir(group_plots)
    
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_group_logger("behavior_analysis", log_name, config=config)
    logger.info(f"Starting group aggregation for {len(subjects)} subjects (task={task})")
    
    process_roi_correlations(subjects, deriv_root, group_stats, group_plots, config, logger)
    aggregate_connectivity_correlations(subjects, deriv_root, group_stats, config, logger)
    
    logger.info("Generating pooled ROI scatters across subjects…")
    partial_covars = config.get("behavior_analysis.statistics.partial_covariates", [])
    scatter_inputs = collect_group_power_roi_inputs(
        subjects, task, deriv_root, config,
        partial_covars=partial_covars, do_temp=True,
    )
    plot_group_power_roi_scatter(
        scatter_inputs, config=config, pooling_strategy=pooling_strategy,
        cluster_bootstrap=int(cluster_bootstrap), subject_fixed_effects=bool(subject_fixed_effects),
        do_temp=True, bootstrap_ci=0, rng=None,
    )
    
    aggregate_channel_level_visuals(subjects, task, logger, deriv_root, config)
    
    aggregate_overall_band_summary(
        subjects, task, pooling_strategy, deriv_root, config
    )


###################################################################
# Channel-Level Visuals
###################################################################



def collect_channel_correlations(subjects: List[str], bands: List[str], deriv_root: Path,
                                 correlation_type: str, config=None) -> Dict[str, Dict[str, List[float]]]:
    if not subjects or not bands:
        return {}
    
    if config is None:
        config = load_settings()
    corr_clip_low = config.get("behavior_analysis.group_aggregation.correlation_clip_low", -0.999999)
    corr_clip_high = config.get("behavior_analysis.group_aggregation.correlation_clip_high", 0.999999)
    
    channel_correlations = {}
    for band in bands:
        correlations_by_channel = {}
        for subject in subjects:
            df = load_channel_correlations(subject, band, deriv_root, correlation_type)
            if df is None:
                continue
            
            for _, row in df.iterrows():
                channel = str(row.get("channel"))
                try:
                    r_value = _safe_float(row.get("r"))
                except (TypeError, ValueError):
                    r_value = np.nan
                
                if np.isfinite(r_value):
                    clipped_r = float(np.clip(r_value, corr_clip_low, corr_clip_high))
                    correlations_by_channel.setdefault(channel, []).append(clipped_r)
        
        if correlations_by_channel:
            channel_correlations[band] = correlations_by_channel
    
    return channel_correlations




def aggregate_channel_statistics(subjects: List[str], bands: List[str], deriv_root: Path,
                                  group_stats: Path, config, correlation_type: str) -> Dict[str, pd.DataFrame]:
    if not subjects or not bands:
        return {}
    
    channel_correlations = collect_channel_correlations(subjects, bands, deriv_root, correlation_type, config=config)
    bands_to_dataframes = {}
    
    for band, correlations_by_channel in channel_correlations.items():
        df_band = compute_group_channel_statistics(correlations_by_channel)
        
        if df_band.empty:
            continue
        
        df_band["band"] = band
        fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
        reject, crit = fdr_bh_reject(df_band["p_group"].to_numpy(), alpha=fdr_alpha)
        df_band["fdr_reject"] = reject
        df_band["fdr_crit_p"] = crit
        df_band = df_band.sort_values("channel").reset_index(drop=True)
        
        suffix = "rating" if correlation_type == "rating" else "temp"
        output_path = group_stats / f"group_corr_pow_{sanitize_label(band)}_vs_{suffix}.tsv"
        write_tsv(df_band, output_path)
        bands_to_dataframes[band] = df_band
    
    return bands_to_dataframes




def plot_channel_correlations(df: pd.DataFrame, band: str, output_path: Path, 
                               config, correlation_type: str) -> None:
    if df.empty or "p_group" not in df.columns or "r_group" not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    x_positions = np.arange(len(df))
    
    fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
    colors = [
        "red" if (np.isfinite(p) and p < fdr_alpha) else "lightblue" 
        for p in df["p_group"]
    ]
    
    ax.bar(x_positions, df["r_group"], color=colors)
    ax.set_xlabel("Channel", fontweight="bold")
    ax.set_ylabel("Spearman ρ", fontweight="bold")
    
    title_suffix, filename_suffix = get_correlation_type_labels(correlation_type)
    ax.set_title(
        f"{band.upper()} Band - Channel-wise Correlations with {title_suffix}\nGroup",
        fontweight="bold", fontsize=14
    )
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df["channel"].tolist(), rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    ax.axhline(y=0.3, color="green", linestyle="--", alpha=0.7)
    ax.axhline(y=-0.3, color="green", linestyle="--", alpha=0.7)
    
    sig_count = int((df["p_group"] < fdr_alpha).sum())
    ax.text(
        0.02, 0.98, f"Significant channels: {sig_count}/{len(df)}",
        transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )
    
    plt.tight_layout()
    save_formats = config.get("output.save_formats", ["svg"])
    save_fig(
        fig, output_path / f"group_power_{filename_suffix}_correlation_{sanitize_label(band)}",
        formats=save_formats, bbox_inches="tight", footer=_get_behavior_footer(config)
    )
    plt.close(fig)


def plot_top_predictors(df_sig: pd.DataFrame, top_n: int, output_path: Path,
                         config, correlation_type: str) -> None:
    if df_sig.empty or top_n <= 0:
        return
    
    df_sig = df_sig.copy()
    df_sig["abs_r"] = df_sig["r_group"].abs()
    df_top = df_sig.nlargest(top_n, "abs_r").copy()
    df_top = df_top.sort_values("abs_r", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
    labels = [f"{ch} ({band})" for ch, band in zip(df_top["channel"], df_top["band"])]
    y_positions = np.arange(len(df_top))
    colors = [_get_band_color(band, config) for band in df_top["band"]]
    
    ax.barh(y_positions, df_top["abs_r"], color=colors, alpha=0.85, 
           edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=11)
    
    fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
    title_suffix, filename_suffix = get_correlation_type_labels(correlation_type)
    title_suffix = f"{title_suffix}al" if correlation_type == "rating" else title_suffix
    
    ax.set_xlabel(f"|Spearman ρ| (p < {fdr_alpha})", fontweight='bold', fontsize=12)
    ax.set_title(
        f"Top {top_n} Significant {title_suffix} Predictors — Group",
        fontweight='bold', fontsize=14, pad=20
    )
    
    for idx, (_, row) in enumerate(df_top.iterrows()):
        ax.text(
            row["abs_r"] + 0.01, idx, 
            f"{row['abs_r']:.3f} (p={row['p_group']:.3f})",
            va='center', ha='left', fontsize=10
        )
    
    ax.set_xlim(0, _safe_float(df_top["abs_r"].max()) * 1.25)
    ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    
    save_formats = config.get("output.save_formats", ["svg"])
    filename_suffix = f"{filename_suffix}al" if correlation_type == "rating" else filename_suffix
    save_fig(
        fig, output_path / f"group_top_{top_n}_{filename_suffix}_predictors",
        formats=save_formats, bbox_inches="tight", footer=_get_behavior_footer(config)
    )
    plt.close(fig)
    
    export_cols = ["channel", "band", "r_group", "p_group", "n_subjects", "abs_r"]
    df_export = df_top[export_cols].sort_values("abs_r", ascending=False)
    write_tsv(df_export, output_path / f"group_top_{top_n}_{filename_suffix}_predictors.tsv")


def plot_group_topomap(bands_data: List[Dict], subjects: List[str], task: str,
                        output_path: Path, config, title_suffix: str, filename_suffix: str,
                        deriv_root: Path) -> None:
    if not bands_data or not subjects:
        return
    
    info = None
    for subject in subjects:
        epochs_path = _find_clean_epochs_path(subject, task, deriv_root=deriv_root)
        if epochs_path and Path(epochs_path).exists():
            info = mne.read_epochs(epochs_path, preload=False, verbose=False).info
            break
    
    if not info:
        return
    
    n_bands = len(bands_data)
    fig, axes = plt.subplots(1, n_bands, figsize=(4.8 * n_bands, 4.8))
    if n_bands == 1:
        axes = [axes]
    plt.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.16, wspace=0.08)
    
    all_correlations = []
    for band_data in bands_data:
        sig_corrs = band_data["correlations"][band_data["significant_mask"]]
        all_correlations.extend(sig_corrs)
    
    topomap_default_vmax = config.get("behavior_analysis.group_aggregation.topomap_default_vmax", 0.5)
    vmax = np.max(np.abs([c for c in all_correlations if np.isfinite(c)])) if all_correlations else topomap_default_vmax
    
    successful_plots = []
    for idx, band_data in enumerate(bands_data):
        ax = axes[idx]
        picks = mne.pick_types(info, meg=False, eeg=True, exclude='bads')
        if not picks:
            continue
        
        n_channels = len(info['ch_names'])
        topo_data = np.zeros(n_channels)
        topo_mask = np.zeros(n_channels, dtype=bool)
        
        for j, channel in enumerate(info['ch_names']):
            if channel in band_data['channels']:
                channel_idx = band_data['channels'].index(channel)
                corr_value = band_data['correlations'][channel_idx]
                topo_data[j] = corr_value if np.isfinite(corr_value) else 0.0
                topo_mask[j] = bool(band_data['significant_mask'][channel_idx])
        
        im, _ = mne.viz.plot_topomap(
            topo_data[picks], mne.pick_info(info, picks), axes=ax, show=False,
            cmap='RdBu_r', vlim=(-vmax, vmax), contours=6,
            mask=topo_mask[picks],
            mask_params=dict(marker='o', markerfacecolor='white', 
                           markeredgecolor='black', linewidth=1, markersize=6)
        )
        successful_plots.append(im)
        
        n_sig = int(topo_mask[picks].sum())
        n_total = sum(1 for ch in band_data['channels'] if ch in info['ch_names'])
        ax.set_title(f"{band_data['band'].upper()}\n{n_sig}/{n_total} significant",
                    fontweight='bold', fontsize=12, pad=10)
    
    fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
    plt.suptitle(f"Group Significant EEG-{title_suffix} Correlations (p < {fdr_alpha})",
                fontweight='bold', fontsize=14, y=0.97)
    
    if successful_plots:
        ax_positions = [ax.get_position() for ax in axes]
        left = min(pos.x0 for pos in ax_positions)
        right = max(pos.x1 for pos in ax_positions)
        bottom = min(pos.y0 for pos in ax_positions)
        span = right - left
        cax = fig.add_axes([left + 0.225 * span, max(0.04, bottom - 0.06), 0.55 * span, 0.028])
        cbar = fig.colorbar(successful_plots[-1], cax=cax, orientation='horizontal')
        cbar.set_label('Correlation (ρ)', fontweight='bold', fontsize=11)
        cbar.ax.tick_params(pad=2, labelsize=9)
    
    save_formats = config.get("output.save_formats", ["svg"])
    save_fig(fig, output_path / f"group_significant_correlations_topomap_{filename_suffix}",
            formats=save_formats, bbox_inches="tight", footer=_get_behavior_footer(config))
    plt.close(fig)




def save_combined_statistics(bands_to_df: Dict[str, pd.DataFrame], 
                              output_path: Path, suffix: str) -> pd.DataFrame:
    if not bands_to_df:
        return pd.DataFrame()
    
    combined = pd.concat([bands_to_df[b] for b in bands_to_df.keys()], ignore_index=True)
    write_tsv(combined, output_path / f"group_corr_pow_combined_vs_{suffix}.tsv")
    return combined

def plot_significant_predictors(combined_df: pd.DataFrame, top_n: int, 
                                 group_plots: Path, config, correlation_type: str,
                                 logger: logging.Logger) -> None:
    if combined_df.empty:
        logger.info(f"No significant group-level {correlation_type} predictors to plot for top-N.")
        return
    
    fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
    df_sig = combined_df[
        (combined_df["p_group"] <= fdr_alpha) & combined_df["r_group"].notna()
    ].copy()
    
    if df_sig.empty:
        logger.info(f"No significant group-level {correlation_type} predictors to plot for top-N.")
        return
    
    plot_top_predictors(df_sig, top_n, group_plots, config, correlation_type)


def aggregate_channel_level_visuals(subjects: List[str], task: str, logger: logging.Logger, 
                                    deriv_root: Path, config) -> None:
    group_stats = deriv_group_stats_path(deriv_root)
    plot_subdir = config.get("plotting.behavioral.plot_subdir", "04_behavior_correlations")
    group_plots = deriv_group_plots_path(deriv_root, subdir=plot_subdir)
    ensure_dir(group_stats)
    ensure_dir(group_plots)
    
    power_bands = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    bands = list(power_bands)
    
    bands_to_df_rating = aggregate_channel_statistics(subjects, bands, deriv_root, group_stats, config, "rating")
    bands_to_df_temp = aggregate_channel_statistics(subjects, bands, deriv_root, group_stats, config, "temp")
    
    if not bands_to_df_rating and not bands_to_df_temp:
        logger.warning("No group channel-level stats aggregated (missing subject inputs?)")
        return
    
    combined_rating = save_combined_statistics(bands_to_df_rating, group_stats, "rating")
    combined_temp = save_combined_statistics(bands_to_df_temp, group_stats, "temp")
    
    for band, df_band in bands_to_df_rating.items():
        plot_channel_correlations(df_band, band, group_plots, config, "rating")
    
    for band, df_band in bands_to_df_temp.items():
        plot_channel_correlations(df_band, band, group_plots, config, "temp")
    
    top_n = int(config.get("behavior_analysis.predictors.top_n", 20))
    fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
    
    plot_significant_predictors(combined_rating, top_n, group_plots, config, "rating", logger)
    plot_significant_predictors(combined_temp, top_n, group_plots, config, "temp", logger)
    
    if bands_to_df_rating:
        bands_data_rating = prepare_bands_data_for_topomap(bands_to_df_rating, fdr_alpha)
        plot_group_topomap(bands_data_rating, subjects, task, group_plots, config, "Pain", "", deriv_root)
    
    if bands_to_df_temp:
        bands_data_temp = prepare_bands_data_for_topomap(bands_to_df_temp, fdr_alpha)
        plot_group_topomap(bands_data_temp, subjects, task, group_plots, config, "Temperature", "temperature", deriv_root)


###################################################################
# Overall Band Summary
###################################################################


def plot_band_correlation(x_values: pd.Series, y_values: pd.Series, band: str,
                           output_path: Path, config, ylabel: str, title: str) -> None:
    if x_values.empty or y_values.empty:
        return
    
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    
    sns.regplot(
        x=x_values, y=y_values, ax=ax, ci=95,
        scatter_kws={
            "s": 25, "alpha": 0.7, "color": _get_band_color(band, config),
            "edgecolor": "white", "linewidths": 0.3
        },
        line_kws={"color": "#666666", "lw": 1.5}
    )
    
    ax.set_xlabel(f"{band.capitalize()} Power\nlog10(power/baseline)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    r, p = stats.spearmanr(x_values, y_values, nan_policy="omit")
    ax.text(
        0.02, 0.98, f"Spearman ρ={r:.3f}\np={p:.3f}\nn={len(x_values)}",
        transform=ax.transAxes, va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_formats = config.get("output.save_formats", ["svg"])
    save_fig(
        fig, output_path, formats=save_formats, bbox_inches="tight",
        footer=_get_behavior_footer(config)
    )
    plt.close(fig)


def get_temperature_ylabel(pooling_strategy: str) -> str:
    if pooling_strategy == "within_subject_centered":
        return "Temperature (°C, centered)"
    if pooling_strategy == "within_subject_zscored":
        return "Temperature (z-scored)"
    return "Temperature (°C)"


def plot_band_correlations_for_type(x_data: Dict[str, List[np.ndarray]], 
                                     y_data: Dict[str, List[np.ndarray]],
                                     bands: List[str], group_plots: Path,
                                     pooling_strategy: str, config,
                                     correlation_type: str) -> None:
    if not bands or not x_data or not y_data:
        return
    
    for band in bands:
        x_lists = x_data.get(band, [])
        y_lists = y_data.get(band, [])
        if not x_lists or not y_lists:
            continue
        
        x_pooled, y_pooled = pool_data_by_strategy(x_lists, y_lists, pooling_strategy)
        min_samples_for_plot = config.get("plotting.validation.min_samples_for_plot", 5)
        if len(x_pooled) < min_samples_for_plot:
            continue
        
        if correlation_type == "rating":
            ylabel = "Rating"
            title = f"{band.capitalize()} vs Rating"
            filename = f"group_power_behavior_correlation_{sanitize_label(band)}"
        else:
            ylabel = get_temperature_ylabel(pooling_strategy)
            title = f"{band.capitalize()} vs Temp"
            filename = f"group_power_temperature_correlation_{sanitize_label(band)}"
        
        output_path = group_plots / filename
        plot_band_correlation(x_pooled, y_pooled, band, output_path, config, ylabel, title)


def aggregate_overall_band_summary(subjects: List[str], task: str, pooling_strategy: str, 
                                    deriv_root: Path, config) -> None:
    rating_x, rating_y, temp_x, temp_y, has_temperature = load_subject_data_for_summary(
        subjects, task, deriv_root, config
    )
    
    power_bands = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    plot_subdir = config.get("plotting.behavioral.plot_subdir", "04_behavior_correlations")
    group_plots = deriv_group_plots_path(deriv_root, subdir=plot_subdir)
    
    plot_band_correlations_for_type(
        rating_x, rating_y, power_bands, group_plots, pooling_strategy, config, "rating"
    )
    
    if has_temperature and temp_x:
        plot_band_correlations_for_type(
            temp_x, temp_y, power_bands, group_plots, pooling_strategy, config, "temperature"
        )
