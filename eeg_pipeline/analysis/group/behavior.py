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

from eeg_pipeline.utils.config.loader import load_settings
from eeg_pipeline.utils.data.loading import (
    get_available_subjects,
    load_subject_data_for_summary,
    load_channel_correlations,
    load_connectivity_files,
    load_epochs_for_analysis,
)
from eeg_pipeline.utils.io.general import (
    deriv_stats_path,
    deriv_features_path,
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
from eeg_pipeline.utils.analysis.stats import (
    _safe_float,
    _get_ttest_pvalue,
    fisher_aggregate as _fisher_aggregate,
    weighted_fisher_aggregate,
    compute_correlation_pvalue,
    compute_group_channel_statistics,
    fdr_bh,
)
from eeg_pipeline.utils.analysis.tfr import (
    prepare_bands_data_for_topomap,
)
from eeg_pipeline.plotting.behavioral import (
    plot_group_power_roi_scatter,
)


###################################################################
# ROI Correlation Aggregation
###################################################################


def load_roi_correlations(subjects: List[str], deriv_root: Path) -> Dict[Tuple[str, str], List[Tuple[float, float]]]:
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
            n_value = row.get("n")
            try:
                r_value = _safe_float(r_value)
            except (ValueError, TypeError):
                r_value = np.nan
            try:
                n_value = int(n_value)
            except (TypeError, ValueError):
                n_value = np.nan
            correlations_by_key.setdefault(key, []).append((r_value, n_value))
    
    return correlations_by_key


def aggregate_roi_correlations(correlations_by_key: Dict[Tuple[str, str], List[Tuple[float, float]]], 
                                 config) -> pd.DataFrame:
    if not correlations_by_key:
        return pd.DataFrame()
    
    records = []
    for (roi, band), r_n_values in correlations_by_key.items():
        r_values = [v for v, _ in r_n_values]
        n_values = [n for _, n in r_n_values]
        r_group, ci_low, ci_high, n_eff, z_stat = weighted_fisher_aggregate(r_values, n_values, config=config)
        if np.isfinite(z_stat):
            p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
        else:
            p_value = compute_correlation_pvalue(r_values)
        
        records.append({
            "roi": roi,
            "band": band,
            "r_group": r_group,
            "r_ci_low": ci_low,
            "r_ci_high": ci_high,
            "n_subjects": len([v for v in r_values if np.isfinite(v)]),
            "n_trials_effective": n_eff,
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


def apply_global_fdr_correction(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Applies FDR Benjamini-Hochberg correction across ALL tests (Bands x ROIs).
    Prevents family-wise error inflation that occurs when correcting per-band.
    """
    if df.empty or "p_group" not in df.columns:
        return df
    
    fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
    return _apply_fdr_to_dataframe(df, "p_group", fdr_alpha, group_by_column=None)


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
    
    all_results = []
    
    for measure_band, dfs in connectivity_by_measure.items():
        concatenated = pd.concat(dfs, ignore_index=True)
        for (roi_i, roi_j), group in concatenated.groupby(["roi_i", "roi_j"], dropna=False):
            row = aggregate_connectivity_roi_group(roi_i, roi_j, group, config)
            row["measure_band"] = measure_band
            all_results.append(row)
            
    if not all_results:
        return

    combined_df = pd.DataFrame(all_results)
    fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
    combined_df = _apply_fdr_to_dataframe(combined_df, "p_group", fdr_alpha, group_by_column=None)
    
    for measure_band, group_df in combined_df.groupby("measure_band"):
        output_path = group_stats / f"group_corr_conn_roi_summary_{sanitize_label(measure_band)}_vs_rating.tsv"
        write_tsv(group_df, output_path)
        logger.info(f"Wrote group connectivity ROI summary: {output_path}")


def process_roi_correlations(subjects: List[str], deriv_root: Path, group_stats: Path,
                             group_plots: Path, config, logger: logging.Logger) -> None:
    correlations_by_key = load_roi_correlations(subjects, deriv_root)
    if not correlations_by_key:
        return
    
    df_grouped = aggregate_roi_correlations(correlations_by_key, config)
    df_fdr = apply_global_fdr_correction(df_grouped, config)
    
    output_path = group_stats / "group_corr_pow_roi_vs_rating.tsv"
    write_tsv(df_fdr, output_path)
    logger.info(f"Wrote group ROI power vs rating summary: {output_path}")
    
    plot_roi_correlations(df_fdr, group_plots, config)


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
    all_band_rows: List[pd.DataFrame] = []
    
    for band, correlations_by_channel in channel_correlations.items():
        df_band = compute_group_channel_statistics(correlations_by_channel)
        if df_band.empty:
            continue
        df_band["band"] = band
        all_band_rows.append(df_band)
    
    if not all_band_rows:
        return {}
    
    combined = pd.concat(all_band_rows, ignore_index=True)
    fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
    reject, crit = fdr_bh_reject(combined["p_group"].to_numpy(), alpha=fdr_alpha)
    combined["fdr_reject"] = reject
    combined["fdr_crit_p"] = crit
    
    bands_to_dataframes = {}
    suffix = "rating" if correlation_type == "rating" else "temp"
    for band, df_band in combined.groupby("band"):
        df_band = df_band.sort_values("channel").reset_index(drop=True)
        output_path = group_stats / f"group_corr_pow_{sanitize_label(band)}_vs_{suffix}.tsv"
        write_tsv(df_band, output_path)
        bands_to_dataframes[band] = df_band
    
    return bands_to_dataframes


def _load_example_info_for_group(subjects: List[str], task: str, deriv_root: Path, config, logger) -> Optional[mne.Info]:
    for subj in subjects:
        epochs, _ = load_epochs_for_analysis(
            subj,
            task,
            align="strict",
            preload=False,
            deriv_root=deriv_root,
            bids_root=config.bids_root,
            config=config,
            logger=logger,
        )
        if epochs is not None:
            return epochs.info
    return None


def _plot_group_channel_topomap(df_stats: pd.DataFrame, info: mne.Info, title: str, output_path: Path, config) -> None:
    if info is None or df_stats.empty:
        return
    if "channel" not in df_stats.columns or "r_group" not in df_stats.columns:
        return

    ch_names = info.ch_names
    values = np.full(len(ch_names), np.nan, dtype=float)
    for _, row in df_stats.iterrows():
        ch = str(row.get("channel"))
        if ch in ch_names:
            idx = ch_names.index(ch)
            values[idx] = _safe_float(row.get("r_group"))

    fig, ax = plt.subplots(figsize=(6, 5))
    im, _ = mne.viz.plot_topomap(
        values,
        info,
        axes=ax,
        show=False,
        cmap="RdBu_r",
        vlim=(-np.nanmax(np.abs(values)), np.nanmax(np.abs(values))),
        contours=6,
    )
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Group r")
    save_fig(fig, output_path)
    plt.close(fig)


def aggregate_aperiodic_correlations(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    group_stats: Path,
    group_plots: Path,
    config,
    logger: logging.Logger,
) -> None:
    if not subjects:
        return

    info_example = _load_example_info_for_group(subjects, task, deriv_root, config, logger)

    metrics = [
        ("aper_slope", "corr_stats_aper_slope_vs_rating.tsv"),
        ("aper_offset", "corr_stats_aper_offset_vs_rating.tsv"),
        ("powcorr", "corr_stats_powcorr_vs_rating.tsv"),
    ]

    for metric, filename in metrics:
        all_df = []
        for subj in subjects:
            fpath = deriv_stats_path(deriv_root, subj) / filename
            if not fpath.exists():
                continue
            try:
                df = read_tsv(fpath)
                if df is None or df.empty:
                    continue
                df["subject"] = subj
                all_df.append(df)
            except Exception as exc:
                logger.warning("Failed to load %s for sub-%s: %s", filename, subj, exc)
        if not all_df:
            continue

        combined = pd.concat(all_df, ignore_index=True)
        grouped_records = []

        if metric == "powcorr":
            group_key = "band"
        else:
            group_key = None

        group_iter = combined.groupby(group_key) if group_key else [(None, combined)]

        for key, df_sub in group_iter:
            correlations_by_channel: Dict[str, List[float]] = {}
            for channel, df_ch in df_sub.groupby("channel"):
                r_vals = pd.to_numeric(df_ch["r"], errors="coerce").to_numpy()
                finite = r_vals[np.isfinite(r_vals)]
                if finite.size == 0:
                    continue
                correlations_by_channel[channel] = finite.tolist()
            if not correlations_by_channel:
                continue
            df_stats = compute_group_channel_statistics(correlations_by_channel, config=config)
            if df_stats.empty:
                continue
            if key is not None:
                df_stats[group_key] = key
            grouped_records.append(df_stats)

        if not grouped_records:
            continue

        summary = pd.concat(grouped_records, ignore_index=True)
        alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
        reject, crit = fdr_bh_reject(summary["p_group"].to_numpy(), alpha=alpha)
        summary["fdr_reject"] = reject
        summary["fdr_crit_p"] = crit

        metric_label = metric
        out_file = group_stats / f"group_corr_{metric}_vs_rating.tsv"
        write_tsv(summary, out_file)
        logger.info("Wrote group aperiodic stats to %s", out_file)

        if info_example is not None:
            if group_key and group_key in summary.columns:
                for band_name, df_band in summary.groupby(group_key):
                    title = f"{metric} {band_name} vs rating (group r)"
                    out_plot = group_plots / f"group_{metric}_{sanitize_label(band_name)}_topomap.png"
                    _plot_group_channel_topomap(df_band, info_example, title, out_plot, config)
            else:
                title = f"{metric_label} vs rating (group r)"
                out_plot = group_plots / f"group_{metric}_topomap.png"
                _plot_group_channel_topomap(summary, info_example, title, out_plot, config)


def aggregate_aperiodic_run_drift(
    subjects: List[str],
    deriv_root: Path,
    group_plots: Path,
    config,
    logger: logging.Logger,
) -> None:
    if not subjects:
        return

    records = []
    for subj in subjects:
        qc_path = deriv_stats_path(deriv_root, subj) / "aperiodic_qc.npz"
        if not qc_path.exists():
            continue
        try:
            qc = np.load(qc_path, allow_pickle=True)
        except Exception as exc:
            logger.warning("Failed to load aperiodic QC for sub-%s: %s", subj, exc)
            continue
        run_labels = qc.get("run_labels")
        slopes = qc.get("slopes")
        offsets = qc.get("offsets")
        if run_labels is None or slopes is None or offsets is None:
            continue
        run_labels = np.asarray(run_labels)
        slopes = np.asarray(slopes)
        offsets = np.asarray(offsets)
        if slopes.ndim == 3:
            slopes = slopes.mean(axis=1)
        if offsets.ndim == 3:
            offsets = offsets.mean(axis=1)
        for run in np.unique(run_labels):
            mask = run_labels == run
            if not np.any(mask):
                continue
            records.append({"subject": subj, "run": str(run), "metric": "slope", "value": float(np.nanmean(slopes[mask]))})
            records.append({"subject": subj, "run": str(run), "metric": "offset", "value": float(np.nanmean(offsets[mask]))})

    if not records:
        return

    df = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, metric in zip(axes, ["slope", "offset"]):
        df_metric = df[df["metric"] == metric]
        if df_metric.empty:
            continue
        sns.lineplot(data=df_metric, x="run", y="value", hue="subject", estimator="mean", errorbar="sd", ax=ax, legend=False)
        ax.set_title(f"Aperiodic {metric} by run")
        ax.set_xlabel("Run")
        ax.set_ylabel(metric.capitalize())
    plt.tight_layout()
    out_path = group_plots / "group_aperiodic_run_drift.png"
    save_fig(fig, out_path)
    plt.close(fig)


def aggregate_itpc_statistics(
    subjects: List[str],
    deriv_root: Path,
    group_stats: Path,
    group_plots: Path,
    config,
    logger: logging.Logger,
) -> None:
    if not subjects:
        return

    all_records = []
    for subj in subjects:
        stats_path = deriv_stats_path(deriv_root, subj) / "corr_stats_itpc_vs_rating.tsv"
        if not stats_path.exists():
            continue
        try:
            df = read_tsv(stats_path)
            if df is None or df.empty:
                continue
            df["subject"] = subj
            all_records.append(df)
        except Exception as exc:
            logger.warning("Failed to load ITPC stats for sub-%s: %s", subj, exc)

    if not all_records:
        logger.warning("No ITPC correlation stats found across subjects; skipping ITPC aggregation")
        return

    combined = pd.concat(all_records, ignore_index=True)
    grouped_stats = []

    for (band, time_bin), df_sub in combined.groupby(["band", "time_bin"]):
        correlations_by_channel: Dict[str, List[float]] = {}
        for channel, df_ch in df_sub.groupby("channel"):
            r_vals = pd.to_numeric(df_ch["r"], errors="coerce").to_numpy()
            finite = r_vals[np.isfinite(r_vals)]
            if finite.size == 0:
                continue
            correlations_by_channel[channel] = finite.tolist()

        if not correlations_by_channel:
            continue

        df_stats = compute_group_channel_statistics(correlations_by_channel, config=config)
        if df_stats.empty:
            continue
        df_stats["band"] = band
        df_stats["time_bin"] = time_bin
        grouped_stats.append(df_stats)

    if not grouped_stats:
        logger.warning("No ITPC group statistics computed; skipping ITPC aggregation")
        return

    summary = pd.concat(grouped_stats, ignore_index=True)
    alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
    reject, crit = fdr_bh_reject(summary["p_group"].to_numpy(), alpha=alpha)
    summary["fdr_reject"] = reject
    summary["fdr_crit_p"] = crit

    output_path = group_stats / "group_corr_itpc_vs_rating.tsv"
    write_tsv(summary, output_path)
    logger.info("Wrote group ITPC channel statistics to %s", output_path)

    bin_summary = (
        summary.groupby(["band", "time_bin"])
        .agg(r_median=("r_group", "median"), frac_sig=("fdr_reject", "mean"))
        .reset_index()
    )
    pivot = bin_summary.pivot(index="band", columns="time_bin", values="r_median")
    frac_sig = bin_summary.pivot(index="band", columns="time_bin", values="frac_sig")

    fig, ax = plt.subplots(figsize=(1.5 + 1.4 * pivot.shape[1], 1.5 + 0.9 * pivot.shape[0]))
    sns.heatmap(
        pivot,
        ax=ax,
        annot=False,
        cmap="RdBu_r",
        center=0.0,
        cbar_kws={"label": "Median r (Fisher z back-transformed)"},
    )
    for (i, band) in enumerate(pivot.index):
        for (j, tbin) in enumerate(pivot.columns):
            try:
                frac = frac_sig.loc[band, tbin]
            except Exception:
                frac = np.nan
            if np.isfinite(frac) and frac > 0:
                ax.text(j + 0.5, i + 0.5, f"{frac:.2f}", ha="center", va="center", color="black", fontsize=8)
    ax.set_xlabel("Time bin")
    ax.set_ylabel("Band")
    ax.set_title("Group ITPC–rating correlations\n(value = median r across channels; numbers = frac sig)")
    plt.tight_layout()
    ensure_dir(group_plots)
    plot_path = group_plots / "group_itpc_correlation_heatmap.png"
    save_fig(ax.figure, plot_path)
    plt.close(fig)
    logger.info("Saved group ITPC heatmap to %s", plot_path)


def aggregate_pac_statistics(
    subjects: List[str],
    deriv_root: Path,
    group_stats: Path,
    group_plots: Path,
    config,
    logger: logging.Logger,
) -> None:
    if not subjects:
        return

    pac_records = []
    pac_time_records = []
    for subj in subjects:
        pac_path = deriv_features_path(deriv_root, subj) / "features_pac.tsv"
        pac_time_path = deriv_features_path(deriv_root, subj) / "features_pac_time.tsv"
        if pac_path.exists():
            try:
                df = read_tsv(pac_path)
                if df is not None and not df.empty:
                    df["subject"] = subj
                    pac_records.append(df)
            except Exception as exc:
                logger.warning("Failed to read PAC comodulograms for sub-%s: %s", subj, exc)
        if pac_time_path.exists():
            try:
                df_time = read_tsv(pac_time_path)
                if df_time is not None and not df_time.empty:
                    df_time["subject"] = subj
                    pac_time_records.append(df_time)
            except Exception as exc:
                logger.warning("Failed to read PAC time data for sub-%s: %s", subj, exc)

    if pac_records:
        pac_df = pd.concat(pac_records, ignore_index=True)
        summaries = []
        for roi, df_roi in pac_df.groupby("roi"):
            pivot_mean = df_roi.pivot_table(index="amp_freq", columns="phase_freq", values="pac", aggfunc="mean")
            p_vals = None
            if "p_perm" in df_roi.columns:
                def _combine_p(series):
                    vals = series.dropna().to_numpy(dtype=float)
                    if vals.size == 0:
                        return np.nan
                    try:
                        _, p_comb = stats.combine_pvalues(vals, method="fisher")
                        return float(p_comb)
                    except Exception:
                        return np.nan
                p_vals = df_roi.pivot_table(index="amp_freq", columns="phase_freq", values="p_perm", aggfunc=_combine_p)

            summary_rows = []
            for (amp, phase), sub_df in df_roi.groupby(["amp_freq", "phase_freq"]):
                pac_mean = np.nanmean(sub_df["pac"].to_numpy(dtype=float))
                p_comb = np.nan
                if "p_perm" in sub_df.columns:
                    try:
                        _, p_comb = stats.combine_pvalues(sub_df["p_perm"].dropna().to_numpy(dtype=float), method="fisher")
                    except Exception:
                        p_comb = np.nan
                summary_rows.append(
                    {
                        "roi": roi,
                        "phase_freq": float(phase),
                        "amp_freq": float(amp),
                        "pac_mean": pac_mean,
                        "p_fisher": p_comb,
                    }
                )

            summary_df = pd.DataFrame(summary_rows)
            if not summary_df.empty and "p_fisher" in summary_df.columns:
                q = fdr_bh(summary_df["p_fisher"].to_numpy(dtype=float), config=config)
                summary_df["q_fdr"] = q
            out_path = group_stats / f"group_pac_comod_{sanitize_label(roi)}.tsv"
            write_tsv(summary_df, out_path)
            summaries.append(summary_df)

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                pivot_mean,
                ax=ax,
                cmap="magma",
                cbar_kws={"label": "PAC (mean)"},
            )
            ax.set_title(f"{roi} PAC (group mean)")
            ax.set_xlabel("Phase freq (Hz)")
            ax.set_ylabel("Amp freq (Hz)")
            plt.tight_layout()
            ensure_dir(group_plots)
            plot_path = group_plots / f"group_pac_comod_{sanitize_label(roi)}.png"
            save_fig(fig, plot_path)
            plt.close(fig)
            logger.info("Saved group PAC comodulogram for %s", roi)

            if p_vals is not None:
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(
                    p_vals,
                    ax=ax,
                    cmap="Reds",
                    cbar_kws={"label": "Fisher-combined p_perm"},
                )
                ax.set_title(f"{roi} PAC significance (Fisher p)")
                ax.set_xlabel("Phase freq (Hz)")
                ax.set_ylabel("Amp freq (Hz)")
                plt.tight_layout()
                sig_path = group_plots / f"group_pac_comod_{sanitize_label(roi)}_p.png"
                save_fig(fig, sig_path)
                plt.close(fig)

    if pac_time_records:
        pac_time_df = pd.concat(pac_time_records, ignore_index=True)
        for roi, df_roi in pac_time_df.groupby("roi"):
            for phase_f, df_phase in df_roi.groupby("phase_freq"):
                pivot = df_phase.pivot_table(index="amp_freq", columns="time", values="pac", aggfunc="mean")
                fig, ax = plt.subplots(figsize=(8, 4))
                im = ax.imshow(
                    pivot.values,
                    aspect="auto",
                    origin="lower",
                    extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
                    cmap="magma",
                )
                ax.set_title(f"{roi} PAC time ribbon (phase {phase_f:.1f} Hz)")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amp freq (Hz)")
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label("PAC (mean)")
                plt.tight_layout()
                ribbon_path = group_plots / f"group_pac_time_roi-{sanitize_label(roi)}_phase-{phase_f:.1f}.png"
                save_fig(fig, ribbon_path)
                plt.close(fig)


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


def plot_band_correlation(subject_r: List[float], band: str,
                           output_path: Path, config, ylabel: str, title: str) -> None:
    r_vals = np.asarray(subject_r, dtype=float)
    r_vals = r_vals[np.isfinite(r_vals)]
    if r_vals.size == 0:
        return

    finite_for_fisher = np.clip(r_vals, -0.999999, 0.999999)
    z_scores = np.arctanh(finite_for_fisher)
    group_r = float(np.tanh(np.mean(z_scores)))
    if z_scores.size >= 2:
        ttest = stats.ttest_1samp(z_scores, popmean=0.0)
        p_group = float(_get_ttest_pvalue(ttest))
    else:
        p_group = np.nan

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    jitter = (np.random.default_rng(int(config.get("random.seed", 42))).random(r_vals.size) - 0.5) * 0.15
    ax.scatter(np.zeros_like(r_vals) + jitter, r_vals, color=_get_band_color(band, config), alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axhline(group_r, color="black", linestyle="--", linewidth=1.2, label=f"group r={group_r:.3f}")
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.8)

    ax.set_xlim(-0.4, 0.4)
    ax.set_ylabel("Subject Spearman ρ")
    ax.set_xticks([])
    ax.set_xlabel(ylabel)
    ax.set_title(title)

    text_lines = [f"n_subjects={r_vals.size}", f"group ρ={group_r:.3f}"]
    if np.isfinite(p_group):
        text_lines.append(f"p={p_group:.3f}")
    ax.text(0.02, 0.98, "\n".join(text_lines), transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower right")
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

        min_samples = int(config.get("plotting.validation.min_samples_for_plot", 5))
        subj_r: List[float] = []
        for x_arr, y_arr in zip(x_lists, y_lists):
            if x_arr is None or y_arr is None:
                continue
            if len(x_arr) != len(y_arr):
                continue
            if len(x_arr) < min_samples:
                continue
            r_subj, _ = stats.spearmanr(x_arr, y_arr, nan_policy="omit")
            if np.isfinite(r_subj):
                subj_r.append(float(r_subj))

        if not subj_r:
            continue

        if correlation_type == "rating":
            ylabel = "Rating correlations (per subject)"
            title = f"{band.capitalize()} vs Rating (subject-level r)"
            filename = f"group_power_behavior_correlation_{sanitize_label(band)}"
        else:
            ylabel = get_temperature_ylabel(pooling_strategy) + " correlations (per subject)"
            title = f"{band.capitalize()} vs Temp (subject-level r)"
            filename = f"group_power_temperature_correlation_{sanitize_label(band)}"
        
        output_path = group_plots / filename
        plot_band_correlation(subj_r, band, output_path, config, ylabel, title)


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


###################################################################
# Main Entry Point
###################################################################


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
    from eeg_pipeline.utils.data.loading import collect_group_power_roi_inputs
    
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
    aggregate_itpc_statistics(subjects, deriv_root, group_stats, group_plots, config, logger)
    aggregate_aperiodic_correlations(subjects, task, deriv_root, group_stats, group_plots, config, logger)
    aggregate_aperiodic_run_drift(subjects, deriv_root, group_plots, config, logger)
    aggregate_pac_statistics(subjects, deriv_root, group_stats, group_plots, config, logger)
    
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

