from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne

from eeg_pipeline.utils.config_loader import load_settings
from eeg_pipeline.utils.data_loading import get_available_subjects
from eeg_pipeline.utils.io_utils import (
    deriv_features_path,
    deriv_stats_path,
    deriv_group_stats_path,
    deriv_group_plots_path,
    ensure_dir,
    fdr_bh_reject,
    get_group_logger,
    get_subject_logger,
    save_fig,
    _find_clean_epochs_path,
)
from eeg_pipeline.utils.tfr_utils import (
    validate_baseline_indices,
    read_tfr_average_with_logratio,
    run_tfr_morlet,
    time_mask,
    find_tfr_path,
)
from eeg_pipeline.utils.stats_utils import (
    _safe_float,
    _get_ttest_pvalue,
    fisher_aggregate as _fisher_aggregate,
)
from eeg_pipeline.utils.io_utils import (
    get_band_color as _get_band_color,
    get_behavior_footer as _get_behavior_footer,
    sanitize_label,
)
from eeg_pipeline.plotting.plot_features import (
    plot_group_power_plots,
    plot_group_band_power_time_courses,
)
from eeg_pipeline.plotting.plot_behavior import (
    plot_group_power_roi_scatter,
)
from eeg_pipeline.analysis.behavior import collect_group_power_roi_inputs

PLOT_SUBDIR = "04_behavior_correlation_analysis"
PARTIAL_COVARS_DEFAULT: List[str] = []


###################################################################
# Helper Functions
###################################################################








###################################################################
# Feature Statistics Aggregation
###################################################################

def aggregate_feature_stats(subjects, task, deriv_root, config):
    if subjects is None or len(subjects) < 2:
        logger = get_group_logger("feature_engineering", config=config)
        logger.info(f"Group-level aggregation requires at least 2 subjects. Found {len(subjects) if subjects else 0} subject(s); skipping group-level aggregation.")
        return
    
    logger = get_group_logger("feature_engineering", config=config)
    gplots = deriv_group_plots_path(deriv_root, "02_feature_engineering")
    gstats = deriv_group_stats_path(deriv_root)
    ensure_dir(gplots)
    ensure_dir(gstats)

    subj_pow = {}
    for s in subjects:
        p = deriv_features_path(deriv_root, s) / "features_eeg_direct.tsv"
        if not p.exists():
            logger.warning(f"Missing features for sub-{s}: {p}")
            continue
        subj_pow[s] = pd.read_csv(p, sep="\t")

    if len(subj_pow) < 2:
        logger.warning(f"Group-level aggregation requires at least 2 subjects with valid feature files. Found {len(subj_pow)} subject(s) with features; skipping group-level aggregation.")
        return

    power_bands = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    bands = list(power_bands)

    band_channels = {}
    for b in bands:
        ch_union = set()
        for s, df in subj_pow.items():
            cols = [c for c in df.columns if c.startswith(f"pow_{b}_")]
            ch_union.update([c.replace(f"pow_{b}_", "") for c in cols])
        band_channels[b] = sorted(ch_union)

    heat_rows = []
    stats_rows = []
    all_ch_union = sorted(set().union(*band_channels.values())) if band_channels else []

    for b in bands:
        subj_means_per_ch = []
        for s, df in subj_pow.items():
            vals = []
            for ch in all_ch_union:
                col = f"pow_{b}_{ch}"
                if col in df.columns:
                    vals.append(float(pd.to_numeric(df[col], errors="coerce").mean()))
                else:
                    vals.append(np.nan)
            subj_means_per_ch.append(vals)
        arr = np.asarray(subj_means_per_ch, dtype=float)
        mean_across_subj = np.nanmean(arr, axis=0)
        heat_rows.append(mean_across_subj)
        n_eff = np.sum(np.isfinite(arr), axis=0)
        std_across_subj = np.nanstd(arr, axis=0, ddof=1)
        for j, ch in enumerate(all_ch_union):
            stats_rows.append({
                "band": b,
                "channel": ch,
                "mean": float(mean_across_subj[j]) if np.isfinite(mean_across_subj[j]) else np.nan,
                "std": float(std_across_subj[j]) if np.isfinite(std_across_subj[j]) else np.nan,
                "n_subjects": int(n_eff[j]),
            })

    plot_group_power_plots(subj_pow, bands, gplots, gstats, config, logger)

    recs: List[dict] = []
    for b in bands:
        for s, df in subj_pow.items():
            cols = [c for c in df.columns if c.startswith(f"pow_{b}_")]
            if not cols:
                continue
            vals = pd.to_numeric(df[cols].stack(), errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            recs.append({
                "subject": s,
                "band": b,
                "mean_power": float(np.mean(vals)),
            })
    dfm = pd.DataFrame(recs)
    if not dfm.empty:
        bands_present = [b for b in bands if b in set(dfm["band"])]
        means = []
        ci_l = []
        ci_h = []
        ns = []
        for b in bands_present:
            v = dfm[dfm["band"] == b]["mean_power"].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            mu = float(np.mean(v)) if v.size else np.nan
            se = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
            delta = 1.96 * se if np.isfinite(se) else np.nan
            means.append(mu)
            ci_l.append(mu - delta if np.isfinite(delta) else np.nan)
            ci_h.append(mu + delta if np.isfinite(delta) else np.nan)
            ns.append(len(v))
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(bands_present))
        ax.bar(x, means, color='steelblue', alpha=0.8)
        yerr = np.array([[mu - lo if np.isfinite(mu) and np.isfinite(lo) else 0 for lo, mu in zip(ci_l, means)],
                         [hi - mu if np.isfinite(mu) and np.isfinite(hi) else 0 for hi, mu in zip(ci_h, means)]])
        ax.errorbar(x, means, yerr=yerr, fmt='none', ecolor='k', capsize=3)
        for i, b in enumerate(bands_present):
            vals = dfm[dfm["band"] == b]["mean_power"].to_numpy(dtype=float)
            jitter = (np.random.rand(len(vals)) - 0.5) * 0.2
            ax.scatter(np.full_like(vals, i, dtype=float) + jitter, vals, color='k', s=12, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([bp.capitalize() for bp in bands_present])
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
            fig,
            gplots / "group_power_distributions_per_band_across_subjects",
            formats=tuple(formats),
            dpi=int(constants["FIG_DPI"]),
            bbox_inches=constants["output.bbox_inches"],
            pad_inches=constants["output.pad_inches"],
            constants=constants,
        )
        out = pd.DataFrame({
            "band": bands_present,
            "group_mean": means,
            "ci_low": ci_l,
            "ci_high": ci_h,
            "n_subjects": ns,
        })
        out.to_csv(gstats / "group_band_power_subject_means.tsv", sep="\t", index=False)
        logger.info("Saved group band power distributions and stats.")

    freq_bands = config.get("time_frequency_analysis.bands", {
        "delta": [1.0, 3.9],
        "theta": [4.0, 7.9],
        "alpha": [8.0, 12.9],
        "beta": [13.0, 30.0],
        "gamma": [30.1, 80.0],
    })
    FEATURES_FREQ_BANDS = {name: tuple(freqs) for name, freqs in freq_bands.items()}
    band_names = list(FEATURES_FREQ_BANDS.keys())
    m = len(band_names)
    per_subject_corrs: List[np.ndarray] = []
    for s, df in subj_pow.items():
        band_vecs: dict[str, dict[str, float]] = {}
        for b in band_names:
            cols = [c for c in df.columns if c.startswith(f"pow_{b}_")]
            if not cols:
                continue
            ser = df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=0)
            ch_means = {c.replace(f"pow_{b}_", ""): float(v) for c, v in ser.items() if np.isfinite(v)}
            if ch_means:
                band_vecs[b] = ch_means
        if len(band_vecs) < 2:
            continue
        corr_mat = np.eye(m, dtype=float)
        for i, bi in enumerate(band_names):
            for j, bj in enumerate(band_names):
                if j <= i:
                    continue
                di = band_vecs.get(bi)
                dj = band_vecs.get(bj)
                if di is None or dj is None:
                    corr = np.nan
                else:
                    common = sorted(set(di.keys()) & set(dj.keys()))
                    if len(common) < 2:
                        corr = np.nan
                    else:
                        vi = np.array([di[ch] for ch in common], dtype=float)
                        vj = np.array([dj[ch] for ch in common], dtype=float)
                        if np.std(vi) < 1e-12 or np.std(vj) < 1e-12:
                            corr = np.nan
                        else:
                            corr = float(np.corrcoef(vi, vj)[0, 1])
                corr_mat[i, j] = corr
                corr_mat[j, i] = corr
        per_subject_corrs.append(corr_mat)

    def _compute_avg_tfr_for_subject(subj: str) -> Optional["mne.time_frequency.AverageTFR"]:
        epo_path = _find_clean_epochs_path(subj, task, deriv_root=deriv_root)
        if epo_path is None or not epo_path.exists():
            return None
        epochs = mne.read_epochs(epo_path, preload=False, verbose=False)
        
        custom_freqs = config.get("time_frequency_analysis.tfr.custom_freqs", None)
        if custom_freqs is not None:
            freqs = np.array(custom_freqs)
        else:
            freq_min = config.get("time_frequency_analysis.tfr.freq_min", 1.0)
            freq_max = config.get("time_frequency_analysis.tfr.freq_max", 100.0)
            n_freqs = config.get("time_frequency_analysis.tfr.n_freqs", 40)
            freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
        
        from eeg_pipeline.utils.tfr_utils import compute_adaptive_n_cycles
        factor = float(config.get("time_frequency_analysis.tfr.n_cycles_factor", 2.0))
        n_cycles = compute_adaptive_n_cycles(freqs, factor)
        decim = config.get("time_frequency_analysis.tfr.decim", 4)
        
        tfr_ep = run_tfr_morlet(
            epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            decim=decim,
            picks="eeg",
            workers=-1,
            config=config,
        )
        times = np.asarray(tfr_ep.times)
        tfr_baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-2.0, 0.0]))
        b_start, b_end, _ = validate_baseline_indices(times, tfr_baseline)
        tfr_ep.apply_baseline(baseline=(b_start, b_end), mode="logratio")
        return tfr_ep.average()

    tfr_list: List["mne.time_frequency.AverageTFR"] = []
    missing_subjects: List[str] = []
    for s in subjects:
        tfr_path = find_tfr_path(s, task, deriv_root)
        if tfr_path is None or not tfr_path.exists():
            missing_subjects.append(s)
            continue
        tfr_baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-2.0, 0.0]))
        t_std = read_tfr_average_with_logratio(tfr_path, tfr_baseline, logger)
        if t_std is None:
            missing_subjects.append(s)
        else:
            tfr_list.append(t_std)

    if len(tfr_list) < 2 and missing_subjects:
        computed = 0
        for s in missing_subjects:
            tavg = _compute_avg_tfr_for_subject(s)
            if tavg is not None:
                tfr_list.append(tavg)
                computed += 1
        if computed > 0:
            logger.info(f"Computed AverageTFR on the fly for {computed} subjects (no saved TFR found)")

    if len(tfr_list) >= 2:
        ref = tfr_list[0]
        tmin = max(float(min(t.times[0] for t in tfr_list)), float(ref.times[0]))
        tmax = min(float(max(t.times[-1] for t in tfr_list)), float(ref.times[-1]))
        ref_mask = time_mask(ref.times, tmin, tmax)
        tref = ref.times[ref_mask]

        band_tc: dict[str, List[np.ndarray]] = {b: [] for b in bands}
        band_tc_pct: dict[str, List[np.ndarray]] = {b: [] for b in bands}
        freq_bands = config.get("time_frequency_analysis.bands", {
            "delta": [1.0, 3.9],
            "theta": [4.0, 7.9],
            "alpha": [8.0, 12.9],
            "beta": [13.0, 30.0],
            "gamma": [30.1, 80.0],
        })
        FEATURES_FREQ_BANDS = {name: tuple(freqs) for name, freqs in freq_bands.items()}
        for t in tfr_list:
            for b in bands:
                if b not in FEATURES_FREQ_BANDS:
                    continue
                fmin, fmax = FEATURES_FREQ_BANDS[b]
                fmask = (t.freqs >= fmin) & (t.freqs <= fmax)
                if fmask.sum() == 0:
                    continue
                series_logr = np.nanmean(t.data[:, fmask, :], axis=(0, 1))
                ratio_data = np.power(10.0, t.data[:, fmask, :])
                series_ratio = np.nanmean(ratio_data, axis=(0, 1))
                s_mask = time_mask(t.times, tmin, tmax)
                if s_mask.sum() < 2:
                    continue
                ts = t.times[s_mask]
                ys_logr = series_logr[s_mask]
                ys_ratio = series_ratio[s_mask]
                if not np.any(np.isfinite(ys_logr)) and not np.any(np.isfinite(ys_ratio)):
                    continue
                fin_logr = np.isfinite(ys_logr)
                if fin_logr.sum() >= 2:
                    ys_logr = np.interp(ts, ts[fin_logr], ys_logr[fin_logr])
                fin_ratio = np.isfinite(ys_ratio)
                if fin_ratio.sum() >= 2:
                    ys_ratio = np.interp(ts, ts[fin_ratio], ys_ratio[fin_ratio])
                yref_logr = np.interp(tref, ts, ys_logr)
                yref_ratio = np.interp(tref, ts, ys_ratio)
                band_tc[b].append(yref_logr)
                yref_pct = 100.0 * (yref_ratio - 1.0)
                band_tc_pct[b].append(yref_pct)

        have_any = any(len(v) >= 2 for v in band_tc.values())
        if have_any:
            valid_bands = [b for b in bands if len(band_tc.get(b, [])) >= 2]
            if len(valid_bands) == 0:
                logger.info("No bands with >=2 subjects for time-course plotting; skipping.")
            else:
                plot_group_band_power_time_courses(valid_bands, band_tc, band_tc_pct, tref, gplots, config, logger)
    else:
        logger.info("Skipping group band power time courses: need at least 2 subjects with TFR (saved or computed).")


###################################################################
# Behavior Correlation Aggregation
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
    if config is None:
        config = load_settings()
    
    if task is None:
        task = config.task
    if deriv_root is None:
        deriv_root = Path(config.deriv_root)
    
    if subjects is None or subjects == ["all"]:
        subjects = get_available_subjects(task=task, config=config)
    gstats = deriv_group_stats_path(deriv_root)
    gplots = deriv_group_plots_path(deriv_root, subdir=PLOT_SUBDIR)
    ensure_dir(gstats)
    ensure_dir(gplots)

    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_group_logger("behavior_analysis", log_name, config=config)
    logger.info(f"Starting group aggregation for {len(subjects)} subjects (task={task})")

    by_key = {}
    for sub in subjects:
        f = deriv_stats_path(deriv_root, sub) / "corr_stats_pow_roi_vs_rating.tsv"
        if not f.exists():
            continue
        df = pd.read_csv(f, sep="\t")
        for _, row in df.iterrows():
            key = (str(row.get("roi")), str(row.get("band")))
            r = row.get("r")
            try:
                r = _safe_float(r)
            except (ValueError, TypeError):
                r = np.nan
            by_key.setdefault(key, []).append(r)
    if by_key:
        recs = []
        for (roi, band), rs in by_key.items():
            r_grp, ci_l, ci_h, n = _fisher_aggregate(rs)
            recs.append({
                "roi": roi,
                "band": band,
                "r_group": r_grp,
                "r_ci_low": ci_l,
                "r_ci_high": ci_h,
                "n_subjects": n,
            })
        dfg = pd.DataFrame(recs)
        pvals = []
        for (roi, band), rs in by_key.items():
            vals = np.array([r for r in rs if np.isfinite(r)])
            vals = np.clip(vals, -0.999999, 0.999999)
            n = vals.size
            if n < 2:
                pvals.append(np.nan)
                continue
            z = np.arctanh(vals)
            from scipy import stats
            res = stats.ttest_1samp(z, popmean=0.0)
            pvals.append(_get_ttest_pvalue(res))
        dfg["p_group"] = pvals
        
        out_rows = []
        for band in sorted(dfg["band"].unique()):
            dfb = dfg[dfg["band"] == band].copy()
            fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
            rej, crit = fdr_bh_reject(dfb["p_group"].to_numpy(), alpha=fdr_alpha)
            dfb["fdr_reject"] = rej
            dfb["fdr_crit_p"] = crit
            out_rows.append(dfb)
        dfg2 = pd.concat(out_rows, ignore_index=True)
        out_path_rating = gstats / "group_corr_pow_roi_vs_rating.tsv"
        dfg2.to_csv(out_path_rating, sep="\t", index=False)
        logger.info(f"Wrote group ROI power vs rating summary: {out_path_rating}")

        for band in sorted(dfg2["band"].unique()):
            dfb = dfg2[dfg2["band"] == band]
            fig, ax = plt.subplots(figsize=(6, 3.2))
            order = sorted(dfb["roi"].unique())
            sns.barplot(data=dfb, x="roi", y="r_group", order=order, color="steelblue", ax=ax)
            for i, roi in enumerate(order):
                row = dfb[dfb["roi"] == roi].iloc[0]
                yv = row["r_group"]
                yerr_low = yv - row["r_ci_low"] if np.isfinite(row["r_ci_low"]) else 0
                yerr_high = row["r_ci_high"] - yv if np.isfinite(row["r_ci_high"]) else 0
                ax.errorbar(i, yv, yerr=[[yerr_low], [yerr_high]], fmt="none", ecolor="k", capsize=3)
            ax.set_ylabel("Group r (Fisher back-transformed)")
            ax.set_xlabel("ROI")
            freq_bands = config.get("time_frequency_analysis.bands", {})
            band_rng = freq_bands.get(band)
            if band_rng:
                band_rng = tuple(band_rng)
            band_label = f"{band} ({band_rng[0]:g}\u2013{band_rng[1]:g} Hz)" if band_rng is not None else band
            ax.set_title(f"Group ROI power vs rating: {band_label}")
            ax.axhline(0, color="k", linewidth=0.8)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")
                fig.tight_layout()
            save_formats = config.get("output.save_formats", ["svg"])
            save_fig(fig, gplots / f"group_roi_power_vs_rating_{sanitize_label(band)}", formats=save_formats, bbox_inches="tight", footer=_get_behavior_footer(config))
            plt.close(fig)

    per_pref = {}
    for sub in subjects:
        subj_stats = deriv_stats_path(deriv_root, sub)
        if not subj_stats.exists():
            continue
        for f in subj_stats.glob("corr_stats_conn_roi_summary_*_vs_rating.tsv"):
            df = pd.read_csv(f, sep="\t")
            if df.empty or "measure_band" not in df.columns:
                continue
            per_pref.setdefault(str(df["measure_band"].iloc[0]), []).append(df)
    for pref, dfs in per_pref.items():
        cat = pd.concat(dfs, ignore_index=True)
        out_rows = []
        for (roi_i, roi_j), grp in cat.groupby(["roi_i", "roi_j"], dropna=False):
            rs = grp["r"].to_numpy(dtype=float)
            r_grp, ci_l, ci_h, n = _fisher_aggregate(rs.tolist())
            vals = np.clip(rs[np.isfinite(rs)], -0.999999, 0.999999)
            if vals.size >= 2:
                from scipy import stats
                res = stats.ttest_1samp(np.arctanh(vals), popmean=0.0)
                pval = _get_ttest_pvalue(res)
            else:
                pval = np.nan
            out_rows.append({
                "measure_band": pref,
                "roi_i": roi_i,
                "roi_j": roi_j,
                "summary_type": "within" if roi_i == roi_j else "between",
                "r_group": r_grp,
                "r_ci_low": ci_l,
                "r_ci_high": ci_h,
                "n_subjects": n,
                "p_group": pval,
            })
        out_df = pd.DataFrame(out_rows)
        fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
        rej, crit = fdr_bh_reject(out_df["p_group"].to_numpy(), alpha=fdr_alpha)
        out_df["fdr_reject"] = rej
        out_df["fdr_crit_p"] = crit
        out_conn = gstats / f"group_corr_conn_roi_summary_{sanitize_label(pref)}_vs_rating.tsv"
        out_df.to_csv(out_conn, sep="\t", index=False)
        logger.info(f"Wrote group connectivity ROI summary: {out_conn}")

    logger.info("Generating pooled ROI scatters across subjects…")
    scatter_inputs = collect_group_power_roi_inputs(
        subjects,
        task,
        deriv_root,
        config,
        partial_covars=PARTIAL_COVARS_DEFAULT,
        do_temp=True,
    )
    plot_group_power_roi_scatter(
        scatter_inputs,
        config=config,
        pooling_strategy=pooling_strategy,
        cluster_bootstrap=int(cluster_bootstrap),
        subject_fixed_effects=bool(subject_fixed_effects),
        do_temp=True,
        bootstrap_ci=0,
        rng=None,
    )
    
    aggregate_channel_level_visuals(subjects, task, logger, deriv_root)

    aggregate_overall_band_summary(subjects, task, logger=logger, use_spearman=True, pooling_strategy=pooling_strategy, cluster_bootstrap=int(cluster_bootstrap))


###################################################################
# Channel-Level Visuals
###################################################################

def aggregate_channel_level_visuals(subjects: List[str], task: str, logger: logging.Logger, deriv_root: Path) -> None:
    config = load_settings()
    gstats = deriv_group_stats_path(deriv_root)
    gplots = deriv_group_plots_path(deriv_root, subdir=PLOT_SUBDIR)
    ensure_dir(gstats)
    ensure_dir(gplots)

    def _load_sub_band_df(sub: str, band: str) -> Optional[pd.DataFrame]:
        f = deriv_stats_path(deriv_root, sub) / f"corr_stats_pow_{band}_vs_rating.tsv"
        if not f.exists():
            return None
        df = pd.read_csv(f, sep="\t")
        if df is None or df.empty or "channel" not in df.columns or "r" not in df.columns:
            return None
        return df

    bands_to_df = {}
    bands_to_df_temp = {}
    power_bands_to_use = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    for band in power_bands_to_use:
        chan_to_r = {}
        chan_to_r_temp = {}
        for sub in subjects:
            df = _load_sub_band_df(sub, band)
            if df is None:
                continue
            for _, row in df.iterrows():
                ch = str(row.get("channel"))
                try:
                    r = _safe_float(row.get("r"))
                except (TypeError, ValueError):
                    r = np.nan
                if np.isfinite(r):
                    chan_to_r.setdefault(ch, []).append(_safe_float(np.clip(r, -0.999999, 0.999999)))
        for sub in subjects:
            ftemp = deriv_stats_path(deriv_root, sub) / f"corr_stats_pow_{band}_vs_temp.tsv"
            if not ftemp.exists():
                continue
            df_t = pd.read_csv(ftemp, sep="\t")
            if df_t is None or df_t.empty or "channel" not in df_t.columns or "r" not in df_t.columns:
                continue
            for _, row in df_t.iterrows():
                ch = str(row.get("channel"))
                try:
                    r = _safe_float(row.get("r"))
                except (TypeError, ValueError):
                    r = np.nan
                if np.isfinite(r):
                    chan_to_r_temp.setdefault(ch, []).append(_safe_float(np.clip(r, -0.999999, 0.999999)))

        if not chan_to_r and not chan_to_r_temp:
            continue

        out_rows = []
        for ch, rs in sorted(chan_to_r.items()):
            vals = np.array(rs, dtype=float)
            vals = vals[np.isfinite(vals)]
            vals = np.clip(vals, -0.999999, 0.999999)
            if vals.size == 0:
                continue
            z = np.arctanh(vals)
            r_grp = _safe_float(np.tanh(np.mean(z)))
            if len(z) >= 2:
                from scipy import stats
                res = stats.ttest_1samp(z, popmean=0.0)
                p = _get_ttest_pvalue(res)
            else:
                p = np.nan
            if len(z) >= 2:
                sd = _safe_float(np.std(z, ddof=1))
                se = sd / np.sqrt(len(z)) if sd > 0 else np.nan
                if np.isfinite(se) and se > 0:
                    tcrit = _safe_float(stats.t.ppf(0.975, df=len(z) - 1))
                    lo = _safe_float(np.tanh(np.mean(z) - tcrit * se))
                    hi = _safe_float(np.tanh(np.mean(z) + tcrit * se))
                else:
                    lo = np.nan
                    hi = np.nan
            else:
                lo = np.nan
                hi = np.nan
            out_rows.append({
                "channel": ch,
                "band": band,
                "r_group": r_grp,
                "p_group": p,
                "r_ci_low": lo,
                "r_ci_high": hi,
                "n_subjects": int(len(z)),
            })

        df_band = pd.DataFrame(out_rows)
        if not df_band.empty:
            fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
            rej, crit = fdr_bh_reject(df_band["p_group"].to_numpy(), alpha=fdr_alpha)
            df_band["fdr_reject"] = rej
            df_band["fdr_crit_p"] = crit
            df_band = df_band.sort_values("channel").reset_index(drop=True)
            df_band.to_csv(gstats / f"group_corr_pow_{sanitize_label(band)}_vs_rating.tsv", sep="\t", index=False)
            bands_to_df[band] = df_band

        out_rows_t = []
        for ch, rs in sorted(chan_to_r_temp.items()):
            vals = np.array(rs, dtype=float)
            vals = vals[np.isfinite(vals)]
            vals = np.clip(vals, -0.999999, 0.999999)
            if vals.size == 0:
                continue
            z = np.arctanh(vals)
            r_grp = _safe_float(np.tanh(np.mean(z)))
            if len(z) >= 2:
                from scipy import stats
                res = stats.ttest_1samp(z, popmean=0.0)
                p = _get_ttest_pvalue(res)
            else:
                p = np.nan
            if len(z) >= 2:
                sd = _safe_float(np.std(z, ddof=1))
                se = sd / np.sqrt(len(z)) if sd > 0 else np.nan
                if np.isfinite(se) and se > 0:
                    tcrit = _safe_float(stats.t.ppf(0.975, df=len(z) - 1))
                    lo = _safe_float(np.tanh(np.mean(z) - tcrit * se))
                    hi = _safe_float(np.tanh(np.mean(z) + tcrit * se))
                else:
                    lo = np.nan
                    hi = np.nan
            else:
                lo = np.nan
                hi = np.nan
            out_rows_t.append({
                "channel": ch,
                "band": band,
                "r_group": r_grp,
                "p_group": p,
                "r_ci_low": lo,
                "r_ci_high": hi,
                "n_subjects": int(len(z)),
            })

        df_band_t = pd.DataFrame(out_rows_t)
        if not df_band_t.empty:
            fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
            rej_t, crit_t = fdr_bh_reject(df_band_t["p_group"].to_numpy(), alpha=fdr_alpha)
            df_band_t["fdr_reject"] = rej_t
            df_band_t["fdr_crit_p"] = crit_t
            df_band_t = df_band_t.sort_values("channel").reset_index(drop=True)
            df_band_t.to_csv(gstats / f"group_corr_pow_{sanitize_label(band)}_vs_temp.tsv", sep="\t", index=False)
            bands_to_df_temp[band] = df_band_t

    if not bands_to_df and not bands_to_df_temp:
        logger.warning("No group channel-level stats aggregated (missing subject inputs?)")
        return

    if bands_to_df:
        combined = pd.concat([bands_to_df[b] for b in bands_to_df.keys()], ignore_index=True)
        combined.to_csv(gstats / "group_corr_pow_combined_vs_rating.tsv", sep="\t", index=False)
    else:
        combined = pd.DataFrame()

    if bands_to_df_temp:
        combined_t = pd.concat([bands_to_df_temp[b] for b in bands_to_df_temp.keys()], ignore_index=True)
        combined_t.to_csv(gstats / "group_corr_pow_combined_vs_temp.tsv", sep="\t", index=False)
    else:
        combined_t = pd.DataFrame()

    for band, dfb in bands_to_df.items():
        fig, ax = plt.subplots(figsize=(12, 7))
        xs = np.arange(len(dfb))
        fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
        colors = ["red" if (np.isfinite(p) and p < fdr_alpha) else "lightblue" for p in dfb["p_group"]]
        ax.bar(xs, dfb["r_group"], color=colors)
        ax.set_xlabel("Channel", fontweight="bold")
        ax.set_ylabel("Spearman ρ", fontweight="bold")
        ax.set_title(f"{band.upper()} Band - Channel-wise Correlations with Behavior\nGroup", fontweight="bold", fontsize=14)
        ax.set_xticks(xs)
        ax.set_xticklabels(dfb["channel"].tolist(), rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        ax.axhline(y=0.3, color="green", linestyle="--", alpha=0.7)
        ax.axhline(y=-0.3, color="green", linestyle="--", alpha=0.7)
        fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
        sig_count = int((dfb["p_group"] < fdr_alpha).sum())
        ax.text(0.02, 0.98, f"Significant channels: {sig_count}/{len(dfb)}",
                transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        plt.tight_layout()
        save_formats = config.get("output.save_formats", ["svg"])
        save_fig(fig, gplots / f"group_power_behavior_correlation_{sanitize_label(band)}", formats=save_formats, bbox_inches="tight", footer=_get_behavior_footer(config))
        plt.close(fig)

    for band, dfb in bands_to_df_temp.items():
        fig, ax = plt.subplots(figsize=(12, 7))
        xs = np.arange(len(dfb))
        fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
        colors = ["red" if (np.isfinite(p) and p < fdr_alpha) else "lightblue" for p in dfb["p_group"]]
        ax.bar(xs, dfb["r_group"], color=colors)
        ax.set_xlabel("Channel", fontweight="bold")
        ax.set_ylabel("Spearman ρ", fontweight="bold")
        ax.set_title(f"{band.upper()} Band - Channel-wise Correlations with Temperature\nGroup", fontweight="bold", fontsize=14)
        ax.set_xticks(xs)
        ax.set_xticklabels(dfb["channel"].tolist(), rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        ax.axhline(y=0.3, color="green", linestyle="--", alpha=0.7)
        ax.axhline(y=-0.3, color="green", linestyle="--", alpha=0.7)
        fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
        sig_count = int((dfb["p_group"] < fdr_alpha).sum())
        ax.text(0.02, 0.98, f"Significant channels: {sig_count}/{len(dfb)}",
                transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        plt.tight_layout()
        save_formats = config.get("output.save_formats", ["svg"])
        save_fig(fig, gplots / f"group_power_temperature_correlation_{sanitize_label(band)}", formats=save_formats, bbox_inches="tight", footer=_get_behavior_footer(config))
        plt.close(fig)

    top_n = int(config.get("behavior_analysis.predictors.top_n", 20))
    fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
    df_sig = combined[(~combined.empty) & (combined["p_group"] <= fdr_alpha) & combined["r_group"].notna()].copy() if not combined.empty else pd.DataFrame()
    if len(df_sig) > 0:
        df_sig["abs_r"] = df_sig["r_group"].abs()
        df_top = df_sig.nlargest(top_n, "abs_r").copy()
        df_top = df_top.sort_values("abs_r", ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
        labels = [f"{ch} ({band})" for ch, band in zip(df_top["channel"], df_top["band"])]
        y_pos = np.arange(len(df_top))
        colors = [_get_band_color(b) for b in df_top["band"]]
        ax.barh(y_pos, df_top["abs_r"], color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=11)
        fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
        ax.set_xlabel(f"|Spearman ρ| (p < {fdr_alpha})", fontweight='bold', fontsize=12)
        ax.set_title(f"Top {top_n} Significant Behavioral Predictors — Group", fontweight='bold', fontsize=14, pad=20)
        for i, (_, row) in enumerate(df_top.iterrows()):
            ax.text(row["abs_r"] + 0.01, i, f"{row['abs_r']:.3f} (p={row['p_group']:.3f})", va='center', ha='left', fontsize=10)
        ax.set_xlim(0, _safe_float(df_top["abs_r"].max()) * 1.25)
        ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()
        save_formats = config.get("output.save_formats", ["svg"])
        save_fig(fig, gplots / f"group_top_{top_n}_behavioral_predictors", formats=save_formats, bbox_inches="tight", footer=_get_behavior_footer(config))
        plt.close(fig)
        df_top_export = df_top[["channel", "band", "r_group", "p_group", "n_subjects", "abs_r"]].sort_values("abs_r", ascending=False)
        df_top_export.to_csv(gstats / f"group_top_{top_n}_behavioral_predictors.tsv", sep="\t", index=False)
    else:
        logger.info("No significant group-level predictors to plot for top-N.")

    top_n = int(config.get("behavior_analysis.predictors.top_n", 20))
    fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
    df_sig_t = combined_t[(~combined_t.empty) & (combined_t["p_group"] <= fdr_alpha) & combined_t["r_group"].notna()].copy() if not combined_t.empty else pd.DataFrame()
    if len(df_sig_t) > 0:
        df_sig_t["abs_r"] = df_sig_t["r_group"].abs()
        df_top_t = df_sig_t.nlargest(top_n, "abs_r").copy()
        df_top_t = df_top_t.sort_values("abs_r", ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
        labels = [f"{ch} ({band})" for ch, band in zip(df_top_t["channel"], df_top_t["band"])]
        y_pos = np.arange(len(df_top_t))
        colors = [_get_band_color(b) for b in df_top_t["band"]]
        ax.barh(y_pos, df_top_t["abs_r"], color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=11)
        fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
        ax.set_xlabel(f"|Spearman ρ| (p < {fdr_alpha})", fontweight='bold', fontsize=12)
        ax.set_title(f"Top {top_n} Significant Temperature Predictors — Group", fontweight='bold', fontsize=14, pad=20)
        for i, (_, row) in enumerate(df_top_t.iterrows()):
            ax.text(row["abs_r"] + 0.01, i, f"{row['abs_r']:.3f} (p={row['p_group']:.3f})", va='center', ha='left', fontsize=10)
        ax.set_xlim(0, _safe_float(df_top_t["abs_r"].max()) * 1.25)
        ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()
        save_formats = config.get("output.save_formats", ["svg"])
        save_fig(fig, gplots / f"group_top_{top_n}_temperature_predictors", formats=save_formats, bbox_inches="tight", footer=_get_behavior_footer(config))
        plt.close(fig)
        df_top_t[["channel", "band", "r_group", "p_group", "n_subjects", "abs_r"]].sort_values("abs_r", ascending=False).to_csv(
            gstats / f"group_top_{top_n}_temperature_predictors.tsv", sep="\t", index=False
        )
    else:
        logger.info("No significant group-level temperature predictors to plot for top-N.")

    def plot_group_topomap(bands_data, title_suffix, filename_suffix):
        info = None
        for sub in subjects:
            epo_path = _find_clean_epochs_path(sub, task)
            if epo_path and Path(epo_path).exists():
                info = mne.read_epochs(epo_path, preload=False, verbose=False).info
                break
        if not info:
            return

        n_bands = len(bands_data)
        fig, axes = plt.subplots(1, n_bands, figsize=(4.8 * n_bands, 4.8))
        if n_bands == 1:
            axes = [axes]
        plt.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.16, wspace=0.08)

        all_corrs = []
        for bd in bands_data:
            all_corrs.extend(bd["correlations"][bd["significant_mask"]])
        vmax = np.max(np.abs([c for c in all_corrs if np.isfinite(c)])) if all_corrs else 0.5

        successful = []
        for i, bd in enumerate(bands_data):
            ax = axes[i]
            picks = mne.pick_types(info, meg=False, eeg=True, exclude='bads')
            if not picks:
                continue
                
            n_chs = len(info['ch_names'])
            topo_data = np.zeros(n_chs)
            topo_mask = np.zeros(n_chs, dtype=bool)
            
            for j, ch in enumerate(info['ch_names']):
                if ch in bd['channels']:
                    idx = bd['channels'].index(ch)
                    val = bd['correlations'][idx]
                    topo_data[j] = val if np.isfinite(val) else 0.0
                    topo_mask[j] = bool(bd['significant_mask'][idx])
            
            im, _ = mne.viz.plot_topomap(
                topo_data[picks], mne.pick_info(info, picks), axes=ax, show=False,
                cmap='RdBu_r', vlim=(-vmax, vmax), contours=6,
                mask=topo_mask[picks],
                mask_params=dict(marker='o', markerfacecolor='white', markeredgecolor='black', linewidth=1, markersize=6)
            )
            successful.append(im)
            n_sig = int(topo_mask[picks].sum())
            n_total = sum(1 for ch in bd['channels'] if ch in info['ch_names'])
            ax.set_title(f"{bd['band'].upper()}\n{n_sig}/{n_total} significant", fontweight='bold', fontsize=12, pad=10)

        fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
        plt.suptitle(f"Group Significant EEG-{title_suffix} Correlations (p < {fdr_alpha})", fontweight='bold', fontsize=14, y=0.97)
        
        if successful:
            ax_positions = [ax.get_position() for ax in axes]
            left = min(pos.x0 for pos in ax_positions)
            right = max(pos.x1 for pos in ax_positions)
            bottom = min(pos.y0 for pos in ax_positions)
            span = right - left
            cax = fig.add_axes([left + 0.225 * span, max(0.04, bottom - 0.06), 0.55 * span, 0.028])
            cbar = fig.colorbar(successful[-1], cax=cax, orientation='horizontal')
            cbar.set_label('Correlation (ρ)', fontweight='bold', fontsize=11)
            cbar.ax.tick_params(pad=2, labelsize=9)
        
        save_formats = config.get("output.save_formats", ["svg"])
        save_fig(fig, gplots / f"group_significant_correlations_topomap_{filename_suffix}", formats=save_formats, bbox_inches="tight", footer=_get_behavior_footer(config))
        plt.close(fig)

    bands_data = []
    for band, dfb in bands_to_df.items():
        chs = dfb["channel"].astype(str).tolist()
        corrs = dfb["r_group"].to_numpy()
        pvals = dfb["p_group"].to_numpy()
        fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
        sig_mask = np.isfinite(pvals) & (pvals < fdr_alpha)
        bands_data.append({
            "band": band, "channels": chs, "correlations": corrs,
            "p_values": pvals, "significant_mask": sig_mask,
        })
    if bands_data:
        plot_group_topomap(bands_data, "Pain", "")

    if bands_to_df_temp:
        bands_data = []
        for band, dfb in bands_to_df_temp.items():
            chs = dfb["channel"].astype(str).tolist()
            corrs = dfb["r_group"].to_numpy()
            pvals = dfb["p_group"].to_numpy()
            fdr_alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
            sig_mask = np.isfinite(pvals) & (pvals < fdr_alpha)
            bands_data.append({
                "band": band, "channels": chs, "correlations": corrs,
                "p_values": pvals, "significant_mask": sig_mask,
            })
        plot_group_topomap(bands_data, "Temperature", "temperature")


###################################################################
# Overall Band Summary
###################################################################

def aggregate_overall_band_summary(subjects, task, logger, use_spearman, pooling_strategy, cluster_bootstrap):
    config = load_settings()
    deriv_root = Path(config.deriv_root)
    
    from eeg_pipeline.utils.data_loading import _load_features_and_targets, _pick_first_column, load_epochs_for_analysis
    
    rating_x = {}
    rating_y = {}
    temp_x = {}
    temp_y = {}
    have_temp = False

    for sub in subjects:
        try:
            _temporal_df, pow_df, _conn_df, y, info = _load_features_and_targets(sub, task, deriv_root, config)
            y = pd.to_numeric(y, errors="coerce")
            epo_path = _find_clean_epochs_path(sub, task)
            if not epo_path:
                continue
            epochs, aligned = load_epochs_for_analysis(sub, task, align="strict", preload=False, deriv_root=deriv_root, bids_root=config.bids_root, config=config)
            if epochs is None:
                continue
        except Exception:
            continue
            
        ts = None
        if aligned is not None:
            psych_temp_columns = config.get("event_columns.temperature", [])
            tcol = _pick_first_column(aligned, psych_temp_columns)
            if tcol:
                ts = pd.to_numeric(aligned[tcol], errors="coerce")
                have_temp = True
                
        power_bands_to_use = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
        for band in power_bands_to_use:
            cols = [c for c in pow_df.columns if c.startswith(f"pow_{band}_")]
            if not cols:
                continue
            vals = pow_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).to_numpy()
            rating_x.setdefault(band, []).append(vals)
            rating_y.setdefault(band, []).append(y.to_numpy())
            if ts is not None:
                temp_x.setdefault(band, []).append(vals)
                temp_y.setdefault(band, []).append(ts.to_numpy())

    def pool_xy(x_lists, y_lists, strategy):
        vis_x, vis_y = [], []
        for xi, yi in zip(x_lists, y_lists):
            xi, yi = pd.Series(xi), pd.Series(yi)
            n = min(len(xi), len(yi))
            xi, yi = xi.iloc[:n], yi.iloc[:n]
            m = xi.notna() & yi.notna()
            xi, yi = xi[m], yi[m]
            
            if strategy == "within_subject_centered":
                xi, yi = xi - xi.mean(), yi - yi.mean()
            elif strategy == "within_subject_zscored":
                sx, sy = xi.std(ddof=1), yi.std(ddof=1)
                if sx <= 0 or sy <= 0:
                    continue
                xi, yi = (xi - xi.mean()) / sx, (yi - yi.mean()) / sy
            elif strategy == "fisher_by_subject":
                xi, yi = xi - xi.mean(), yi - yi.mean()
            vis_x.append(xi)
            vis_y.append(yi)
        return pd.concat(vis_x, ignore_index=True) if vis_x else pd.Series(dtype=float), pd.concat(vis_y, ignore_index=True) if vis_y else pd.Series(dtype=float)

    power_bands_to_use = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
    for band in power_bands_to_use:
        x_lists = rating_x.get(band, [])
        y_lists = rating_y.get(band, [])
        if not x_lists or not y_lists:
            continue
        X, Y = pool_xy(x_lists, y_lists, pooling_strategy)
        if len(X) < 5:
            continue
            
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        sns.regplot(x=X, y=Y, ax=ax, ci=95,
                   scatter_kws={"s": 25, "alpha": 0.7, "color": _get_band_color(band), "edgecolor": "white", "linewidths": 0.3},
                   line_kws={"color": "#666666", "lw": 1.5})
        ax.set_xlabel(f"{band.capitalize()} Power\nlog10(power/baseline)")
        ax.set_ylabel("Rating")
        ax.set_title(f"{band.capitalize()} vs Rating")
        
        from scipy import stats
        r, p = stats.spearmanr(X, Y, nan_policy="omit")
        ax.text(0.02, 0.98, f"Spearman ρ={r:.3f}\np={p:.3f}\nn={len(X)}",
               transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_formats = config.get("output.save_formats", ["svg"])
        save_fig(fig, deriv_group_plots_path(deriv_root, subdir=PLOT_SUBDIR) / f"group_power_behavior_correlation_{sanitize_label(band)}", formats=save_formats, bbox_inches="tight", footer=_get_behavior_footer(config))
        plt.close(fig)

    if have_temp and temp_x:
        power_bands_to_use = config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
        for band in power_bands_to_use:
            x_lists = temp_x.get(band, [])
            y_lists = temp_y.get(band, [])
            if not x_lists or not y_lists:
                continue
            X, Y = pool_xy(x_lists, y_lists, pooling_strategy)
            if len(X) < 5:
                continue
                
            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            sns.regplot(x=X, y=Y, ax=ax, ci=95,
                       scatter_kws={"s": 25, "alpha": 0.7, "color": _get_band_color(band), "edgecolor": "white", "linewidths": 0.3},
                       line_kws={"color": "#666666", "lw": 1.5})
            ax.set_xlabel(f"{band.capitalize()} Power\nlog10(power/baseline)")
            
            if pooling_strategy == "within_subject_centered":
                ax.set_ylabel("Temperature (°C, centered)")
            elif pooling_strategy == "within_subject_zscored":
                ax.set_ylabel("Temperature (z-scored)")
            else:
                ax.set_ylabel("Temperature (°C)")
            ax.set_title(f"{band.capitalize()} vs Temp")
            
            from scipy import stats
            r, p = stats.spearmanr(X, Y, nan_policy="omit")
            ax.text(0.02, 0.98, f"Spearman ρ={r:.3f}\np={p:.3f}\nn={len(X)}",
                   transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save_formats = config.get("output.save_formats", ["svg"])
            save_fig(fig, deriv_group_plots_path(deriv_root, subdir=PLOT_SUBDIR) / f"group_power_temperature_correlation_{sanitize_label(band)}", formats=save_formats, bbox_inches="tight", footer=_get_behavior_footer(config))
            plt.close(fig)

