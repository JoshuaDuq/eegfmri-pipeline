import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
from scipy.stats import ttest_rel, t as t_dist

from eeg_pipeline.utils.io_utils import (
    ensure_aligned_lengths,
    deriv_group_plots_path,
    deriv_group_stats_path,
)
from eeg_pipeline.utils.tfr_utils import (
    apply_baseline_and_crop,
    validate_baseline_indices,
    avg_alltrials_to_avg_tfr,
    avg_by_mask_to_avg_tfr,
    align_avg_tfrs,
    align_paired_avg_tfrs,
    get_rois,
    canonicalize_ch_name as _canonicalize_ch_name,
    find_roi_channels as _find_roi_channels,
)
from eeg_pipeline.utils.io_utils import (
    get_viz_params as __get_viz_params,
    unwrap_figure as __unwrap_figure,
    plot_topomap_on_ax as __plot_topomap_on_ax,
    format_cluster_ann as __format_cluster_ann,
    robust_sym_vlim as __robust_sym_vlim,
)
from eeg_pipeline.utils.stats_utils import (
    fdr_bh_mask as _fdr_bh_mask,
    fdr_bh_values as _fdr_bh_values,
    cluster_test_epochs as _cluster_test_epochs,
    cluster_test_two_sample_arrays as _cluster_test_two_sample_arrays,
)


###################################################################
# Utilities
###################################################################

def _get_viz_params(config=None):
    return __get_viz_params(config)


def _log(msg, logger=None, level: str = "info"):
    if logger is None:
        logger = logging.getLogger(__name__)
    getattr(logger, level)(msg)


def _fig(obj):
    return __unwrap_figure(obj)


def _clip_time_window(times: np.ndarray, window: Tuple[float, float]) -> Tuple[float, float]:
    tmin_req, tmax_req = window
    tmin_eff = float(max(float(np.min(times)), float(tmin_req)))
    tmax_eff = float(min(float(np.max(times)), float(tmax_req)))
    return tmin_eff, tmax_eff


def _get_bands_for_tfr(tfr=None, max_freq_available=None, config=None, logger=None):
    max_freq = max_freq_available or (float(np.max(tfr.freqs)) if tfr is not None else 80.0)
    band_bounds = config.get("tfr_topography_pipeline.bands", {
        "theta": [4.0, 7.9],
        "alpha": [8.0, 12.9],
        "beta": [13.0, 30.0],
        "gamma": [30.1, 80.0],
    }) if config else {}

    bands = {}
    for band_name in ["delta", "theta", "alpha", "beta"]:
        if band_name in band_bounds:
            fmin, fmax = band_bounds[band_name]
            if fmax is None:
                fmax_eff = max_freq
            else:
                fmax_eff = min(fmax, max_freq)
            if fmax is not None and fmax_eff < fmax and logger:
                logger.info(f"Band {band_name} reduced from {fmax} to {fmax_eff} Hz due to data limits")
            bands[band_name] = (float(fmin), float(fmax_eff))

    gamma_lower, gamma_upper = band_bounds.get("gamma", (30.1, 80.0))
    fmax_gamma = min(gamma_upper if gamma_upper is not None else 80.0, max_freq)
    if gamma_upper is not None and fmax_gamma < gamma_upper and logger:
        logger.info(f"Band gamma reduced from {gamma_upper} to {fmax_gamma} Hz due to data limits")
    bands["gamma"] = (float(gamma_lower), float(fmax_gamma))
    return bands


def _average_tfr_band(tfr_avg, fmin, fmax, tmin, tmax):
    freqs = np.asarray(tfr_avg.freqs)
    times = np.asarray(tfr_avg.times)
    f_mask = (freqs >= float(fmin)) & (freqs <= float(fmax))
    t_mask = (times >= float(tmin)) & (times < float(tmax))
    if f_mask.sum() == 0 or t_mask.sum() == 0:
        return None
    sel = np.asarray(tfr_avg.data)[:, f_mask, :][:, :, t_mask]
    return sel.mean(axis=(1, 2))


def _plot_topomap_on_ax(ax, data, info, mask=None, mask_params=None, vmin=None, vmax=None, config=None):
    return __plot_topomap_on_ax(ax, data, info, mask=mask, mask_params=mask_params, vmin=vmin, vmax=vmax, config=config)


def _format_cluster_ann(p, k=None, mass=None, config=None):
    return __format_cluster_ann(p, k=k, mass=mass, config=config)


def _robust_sym_vlim(arrs, q_low: float = 0.02, q_high: float = 0.98, cap: float = 0.25, min_v: float = 1e-6) -> float:
    return __robust_sym_vlim(arrs, q_low=q_low, q_high=q_high, cap=cap, min_v=min_v)


def _sanitize(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _pick_central_channel(info, preferred="Cz", logger=None, group_mode=False):
    ch_names = info["ch_names"]
    if preferred in ch_names:
        return preferred
    for nm in ch_names:
        if nm.lower() == preferred.lower():
            return nm
    picks = mne.pick_types(info, eeg=True, exclude=[])
    if len(picks) == 0:
        raise RuntimeError("No EEG channels available for plotting.")
    fallback = ch_names[picks[0]]
    if group_mode:
        raise ValueError(
            f"Channel '{preferred}' not found in group-level analysis. Ensure all subjects have '{preferred}' channel."
        )
    _log(f"Channel '{preferred}' not found; using '{fallback}' instead.", logger, "warning")
    return fallback


def _format_baseline_in_filename(baseline_used: Tuple[float, float]) -> str:
    b_start, b_end = baseline_used
    return f"bl{abs(b_start):.1f}to{abs(b_end):.2f}"


def _save_fig(
    fig_obj,
    out_dir: Path,
    name: str,
    config,
    formats=None,
    logger: Optional[logging.Logger] = None,
    baseline_used: Optional[Tuple[float, float]] = None,
    subject: Optional[str] = None,
    task: Optional[str] = None,
    band: Optional[str] = None,
):
    from eeg_pipeline.utils.io_utils import save_fig as _central_save_fig, build_footer as _build_footer
    out_dir.mkdir(parents=True, exist_ok=True)

    if baseline_used is None:
        baseline_used = tuple(config.get("time_frequency_analysis.baseline_window", [-5.0, -0.01]))

    figs = fig_obj if isinstance(fig_obj, list) else [fig_obj]
    stem, ext = (name.rsplit(".", 1) + [""])[:2]
    # Enforce config save formats only; fallback to ["png"] if missing
    exts = []
    if hasattr(config, "get"):
        try:
            exts = list(config.get("output.save_formats", []))
        except Exception:
            exts = []
    if not exts:
        exts = ["png"]

    header_parts = []
    if subject:
        header_parts.append(f"sub-{subject}")
    if task:
        header_parts.append(f"task-{task}")
    if band:
        header_parts.append(f"band-{band}")

    baseline_str = _format_baseline_in_filename(baseline_used)
    if baseline_str not in stem:
        stem = f"{stem}_{baseline_str}"
    if header_parts:
        stem = f"{'_'.join(header_parts)}_{stem}"

    template_name = (
        config.get("output.tfr_footer_template", "tfr_baseline")
        if hasattr(config, "get") else "tfr_baseline"
    )
    footer_kwargs = {
        "baseline_window": baseline_used,
        "baseline": f"[{float(baseline_used[0]):.2f}, {float(baseline_used[1]):.2f}] s",
    }
    footer_text = _build_footer(
        template_name,
        config,
        **footer_kwargs,
    ) if hasattr(config, 'get') else None

    constants = {
        "FIG_DPI": int(config.get("output.fig_dpi", 300)),
        "SAVE_FORMATS": list(exts),
        "output.bbox_inches": config.get("output.bbox_inches", "tight"),
        "output.pad_inches": float(config.get("output.pad_inches", 0.02)),
    }

    for i, f in enumerate(figs):
        out_name = f"{stem}.{exts[0]}" if i == 0 else f"{stem}_{i+1}.{exts[0]}"
        out_path = out_dir / out_name
        _central_save_fig(
            f,
            out_path,
            logger=logger,
            footer=footer_text,
            formats=tuple(exts),
            dpi=constants["FIG_DPI"],
            bbox_inches=constants["output.bbox_inches"],
            pad_inches=constants["output.pad_inches"],
            constants=constants,
        )


###################################################################
# Single-Subject Plots
###################################################################

def plot_cz_all_trials_raw(tfr, out_dir: Path, config, logger: Optional[logging.Logger] = None) -> None:
    tfr_avg = tfr.copy().average() if isinstance(tfr, mne.time_frequency.EpochsTFR) else tfr.copy()
    central_ch = _pick_central_channel(tfr_avg.info, preferred="Cz", logger=logger)
    fig = _fig(tfr_avg.plot(picks=central_ch, show=False))
    fig.suptitle(f"{central_ch} TFR — all trials (raw, no baseline)", fontsize=12)
    _save_fig(fig, out_dir, f"tfr_{central_ch}_all_trials_raw.png", config=config, logger=logger)


def plot_cz_all_trials(
    tfr,
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    plateau_window: Tuple[float, float],
    logger: Optional[logging.Logger] = None,
    subject: Optional[str] = None,
    task: Optional[str] = None,
) -> None:
    tfr_copy = tfr.copy()
    baseline_used = apply_baseline_and_crop(tfr_copy, baseline=baseline, mode="logratio", logger=logger)
    tfr_avg = tfr_copy.average() if isinstance(tfr_copy, mne.time_frequency.EpochsTFR) else tfr_copy

    central_ch = _pick_central_channel(tfr_avg.info, preferred="Cz", logger=logger)
    ch_idx = tfr_avg.info["ch_names"].index(central_ch)
    arr = np.asarray(tfr_avg.data[ch_idx])
    vabs = _robust_sym_vlim(arr)
    times = np.asarray(tfr_avg.times)
    tmin_req, tmax_req = plateau_window
    tmask = (times >= float(tmin_req)) & (times < float(tmax_req))
    if not np.any(tmask):
        msg = f"Plateau window [{tmin_req}, {tmax_req}] outside data range [{times.min():.2f}, {times.max():.2f}]"
        strict_mode = config.get("analysis.strict_mode", True) if config else True
        if strict_mode:
            raise ValueError(msg)
        _log(f"{msg}; using entire time span", logger, "warning")
        tmask = np.ones_like(times, dtype=bool)

    mu = float(np.nanmean(arr[:, tmask]))
    pct = (10.0 ** (mu) - 1.0) * 100.0
    fig = tfr_avg.plot(picks=central_ch, vlim=(-vabs, +vabs), show=False)
    fig = _fig(fig)
    fig.suptitle(
        f"{central_ch} TFR — all trials (baseline logratio)\nvlim ±{vabs:.2f}; mean %Δ vs BL={pct:+.0f}%",
        fontsize=12,
    )
    _save_fig(
        fig,
        out_dir,
        f"tfr_{central_ch}_all_trials.png",
        config=config,
        logger=logger,
        baseline_used=baseline_used,
        subject=subject,
        task=task,
    )


def plot_channels_all_trials(
    tfr,
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    logger: Optional[logging.Logger] = None,
    subject: Optional[str] = None,
    task: Optional[str] = None,
) -> None:
    tfr_copy = tfr.copy()
    baseline_used = apply_baseline_and_crop(tfr_copy, baseline=baseline, mode="logratio", logger=logger)
    tfr_avg = tfr_copy.average() if isinstance(tfr_copy, mne.time_frequency.EpochsTFR) else tfr_copy

    ch_names = tfr_avg.info["ch_names"]
    ch_dir = out_dir / "channels"
    ch_dir.mkdir(parents=True, exist_ok=True)

    fmax_available = float(np.max(tfr_avg.freqs))
    bands = _get_bands_for_tfr(max_freq_available=fmax_available, logger=logger, config=config)

    for ch in ch_names:
        fig = _fig(tfr_avg.plot(picks=ch, show=False))
        fig.suptitle(f"{ch} — all trials (baseline logratio)", fontsize=12)
        _save_fig(
            fig,
            ch_dir,
            f"tfr_{ch}_all_trials.png",
            config=config,
            logger=logger,
            baseline_used=baseline_used,
            subject=subject,
            task=task,
        )

        for band, (fmin, fmax) in bands.items():
            fmax_eff = min(fmax, fmax_available)
            if fmin >= fmax_eff:
                continue
            band_dir = ch_dir / band
            band_dir.mkdir(parents=True, exist_ok=True)

            fig_b = _fig(tfr_avg.plot(picks=ch, fmin=fmin, fmax=fmax_eff, show=False))
            fig_b.suptitle(f"{ch} — {band} band (baseline logratio)", fontsize=12)
            _save_fig(
                fig_b,
                band_dir,
                f"tfr_{ch}_{band}_all_trials.png",
                config=config,
                logger=logger,
                baseline_used=baseline_used,
                subject=subject,
                task=task,
                band=band,
            )


###################################################################
# Pain vs Non-pain (Subject)
###################################################################

def contrast_channels_pain_nonpain(
    tfr,
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    logger: Optional[logging.Logger] = None,
    subject: Optional[str] = None,
) -> None:
    pain_col = next((c for c in config.get("event_columns.pain_binary", []) if events_df is not None and c in events_df.columns), None)
    if pain_col is None:
        _log("Events with pain binary column required; skipping channel contrasts.", logger, "warning")
        return

    if not isinstance(tfr, mne.time_frequency.EpochsTFR):
        _log("Contrast requires EpochsTFR; skipping channel contrasts.", logger, "warning")
        return

    n_epochs, n_meta = tfr.data.shape[0], len(events_df)
    n = min(n_epochs, n_meta)

    if getattr(tfr, "metadata", None) is not None and pain_col in tfr.metadata.columns:
        pain_vec = pd.to_numeric(tfr.metadata.iloc[:n][pain_col], errors="coerce").fillna(0).astype(int).values
    else:
        pain_vec = pd.to_numeric(events_df.iloc[:n][pain_col], errors="coerce").fillna(0).astype(int).values

    pain_mask = pain_vec == 1
    non_mask = pain_vec == 0

    if pain_mask.sum() == 0 or non_mask.sum() == 0:
        _log("One group has zero trials; skipping channel contrasts.", logger, "warning")
        return

    tfr_sub = tfr.copy()[:n]
    try:
        ensure_aligned_lengths(
            tfr_sub, pain_mask, non_mask,
            context=f"Pain contrast",
            strict=config.get("analysis.strict_mode", True),
            logger=logger
        )
    except ValueError as e:
        _log(f"{e}. Skipping contrast.", logger, "error")
        return
    if len(pain_mask) != len(tfr_sub):
        pain_mask = pain_mask[:len(tfr_sub)]
        non_mask = non_mask[:len(tfr_sub)]

    baseline_used = apply_baseline_and_crop(tfr_sub, baseline=baseline, mode="logratio", logger=logger)
    tfr_pain = tfr_sub[pain_mask].average()
    tfr_non = tfr_sub[non_mask].average()
    tfr_diff = tfr_pain.copy()
    tfr_diff.data = tfr_pain.data - tfr_non.data

    ch_names = tfr_pain.info["ch_names"]
    ch_dir = out_dir / "channels"
    ch_dir.mkdir(parents=True, exist_ok=True)

    fmax_available = float(np.max(tfr_pain.freqs))
    bands = _get_bands_for_tfr(max_freq_available=fmax_available, logger=logger, config=config)

    for ch in ch_names:
        fig = _fig(tfr_pain.plot(picks=ch, show=False))
        fig.suptitle(f"{ch} — Painful (baseline logratio)", fontsize=12)
        _save_fig(fig, ch_dir, f"tfr_{ch}_painful_bl.png", config=config, logger=logger, baseline_used=baseline_used)

        fig = _fig(tfr_non.plot(picks=ch, show=False))
        fig.suptitle(f"{ch} — Non-pain (baseline logratio)", fontsize=12)
        _save_fig(fig, ch_dir, f"tfr_{ch}_nonpain_bl.png", config=config, logger=logger, baseline_used=baseline_used)

        fig = _fig(tfr_diff.plot(picks=ch, show=False))
        fig.suptitle(f"{ch} — Pain minus Non-pain (baseline logratio)", fontsize=12)
        _save_fig(fig, ch_dir, f"tfr_{ch}_pain_minus_nonpain_bl.png", config=config, logger=logger, baseline_used=baseline_used)

        for band, (fmin, fmax) in bands.items():
            fmax_eff = min(fmax, fmax_available)
            if fmin >= fmax_eff:
                continue
            band_dir = ch_dir / band
            band_dir.mkdir(parents=True, exist_ok=True)

            fig_b = _fig(tfr_pain.plot(picks=ch, fmin=fmin, fmax=fmax_eff, show=False))
            fig_b.suptitle(f"{ch} — {band} Painful (baseline logratio)", fontsize=12)
            _save_fig(fig_b, band_dir, f"tfr_{ch}_{band}_painful_bl.png", config=config, logger=logger, baseline_used=baseline_used)

            fig_b = _fig(tfr_non.plot(picks=ch, fmin=fmin, fmax=fmax_eff, show=False))
            fig_b.suptitle(f"{ch} — {band} Non-pain (baseline logratio)", fontsize=12)
            _save_fig(fig_b, band_dir, f"tfr_{ch}_{band}_nonpain_bl.png", config=config, logger=logger, baseline_used=baseline_used)

            fig_b = _fig(tfr_diff.plot(picks=ch, fmin=fmin, fmax=fmax_eff, show=False))
            fig_b.suptitle(f"{ch} — {band} Pain minus Non-pain (baseline logratio)", fontsize=12)
            _save_fig(fig_b, band_dir, f"tfr_{ch}_{band}_pain_minus_nonpain_bl.png", config=config, logger=logger, baseline_used=baseline_used)


def qc_baseline_plateau_power(
    tfr,
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    logger: Optional[logging.Logger] = None,
) -> None:
    if baseline is None:
        baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-5.0, -0.01]))
    qc_dir = out_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    data = getattr(tfr, "data", None)
    if data is None or data.ndim not in [3, 4]:
        return

    if data.ndim == 3:
        data = data[None, ...]

    freqs = np.asarray(tfr.freqs)
    times = np.asarray(tfr.times)

    min_baseline_samples = config.get("tfr_topography_pipeline.min_baseline_samples", 5)
    b_start, b_end, tmask_base_idx = validate_baseline_indices(times, baseline, min_samples=min_baseline_samples, logger=logger)
    tmask_base = np.zeros(len(times), dtype=bool)
    tmask_base[tmask_base_idx] = True
    tmask_plat = (times >= plateau_window[0]) & (times < plateau_window[1])

    if not np.any(tmask_plat):
        _log(f"QC skipped: plateau samples={int(tmask_plat.sum())}", logger, "warning")
        return

    tfr_copy = tfr.copy()
    baseline_used = apply_baseline_and_crop(tfr_copy, baseline=baseline, mode="logratio", logger=logger)
    tfr_topo_avg = tfr_copy.average() if isinstance(tfr_copy, mne.time_frequency.EpochsTFR) else tfr_copy

    rows = []
    eps = 1e-20

    band_bounds = config.get("time_frequency_analysis.bands") or config.frequency_bands
    BAND_BOUNDS = {k: tuple(v) for k, v in band_bounds.items()}
    for band, (fmin, fmax) in BAND_BOUNDS.items():
        fmask = (freqs >= float(fmin)) & (freqs <= (float(fmax) if fmax is not None else freqs.max()))
        if not np.any(fmask):
            continue

        base = data[:, :, fmask, :][:, :, :, tmask_base].mean(axis=(2, 3))
        plat = data[:, :, fmask, :][:, :, :, tmask_plat].mean(axis=(2, 3))

        base_flat = base.reshape(-1)
        ratio_log = np.log10((plat.reshape(-1) + eps) / (base_flat + eps))

        fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
        axes[0].hist(base_flat, bins=50, color="tab:blue", alpha=0.8)
        axes[0].set_title(f"Baseline power — {band}")
        axes[0].set_xlabel("Power (a.u.)")
        axes[0].set_ylabel("Count")
        axes[1].hist(ratio_log, bins=50, color="tab:orange", alpha=0.8)
        axes[1].set_title(f"log10(plateau/baseline) — {band}")
        axes[1].set_xlabel("log10 ratio")
        axes[1].set_ylabel("Count")
        fig.suptitle(
            f"Baseline vs Plateau QC — {band}\n(baseline={b_start:.2f}–{b_end:.2f}s; plateau={plateau_window[0]:.2f}–{plateau_window[1]:.2f}s)",
            fontsize=10,
        )
        _save_fig(fig, qc_dir, f"qc_baseline_plateau_hist_{band}.png", config=config, logger=logger)

        topo_vals = None
        if tfr_topo_avg is not None:
            fmin_eff = float(fmin)
            fmax_eff = float(fmax) if fmax is not None else float(freqs.max())
            topo_vals = _average_tfr_band(
                tfr_topo_avg,
                fmin=fmin_eff,
                fmax=fmax_eff,
                tmin=float(plateau_window[0]),
                tmax=float(plateau_window[1]),
            )

        if topo_vals is not None and np.isfinite(topo_vals).any():
            fig2, ax2 = plt.subplots(1, 1, figsize=(4.8, 3.2), constrained_layout=True)
            ax2.hist(topo_vals, bins=50, color="tab:green", alpha=0.8)
            ax2.set_title(f"Per-channel Δ (topomap-consistent) — {band}")
            ax2.set_xlabel("log10 ratio")
            ax2.set_ylabel("Count")
            _save_fig(fig2, qc_dir, f"qc_band_topomap_values_hist_{band}.png", config=config, logger=logger)

        row = {
            "band": band,
            "baseline_mean": float(np.nanmean(base_flat)),
            "baseline_median": float(np.nanmedian(base_flat)),
            "plateau_mean": float(np.nanmean(plat.reshape(-1))),
            "plateau_median": float(np.nanmedian(plat.reshape(-1))),
            "log10_ratio_mean": float(np.nanmean(ratio_log)),
            "log10_ratio_median": float(np.nanmedian(ratio_log)),
            "n_baseline_samples": int(tmask_base.sum()),
            "n_plateau_samples": int(tmask_plat.sum()),
        }
        if topo_vals is not None and np.isfinite(topo_vals).any():
            row["log10_ratio_mean_topomap"] = float(np.nanmean(topo_vals))
            row["log10_ratio_median_topomap"] = float(np.nanmedian(topo_vals))
        else:
            row["log10_ratio_mean_topomap"] = float("nan")
            row["log10_ratio_median_topomap"] = float("nan")
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df_path = qc_dir / "qc_baseline_plateau_summary.tsv"
        df.to_csv(df_path, sep="\t", index=False)
        _log(f"Saved QC summary: {df_path}", logger)


def contrast_pain_nonpain(
    tfr: "mne.time_frequency.EpochsTFR | mne.time_frequency.AverageTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    plateau_window: Tuple[float, float],
    logger: Optional[logging.Logger] = None,
    subject: Optional[str] = None,
) -> None:
    if not isinstance(tfr, mne.time_frequency.EpochsTFR):
        _log("Contrast requires EpochsTFR (trial-level). Skipping contrasts and using only overall average.", logger, "warning")
        return

    pain_col = next((c for c in config.get("event_columns.pain_binary", []) if events_df is not None and c in events_df.columns), None)
    if pain_col is None:
        _log(f"Events with pain binary column required for contrast; skipping.", logger, "warning")
        return

    n_epochs, n_meta = tfr.data.shape[0], len(events_df)
    n = min(n_epochs, n_meta)

    if getattr(tfr, "metadata", None) is not None and pain_col in tfr.metadata.columns:
        pain_vec = pd.to_numeric(tfr.metadata.iloc[:n][pain_col], errors="coerce").fillna(0).astype(int).values
    else:
        pain_vec = pd.to_numeric(events_df.iloc[:n][pain_col], errors="coerce").fillna(0).astype(int).values

    pain_mask = pain_vec == 1
    non_mask = pain_vec == 0

    _log(f"Pain/non-pain counts (n={n}): pain={int(pain_mask.sum())}, non-pain={int(non_mask.sum())}.", logger)

    if pain_mask.sum() == 0 or non_mask.sum() == 0:
        _log("One of the groups has zero trials; skipping contrasts.", logger, "warning")
        return

    tfr_sub = tfr.copy()[:n]
    try:
        ensure_aligned_lengths(
            tfr_sub, pain_mask, non_mask,
            context=f"Pain contrast",
            strict=config.get("analysis.strict_mode", True),
            logger=logger
        )
    except ValueError as e:
        _log(f"{e}. Skipping contrast.", logger, "error")
        return
    if len(pain_mask) != len(tfr_sub):
        pain_mask = pain_mask[:len(tfr_sub)]
        non_mask = non_mask[:len(tfr_sub)]

    baseline_used = apply_baseline_and_crop(tfr_sub, baseline=baseline, mode="logratio", logger=logger)
    tfr_pain = tfr_sub[pain_mask].average()
    tfr_non = tfr_sub[non_mask].average()

    central_ch = _pick_central_channel(tfr_pain.info, preferred="Cz", logger=logger)
    ch_idx = tfr_pain.info["ch_names"].index(central_ch)
    arr_pain = np.asarray(tfr_pain.data[ch_idx])
    arr_non = np.asarray(tfr_non.data[ch_idx])
    vabs_pn = _robust_sym_vlim([arr_pain, arr_non])

    times = np.asarray(tfr_pain.times)
    tmin_req, tmax_req = plateau_window
    tmask = (times >= float(tmin_req)) & (times < float(tmax_req))
    if not np.any(tmask):
        msg = f"Plateau window [{tmin_req}, {tmax_req}] outside data range [{times.min():.2f}, {times.max():.2f}]"
        strict_mode = config.get("analysis.strict_mode", True) if config else True
        if strict_mode:
            raise ValueError(msg)
        _log(f"{msg}; using entire time span", logger, "warning")
        tmask = np.ones_like(times, dtype=bool)

    mu_pain = float(np.nanmean(arr_pain[:, tmask]))
    pct_pain = (10.0 ** (mu_pain) - 1.0) * 100.0
    mu_non = float(np.nanmean(arr_non[:, tmask]))
    pct_non = (10.0 ** (mu_non) - 1.0) * 100.0

    fig = _fig(tfr_pain.plot(picks=central_ch, vlim=(-vabs_pn, +vabs_pn), show=False))
    fig.suptitle(
        f"{central_ch} — Pain (baseline logratio)\nvlim ±{vabs_pn:.2f}; mean %Δ vs BL={pct_pain:+.0f}%",
        fontsize=12,
    )
    _save_fig(fig, out_dir, f"tfr_{central_ch}_pain_bl.png", config=config, logger=logger, baseline_used=baseline_used)

    fig = _fig(tfr_non.plot(picks=central_ch, vlim=(-vabs_pn, +vabs_pn), show=False))
    fig.suptitle(
        f"{central_ch} — Non-pain (baseline logratio)\nvlim ±{vabs_pn:.2f}; mean %Δ vs BL={pct_non:+.0f}%",
        fontsize=12,
    )
    _save_fig(fig, out_dir, f"tfr_{central_ch}_nonpain_bl.png", config=config, logger=logger, baseline_used=baseline_used)

    tfr_diff = tfr_pain.copy()
    tfr_diff.data = tfr_pain.data - tfr_non.data
    tfr_diff.comment = "pain-minus-nonpain"

    arr_diff = np.asarray(arr_pain) - np.asarray(arr_non)
    vabs_diff = _robust_sym_vlim(arr_diff)
    mu_diff = float(np.nanmean(arr_diff[:, tmask]))
    pct_diff = (10.0 ** (mu_diff) - 1.0) * 100.0

    fig = _fig(tfr_diff.plot(picks=central_ch, vlim=(-vabs_diff, +vabs_diff), show=False))
    fig.suptitle(
        f"{central_ch} — Pain minus Non (baseline logratio)\nvlim ±{vabs_diff:.2f}; Δ% vs BL={pct_diff:+.0f}%",
        fontsize=12,
    )
    _save_fig(fig, out_dir, f"tfr_{central_ch}_pain_minus_non_bl.png", config=config, logger=logger, baseline_used=baseline_used)

    times = np.asarray(tfr_pain.times)
    tmin_eff, tmax_eff = _clip_time_window(times, plateau_window)
    fmax_available = float(np.max(tfr_pain.freqs))
    bands = _get_bands_for_tfr(max_freq_available=fmax_available, logger=logger, config=config)
    tmin, tmax = tmin_eff, tmax_eff

    n_pain = int(pain_mask.sum())
    n_non = int(non_mask.sum())
    row_labels = [f"Pain (n={n_pain})", f"Non-pain (n={n_non})", "Pain - Non"]
    n_rows = 3
    n_cols = len(bands)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 3.5 * n_rows),
        squeeze=False,
        gridspec_kw={"wspace": 0.3, "hspace": 0.3},
    )
    viz_params = _get_viz_params(config)
    for c, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            for r in range(n_rows):
                axes[r, c].axis('off')
            continue

        pain_data = _average_tfr_band(tfr_pain, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        non_data = _average_tfr_band(tfr_non, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        if pain_data is None or non_data is None:
            for r in range(n_rows):
                axes[r, c].axis('off')
            continue

        diff_data = pain_data - non_data
        vabs_pn = _robust_sym_vlim([pain_data, non_data])
        diff_abs = _robust_sym_vlim(diff_data) if np.isfinite(diff_data).any() else 0.0

        sig_mask = None
        cluster_p_min = cluster_k = cluster_mass = None
        if viz_params["diff_annotation_enabled"]:
            sig_mask, cluster_p_min, cluster_k, cluster_mass = _cluster_test_epochs(
                tfr_sub, pain_mask, non_mask, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax, paired=False
            )

        ax = axes[0, c]
        _plot_topomap_on_ax(ax, pain_data, tfr_pain.info, vmin=-vabs_pn, vmax=+vabs_pn, config=config)
        pain_mu = float(np.nanmean(pain_data))
        ax.text(0.5, 1.02, f"%Δ={(10**pain_mu - 1)*100:+.1f}%", transform=ax.transAxes, ha="center", va="top", fontsize=9)

        ax = axes[1, c]
        _plot_topomap_on_ax(ax, non_data, tfr_pain.info, vmin=-vabs_pn, vmax=+vabs_pn, config=config)
        non_mu = float(np.nanmean(non_data))
        ax.text(0.5, 1.02, f"%Δ={(10**non_mu - 1)*100:+.1f}%", transform=ax.transAxes, ha="center", va="top", fontsize=9)

        ax = axes[2, c]
        _plot_topomap_on_ax(
            ax,
            diff_data,
            tfr_pain.info,
            mask=(sig_mask if viz_params["diff_annotation_enabled"] else None),
            mask_params=viz_params["sig_mask_params"],
            vmin=(-diff_abs if diff_abs > 0 else None),
            vmax=(+diff_abs if diff_abs > 0 else None),
            config=config,
        )
        diff_mu = float(np.nanmean(diff_data))
        pct_mu = (10**(diff_mu) - 1.0) * 100.0
        cl_txt = _format_cluster_ann(cluster_p_min, cluster_k, cluster_mass, config=config) if viz_params["diff_annotation_enabled"] else ""
        label = f"Δ%={pct_mu:+.1f}%" + (f" | {cl_txt}" if cl_txt else "")
        ax.text(0.5, 1.02, label, transform=ax.transAxes, ha="center", va="top", fontsize=9)

        axes[0, c].set_title(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=9, pad=4, y=1.04)

        sm_pn = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-vabs_pn, vcenter=0.0, vmax=vabs_pn), cmap=viz_params["topo_cmap"])
        sm_pn.set_array([])
        fig.colorbar(sm_pn, ax=[axes[0, c], axes[1, c]], fraction=viz_params["colorbar_fraction"], pad=viz_params["colorbar_pad"])
        if diff_abs > 0:
            sm_diff = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-diff_abs, vcenter=0.0, vmax=diff_abs), cmap=viz_params["topo_cmap"])
            sm_diff.set_array([])
            fig.colorbar(sm_diff, ax=axes[2, c], fraction=viz_params["colorbar_fraction"], pad=viz_params["colorbar_pad"])
    axes[0, 0].set_ylabel(row_labels[0], fontsize=10)
    axes[1, 0].set_ylabel(row_labels[1], fontsize=10)
    axes[2, 0].set_ylabel(row_labels[2], fontsize=10)
    fig.suptitle(f"Topomaps (baseline: logratio; t=[{tmin:.1f}, {tmax:.1f}] s)", fontsize=12)
    fig.supxlabel("Frequency bands", fontsize=10)
    _save_fig(fig, out_dir, "topomap_grid_bands_pain_non_diff_bl.png", config=config)


###################################################################
# Temperature Contrasts (Subject)
###################################################################

def contrast_maxmin_temperature(
    tfr: "mne.time_frequency.EpochsTFR | mne.time_frequency.AverageTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    logger: Optional[logging.Logger] = None,
) -> None:
    if baseline is None:
        baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-5.0, -0.01]))
    if not isinstance(tfr, mne.time_frequency.EpochsTFR):
        _log("Max-vs-min temperature contrast requires EpochsTFR; skipping.")
        return
    if events_df is None:
        _log("Max-vs-min temperature contrast requires events_df; skipping.")
        return
    temp_col = next((c for c in config.get("event_columns.temperature", []) if c in events_df.columns), None)
    if temp_col is None:
        _log("Max-vs-min temperature contrast: no temperature column found; skipping.")
        return

    n_epochs = tfr.data.shape[0]
    n_meta = len(events_df)
    n = min(n_epochs, n_meta)
    if getattr(tfr, "metadata", None) is not None and temp_col in tfr.metadata.columns:
        temp_series = tfr.metadata.iloc[:n][temp_col]
    else:
        temp_series = events_df.iloc[:n][temp_col]

    s_round = pd.to_numeric(temp_series, errors="coerce").round(1)
    temps = sorted(map(float, s_round.dropna().unique()))
    if len(temps) < 2:
        _log("Max-vs-min temperature contrast: need at least 2 temperature levels; skipping.")
        return

    t_min = float(min(temps))
    t_max = float(max(temps))

    mask_min = np.asarray(s_round == round(t_min, 1), dtype=bool)
    mask_max = np.asarray(s_round == round(t_max, 1), dtype=bool)
    if mask_min.sum() == 0 or mask_max.sum() == 0:
        _log(f"Max-vs-min temperature contrast: zero trials in one group (min n={int(mask_min.sum())}, max n={int(mask_max.sum())}); skipping.")
        return

    tfr_sub = tfr.copy()[:n]
    try:
        strict_mode = config.get("analysis.strict_mode", True)
        ensure_aligned_lengths(
            tfr_sub, mask_min, mask_max,
            context=f"Temperature contrast",
            strict=strict_mode,
            logger=logger
        )
    except ValueError as e:
        _log(f"{e}. Skipping contrast.", logger, "error")
        return
    if len(mask_min) != len(tfr_sub) or len(mask_max) != len(tfr_sub):
        mask_min = mask_min[:len(tfr_sub)]
        mask_max = mask_max[:len(tfr_sub)]

    baseline_used = apply_baseline_and_crop(tfr_sub, baseline=baseline, mode="logratio", logger=logger)
    tfr_min = tfr_sub[mask_min].average()
    tfr_max = tfr_sub[mask_max].average()

    times = np.asarray(tfr_max.times)
    tmin_req, tmax_req = plateau_window
    tmin_eff = float(max(times.min(), tmin_req))
    tmax_eff = float(min(times.max(), tmax_req))
    tmin, tmax = tmin_eff, tmax_eff

    fmax_available = float(np.max(tfr_max.freqs))
    bands = _get_bands_for_tfr(max_freq_available=fmax_available, logger=logger, config=config)

    n_rows = len(bands)
    n_cols = 4
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 3.5 * n_rows),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.25, 1.0], "wspace": 0.3},
    )
    for r, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            for c in range(n_cols):
                axes[r, c].axis("off")
            continue
        max_data = _average_tfr_band(tfr_max, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        min_data = _average_tfr_band(tfr_min, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        if max_data is None or min_data is None:
            for c in range(n_cols):
                axes[r, c].axis("off")
            continue

        sig_mask = None
        cluster_p_min = cluster_k = cluster_mass = None
        if _get_viz_params(config)["diff_annotation_enabled"]:
            sig_mask, cluster_p_min, cluster_k, cluster_mass = _cluster_test_epochs(
                tfr_sub, mask_max, mask_min, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax, paired=False
            )

        diff_data = max_data - min_data
        max_mu = float(np.nanmean(max_data))
        min_mu = float(np.nanmean(min_data))
        diff_mu = float(np.nanmean(diff_data))
        vabs_pn = _robust_sym_vlim([max_data, min_data])
        diff_abs = _robust_sym_vlim(diff_data) if np.isfinite(diff_data).any() else 0.0

        ax = axes[r, 0]
        _plot_topomap_on_ax(ax, max_data, tfr_max.info, vmin=-vabs_pn, vmax=+vabs_pn)
        ax.text(0.5, 1.02, f"%Δ={(10**max_mu - 1)*100:+.1f}%", transform=ax.transAxes, ha="center", va="top", fontsize=9)

        ax = axes[r, 1]
        _plot_topomap_on_ax(ax, min_data, tfr_min.info, vmin=-vabs_pn, vmax=+vabs_pn)
        ax.text(0.5, 1.02, f"%Δ={(10**min_mu - 1)*100:+.1f}%", transform=ax.transAxes, ha="center", va="top", fontsize=9)

        axes[r, 2].axis("off")

        ax = axes[r, 3]
        _plot_topomap_on_ax(
            ax,
            diff_data,
            tfr_max.info,
            mask=(sig_mask if _get_viz_params(config)["diff_annotation_enabled"] else None),
            mask_params=_get_viz_params(config)["sig_mask_params"],
            vmin=(-diff_abs if diff_abs > 0 else None),
            vmax=(+diff_abs if diff_abs > 0 else None),
        )
        pct_mu = (10**(diff_mu) - 1.0) * 100.0
        cl_txt = _format_cluster_ann(cluster_p_min, cluster_k, cluster_mass, config=config) if _get_viz_params(config)["diff_annotation_enabled"] else ""
        label = f"Δ%={pct_mu:+.1f}%" + (f" | {cl_txt}" if cl_txt else "")
        ax.text(0.5, 1.02, label, transform=ax.transAxes, ha="center", va="top", fontsize=9)

        if r == 0:
            axes[r, 0].set_title(f"Max {t_max:.1f}°C (n={int(mask_max.sum())})", fontsize=9, pad=4, y=1.04)
            axes[r, 1].set_title(f"Min {t_min:.1f}°C (n={int(mask_min.sum())})", fontsize=9, pad=4, y=1.04)
            axes[r, 3].set_title("Max - Min", fontsize=9, pad=4, y=1.04)
        axes[r, 0].set_ylabel(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=10)

        sm_pn = ScalarMappable(norm=mcolors.Normalize(vmin=-vabs_pn, vmax=+vabs_pn), cmap=_get_viz_params(config)["topo_cmap"])
        sm_pn.set_array([])
        fig.colorbar(
            sm_pn, ax=[axes[r, 0], axes[r, 1]], fraction=_get_viz_params(config)["colorbar_fraction"], pad=_get_viz_params(config)["colorbar_pad"]
        )
        if diff_abs > 0:
            sm_diff = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-diff_abs, vcenter=0.0, vmax=diff_abs), cmap=_get_viz_params(config)["topo_cmap"])
            sm_diff.set_array([])
            fig.colorbar(
                sm_diff, ax=axes[r, 3], fraction=_get_viz_params(config)["colorbar_fraction"], pad=_get_viz_params(config)["colorbar_pad"]
            )

    fig.suptitle(
        f"Topomaps: Max vs Min temperature (baseline: logratio; t=[{tmin:.1f}, {tmax:.1f}] s)",
        fontsize=12,
    )
    fig.supylabel("Frequency bands", fontsize=10)
    _save_fig(
        fig,
        out_dir,
        "topomap_grid_bands_maxmin_temp_diff_bl.png",
        config,
    )


def contrast_pain_nonpain_topomaps_rois(
    tfr: "mne.time_frequency.EpochsTFR",
    events_df: Optional[pd.DataFrame],
    roi_map: Dict[str, list[str]],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    logger: Optional[logging.Logger] = None,
) -> None:
    if baseline is None:
        baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-5.0, -0.01]))
    if not isinstance(tfr, mne.time_frequency.EpochsTFR):
        _log("ROI topomap contrast requires EpochsTFR; skipping.")
        return
    pain_col = next((c for c in config.get("event_columns.pain_binary", []) if events_df is not None and c in events_df.columns), None)
    if pain_col is None:
        _log(f"Events with pain binary column required for ROI topomap contrasts; skipping.", logger, "warning")
        return

    n_epochs = tfr.data.shape[0]
    n_meta = len(events_df)
    n = min(n_epochs, n_meta)
    if n_epochs != n_meta and not (getattr(tfr, "metadata", None) is not None and pain_col in tfr.metadata.columns):
        _log(f"ROI topomaps: tfr epochs ({n_epochs}) != events rows ({n_meta}) and no matching pain column in TFR metadata; skipping.", None, "warning")
        return

    if getattr(tfr, "metadata", None) is not None and pain_col in tfr.metadata.columns:
        pain_vec = pd.to_numeric(tfr.metadata.iloc[:n][pain_col], errors="coerce").fillna(0).astype(int).values
    else:
        pain_vec = pd.to_numeric(events_df.iloc[:n][pain_col], errors="coerce").fillna(0).astype(int).values

    pain_mask = np.asarray(pain_vec == 1, dtype=bool)
    non_mask = np.asarray(pain_vec == 0, dtype=bool)
    _log(f"ROI topomaps pain/non-pain counts (n={n}): pain={int(pain_mask.sum())}, non-pain={int(non_mask.sum())}.")
    if pain_mask.sum() == 0 or non_mask.sum() == 0:
        _log("ROI topomaps: one of the groups has zero trials; skipping.", None, "warning")
        return

    tfr_sub = tfr.copy()[:n]
    try:
        ensure_aligned_lengths(
            tfr_sub, pain_mask, non_mask,
            context=f"Pain contrast",
            strict=config.get("analysis.strict_mode", True),
            logger=logger
        )
        if len(pain_mask) != len(tfr_sub):
            pain_mask = pain_mask[:len(tfr_sub)]
            non_mask = non_mask[:len(tfr_sub)]
    except ValueError as e:
        _log(f"{e}. Skipping contrast.", logger, "error")
        return

    baseline_used = apply_baseline_and_crop(tfr_sub, baseline=baseline, mode="logratio", logger=logger)
    tfr_pain = tfr_sub[pain_mask].average()
    tfr_non = tfr_sub[non_mask].average()

    fmax_available = float(np.max(tfr_pain.freqs))
    band_bounds = config.get("time_frequency_analysis.bands") or config.frequency_bands
    BAND_BOUNDS = {k: tuple(v) for k, v in band_bounds.items()}
    bands = {}
    if "theta" in BAND_BOUNDS:
        bands["theta"] = BAND_BOUNDS["theta"]
    bands["alpha"] = BAND_BOUNDS["alpha"]
    bands["beta"] = BAND_BOUNDS["beta"]
    bands["gamma"] = (
        BAND_BOUNDS["gamma"][0],
        fmax_available if BAND_BOUNDS["gamma"][1] is None else BAND_BOUNDS["gamma"][1],
    )

    if len(bands) == 0:
        _log("ROI topomaps: no frequency bands available; skipping.")
        return

    times = np.asarray(tfr_pain.times)
    tmin_req, tmax_req = plateau_window
    tmin_eff = float(max(times.min(), tmin_req))
    tmax_eff = float(min(times.max(), tmax_req))
    tmin, tmax = tmin_eff, tmax_eff

    ch_names = tfr_pain.info["ch_names"]
    n_pain = int(pain_mask.sum())
    n_non = int(non_mask.sum())
    cond_labels = [f"Pain (n={n_pain})", f"Non-pain (n={n_non})", "", "Pain - Non"]
    roi_highlight_enabled = config.get("time_frequency_analysis.roi_highlight_enabled", False)
    mask_params = config.get("time_frequency_analysis.roi_mask_params", {
        "marker": "o",
        "markerfacecolor": "w",
        "markeredgecolor": "k",
        "linewidth": 0.5,
        "markersize": 4,
    })
    for roi, roi_chs in roi_map.items():
        mask_vec = np.array([ch in roi_chs for ch in ch_names], dtype=bool)

        n_rows = 3
        n_cols = len(bands)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.5 * n_cols, 3.5 * n_rows),
            squeeze=False,
            gridspec_kw={"wspace": 0.3, "hspace": 0.3},
        )
        for c, (band, (fmin, fmax)) in enumerate(bands.items()):
            fmax_eff = min(fmax, fmax_available)
            if fmin >= fmax_eff:
                for r in range(n_rows):
                    axes[r, c].axis('off')
                continue

            pain_data = _average_tfr_band(tfr_pain, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
            non_data = _average_tfr_band(tfr_non, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
            if pain_data is None or non_data is None:
                for r in range(n_rows):
                    axes[r, c].axis('off')
                continue

            diff_data = pain_data - non_data
            vabs_pn = _robust_sym_vlim([pain_data, non_data])
            vmin, vmax = -vabs_pn, +vabs_pn
            diff_abs = _robust_sym_vlim(diff_data) if np.isfinite(diff_data).any() else 0.0

            if mask_vec.any():
                pain_mu = float(np.nanmean(pain_data[mask_vec]))
                non_mu = float(np.nanmean(non_data[mask_vec]))
                diff_mu = float(np.nanmean(diff_data[mask_vec]))
            else:
                pain_mu = float(np.nanmean(pain_data))
                non_mu = float(np.nanmean(non_data))
                diff_mu = float(np.nanmean(diff_data))
            ax = axes[0, c]
            _plot_topomap_on_ax(
                ax,
                pain_data,
                tfr_pain.info,
                mask=(mask_vec if roi_highlight_enabled else None),
                mask_params=mask_params,
                vmin=vmin,
                vmax=vmax,
            )
            ax.text(0.5, 1.02, f"%Δ_ROI={(10**(pain_mu) - 1.0)*100:+.1f}%", transform=ax.transAxes, ha="center", va="top", fontsize=8)

            ax = axes[1, c]
            _plot_topomap_on_ax(
                ax,
                non_data,
                tfr_pain.info,
                mask=(mask_vec if roi_highlight_enabled else None),
                mask_params=mask_params,
                vmin=vmin,
                vmax=vmax,
            )
            ax.text(0.5, 1.02, f"%Δ_ROI={(10**(non_mu) - 1.0)*100:+.1f}%", transform=ax.transAxes, ha="center", va="top", fontsize=8)
            roi_picks = np.where(mask_vec)[0]
            sig_mask = None
            cluster_p_min = cluster_k = cluster_mass = None
            if _get_viz_params(config)["diff_annotation_enabled"]:
                sig_mask, cluster_p_min, cluster_k, cluster_mass = _cluster_test_epochs(
                    tfr_sub, pain_mask, non_mask, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax, paired=False,
                    restrict_picks=roi_picks,
                )

            ax = axes[2, c]
            _plot_topomap_on_ax(
                ax,
                diff_data,
                tfr_pain.info,
                mask=(mask_vec if roi_highlight_enabled else None),
                mask_params=mask_params,
                vmin=(-diff_abs if diff_abs > 0 else None),
                vmax=(+diff_abs if diff_abs > 0 else None),
            )
            if _get_viz_params(config)["diff_annotation_enabled"] and sig_mask is not None and sig_mask.any():
                _plot_topomap_on_ax(
                    ax,
                    diff_data,
                    tfr_pain.info,
                    mask=sig_mask,
                    mask_params=_get_viz_params(config)["sig_mask_params"],
                    vmin=(-diff_abs if diff_abs > 0 else None),
                    vmax=(+diff_abs if diff_abs > 0 else None),
                )

            pct_mu = (10**(diff_mu) - 1.0) * 100.0
            cl_txt = _format_cluster_ann(cluster_p_min, cluster_k, cluster_mass, config=config) if _get_viz_params(config)["diff_annotation_enabled"] else ""
            label = f"Δ%_ROI={pct_mu:+.1f}%" + (f" | {cl_txt}" if cl_txt else "")
            ax.text(0.5, 1.02, label, transform=ax.transAxes, ha="center", va="top", fontsize=8)

            axes[0, c].set_title(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=9, pad=4, y=1.04)

            sm_pn = ScalarMappable(norm=mcolors.Normalize(vmin=-vabs_pn, vmax=+vabs_pn), cmap=_get_viz_params(config)["topo_cmap"])
            sm_pn.set_array([])
            fig.colorbar(sm_pn, ax=[axes[0, c], axes[1, c]], fraction=_get_viz_params(config)["colorbar_fraction"], pad=_get_viz_params(config)["colorbar_pad"])
            if diff_abs > 0:
                sm_diff = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-diff_abs, vcenter=0.0, vmax=diff_abs), cmap=_get_viz_params(config)["topo_cmap"])
                sm_diff.set_array([])
                fig.colorbar(sm_diff, ax=axes[2, c], fraction=_get_viz_params(config)["colorbar_fraction"], pad=_get_viz_params(config)["colorbar_pad"])
        axes[0, 0].set_ylabel(cond_labels[0], fontsize=10)
        axes[1, 0].set_ylabel(cond_labels[1], fontsize=10)
        axes[2, 0].set_ylabel(cond_labels[3], fontsize=10)
        fig.suptitle(f"ROI: {roi} — Topomaps (baseline: logratio; t=[{tmin:.1f}, {tmax:.1f}] s)", fontsize=12)
        fig.supxlabel("Frequency bands", fontsize=10)
        _save_fig(
            fig,
            out_dir,
            f"topomap_ROI-{_sanitize(roi)}_grid_bands_pain_non_diff_bl.png",
            config,
            
        )


###################################################################
# ROI Processing (Subject)
###################################################################

def compute_roi_tfrs(
    epochs: mne.Epochs,
    freqs: np.ndarray,
    n_cycles: np.ndarray,
    config,
    roi_map: Optional[Dict[str, list[str]]] = None,
) -> Dict[str, mne.time_frequency.EpochsTFR]:
    if roi_map is None:
        from eeg_pipeline.utils.tfr_utils import build_rois_from_info as _build_rois
        roi_map = _build_rois(epochs.info, config=config)
    roi_tfrs = {}
    for roi, chs in roi_map.items():
        picks = mne.pick_channels(epochs.ch_names, include=chs, ordered=True)
        if len(picks) == 0:
            continue
        data = epochs.get_data()
        roi_data = data[:, picks, :].mean(axis=1, keepdims=True)
        info = mne.create_info([roi], sfreq=epochs.info['sfreq'], ch_types='eeg')
        epo_roi = mne.EpochsArray(
            roi_data,
            info,
            events=epochs.events,
            event_id=epochs.event_id,
            tmin=epochs.tmin,
            metadata=epochs.metadata,
            verbose=False,
        )
        power = mne.time_frequency.tfr_morlet(
            epo_roi, freqs=freqs, n_cycles=n_cycles, use_fft=True,
            return_itc=False, average=False, decim=config.get("tfr_topography_pipeline.tfr.decim", 4),
            n_jobs=config.get("tfr_topography_pipeline.tfr.workers", -1), picks="eeg", verbose=False
        )
        roi_tfrs[roi] = power
    return roi_tfrs


def plot_rois_all_trials(
    roi_tfrs: Dict[str, mne.time_frequency.EpochsTFR],
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    logger: Optional[logging.Logger] = None,
) -> None:
    rois_dir = out_dir / "rois"
    for roi, tfr in roi_tfrs.items():
        tfr_c = tfr.copy()
        baseline_used = apply_baseline_and_crop(tfr_c, baseline=baseline, mode="logratio", logger=logger)
        tfr_avg = tfr_c.average()
        ch = tfr_avg.info['ch_names'][0]
        roi_tag = _sanitize(roi)
        roi_dir = rois_dir / roi_tag

        fig = _fig(tfr_avg.plot(picks=ch, show=False))
        fig.suptitle(f"ROI: {roi} — all trials (baseline logratio)", fontsize=12)
        _save_fig(fig, roi_dir, "tfr_all_trials_bl.png", config=config, logger=logger, baseline_used=baseline_used)

        fmax_available = float(np.max(tfr_avg.freqs))
        bands = _get_bands_for_tfr(max_freq_available=fmax_available, logger=logger, config=config)
        for band, (fmin, fmax) in bands.items():
            fmax_eff = min(fmax, fmax_available)
            if fmin >= fmax_eff:
                continue
            band_dir = roi_dir / band
            fig_b = tfr_avg.plot(picks=ch, fmin=fmin, fmax=fmax_eff, show=False)
            fig_b = fig_b[0] if isinstance(fig_b, list) else fig_b
            fig_b.suptitle(f"ROI: {roi} — {band} band (baseline logratio)", fontsize=12)
            _save_fig(fig_b, band_dir, f"tfr_{band}_all_trials_bl.png", config=config, logger=logger, baseline_used=baseline_used)


def plot_topomaps_bands_all_trials(
    tfr: "mne.time_frequency.EpochsTFR | mne.time_frequency.AverageTFR",
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    logger: Optional[logging.Logger] = None,
) -> None:
    if baseline is None:
        baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-5.0, -0.01]))
    tfr_all = tfr.copy()
    baseline_used = apply_baseline_and_crop(tfr_all, baseline=baseline, mode="logratio", logger=logger)
    tfr_avg = tfr_all.average() if isinstance(tfr_all, mne.time_frequency.EpochsTFR) else tfr_all

    fmax_available = float(np.max(tfr_avg.freqs))
    bands = _get_bands_for_tfr(max_freq_available=fmax_available, logger=logger, config=config)
    times = np.asarray(tfr_avg.times)
    tmin, tmax = _clip_time_window(times, plateau_window)

    tfr_corrected = tfr_avg
    n_rows = len(bands)
    fig, axes = plt.subplots(n_rows, 1, figsize=(4.0, 3.5 * n_rows), squeeze=False)
    for r, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            axes[r, 0].axis('off')
            continue
        data = _average_tfr_band(tfr_corrected, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        if data is None:
            axes[r, 0].axis('off')
            continue
        vabs = _robust_sym_vlim(data)
        vmin, vmax = -vabs, +vabs

        _plot_topomap_on_ax(axes[r, 0], data, tfr_avg.info, vmin=vmin, vmax=vmax)

        eeg_picks = mne.pick_types(tfr_avg.info, eeg=True, exclude=[])
        mu = float(np.nanmean(data[eeg_picks]))
        pct = (10.0 ** (mu) - 1.0) * 100.0
        axes[r, 0].text(
            0.5,
            1.02,
            f"%Δ={pct:+.0f}%",
            transform=axes[r, 0].transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )
        axes[r, 0].set_ylabel(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=10)

        sm = ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=_get_viz_params(config)["topo_cmap"])
        sm.set_array([])
        cbar = fig.colorbar(
            sm, ax=axes[r, 0], fraction=_get_viz_params(config)["colorbar_fraction"], pad=_get_viz_params(config)["colorbar_pad"]
        )
        cbar.set_label("log10(power/baseline)")
    fig.suptitle(f"Topomaps (all trials; baseline: logratio; t=[{tmin:.1f}, {tmax:.1f}] s)", fontsize=12)
    fig.supylabel("Frequency bands", fontsize=10)
    fig.supxlabel("All trials", fontsize=10)
    _save_fig(fig, out_dir, f"topomap_grid_bands_all_trials_bl.png", config=config)


###################################################################
# Group Plots
###################################################################

def group_topomaps_bands_all_trials(
    powers: List["mne.time_frequency.EpochsTFR"],
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    plateau_window: Tuple[float, float],
    logger: Optional[logging.Logger] = None,
) -> None:
    if not powers:
        return
    avg_list = []
    for p in powers:
        avg_list.append(avg_alltrials_to_avg_tfr(p, baseline=baseline, logger=logger))
    info_common, data = align_avg_tfrs(avg_list, logger=logger)
    if info_common is None or data is None:
        logger and logger.warning("Group all-trials: no aligned data across subjects")
        return

    mean_data = data.mean(axis=0)
    freqs = np.asarray(avg_list[0].freqs)
    times = np.asarray(avg_list[0].times)
    fmax_available = float(freqs.max())
    bands = _get_bands_for_tfr(max_freq_available=fmax_available, logger=logger, config=config)
    tmin, tmax = _clip_time_window(times, plateau_window)

    n_cols = len(bands)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.8 * n_cols, 4.8))
    if n_cols == 1:
        axes = [axes]
    plt.subplots_adjust(left=0.06, right=0.98, top=0.83, bottom=0.20, wspace=0.08)

    for c, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        ax = axes[c]
        if fmin >= fmax_eff:
            ax.axis('off')
            continue
        fmask = (freqs >= fmin) & (freqs <= fmax_eff)
        tmask = (times >= tmin) & (times < tmax)
        if fmask.sum() == 0 or tmask.sum() == 0:
            ax.axis('off')
            continue
        vec = mean_data[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
        vabs = _robust_sym_vlim(vec)

        viz_params = _get_viz_params(config)
        _plot_topomap_on_ax(ax, vec, info_common, vmin=-vabs, vmax=+vabs, config=config)

        eeg_picks = mne.pick_types(info_common, eeg=True, exclude=[])
        mu = float(np.nanmean(vec[eeg_picks])) if len(eeg_picks) > 0 else float(np.nanmean(vec))
        pct = (10.0 ** (mu) - 1.0) * 100.0
        ax.text(0.5, 1.02, f"%Δ={pct:+.0f}%", transform=ax.transAxes, ha="center", va="top", fontsize=9)

        ax.set_title(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=10)

        sm = ScalarMappable(norm=mcolors.Normalize(vmin=-vabs, vmax=+vabs), cmap=viz_params["topo_cmap"])
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=viz_params["colorbar_fraction"], pad=viz_params["colorbar_pad"])

    fig.suptitle(f"Group topomaps (all trials; baseline: logratio; t=[{tmin:.1f}, {tmax:.1f}] s)", fontsize=12)
    _save_fig(fig, out_dir, f"group_topomap_grid_bands_all_trials_bl.png", config=config, logger=logger, baseline_used=baseline)


def group_topomaps_pain_nonpain(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_by_subj: List[Optional[pd.DataFrame]],
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    plateau_window: Tuple[float, float],
    logger: Optional[logging.Logger] = None,
) -> None:
    if not powers:
        return
    pain_cols = config.get("event_columns.pain_binary", [])
    avg_pain = []
    avg_non = []
    for power, ev in zip(powers, events_by_subj):
        if ev is None:
            continue
        pain_col = next((c for c in pain_cols if c in ev.columns), None)
        if pain_col is None:
            continue
        vals = pd.to_numeric(ev[pain_col], errors="coerce").fillna(0).astype(int).values
        pain_mask = np.asarray(vals == 1, dtype=bool)
        non_mask = np.asarray(vals == 0, dtype=bool)
        if pain_mask.sum() == 0 or non_mask.sum() == 0:
            continue
        a_p = avg_by_mask_to_avg_tfr(power, pain_mask, baseline=baseline, logger=logger)
        a_n = avg_by_mask_to_avg_tfr(power, non_mask, baseline=baseline, logger=logger)
        if a_p is not None and a_n is not None:
            avg_pain.append(a_p)
            avg_non.append(a_n)
    if len(avg_pain) == 0 or len(avg_non) == 0:
        logger and logger.warning("Group pain/non-pain: insufficient aligned subjects")
        return

    info_p, data_p, data_n = align_paired_avg_tfrs(avg_pain, avg_non, logger=logger)
    if info_p is None or data_p is None or data_n is None:
        logger and logger.warning("Group pain/non-pain: could not align paired data")
        return

    n_subj = int(data_p.shape[0])
    mean_p = data_p.mean(axis=0)
    mean_n = data_n.mean(axis=0)

    fmax_available = float(np.max(avg_pain[0].freqs))
    bands = _get_bands_for_tfr(max_freq_available=fmax_available, logger=logger, config=config)
    times = np.asarray(avg_pain[0].times)
    tmin, tmax = _clip_time_window(times, plateau_window)

    n_rows = 3
    n_cols = len(bands)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows), squeeze=False)
    for c, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            for r in range(n_rows):
                axes[r, c].axis('off')
            continue
        freqs = np.asarray(avg_pain[0].freqs)
        times = np.asarray(avg_pain[0].times)
        fmask = (freqs >= fmin) & (freqs <= fmax_eff)
        tmask = (times >= tmin) & (times < tmax)
        if fmask.sum() == 0 or tmask.sum() == 0:
            for r in range(n_rows):
                axes[r, c].axis('off')
            continue
        v_p = mean_p[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
        v_n = mean_n[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
        v_d = v_p - v_n
        vabs_pn = _robust_sym_vlim([v_p, v_n])
        vabs_d = _robust_sym_vlim(v_d)
        _plot_topomap_on_ax(axes[0, c], v_p, info_p, vmin=-vabs_pn, vmax=+vabs_pn)
        eeg_picks = mne.pick_types(info_p, eeg=True, exclude=[])
        mu_p = float(np.nanmean(v_p[eeg_picks])) if len(eeg_picks) > 0 else float(np.nanmean(v_p))
        pct_p = (10.0 ** (mu_p) - 1.0) * 100.0
        axes[0, c].text(0.5, 1.02, f"%Δ={pct_p:+.0f}%", transform=axes[0, c].transAxes, ha="center", va="top", fontsize=9)
        _plot_topomap_on_ax(axes[1, c], v_n, info_p, vmin=-vabs_pn, vmax=+vabs_pn)
        mu_n = float(np.nanmean(v_n[eeg_picks])) if len(eeg_picks) > 0 else float(np.nanmean(v_n))
        pct_n = (10.0 ** (mu_n) - 1.0) * 100.0
        axes[1, c].text(0.5, 1.02, f"%Δ={pct_n:+.0f}%", transform=axes[1, c].transAxes, ha="center", va="top", fontsize=9)
        sig_mask = None
        cluster_p_min = cluster_k = cluster_mass = None
        fdr_txt = ""
        subj_p = data_p[:, :, fmask, :][:, :, :, tmask].mean(axis=(2, 3))
        subj_n = data_n[:, :, fmask, :][:, :, :, tmask].mean(axis=(2, 3))
        if _get_viz_params(config)["diff_annotation_enabled"]:
            sig_mask, cluster_p_min, cluster_k, cluster_mass = _cluster_test_two_sample_arrays(
                subj_p, subj_n, info_p, alpha=config.get("statistics.sig_alpha", 0.05), paired=True, n_permutations=config.get("statistics.cluster_n_perm", 1024)
            )
            if sig_mask is None:
                res = ttest_rel(subj_p, subj_n, axis=0, nan_policy="omit")
                p_ch = np.asarray(res.pvalue)
                sig_mask = _fdr_bh_mask(p_ch, alpha=config.get("statistics.sig_alpha", 0.05))
                rej, q = _fdr_bh_values(p_ch, alpha=config.get("statistics.sig_alpha", 0.05))
                k_rej = int(np.nansum(rej)) if rej is not None else 0
                q_min = float(np.nanmin(q)) if q is not None and np.isfinite(q).any() else None
                fdr_txt = _format_cluster_ann(q_min, k_rej if k_rej > 0 else None, config=config)
        _plot_topomap_on_ax(
            axes[2, c], v_d, info_p,
            mask=(sig_mask if _get_viz_params(config)["diff_annotation_enabled"] else None), mask_params=_get_viz_params(config)["sig_mask_params"],
            vmin=-vabs_d, vmax=+vabs_d,
        )
        mu_d = float(np.nanmean(v_d))
        pct_d = (10.0 ** (mu_d) - 1.0) * 100.0
        cl_txt = (_format_cluster_ann(cluster_p_min, cluster_k, cluster_mass, config=config) or fdr_txt) if _get_viz_params(config)["diff_annotation_enabled"] else ""
        label = f"Δ%={pct_d:+.1f}%" + (f" | {cl_txt}" if cl_txt else "")
        axes[2, c].text(0.5, 1.02, label, transform=axes[2, c].transAxes, ha="center", va="top", fontsize=9)
        axes[0, c].set_title(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=9, pad=4, y=1.04)
        sm_pn = ScalarMappable(norm=mcolors.Normalize(vmin=-vabs_pn, vmax=+vabs_pn), cmap=_get_viz_params(config)["topo_cmap"])
        sm_pn.set_array([])
        fig.colorbar(sm_pn, ax=[axes[0, c], axes[1, c]], fraction=_get_viz_params(config)["colorbar_fraction"], pad=_get_viz_params(config)["colorbar_pad"])
        sm_d = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-vabs_d, vcenter=0.0, vmax=vabs_d), cmap=_get_viz_params(config)["topo_cmap"])
        sm_d.set_array([])
        fig.colorbar(sm_d, ax=axes[2, c], fraction=_get_viz_params(config)["colorbar_fraction"], pad=_get_viz_params(config)["colorbar_pad"])
    axes[0, 0].set_ylabel("Pain", fontsize=10)
    axes[1, 0].set_ylabel("Non-pain", fontsize=10)
    axes[2, 0].set_ylabel("Pain - Non", fontsize=10)
    fig.suptitle(f"Group topomaps: Pain vs Non-pain (baseline: logratio; t=[{tmin:.1f}, {tmax:.1f}] s)", fontsize=12)
    fig.supxlabel("Frequency bands", fontsize=10)
    _save_fig(fig, out_dir, "group_topomap_grid_bands_pain_vs_nonpain_bl.png", config=config, logger=logger)


def group_pain_nonpain_temporal_topomaps(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_by_subj: List[Optional[pd.DataFrame]],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    window_count: int = 5,
    logger: Optional[logging.Logger] = None,
) -> None:
    if not powers:
        return
    if baseline is None:
        baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-5.0, -0.01]))
    pain_cols = config.get("event_columns.pain_binary", [])
    avg_pain, avg_non = [], []
    for power, ev in zip(powers, events_by_subj):
        if ev is None:
            continue
        pain_col = next((c for c in pain_cols if c in ev.columns), None)
        if pain_col is None:
            continue
        vals = pd.to_numeric(ev[pain_col], errors="coerce").fillna(0).astype(int).values
        pain_mask, non_mask = vals == 1, vals == 0
        if pain_mask.sum() == 0 or non_mask.sum() == 0:
            continue
        a_p = avg_by_mask_to_avg_tfr(power, pain_mask, baseline=baseline, logger=logger)
        a_n = avg_by_mask_to_avg_tfr(power, non_mask, baseline=baseline, logger=logger)
        if a_p is not None and a_n is not None:
            avg_pain.append(a_p)
            avg_non.append(a_n)

    if len(avg_pain) < 2 or len(avg_non) < 2:
        _log("Group temporal topomaps: insufficient subjects with pain/non trials", logger, "warning")
        return

    info_p, data_p, data_n = align_paired_avg_tfrs(avg_pain, avg_non, logger=logger)
    if info_p is None or data_p is None or data_n is None:
        _log("Group temporal topomaps: could not align paired pain/non data", logger, "warning")
        return

    mean_p = data_p.mean(axis=0)
    mean_n = data_n.mean(axis=0)
    freqs = np.asarray(avg_pain[0].freqs)
    times = np.asarray(avg_pain[0].times)

    tmin, tmax = _clip_time_window(times, plateau_window)
    if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
        logger and logger.warning("Group temporal topomaps: invalid plateau window after clipping")
        return

    edges = np.linspace(tmin, tmax, int(window_count) + 1)
    win_starts = edges[:-1]
    win_ends = edges[1:]

    fmax_available = float(freqs.max())
    bands = _get_bands_for_tfr(max_freq_available=fmax_available, logger=logger, config=config)
    for band_name, (fmin, fmax) in bands.items():
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            continue
        fmask = (freqs >= fmin) & (freqs <= fmax_eff)
        pain_vecs = []
        non_vecs = []
        diff_vecs = []
        for w_start, w_end in zip(win_starts, win_ends):
            tmask = (times >= w_start) & (times < w_end)
            if fmask.sum() == 0 or tmask.sum() == 0:
                pain_vecs.append(None)
                non_vecs.append(None)
                diff_vecs.append(None)
                continue
            v_p = mean_p[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
            v_n = mean_n[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
            pain_vecs.append(v_p)
            non_vecs.append(v_n)
            diff_vecs.append(v_p - v_n)
        vals = [v for v in (pain_vecs + non_vecs) if v is not None]
        if len(vals) == 0:
            continue
        vabs_cond = _robust_sym_vlim(vals)
        diff_vals = [v for v in diff_vecs if v is not None]
        vabs_diff = _robust_sym_vlim(diff_vals) if len(diff_vals) > 0 else vabs_cond

        n_rows = 3
        n_cols = len(win_starts)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.0 * n_cols, 9.0),
            squeeze=False,
            gridspec_kw={"hspace": 0.08, "wspace": 0.2},
        )
        for col, (w_start, w_end) in enumerate(zip(win_starts, win_ends)):
            try:
                axes[0, col].set_title(f"{w_start:.1f}-{w_end:.1f}s", fontsize=10, pad=10)
            except Exception:
                pass
            v_p = pain_vecs[col]
            v_n = non_vecs[col]
            v_d = diff_vecs[col]
            if v_p is not None:
                _plot_topomap_on_ax(axes[0, col], v_p, info_p, vmin=-vabs_cond, vmax=+vabs_cond)
                try:
                    eeg_picks = mne.pick_types(info_p, eeg=True, exclude=[])
                    mu = float(np.nanmean(v_p[eeg_picks])) if len(eeg_picks) > 0 else float(np.nanmean(v_p))
                    pct = (10.0 ** (mu) - 1.0) * 100.0
                    axes[0, col].text(0.5, 1.02, f"%Δ={pct:+.0f}%", transform=axes[0, col].transAxes, ha="center", va="bottom", fontsize=8)
                except Exception:
                    pass
            else:
                axes[0, col].axis('off')
            if v_n is not None:
                _plot_topomap_on_ax(axes[1, col], v_n, info_p, vmin=-vabs_cond, vmax=+vabs_cond)
                try:
                    eeg_picks = mne.pick_types(info_p, eeg=True, exclude=[])
                    mu = float(np.nanmean(v_n[eeg_picks])) if len(eeg_picks) > 0 else float(np.nanmean(v_n))
                    pct = (10.0 ** (mu) - 1.0) * 100.0
                    axes[1, col].text(0.5, 1.02, f"%Δ={pct:+.0f}%", transform=axes[1, col].transAxes, ha="center", va="bottom", fontsize=8)
                except Exception:
                    pass
            else:
                axes[1, col].axis('off')
            if v_d is not None:
                sig_mask = None
                p_min_window = k_win = mass_win = None
                fdr_txt = ""
                if _get_viz_params(config)["diff_annotation_enabled"]:
                    try:
                        tmask_w = (times >= w_start) & (times < w_end)
                        subj_p = data_p[:, :, fmask, :][:, :, :, tmask_w].mean(axis=(2, 3))
                        subj_n = data_n[:, :, fmask, :][:, :, :, tmask_w].mean(axis=(2, 3))
                        sig_mask, p_min_window, k_win, mass_win = _cluster_test_two_sample_arrays(
                            subj_p, subj_n, info_p, alpha=config.get("statistics.sig_alpha", 0.05), paired=True, n_permutations=config.get("statistics.cluster_n_perm", 1024)
                        )
                    except Exception:
                        sig_mask, p_min_window, k_win, mass_win = None, None, None, None
                    if sig_mask is None:
                        try:
                            res = ttest_rel(subj_p, subj_n, axis=0, nan_policy="omit")
                            p_ch = np.asarray(res.pvalue)
                            sig_mask = _fdr_bh_mask(p_ch, alpha=config.get("statistics.sig_alpha", 0.05))
                            rej, q = _fdr_bh_values(p_ch, alpha=config.get("statistics.sig_alpha", 0.05))
                            k_rej = int(np.nansum(rej)) if rej is not None else 0
                            q_min = float(np.nanmin(q)) if q is not None and np.isfinite(q).any() else None
                            fdr_txt = _format_cluster_ann(q_min, k_rej if k_rej > 0 else None, config=config)
                        except Exception:
                            sig_mask = None
                _plot_topomap_on_ax(
                    axes[2, col], v_d, info_p,
                    vmin=-vabs_diff, vmax=+vabs_diff,
                    mask=(sig_mask if _get_viz_params(config)["diff_annotation_enabled"] else None),
                    mask_params=_get_viz_params(config)["sig_mask_params"],
                )
                try:
                    mu = float(np.nanmean(v_d))
                    pct = (10.0 ** (mu) - 1.0) * 100.0
                    cl_txt = (_format_cluster_ann(p_min_window, k_win, mass_win, config=config) or fdr_txt) if _get_viz_params(config)["diff_annotation_enabled"] else ""
                    if cl_txt:
                        axes[2, col].set_title(f"Δ%={pct:+.1f}% | {cl_txt}", fontsize=8, pad=10, y=1.06)
                    else:
                        axes[2, col].set_title(f"Δ%={pct:+.1f}%", fontsize=8, pad=10, y=1.06)
                except Exception:
                    pass
            else:
                axes[2, col].axis('off')
        try:
            axes[0, 0].set_ylabel("Pain", fontsize=10)
            axes[1, 0].set_ylabel("Non-pain", fontsize=10)
            axes[2, 0].set_ylabel("Pain - Non", fontsize=10)
        except Exception:
            pass
        try:
            sm_cond = ScalarMappable(norm=mcolors.Normalize(vmin=-vabs_cond, vmax=+vabs_cond), cmap=_get_viz_params(config)["topo_cmap"])
            sm_cond.set_array([])
            fig.colorbar(sm_cond, ax=axes[0:2, :].ravel().tolist(), fraction=_get_viz_params(config)["colorbar_fraction"], pad=_get_viz_params(config)["colorbar_pad"])
            sm_diff = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-vabs_diff, vcenter=0.0, vmax=vabs_diff), cmap=_get_viz_params(config)["topo_cmap"])
            sm_diff.set_array([])
            fig.colorbar(sm_diff, ax=axes[2, :].ravel().tolist(), fraction=_get_viz_params(config)["colorbar_fraction"], pad=_get_viz_params(config)["colorbar_pad"])
        except Exception:
            pass
        try:
            fig.suptitle(
                f"Group Temporal Topomaps: Pain vs Non-pain — {band_name} (t=[{tmin:.1f},{tmax:.1f}]s; {len(win_starts)} windows)",
                fontsize=12,
            )
        except Exception:
            pass
        band_suffix = band_name.lower()
        fname = f"group_temporal_topomaps_pain_vs_nonpain_{band_suffix}_plateau_{tmin:.0f}-{tmax:.0f}s_{len(win_starts)}windows.png"
        _save_fig(fig, out_dir, fname, config=config, logger=logger)


def group_maxmin_temperature_temporal_topomaps(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_by_subj: List[Optional[pd.DataFrame]],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    window_count: int = 5,
    logger: Optional[logging.Logger] = None,
) -> None:
    if baseline is None:
        baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-5.0, -0.01]))
    if not powers:
        return
    temps = []
    for ev in events_by_subj:
        if ev is None:
            continue
        tcol = next((c for c in config.get("event_columns.temperature", []) if c in ev.columns), None)
        if tcol is None:
            continue
        vals = pd.to_numeric(ev[tcol], errors="coerce").round(1)
        temps.extend(list(vals.dropna().unique()))
    temps = sorted(set(map(float, temps)))
    if len(temps) < 2:
        logger and logger.warning("Group temporal (max/min): fewer than 2 temperature levels; skipping")
        return
    t_min = float(min(temps))
    t_max = float(max(temps))

    avg_min = []
    avg_max = []
    for power, ev in zip(powers, events_by_subj):
        if ev is None:
            continue
        tcol = next((c for c in config.get("event_columns.temperature", []) if c in ev.columns), None)
        if tcol is None:
            continue
        vals = pd.to_numeric(ev[tcol], errors="coerce").round(1)
        mask_min = np.asarray(vals == round(t_min, 1), dtype=bool)
        mask_max = np.asarray(vals == round(t_max, 1), dtype=bool)
        if mask_min.sum() == 0 or mask_max.sum() == 0:
            continue
        a_min = avg_by_mask_to_avg_tfr(power, mask_min)
        a_max = avg_by_mask_to_avg_tfr(power, mask_max)
        if a_min is not None and a_max is not None:
            avg_min.append(a_min)
            avg_max.append(a_max)

    if len(avg_min) < 2 or len(avg_max) < 2:
        logger and logger.warning("Group temporal (max/min): insufficient subjects with both min and max trials")
        return
    info_c, data_min, data_max = align_paired_avg_tfrs(avg_min, avg_max, logger=logger)
    if info_c is None or data_min is None or data_max is None:
        logger and logger.warning("Group temporal (max/min): could not align paired min/max data")
        return

    mean_min = data_min.mean(axis=0)
    mean_max = data_max.mean(axis=0)
    freqs = np.asarray(avg_min[0].freqs if avg_min else avg_max[0].freqs)
    times = np.asarray(avg_min[0].times if avg_min else avg_max[0].times)

    tmin, tmax = _clip_time_window(times, plateau_window)
    if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
        logger and logger.warning("Group temporal (max/min): invalid plateau window after clipping")
        return

    edges = np.linspace(tmin, tmax, int(window_count) + 1)
    win_starts = edges[:-1]
    win_ends = edges[1:]

    fmax_available = float(freqs.max())
    bands = _get_bands_for_tfr(max_freq_available=fmax_available, logger=logger, config=config)
    for band_name, (fmin, fmax) in bands.items():
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            continue
        fmask = (freqs >= fmin) & (freqs <= fmax_eff)
        max_vecs = []
        min_vecs = []
        diff_vecs = []
        for w_start, w_end in zip(win_starts, win_ends):
            tmask = (times >= w_start) & (times < w_end)
            if fmask.sum() == 0 or tmask.sum() == 0:
                max_vecs.append(None)
                min_vecs.append(None)
                diff_vecs.append(None)
                continue
            v_max = mean_max[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
            v_min = mean_min[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
            max_vecs.append(v_max)
            min_vecs.append(v_min)
            diff_vecs.append(v_max - v_min)
        vals = [v for v in (max_vecs + min_vecs) if v is not None]
        if len(vals) == 0:
            continue
        vabs_cond = _robust_sym_vlim(vals)
        diff_vals = [v for v in diff_vecs if v is not None]
        vabs_diff = _robust_sym_vlim(diff_vals) if len(diff_vals) > 0 else vabs_cond

        n_rows = 3
        n_cols = len(win_starts)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.0 * n_cols, 9.0),
            squeeze=False,
            gridspec_kw={"hspace": 0.08, "wspace": 0.2},
        )
        for col, (w_start, w_end) in enumerate(zip(win_starts, win_ends)):
            try:
                axes[0, col].set_title(f"{w_start:.1f}-{w_end:.1f}s", fontsize=10, pad=10)
            except Exception:
                pass
            v_max = max_vecs[col]
            v_min = min_vecs[col]
            v_diff = diff_vecs[col]
            if v_max is not None:
                _plot_topomap_on_ax(axes[0, col], v_max, info_c, vmin=-vabs_cond, vmax=+vabs_cond)
                try:
                    eeg_picks = mne.pick_types(info_c, eeg=True, exclude=[])
                    mu = float(np.nanmean(v_max[eeg_picks])) if len(eeg_picks) > 0 else float(np.nanmean(v_max))
                    pct = (10.0 ** (mu) - 1.0) * 100.0
                    axes[0, col].text(0.5, 1.02, f"%Δ={pct:+.0f}%", transform=axes[0, col].transAxes, ha="center", va="bottom", fontsize=8)
                except Exception:
                    pass
            else:
                axes[0, col].axis('off')
            if v_min is not None:
                _plot_topomap_on_ax(axes[1, col], v_min, info_c, vmin=-vabs_cond, vmax=+vabs_cond)
                try:
                    eeg_picks = mne.pick_types(info_c, eeg=True, exclude=[])
                    mu = float(np.nanmean(v_min[eeg_picks])) if len(eeg_picks) > 0 else float(np.nanmean(v_min))
                    pct = (10.0 ** (mu) - 1.0) * 100.0
                    axes[1, col].text(0.5, 1.02, f"%Δ={pct:+.0f}%", transform=axes[1, col].transAxes, ha="center", va="bottom", fontsize=8)
                except Exception:
                    pass
            else:
                axes[1, col].axis('off')
            if v_diff is not None:
                sig_mask = None
                p_min_window = k_win = mass_win = None
                fdr_txt = ""
                if _get_viz_params(config)["diff_annotation_enabled"]:
                    try:
                        tmask_w = (times >= w_start) & (times < w_end)
                        subj_max = data_max[:, :, fmask, :][:, :, :, tmask_w].mean(axis=(2, 3))
                        subj_min = data_min[:, :, fmask, :][:, :, :, tmask_w].mean(axis=(2, 3))
                        sig_mask, p_min_window, k_win, mass_win = _cluster_test_two_sample_arrays(
                            subj_max, subj_min, info_c, alpha=config.get("statistics.sig_alpha", 0.05), paired=True, n_permutations=config.get("statistics.cluster_n_perm", 1024)
                        )
                    except Exception:
                        sig_mask, p_min_window, k_win, mass_win = None, None, None, None
                    if sig_mask is None:
                        try:
                            res = ttest_rel(subj_max, subj_min, axis=0, nan_policy="omit")
                            p_ch = np.asarray(res.pvalue)
                            sig_mask = _fdr_bh_mask(p_ch, alpha=config.get("statistics.sig_alpha", 0.05))
                            rej, q = _fdr_bh_values(p_ch, alpha=config.get("statistics.sig_alpha", 0.05))
                            k_rej = int(np.nansum(rej)) if rej is not None else 0
                            q_min = float(np.nanmin(q)) if q is not None and np.isfinite(q).any() else None
                            fdr_txt = _format_cluster_ann(q_min, k_rej if k_rej > 0 else None, config=config)
                        except Exception:
                            sig_mask = None
                _plot_topomap_on_ax(
                    axes[2, col], v_diff, info_c,
                    vmin=-vabs_diff, vmax=+vabs_diff,
                    mask=(sig_mask if _get_viz_params(config)["diff_annotation_enabled"] else None),
                    mask_params=_get_viz_params(config)["sig_mask_params"],
                )
                try:
                    mu = float(np.nanmean(v_diff))
                    pct = (10.0 ** (mu) - 1.0) * 100.0
                    cl_txt = (_format_cluster_ann(p_min_window, k_win, mass_win, config=config) or fdr_txt) if _get_viz_params(config)["diff_annotation_enabled"] else ""
                    if cl_txt:
                        axes[2, col].set_title(f"Δ%={pct:+.1f}% | {cl_txt}", fontsize=8, pad=10, y=1.06)
                    else:
                        axes[2, col].set_title(f"Δ%={pct:+.1f}%", fontsize=8, pad=10, y=1.06)
                except Exception:
                    pass
            else:
                axes[2, col].axis('off')
        try:
            axes[0, 0].set_ylabel(f"Max {t_max:.1f}°C", fontsize=10)
            axes[1, 0].set_ylabel(f"Min {t_min:.1f}°C", fontsize=10)
            axes[2, 0].set_ylabel("Max - Min", fontsize=10)
        except Exception:
            pass
        try:
            sm_cond = ScalarMappable(norm=mcolors.Normalize(vmin=-vabs_cond, vmax=+vabs_cond), cmap=_get_viz_params(config)["topo_cmap"])
            sm_cond.set_array([])
            fig.colorbar(sm_cond, ax=axes[0:2, :].ravel().tolist(), fraction=_get_viz_params(config)["colorbar_fraction"], pad=_get_viz_params(config)["colorbar_pad"])
            sm_diff = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-vabs_diff, vcenter=0.0, vmax=vabs_diff), cmap=_get_viz_params(config)["topo_cmap"])
            sm_diff.set_array([])
            fig.colorbar(sm_diff, ax=axes[2, :].ravel().tolist(), fraction=_get_viz_params(config)["colorbar_fraction"], pad=_get_viz_params(config)["colorbar_pad"])
        except Exception:
            pass
        try:
            fig.suptitle(
                f"Group Temporal Topomaps: Max vs Min temperature — {band_name} (t=[{tmin:.1f},{tmax:.1f}]s; {len(win_starts)} windows)",
                fontsize=12,
            )
        except Exception:
            pass
        band_suffix = band_name.lower()
        fname = f"group_temporal_topomaps_max_vs_min_temp_{band_suffix}_plateau_{tmin:.0f}-{tmax:.0f}s_{len(win_starts)}windows.png"
        _save_fig(fig, out_dir, fname, config=config, logger=logger)


def group_topomap_grid_baseline_temps(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_by_subj: List[Optional[pd.DataFrame]],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    logger: Optional[logging.Logger] = None,
) -> None:
    if baseline is None:
        baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-5.0, -0.01]))
    if not powers:
        return
    avg_all = []
    for p in powers:
        avg_all.append(avg_alltrials_to_avg_tfr(p, baseline=baseline, logger=logger))
    info_all, data_all = align_avg_tfrs(avg_all, logger=logger)
    if info_all is None or data_all is None:
        logger and logger.warning("Group temperature grid: could not align all-trials TFRs")
        return
    mean_all = data_all.mean(axis=0)

    temps = []
    for ev in events_by_subj:
        if ev is None:
            continue
        tcol = next((c for c in config.get("event_columns.temperature", []) if c in ev.columns), None)
        if tcol is None:
            continue
        vals = pd.to_numeric(ev[tcol], errors="coerce").round(1)
        temps.extend(list(vals.dropna().unique()))
    temps = sorted(set(map(float, temps)))

    cond_map = {}
    cond_map["All trials"] = (info_all, mean_all, data_all.shape[0], float("nan"))
    for tval in temps:
        avg_list = []
        for p, ev in zip(powers, events_by_subj):
            if ev is None:
                continue
            tcol = next((c for c in config.get("event_columns.temperature", []) if c in ev.columns), None)
            if tcol is None:
                continue
            mask = pd.to_numeric(ev[tcol], errors="coerce").round(1) == round(float(tval), 1)
            if mask.sum() == 0:
                continue
            a = avg_by_mask_to_avg_tfr(p, mask)
            if a is not None:
                avg_list.append(a)
        if not avg_list:
            continue
        info_t, data_t = align_avg_tfrs(avg_list, logger=logger)
        if info_t is None or data_t is None:
            continue
        mean_t = data_t.mean(axis=0)
        cond_map[f"{tval:.1f}°C"] = (info_t, mean_t, data_t.shape[0], float(tval))

    if len(cond_map) <= 1:
        logger and logger.warning("Group temperature grid: no temperature-specific data available")
        return

    labels = list(cond_map.keys())
    freqs = np.asarray(avg_all[0].freqs)
    times = np.asarray(avg_all[0].times)
    fmax_available = float(freqs.max())
    bands = _get_bands_for_tfr(max_freq_available=fmax_available, logger=logger, config=config)
    tmin_req, tmax_req = plateau_window
    tmin = float(max(times.min(), tmin_req))
    tmax = float(min(times.max(), tmax_req))

    n_rows = len(bands)
    n_cols = len(labels)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.6 * n_cols, 3.6 * n_rows),
        squeeze=False,
        gridspec_kw={"wspace": 0.18, "hspace": 0.35},
    )

    for r, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            for c in range(n_cols):
                axes[r, c].axis('off')
            continue
        col_vecs = []
        for label in labels:
            info_c, data_c, _, _ = cond_map[label]
            fmask = (freqs >= fmin) & (freqs <= fmax_eff)
            tmask = (times >= tmin) & (times < tmax)
            if fmask.sum() == 0 or tmask.sum() == 0:
                col_vecs.append(None)
                continue
            vec = data_c[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
            col_vecs.append(vec)
        vals = [v for v in col_vecs if v is not None]
        if len(vals) == 0:
            for c in range(n_cols):
                axes[r, c].axis('off')
            continue
        vabs = _robust_sym_vlim(vals)
        for c, label in enumerate(labels):
            ax = axes[r, c]
            info_c, data_c, nsub, _tval = cond_map[label]
            fmask = (freqs >= fmin) & (freqs <= fmax_eff)
            tmask = (times >= tmin) & (times < tmax)
            if fmask.sum() == 0 or tmask.sum() == 0:
                ax.axis('off')
                continue
            vec = data_c[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
            _plot_topomap_on_ax(ax, vec, info_c, vmin=-vabs, vmax=+vabs)
            title = f"{label} (n={nsub})"
            ax.set_title(title, fontsize=9, pad=2)
            eeg_picks = mne.pick_types(info_c, eeg=True, exclude=[])
            mu = float(np.nanmean(vec[eeg_picks])) if len(eeg_picks) > 0 else float(np.nanmean(vec))
            pct = (10.0 ** (mu) - 1.0) * 100.0
            ax.text(0.5, 1.01, f"%Δ={pct:+.0f}%", transform=ax.transAxes, ha="center", va="top", fontsize=8)

        axes[r, 0].set_ylabel(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)")
        sm = ScalarMappable(norm=mcolors.Normalize(vmin=-vabs, vmax=+vabs), cmap=_get_viz_params(config)["topo_cmap"])
        sm.set_array([])
        fig.colorbar(sm, ax=axes[r, :].ravel().tolist(), fraction=_get_viz_params(config)["colorbar_fraction"], pad=_get_viz_params(config)["colorbar_pad"])
    fig.suptitle(
        f"Group Δ=log10(power/baseline) by temperature (t=[{tmin:.1f}, {tmax:.1f}] s)",
        fontsize=12,
    )
    _save_fig(
        fig,
        out_dir,
        "group_topomap_grid_bands_by_temperature_bl.png",
        config=config,
        logger=logger,
    )


def group_contrast_maxmin_temperature(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_by_subj: List[Optional[pd.DataFrame]],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    logger: Optional[logging.Logger] = None,
) -> None:
    if baseline is None:
        baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-5.0, -0.01]))
    if not powers:
        return
    temps = []
    for ev in events_by_subj:
        if ev is None:
            continue
        tcol = next((c for c in config.get("event_columns.temperature", []) if c in ev.columns), None)
        if tcol is None:
            continue
        vals = pd.to_numeric(ev[tcol], errors="coerce").round(1)
        temps.extend(list(vals.dropna().unique()))
    temps = sorted(set(map(float, temps)))
    if len(temps) < 2:
        logger and logger.info("Group max/min: fewer than 2 temperature levels; skipping")
        return
    t_min, t_max = float(min(temps)), float(max(temps))

    avg_min: List["mne.time_frequency.AverageTFR"] = []
    avg_max: List["mne.time_frequency.AverageTFR"] = []
    for power, ev in zip(powers, events_by_subj):
        if ev is None:
            continue
        tcol = next((c for c in config.get("event_columns.temperature", []) if c in ev.columns), None)
        if tcol is None:
            continue
        vals = pd.to_numeric(ev[tcol], errors="coerce").round(1)
        mask_min = np.asarray(vals == round(t_min, 1), dtype=bool)
        mask_max = np.asarray(vals == round(t_max, 1), dtype=bool)
        if mask_min.sum() == 0 or mask_max.sum() == 0:
            continue
        a_min = avg_by_mask_to_avg_tfr(power, mask_min)
        a_max = avg_by_mask_to_avg_tfr(power, mask_max)
        if a_min is not None and a_max is not None:
            avg_min.append(a_min)
            avg_max.append(a_max)

    info_common, data_min, data_max = align_paired_avg_tfrs(avg_min, avg_max, logger=logger)
    if info_common is None or data_min is None or data_max is None:
        logger and logger.info("Group max/min: could not align paired min/max TFRs; skipping")
        return

    mean_min = data_min.mean(axis=0)
    mean_max = data_max.mean(axis=0)
    freqs = np.asarray(avg_min[0].freqs if avg_min else avg_max[0].freqs)
    times = np.asarray(avg_min[0].times if avg_min else avg_max[0].times)
    fmax_available = float(freqs.max())
    bands = _get_bands_for_tfr(max_freq_available=fmax_available, logger=logger, config=config)
    tmin_req, tmax_req = plateau_window
    tmin = float(max(times.min(), tmin_req))
    tmax = float(min(times.max(), tmax_req))

    n_rows, n_cols = 3, len(bands)
    row_labels = [f"Max {t_max:.1f}°C (n={data_max.shape[0]})", f"Min {t_min:.1f}°C (n={data_min.shape[0]})", "Max - Min"]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows), squeeze=False, gridspec_kw={"wspace": 0.3, "hspace": 0.3})

    for c, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            for r in range(n_rows):
                axes[r, c].axis('off')
            continue
        fmask = (freqs >= fmin) & (freqs <= fmax_eff)
        tmask = (times >= tmin) & (times < tmax)
        if fmask.sum() == 0 or tmask.sum() == 0:
            for r in range(n_rows):
                axes[r, c].axis('off')
            continue
        v_max = mean_max[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
        v_min = mean_min[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
        v_diff = v_max - v_min
        vabs_pn = _robust_sym_vlim([v_max, v_min])
        vabs_diff = _robust_sym_vlim(v_diff)
        ax = axes[0, c]
        _plot_topomap_on_ax(ax, v_max, info_common, vmin=-vabs_pn, vmax=+vabs_pn)
        mu_max = float(np.nanmean(v_max))
        pct_max = (10.0 ** mu_max - 1.0) * 100.0
        ax.text(0.5, 1.02, f"%Δ={pct_max:+.1f}%", transform=ax.transAxes, ha="center", va="top", fontsize=9)

        ax = axes[1, c]
        _plot_topomap_on_ax(ax, v_min, info_common, vmin=-vabs_pn, vmax=+vabs_pn)
        mu_min = float(np.nanmean(v_min))
        pct_min = (10.0 ** mu_min - 1.0) * 100.0
        ax.text(0.5, 1.02, f"%Δ={pct_min:+.1f}%", transform=ax.transAxes, ha="center", va="top", fontsize=9)
        sig_mask = cluster_p_min = cluster_k = cluster_mass = None
        fdr_txt = ""
        if _get_viz_params(config)["diff_annotation_enabled"]:
            subj_max = data_max[:, :, fmask, :][:, :, :, tmask].mean(axis=(2, 3))
            subj_min = data_min[:, :, fmask, :][:, :, :, tmask].mean(axis=(2, 3))
            sig_mask, cluster_p_min, cluster_k, cluster_mass = _cluster_test_two_sample_arrays(
                subj_max, subj_min, info_common, alpha=config.get("statistics.sig_alpha", 0.05), paired=True, n_permutations=config.get("statistics.cluster_n_perm", 1024)
            )
            if sig_mask is None:
                res = ttest_rel(subj_max, subj_min, axis=0, nan_policy="omit")
                p_ch = np.asarray(res.pvalue)
                sig_mask = _fdr_bh_mask(p_ch, alpha=config.get("statistics.sig_alpha", 0.05))
                rej, q = _fdr_bh_values(p_ch, alpha=config.get("statistics.sig_alpha", 0.05))
                k_rej = int(np.nansum(rej)) if rej is not None else 0
                q_min = float(np.nanmin(q)) if q is not None and np.isfinite(q).any() else None
                fdr_txt = _format_cluster_ann(q_min, k_rej if k_rej > 0 else None, config=config)

        ax = axes[2, c]
        _plot_topomap_on_ax(
            ax,
            v_diff,
            info_common,
            mask=(sig_mask if _get_viz_params(config)["diff_annotation_enabled"] else None),
            mask_params=_get_viz_params(config)["sig_mask_params"],
            vmin=-vabs_diff,
            vmax=+vabs_diff,
        )
        mu_d = float(np.nanmean(v_diff))
        pct_d = (10.0 ** mu_d - 1.0) * 100.0
        cl_txt = (_format_cluster_ann(cluster_p_min, cluster_k, cluster_mass, config=config) or fdr_txt) if _get_viz_params(config)["diff_annotation_enabled"] else ""
        label = f"Δ%={pct_d:+.1f}%" + (f" | {cl_txt}" if cl_txt else "")
        ax.text(0.5, 1.02, label, transform=ax.transAxes, ha="center", va="top", fontsize=9)
        axes[0, c].set_title(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=9, pad=4, y=1.04)

        sm_pn = ScalarMappable(norm=mcolors.Normalize(vmin=-vabs_pn, vmax=+vabs_pn), cmap=_get_viz_params(config)["topo_cmap"])
        sm_pn.set_array([])
        fig.colorbar(sm_pn, ax=[axes[0, c], axes[1, c]], fraction=_get_viz_params(config)["colorbar_fraction"], pad=_get_viz_params(config)["colorbar_pad"])
        sm_d = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-vabs_diff, vcenter=0.0, vmax=vabs_diff), cmap=_get_viz_params(config)["topo_cmap"])
        sm_d.set_array([])
        fig.colorbar(sm_d, ax=axes[2, c], fraction=_get_viz_params(config)["colorbar_fraction"], pad=_get_viz_params(config)["colorbar_pad"])

    axes[0, 0].set_ylabel(row_labels[0], fontsize=10)
    axes[1, 0].set_ylabel(row_labels[1], fontsize=10)
    axes[2, 0].set_ylabel(row_labels[2], fontsize=10)
    fig.suptitle(
        f"Group Topomaps: Max vs Min temperature (baseline logratio; t=[{tmin:.1f}, {tmax:.1f}] s)",
        fontsize=12,
    )
    _save_fig(fig, out_dir, "group_topomap_grid_bands_maxmin_temp_diff_baseline_logratio.png", config=config, logger=logger)


def group_rois_all_trials(
    powers: List["mne.time_frequency.EpochsTFR"],
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    logger: Optional[logging.Logger] = None,
    roi_map: Optional[Dict[str, list[str]]] = None,
) -> None:
    if not powers:
        return

    avg_list = []
    for p in powers:
        t = p.copy()
        baseline_used = apply_baseline_and_crop(t, baseline=baseline, mode="logratio", logger=logger)
        avg_list.append(t.average())

    if not avg_list:
        return

    if roi_map is not None:
        rois = list(roi_map.keys())
    else:
        if config is None:
            raise ValueError("Either roi_map or config is required for group_rois_all_trials")
        roi_defs = get_rois(config)
        rois = list(roi_defs.keys())

    for roi in rois:
        per_subj: List["mne.time_frequency.AverageTFR"] = []
        for a in avg_list:
            chs_all = None
            if roi_map is not None:
                chs_all = roi_map.get(roi)
            if chs_all is not None:
                subj_chs = a.info['ch_names']
                canon_subj = {_canonicalize_ch_name(ch).upper(): ch for ch in subj_chs}
                want = {_canonicalize_ch_name(ch).upper() for ch in chs_all}
                chs = [canon_subj[_canonicalize_ch_name(ch).upper()] for ch in subj_chs if _canonicalize_ch_name(ch).upper() in want]
            else:
                roi_defs = get_rois(config)
                pats = roi_defs.get(roi, [])
                chs = _find_roi_channels(a.info, pats)
            if len(chs) == 0:
                continue
            picks = mne.pick_channels(a.info['ch_names'], include=chs, exclude=[])
            if len(picks) == 0:
                continue
            data = np.asarray(a.data)[picks, :, :].mean(axis=0, keepdims=True)
            ra = a.copy()
            ra.data = data
            ra.info = mne.create_info([f"ROI:{roi}"], sfreq=a.info['sfreq'], ch_types='eeg')
            per_subj.append(ra)

        if len(per_subj) < 1 and roi_map is not None:
            for a in avg_list:
                roi_defs = get_rois(config)
                pats = roi_defs.get(roi, [])
                chs_rx = _find_roi_channels(a.info, pats)
                if chs_rx:
                    picks = mne.pick_channels(a.info['ch_names'], include=chs_rx, exclude=[])
                    if len(picks) == 0:
                        continue
                    data = np.asarray(a.data)[picks, :, :].mean(axis=0, keepdims=True)
                    ra = a.copy()
                    ra.data = data
                    ra.info = mne.create_info([f"ROI:{roi}"], sfreq=a.info['sfreq'], ch_types='eeg')
                    per_subj.append(ra)

        if len(per_subj) < 1:
            logger and logger.info(f"Group ROI all-trials: no subjects contributed to ROI '{roi}'")
            continue

        info_c, data_c = align_avg_tfrs(per_subj, logger=logger)
        if info_c is None or data_c is None:
            continue

        mean_roi = data_c.mean(axis=0)
        grp = per_subj[0].copy()
        grp.data = mean_roi
        grp.info = info_c
        grp.nave = int(data_c.shape[0])
        grp.comment = f"Group ROI:{roi}"
        ch = grp.info['ch_names'][0]
        fig = _fig(grp.plot(picks=ch, show=False))
        fig.suptitle(f"Group ROI: {roi} — all trials (baseline logratio, n={data_c.shape[0]})", fontsize=12)
        _save_fig(fig, out_dir, f"group_tfr_ROI-{_sanitize(roi)}_all_trials_baseline_logratio.png", config=config, logger=logger, baseline_used=baseline)


def group_contrast_pain_nonpain_rois(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_by_subj: List[Optional[pd.DataFrame]],
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    logger: Optional[logging.Logger] = None,
    roi_map: Optional[Dict[str, list[str]]] = None,
) -> None:
    if not powers:
        return

    if roi_map is not None:
        rois = list(roi_map.keys())
    else:
        if config is None:
            raise ValueError("Either roi_map or config is required for group_contrast_pain_nonpain_rois")
        roi_defs = get_rois(config)
        rois = list(roi_defs.keys())
    for roi in rois:
        roi_p_list: List["mne.time_frequency.AverageTFR"] = []
        roi_n_list: List["mne.time_frequency.AverageTFR"] = []

        for power, ev in zip(powers, events_by_subj):
            if ev is None:
                continue
            pain_col = next((c for c in config.get("event_columns.pain_binary", []) if c in ev.columns), None)
            if pain_col is None:
                continue
            vals = pd.to_numeric(ev[pain_col], errors="coerce").fillna(0).astype(int).values
            pain_mask = np.asarray(vals == 1, dtype=bool)
            non_mask = np.asarray(vals == 0, dtype=bool)
            if pain_mask.sum() == 0 or non_mask.sum() == 0:
                continue
            a_p = avg_by_mask_to_avg_tfr(power, pain_mask)
            a_n = avg_by_mask_to_avg_tfr(power, non_mask)
            if a_p is None or a_n is None:
                continue
            r_p = None
            r_n = None
            if roi_map is not None:
                chs_all = roi_map.get(roi)
                if chs_all is not None:
                    subj_chs = a_p.info['ch_names']
                    canon_subj = {_canonicalize_ch_name(ch).upper(): ch for ch in subj_chs}
                    want = {_canonicalize_ch_name(ch).upper() for ch in chs_all}
                    chs = [canon_subj[_canonicalize_ch_name(ch).upper()] for ch in subj_chs if _canonicalize_ch_name(ch).upper() in want]
                    if len(chs) > 0:
                        picks = mne.pick_channels(subj_chs, include=chs, exclude=[])
                        r_p = a_p.copy(); r_p.data = np.asarray(a_p.data)[picks, :, :].mean(axis=0, keepdims=True); r_p.info = mne.create_info([f"ROI:{roi}"], sfreq=a_p.info['sfreq'], ch_types='eeg')
                        picks_n = mne.pick_channels(a_n.info['ch_names'], include=chs, exclude=[])
                        r_n = a_n.copy(); r_n.data = np.asarray(a_n.data)[picks_n, :, :].mean(axis=0, keepdims=True); r_n.info = mne.create_info([f"ROI:{roi}"], sfreq=a_n.info['sfreq'], ch_types='eeg')
            if r_p is None or r_n is None:
                roi_defs = get_rois(config)
                pats = roi_defs.get(roi, [])
                chs = _find_roi_channels(a_p.info, pats)
                if len(chs) > 0:
                    picks = mne.pick_channels(a_p.info['ch_names'], include=chs, exclude=[])
                    r_p = a_p.copy(); r_p.data = np.asarray(a_p.data)[picks, :, :].mean(axis=0, keepdims=True); r_p.info = mne.create_info([f"ROI:{roi}"], sfreq=a_p.info['sfreq'], ch_types='eeg')
                    picks_n = mne.pick_channels(a_n.info['ch_names'], include=chs, exclude=[])
                    r_n = a_n.copy(); r_n.data = np.asarray(a_n.data)[picks_n, :, :].mean(axis=0, keepdims=True); r_n.info = mne.create_info([f"ROI:{roi}"], sfreq=a_n.info['sfreq'], ch_types='eeg')
            if r_p is not None and r_n is not None:
                roi_p_list.append(r_p)
                roi_n_list.append(r_n)

        if (len(roi_p_list) < 1 or len(roi_n_list) < 1) and roi_map is not None:
            roi_p_list = []
            roi_n_list = []
            for power, ev in zip(powers, events_by_subj):
                if ev is None:
                    continue
                pain_col = next((c for c in config.get("event_columns.pain_binary", []) if c in ev.columns), None)
                if pain_col is None:
                    continue
                vals = pd.to_numeric(ev[pain_col], errors="coerce").fillna(0).astype(int).values
                pain_mask = np.asarray(vals == 1, dtype=bool)
                non_mask = np.asarray(vals == 0, dtype=bool)
                if pain_mask.sum() == 0 or non_mask.sum() == 0:
                    continue
                a_p = avg_by_mask_to_avg_tfr(power, pain_mask)
                a_n = avg_by_mask_to_avg_tfr(power, non_mask)
                if a_p is None or a_n is None:
                    continue
                roi_defs = get_rois(config)
                pats = roi_defs.get(roi, [])
                chs = _find_roi_channels(a_p.info, pats)
                if len(chs) > 0:
                    picks = mne.pick_channels(a_p.info['ch_names'], include=chs, exclude=[])
                    r_p = a_p.copy(); r_p.data = np.asarray(a_p.data)[picks, :, :].mean(axis=0, keepdims=True); r_p.info = mne.create_info([f"ROI:{roi}"], sfreq=a_p.info['sfreq'], ch_types='eeg')
                    picks_n = mne.pick_channels(a_n.info['ch_names'], include=chs, exclude=[])
                    r_n = a_n.copy(); r_n.data = np.asarray(a_n.data)[picks_n, :, :].mean(axis=0, keepdims=True); r_n.info = mne.create_info([f"ROI:{roi}"], sfreq=a_n.info['sfreq'], ch_types='eeg')
                    roi_p_list.append(r_p)
                    roi_n_list.append(r_n)

        if len(roi_p_list) < 1 or len(roi_n_list) < 1:
            logger and logger.info(f"Group ROI pain/non: no subjects contributed to ROI '{roi}'")
            continue

        info_p, data_p = align_avg_tfrs(roi_p_list, logger=logger)
        info_n, data_n = align_avg_tfrs(roi_n_list, logger=logger)
        if info_p is None or info_n is None or data_p is None or data_n is None:
            continue

        mean_p = data_p.mean(axis=0)
        mean_n = data_n.mean(axis=0)

        grp_p = roi_p_list[0].copy()
        grp_p.data = mean_p
        grp_p.info = info_p
        grp_p.nave = int(data_p.shape[0])
        grp_p.comment = f"Group ROI:{roi} Pain"

        grp_n = roi_n_list[0].copy()
        grp_n.data = mean_n
        grp_n.info = info_n
        grp_n.nave = int(data_n.shape[0])
        grp_n.comment = f"Group ROI:{roi} Non"

        diff = mean_p - mean_n
        grp_d = roi_p_list[0].copy()
        grp_d.data = diff
        grp_d.info = info_p
        grp_d.nave = int(min(data_p.shape[0], data_n.shape[0]))
        grp_d.comment = f"Group ROI:{roi} Diff"
        ch = grp_p.info['ch_names'][0]

        fig = _fig(grp_p.plot(picks=ch, show=False))
        fig.suptitle(f"Group ROI: {roi} — Pain (baseline logratio, n={data_p.shape[0]})", fontsize=12)
        _save_fig(fig, out_dir, f"group_tfr_ROI-{_sanitize(roi)}_pain_baseline_logratio.png", config=config, logger=logger, baseline_used=baseline)

        fig = _fig(grp_n.plot(picks=ch, show=False))
        fig.suptitle(f"Group ROI: {roi} — Non-pain (baseline logratio, n={data_n.shape[0]})", fontsize=12)
        _save_fig(fig, out_dir, f"group_tfr_ROI-{_sanitize(roi)}_nonpain_baseline_logratio.png", config=config, logger=logger, baseline_used=baseline)

        fig = _fig(grp_d.plot(picks=ch, show=False))
        n_diff = min(data_p.shape[0], data_n.shape[0])
        fig.suptitle(f"Group ROI: {roi} — Pain minus Non (baseline logratio, n={n_diff})", fontsize=12)
        _save_fig(fig, out_dir, f"group_tfr_ROI-{_sanitize(roi)}_pain_minus_non_baseline_logratio.png", config=config, logger=logger, baseline_used=baseline)


def group_contrast_pain_nonpain_scalpmean(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_by_subj: List[Optional[pd.DataFrame]],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    if baseline is None:
        baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-5.0, -0.01]))
    if not powers:
        return

    pain_cols = config.get("event_columns.pain_binary", [])
    avg_pain: List["mne.time_frequency.AverageTFR"] = []
    avg_non: List["mne.time_frequency.AverageTFR"] = []

    for power, ev in zip(powers, events_by_subj):
        if ev is None:
            continue
        pain_col = next((c for c in pain_cols if c in ev.columns), None)
        if pain_col is None:
            continue
        vals = pd.to_numeric(ev[pain_col], errors="coerce").fillna(0).astype(int).values
        pain_mask = np.asarray(vals == 1, dtype=bool)
        non_mask = np.asarray(vals == 0, dtype=bool)
        if pain_mask.sum() == 0 or non_mask.sum() == 0:
            continue
        a_p = avg_by_mask_to_avg_tfr(power, pain_mask, baseline=baseline, logger=logger)
        a_n = avg_by_mask_to_avg_tfr(power, non_mask, baseline=baseline, logger=logger)
        if a_p is None or a_n is None:
            continue
        avg_pain.append(a_p)
        avg_non.append(a_n)

    if len(avg_pain) < 1 or len(avg_non) < 1:
        return

    info_p, data_p, data_n = align_paired_avg_tfrs(avg_pain, avg_non, logger=logger)
    if info_p is None or data_p is None or data_n is None:
        return

    n_subj = int(min(data_p.shape[0], data_n.shape[0]))
    mean_p = data_p.mean(axis=0)
    mean_n = data_n.mean(axis=0)

    data_p_sm = np.asarray(mean_p).mean(axis=0, keepdims=True)
    data_n_sm = np.asarray(mean_n).mean(axis=0, keepdims=True)
    diff_sm = data_p_sm - data_n_sm

    tmpl = avg_pain[0].copy()
    sfreq = tmpl.info['sfreq']

    def create_group_tfr(data, comment):
        tfr = tmpl.copy()
        tfr.data = data
        tfr.info = mne.create_info(["AllEEG"], sfreq=sfreq, ch_types='eeg')
        tfr.nave = n_subj
        tfr.comment = comment
        return tfr

    grp_p = create_group_tfr(data_p_sm, "Group AllEEG Pain")
    grp_n = create_group_tfr(data_n_sm, "Group AllEEG Non")
    grp_d = create_group_tfr(diff_sm, "Group AllEEG Diff")

    for grp, title, filename in [
        (grp_p, f"Group TFR: All EEG — Pain (baseline logratio, n={n_subj})", "group_tfr_AllEEG_pain_baseline_logratio.png"),
        (grp_n, f"Group TFR: All EEG — Non-pain (baseline logratio, n={n_subj})", "group_tfr_AllEEG_nonpain_baseline_logratio.png"),
        (grp_d, "Group TFR: All EEG — Pain minus Non (baseline logratio)", "group_tfr_AllEEG_pain_minus_non_baseline_logratio.png")
    ]:
        fig = grp.plot(picks="AllEEG", show=False)
        fig = _fig(fig)
        fig.suptitle(title, fontsize=12)
        _save_fig(fig, out_dir, filename, config=config, logger=logger)


def plot_topomap_grid_baseline_temps(
    tfr: "mne.time_frequency.EpochsTFR | mne.time_frequency.AverageTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    logger: Optional[logging.Logger] = None,
) -> None:
    if baseline is None:
        baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-5.0, -0.01]))
    if events_df is None:
        _log("Temperature grid: events_df is None; skipping.")
        return
    temp_col = next((c for c in config.get("event_columns.temperature", []) if c in events_df.columns), None)
    if temp_col is None:
        _log("Temperature grid: no temperature column found; skipping.")
        return

    tfr_corr = tfr.copy()
    baseline_used = apply_baseline_and_crop(tfr_corr, baseline=baseline, mode="logratio", logger=logger)
    tfr_avg_all_corr = tfr_corr.average() if isinstance(tfr_corr, mne.time_frequency.EpochsTFR) else tfr_corr

    temps = (
        pd.to_numeric(events_df[temp_col], errors="coerce")
        .round(1)
        .dropna()
        .unique()
    )
    temps = sorted(map(float, temps))
    if len(temps) == 0:
        _log("Temperature grid: no temperature levels; skipping.")
        return

    times_corr = np.asarray(tfr_avg_all_corr.times)
    tmin_req, tmax_req = plateau_window
    tmin = float(max(times_corr.min(), tmin_req))
    tmax = float(min(times_corr.max(), tmax_req))

    fmax_available = float(np.max(tfr_avg_all_corr.freqs))
    bands = _get_bands_for_tfr(max_freq_available=fmax_available, logger=logger, config=config)

    cond_tfrs: list[tuple[str, "mne.time_frequency.AverageTFR", int, float]] = []
    n_all = len(tfr_corr) if isinstance(tfr_corr, mne.time_frequency.EpochsTFR) else 1
    cond_tfrs.append(("All trials", tfr_avg_all_corr, n_all, np.nan))

    if isinstance(tfr_corr, mne.time_frequency.EpochsTFR):
        for tval in temps:
            mask = pd.to_numeric(events_df[temp_col], errors="coerce").round(1) == round(float(tval), 1)
            mask = np.asarray(mask, dtype=bool)
            if mask.sum() == 0:
                continue
            tfr_temp = tfr_corr.copy()[mask].average()
            cond_tfrs.append((f"{tval:.1f}°C", tfr_temp, int(mask.sum()), float(tval)))
    else:
        _log("Temperature grid: input is AverageTFR; cannot split by temperature; showing only All trials.")

    n_cols, n_rows = len(cond_tfrs), len(bands)
    width_ratios = [1.0] * n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.8 * n_cols, 3.8 * n_rows), squeeze=False,
        gridspec_kw={"wspace": 0.30, "hspace": 0.55, "width_ratios": width_ratios},
    )

    for r, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            for c in range(n_cols):
                axes[r, c].axis("off")
            continue

        diff_datas: list[Optional[np.ndarray]] = []
        for _, tfr_cond, _, _ in cond_tfrs:
            d = _average_tfr_band(tfr_cond, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
            diff_datas.append(d)

        vals = [v for v in diff_datas if v is not None and np.isfinite(v).any()]
        if len(vals) == 0:
            for c in range(n_cols):
                axes[r, c].axis("off")
            continue

        diff_abs = _robust_sym_vlim(vals)
        if not np.isfinite(diff_abs) or diff_abs == 0:
            diff_abs = 1e-6

        for idx, (label, tfr_cond, n_cond, _tval) in enumerate(cond_tfrs, start=0):
            ax = axes[r, idx]
            data = diff_datas[idx]
            if data is None:
                ax.axis("off")
                continue

            _plot_topomap_on_ax(ax, data, tfr_cond.info, vmin=-diff_abs, vmax=+diff_abs)
            mu = float(np.nanmean(data))
            pct = (10.0 ** mu - 1.0) * 100.0
            ax.text(0.5, 1.02, f"%Δ={pct:+.0f}%", transform=ax.transAxes, ha="center", va="top", fontsize=9)
            if r == 0:
                ax.set_title(f"{label} (n={n_cond})", fontsize=9, pad=4, y=1.04)

        axes[r, 0].set_ylabel(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=10)

        sm_diff = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-diff_abs, vcenter=0.0, vmax=diff_abs), cmap=_get_viz_params(config)["topo_cmap"])
        sm_diff.set_array([])
        cbar_d = fig.colorbar(sm_diff, ax=axes[r, :].ravel().tolist(), fraction=0.045, pad=0.06, shrink=0.9)
        cbar_d.set_label("log10(power/baseline)")

    fig.suptitle(
        f"Topomaps by temperature: Δ=log10(power/baseline) over plateau t=[{tmin:.1f}, {tmax:.1f}] s",
        fontsize=12,
    )
    _save_fig(fig, out_dir, "topomap_grid_bands_alltrials_plus_temperatures_baseline_logratio.png", config=config)


def plot_pain_nonpain_temporal_topomaps(
    tfr: "mne.time_frequency.EpochsTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    window_count: int = 5,
    logger: Optional[logging.Logger] = None,
) -> None:
    if not isinstance(tfr, mne.time_frequency.EpochsTFR):
        _log("Temporal topomaps require EpochsTFR (trial-level data). Skipping.", logger, "warning")
        return

    if baseline is None:
        baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-5.0, -0.01]))

    pain_col = next((c for c in config.get("event_columns.pain_binary", []) if events_df is not None and c in events_df.columns), None)
    if pain_col is None:
        _log(f"Events with pain binary column required for temporal topomaps; skipping.", logger, "warning")
        return

    n_epochs = tfr.data.shape[0]
    n_meta = len(events_df)
    n = min(n_epochs, n_meta)

    if getattr(tfr, "metadata", None) is not None and pain_col in tfr.metadata.columns:
        pain_vec = pd.to_numeric(tfr.metadata.iloc[:n][pain_col], errors="coerce").fillna(0).astype(int).values
    else:
        pain_vec = pd.to_numeric(events_df.iloc[:n][pain_col], errors="coerce").fillna(0).astype(int).values

    pain_mask = np.asarray(pain_vec == 1, dtype=bool)
    non_mask = np.asarray(pain_vec == 0, dtype=bool)

    if pain_mask.sum() == 0 or non_mask.sum() == 0:
        _log("One of the groups has zero trials; skipping temporal topomaps.", logger, "warning")
        return

    _log(f"Temporal topomaps: pain={int(pain_mask.sum())}, non-pain={int(non_mask.sum())} trials.", logger)

    tfr_sub = tfr.copy()[:n]
    try:
        ensure_aligned_lengths(
            tfr_sub, pain_mask, non_mask,
            context=f"Pain contrast",
            strict=config.get("analysis.strict_mode", True),
            logger=logger
        )
    except ValueError as e:
        _log(f"{e}. Skipping contrast.", logger, "error")
        return
    if len(pain_mask) != len(tfr_sub):
        pain_mask = pain_mask[:len(tfr_sub)]
        non_mask = non_mask[:len(tfr_sub)]

    baseline_used = apply_baseline_and_crop(tfr_sub, baseline=baseline, mode="logratio", logger=logger)

    tfr_pain = tfr_sub[pain_mask].average()
    tfr_non = tfr_sub[non_mask].average()

    times = np.asarray(tfr_pain.times)
    tmin_req, tmax_req = plateau_window
    tmin_clip = float(max(times.min(), tmin_req))
    tmax_clip = float(min(times.max(), tmax_req))
    if not np.isfinite(tmin_clip) or not np.isfinite(tmax_clip) or (tmax_clip <= tmin_clip):
        _log(f"No valid plateau interval within data range; skipping temporal topomaps (requested [{tmin_req}, {tmax_req}] s, available [{times.min():.2f}, {times.max():.2f}] s).", logger, "warning")
        return

    n_windows = window_count
    edges = np.linspace(tmin_clip, tmax_clip, n_windows + 1)
    window_starts = edges[:-1]
    window_ends = edges[1:]
    window_size_eff = float((tmax_clip - tmin_clip) / n_windows)
    _log(f"Creating temporal topomaps over plateau [{tmin_clip:.2f}, {tmax_clip:.2f}] s using {n_windows} windows (~{window_size_eff:.2f}s each).", logger)

    fmax_available = float(np.max(tfr_pain.freqs))
    bands = _get_bands_for_tfr(max_freq_available=fmax_available, logger=logger, config=config)

    for band_name, (fmin, fmax) in bands.items():
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            continue

        freq_label = f"{band_name} ({fmin:.0f}-{fmax_eff:.0f}Hz)"

        pain_data_windows = []
        non_data_windows = []
        diff_data_windows = []

        for tmin_win, tmax_win in zip(window_starts, window_ends):
            pain_data = _average_tfr_band(tfr_pain, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
            non_data = _average_tfr_band(tfr_non, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)

            if pain_data is not None and non_data is not None:
                diff_data = pain_data - non_data
                pain_data_windows.append(pain_data)
                non_data_windows.append(non_data)
                diff_data_windows.append(diff_data)
            else:
                pain_data_windows.append(None)
                non_data_windows.append(None)
                diff_data_windows.append(None)

        all_data = [d for d in pain_data_windows + non_data_windows if d is not None]
        diff_data_valid = [d for d in diff_data_windows if d is not None]

        if len(all_data) == 0:
            _log(f"No valid data found for {band_name} temporal topomaps; skipping this band.", logger, "warning")
            continue

        vabs_cond = _robust_sym_vlim(all_data)
        vabs_diff = _robust_sym_vlim(diff_data_valid) if len(diff_data_valid) > 0 else vabs_cond

        fig, axes = plt.subplots(
            3, n_windows, figsize=(3.0 * n_windows, 9.0), squeeze=False,
            gridspec_kw={"hspace": 0.25, "wspace": 0.3}
        )

        row_labels = [f"Pain (n={int(pain_mask.sum())})", f"Non-pain (n={int(non_mask.sum())})", "Pain - Non"]

        for col, (tmin_win, tmax_win) in enumerate(zip(window_starts, window_ends)):
            axes[0, col].set_title(f"{tmin_win:.1f}-{tmax_win:.1f}s", fontsize=10, pad=25)

            pain_data = pain_data_windows[col]
            if pain_data is not None:
                _plot_topomap_on_ax(
                    axes[0, col], pain_data, tfr_pain.info,
                    vmin=-vabs_cond, vmax=+vabs_cond
                )
                mu = float(np.nanmean(pain_data))
                pct = (10.0 ** mu - 1.0) * 100.0
                axes[0, col].text(0.5, 1.08, f"%Δ={pct:+.0f}%",
                                transform=axes[0, col].transAxes, ha="center", va="bottom", fontsize=8)
            else:
                axes[0, col].axis('off')

            non_data = non_data_windows[col]
            if non_data is not None:
                _plot_topomap_on_ax(
                    axes[1, col], non_data, tfr_non.info,
                    vmin=-vabs_cond, vmax=+vabs_cond
                )
                mu = float(np.nanmean(non_data))
                pct = (10.0 ** mu - 1.0) * 100.0
                axes[1, col].text(0.5, 1.08, f"%Δ={pct:+.0f}%",
                                transform=axes[1, col].transAxes, ha="center", va="bottom", fontsize=8)
            else:
                axes[1, col].axis('off')

            diff_data = diff_data_windows[col]
            if diff_data is not None:
                sig_mask = cluster_p_min = cluster_k = cluster_mass = None
                if _get_viz_params(config)["diff_annotation_enabled"]:
                    sig_mask, cluster_p_min, cluster_k, cluster_mass = _cluster_test_epochs(
                        tfr_sub, pain_mask, non_mask,
                        fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win,
                        paired=False,
                    )

                _plot_topomap_on_ax(
                    axes[2, col], diff_data, tfr_pain.info,
                    vmin=-vabs_diff, vmax=+vabs_diff,
                    mask=(sig_mask if _get_viz_params(config)["diff_annotation_enabled"] else None), mask_params=_get_viz_params(config)["sig_mask_params"]
                )
                mu = float(np.nanmean(diff_data))
                pct = (10.0 ** mu - 1.0) * 100.0
                cl_txt = _format_cluster_ann(cluster_p_min, cluster_k, cluster_mass, config=config) if _get_viz_params(config)["diff_annotation_enabled"] else ""
                label = f"Δ%={pct:+.1f}%" + (f" | {cl_txt}" if cl_txt else "")
                axes[2, col].text(0.5, 1.08, label, transform=axes[2, col].transAxes, ha="center", va="bottom", fontsize=8)
            else:
                axes[2, col].axis('off')

        for row, label in enumerate(row_labels):
            axes[row, 0].set_ylabel(label, fontsize=11, labelpad=10)

        sm_cond = ScalarMappable(
            norm=mcolors.TwoSlopeNorm(vmin=-vabs_cond, vcenter=0.0, vmax=vabs_cond),
            cmap=_get_viz_params(config)["topo_cmap"]
        )
        sm_cond.set_array([])
        cbar_cond = fig.colorbar(
            sm_cond, ax=axes[:2, :].ravel().tolist(),
            fraction=0.03, pad=0.02, shrink=0.8, aspect=20
        )
        cbar_cond.set_label("log10(power/baseline)", fontsize=10)

        sm_diff = ScalarMappable(
            norm=mcolors.TwoSlopeNorm(vmin=-vabs_diff, vcenter=0.0, vmax=vabs_diff),
            cmap=_get_viz_params(config)["topo_cmap"]
        )
        sm_diff.set_array([])
        cbar_diff = fig.colorbar(
            sm_diff, ax=axes[2, :].ravel().tolist(),
            fraction=0.03, pad=0.02, shrink=0.8, aspect=20
        )
        cbar_diff.set_label("log10(power/baseline) difference", fontsize=10)

        fig.suptitle(
            f"Temporal topomaps: Pain vs Non-pain - {freq_label} (plateau {tmin_clip:.1f}–{tmax_clip:.1f}s; 5 windows)\n"
            f"log10(power/baseline), vlim ±{vabs_cond:.2f} (conditions), ±{vabs_diff:.2f} (difference)",
            fontsize=12, y=1.08
        )

        band_suffix = band_name.lower()
        filename = f"temporal_topomaps_pain_vs_nonpain_{band_suffix}_plateau_{tmin_clip:.0f}-{tmax_clip:.0f}s_5windows.png"
        _save_fig(fig, out_dir, filename, config=config, logger=logger)


def contrast_pain_nonpain_rois(
    roi_tfrs: Dict[str, mne.time_frequency.EpochsTFR],
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    logger: Optional[logging.Logger] = None,
) -> None:
    pain_col = next((c for c in config.get("event_columns.pain_binary", []) if events_df is not None and c in events_df.columns), None)
    if pain_col is None:
        _log(f"Events with pain binary column required for ROI contrasts; skipping.", logger=logger)
        return

    rois_dir = out_dir / "rois"
    for roi, tfr in roi_tfrs.items():
        try:
            n_epochs = tfr.data.shape[0]
            n_meta = len(events_df)
            n = min(n_epochs, n_meta)
            if n_epochs != n_meta:
                _log(f"ROI {roi}: trimming to {n} epochs to match events.")

            if getattr(tfr, "metadata", None) is not None and pain_col in tfr.metadata.columns:
                pain_vec = pd.to_numeric(tfr.metadata.iloc[:n][pain_col], errors="coerce").fillna(0).astype(int).values
            else:
                pain_vec = pd.to_numeric(events_df.iloc[:n][pain_col], errors="coerce").fillna(0).astype(int).values
            pain_mask = np.asarray(pain_vec == 1, dtype=bool)
            non_mask = np.asarray(pain_vec == 0, dtype=bool)
            if pain_mask.sum() == 0 or non_mask.sum() == 0:
                _log(f"ROI {roi}: one group has zero trials; skipping.")
                continue

            tfr_sub = tfr.copy()[:n]
            strict_mode = config.get("analysis.strict_mode", True) if config else True
            try:
                ensure_aligned_lengths(
                    tfr_sub, pain_mask, non_mask,
                    context=f"ROI {roi}",
                    strict=strict_mode,
                    logger=logger
                )
                if len(pain_mask) != len(tfr_sub):
                    pain_mask = pain_mask[:len(tfr_sub)]
                    non_mask = non_mask[:len(tfr_sub)]
            except ValueError as e:
                _log(f"{e}. Skipping ROI {roi}.", logger, "error")
                continue
            baseline_used = apply_baseline_and_crop(tfr_sub, baseline=baseline, mode="logratio", logger=logger)
            tfr_pain = tfr_sub[pain_mask].average()
            tfr_non = tfr_sub[non_mask].average()

            ch = tfr_pain.info['ch_names'][0]
            roi_tag = _sanitize(roi)
            roi_dir = rois_dir / roi_tag

            fig = _fig(tfr_pain.plot(picks=ch, show=False))
            fig.suptitle(f"ROI: {roi} — Painful (baseline logratio)", fontsize=12)
            _save_fig(fig, roi_dir, "tfr_painful_bl.png", config=config, logger=logger, baseline_used=baseline_used)

            fig = _fig(tfr_non.plot(picks=ch, show=False))
            fig.suptitle(f"ROI: {roi} — Non-pain (baseline logratio)", fontsize=12)
            _save_fig(fig, roi_dir, "tfr_nonpain_bl.png", config=config, logger=logger, baseline_used=baseline_used)

            tfr_diff = tfr_pain.copy()
            tfr_diff.data = tfr_pain.data - tfr_non.data
            fig = _fig(tfr_diff.plot(picks=ch, show=False))
            fig.suptitle(f"ROI: {roi} — Pain minus Non-pain (baseline logratio)", fontsize=12)
            _save_fig(fig, roi_dir, "tfr_pain_minus_nonpain_bl.png", config=config, logger=logger, baseline_used=baseline_used)

            fmax_available = float(np.max(tfr_pain.freqs))
            bands = _get_bands_for_tfr(max_freq_available=fmax_available, logger=logger, config=config)
            for band, (fmin, fmax) in bands.items():
                fmax_eff = min(fmax, fmax_available)
                if fmin >= fmax_eff:
                    continue
                band_dir = roi_dir / band
                fig_b = _fig(tfr_pain.plot(picks=ch, fmin=fmin, fmax=fmax_eff, show=False))
                fig_b.suptitle(f"ROI: {roi} — {band} Painful (baseline logratio)", fontsize=12)
                _save_fig(fig_b, band_dir, f"tfr_{band}_painful_bl.png", config=config, logger=logger, baseline_used=baseline_used)

                fig_b = _fig(tfr_non.plot(picks=ch, fmin=fmin, fmax=fmax_eff, show=False))
                fig_b.suptitle(f"ROI: {roi} — {band} Non-pain (baseline logratio)", fontsize=12)
                _save_fig(fig_b, band_dir, f"tfr_{band}_nonpain_bl.png", config=config, logger=logger, baseline_used=baseline_used)

                fig_b = _fig(tfr_diff.plot(picks=ch, fmin=fmin, fmax=fmax_eff, show=False))
                fig_b.suptitle(f"ROI: {roi} — {band} Pain minus Non-pain (baseline logratio)", fontsize=12)
                _save_fig(fig_b, band_dir, f"tfr_{band}_pain_minus_nonpain_bl.png", config=config, logger=logger, baseline_used=baseline_used)
        except (FileNotFoundError, ValueError, RuntimeError, KeyError, IndexError) as exc:
            _log(f"ROI {roi}: error while computing ROI contrasts ({exc})", logger, "error")
            continue


###################################################################
# Group TF correlation (moved for completeness)
###################################################################

def group_tf_correlation(subjects=None, roi=None, method="auto", alpha=None, min_subjects=3, config=None, logger=None):
    if alpha is None:
        alpha = config.get("statistics.sig_alpha", 0.05) if config else 0.05
    roi_raw = roi.lower() if isinstance(roi, str) else None
    roi_suffix = f"_{re.sub(r'[^A-Za-z0-9._-]+', '_', roi_raw)}" if roi_raw else ""
    allowed = set(subjects) if subjects else None

    def _log_local(msg, level="info"):
        _log(msg, logger, level)

    def _discover_subjects_with_tf(roi_suffix, method_suffix, config, allowed_subjects=None):
        subs = []
        for sd in sorted(config.deriv_root.glob("sub-*")):
            if not sd.is_dir():
                continue
            sub = sd.name[4:]
            if allowed_subjects is not None and sub not in allowed_subjects:
                continue
            cand = sd / "eeg" / "stats" / f"tf_corr_stats{roi_suffix}{method_suffix}.tsv"
            if cand.exists():
                subs.append(sub)
        return subs

    def _load_subject_tf(sub, roi_suffix, method_suffix, config):
        p = config.deriv_root / f"sub-{sub}" / "eeg" / "stats" / f"tf_corr_stats{roi_suffix}{method_suffix}.tsv"
        return pd.read_csv(p, sep="\t") if p.exists() else None

    if method == "auto":
        for ms in ("_spearman", "_pearson"):
            subs_ms = _discover_subjects_with_tf(roi_suffix, ms, config, allowed)
            if len(subs_ms) >= min_subjects:
                method_suffix = ms
                subjects_to_use = subs_ms
                break
        else:
            _log_local(f"Group TF correlation skipped for ROI '{roi or 'all'}' — insufficient subject heatmaps.", level="warning")
            return None
    else:
        method_suffix = f"_{method.lower()}"
        subjects_to_use = subjects or _discover_subjects_with_tf(roi_suffix, method_suffix, config, allowed)

    if not subjects_to_use:
        _log_local(f"Group TF correlation skipped for ROI '{roi or 'all'}' — no subject files for method '{method}'.", level="warning")
        return None

    dfs = []
    used_subjects = []
    for sub in subjects_to_use:
        df = _load_subject_tf(sub, roi_suffix, method_suffix, config)
        if df is None or df.empty or df.dropna(subset=["correlation", "frequency", "time"]).empty:
            continue
        dfs.append(df.dropna(subset=["correlation", "frequency", "time"]))
        used_subjects.append(sub)

    if len(dfs) < min_subjects:
        _log_local(f"Group TF correlation skipped for ROI '{roi or 'all'}' — fewer than {min_subjects} subjects with valid data.", level="warning")
        return None

    def _grid(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        freqs = np.unique(np.round(df["frequency"].to_numpy(dtype=float), 6))
        times = np.unique(np.round(df["time"].to_numpy(dtype=float), 6))
        return freqs, times

    f_common, t_common = _grid(dfs[0])
    for df in dfs[1:]:
        f, t = _grid(df)
        f_common = np.intersect1d(f_common, f)
        t_common = np.intersect1d(t_common, t)

    if f_common.size == 0 or t_common.size == 0:
        _log_local(f"Group TF correlation skipped for ROI '{roi or 'all'}' — unable to find common TF grid.", level="warning")
        return None

    mats: list[np.ndarray] = []
    for df in dfs:
        df_use = df.copy()
        df_use["frequency"] = np.round(df_use["frequency"].astype(float), 6)
        df_use["time"] = np.round(df_use["time"].astype(float), 6)
        pivot = df_use.pivot_table(index="frequency", columns="time", values="correlation", aggfunc="mean")
        pivot = pivot.reindex(index=f_common, columns=t_common)
        mats.append(pivot.to_numpy())

    Z = np.stack([np.arctanh(np.clip(m, -0.999999, 0.999999)) for m in mats], axis=0)
    z_mean = np.nanmean(Z, axis=0)
    z_sd = np.nanstd(Z, axis=0, ddof=1)
    n = np.sum(np.isfinite(Z), axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = z_sd / np.sqrt(np.maximum(n, 1))
        denom[denom == 0] = np.nan
        t_stat = z_mean / denom

    p_vals = np.full_like(t_stat, np.nan, dtype=float)
    finite = np.isfinite(t_stat) & (n > 1)
    if np.any(finite):
        df = np.maximum(n[finite] - 1, 1)
        t_abs = np.abs(t_stat[finite])
        p_vals[finite] = 2.0 * t_dist.sf(t_abs, df=df)

    rej, q_flat = _fdr_bh_values(p_vals[np.isfinite(p_vals)], alpha=alpha)
    q_vals = np.full_like(p_vals, np.nan)
    if q_flat is not None:
        q_vals[np.isfinite(p_vals)] = q_flat

    sig_mask = np.zeros_like(p_vals, dtype=bool)
    if rej is not None:
        sig_mask[np.isfinite(p_vals)] = rej.astype(bool)
    sig_mask &= (n >= min_subjects)
    r_mean = np.tanh(z_mean)

    stats_dir = deriv_group_stats_path(config.deriv_root)
    plots_dir = deriv_group_plots_path(config.deriv_root, "tf_corr")
    stats_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    out_tsv = stats_dir / f"tf_corr_group{roi_suffix}{method_suffix}.tsv"
    df_out = pd.DataFrame(
        {
            "frequency": np.repeat(f_common, len(t_common)),
            "time": np.tile(t_common, len(f_common)),
            "r_mean": r_mean.flatten(),
            "z_mean": z_mean.flatten(),
            "n": n.flatten(),
            "p": p_vals.flatten(),
            "q": q_vals.flatten(),
            "significant": sig_mask.flatten(),
        }
    )
    df_out.to_csv(out_tsv, sep="\t", index=False)

    extent = [t_common[0], t_common[-1], f_common[0], f_common[-1]]
    cmap = "RdBu_r"
    vmin = -0.6
    vmax = 0.6
    figure_paths = []

    def _annotate(fig):
        try:
            bwin = config.get("time_frequency_analysis.baseline_window", [-0.5, -0.01]) if config else [-0.5, -0.01]
            corr_txt = f"FDR BH α={alpha}"
            text = (
                f"Group TF correlation | Baseline: [{float(bwin[0]):.2f}, {float(bwin[1]):.2f}] s | "
                f"{corr_txt}"
            )
            fig.text(0.01, 0.01, text, fontsize=8, alpha=0.8)
        except Exception:
            pass

    fig1, ax1 = plt.subplots(figsize=(7.2, 5.4))
    im1 = ax1.imshow(
        r_mean,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax1.axvline(0.0, color="k", linestyle="--", alpha=0.6)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency (Hz)")
    title_roi = roi or "All channels"
    title_method = method_suffix.strip("_").title()
    ax1.set_title(f"Group TF correlation — mean r ({title_method}, {title_roi})")
    cb1 = plt.colorbar(im1, ax=ax1)
    cb1.set_label("r")
    plt.tight_layout()
    _annotate(fig1)
    save_formats = config.get("output.save_formats", ["png"]) if config else ["png"]
    _save_fig(
        fig1,
        plots_dir,
        f"tf_corr_group_rmean{roi_suffix}{method_suffix}",
        config,
        
        logger=logger,
    )
    for ext in save_formats:
        figure_paths.append(
            plots_dir / f"tf_corr_group_rmean{roi_suffix}{method_suffix}.{ext}"
        )

    fig2, ax2 = plt.subplots(figsize=(7.2, 5.4))
    im2 = ax2.imshow(
        np.where(sig_mask, r_mean, np.nan),
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax2.axvline(0.0, color="k", linestyle="--", alpha=0.6)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    sig_title = f"Group TF correlation — FDR<{alpha:g} ({title_method}, {title_roi})"
    ax2.set_title(sig_title)
    cb2 = plt.colorbar(im2, ax=ax2)
    cb2.set_label("r (significant)")
    plt.tight_layout()
    _annotate(fig2)
    save_formats = config.get("output.save_formats", ["png"]) if config else ["png"]
    _save_fig(
        fig2,
        plots_dir,
        f"tf_corr_group_sig{roi_suffix}{method_suffix}",
        config,
        
        logger=logger,
    )
    for ext in save_formats:
        figure_paths.append(
            plots_dir / f"tf_corr_group_sig{roi_suffix}{method_suffix}.{ext}"
        )

    _log_local(
        (
            f"Group TF correlation saved (ROI={roi or 'all'}, method={method_suffix.strip('_')}): "
            f"{out_tsv}"
        )
    )
    return out_tsv, figure_paths



