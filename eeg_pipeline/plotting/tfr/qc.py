"""
TFR quality control plotting functions.

Functions for creating quality control visualizations for time-frequency
representations, including baseline vs active comparisons.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

from eeg_pipeline.plotting.io.figures import save_fig as central_save_fig
from ...utils.analysis.tfr import (
    validate_baseline_indices,
    average_tfr_band,
)
from ..config import get_plot_config
from ..core.utils import get_font_sizes, log
from .contrasts import _get_baseline_window
from .channels import _save_fig


###################################################################
# QC Plotting Functions
###################################################################


def qc_baseline_active_power(
    tfr,
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    active_window: Tuple[float, float] = (3.0, 10.5),
    logger: Optional[logging.Logger] = None,
) -> None:
    """Create quality control plots comparing baseline vs active power.
    
    Generates histograms and summary statistics comparing power during baseline
    and active periods across frequency bands. Creates topomap visualizations
    showing percentage change from baseline to active.
    
    Args:
        tfr: MNE TFR object (EpochsTFR or AverageTFR)
        out_dir: Output directory path
        config: Configuration object
        baseline: Optional baseline window tuple (defaults to config)
        active_window: Active window tuple for statistics
        logger: Optional logger instance
    """
    baseline = _get_baseline_window(config, baseline)
    qc_dir = out_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    data = getattr(tfr, "data", None)
    if data is None or data.ndim not in [3, 4]:
        return

    if data.ndim == 3:
        data = data[None, ...]

    freqs = np.asarray(tfr.freqs)
    times = np.asarray(tfr.times)

    from ...utils.config.loader import get_config_value, ensure_config
    config = ensure_config(config)
    min_baseline_samples = int(get_config_value(config, "time_frequency_analysis.constants.min_samples_for_baseline_validation", 5))
    b_start, b_end, tmask_base_idx = validate_baseline_indices(times, baseline, min_samples=min_baseline_samples, logger=logger)
    tmask_base = np.zeros(len(times), dtype=bool)
    tmask_base[tmask_base_idx] = True
    tmask_plat = (times >= active_window[0]) & (times < active_window[1])

    if not np.any(tmask_plat):
        log(f"QC skipped: active samples={int(tmask_plat.sum())}", logger, "warning")
        return

    tfr_avg = tfr.average() if isinstance(tfr, mne.time_frequency.EpochsTFR) else tfr
    
    rows = []
    font_sizes = get_font_sizes()

    band_bounds = config.get("time_frequency_analysis.bands") if config else None
    if band_bounds is None and hasattr(config, "frequency_bands"):
        band_bounds = config.frequency_bands
    if band_bounds is None:
        log("QC skipped: no frequency bands found in config", logger, "warning")
        return
    
    band_bounds_dict = {k: tuple(v) for k, v in band_bounds.items()}
    for band, (fmin, fmax) in band_bounds_dict.items():
        fmask = (freqs >= float(fmin)) & (freqs <= (float(fmax) if fmax is not None else freqs.max()))
        if not np.any(fmask):
            continue

        base = data[:, :, fmask, :][:, :, :, tmask_base].mean(axis=(2, 3))
        plat = data[:, :, fmask, :][:, :, :, tmask_plat].mean(axis=(2, 3))

        base_flat = base.reshape(-1)
        plat_flat = plat.reshape(-1)
        plot_cfg = get_plot_config(config)
        tfr_config = plot_cfg.plot_type_configs.get("tfr", {}) if plot_cfg else {}
        qc_config = tfr_config.get("qc", {})
        epsilon_for_division = qc_config.get("epsilon_for_division", 1e-20)
        percentage_multiplier = tfr_config.get("percentage_multiplier", 100.0)
        pct_change = ((plat_flat - base_flat) / (base_flat + epsilon_for_division)) * percentage_multiplier

        qc_fig_width = qc_config.get("fig_width", 8)
        qc_fig_height = qc_config.get("fig_height", 3)
        histogram_bins = qc_config.get("histogram_bins", 50)
        histogram_alpha = qc_config.get("histogram_alpha", 0.8)
        fig, axes = plt.subplots(1, 2, figsize=(qc_fig_width, qc_fig_height), constrained_layout=True)
        axes[0].hist(base_flat, bins=histogram_bins, color="tab:blue", alpha=histogram_alpha)
        axes[0].set_title(f"Baseline power — {band}")
        axes[0].set_xlabel("Power (a.u.)")
        axes[0].set_ylabel("Count")
        axes[1].hist(pct_change, bins=histogram_bins, color="tab:orange", alpha=histogram_alpha)
        axes[1].set_title(f"% signal change (active vs baseline) — {band}")
        axes[1].set_xlabel("% change")
        axes[1].set_ylabel("Count")
        fig.suptitle(
            f"Baseline vs Active QC — {band}\n(baseline={b_start:.2f}–{b_end:.2f}s; active={active_window[0]:.2f}–{active_window[1]:.2f}s)",
            fontsize=font_sizes["ylabel"],
        )
        _save_fig(fig, qc_dir, f"qc_baseline_active_hist_{band}.png", config=config, logger=logger)

        topo_vals = None
        if tfr_avg is not None:
            fmin_eff = float(fmin)
            fmax_eff = float(fmax) if fmax is not None else float(freqs.max())
            topo_plat = average_tfr_band(
                tfr_avg,
                fmin=fmin_eff,
                fmax=fmax_eff,
                tmin=float(active_window[0]),
                tmax=float(active_window[1]),
            )
            topo_base = average_tfr_band(
                tfr_avg,
                fmin=fmin_eff,
                fmax=fmax_eff,
                tmin=float(b_start),
                tmax=float(b_end),
            )
            if topo_plat is not None and topo_base is not None:
                plot_cfg_topo_pct = get_plot_config(config)
                tfr_config_topo_pct = plot_cfg_topo_pct.plot_type_configs.get("tfr", {}) if plot_cfg_topo_pct else {}
                qc_config_topo_pct = tfr_config_topo_pct.get("qc", {})
                epsilon_for_division_topo_pct = qc_config_topo_pct.get("epsilon_for_division", 1e-20)
                percentage_multiplier_topo_pct = tfr_config_topo_pct.get("percentage_multiplier", 100.0)
                topo_vals = ((topo_plat - topo_base) / (topo_base + epsilon_for_division_topo_pct)) * percentage_multiplier_topo_pct

        row = {
            "band": band,
            "baseline_mean": float(np.nanmean(base_flat)),
            "baseline_median": float(np.nanmedian(base_flat)),
            "active_mean": float(np.nanmean(plat_flat)),
            "active_median": float(np.nanmedian(plat_flat)),
            "pct_change_mean": float(np.nanmean(pct_change)),
            "pct_change_median": float(np.nanmedian(pct_change)),
            "n_baseline_samples": int(tmask_base.sum()),
            "n_active_samples": int(tmask_plat.sum()),
        }
        if topo_vals is not None and np.isfinite(topo_vals).any():
            row["pct_change_mean_topomap"] = float(np.nanmean(topo_vals))
            row["pct_change_median_topomap"] = float(np.nanmedian(topo_vals))
        else:
            row["pct_change_mean_topomap"] = float("nan")
            row["pct_change_median_topomap"] = float("nan")
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df_path = qc_dir / "qc_baseline_active_summary.tsv"
        df.to_csv(df_path, sep="\t", index=False)
        log(f"Saved QC summary: {df_path}", logger)


__all__ = [
    "qc_baseline_active_power",
]

