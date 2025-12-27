"""
Trial Table Overview Plots
==========================

Quick sanity-check visuals for the canonical subject-level trial table:
- Rating vs Temperature (with fitted curve if available)
- Pain residual distribution
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eeg_pipeline.infra.paths import deriv_stats_path, ensure_dir, deriv_plots_path, resolve_deriv_root
from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.plotting.config import get_plot_config


def _load_trial_table(stats_dir: Path) -> Optional[pd.DataFrame]:
    candidates = sorted(stats_dir.glob("trials*.parquet"))
    if candidates:
        return pd.read_parquet(candidates[0])
    candidates = sorted(stats_dir.glob("trials*.tsv"))
    if candidates:
        return pd.read_csv(candidates[0], sep="\t")
    return None


def plot_trial_table_overview(
    subject: str,
    task: str,
    deriv_root: Path,
    config: Any,
    logger: Optional[logging.Logger] = None,
    plots_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    if logger is None:
        logger = logging.getLogger(__name__)
    deriv_root = resolve_deriv_root(deriv_root=deriv_root, config=config)

    stats_dir = deriv_stats_path(deriv_root, subject)
    if plots_dir is None:
        plot_cfg = get_plot_config(config)
        behavioral_config = plot_cfg.get_behavioral_config()
        plot_subdir = behavioral_config.get("plot_subdir", "behavior")
        plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)
    out_dir = plots_dir / "trial_table"
    ensure_dir(out_dir)

    df = _load_trial_table(stats_dir)
    if df is None or df.empty:
        logger.debug("Trial table overview: no trial table found for sub-%s", subject)
        return {}

    if "rating" not in df.columns:
        return {}

    plot_cfg = get_plot_config(config)
    fig, axes = plt.subplots(1, 2, figsize=plot_cfg.get_figure_size("wide", plot_type="behavioral"))
    ax0, ax1 = axes

    # Rating vs Temperature
    if "temperature" in df.columns:
        x = pd.to_numeric(df["temperature"], errors="coerce")
        y = pd.to_numeric(df["rating"], errors="coerce")
        mask = x.notna() & y.notna()
        ax0.scatter(x[mask], y[mask], s=25, alpha=0.75, edgecolor="white", linewidth=0.3)
        ax0.set_xlabel("Temperature")
        ax0.set_ylabel("Pain rating")
        ax0.set_title("Rating vs Temperature")

        if "rating_hat_from_temp" in df.columns:
            yhat = pd.to_numeric(df["rating_hat_from_temp"], errors="coerce")
            mask2 = mask & yhat.notna()
            if mask2.any():
                order = np.argsort(x[mask2].to_numpy())
                ax0.plot(x[mask2].to_numpy()[order], yhat[mask2].to_numpy()[order], color="black", linewidth=2.0)
    else:
        ax0.axis("off")

    # Pain residual distribution
    if "pain_residual" in df.columns:
        r = pd.to_numeric(df["pain_residual"], errors="coerce").dropna()
        ax1.hist(r.to_numpy(), bins=20, alpha=0.8, edgecolor="white")
        ax1.axvline(0, color="black", linewidth=1.0, alpha=0.7)
        ax1.set_title("Pain residual")
        ax1.set_xlabel("rating - f(temp)")
        ax1.set_ylabel("Count")
    else:
        ax1.axis("off")

    fig.suptitle(f"sub-{subject} trial table overview", fontsize=plot_cfg.font.title, fontweight="bold")
    out_path = out_dir / f"sub-{subject}_trial_table_overview.png"
    save_fig(fig, out_path, logger=logger)
    plt.close(fig)
    return {"trial_table_overview": out_path}


__all__ = ["plot_trial_table_overview"]

