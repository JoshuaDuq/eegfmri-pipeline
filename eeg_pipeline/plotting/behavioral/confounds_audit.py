"""
Confounds Audit Plots
=====================

Visualizes QC metrics that correlate with rating/temperature.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eeg_pipeline.infra.paths import deriv_stats_path, ensure_dir, deriv_plots_path, resolve_deriv_root
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.io.figures import save_fig


def _load_confounds(stats_dir: Path) -> Optional[pd.DataFrame]:
    candidates = sorted(stats_dir.glob("confounds_audit*.tsv"))
    if not candidates:
        return None
    try:
        return pd.read_csv(candidates[0], sep="\t")
    except Exception:
        return None


def plot_confounds_audit(
    subject: str,
    task: str,
    deriv_root: Path,
    config: Any,
    logger: Optional[logging.Logger] = None,
    plots_dir: Optional[Path] = None,
    top_n: int = 15,
) -> Dict[str, Path]:
    if logger is None:
        logger = logging.getLogger(__name__)
    deriv_root = resolve_deriv_root(deriv_root=deriv_root, config=config)
    stats_dir = deriv_stats_path(deriv_root, subject)

    df = _load_confounds(stats_dir)
    if df is None or df.empty:
        return {}

    plot_cfg = get_plot_config(config)
    if plots_dir is None:
        behavioral_config = plot_cfg.get_behavioral_config()
        plot_subdir = behavioral_config.get("plot_subdir", "behavior")
        plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)
    out_dir = plots_dir / "confounds"
    ensure_dir(out_dir)

    df["q"] = pd.to_numeric(df.get("q", np.nan), errors="coerce")
    df["r"] = pd.to_numeric(df.get("r", np.nan), errors="coerce")
    df = df[np.isfinite(df["r"])]
    if df.empty:
        return {}

    saved: Dict[str, Path] = {}
    for target in sorted(df["target"].dropna().unique()):
        sub = df[df["target"] == target].copy()
        sub["abs_r"] = sub["r"].abs()
        sub = sub.sort_values(["q", "abs_r"], ascending=[True, False])
        sub = sub.head(int(top_n))
        if sub.empty:
            continue

        fig, ax = plt.subplots(1, 1, figsize=plot_cfg.get_figure_size("wide", plot_type="behavioral"))
        y_labels = [str(s).replace("quality_", "").replace("_", " ") for s in sub["qc_metric"].astype(str)]
        ax.barh(y_labels, sub["r"].to_numpy(), color="tab:blue", alpha=0.85)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Spearman r")
        ax.set_title(f"QC confounds vs {target} (top {len(sub)})", fontsize=plot_cfg.font.title, fontweight="bold")

        out_path = out_dir / f"sub-{subject}_confounds_{target}.png"
        save_fig(fig, out_path, logger=logger)
        plt.close(fig)
        saved[f"confounds_{target}"] = out_path

    return saved


__all__ = ["plot_confounds_audit"]

