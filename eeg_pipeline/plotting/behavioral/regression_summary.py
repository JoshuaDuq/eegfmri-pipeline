"""
Trialwise Regression Summary Plots
==================================

Summarizes subject-level regression results:
- Top features by delta R² (incremental variance explained)
- Signed beta bars for interpretability
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


def _load_regression(stats_dir: Path) -> Optional[pd.DataFrame]:
    candidates = sorted(stats_dir.glob("regression_feature_effects*.tsv"))
    if not candidates:
        return None
    try:
        return pd.read_csv(candidates[0], sep="\t")
    except Exception:
        return None


def plot_regression_summary(
    subject: str,
    task: str,
    deriv_root: Path,
    config: Any,
    logger: Optional[logging.Logger] = None,
    plots_dir: Optional[Path] = None,
    top_n: int = 20,
) -> Dict[str, Path]:
    if logger is None:
        logger = logging.getLogger(__name__)
    deriv_root = resolve_deriv_root(deriv_root=deriv_root, config=config)
    stats_dir = deriv_stats_path(deriv_root, subject)

    df = _load_regression(stats_dir)
    if df is None or df.empty:
        return {}

    plot_cfg = get_plot_config(config)
    if plots_dir is None:
        behavioral_config = plot_cfg.get_behavioral_config()
        plot_subdir = behavioral_config.get("plot_subdir", "behavior")
        plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)
    out_dir = plots_dir / "regression"
    ensure_dir(out_dir)

    df["delta_r2"] = pd.to_numeric(df.get("delta_r2", np.nan), errors="coerce")
    df["beta_feature"] = pd.to_numeric(df.get("beta_feature", np.nan), errors="coerce")
    df["p_primary"] = pd.to_numeric(df.get("p_primary", np.nan), errors="coerce")
    df = df[np.isfinite(df["beta_feature"])]
    if df.empty:
        return {}

    df = df.sort_values(["delta_r2", "p_primary"], ascending=[False, True]).head(int(top_n))

    fig, ax = plt.subplots(1, 1, figsize=plot_cfg.get_figure_size("wide", plot_type="behavioral"))
    labels = [str(c).replace("_", " ") for c in df["feature"].astype(str)]
    colors = ["tab:red" if b < 0 else "tab:green" for b in df["beta_feature"].to_numpy()]
    ax.barh(labels, df["delta_r2"].to_numpy(), color=colors, alpha=0.85)
    ax.invert_yaxis()
    ax.set_xlabel("ΔR² (feature added)")
    ax.set_title(f"Trialwise regression: top {len(df)} features", fontsize=plot_cfg.font.title, fontweight="bold")

    out_path = out_dir / f"sub-{subject}_regression_top_features.png"
    save_fig(fig, out_path, logger=logger)
    plt.close(fig)
    return {"regression_summary": out_path}


__all__ = ["plot_regression_summary"]

