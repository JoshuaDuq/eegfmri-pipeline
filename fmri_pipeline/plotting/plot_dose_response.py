from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math

from .plot_utils import FigureSpec, save_figure, subject_palette, _add_panel_label, _format_pvalue
from utils import get_log_function

log, _ = get_log_function(Path(__file__).stem)


def plot_subject_dose_response(
    level_df: pd.DataFrame,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
    signature: str = "nps"
) -> Optional[FigureSpec]:
    
    required_cols = {"temp_celsius", "br_score", "subject"}
    if not required_cols.issubset(level_df.columns):
        return None

    data = level_df.dropna(subset=["temp_celsius", "br_score"])
    if data.empty:
        return None

    subjects = sorted(data["subject"].unique())
    palette = subject_palette(subjects)
    n_subjects = len(subjects)
    ncols = min(3, n_subjects)
    nrows = math.ceil(n_subjects / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.0 * ncols, 3.2 * nrows),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel()

    for axis in axes[n_subjects:]:
        axis.axis("off")

    for idx, subject in enumerate(subjects):
        axis = axes[idx]
        sub_df = data.loc[data["subject"] == subject].sort_values("temp_celsius")
        
        _add_panel_label(axis, chr(65 + idx), x=-0.12, y=1.05)
        
        color = palette.get(subject, (0.3, 0.3, 0.3))
        
        axis.plot(
            sub_df["temp_celsius"],
            sub_df["br_score"],
            color=color,
            marker="o",
            markersize=5,
            linewidth=1.2,
            markeredgewidth=0.5,
            markeredgecolor="white",
            zorder=3,
        )
        
        if len(sub_df) >= 3:
            try:
                fit = stats.linregress(sub_df["temp_celsius"], sub_df["br_score"])
                fit_line = fit.intercept + fit.slope * sub_df["temp_celsius"]
                axis.plot(
                    sub_df["temp_celsius"],
                    fit_line,
                    color=color,
                    linestyle="-",
                    linewidth=0.8,
                    alpha=0.5,
                    zorder=2,
                )
                p_str = _format_pvalue(fit.pvalue)
                stats_text = f"r² = {fit.rvalue ** 2:.2f}, {p_str}"
                axis.text(0.98, 0.02, stats_text, transform=axis.transAxes, fontsize=7, va="bottom", ha="right", color="0.3")
            except Exception:
                pass
        
        axis.axhline(0, color="0.7", linewidth=0.5, linestyle="-", alpha=0.8, zorder=1)
        axis.set_title(subject, fontsize=10, fontweight="normal")
        if idx % ncols == 0:
            axis.set_ylabel(f"{signature.upper()} (a.u.)", fontsize=8.5)
        if idx >= (nrows - 1) * ncols:
            axis.set_xlabel("Temperature (°C)")
        sns.despine(ax=axis, trim=True)

    fig_name = f"Fig1a_{signature.upper()}_DoseResponse_BySubject_ConditionLevel"
    paths = save_figure(fig, fig_name, figure_dir, formats, dpi)
    
    return FigureSpec(
        name=fig_name,
        description=f"Subject-specific {signature.upper()} dose-response curves",
        paths=paths,
        stats_paths=[],
    )


def plot_group_dose_response(
    level_df: pd.DataFrame,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
    signature: str = "nps"
) -> Optional[FigureSpec]:
    
    required_cols = {"temp_celsius", "br_score", "subject"}
    if not required_cols.issubset(level_df.columns):
        return None

    data = level_df.dropna(subset=["temp_celsius", "br_score"])
    if data.empty or data["subject"].nunique() < 1:
        return None

    summary = (
        data.groupby("temp_celsius")
        .agg(
            mean_br=("br_score", "mean"),
            std_br=("br_score", "std"),
            n=("br_score", "count"),
        )
        .reset_index()
        .sort_values("temp_celsius")
    )
    
    if summary.empty:
        return None

    fig, axis = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    
    axis.errorbar(
        summary["temp_celsius"],
        summary["mean_br"],
        yerr=summary["std_br"] / np.sqrt(summary["n"]),
        marker="o",
        markersize=6,
        linewidth=1.5,
        capsize=4,
        capthick=1,
        color="0.2",
    )
    
    if len(summary) >= 3:
        try:
            fit = stats.linregress(summary["temp_celsius"], summary["mean_br"])
            fit_line = fit.intercept + fit.slope * summary["temp_celsius"]
            axis.plot(summary["temp_celsius"], fit_line, color="0.4", linestyle="--", linewidth=1.0, alpha=0.7)
        except Exception:
            pass
    
    axis.axhline(0, color="0.7", linewidth=0.5, linestyle="-", alpha=0.8)
    axis.set_xlabel("Temperature (°C)")
    axis.set_ylabel(f"{signature.upper()} (a.u.)")
    axis.set_title("Group Dose-Response", fontsize=11, fontweight="normal")
    sns.despine(ax=axis, trim=True)

    fig_name = f"Fig1b_{signature.upper()}_DoseResponse_Group"
    paths = save_figure(fig, fig_name, figure_dir, formats, dpi)
    
    return FigureSpec(
        name=fig_name,
        description=f"Group-level {signature.upper()} dose-response curve",
        paths=paths,
        stats_paths=[],
    )

