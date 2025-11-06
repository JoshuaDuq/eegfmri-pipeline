from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math

from .plot_utils import FigureSpec, save_figure, subject_palette, _format_pvalue
from utils import get_log_function

log, _ = get_log_function(Path(__file__).stem)


def plot_vas_br_relationship(
    level_df: pd.DataFrame,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
    signature: str = "nps"
) -> Optional[FigureSpec]:
    columns = {"mean_vas", "br_score", "subject"}
    if not columns.issubset(level_df.columns):
        return None

    data = level_df.dropna(subset=["mean_vas", "br_score"])
    if data.empty:
        return None

    palette = subject_palette(sorted(data["subject"].unique()))

    fig, axis = plt.subplots(figsize=(3.0, 3.0), constrained_layout=True)
    
    for subject in data["subject"].unique():
        sub_data = data[data["subject"] == subject]
        axis.scatter(
            sub_data["mean_vas"],
            sub_data["br_score"],
            color=palette.get(subject, (0.5, 0.5, 0.5)),
            s=20,
            alpha=0.6,
            linewidth=0,
            zorder=2,
        )
    
    sns.regplot(
        data=data,
        x="mean_vas",
        y="br_score",
        scatter=False,
        color="0.3",
        line_kws={"linewidth": 1.0},
        ci=95,
        ax=axis,
    )
    axis.axhline(0, color="0.7", linewidth=0.5, linestyle="-", alpha=0.6)
    axis.set_xlabel("VAS rating")
    axis.set_ylabel(f"{signature.upper()} response (a.u.)")
    sns.despine(ax=axis, trim=True)

    if len(data) >= 3:
        try:
            vas_arr = data["mean_vas"].values
            br_arr = data["br_score"].values
            pearson_r, pearson_p = stats.pearsonr(vas_arr, br_arr)
            annotation = f"r={pearson_r:.2f}, {_format_pvalue(pearson_p)}, n={len(data)}"
            axis.text(
                0.02,
                0.98,
                annotation,
                transform=axis.transAxes,
                fontsize=7,
                va="top",
                ha="left",
                color="0.3",
            )
        except Exception:
            pass

    fig_name = f"Fig3_VAS_vs_{signature.upper()}"
    paths = save_figure(fig, fig_name, figure_dir, formats, dpi)
    
    return FigureSpec(
        name=fig_name,
        description=f"Brain-behavior correlation between subjective pain ratings (VAS) and {signature.upper()}.",
        paths=paths,
        stats_paths=[],
    )


def plot_temperature_vas_curve(
    trial_df: pd.DataFrame,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
    signature: str = "nps"
) -> Optional[FigureSpec]:
    required_cols = {"temp_celsius", "mean_vas"}
    if not required_cols.issubset(trial_df.columns):
        if "vas_rating" in trial_df.columns:
            trial_df = trial_df.copy()
            trial_df["mean_vas"] = trial_df.groupby(["subject", "temp_celsius"])["vas_rating"].transform("mean")
        else:
            return None

    data = trial_df.dropna(subset=["temp_celsius", "mean_vas"])
    if data.empty:
        return None

    summary = (
        data.groupby("temp_celsius")
        .agg(mean_vas=("mean_vas", "mean"), std_vas=("mean_vas", "std"), n=("mean_vas", "count"))
        .reset_index()
        .sort_values("temp_celsius")
    )
    if summary.empty:
        return None

    summary["ci"] = summary.apply(
        lambda row: stats.t.ppf(0.975, int(row["n"]) - 1) * row["std_vas"] / math.sqrt(row["n"])
        if row["n"] > 1 and not np.isnan(row["std_vas"])
        else np.nan,
        axis=1
    )

    fig, axis = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
    axis.fill_between(
        summary["temp_celsius"],
        summary["mean_vas"] - summary["ci"],
        summary["mean_vas"] + summary["ci"],
        color="0.3",
        alpha=0.15,
        linewidth=0,
        zorder=2,
    )
    axis.plot(
        summary["temp_celsius"],
        summary["mean_vas"],
        marker="o",
        markersize=4,
        markeredgewidth=0,
        color="0.2",
        linewidth=1.5,
        zorder=3,
    )
    axis.set_xlabel("Temperature (°C)")
    axis.set_ylabel("VAS rating")
    sns.despine(ax=axis, trim=True)
    
    if len(summary) >= 3:
        try:
            vas_fit = stats.linregress(summary["temp_celsius"], summary["mean_vas"])
            vas_stats = f"r²={vas_fit.rvalue**2:.2f}, {_format_pvalue(vas_fit.pvalue)}"
            axis.text(
                0.02, 0.98,
                vas_stats,
                transform=axis.transAxes,
                fontsize=7,
                va="top",
                ha="left",
                color="0.3",
            )
        except Exception:
            pass

    fig_name = f"Fig7_{signature.upper()}_TemperatureVAS"
    paths = save_figure(fig, fig_name, figure_dir, formats, dpi)
    
    return FigureSpec(
        name=fig_name,
        description=f"Perceived pain intensity (VAS ratings) as a function of thermal stimulus temperature.",
        paths=paths,
        stats_paths=[],
    )


def plot_bland_altman(
    level_df: pd.DataFrame,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int
) -> Optional[FigureSpec]:
    required_cols = {"mean_vas", "br_score"}
    if not required_cols.issubset(level_df.columns):
        return None
    
    data = level_df.dropna(subset=["mean_vas", "br_score"])
    if len(data) < 3:
        return None
    
    vas_z = (data["mean_vas"] - data["mean_vas"].mean()) / data["mean_vas"].std()
    br_z = (data["br_score"] - data["br_score"].mean()) / data["br_score"].std()
    
    mean_val = (vas_z + br_z) / 2
    diff_val = br_z - vas_z
    
    mean_diff = np.mean(diff_val)
    std_diff = np.std(diff_val, ddof=1)
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff
    
    fig, axis = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)
    
    axis.scatter(mean_val, diff_val, s=15, alpha=0.5, color="0.3", linewidth=0, zorder=2)
    axis.axhline(mean_diff, color="0.2", linestyle="-", linewidth=1.0, label=f"Mean: {mean_diff:.3f}")
    axis.axhline(upper_loa, color="0.5", linestyle="--", linewidth=0.8, label=f"+1.96 SD: {upper_loa:.3f}")
    axis.axhline(lower_loa, color="0.5", linestyle="--", linewidth=0.8, label=f"-1.96 SD: {lower_loa:.3f}")
    
    axis.set_xlabel("Mean (VAS, NPS) [standardized]")
    axis.set_ylabel("Difference (NPS - VAS) [standardized]")
    axis.legend(fontsize=7, loc="best")
    sns.despine(ax=axis, trim=True)

    fig_name = "Fig8_BlandAltman_VAS_NPS"
    paths = save_figure(fig, fig_name, figure_dir, formats, dpi)
    
    return FigureSpec(
        name=fig_name,
        description="Bland-Altman plot assessing agreement between VAS ratings and NPS responses (both standardized).",
        paths=paths,
        stats_paths=[],
    )
