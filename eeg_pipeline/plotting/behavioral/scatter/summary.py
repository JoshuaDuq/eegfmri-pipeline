from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eeg_pipeline.infra.tsv import read_tsv
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.utils.config.loader import load_settings
from eeg_pipeline.io.formatting import sanitize_label
from eeg_pipeline.infra.logging import get_subject_logger
from eeg_pipeline.infra.paths import deriv_plots_path, deriv_stats_path, ensure_dir
from eeg_pipeline.plotting.io.figures import (
    get_band_color,
    get_behavior_footer as _get_behavior_footer,
    get_default_config as _get_default_config,
    save_fig,
)


def _get_behavioral_config(plot_cfg):
    return plot_cfg.plot_type_configs.get("behavioral", {})


def _create_rating_distribution_plot(
    y: pd.Series,
    config=None,
) -> Tuple[plt.Figure, plt.Axes]:
    config = config or _get_default_config()
    plot_cfg = get_plot_config(config)
    behavioral_config = _get_behavioral_config(plot_cfg)
    rating_config = behavioral_config.get("rating_distribution", {})

    figsize_width = rating_config.get("figsize_width", 8)
    figsize_height = rating_config.get("figsize_height", 6)
    bins = rating_config.get("bins", 20)
    alpha = rating_config.get("alpha", 0.7)
    edgecolor = rating_config.get("edgecolor", "black")

    fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
    ax.hist(y.dropna(), bins=bins, alpha=alpha, edgecolor=edgecolor)
    grid_alpha = rating_config.get("grid_alpha", 0.3)
    ax.set_xlabel("Pain Rating")
    ax.set_ylabel("Frequency")
    ax.set_title("Rating Distribution")
    ax.grid(True, alpha=grid_alpha)

    mean_rating = y.mean()
    std_rating = y.std()
    text_box_alpha = rating_config.get("text_box_alpha", 0.8)
    ax.text(
        0.02,
        0.98,
        f"Mean: {mean_rating:.2f} ± {std_rating:.2f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=text_box_alpha),
    )

    return fig, ax


def plot_behavioral_response_patterns(
    y: pd.Series,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config=None,
) -> None:
    config = config or _get_default_config()
    plot_cfg = get_plot_config(config)
    fig, _ = _create_rating_distribution_plot(y, config)

    plt.tight_layout()
    save_fig(
        fig,
        save_dir / f"sub-{subject}_rating_distribution",
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        footer=_get_behavior_footer(config),
    )
    plt.close(fig)


def _load_correlation_stats(
    stats_dir: Path,
    logger: logging.Logger,
    config: Optional[Any] = None,
) -> Optional[pd.DataFrame]:
    if config is None:
        target_rating = "rating"
        target_temperature = "temperature"
    else:
        plot_cfg = get_plot_config(config)
        behavioral_config = _get_behavioral_config(plot_cfg)
        target_rating = behavioral_config.get("target_rating", "rating")
        target_temperature = behavioral_config.get("target_temperature", "temperature")

    candidate_files = [
        (target_rating, stats_dir / "corr_stats_pow_combined_vs_rating.tsv"),
        (target_rating, stats_dir / "corr_stats_power_combined_vs_rating.tsv"),
        (target_temperature, stats_dir / "corr_stats_pow_combined_vs_temp.tsv"),
        (target_temperature, stats_dir / "corr_stats_power_combined_vs_temp.tsv"),
    ]

    frames = []
    for target_label, path in candidate_files:
        if path.exists():
            df = read_tsv(path)
            df["target"] = target_label
            frames.append(df)

    if not frames:
        logger.warning(
            "No combined correlation stats found for rating or temperature. Expected one of: "
            + ", ".join(str(p) for _, p in candidate_files)
        )
        return None

    return pd.concat(frames, axis=0, ignore_index=True)


def _filter_significant_predictors(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    return df[
        (df["p"] <= alpha)
        & df["r"].notna()
        & df["p"].notna()
        & df["channel"].notna()
        & df["band"].notna()
    ].copy()


def _create_predictor_labels(df: pd.DataFrame) -> pd.Series:
    if "target" in df.columns:
        return (
            df["channel"]
            + " ("
            + df["band"]
            + ") ["
            + df["target"].astype(str)
            + "]"
        )
    return df["channel"] + " (" + df["band"] + ")"


def _create_predictors_plot(
    df_top: pd.DataFrame,
    top_n: int,
    alpha: float,
    config,
) -> Tuple[plt.Figure, plt.Axes]:
    plot_cfg = get_plot_config(config)
    behavioral_config = _get_behavioral_config(plot_cfg)
    predictors_config = behavioral_config.get("predictors_plot", {})

    figsize_width = predictors_config.get("figsize_width", 10)
    figsize_height_base = predictors_config.get("figsize_height_base", 8)
    figsize_height_per_predictor = predictors_config.get("figsize_height_per_predictor", 0.4)
    fig, ax = plt.subplots(
        figsize=(figsize_width, max(figsize_height_base, top_n * figsize_height_per_predictor))
    )

    band_colors = {str(band): get_band_color(band, config) for band in df_top["band"].unique()}
    colors = [band_colors[band] for band in df_top["band"]]

    bar_alpha = predictors_config.get("bar_alpha", 0.8)
    bar_edgecolor = predictors_config.get("bar_edgecolor", "black")
    bar_linewidth = predictors_config.get("bar_linewidth", 0.5)
    y_pos = np.arange(len(df_top))
    ax.barh(
        y_pos,
        df_top["abs_r"],
        color=colors,
        alpha=bar_alpha,
        edgecolor=bar_edgecolor,
        linewidth=bar_linewidth,
    )

    ax.set_yticks(y_pos)
    label_fontsize = predictors_config.get("label_fontsize", 11)
    ax.set_yticklabels(df_top["predictor"], fontsize=label_fontsize)
    xlabel_fontsize = predictors_config.get("xlabel_fontsize", 12)
    ax.set_xlabel(
        f"|Spearman ρ| with Behavior (p < {alpha})",
        fontweight="bold",
        fontsize=xlabel_fontsize,
    )
    title_fontsize = predictors_config.get("title_fontsize", 14)
    title_pad = predictors_config.get("title_pad", 20)
    ax.set_title(
        f"Top {top_n} Significant Behavioral Predictors",
        fontweight="bold",
        fontsize=title_fontsize,
        pad=title_pad,
    )

    value_fontsize = predictors_config.get("value_fontsize", 10)
    value_x_offset = predictors_config.get("value_x_offset", 0.01)
    for i, (_, row) in enumerate(df_top.iterrows()):
        r_val, p_val, abs_r_val = row["r"], row["p"], row["abs_r"]
        sign_str = "(+)" if r_val >= 0 else "(-)"
        x_pos = abs_r_val + value_x_offset
        ax.text(
            x_pos,
            i,
            f"{abs_r_val:.3f} {sign_str} (p={p_val:.3f})",
            va="center",
            ha="left",
            fontsize=value_fontsize,
            fontweight="normal",
        )

    max_r = df_top["abs_r"].max()
    xlim_multiplier = predictors_config.get("xlim_multiplier", 1.25)
    ax.set_xlim(0, max_r * xlim_multiplier)
    grid_alpha = predictors_config.get("grid_alpha", 0.3)
    grid_linestyle = predictors_config.get("grid_linestyle", "-")
    grid_linewidth = predictors_config.get("grid_linewidth", 0.5)
    ax.grid(True, axis="x", alpha=grid_alpha, linestyle=grid_linestyle, linewidth=grid_linewidth)
    ax.set_axisbelow(True)

    return fig, ax


def _export_top_predictors(
    df_top: pd.DataFrame,
    stats_dir: Path,
    top_n: int,
    logger: logging.Logger,
) -> None:
    top_predictors_file = stats_dir / f"top_{top_n}_behavioral_predictors.tsv"
    export_cols = ["predictor", "channel", "band", "r", "abs_r", "p", "n"]
    if "target" in df_top.columns and "target" not in export_cols:
        export_cols = ["target"] + export_cols

    df_top_export = df_top[export_cols].copy()
    df_top_export = df_top_export.sort_values("abs_r", ascending=False)
    df_top_export.to_csv(top_predictors_file, sep="\t", index=False)
    logger.info(f"Exported top predictors data: {top_predictors_file}")


def plot_top_behavioral_predictors(
    subject: str,
    task: Optional[str] = None,
    alpha: float = None,
    top_n: int = None,
    plots_dir: Optional[Path] = None,
) -> None:
    config = load_settings()
    if task is None:
        task = config.task

    alpha = alpha or config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
    top_n = top_n or int(config.get("behavior_analysis.predictors.top_n", 20))
    log_name = config.get("output.log_file_name", "behavior_analysis.log")
    logger = get_subject_logger("behavior_analysis", subject, log_name, config=config)
    logger.info(f"Creating top {top_n} behavioral predictors plot for sub-{subject}")

    if plots_dir is None:
        deriv_root = Path(config.deriv_root)
        plot_cfg = get_plot_config(config) if config else None
        behavioral_config = (
            plot_cfg.plot_type_configs.get("behavioral", {}) if plot_cfg else {}
        )
        plot_subdir = behavioral_config.get("plot_subdir", "behavior") if plot_cfg else "behavior"
        plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)

    stats_dir = deriv_stats_path(Path(config.deriv_root), subject)
    ensure_dir(plots_dir)

    df = _load_correlation_stats(stats_dir, logger)
    if df is None:
        return

    df_sig = _filter_significant_predictors(df, alpha)
    if df_sig.empty:
        logger.warning(f"No significant correlations found (p <= {alpha})")
        return

    df_sig["abs_r"] = df_sig["r"].abs()
    df_top = df_sig.nlargest(top_n, "abs_r")

    if df_top.empty:
        logger.warning("No top correlations to plot")
        return

    df_top["predictor"] = _create_predictor_labels(df_top)
    df_top = df_top.sort_values("abs_r", ascending=True)

    fig, _ = _create_predictors_plot(df_top, top_n, alpha, config)
    plt.tight_layout()

    plot_cfg = get_plot_config(config)
    output_path = plots_dir / f"sub-{subject}_top_{top_n}_behavioral_predictors"
    save_fig(
        fig,
        output_path,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        footer=_get_behavior_footer(config),
    )
    plt.close(fig)

    logger.info(f"Saved top {top_n} behavioral predictors plot: {output_path}.png")

    if "target" in df.columns:
        counts_by_tgt = df_sig["target"].value_counts().to_dict() if len(df_sig) else {}
        logger.info(
            "Found %d significant predictors across targets %s (out of %d total correlations)",
            len(df_top),
            counts_by_tgt,
            len(df),
        )
    else:
        logger.info(
            "Found %d significant predictors (out of %d total correlations)",
            len(df_top),
            len(df),
        )

    _export_top_predictors(df_top, stats_dir, top_n, logger)
