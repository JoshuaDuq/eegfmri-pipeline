from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.infra.logging import get_subject_logger
from eeg_pipeline.infra.paths import deriv_plots_path, deriv_stats_path, ensure_dir
from eeg_pipeline.infra.tsv import read_tsv
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.io.figures import (
    get_band_color,
    get_behavior_footer as _get_behavior_footer,
    save_fig,
)
from eeg_pipeline.utils.analysis.stats.correlation import (
    format_correlation_method_label,
    normalize_correlation_method,
)
from eeg_pipeline.utils.config.loader import get_config_value


def _get_behavioral_config(plot_cfg: Any) -> dict:
    """Extract behavioral plotting configuration from plot config."""
    if plot_cfg is None:
        return {}
    return plot_cfg.plot_type_configs.get("behavioral", {})


def _create_rating_distribution_plot(
    y: pd.Series,
    config: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create histogram plot of rating distribution with statistics."""
    plot_cfg = get_plot_config(config)
    behavioral_config = _get_behavioral_config(plot_cfg)
    rating_config = behavioral_config.get("rating_distribution", {})

    figsize_width = rating_config.get("figsize_width", 8)
    figsize_height = rating_config.get("figsize_height", 6)
    bins = rating_config.get("bins", 20)
    alpha = rating_config.get("alpha", 0.7)
    edgecolor = rating_config.get("edgecolor", "black")
    grid_alpha = rating_config.get("grid_alpha", 0.3)
    text_box_alpha = rating_config.get("text_box_alpha", 0.8)

    fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
    ax.hist(y.dropna(), bins=bins, alpha=alpha, edgecolor=edgecolor)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Frequency")
    ax.set_title("Rating Distribution")
    ax.grid(True, alpha=grid_alpha)

    mean_rating = y.mean()
    std_rating = y.std()
    stats_text = f"Mean: {mean_rating:.2f} ± {std_rating:.2f}"
    ax.text(
        0.02,
        0.98,
        stats_text,
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
    config: Any,
) -> None:
    """Plot and save rating distribution for a subject."""
    if config is None:
        raise ValueError("config is required for behavioral plotting")

    plot_cfg = get_plot_config(config)
    fig, _ = _create_rating_distribution_plot(y, config)

    plt.tight_layout()
    output_path = save_dir / f"sub-{subject}_rating_distribution"
    save_fig(
        fig,
        output_path,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        footer=_get_behavior_footer(config),
        config=config,
    )
    plt.close(fig)


def _extract_correlation_method_label(config: Optional[Any]) -> Optional[str]:
    """Extract formatted correlation method label from config."""
    if config is None:
        return None

    raw_method = get_config_value(
        config, "behavior_analysis.statistics.correlation_method", "spearman"
    )
    method = normalize_correlation_method(raw_method, default="spearman")
    robust_method = get_config_value(config, "behavior_analysis.robust_correlation", None)
    if robust_method is not None:
        robust_method = str(robust_method).strip().lower() or None

    return format_correlation_method_label(method, robust_method)


def _try_read_tsv(path: Path) -> Optional[pd.DataFrame]:
    """Attempt to read a TSV file, returning None if it doesn't exist or is empty."""
    if not path.exists():
        return None
    df = read_tsv(path)
    if df is None or df.empty:
        return None
    return df


def _find_unified_correlation_file(
    stats_dir: Path, method_label: Optional[str], target_rating: str
) -> Optional[pd.DataFrame]:
    """Search for unified correlations*.tsv files with optional method suffix."""
    glob_candidates: list[Path] = []
    if method_label:
        pattern = f"correlations*_{method_label}.tsv"
        glob_candidates.extend(sorted(stats_dir.glob(pattern)))
    glob_candidates.extend(sorted(stats_dir.glob("correlations*.tsv")))

    for path in glob_candidates:
        df = _try_read_tsv(path)
        if df is None:
            continue

        if "target" in df.columns:
            target_filtered = df[
                df["target"].astype(str).str.lower() == str(target_rating).lower()
            ].copy()
            if not target_filtered.empty:
                return target_filtered

        df_with_target = df.copy()
        df_with_target["target"] = target_rating
        return df_with_target

    return None


def _load_correlation_stats(
    stats_dir: Path,
    logger: logging.Logger,
    config: Optional[Any] = None,
) -> Optional[pd.DataFrame]:
    """Load correlation statistics from TSV files."""
    plot_cfg = get_plot_config(config) if config is not None else None
    behavioral_config = _get_behavioral_config(plot_cfg)
    target_rating = behavioral_config.get("target_rating", "rating")
    method_label = _extract_correlation_method_label(config)

    df = _find_unified_correlation_file(stats_dir, method_label, target_rating)
    if df is not None:
        return df

    logger.warning("No correlation stats found in %s", stats_dir)
    return None


def _extract_correlation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and normalize correlation coefficient and sample size columns."""
    r_col = "r_primary" if "r_primary" in df.columns else "r"
    df["r"] = pd.to_numeric(df.get(r_col), errors="coerce")
    df["n"] = pd.to_numeric(df.get("n"), errors="coerce")
    return df


def _extract_p_value_column(df: pd.DataFrame) -> Optional[str]:
    """Find the best available p-value column, preferring corrected values."""
    p_column_priority = ["q_global", "p_fdr", "p_primary_perm", "p_primary", "p"]
    for col in p_column_priority:
        if col in df.columns:
            numeric_values = pd.to_numeric(df[col], errors="coerce")
            if numeric_values.notna().any():
                return col
    return None


def _parse_feature_name(feat: Any) -> dict:
    """Parse feature name using NamingSchema into display components."""
    parsed = NamingSchema.parse(str(feat))
    if not parsed.get("valid"):
        return {
            "band": None,
            "identifier": None,
            "group": None,
            "segment": None,
            "scope": None,
        }
    return {
        "band": parsed.get("band"),
        "identifier": parsed.get("identifier"),
        "group": parsed.get("group"),
        "segment": parsed.get("segment"),
        "scope": parsed.get("scope"),
    }


def _create_predictor_label(row: pd.Series) -> str:
    """Create human-readable label for a predictor from parsed feature components."""
    group = str(row.get("group") or "feature")
    band = row.get("band")
    identifier = row.get("identifier")
    segment = row.get("segment")

    if identifier:
        label = f"{group}:{identifier}"
    else:
        label = group

    if band:
        label = f"{label} ({band})"

    if segment:
        label = f"{label} [{segment}]"

    if "target" in row and pd.notna(row["target"]):
        label = f"{label} → {row['target']}"

    return label


def _prepare_predictor_table(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare correlation dataframe for plotting by extracting and formatting columns."""
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df = _extract_correlation_columns(df)

    p_kind = _extract_p_value_column(df)
    if p_kind is None:
        return pd.DataFrame()

    df["p_plot"] = pd.to_numeric(df[p_kind], errors="coerce")
    df["p_kind"] = p_kind

    parsed_rows = df["feature"].apply(_parse_feature_name)
    parsed_df = pd.DataFrame(list(parsed_rows))
    for col in parsed_df.columns:
        df[col] = parsed_df[col]

    df["predictor"] = df.apply(_create_predictor_label, axis=1)
    df["abs_r"] = df["r"].abs()
    return df


def _filter_significant_predictors(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Filter predictors by significance threshold."""
    if df.empty:
        return pd.DataFrame()
    return df[
        (df["p_plot"].notna())
        & (df["p_plot"] <= alpha)
        & df["r"].notna()
    ].copy()


def _get_correlation_method_label_for_plot(config: Any) -> str:
    """Get formatted correlation method label for plot axis."""
    method_label = _extract_correlation_method_label(config)
    if method_label:
        return method_label
    return "Spearman ρ"


def _create_bar_colors(df_top: pd.DataFrame, config: Any, plot_cfg: Any) -> list[str]:
    """Create color list for bars based on frequency bands."""
    unique_bands = df_top["band"].dropna().unique()
    band_colors = {
        str(band): get_band_color(band, config) for band in unique_bands
    }
    default_color = plot_cfg.get_color("gray", plot_type="behavioral")
    return [
        band_colors.get(str(band), default_color) for band in df_top["band"]
    ]


def _add_value_labels(ax: plt.Axes, df_top: pd.DataFrame, config: dict) -> None:
    """Add correlation value and p-value labels to each bar."""
    value_fontsize = config.get("value_fontsize", 10)
    value_x_offset = config.get("value_x_offset", 0.01)

    for i, (_, row) in enumerate(df_top.iterrows()):
        r_val = row["r"]
        p_val = row["p_plot"]
        abs_r_val = row["abs_r"]
        sign_str = "(+)" if r_val >= 0 else "(-)"
        x_pos = abs_r_val + value_x_offset
        label_text = f"{abs_r_val:.3f} {sign_str} (p={p_val:.3f})"

        ax.text(
            x_pos,
            i,
            label_text,
            va="center",
            ha="left",
            fontsize=value_fontsize,
            fontweight="normal",
        )


def _create_predictors_plot(
    df_top: pd.DataFrame,
    top_n: int,
    alpha: float,
    config: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create horizontal bar plot of top behavioral predictors."""
    plot_cfg = get_plot_config(config)
    behavioral_config = _get_behavioral_config(plot_cfg)
    predictors_config = behavioral_config.get("predictors_plot", {})

    figsize_width = predictors_config.get("figsize_width", 10)
    figsize_height_base = predictors_config.get("figsize_height_base", 8)
    figsize_height_per_predictor = predictors_config.get(
        "figsize_height_per_predictor", 0.4
    )
    figsize_height = max(
        figsize_height_base, top_n * figsize_height_per_predictor
    )
    fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))

    colors = _create_bar_colors(df_top, config, plot_cfg)
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

    method_label = _get_correlation_method_label_for_plot(config)
    xlabel_text = f"|{method_label}| with Behavior (p < {alpha})"
    xlabel_fontsize = predictors_config.get("xlabel_fontsize", 12)
    ax.set_xlabel(xlabel_text, fontweight="bold", fontsize=xlabel_fontsize)

    title_fontsize = predictors_config.get("title_fontsize", 14)
    title_pad = predictors_config.get("title_pad", 20)
    ax.set_title(
        f"Top {top_n} Significant Behavioral Predictors",
        fontweight="bold",
        fontsize=title_fontsize,
        pad=title_pad,
    )

    _add_value_labels(ax, df_top, predictors_config)

    max_r = df_top["abs_r"].max()
    xlim_multiplier = predictors_config.get("xlim_multiplier", 1.25)
    ax.set_xlim(0, max_r * xlim_multiplier)

    grid_alpha = predictors_config.get("grid_alpha", 0.3)
    grid_linestyle = predictors_config.get("grid_linestyle", "-")
    grid_linewidth = predictors_config.get("grid_linewidth", 0.5)
    ax.grid(
        True,
        axis="x",
        alpha=grid_alpha,
        linestyle=grid_linestyle,
        linewidth=grid_linewidth,
    )
    ax.set_axisbelow(True)

    return fig, ax


def _export_top_predictors(
    df_top: pd.DataFrame,
    stats_dir: Path,
    top_n: int,
    logger: logging.Logger,
) -> None:
    """Export top predictors to TSV file."""
    top_predictors_file = stats_dir / f"top_{top_n}_behavioral_predictors.tsv"
    export_cols = ["predictor", "r", "abs_r", "p_plot", "p_kind", "n"]

    if "target" in df_top.columns:
        export_cols = ["target"] + export_cols

    df_top_export = df_top[export_cols].copy()
    df_top_export = df_top_export.sort_values("abs_r", ascending=False)
    df_top_export.to_csv(top_predictors_file, sep="\t", index=False)
    logger.info(f"Exported top predictors data: {top_predictors_file}")


def _get_plots_directory(config: Any, subject: str) -> Path:
    """Get plots directory path from config."""
    deriv_root = Path(config.deriv_root)
    plot_cfg = get_plot_config(config)
    behavioral_config = _get_behavioral_config(plot_cfg)
    plot_subdir = behavioral_config.get("plot_subdir", "behavior")
    return deriv_plots_path(deriv_root, subject, subdir=plot_subdir)


def _log_predictor_summary(
    logger: logging.Logger,
    df_top: pd.DataFrame,
    df_sig: pd.DataFrame,
    df: pd.DataFrame,
) -> None:
    """Log summary statistics about significant predictors."""
    if "target" in df.columns:
        counts_by_target = (
            df_sig["target"].value_counts().to_dict() if len(df_sig) > 0 else {}
        )
        logger.info(
            "Found %d significant predictors across targets %s (out of %d total correlations)",
            len(df_top),
            counts_by_target,
            len(df),
        )
    else:
        logger.info(
            "Found %d significant predictors (out of %d total correlations)",
            len(df_top),
            len(df),
        )


def plot_top_behavioral_predictors(
    subject: str,
    task: Optional[str] = None,
    alpha: Optional[float] = None,
    top_n: Optional[int] = None,
    plots_dir: Optional[Path] = None,
    *,
    config: Optional[Any] = None,
) -> None:
    """Plot and save top behavioral predictors based on correlation statistics."""
    if config is None:
        raise ValueError("config is required for behavioral plotting")

    alpha = (
        alpha
        if alpha is not None
        else float(config.get("behavior_analysis.statistics.fdr_alpha", 0.05))
    )
    top_n = top_n or int(config.get("behavior_analysis.predictors.top_n", 20))
    logger = get_subject_logger("behavior_analysis", subject)
    logger.info(f"Creating top {top_n} behavioral predictors plot for sub-{subject}")

    if plots_dir is None:
        plots_dir = _get_plots_directory(config, subject)

    stats_dir = deriv_stats_path(Path(config.deriv_root), subject)
    ensure_dir(plots_dir)

    df = _load_correlation_stats(stats_dir, logger, config=config)
    if df is None:
        return

    df = _prepare_predictor_table(df)
    if df.empty:
        logger.warning("No usable correlation table for top predictors")
        return

    df_sig = _filter_significant_predictors(df, alpha)
    if df_sig.empty:
        logger.warning(f"No significant correlations found (p <= {alpha})")
        return

    df_top = df_sig.nlargest(top_n, "abs_r").sort_values("abs_r", ascending=True)

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
        config=config,
    )
    plt.close(fig)

    logger.info(f"Saved top {top_n} behavioral predictors plot: {output_path}.png")

    _log_predictor_summary(logger, df_top, df_sig, df)
    _export_top_predictors(df_top, stats_dir, top_n, logger)
