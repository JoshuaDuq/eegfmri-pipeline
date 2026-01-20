from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eeg_pipeline.infra.paths import deriv_stats_path, ensure_dir
from eeg_pipeline.plotting.io.figures import save_fig


def _find_first_file(stats_dir: Path, pattern: str) -> Optional[Path]:
    """Find the first matching file in stats directory."""
    matches = sorted(stats_dir.glob(pattern))
    return matches[0] if matches else None


def _validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate that dataframe has required columns and is not empty."""
    required_columns = {"r_overall", "r_group_std"}
    return not df.empty and required_columns.issubset(df.columns)


def _prepare_numeric_data(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Extract and validate numeric data for plotting."""
    overall_correlation = pd.to_numeric(df["r_overall"], errors="coerce")
    group_std = pd.to_numeric(df["r_group_std"], errors="coerce")
    valid_mask = overall_correlation.notna() & group_std.notna()
    return overall_correlation, group_std, valid_mask


def _calculate_stability_scores(
    overall_correlation: pd.Series,
    group_std: pd.Series,
    valid_mask: pd.Series,
) -> pd.Series:
    """Calculate stability scores as |r| / (std_r + epsilon)."""
    epsilon = 1e-6
    denominator = group_std + epsilon
    scores = (overall_correlation.abs() / denominator).where(valid_mask, np.nan)
    return scores


def _get_top_feature_indices(scores: pd.Series, n_top: int = 10) -> list:
    """Get indices of top N features by stability score."""
    return scores.sort_values(ascending=False).head(n_top).index.tolist()


def _get_group_column_name(df: pd.DataFrame) -> str:
    """Extract group column name from dataframe, defaulting to 'groups'."""
    if "group_column" in df.columns and not df["group_column"].empty:
        return str(df["group_column"].iloc[0])
    return "groups"


def _annotate_top_features(
    ax: Any,
    overall_correlation: pd.Series,
    group_std: pd.Series,
    df: pd.DataFrame,
    top_indices: list,
) -> None:
    """Annotate top features on the plot."""
    has_feature_column = "feature" in df.columns

    for idx in top_indices:
        if idx not in overall_correlation.index or idx not in group_std.index:
            continue

        x_value = overall_correlation.loc[idx]
        y_value = group_std.loc[idx]

        ax.scatter([x_value], [y_value], s=40)

        if has_feature_column:
            feature_name = str(df.loc[idx, "feature"])
            ax.annotate(
                feature_name,
                (x_value, y_value),
                fontsize=7,
                alpha=0.85,
            )


def _create_stability_plot(
    overall_correlation: pd.Series,
    group_std: pd.Series,
    valid_mask: pd.Series,
    subject: str,
    task: str,
    group_column_name: str,
) -> tuple[Any, Any]:
    """Create the stability scatter plot figure and axes."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=150)

    ax.scatter(
        overall_correlation[valid_mask],
        group_std[valid_mask],
        s=18,
        alpha=0.65,
    )
    ax.set_xlabel("Overall association (r)")
    ax.set_ylabel("Across-group variability (std r)")
    ax.set_title(
        f"sub-{subject} {task}: stability across {group_column_name}"
    )
    ax.grid(True, alpha=0.2)

    return fig, ax


def plot_stability_groupwise(
    *,
    subject: str,
    task: str,
    deriv_root: Path,
    config: Any,
    logger: Any,
    plots_dir: Path,
) -> Dict[str, Path]:
    """Plot stability groupwise analysis for a subject and task."""
    stats_dir = deriv_stats_path(deriv_root, subject)
    if not stats_dir.exists():
        return {}

    stability_path = _find_first_file(stats_dir, "stability_groupwise*.tsv")
    if stability_path is None or not stability_path.exists():
        return {}

    df = pd.read_csv(stability_path, sep="\t")
    if not _validate_dataframe(df):
        return {}

    out_dir = plots_dir / "stability"
    ensure_dir(out_dir)

    overall_correlation, group_std, valid_mask = _prepare_numeric_data(df)

    min_valid_points = 3
    if valid_mask.sum() < min_valid_points:
        return {}

    stability_scores = _calculate_stability_scores(
        overall_correlation, group_std, valid_mask
    )
    top_indices = _get_top_feature_indices(stability_scores, n_top=10)
    group_column_name = _get_group_column_name(df)

    fig, ax = _create_stability_plot(
        overall_correlation,
        group_std,
        valid_mask,
        subject,
        task,
        group_column_name,
    )

    _annotate_top_features(ax, overall_correlation, group_std, df, top_indices)

    fig.tight_layout()
    out_path = out_dir / f"sub-{subject}_stability_groupwise.png"
    save_fig(fig, out_path, logger=logger, config=config)
    plt.close(fig)
    logger.info("Saved stability plot: %s", out_path.name)
    return {"stability_groupwise": out_path}


__all__ = ["plot_stability_groupwise"]

