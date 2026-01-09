from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eeg_pipeline.infra.paths import deriv_stats_path, ensure_dir
from eeg_pipeline.infra.tsv import read_table
from eeg_pipeline.plotting.io.figures import save_fig


_MIN_VALID_POINTS = 5
_NUM_BIN_QUANTILES = 9
_MIN_BINS_FOR_PLOT = 4
_FIGURE_WIDTH = 7.5
_FIGURE_HEIGHT = 4.5
_FIGURE_DPI = 150
_SCATTER_SIZE = 16
_SCATTER_ALPHA = 0.65
_BINNED_LINE_WIDTH = 2.0
_BREAKPOINT_LINE_WIDTH = 1.5
_GRID_ALPHA = 0.2


def _find_first(stats_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(stats_dir.glob(pattern))
    return matches[0] if matches else None


def _find_trials_file(stats_dir: Path) -> Optional[Path]:
    tsv_path = _find_first(stats_dir, "trials*.tsv")
    if tsv_path is not None and tsv_path.exists():
        return tsv_path
    parquet_path = _find_first(stats_dir, "trials*.parquet")
    if parquet_path is not None and parquet_path.exists():
        return parquet_path
    return None


def _find_column_in_dataframe(
    df: pd.DataFrame, config: Any, config_key: str, default_candidates: List[str]
) -> Optional[str]:
    config_candidates = list(config.get(config_key, []) or [])
    all_candidates = config_candidates + default_candidates
    return next((col for col in all_candidates if col in df.columns), None)


def _load_metadata_value(metadata_path: Optional[Path], key: str) -> Optional[Any]:
    if metadata_path is None or not metadata_path.exists():
        return None
    try:
        content = metadata_path.read_text()
        metadata = json.loads(content)
        return metadata.get(key)
    except (json.JSONDecodeError, OSError):
        return None


def _load_model_metadata(stats_dir: Path) -> Tuple[Optional[str], Optional[float]]:
    comparison_path = _find_first(stats_dir, "temperature_model_comparison*.metadata.json")
    breakpoint_path = _find_first(stats_dir, "temperature_breakpoint_test*.metadata.json")
    best_model = _load_metadata_value(comparison_path, "best_model")
    best_breakpoint = _load_metadata_value(breakpoint_path, "best_breakpoint")
    return best_model, best_breakpoint


def _prepare_temperature_rating_data(
    df: pd.DataFrame, temp_col: str, rating_col: str
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    temperature = pd.to_numeric(df[temp_col], errors="coerce")
    rating = pd.to_numeric(df[rating_col], errors="coerce")
    valid_mask = temperature.notna() & rating.notna()
    return temperature, rating, valid_mask


def _plot_binned_means(ax: plt.Axes, temperature: pd.Series, rating: pd.Series) -> None:
    try:
        quantiles = np.linspace(0, 1, _NUM_BIN_QUANTILES)
        bin_edges = np.unique(np.quantile(temperature, quantiles))
        if bin_edges.size < _MIN_BINS_FOR_PLOT:
            return

        temperature_array = temperature.to_numpy()
        rating_array = rating.to_numpy()
        binned_data = pd.DataFrame(
            {"temperature": temperature_array, "rating": rating_array}
        )
        binned_data["bin"] = pd.cut(
            binned_data["temperature"],
            bins=bin_edges,
            include_lowest=True,
            duplicates="drop",
        )
        binned_means = (
            binned_data.groupby("bin", observed=True)
            .agg(temperature_mean=("temperature", "mean"), rating_mean=("rating", "mean"))
            .dropna()
        )
        ax.plot(
            binned_means["temperature_mean"],
            binned_means["rating_mean"],
            linewidth=_BINNED_LINE_WIDTH,
        )
    except (ValueError, KeyError):
        pass


def _create_temperature_plot(
    temperature: pd.Series,
    rating: pd.Series,
    valid_mask: pd.Series,
    temp_col: str,
    rating_col: str,
    subject: str,
    task: str,
    best_model: Optional[str],
    best_breakpoint: Optional[float],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(_FIGURE_WIDTH, _FIGURE_HEIGHT), dpi=_FIGURE_DPI)

    valid_temperature = temperature[valid_mask]
    valid_rating = rating[valid_mask]
    ax.scatter(valid_temperature, valid_rating, s=_SCATTER_SIZE, alpha=_SCATTER_ALPHA)

    _plot_binned_means(ax, valid_temperature, valid_rating)

    if best_breakpoint is not None and np.isfinite(best_breakpoint):
        ax.axvline(float(best_breakpoint), linestyle="--", linewidth=_BREAKPOINT_LINE_WIDTH)

    title = f"sub-{subject} {task}: {rating_col} vs {temp_col}"
    if best_model:
        title += f" (best: {best_model})"
    ax.set_title(title)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Rating")
    ax.grid(True, alpha=_GRID_ALPHA)
    fig.tight_layout()

    return fig


def plot_temperature_models(
    *,
    subject: str,
    task: str,
    deriv_root: Path,
    config: Any,
    logger: Any,
    plots_dir: Path,
) -> Dict[str, Path]:
    stats_dir = deriv_stats_path(deriv_root, subject)
    if not stats_dir.exists():
        return {}

    out_dir = plots_dir / "temperature_models"
    ensure_dir(out_dir)

    trials_path = _find_trials_file(stats_dir)
    if trials_path is None:
        return {}

    df = read_table(trials_path)

    temp_col = _find_column_in_dataframe(
        df, config, "event_columns.temperature", ["temperature"]
    )
    rating_col = _find_column_in_dataframe(
        df, config, "event_columns.rating", ["rating"]
    )
    if temp_col is None or rating_col is None:
        return {}

    temperature, rating, valid_mask = _prepare_temperature_rating_data(
        df, temp_col, rating_col
    )
    num_valid_points = int(valid_mask.sum())
    if num_valid_points < _MIN_VALID_POINTS:
        return {}

    best_model, best_breakpoint = _load_model_metadata(stats_dir)

    fig = _create_temperature_plot(
        temperature,
        rating,
        valid_mask,
        temp_col,
        rating_col,
        subject,
        task,
        best_model,
        best_breakpoint,
    )

    out_path = out_dir / f"sub-{subject}_temperature_models.png"
    save_fig(fig, out_path, logger=logger)
    plt.close(fig)
    logger.info("Saved temperature model plot: %s", out_path.name)
    return {"temperature_models": out_path}


__all__ = ["plot_temperature_models"]
