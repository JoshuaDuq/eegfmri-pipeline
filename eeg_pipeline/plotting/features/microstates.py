"""
Microstate Visualization
========================

Condition-comparison plots for EEG microstate dynamics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.features.utils import get_named_segments
from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.utils.analysis.events import extract_comparison_mask
from eeg_pipeline.utils.config.loader import get_config_value


_METRIC_PREFIXES = ("coverage", "duration_ms", "occurrence_hz")


def _select_segment(features_df: pd.DataFrame, config: Any) -> Optional[str]:
    segments = get_named_segments(features_df, group="microstates")
    if not segments:
        return None

    preferred = str(get_config_value(config, "plotting.comparisons.comparison_segment", "")).strip()
    if preferred and preferred in segments:
        return preferred
    non_baseline = [segment for segment in segments if str(segment).strip().lower() != "baseline"]
    if non_baseline:
        return non_baseline[0]
    return segments[0]


def _collect_metric_columns(
    features_df: pd.DataFrame,
    segment: str,
) -> Dict[str, Dict[str, str]]:
    collected: Dict[str, Dict[str, str]] = {m: {} for m in _METRIC_PREFIXES}

    for column in features_df.columns:
        parsed = NamingSchema.parse(str(column))
        if not parsed.get("valid"):
            continue
        if parsed.get("group") != "microstates":
            continue
        if parsed.get("scope") != "global":
            continue
        if parsed.get("segment") != segment:
            continue

        stat = str(parsed.get("stat") or "")
        for metric in _METRIC_PREFIXES:
            if stat.startswith(metric + "_"):
                label = stat.replace(metric + "_", "", 1)
                if label:
                    collected[metric][label] = str(column)
                break

    return {metric: by_label for metric, by_label in collected.items() if by_label}


def _comparison_masks(events_df: pd.DataFrame, config: Any) -> Optional[Tuple[np.ndarray, np.ndarray, str, str]]:
    comparison = extract_comparison_mask(events_df, config, require_enabled=False)
    if comparison is not None:
        return comparison

    # Fallback to binary_outcome if explicit comparison config is missing.
    candidate_columns = ("binary_outcome", "binary_outcome_coded")
    for column in candidate_columns:
        if column not in events_df.columns:
            continue
        values = pd.to_numeric(events_df[column], errors="coerce")
        mask1 = (values == 0).to_numpy()
        mask2 = (values == 1).to_numpy()
        if np.any(mask1) and np.any(mask2):
            return mask1, mask2, "condition_1", "condition_2"
    return None


def _plot_metric(
    metric: str,
    by_label: Dict[str, str],
    features_df: pd.DataFrame,
    mask1: np.ndarray,
    mask2: np.ndarray,
    label1: str,
    label2: str,
    subject: str,
    save_dir: Path,
    logger: Any,
    config: Any,
) -> None:
    plot_cfg = get_plot_config(config)
    ordered_labels = sorted(by_label.keys())
    if not ordered_labels:
        return

    fig, ax = plt.subplots(figsize=(max(6.5, 1.6 * len(ordered_labels)), 5.0))
    x_positions = np.arange(len(ordered_labels), dtype=float)
    width = 0.32

    vals_1: List[np.ndarray] = []
    vals_2: List[np.ndarray] = []
    for label in ordered_labels:
        col = by_label[label]
        series = pd.to_numeric(features_df[col], errors="coerce")
        v1 = series[mask1].dropna().to_numpy()
        v2 = series[mask2].dropna().to_numpy()
        vals_1.append(v1)
        vals_2.append(v2)

    b1 = ax.boxplot(
        vals_1,
        positions=x_positions - width / 2.0,
        widths=width,
        patch_artist=True,
        showfliers=False,
    )
    b2 = ax.boxplot(
        vals_2,
        positions=x_positions + width / 2.0,
        widths=width,
        patch_artist=True,
        showfliers=False,
    )

    color_1 = plot_cfg.get_color("condition_1")
    color_2 = plot_cfg.get_color("condition_2")
    for patch in b1["boxes"]:
        patch.set_facecolor(color_1)
        patch.set_alpha(0.75)
    for patch in b2["boxes"]:
        patch.set_facecolor(color_2)
        patch.set_alpha(0.75)

    y_max = float(np.nanmax([np.nanmax(v) if len(v) else np.nan for v in vals_1 + vals_2]))
    y_min = float(np.nanmin([np.nanmin(v) if len(v) else np.nan for v in vals_1 + vals_2]))
    if not np.isfinite(y_max) or not np.isfinite(y_min):
        plt.close(fig)
        return
    y_span = max(1e-9, y_max - y_min)

    for idx, (v1, v2) in enumerate(zip(vals_1, vals_2)):
        if len(v1) < 3 or len(v2) < 3:
            continue
        try:
            _, p_value = mannwhitneyu(v1, v2, alternative="two-sided")
        except ValueError:
            continue
        p_text = f"p={p_value:.3f}" if p_value >= 0.001 else "p<0.001"
        ax.text(
            x_positions[idx],
            y_max + (0.06 + 0.05 * idx) * y_span,
            p_text,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([label.upper() for label in ordered_labels])
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("Microstate Class")
    ax.set_title(f"Sub-{subject} Microstates: {metric.replace('_', ' ').title()}")
    ax.legend([b1["boxes"][0], b2["boxes"][0]], [label1, label2], loc="best")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    save_path = save_dir / f"sub-{subject}_microstates_{metric}_by_condition"
    save_fig(fig, save_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi)
    plt.close(fig)

    if logger is not None:
        logger.info("Saved microstate %s condition comparison: %s", metric, save_path)


def plot_microstates_by_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: Any,
    config: Any = None,
    *,
    stats_dir: Optional[Path] = None,  # kept for API consistency
) -> None:
    """Plot microstate metrics by condition."""
    _ = stats_dir
    if features_df is None or features_df.empty or events_df is None or events_df.empty:
        return

    ensure_dir(save_dir)

    segment = _select_segment(features_df, config)
    if segment is None:
        return

    metric_columns = _collect_metric_columns(features_df, segment)
    if not metric_columns:
        return

    comparison = _comparison_masks(events_df, config)
    if comparison is None:
        if logger is not None:
            logger.warning("Microstates plotting: no valid condition comparison found; skipping.")
        return
    mask1, mask2, label1, label2 = comparison

    if len(mask1) != len(features_df) or len(mask2) != len(features_df):
        n = min(len(features_df), len(mask1), len(mask2))
        mask1 = mask1[:n]
        mask2 = mask2[:n]
        features_df = features_df.iloc[:n].reset_index(drop=True)

    for metric, by_label in metric_columns.items():
        _plot_metric(
            metric=metric,
            by_label=by_label,
            features_df=features_df,
            mask1=mask1,
            mask2=mask2,
            label1=label1,
            label2=label2,
            subject=subject,
            save_dir=save_dir,
            logger=logger,
            config=config,
        )


__all__ = ["plot_microstates_by_condition"]
