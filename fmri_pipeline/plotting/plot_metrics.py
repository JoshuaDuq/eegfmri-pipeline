from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .plot_utils import FigureSpec, save_figure, subject_palette, _add_panel_label, load_group_metrics
from utils import get_log_function

log, _ = get_log_function(Path(__file__).stem)


###################################################################
# Subject Metric Panels
###################################################################


def plot_subject_metric_panels(
    metrics_df: pd.DataFrame,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
    signature: str = "nps"
) -> Optional[FigureSpec]:
    if metrics_df.empty:
        return None

    subjects = metrics_df["subject"].unique().tolist()
    palette = subject_palette(subjects)
    
    group_metrics = {}
    group_dir = figure_dir.parent.parent / "group" / signature
    if group_dir.exists():
        group_metrics = load_group_metrics(group_dir)
    
    metric_specs = [
        {"column": "slope_BR_temp", "label": "Slope BR~Temp (a.u./°C)", "baseline": 0.0, "group_key": "slope_BR_temp"},
        {"column": "r_BR_temp", "label": "r(BR, Temp)", "baseline": 0.0, "group_key": "r_BR_temp", "xlim": (-1.0, 1.0)},
        {"column": "r_BR_VAS", "label": "r(BR, VAS)", "baseline": 0.0, "group_key": "r_BR_VAS", "xlim": (-1.0, 1.0)},
        {"column": "auc_pain", "label": "Pain classification AUC", "baseline": 0.5, "group_key": "auc_pain", "xlim": (0.0, 1.0)},
        {
            "column": "forced_choice_accuracy",
            "label": "Forced-choice accuracy",
            "baseline": 0.5,
            "group_key": "forced_choice_acc",
            "xlim": (0.0, 1.0),
        },
    ]

    n_panels = len(metric_specs)
    ncols = 3
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7.0, 1.8 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for axis in axes[n_panels:]:
        axis.axis("off")

    subject_order = subjects
    y_positions = np.arange(len(subject_order))

    for panel_idx, (axis, spec) in enumerate(zip(axes, metric_specs)):
        _add_panel_label(axis, chr(65 + panel_idx), x=-0.18, y=1.05)
        
        series = (
            metrics_df[["subject", spec["column"]]]
            .dropna(subset=[spec["column"]])
            .set_index("subject")
            .reindex(subject_order)
        )

        if series.dropna().empty:
            axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes, fontsize=11)
            axis.set_title(spec["label"], fontweight="bold")
            axis.set_yticks(y_positions)
            axis.set_yticklabels(subject_order)
            continue

        for subject_idx, subject in enumerate(subject_order):
            if pd.isna(series.loc[subject][spec["column"]]):
                continue
            value = series.loc[subject][spec["column"]]
            axis.scatter(
                value,
                subject_idx,
                color=palette.get(subject, "0.3"),
                s=40,
                linewidth=0,
                zorder=3,
            )

        if "baseline" in spec and spec["baseline"] is not None:
            axis.axvline(spec["baseline"], color="0.7", linestyle="-", linewidth=0.5, alpha=0.5, zorder=1)

        group_key = spec.get("group_key")
        if group_key and group_key in group_metrics and "mean" in group_metrics[group_key]:
            gm = group_metrics[group_key]
            mean = gm.get("mean")
            lower = gm.get("ci_lower")
            upper = gm.get("ci_upper")
            if mean is not None:
                axis.axvline(mean, color="0.2", linewidth=1.2, alpha=0.8, zorder=2)
            if lower is not None and upper is not None and lower <= upper:
                axis.fill_betweenx(
                    y=[-0.5, len(subject_order) - 0.5],
                    x1=lower,
                    x2=upper,
                    color="0.2",
                    alpha=0.08,
                    linewidth=0,
                    zorder=2,
                )

        axis.set_title(spec["label"], fontsize=9, fontweight="normal")
        if spec.get("xlim"):
            axis.set_xlim(spec["xlim"])
        axis.set_yticks(y_positions)
        axis.set_yticklabels(subject_order, fontsize=8)
        sns.despine(ax=axis, trim=True)

    fig_name = f"Fig4_{signature.upper()}_SubjectMetrics"
    paths = save_figure(fig, fig_name, figure_dir, formats, dpi)
    
    return FigureSpec(
        name=fig_name,
        description=f"Individual subject performance across five key metrics (panels A-E): slope of {signature.upper()}-temperature relationship, correlation with temperature, correlation with VAS, pain classification AUC, and forced-choice accuracy.",
        paths=paths,
        stats_paths=[],
    )


###################################################################
# ROC Curves
###################################################################


def plot_subject_roc_curves(
    trial_df: pd.DataFrame,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
    signature: str = "nps"
) -> Optional[FigureSpec]:
    if trial_df.empty:
        return None

    try:
        from sklearn.metrics import auc as sklearn_auc
        from sklearn.metrics import roc_curve
    except ImportError:
        return None

    palette = subject_palette(sorted(trial_df["subject"].unique()))
    pain_threshold = 100.0
    
    curves: List[Tuple[str, np.ndarray, np.ndarray, float, Tuple[float, float]]] = []
    for subject, sub_df in trial_df.groupby("subject"):
        valid = sub_df.dropna(subset=["br_score"])
        if valid.empty:
            continue
        if "pain_binary" in valid.columns and valid["pain_binary"].notna().any():
            labels = valid["pain_binary"].astype(int)
        elif "vas_rating" in valid.columns:
            labels = (valid["vas_rating"] > pain_threshold).astype(int)
        else:
            continue

        if labels.nunique() < 2:
            continue

        try:
            fpr, tpr, _ = roc_curve(labels, valid["br_score"])
            auc_value = sklearn_auc(fpr, tpr)
            
            n_boot = 500
            boot_aucs = []
            for _ in range(n_boot):
                boot_idx = np.random.choice(len(labels), len(labels), replace=True)
                if labels.iloc[boot_idx].nunique() < 2:
                    continue
                try:
                    boot_fpr, boot_tpr, _ = roc_curve(labels.iloc[boot_idx], valid["br_score"].iloc[boot_idx])
                    boot_aucs.append(sklearn_auc(boot_fpr, boot_tpr))
                except Exception:
                    continue
            
            if boot_aucs:
                auc_ci = tuple(np.percentile(boot_aucs, [2.5, 97.5]))
            else:
                auc_ci = (np.nan, np.nan)
                
        except Exception:
            continue
        curves.append((subject, fpr, tpr, auc_value, auc_ci))

    if not curves:
        return None

    fig, axis = plt.subplots(figsize=(2.8, 2.8), constrained_layout=True)
    axis.plot([0, 1], [0, 1], color="0.7", linestyle="-", linewidth=0.8, alpha=0.5, zorder=1)
    
    for subject, fpr, tpr, auc_value, auc_ci in curves:
        label_text = f"{subject} ({auc_value:.2f})"
        axis.plot(
            fpr,
            tpr,
            color=palette.get(subject, "0.3"),
            linewidth=1.5,
            label=label_text,
            zorder=2,
        )
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_aspect("equal", "box")
    axis.legend(loc="lower right", fontsize=7, title="AUC", title_fontsize=7)
    sns.despine(ax=axis, trim=True)

    fig_name = f"Fig5_{signature.upper()}_PainROC"
    paths = save_figure(fig, fig_name, figure_dir, formats, dpi)
    
    return FigureSpec(
        name=fig_name,
        description=f"Receiver operating characteristic (ROC) curves for pain vs. non-pain classification at the trial level using {signature.upper()}.",
        paths=paths,
        stats_paths=[],
    )


###################################################################
# Temperature Distributions
###################################################################


def plot_trial_temperature_distributions(
    trial_df: pd.DataFrame,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
    signature: str = "nps"
) -> Optional[FigureSpec]:
    required_cols = {"temp_celsius", "br_score", "subject"}
    if not required_cols.issubset(trial_df.columns):
        return None

    data = trial_df.dropna(subset=["temp_celsius", "br_score"])
    if data.empty:
        return None

    palette = subject_palette(sorted(data["subject"].unique()))

    unique_temps = np.sort(data["temp_celsius"].unique())
    temp_labels = [f"{temp:.1f}°C" for temp in unique_temps]
    label_map = dict(zip(unique_temps, temp_labels))
    data = data.copy()
    data["temp_label"] = data["temp_celsius"].map(label_map)

    q01, q99 = data["br_score"].quantile([0.01, 0.99])
    y_range = q99 - q01
    y_min_robust = q01 - 0.1 * y_range
    y_max_robust = q99 + 0.1 * y_range
    
    outliers = data[(data["br_score"] < y_min_robust) | (data["br_score"] > y_max_robust)]

    fig, axis = plt.subplots(figsize=(5.0, 3.0), constrained_layout=True)
    
    sns.violinplot(
        data=data,
        x="temp_label",
        y="br_score",
        order=temp_labels,
        color="0.85",
        inner=None,
        cut=0,
        linewidth=0.5,
        ax=axis,
        saturation=1,
    )
    
    sns.stripplot(
        data=data,
        x="temp_label",
        y="br_score",
        order=temp_labels,
        hue="subject",
        dodge=False,
        alpha=0.5,
        size=2.5,
        linewidth=0,
        palette=palette,
        ax=axis,
        legend=False,
    )
    axis.axhline(0, color="0.6", linestyle="-", linewidth=0.5, alpha=0.5)
    axis.set_xlabel("Temperature (°C)")
    axis.set_ylabel(f"{signature.upper()} response (a.u.)")
    sns.despine(ax=axis, trim=True)
    axis.set_xticks(range(len(temp_labels)))
    temp_values = [float(label.replace("°C", "")) for label in temp_labels]
    axis.set_xticklabels([f"{val:.0f}" for val in temp_values], fontsize=8)
    axis.tick_params(axis="x", rotation=0)
    
    axis.set_ylim(y_min_robust, y_max_robust)
    
    if len(outliers) > 0:
        n_total = len(data)
        pct_outliers = 100 * len(outliers) / n_total
        axis.text(
            0.98, 0.02,
            f"{len(outliers)} outliers excluded ({pct_outliers:.1f}%)",
            transform=axis.transAxes,
            fontsize=6.5,
            va="bottom",
            ha="right",
            color="0.4",
            style="italic",
        )
    
    for temp_idx, temp_label in enumerate(temp_labels):
        temp_data = data[data["temp_label"] == temp_label]["br_score"]
        if len(temp_data) >= 3:
            median_val = temp_data.median()
            axis.plot(
                [temp_idx - 0.3, temp_idx + 0.3],
                [median_val, median_val],
                color="0.2",
                linewidth=1.0,
                zorder=10,
                alpha=0.8
            )

    fig_name = f"Fig6_{signature.upper()}_TemperatureDistributions"
    paths = save_figure(fig, fig_name, figure_dir, formats, dpi)
    
    return FigureSpec(
        name=fig_name,
        description=f"Trial-level {signature.upper()} response distributions by temperature condition.",
        paths=paths,
        stats_paths=[],
    )
