#!/usr/bin/env python3

"""
Generate publication-quality visualizations for the NPS pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)
import seaborn as sns  # noqa: E402

from utils import load_config, get_log_function, PipelinePaths


SCRIPT_NAME = Path(__file__).stem
log, LOG_FILE = get_log_function(SCRIPT_NAME)

EXTREME_TRIAL_THRESHOLD = 1000.0  # Exclude extreme artifacts (typical NPS range: -25 to +25)


@dataclass(frozen=True)
class FigureSpec:
    name: str
    description: str
    paths: List[str]
    stats_paths: List[str] = field(default_factory=list)


def rename_figure_for_signature(fig_spec: Optional[FigureSpec], signature: str) -> Optional[FigureSpec]:
    """Rename figure to use signature-specific naming."""
    if fig_spec is None:
        return None
    
    sig_upper = signature.upper()
    # Replace "NPS" in name and description with the actual signature
    new_name = fig_spec.name.replace("_NPS_", f"_{sig_upper}_").replace("NPS", sig_upper)
    new_description = fig_spec.description.replace("NPS", sig_upper)
    
    # Update file paths to match new name
    new_paths = []
    for path in fig_spec.paths:
        path_obj = Path(path)
        # Replace in filename only
        new_filename = path_obj.name.replace("_NPS_", f"_{sig_upper}_").replace("NPS", sig_upper)
        new_path = path_obj.parent / new_filename
        # Rename the actual file if it exists
        if path_obj.exists():
            # Use replace() which handles overwrites atomically on all platforms
            path_obj.replace(new_path)
        new_paths.append(str(new_path))

    new_stats_paths = []
    for path in fig_spec.stats_paths:
        path_obj = Path(path)
        new_filename = path_obj.name.replace("_NPS_", f"_{sig_upper}_").replace("NPS", sig_upper)
        new_path = path_obj.parent / new_filename
        if path_obj.exists():
            path_obj.replace(new_path)
        new_stats_paths.append(str(new_path))

    return FigureSpec(name=new_name, description=new_description, paths=new_paths, stats_paths=new_stats_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate summary figures for pain signature behavioral metrics (NPS, SIIPS1, etc.)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="utils/config.yaml", help="Pipeline configuration file")
    parser.add_argument("--scores-dir", default="outputs", help="Base directory for signature scores (e.g., outputs/nps_scores, outputs/siips1_scores)")
    parser.add_argument("--group-dir", default="outputs/group", help="Base directory for group statistics")
    parser.add_argument("--figures-dir", default="outputs/figures", help="Base directory for figure outputs")
    parser.add_argument("--signatures", nargs="+", default=None, help="Signatures to plot (e.g., nps siips1). If not specified, uses config.")
    parser.add_argument("--formats", default="svg,png", help="Comma-separated list of output formats (e.g., svg,png)")
    parser.add_argument("--dpi", type=int, default=300, help="Figure resolution for raster outputs")
    parser.add_argument(
        "--pain-threshold",
        type=float,
        default=100.0,
        help="VAS threshold separating pain from non-pain when labels are unavailable",
    )
    parser.add_argument(
        "--trial-outlier-threshold",
        type=float,
        default=EXTREME_TRIAL_THRESHOLD,
        help="Absolute signature value above which trial-level responses are excluded as extreme outliers",
    )
    parser.add_argument(
        "--style-context",
        default="talk",
        choices=["paper", "notebook", "talk", "poster"],
        help="Seaborn context preset for figure styling",
    )
    return parser.parse_args()


def configure_matplotlib(context: str = "paper") -> None:
    """Configure matplotlib for Nature-style publication figures."""
    sns.set_theme(style="ticks", context=context)
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "0.15",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.titlesize": 11,
            "axes.titleweight": "normal",
            "axes.titlepad": 10,
            "axes.labelsize": 9,
            "axes.labelweight": "normal",
            "axes.labelpad": 4,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            "patch.linewidth": 0.5,
            "grid.alpha": 0.0,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_formats(formats: str) -> List[str]:
    resolved = [fmt.strip().lower() for fmt in formats.split(",") if fmt.strip()]
    return resolved or ["svg"]


def save_stats_csv(
    data: Any,
    figure_dir: Path,
    fig_name: str,
    include_index: bool = False,
) -> str:
    """
    Persist statistics associated with a figure as CSV.

    Accepts either a pandas DataFrame or any iterable of mappings. Empty inputs
    are converted to a placeholder table to keep downstream tooling consistent.
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        try:
            df = pd.DataFrame(list(data))
        except Exception:
            df = pd.DataFrame()

    if df.empty:
        df = pd.DataFrame({"note": ["No statistics available"]})

    stats_path = figure_dir / f"{fig_name}_stats.csv"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stats_path, index=include_index)
    return str(stats_path)


def attach_basic_stats(
    fig_spec: Optional[FigureSpec],
    figure_dir: Path,
    stats_sources: Dict[str, Optional[pd.DataFrame]],
    extra_rows: Optional[pd.DataFrame] = None,
) -> Optional[FigureSpec]:
    """
    Attach a default descriptive-statistics CSV to a figure when plot-specific
    functions have not generated custom outputs.
    """
    if fig_spec is None:
        return None
    if fig_spec.stats_paths:
        return fig_spec

    frames: List[pd.DataFrame] = []
    for scope, df in stats_sources.items():
        if df is None or df.empty:
            frames.append(pd.DataFrame({"stat_scope": [scope], "note": ["No data available"]}))
            continue
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            frames.append(pd.DataFrame({"stat_scope": [scope], "note": ["No numeric columns"]}))
            continue
        summary = numeric.describe().T.reset_index().rename(columns={"index": "metric"})
        summary.insert(0, "stat_scope", scope)
        frames.append(summary)

    if extra_rows is not None and not extra_rows.empty:
        frames.append(extra_rows)

    if not frames:
        frames.append(pd.DataFrame({"stat_scope": ["overall"], "note": ["No statistics computed"]}))

    stats_df = pd.concat(frames, ignore_index=True, sort=False)
    stats_path = save_stats_csv(stats_df, figure_dir, fig_spec.name)

    return FigureSpec(
        name=fig_spec.name,
        description=fig_spec.description,
        paths=fig_spec.paths,
        stats_paths=[stats_path],
    )


def append_figure_with_stats(
    figures: List[FigureSpec],
    fig_spec: Optional[FigureSpec],
    figure_dir: Path,
    stats_sources: Dict[str, Optional[pd.DataFrame]],
    extra_rows: Optional[pd.DataFrame] = None,
) -> None:
    """Attach basic stats to a figure (if needed) and append to list."""
    if fig_spec is None:
        return
    enriched = attach_basic_stats(fig_spec, figure_dir, stats_sources, extra_rows)
    figures.append(enriched if enriched is not None else fig_spec)


def load_subject_metrics(scores_dir: Path, subjects: Iterable[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for subject in subjects:
        metrics_path = scores_dir / subject / "subject_metrics.tsv"
        if not metrics_path.exists():
            log(f"Missing subject_metrics.tsv for {subject}", "WARNING")
            continue
        try:
            df = pd.read_csv(metrics_path, sep="\t")
            df["subject"] = subject
            frames.append(df)
        except Exception as exc:  # pragma: no cover - defensive
            log(f"Failed to load {metrics_path}: {exc}", "WARNING")
    if not frames:
        return pd.DataFrame()
    metrics = pd.concat(frames, ignore_index=True)
    numeric_cols = metrics.select_dtypes(include="object").columns.difference(["subject", "notes"])
    for col in numeric_cols:
        metrics[col] = pd.to_numeric(metrics[col], errors="coerce")
    return metrics


def load_level_br(scores_dir: Path, subjects: Iterable[str], signature: str = "nps") -> pd.DataFrame:
    """Load condition-level signature scores.
    
    Parameters
    ----------
    scores_dir : Path
        Base directory (e.g., outputs/nps_scores or outputs/siips1_scores)
    subjects : Iterable[str]
        Subject IDs
    signature : str
        Signature name (e.g., 'nps', 'siips1')
    """
    frames: List[pd.DataFrame] = []
    filename = f"level_{signature}.tsv"
    for subject in subjects:
        level_path = scores_dir / subject / filename
        if not level_path.exists():
            # Try legacy name for backward compatibility
            if signature == "nps":
                legacy_path = scores_dir / subject / "level_br.tsv"
                if legacy_path.exists():
                    level_path = legacy_path
                else:
                    log(f"Missing {filename} for {subject}", "WARNING")
                    continue
            else:
                log(f"Missing {filename} for {subject}", "WARNING")
                continue
        try:
            df = pd.read_csv(level_path, sep="\t")
            df["subject"] = subject
            # Rename signature-specific column to generic 'br_score' for plotting
            score_col = f"{signature}_score"
            if score_col in df.columns and "br_score" not in df.columns:
                df["br_score"] = df[score_col]
            frames.append(df)
        except Exception as exc:  # pragma: no cover
            log(f"Failed to load {level_path}: {exc}", "WARNING")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_trial_br(scores_dir: Path, subjects: Iterable[str], signature: str = "nps") -> pd.DataFrame:
    """Load trial-level signature scores.
    
    Parameters
    ----------
    scores_dir : Path
        Base directory (e.g., outputs/nps_scores or outputs/siips1_scores)
    subjects : Iterable[str]
        Subject IDs
    signature : str
        Signature name (e.g., 'nps', 'siips1')
    """
    frames: List[pd.DataFrame] = []
    filename = f"trial_{signature}.tsv"
    for subject in subjects:
        trial_path = scores_dir / subject / filename
        if not trial_path.exists():
            # Try legacy name for backward compatibility
            if signature == "nps":
                legacy_path = scores_dir / subject / "trial_br.tsv"
                if legacy_path.exists():
                    trial_path = legacy_path
                else:
                    log(f"Missing {filename} for {subject}", "INFO")
                    continue
            else:
                log(f"Missing {filename} for {subject}", "INFO")
                continue
        try:
            df = pd.read_csv(trial_path, sep="\t")
            df["subject"] = subject
            # Rename signature-specific column to generic 'br_score' for plotting
            score_col = f"{signature}_score"
            if score_col in df.columns and "br_score" not in df.columns:
                df["br_score"] = df[score_col]
            frames.append(df)
        except Exception as exc:  # pragma: no cover
            log(f"Failed to load {trial_path}: {exc}", "WARNING")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_group_metrics(group_dir: Path) -> Dict[str, Dict[str, float]]:
    verbose_path = group_dir / "group_stats_verbose.json"
    if not verbose_path.exists():
        log(f"Group statistics not found at {verbose_path}", "INFO")
        return {}
    try:
        data = json.loads(verbose_path.read_text())
    except Exception as exc:  # pragma: no cover
        log(f"Failed to parse {verbose_path}: {exc}", "WARNING")
        return {}
    return data.get("metrics", {})


def aggregate_trials_to_conditions(trial_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trial-level data to condition-level format for comparison."""
    if trial_df.empty:
        return pd.DataFrame()
    
    required_cols = {"subject", "temp_celsius", "br_score"}
    if not required_cols.issubset(trial_df.columns):
        return pd.DataFrame()
    
    # Aggregate by subject and temperature
    agg_dict = {
        "br_score": "mean",
        "n_trials_removed_run": "sum",
    }
    
    # Optional columns
    if "vas_rating" in trial_df.columns:
        agg_dict["vas_rating"] = "mean"
    if "pain_binary" in trial_df.columns:
        agg_dict["pain_binary"] = "mean"
    
    level_from_trials = trial_df.groupby(["subject", "temp_celsius"]).agg(agg_dict).reset_index()
    
    # Rename for consistency with level_br.tsv format
    if "vas_rating" in level_from_trials.columns:
        level_from_trials.rename(columns={"vas_rating": "mean_vas"}, inplace=True)
    
    # Add trial count
    trial_counts = trial_df.groupby(["subject", "temp_celsius"]).size().reset_index(name="n_trials")
    level_from_trials = level_from_trials.merge(trial_counts, on=["subject", "temp_celsius"], how="left")
    
    return level_from_trials


def filter_extreme_trials(
    trial_df: pd.DataFrame,
    threshold: float = EXTREME_TRIAL_THRESHOLD,
) -> Tuple[pd.DataFrame, int, pd.DataFrame]:
    if trial_df.empty or "br_score" not in trial_df.columns:
        return trial_df, 0, pd.DataFrame()
    mask = trial_df["br_score"].abs() > threshold
    removed = trial_df.loc[mask].copy()
    if removed.empty:
        return trial_df, 0, removed
    cleaned = trial_df.loc[~mask].copy()
    removed_details = removed[["subject", "run", "trial_regressor", "temp_celsius", "br_score"]].sort_values(
        ["subject", "run", "trial_regressor"]
    )
    if "run" in removed_details.columns:
        removed_details["run"] = pd.to_numeric(removed_details["run"], errors="coerce")
    for col in ("temp_celsius", "br_score"):
        if col in removed_details.columns:
            removed_details[col] = pd.to_numeric(removed_details[col], errors="coerce")
    log(
        f"Excluded {len(removed_details)} trial(s) with |NPS| > {threshold:.1f} (extreme outliers).",
        "WARNING",
    )
    for _, row in removed_details.iterrows():
        log(
            f"  Removed {row['subject']} run-{int(row['run']):02d} {row['trial_regressor']}: "
            f"NPS={row['br_score']:.3f}, temp={row['temp_celsius']:.2f} C",
            "WARNING",
        )
    return cleaned, len(removed_details), removed_details


def save_figure(
    fig: plt.Figure,
    base_name: str,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[str]:
    fig.patch.set_facecolor("white")
    for axis in fig.axes:
        axis.set_facecolor("white")
    paths: List[str] = []
    for ext in formats:
        out_path = figure_dir / f"{base_name}.{ext}"
        fig.savefig(
            out_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        paths.append(str(out_path))
    plt.close(fig)
    return paths


def subject_palette(subjects: Sequence[str]) -> Dict[str, Tuple[float, float, float]]:
    """Generate muted, professional color palette for subjects."""
    if not subjects:
        return {}
    colors = [
        (0.30, 0.45, 0.69),
        (0.87, 0.56, 0.36),
        (0.47, 0.71, 0.48),
        (0.91, 0.54, 0.76),
        (0.60, 0.60, 0.60),
        (0.98, 0.75, 0.36),
        (0.55, 0.83, 0.78),
        (0.91, 0.33, 0.32),
    ]
    return {subject: colors[idx % len(colors)] for idx, subject in enumerate(subjects)}


def _add_panel_label(axis: plt.Axes, label: str, x: float = -0.12, y: float = 1.05) -> None:
    """Add panel label (A, B, C, etc.) to axis."""
    axis.text(
        x, y, label,
        transform=axis.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right"
    )


def _format_pvalue(p: float) -> str:
    """Format p-value with appropriate precision."""
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.2f}"


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)


def _compute_correlation_with_ci(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> dict:
    """Compute correlation with bootstrap CI and both Pearson and Spearman."""
    if method == "pearson":
        r, p = stats.pearsonr(x, y)
    else:
        r, p = stats.spearmanr(x, y)
    
    n_boot = 1000
    boot_rs = []
    n = len(x)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        if method == "pearson":
            boot_r, _ = stats.pearsonr(x[idx], y[idx])
        else:
            boot_r, _ = stats.spearmanr(x[idx], y[idx])
        boot_rs.append(boot_r)
    
    ci_lower, ci_upper = np.percentile(boot_rs, [2.5, 97.5])
    
    return {
        "r": r,
        "p": p,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n": n,
    }


def _safe_sem(data: pd.Series) -> float:
    """Calculate SEM with small-sample warning."""
    data = data.dropna()
    if len(data) < 2:
        return np.nan
    return stats.sem(data)


def plot_subject_dose_response(
    level_df: pd.DataFrame,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
    palette: Dict[str, Tuple[float, float, float]],
    data_type: str = "condition",  # "condition" or "trial"
    signature: str = "NPS",  # Signature name for figure titles
) -> Optional[FigureSpec]:
    required_cols = {"temp_celsius", "br_score", "subject"}
    if not required_cols.issubset(level_df.columns):
        log("Condition-level data missing required columns; skipping dose-response plots", "WARNING")
        return None

    data = level_df.dropna(subset=["temp_celsius", "br_score"])
    if data.empty:
        log("No condition-level data available for dose-response plots", "WARNING")
        return None

    subjects = sorted(data["subject"].unique())
    stats_records: List[Dict[str, Any]] = []
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
        record: Dict[str, Any] = {
            "signature": signature,
            "data_type": data_type,
            "subject": subject,
            "n_points": int(len(sub_df)),
            "temp_min": float(sub_df["temp_celsius"].min()) if not sub_df.empty else np.nan,
            "temp_max": float(sub_df["temp_celsius"].max()) if not sub_df.empty else np.nan,
            "br_mean": float(sub_df["br_score"].mean()) if not sub_df.empty else np.nan,
            "br_std": float(sub_df["br_score"].std(ddof=0)) if len(sub_df) > 1 else np.nan,
        }
        
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
                
                axis.text(
                    0.98,
                    0.02,
                    stats_text,
                    transform=axis.transAxes,
                    fontsize=7,
                    va="bottom",
                    ha="right",
                    color="0.3",
                )
                record.update(
                    {
                        "slope": float(fit.slope),
                        "intercept": float(fit.intercept),
                        "r_value": float(fit.rvalue),
                        "r_squared": float(fit.rvalue ** 2),
                        "p_value": float(fit.pvalue),
                        "stderr": float(fit.stderr),
                    }
                )
            except Exception as exc:  # pragma: no cover
                log(f"Linregress failed for {subject}: {exc}", "WARNING")
                record.update(
                    {
                        "slope": np.nan,
                        "intercept": np.nan,
                        "r_value": np.nan,
                        "r_squared": np.nan,
                        "p_value": np.nan,
                        "stderr": np.nan,
                    }
                )
        else:
            record.update(
                {
                    "slope": np.nan,
                    "intercept": np.nan,
                    "r_value": np.nan,
                    "r_squared": np.nan,
                    "p_value": np.nan,
                    "stderr": np.nan,
                }
            )
        axis.axhline(0, color="0.7", linewidth=0.5, linestyle="-", alpha=0.8, zorder=1)
        axis.set_title(subject, fontsize=10, fontweight="normal")
        if idx % ncols == 0:
            ylabel = f"{signature} (a.u.)" if data_type == "trial" else f"{signature} (a.u.)\n[cond-level]"
            axis.set_ylabel(ylabel, fontsize=8.5)
        if idx >= (nrows - 1) * ncols:
            axis.set_xlabel("Temperature (°C)")
        sns.despine(ax=axis, trim=True)
        stats_records.append(record)

    # No suptitle - cleaner look
    # Determine filename based on data type
    if data_type == "condition":
        fig_name = f"Fig1a_{signature}_DoseResponse_BySubject_ConditionLevel"
        description_suffix = "CONDITION-LEVEL GLM: betas per temperature condition"
    else:
        fig_name = f"Fig1b_{signature}_DoseResponse_BySubject_TrialLevel"
        description_suffix = "TRIAL-LEVEL LSS GLM: betas per individual trial"
    
    paths = save_figure(fig, fig_name, figure_dir, formats, dpi)
    stats_path = save_stats_csv(stats_records, figure_dir, fig_name)
    return FigureSpec(
        name=fig_name,
        description=f"Subject-specific {signature} dose-response curves ({description_suffix}). Each panel shows one subject with linear regression line and 95% CI (shaded). Correlation statistics (r², p-value) quantify monotonic stimulus-response relationship.",
        paths=paths,
        stats_paths=[stats_path],
    )


def plot_group_dose_response(
    level_df: pd.DataFrame,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
    palette: Dict[str, Tuple[float, float, float]],
    data_type: str = "condition",  # "condition" or "trial"
    trial_df_raw: Optional[pd.DataFrame] = None,  # Raw trial data for overlay
    signature: str = "NPS",  # Signature name for figure titles
) -> Optional[FigureSpec]:
    required_cols = {"temp_celsius", "br_score", "subject"}
    if not required_cols.issubset(level_df.columns):
        log("Condition-level data missing required columns; skipping group dose-response plot", "WARNING")
        return None

    data = level_df.dropna(subset=["temp_celsius", "br_score"])
    if data.empty or data["subject"].nunique() < 1:
        log("Skipping group dose-response summary (insufficient data)", "WARNING")
        return None

    summary = (
        data.groupby("temp_celsius")
        .agg(
            mean_br=("br_score", "mean"),
            std_br=("br_score", "std"),
            n=("br_score", "count"),
            mean_vas=("mean_vas", "mean"),
        )
        .reset_index()
        .sort_values("temp_celsius")
    )
    if summary.empty:
        log("Group dose-response summary is empty after aggregation", "WARNING")
        return None

    ci = []
    for _, row in summary.iterrows():
        if row["n"] > 1 and not np.isnan(row["std_br"]):
            t_multiplier = stats.t.ppf(0.975, int(row["n"]) - 1)
            ci.append(t_multiplier * row["std_br"] / math.sqrt(row["n"]))
        else:
            ci.append(np.nan)
    summary["ci"] = ci

    temp_fit = None
    regression_summary: Dict[str, Any] = {
        "slope": np.nan,
        "intercept": np.nan,
        "r_value": np.nan,
        "r_squared": np.nan,
        "p_value": np.nan,
        "cohens_d": np.nan,
        "predicted_change": np.nan,
        "n_temperatures": int(len(summary)),
    }
    if len(summary) >= 3:
        try:
            temp_fit = stats.linregress(summary["temp_celsius"], summary["mean_br"])
            temp_range = summary["temp_celsius"].max() - summary["temp_celsius"].min()
            predicted_change = temp_fit.slope * temp_range if temp_range > 0 else np.nan
            pooled_std = np.sqrt(np.nanmean(summary["std_br"] ** 2))
            cohens_d = (
                predicted_change / pooled_std
                if np.isfinite(predicted_change) and np.isfinite(pooled_std) and pooled_std > 0
                else np.nan
            )
            regression_summary.update(
                {
                    "slope": float(temp_fit.slope),
                    "intercept": float(temp_fit.intercept),
                    "r_value": float(temp_fit.rvalue),
                    "r_squared": float(temp_fit.rvalue ** 2),
                    "p_value": float(temp_fit.pvalue),
                    "cohens_d": float(cohens_d) if np.isfinite(cohens_d) else np.nan,
                    "predicted_change": float(predicted_change)
                    if np.isfinite(predicted_change)
                    else np.nan,
                }
            )
        except Exception:
            temp_fit = None

    if temp_fit is not None:
        summary["predicted_br"] = temp_fit.intercept + temp_fit.slope * summary["temp_celsius"]
        summary["r_squared"] = regression_summary["r_squared"]
    else:
        summary["predicted_br"] = np.nan
        summary["r_squared"] = np.nan
    summary["signature"] = signature
    summary["data_type"] = data_type

    fig, axis = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
    
    # Optional: Show raw trial data as semi-transparent background
    if trial_df_raw is not None and not trial_df_raw.empty and data_type == "trial":
        required_trial_cols = {"subject", "temp_celsius", "br_score"}
        if required_trial_cols.issubset(trial_df_raw.columns):
            for subject, color in palette.items():
                sub_trials = trial_df_raw.loc[trial_df_raw["subject"] == subject]
                if not sub_trials.empty:
                    axis.scatter(
                        sub_trials["temp_celsius"],
                        sub_trials["br_score"],
                        color=color,
                        s=4,
                        alpha=0.15,
                        linewidth=0,
                        zorder=1,
                    )
    
    # Overlay condition-level means
    for subject, color in palette.items():
        sub_points = data.loc[data["subject"] == subject]
        if sub_points.empty:
            continue
        axis.scatter(
            sub_points["temp_celsius"],
            sub_points["br_score"],
            color=color,
            s=25 if trial_df_raw is not None else 15,
            alpha=0.7 if trial_df_raw is not None else 0.4,
            linewidth=0.5,
            edgecolors="white" if trial_df_raw is not None else "none",
            zorder=3 if trial_df_raw is not None else 2,
        )
    
    axis.fill_between(
        summary["temp_celsius"],
        summary["mean_br"] - summary["ci"],
        summary["mean_br"] + summary["ci"],
        color="0.2",
        alpha=0.1,
        linewidth=0,
        zorder=3,
    )
    axis.plot(
        summary["temp_celsius"],
        summary["mean_br"],
        marker="o",
        color="0.2",
        linewidth=1.5,
        markersize=4,
        markeredgewidth=0,
        zorder=4,
    )

    axis.axhline(0, color="0.6", linewidth=0.5, linestyle="-", alpha=0.6, zorder=1)
    axis.set_xlabel("Temperature (°C)")
    
    # Label depends on data type
    if data_type == "condition":
        axis.set_ylabel(f"{signature} response (a.u.)\n[Condition-level GLM]")
        axis.set_title(f"Group {signature} dose-response (condition-level)", fontsize=10, fontweight="normal", pad=8)
    else:
        axis.set_ylabel(f"{signature} response (a.u.)\n[Trial-level LSS GLM]")
        axis.set_title(f"Group {signature} dose-response (trial-level)", fontsize=10, fontweight="normal", pad=8)
    
    sns.despine(ax=axis, trim=True)

    # Enhanced statistical annotations

    if temp_fit is not None:

        p_value = regression_summary["p_value"]

        p_text = _format_pvalue(p_value) if np.isfinite(p_value) else "p = NA"

        cohens_d_val = regression_summary["cohens_d"]

        d_text = f"d = {cohens_d_val:.2f}" if np.isfinite(cohens_d_val) else "d = NA"

        r_squared = regression_summary["r_squared"]

        fit_text = (

            f"rA? = {r_squared:.2f}\n"

            f"{p_text}\n"

            f"{d_text}\n"

            f"n = {regression_summary['n_temperatures']} temps"

        )

        axis.text(

            0.98, 0.02,

            fit_text,

            transform=axis.transAxes,

            fontsize=6.5,

            va="bottom",

            ha="right",

            color="0.3",

        )

    if summary["mean_vas"].notna().any():
        twin = axis.twinx()
        twin.plot(
            summary["temp_celsius"],
            summary["mean_vas"],
            color="0.5",
            marker="s",
            markersize=3,
            linewidth=1.0,
            alpha=0.6,
            zorder=5,
        )
        twin.set_ylabel("VAS rating", color="0.5")
        twin.tick_params(axis="y", labelcolor="0.5", labelsize=8)
        twin.spines["right"].set_visible(True)
        twin.spines["right"].set_edgecolor("0.5")
        twin.spines["right"].set_linewidth(0.8)
        vmax = summary["mean_vas"].max()
        if np.isfinite(vmax) and vmax > 0:
            twin.set_ylim(0, vmax * 1.08)

    summary_stats = summary.copy()
    summary_stats.insert(0, "stat_scope", "per_temperature")
    overall_summary = regression_summary.copy()
    overall_summary.update(
        {
            "stat_scope": "overall",
            "signature": signature,
            "data_type": data_type,
        }
    )
    stats_df = pd.concat(
        [summary_stats, pd.DataFrame([overall_summary])],
        ignore_index=True,
        sort=False,
    )

    # Different file names and descriptions based on data type
    if data_type == "condition":
        fig_name = f"Fig2a_{signature}_DoseResponse_GroupSummary_ConditionLevel"
        description = f"Group-average {signature} dose-response curve (CONDITION-LEVEL GLM: betas per temperature condition). Individual subject condition means (small, colored) overlay group mean (black line) with 95% t-based CI band (shaded). Linear regression fit quantifies temperature sensitivity (r², p-value). Dual y-axis shows VAS pain ratings (gray, right axis). Note: Condition-level betas reflect cumulative response across trials."
    else:
        fig_name = f"Fig2b_{signature}_DoseResponse_GroupSummary_TrialLevel"
        description = f"Group-average {signature} dose-response curve (TRIAL-LEVEL LSS GLM: betas per individual trial). Individual subject trial-averaged data (small, colored) overlay group mean (black line) with 95% t-based CI band (shaded). Linear regression fit quantifies temperature sensitivity (r², p-value). Dual y-axis shows VAS pain ratings (gray, right axis). Note: Trial-level betas reflect single-trial responses with proper scaling."
    
    paths = save_figure(fig, fig_name, figure_dir, formats, dpi)
    stats_path = save_stats_csv(stats_df, figure_dir, fig_name)
    return FigureSpec(
        name=fig_name,
        description=description,
        paths=paths,
        stats_paths=[stats_path],
    )


def plot_vas_br_relationship(
    level_df: pd.DataFrame,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
    palette: Dict[str, Tuple[float, float, float]],
) -> Optional[FigureSpec]:
    columns = {"mean_vas", "br_score", "subject"}
    if not columns.issubset(level_df.columns):
        log("VAS columns not found; skipping VAS vs NPS plot", "WARNING")
        return None

    data = level_df.dropna(subset=["mean_vas", "br_score"])
    if data.empty:
        log("Insufficient VAS data for scatter plot", "WARNING")
        return None

    pearson_r = np.nan
    pearson_p = np.nan
    spearman_r = np.nan
    spearman_p = np.nan

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
    axis.set_ylabel("NPS response (a.u.)")
    sns.despine(ax=axis, trim=True)

    if len(data) >= 3:
        try:
            vas_arr = data["mean_vas"].values
            br_arr = data["br_score"].values
            pearson_r, pearson_p = stats.pearsonr(vas_arr, br_arr)
            spearman_r, spearman_p = stats.spearmanr(vas_arr, br_arr)
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
        except Exception as exc:  # pragma: no cover
            log(f"Correlation failed: {exc}", "WARNING")


    paths = save_figure(fig, "Fig3_VAS_vs_NPS", figure_dir, formats, dpi)
    return FigureSpec(
        name="Fig3_VAS_vs_NPS",
        description="Brain-behavior correlation between subjective pain ratings (VAS) and neurologic pain signature (NPS). Both Pearson (parametric) and Spearman (non-parametric) correlations reported with R², p-value, and sample size. Kernel density estimation contours show data distribution. Shaded band = 95% CI around regression line.",
        paths=paths,
    )


def plot_subject_metric_panels(
    subject_metrics: pd.DataFrame,
    group_metrics: Dict[str, Dict[str, float]],
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
    palette: Dict[str, Tuple[float, float, float]],
) -> Optional[FigureSpec]:
    if subject_metrics.empty:
        log("Subject metrics table is empty; skipping metric panels", "WARNING")
        return None

    subjects = subject_metrics["subject"].unique().tolist()
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
            subject_metrics[["subject", spec["column"]]]
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
            value = series.loc[subject][spec["column"]] if pd.notna(series.loc[subject][spec["column"]]) else None
            if value is None:
                continue
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

    # No suptitle

    paths = save_figure(fig, "Fig4_NPS_SubjectMetrics", figure_dir, formats, dpi)
    return FigureSpec(
        name="Fig4_NPS_SubjectMetrics",
        description="Individual subject performance across five key metrics (panels A-E): slope of NPS-temperature relationship, correlation with temperature, correlation with VAS, pain classification AUC, and forced-choice accuracy. Dashed vertical lines mark theoretical baselines (0 for correlations/slopes, 0.5 for classification). Solid vertical line with shaded band = group mean with 95% CI.",
        paths=paths,
    )


def plot_subject_roc_curves(
    trial_df: pd.DataFrame,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
    palette: Dict[str, Tuple[float, float, float]],
    pain_threshold: float,
) -> Optional[FigureSpec]:
    if trial_df.empty:
        log("Trial-level table empty; skipping ROC curves", "WARNING")
        return None

    try:
        from sklearn.metrics import auc as sklearn_auc
        from sklearn.metrics import roc_curve
    except ImportError:  # pragma: no cover
        log("scikit-learn not available; skipping ROC curves", "WARNING")
        return None

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
            log(f"No pain labels available for {subject}; skipping", "INFO")
            continue

        if labels.nunique() < 2:
            log(f"Only one pain class for {subject}; skipping ROC", "INFO")
            continue

        try:
            fpr, tpr, _ = roc_curve(labels, valid["br_score"])
            auc_value = sklearn_auc(fpr, tpr)
            
            n_boot = 500
            boot_aucs = []
            for _ in range(n_boot):
                idx = np.random.choice(len(labels), len(labels), replace=True)
                if labels.iloc[idx].nunique() < 2:
                    continue
                try:
                    boot_fpr, boot_tpr, _ = roc_curve(labels.iloc[idx], valid["br_score"].iloc[idx])
                    boot_aucs.append(sklearn_auc(boot_fpr, boot_tpr))
                except Exception:
                    continue
            
            if boot_aucs:
                auc_ci = tuple(np.percentile(boot_aucs, [2.5, 97.5]))
            else:
                auc_ci = (np.nan, np.nan)
                
        except Exception as exc:  # pragma: no cover
            log(f"ROC computation failed for {subject}: {exc}", "WARNING")
            continue
        curves.append((subject, fpr, tpr, auc_value, auc_ci))

    if not curves:
        log("No valid ROC curves computed", "WARNING")
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

    paths = save_figure(fig, "Fig5_NPS_PainROC", figure_dir, formats, dpi)
    return FigureSpec(
        name="Fig5_NPS_PainROC",
        description="Receiver operating characteristic (ROC) curves for pain vs. non-pain classification at the trial level. AUC values with bootstrap 95% confidence intervals (500 iterations) quantify classification performance relative to chance (diagonal line, AUC=0.50). Background shading highlights feasible operating region.",
        paths=paths,
    )


def plot_trial_temperature_distributions(
    trial_df: pd.DataFrame,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
    subject_colors: Dict[str, Tuple[float, float, float]],
) -> Optional[FigureSpec]:
    required_cols = {"temp_celsius", "br_score", "subject"}
    if not required_cols.issubset(trial_df.columns):
        log("Trial-level table missing required columns; skipping temperature distributions", "INFO")
        return None

    data = trial_df.dropna(subset=["temp_celsius", "br_score"])
    if data.empty:
        log("No valid trial-level data for temperature distributions", "WARNING")
        return None

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
    if len(outliers) > 0:
        log(f"Fig6: {len(outliers)} extreme outliers detected and will be shown separately", "WARNING")
        for _, row in outliers.iterrows():
            log(f"  Outlier: {row['subject']} at {row['temp_celsius']:.1f}°C, NPS={row['br_score']:.2f}", "INFO")

    palette = sns.color_palette("YlOrRd", len(unique_temps))
    color_lookup = dict(zip(temp_labels, palette))
    counts = data.groupby("temp_label").size().reindex(temp_labels, fill_value=0)

    fig, axis = plt.subplots(figsize=(5.0, 3.0), constrained_layout=True)
    
    gray_palette = {label: (0.7, 0.7, 0.7) for label in temp_labels}
    
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
    
    if subject_colors:
        hue_palette = subject_colors
    else:
        hue_palette = "colorblind"
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
        palette=hue_palette,
        ax=axis,
        legend=False,
    )
    axis.axhline(0, color="0.6", linestyle="-", linewidth=0.5, alpha=0.5)
    axis.set_xlabel("Temperature (°C)")
    axis.set_ylabel("NPS response (a.u.)")
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

    paths = save_figure(fig, "Fig6_NPS_TemperatureDistributions", figure_dir, formats, dpi)
    return FigureSpec(
        name="Fig6_NPS_TemperatureDistributions",
        description="Trial-level NPS response distributions by temperature condition. Violin plots show probability density with embedded box plots (quartiles). Overlaid strip plots distinguish individual subjects. White horizontal lines mark medians. Y-axis uses robust percentile-based scaling (1st-99th percentile) to prevent extreme outliers from compressing visualization. Sample sizes indicated below each temperature.",
        paths=paths,
    )


def plot_temperature_vas_curve(
    level_df: pd.DataFrame,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> Optional[FigureSpec]:
    required_cols = {"temp_celsius", "mean_vas"}
    if not required_cols.issubset(level_df.columns):
        log("Condition-level data missing VAS columns; skipping VAS curve", "INFO")
        return None

    data = level_df.dropna(subset=["temp_celsius", "mean_vas"])
    if data.empty:
        log("Insufficient VAS data for temperature curve", "WARNING")
        return None

    summary = (
        data.groupby("temp_celsius")
        .agg(mean_vas=("mean_vas", "mean"), std_vas=("mean_vas", "std"), n=("mean_vas", "count"))
        .reset_index()
        .sort_values("temp_celsius")
    )
    if summary.empty:
        return None

    cis = []
    for _, row in summary.iterrows():
        if row["n"] > 1 and not np.isnan(row["std_vas"]):
            t_multiplier = stats.t.ppf(0.975, int(row["n"]) - 1)
            cis.append(t_multiplier * row["std_vas"] / math.sqrt(row["n"]))
        else:
            cis.append(np.nan)
    summary["ci"] = cis

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

    paths = save_figure(fig, "Fig7_NPS_TemperatureVAS", figure_dir, formats, dpi)
    return FigureSpec(
        name="Fig7_NPS_TemperatureVAS",
        description="Perceived pain intensity (VAS ratings) as a function of thermal stimulus temperature. Group mean with 95% t-based CI band. Linear regression statistics (slope β, R², p-value) quantify psychophysical relationship. Demonstrates stimulus-response mapping independent of NPS.",
        paths=paths,
    )


def plot_bland_altman(level_df: pd.DataFrame, figure_dir: Path, formats: Sequence[str], dpi: int) -> Optional[FigureSpec]:
    """Bland-Altman plot for VAS-NPS agreement."""
    required_cols = {"mean_vas", "br_score"}
    if not required_cols.issubset(level_df.columns):
        log("Missing VAS or NPS columns for Bland-Altman plot", "WARNING")
        return None
    
    data = level_df.dropna(subset=["mean_vas", "br_score"])
    if len(data) < 3:
        log("Insufficient data for Bland-Altman plot", "WARNING")
        return None
    
    # Standardize to same scale
    vas_z = (data["mean_vas"] - data["mean_vas"].mean()) / data["mean_vas"].std()
    nps_z = (data["br_score"] - data["br_score"].mean()) / data["br_score"].std()
    
    mean_val = (vas_z + nps_z) / 2
    diff_val = nps_z - vas_z
    
    mean_diff = diff_val.mean()
    std_diff = diff_val.std()
    
    fig, axis = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)
    
    axis.scatter(mean_val, diff_val, s=20, alpha=0.6, color="0.3", linewidth=0)
    axis.axhline(mean_diff, color="0.2", linewidth=1.0, linestyle="-", label="Mean difference")
    axis.axhline(mean_diff + 1.96*std_diff, color="0.5", linewidth=0.8, linestyle="--", label="±1.96 SD")
    axis.axhline(mean_diff - 1.96*std_diff, color="0.5", linewidth=0.8, linestyle="--")
    axis.axhline(0, color="0.7", linewidth=0.5, alpha=0.5)
    
    axis.set_xlabel("Mean (VAS, NPS) [z-score]")
    axis.set_ylabel("Difference (NPS - VAS) [z-score]")
    axis.legend(loc="best", fontsize=7)
    sns.despine(ax=axis, trim=True)
    
    stats_text = f"Bias={mean_diff:.2f}\nLoA=[{mean_diff-1.96*std_diff:.2f}, {mean_diff+1.96*std_diff:.2f}]"
    axis.text(0.02, 0.98, stats_text, transform=axis.transAxes, fontsize=7, va="top", ha="left", color="0.3")
    
    paths = save_figure(fig, "Fig9_BlandAltman_VAS_NPS", figure_dir, formats, dpi)
    return FigureSpec(
        name="Fig9_BlandAltman_VAS_NPS",
        description="Bland-Altman agreement plot comparing standardized VAS and NPS responses. Mean difference (bias) and limits of agreement (±1.96 SD) assess systematic differences and measurement consistency.",
        paths=paths,
    )


def plot_effect_sizes_by_temperature(
    level_df: pd.DataFrame, figure_dir: Path, formats: Sequence[str], dpi: int
) -> Optional[FigureSpec]:
    """Effect sizes (Cohen's d) for NPS response vs baseline at each temperature."""
    if "temp_celsius" not in level_df.columns or "br_score" not in level_df.columns:
        log("Missing temperature or NPS data for effect size plot", "WARNING")
        return None
    
    data = level_df.dropna(subset=["temp_celsius", "br_score"])
    if data.empty:
        return None
    
    temps = sorted(data["temp_celsius"].unique())
    if len(temps) < 2:
        return None
    
    effect_sizes = []
    temp_labels = []
    cis_lower = []
    cis_upper = []
    
    baseline_temp = temps[0]
    baseline_data = data[data["temp_celsius"] == baseline_temp]["br_score"].values
    
    for temp in temps[1:]:
        temp_data = data[data["temp_celsius"] == temp]["br_score"].values
        if len(temp_data) < 2 or len(baseline_data) < 2:
            continue
        
        d = _cohens_d(temp_data, baseline_data)
        effect_sizes.append(d)
        temp_labels.append(f"{temp:.1f}")
        
        # Bootstrap CI for Cohen's d
        boot_d = []
        for _ in range(500):
            bt_idx = np.random.choice(len(temp_data), len(temp_data), replace=True)
            bb_idx = np.random.choice(len(baseline_data), len(baseline_data), replace=True)
            try:
                boot_d.append(_cohens_d(temp_data[bt_idx], baseline_data[bb_idx]))
            except:
                continue
        if boot_d:
            ci_l, ci_u = np.percentile(boot_d, [2.5, 97.5])
            cis_lower.append(ci_l)
            cis_upper.append(ci_u)
        else:
            cis_lower.append(d)
            cis_upper.append(d)
    
    if not effect_sizes:
        return None
    
    fig, axis = plt.subplots(figsize=(4.0, 2.5), constrained_layout=True)
    
    x_pos = np.arange(len(effect_sizes))
    axis.bar(x_pos, effect_sizes, color="0.6", edgecolor="0.2", linewidth=0.5, alpha=0.7)
    axis.errorbar(x_pos, effect_sizes, 
                  yerr=[np.array(effect_sizes) - np.array(cis_lower), 
                        np.array(cis_upper) - np.array(effect_sizes)],
                  fmt='none', ecolor='0.3', elinewidth=1.0, capsize=2)
    
    axis.axhline(0, color="0.7", linewidth=0.5, alpha=0.5)
    axis.axhline(0.2, color="0.8", linewidth=0.5, linestyle=":", alpha=0.5)
    axis.axhline(0.5, color="0.8", linewidth=0.5, linestyle=":", alpha=0.5)
    axis.axhline(0.8, color="0.8", linewidth=0.5, linestyle=":", alpha=0.5)
    
    axis.set_xlabel(f"Temperature (°C) vs {baseline_temp:.1f}°C")
    axis.set_ylabel("Cohen's d")
    axis.set_xticks(x_pos)
    axis.set_xticklabels(temp_labels)
    sns.despine(ax=axis, trim=True)
    
    axis.text(0.98, 0.98, "Small=0.2\nMed=0.5\nLarge=0.8", 
              transform=axis.transAxes, fontsize=6.5, va="top", ha="right", color="0.5", style="italic")
    
    paths = save_figure(fig, "Fig10_EffectSizes_Temperature", figure_dir, formats, dpi)
    return FigureSpec(
        name="Fig10_EffectSizes_Temperature",
        description="Effect sizes (Cohen's d) for NPS response at each temperature relative to baseline. Error bars show 95% bootstrap CI. Horizontal reference lines mark conventional thresholds (small=0.2, medium=0.5, large=0.8).",
        paths=paths,
    )


def plot_residual_diagnostics(
    level_df: pd.DataFrame, figure_dir: Path, formats: Sequence[str], dpi: int
) -> Optional[FigureSpec]:
    """Residual plots for NPS~Temperature linear model."""
    required = {"temp_celsius", "br_score", "subject"}
    if not required.issubset(level_df.columns):
        return None
    
    data = level_df.dropna(subset=["temp_celsius", "br_score"])
    if len(data) < 5:
        return None
    
    # Fit simple linear model
    fit = stats.linregress(data["temp_celsius"], data["br_score"])
    predicted = fit.slope * data["temp_celsius"] + fit.intercept
    residuals = data["br_score"] - predicted
    
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.5), constrained_layout=True)
    
    # Residuals vs fitted
    axes[0].scatter(predicted, residuals, s=20, alpha=0.6, color="0.3", linewidth=0)
    axes[0].axhline(0, color="0.6", linewidth=0.5, linestyle="-")
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")
    sns.despine(ax=axes[0], trim=True)
    
    # Q-Q plot
    from scipy import stats as sp_stats
    (osm, osr), (slope_qq, intercept_qq, r_qq) = sp_stats.probplot(residuals, dist="norm")
    axes[1].scatter(osm, osr, s=15, alpha=0.6, color="0.3", linewidth=0)
    axes[1].plot(osm, slope_qq * osm + intercept_qq, color="0.5", linewidth=1.0, linestyle="--")
    axes[1].set_xlabel("Theoretical quantiles")
    axes[1].set_ylabel("Sample quantiles")
    axes[1].text(0.02, 0.98, f"r={r_qq:.2f}", transform=axes[1].transAxes, 
                 fontsize=7, va="top", color="0.3")
    sns.despine(ax=axes[1], trim=True)
    
    paths = save_figure(fig, "Fig11_Residual_Diagnostics", figure_dir, formats, dpi)
    return FigureSpec(
        name="Fig11_Residual_Diagnostics",
        description="Residual diagnostics for NPS~Temperature linear regression. Left: residuals vs fitted values checks homoscedasticity. Right: Q-Q plot assesses normality assumption.",
        paths=paths,
    )


def plot_within_subject_reliability(
    trial_df: pd.DataFrame, figure_dir: Path, formats: Sequence[str], dpi: int, palette: Dict[str, Tuple[float, float, float]]
) -> Optional[FigureSpec]:
    """Within-subject reliability across temperature conditions."""
    if "subject" not in trial_df.columns or "temp_celsius" not in trial_df.columns or "br_score" not in trial_df.columns:
        return None
    
    data = trial_df.dropna(subset=["subject", "temp_celsius", "br_score"])
    if data.empty:
        return None
    
    # Calculate within-subject coefficient of variation for each temp
    reliability_data = []
    for (subject, temp), group in data.groupby(["subject", "temp_celsius"]):
        if len(group) >= 2:
            cv = group["br_score"].std() / abs(group["br_score"].mean()) if group["br_score"].mean() != 0 else np.nan
            icc_data = group["br_score"].values
            reliability_data.append({
                "subject": subject,
                "temp": temp,
                "cv": cv,
                "n_trials": len(group),
                "std": group["br_score"].std(),
            })
    
    if not reliability_data:
        return None
    
    rel_df = pd.DataFrame(reliability_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.5), constrained_layout=True)
    
    # CV by temperature
    for subject in rel_df["subject"].unique():
        sub_data = rel_df[rel_df["subject"] == subject]
        axes[0].plot(sub_data["temp"], sub_data["cv"], 
                     marker="o", markersize=3, linewidth=0.8, alpha=0.7,
                     color=palette.get(subject, (0.5, 0.5, 0.5)))
    
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Coefficient of variation")
    axes[0].axhline(0.5, color="0.8", linewidth=0.5, linestyle=":", alpha=0.5)
    sns.despine(ax=axes[0], trim=True)
    
    # SD distribution
    axes[1].hist(rel_df["std"].dropna(), bins=15, color="0.6", edgecolor="0.2", linewidth=0.5, alpha=0.7)
    axes[1].set_xlabel("Within-condition SD")
    axes[1].set_ylabel("Frequency")
    median_std = rel_df["std"].median()
    axes[1].axvline(median_std, color="0.2", linewidth=1.0, linestyle="--")
    axes[1].text(0.98, 0.98, f"Median={median_std:.2f}", 
                 transform=axes[1].transAxes, fontsize=7, va="top", ha="right", color="0.3")
    sns.despine(ax=axes[1], trim=True)
    
    paths = save_figure(fig, "Fig12_Within_Subject_Reliability", figure_dir, formats, dpi)
    return FigureSpec(
        name="Fig12_Within_Subject_Reliability",
        description="Within-subject measurement reliability. Left: coefficient of variation across temperatures. Right: distribution of within-condition standard deviations. Lower values indicate more consistent responses.",
        paths=paths,
    )


def plot_correlation_matrix(
    level_df: pd.DataFrame, figure_dir: Path, formats: Sequence[str], dpi: int
) -> Optional[FigureSpec]:
    """Correlation matrix of key continuous variables."""
    var_cols = ["temp_celsius", "br_score", "mean_vas"]
    available = [c for c in var_cols if c in level_df.columns]
    
    if len(available) < 2:
        return None
    
    data = level_df[available].dropna()
    if len(data) < 3:
        return None
    
    corr_matrix = data.corr(method="pearson")
    
    # Compute p-values
    n = len(data)
    p_matrix = np.zeros_like(corr_matrix)
    for i, col1 in enumerate(available):
        for j, col2 in enumerate(available):
            if i != j:
                r = corr_matrix.iloc[i, j]
                t_stat = r * np.sqrt((n-2) / (1 - r**2)) if abs(r) < 1 else np.inf
                p_matrix[i, j] = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
    
    fig, axis = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)
    
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    
    cmap = sns.diverging_palette(240, 10, s=50, l=60, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
        annot_kws={"fontsize": 8},
        ax=axis,
    )
    
    # Add significance stars
    for i in range(len(available)):
        for j in range(i):
            if p_matrix[i, j] < 0.001:
                marker = "***"
            elif p_matrix[i, j] < 0.01:
                marker = "**"
            elif p_matrix[i, j] < 0.05:
                marker = "*"
            else:
                marker = ""
            if marker:
                axis.text(j + 0.5, i + 0.3, marker, ha="center", va="center", 
                         fontsize=9, color="0.2", weight="bold")
    
    axis.set_xticklabels(axis.get_xticklabels(), rotation=45, ha="right")
    axis.set_yticklabels(axis.get_yticklabels(), rotation=0)
    
    paths = save_figure(fig, "Fig13_Correlation_Matrix", figure_dir, formats, dpi)
    return FigureSpec(
        name="Fig13_Correlation_Matrix",
        description="Pearson correlation matrix for key continuous variables. Lower triangle shows correlation coefficients; significance markers indicate *p<0.05, **p<0.01, ***p<0.001.",
        paths=paths,
    )


def plot_temporal_dynamics_enhanced(
    trial_df: pd.DataFrame, figure_dir: Path, formats: Sequence[str], dpi: int, palette: Dict[str, Tuple[float, float, float]]
) -> Optional[FigureSpec]:
    """Enhanced trial-by-trial temporal evolution with moving average and trend analysis."""
    if "br_score" not in trial_df.columns:
        return None
    
    # Extract trial number
    trial_df_copy = trial_df.copy()
    if "trial_regressor" in trial_df_copy.columns:
        trial_df_copy["trial_num"] = trial_df_copy["trial_regressor"].str.extract(r"(\d+)").astype(float)
    else:
        trial_df_copy["trial_num"] = trial_df_copy.groupby("subject").cumcount() + 1
    
    data = trial_df_copy.dropna(subset=["trial_num", "br_score"])
    if len(data) < 10:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    
    # Panel A: Individual subject trajectories with raw trial data
    axes[0].text(-0.15, 1.05, "A", transform=axes[0].transAxes,
                fontsize=16, fontweight="bold", va="top", ha="right")
    
    if "subject" in data.columns:
        for subject in data["subject"].unique():
            sub_data = data[data["subject"] == subject].sort_values("trial_num")
            axes[0].plot(sub_data["trial_num"], sub_data["br_score"],
                        alpha=0.5, linewidth=0.8, color=palette.get(subject, (0.5, 0.5, 0.5)),
                        marker='o', markersize=2.5, markeredgewidth=0, label=subject)
    else:
        axes[0].plot(data["trial_num"], data["br_score"],
                    alpha=0.6, linewidth=0.8, color="0.3", marker='o', markersize=2.5)
    
    axes[0].axhline(0, color="0.7", linewidth=0.5, alpha=0.5)
    axes[0].set_xlabel("Trial number")
    axes[0].set_ylabel("NPS response (a.u.)")
    axes[0].set_title("Individual trial sequences", fontsize=9, fontweight="normal")
    
    # Legend only if not too many subjects
    if "subject" in data.columns and data["subject"].nunique() <= 4:
        axes[0].legend(loc="best", fontsize=6.5, frameon=False, ncol=1)
    
    sns.despine(ax=axes[0], trim=True)
    
    # Panel B: Moving average with confidence envelope
    axes[1].text(-0.15, 1.05, "B", transform=axes[1].transAxes,
                fontsize=16, fontweight="bold", va="top", ha="right")
    
    # Sort and compute moving average
    data_sorted = data.sort_values("trial_num")
    
    # Group by trial number for multi-subject data
    if "subject" in data.columns and data["subject"].nunique() > 1:
        trial_summary = data.groupby("trial_num")["br_score"].agg(["mean", "sem", "count"]).reset_index()
        x_data = trial_summary["trial_num"].values
        y_data = trial_summary["mean"].values
        sem_data = trial_summary["sem"].values
    else:
        x_data = data_sorted["trial_num"].values
        y_data = data_sorted["br_score"].values
        sem_data = np.zeros_like(y_data)
    
    # Compute moving average (window=5)
    window = min(5, max(3, len(x_data) // 10))
    if len(y_data) >= window:
        from scipy.ndimage import uniform_filter1d
        y_smooth = uniform_filter1d(y_data, size=window, mode='nearest')
        sem_smooth = uniform_filter1d(sem_data, size=window, mode='nearest') if sem_data.any() else np.zeros_like(y_smooth)
    else:
        y_smooth = y_data
        sem_smooth = sem_data
    
    # Plot moving average
    axes[1].plot(x_data, y_smooth, linewidth=2.0, color="0.2", label=f"Moving avg (w={window})")
    
    # Confidence envelope
    if sem_smooth.any():
        axes[1].fill_between(x_data,
                            y_smooth - 1.96 * sem_smooth,
                            y_smooth + 1.96 * sem_smooth,
                            alpha=0.15, color="0.2", linewidth=0)
    
    # Linear trend
    if len(x_data) >= 3:
        try:
            fit = stats.linregress(x_data, y_smooth)
            y_trend = fit.slope * x_data + fit.intercept
            axes[1].plot(x_data, y_trend, linestyle="--", linewidth=1.2, color="red", alpha=0.7,
                        label=f"Linear trend")
            
            # Determine trend direction and significance
            if abs(fit.slope) < 0.01:
                trend_direction = "Stable"
            elif fit.slope < 0:
                trend_direction = "Habituation" if fit.pvalue < 0.05 else "Habituation (n.s.)"
            else:
                trend_direction = "Sensitization" if fit.pvalue < 0.05 else "Sensitization (n.s.)"
            
            # Enhanced statistical annotation
            stats_text = (
                f"{trend_direction}\n"
                f"slope = {fit.slope:.3f}\n"
                f"{_format_pvalue(fit.pvalue)}\n"
                f"r² = {fit.rvalue**2:.3f}\n"
                f"n = {len(x_data)} trials"
            )
            axes[1].text(0.98, 0.98, stats_text, transform=axes[1].transAxes,
                        fontsize=6.5, va="top", ha="right", color="0.3",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                                edgecolor="0.8", linewidth=0.5, alpha=0.9))
        except:
            pass
    
    axes[1].axhline(0, color="0.7", linewidth=0.5, alpha=0.5)
    axes[1].set_xlabel("Trial number")
    axes[1].set_ylabel("NPS response (a.u.)")
    axes[1].set_title("Temporal dynamics (smoothed)", fontsize=9, fontweight="normal")
    axes[1].legend(loc="upper left", fontsize=7, frameon=False)
    
    sns.despine(ax=axes[1], trim=True)
    
    paths = save_figure(fig, "Fig14a_Temporal_Dynamics_Enhanced", figure_dir, formats, dpi)
    return FigureSpec(
        name="Fig14a_Temporal_Dynamics_Enhanced",
        description="Enhanced trial-by-trial temporal evolution of NPS responses. Panel A: Individual subject trajectories showing raw trial-level data across the experiment. Panel B: Moving average (smoothed) group response with 95% CI envelope and linear trend analysis. Detects habituation (negative slope), sensitization (positive slope), or stable responding over time. Statistical annotations include slope, significance, effect size (r²), and sample size.",
        paths=paths,
    )


def plot_trial_order_effects(
    trial_df: pd.DataFrame, figure_dir: Path, formats: Sequence[str], dpi: int, palette: Dict[str, Tuple[float, float, float]]
) -> Optional[FigureSpec]:
    """Trial order effects to detect habituation or sensitization."""
    if "br_score" not in trial_df.columns:
        return None
    
    # Extract trial number if possible
    trial_df_copy = trial_df.copy()
    if "trial_regressor" in trial_df_copy.columns:
        trial_df_copy["trial_num"] = trial_df_copy["trial_regressor"].str.extract(r"(\d+)").astype(float)
    else:
        trial_df_copy["trial_num"] = trial_df_copy.groupby("subject").cumcount() + 1
    
    data = trial_df_copy.dropna(subset=["trial_num", "br_score"])
    if len(data) < 5:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.5), constrained_layout=True)
    
    # Scatter with trend
    if "subject" in data.columns:
        for subject in data["subject"].unique():
            sub_data = data[data["subject"] == subject]
            axes[0].scatter(sub_data["trial_num"], sub_data["br_score"],
                           s=15, alpha=0.4, linewidth=0,
                           color=palette.get(subject, (0.5, 0.5, 0.5)))
    else:
        axes[0].scatter(data["trial_num"], data["br_score"], s=15, alpha=0.4, color="0.3", linewidth=0)
    
    # Overall trend
    if len(data) >= 3:
        fit = stats.linregress(data["trial_num"], data["br_score"])
        trend_x = np.array([data["trial_num"].min(), data["trial_num"].max()])
        trend_y = fit.slope * trend_x + fit.intercept
        axes[0].plot(trend_x, trend_y, color="0.2", linewidth=1.5, linestyle="--", alpha=0.8)
        
        trend_text = f"slope={fit.slope:.3f}\n{_format_pvalue(fit.pvalue)}"
        axes[0].text(0.02, 0.98, trend_text, transform=axes[0].transAxes,
                     fontsize=7, va="top", ha="left", color="0.3")
    
    axes[0].set_xlabel("Trial number")
    axes[0].set_ylabel("NPS response (a.u.)")
    axes[0].axhline(0, color="0.7", linewidth=0.5, alpha=0.5)
    sns.despine(ax=axes[0], trim=True)
    
    # Binned averages
    n_bins = min(10, int(data["trial_num"].max() // 5))
    if n_bins >= 2:
        data["trial_bin"] = pd.cut(data["trial_num"], bins=n_bins, labels=False)
        binned = data.groupby("trial_bin")["br_score"].agg(["mean", "sem"]).reset_index()
        bin_centers = binned["trial_bin"] * (data["trial_num"].max() / n_bins)
        
        axes[1].errorbar(bin_centers, binned["mean"], yerr=binned["sem"],
                        marker="o", markersize=4, linewidth=1.0, color="0.2",
                        elinewidth=0.8, capsize=2, alpha=0.8)
        axes[1].axhline(0, color="0.7", linewidth=0.5, alpha=0.5)
        axes[1].set_xlabel("Trial bin")
        axes[1].set_ylabel("Mean NPS (±SEM)")
        sns.despine(ax=axes[1], trim=True)
    else:
        axes[1].axis("off")
    
    paths = save_figure(fig, "Fig14b_Trial_Order_Effects", figure_dir, formats, dpi)
    return FigureSpec(
        name="Fig14b_Trial_Order_Effects",
        description="Trial order effects across experiment (binned analysis). Left: trial-by-trial NPS with linear trend to detect habituation (negative slope) or sensitization (positive slope). Right: binned averages with SEM error bars showing aggregate temporal patterns.",
        paths=paths,
    )


def plot_variance_components(
    trial_df: pd.DataFrame, figure_dir: Path, formats: Sequence[str], dpi: int
) -> Optional[FigureSpec]:
    """Variance decomposition: between-subject, within-subject, temperature, residual."""
    required = {"subject", "temp_celsius", "br_score"}
    if not required.issubset(trial_df.columns):
        return None
    
    data = trial_df.dropna(subset=["subject", "temp_celsius", "br_score"])
    if len(data) < 10:
        return None
    
    # Calculate variance components
    grand_mean = data["br_score"].mean()
    total_var = data["br_score"].var()
    
    # Between-subject variance
    subject_means = data.groupby("subject")["br_score"].mean()
    between_subject_var = subject_means.var()
    
    # Temperature effect variance
    temp_means = data.groupby("temp_celsius")["br_score"].mean()
    temp_var = temp_means.var()
    
    # Within-subject variance (pooled)
    within_subject_vars = []
    for subject in data["subject"].unique():
        sub_data = data[data["subject"] == subject]["br_score"]
        if len(sub_data) > 1:
            within_subject_vars.append(sub_data.var())
    within_subject_var = np.mean(within_subject_vars) if within_subject_vars else 0
    
    # Residual
    residual_var = total_var - between_subject_var - temp_var
    residual_var = max(0, residual_var)
    
    components = {
        "Between\nsubjects": between_subject_var,
        "Temperature\neffect": temp_var,
        "Within\nsubject": within_subject_var,
        "Residual": residual_var,
    }
    
    # Normalize to percentages
    total_components = sum(components.values())
    if total_components == 0:
        return None
    
    percentages = {k: (v / total_components) * 100 for k, v in components.items()}
    
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.5), constrained_layout=True)
    
    # Bar plot
    labels = list(percentages.keys())
    values = list(percentages.values())
    colors = ["0.3", "0.5", "0.6", "0.8"]
    
    axes[0].bar(range(len(labels)), values, color=colors, edgecolor="0.2", linewidth=0.5, alpha=0.8)
    axes[0].set_ylabel("Variance explained (%)")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, fontsize=7.5)
    axes[0].set_ylim(0, 100)
    axes[0].axhline(50, color="0.7", linewidth=0.5, linestyle=":", alpha=0.5)
    sns.despine(ax=axes[0], trim=True)
    
    # Pie chart
    axes[1].pie(values, labels=labels, autopct='%1.1f%%', colors=colors,
                textprops={'fontsize': 7.5}, wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'})
    axes[1].set_aspect('equal')
    
    paths = save_figure(fig, "Fig15_Variance_Components", figure_dir, formats, dpi)
    return FigureSpec(
        name="Fig15_Variance_Components",
        description="Variance decomposition showing relative contributions of between-subject differences, temperature effects, within-subject variability, and residual variance to total NPS response variance.",
        paths=paths,
    )


def plot_pairwise_temperature_contrasts(
    level_df: pd.DataFrame, figure_dir: Path, formats: Sequence[str], dpi: int
) -> Optional[FigureSpec]:
    """Pairwise temperature comparisons with multiple comparison correction."""
    if "temp_celsius" not in level_df.columns or "br_score" not in level_df.columns:
        return None
    
    data = level_df.dropna(subset=["temp_celsius", "br_score"])
    temps = sorted(data["temp_celsius"].unique())
    
    if len(temps) < 2:
        return None
    
    from itertools import combinations
    
    contrasts = []
    for t1, t2 in combinations(temps, 2):
        data1 = data[data["temp_celsius"] == t1]["br_score"].values
        data2 = data[data["temp_celsius"] == t2]["br_score"].values
        
        if len(data1) < 2 or len(data2) < 2:
            continue
        
        t_stat, p_val = stats.ttest_ind(data2, data1)
        cohens_d = _cohens_d(data2, data1)
        mean_diff = np.mean(data2) - np.mean(data1)
        
        contrasts.append({
            "comparison": f"{t2:.1f} vs {t1:.1f}",
            "t1": t1,
            "t2": t2,
            "mean_diff": mean_diff,
            "cohens_d": cohens_d,
            "p_value": p_val,
        })
    
    if not contrasts:
        return None
    
    # Bonferroni correction
    n_comparisons = len(contrasts)
    for c in contrasts:
        c["p_corrected"] = min(1.0, c["p_value"] * n_comparisons)
        c["significant"] = c["p_corrected"] < 0.05
    
    contrast_df = pd.DataFrame(contrasts).sort_values("mean_diff")
    
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    
    # Mean differences
    y_pos = np.arange(len(contrast_df))
    colors = ["0.2" if sig else "0.6" for sig in contrast_df["significant"]]
    
    axes[0].barh(y_pos, contrast_df["mean_diff"], color=colors, edgecolor="0.2", linewidth=0.5, alpha=0.8)
    axes[0].axvline(0, color="0.7", linewidth=0.8, linestyle="-", alpha=0.6)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(contrast_df["comparison"], fontsize=7.5)
    axes[0].set_xlabel("Mean difference (NPS a.u.)")
    axes[0].set_title("Pairwise contrasts", fontsize=9, fontweight="normal")
    sns.despine(ax=axes[0], trim=True)
    
    # Effect sizes
    axes[1].scatter(contrast_df["cohens_d"], y_pos, s=40, 
                   c=["0.2" if sig else "0.6" for sig in contrast_df["significant"]],
                   linewidth=0, alpha=0.8)
    axes[1].axvline(0, color="0.7", linewidth=0.5, alpha=0.5)
    axes[1].axvline(0.2, color="0.8", linewidth=0.5, linestyle=":", alpha=0.5)
    axes[1].axvline(0.5, color="0.8", linewidth=0.5, linestyle=":", alpha=0.5)
    axes[1].axvline(0.8, color="0.8", linewidth=0.5, linestyle=":", alpha=0.5)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([""] * len(y_pos))
    axes[1].set_xlabel("Cohen's d")
    axes[1].set_title("Effect sizes", fontsize=9, fontweight="normal")
    axes[1].text(0.98, 0.02, "Dark=sig\n(Bonferroni)", 
                transform=axes[1].transAxes, fontsize=6.5, va="bottom", ha="right", color="0.4")
    sns.despine(ax=axes[1], trim=True)
    
    paths = save_figure(fig, "Fig16_Pairwise_Contrasts", figure_dir, formats, dpi)
    return FigureSpec(
        name="Fig16_Pairwise_Contrasts",
        description="All pairwise temperature comparisons with Bonferroni correction. Left: mean differences. Right: effect sizes (Cohen's d). Dark bars/points indicate significance after correction for multiple comparisons (p<0.05).",
        paths=paths,
    )


def plot_bootstrap_uncertainty_distributions(
    level_df: pd.DataFrame, figure_dir: Path, formats: Sequence[str], dpi: int
) -> Optional[FigureSpec]:
    """Bootstrap distributions for key statistics."""
    if "temp_celsius" not in level_df.columns or "br_score" not in level_df.columns:
        return None
    
    data = level_df.dropna(subset=["temp_celsius", "br_score"])
    if len(data) < 5:
        return None
    
    n_boot = 2000
    n = len(data)
    
    boot_slopes = []
    boot_correlations = []
    boot_means = []
    
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        boot_sample = data.iloc[idx]
        
        # Slope
        try:
            fit = stats.linregress(boot_sample["temp_celsius"], boot_sample["br_score"])
            boot_slopes.append(fit.slope)
        except:
            pass
        
        # Correlation
        if "mean_vas" in boot_sample.columns:
            vas_data = boot_sample.dropna(subset=["mean_vas"])
            if len(vas_data) >= 3:
                try:
                    r, _ = stats.pearsonr(vas_data["mean_vas"], vas_data["br_score"])
                    boot_correlations.append(r)
                except:
                    pass
        
        # Mean
        boot_means.append(boot_sample["br_score"].mean())
    
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.5), constrained_layout=True)
    
    # Slope distribution
    if boot_slopes:
        axes[0].hist(boot_slopes, bins=30, color="0.6", edgecolor="0.2", linewidth=0.5, alpha=0.8)
        ci_low, ci_high = np.percentile(boot_slopes, [2.5, 97.5])
        axes[0].axvline(np.median(boot_slopes), color="0.2", linewidth=1.2, linestyle="-")
        axes[0].axvline(ci_low, color="0.4", linewidth=0.8, linestyle="--")
        axes[0].axvline(ci_high, color="0.4", linewidth=0.8, linestyle="--")
        axes[0].set_xlabel("Temperature slope")
        axes[0].set_ylabel("Bootstrap frequency")
        axes[0].text(0.98, 0.98, f"95% CI\n[{ci_low:.3f},\n{ci_high:.3f}]", 
                    transform=axes[0].transAxes, fontsize=6.5, va="top", ha="right", color="0.3")
    else:
        axes[0].text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=axes[0].transAxes)
    sns.despine(ax=axes[0], trim=True)
    
    # Correlation distribution
    if boot_correlations:
        axes[1].hist(boot_correlations, bins=30, color="0.6", edgecolor="0.2", linewidth=0.5, alpha=0.8)
        ci_low, ci_high = np.percentile(boot_correlations, [2.5, 97.5])
        axes[1].axvline(np.median(boot_correlations), color="0.2", linewidth=1.2, linestyle="-")
        axes[1].axvline(ci_low, color="0.4", linewidth=0.8, linestyle="--")
        axes[1].axvline(ci_high, color="0.4", linewidth=0.8, linestyle="--")
        axes[1].set_xlabel("VAS-NPS correlation")
        axes[1].text(0.98, 0.98, f"95% CI\n[{ci_low:.3f},\n{ci_high:.3f}]", 
                    transform=axes[1].transAxes, fontsize=6.5, va="top", ha="right", color="0.3")
    else:
        axes[1].text(0.5, 0.5, "No VAS data", ha="center", va="center", transform=axes[1].transAxes)
    sns.despine(ax=axes[1], trim=True)
    
    # Mean distribution
    if boot_means:
        axes[2].hist(boot_means, bins=30, color="0.6", edgecolor="0.2", linewidth=0.5, alpha=0.8)
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        axes[2].axvline(np.median(boot_means), color="0.2", linewidth=1.2, linestyle="-")
        axes[2].axvline(ci_low, color="0.4", linewidth=0.8, linestyle="--")
        axes[2].axvline(ci_high, color="0.4", linewidth=0.8, linestyle="--")
        axes[2].set_xlabel("Mean NPS")
        axes[2].text(0.98, 0.98, f"95% CI\n[{ci_low:.3f},\n{ci_high:.3f}]", 
                    transform=axes[2].transAxes, fontsize=6.5, va="top", ha="right", color="0.3")
    sns.despine(ax=axes[2], trim=True)
    
    paths = save_figure(fig, "Fig17_Bootstrap_Distributions", figure_dir, formats, dpi)
    return FigureSpec(
        name="Fig17_Bootstrap_Distributions",
        description="Bootstrap sampling distributions (2000 iterations) for key statistics. Solid line=median, dashed lines=95% CI. Shows uncertainty in temperature slope, VAS-NPS correlation, and mean NPS response.",
        paths=paths,
    )


def plot_prediction_intervals(
    level_df: pd.DataFrame, figure_dir: Path, formats: Sequence[str], dpi: int
) -> Optional[FigureSpec]:
    """Temperature-NPS relationship with prediction intervals."""
    if "temp_celsius" not in level_df.columns or "br_score" not in level_df.columns:
        return None
    
    data = level_df.dropna(subset=["temp_celsius", "br_score"])
    if len(data) < 5:
        return None
    
    fit = stats.linregress(data["temp_celsius"], data["br_score"])
    x = data["temp_celsius"].values
    y = data["br_score"].values
    y_pred = fit.slope * x + fit.intercept
    residuals = y - y_pred
    
    # Standard error
    n = len(data)
    mse = np.sum(residuals**2) / (n - 2)
    x_mean = x.mean()
    se_pred = np.sqrt(mse * (1 + 1/n + (x - x_mean)**2 / np.sum((x - x_mean)**2)))
    
    # Create smooth prediction curve
    x_smooth = np.linspace(x.min(), x.max(), 100)
    y_smooth = fit.slope * x_smooth + fit.intercept
    se_smooth = np.sqrt(mse * (1 + 1/n + (x_smooth - x_mean)**2 / np.sum((x - x_mean)**2)))
    
    # 95% prediction interval (t-distribution)
    t_val = stats.t.ppf(0.975, n - 2)
    pi_lower = y_smooth - t_val * se_smooth
    pi_upper = y_smooth + t_val * se_smooth
    
    # 95% confidence interval for the mean
    se_mean = np.sqrt(mse * (1/n + (x_smooth - x_mean)**2 / np.sum((x - x_mean)**2)))
    ci_lower = y_smooth - t_val * se_mean
    ci_upper = y_smooth + t_val * se_mean
    
    fig, axis = plt.subplots(figsize=(4.0, 3.5), constrained_layout=True)
    
    # Prediction interval (wider)
    axis.fill_between(x_smooth, pi_lower, pi_upper, color="0.8", alpha=0.3, linewidth=0, label="95% PI")
    
    # Confidence interval (narrower)
    axis.fill_between(x_smooth, ci_lower, ci_upper, color="0.4", alpha=0.25, linewidth=0, label="95% CI")
    
    # Regression line
    axis.plot(x_smooth, y_smooth, color="0.2", linewidth=1.5, label="Regression")
    
    # Data points
    axis.scatter(x, y, s=25, alpha=0.6, color="0.3", linewidth=0, zorder=3)
    
    axis.set_xlabel("Temperature (°C)")
    axis.set_ylabel("NPS response (a.u.)")
    axis.legend(loc="best", fontsize=7)
    sns.despine(ax=axis, trim=True)
    
    stats_text = f"r²={fit.rvalue**2:.2f}\n{_format_pvalue(fit.pvalue)}"
    axis.text(0.02, 0.98, stats_text, transform=axis.transAxes, fontsize=7, va="top", color="0.3")
    
    paths = save_figure(fig, "Fig18_Prediction_Intervals", figure_dir, formats, dpi)
    return FigureSpec(
        name="Fig18_Prediction_Intervals",
        description="Temperature-NPS regression with prediction intervals. Inner band (dark)=95% CI for mean response. Outer band (light)=95% prediction interval for individual observations. Shows expected range of future measurements.",
        paths=paths,
    )


def plot_raincloud_distributions(
    trial_df: pd.DataFrame, figure_dir: Path, formats: Sequence[str], dpi: int, palette: Dict[str, Tuple[float, float, float]]
) -> Optional[FigureSpec]:
    """Raincloud plots: violin + boxplot + scatter for NPS by temperature."""
    if "temp_celsius" not in trial_df.columns or "br_score" not in trial_df.columns:
        return None
    
    data = trial_df.dropna(subset=["temp_celsius", "br_score"])
    temps = sorted(data["temp_celsius"].unique())
    
    if len(temps) < 2:
        return None
    
    # Limit extreme outliers for visualization
    q1 = data["br_score"].quantile(0.01)
    q99 = data["br_score"].quantile(0.99)
    data_plot = data[(data["br_score"] >= q1) & (data["br_score"] <= q99)].copy()
    
    temp_labels = [f"{t:.0f}" for t in temps]
    data_plot["temp_label"] = data_plot["temp_celsius"].apply(lambda t: f"{t:.0f}")
    
    fig, axis = plt.subplots(figsize=(6.0, 3.5), constrained_layout=True)
    
    # Half-violin plots
    parts = axis.violinplot(
        [data_plot[data_plot["temp_label"] == tl]["br_score"].values for tl in temp_labels],
        positions=np.arange(len(temp_labels)),
        widths=0.6,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor("0.75")
        pc.set_edgecolor("0.4")
        pc.set_alpha(0.7)
        pc.set_linewidth(0.5)
        # Make half-violin by modifying paths
        m = np.mean(pc.get_paths()[0].vertices[:, 0])
        pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], m, np.inf)
    
    # Box plots (offset)
    bp = axis.boxplot(
        [data_plot[data_plot["temp_label"] == tl]["br_score"].values for tl in temp_labels],
        positions=np.arange(len(temp_labels)) - 0.25,
        widths=0.15,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="0.2", linewidth=1.0),
        boxprops=dict(facecolor="0.9", edgecolor="0.3", linewidth=0.5, alpha=0.8),
        whiskerprops=dict(color="0.3", linewidth=0.5),
        capprops=dict(color="0.3", linewidth=0.5),
    )
    
    # Scatter points (jittered, offset)
    for i, tl in enumerate(temp_labels):
        temp_data = data_plot[data_plot["temp_label"] == tl]["br_score"].values
        y_jitter = temp_data
        x_jitter = np.random.normal(i - 0.25, 0.04, len(temp_data))
        
        # Color by subject if available
        if "subject" in data_plot.columns:
            for subject in data_plot["subject"].unique():
                sub_mask = (data_plot["temp_label"] == tl) & (data_plot["subject"] == subject)
                sub_data = data_plot[sub_mask]["br_score"].values
                if len(sub_data) > 0:
                    sub_x = np.random.normal(i - 0.25, 0.04, len(sub_data))
                    axis.scatter(sub_x, sub_data, s=8, alpha=0.4, linewidth=0,
                               color=palette.get(subject, (0.5, 0.5, 0.5)), zorder=3)
        else:
            axis.scatter(x_jitter, y_jitter, s=8, alpha=0.4, color="0.3", linewidth=0, zorder=3)
    
    axis.axhline(0, color="0.7", linewidth=0.5, alpha=0.5)
    axis.set_xticks(range(len(temp_labels)))
    axis.set_xticklabels(temp_labels)
    axis.set_xlabel("Temperature (°C)")
    axis.set_ylabel("NPS response (a.u.)")
    sns.despine(ax=axis, trim=True)
    
    paths = save_figure(fig, "Fig19_Raincloud_Distributions", figure_dir, formats, dpi)
    return FigureSpec(
        name="Fig19_Raincloud_Distributions",
        description="Raincloud plots showing NPS distributions by temperature. Combines half-violins (distribution), box plots (quartiles), and jittered scatter (individual trials) for comprehensive visualization.",
        paths=paths,
    )


def plot_cumulative_distributions(
    trial_df: pd.DataFrame, figure_dir: Path, formats: Sequence[str], dpi: int
) -> Optional[FigureSpec]:
    """Cumulative distribution functions comparing temperatures."""
    if "temp_celsius" not in trial_df.columns or "br_score" not in trial_df.columns:
        return None
    
    data = trial_df.dropna(subset=["temp_celsius", "br_score"])
    temps = sorted(data["temp_celsius"].unique())
    
    if len(temps) < 2:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), constrained_layout=True)
    
    # Empirical CDFs
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(temps)))
    for i, temp in enumerate(temps):
        temp_data = data[data["temp_celsius"] == temp]["br_score"].values
        temp_data_sorted = np.sort(temp_data)
        cdf = np.arange(1, len(temp_data_sorted) + 1) / len(temp_data_sorted)
        
        axes[0].plot(temp_data_sorted, cdf, linewidth=1.5, alpha=0.8,
                    color=colors[i], label=f"{temp:.1f}°C")
    
    axes[0].set_xlabel("NPS response (a.u.)")
    axes[0].set_ylabel("Cumulative probability")
    axes[0].legend(loc="best", fontsize=6.5, ncol=2)
    axes[0].grid(alpha=0.15, linewidth=0.5)
    axes[0].set_ylim(0, 1)
    sns.despine(ax=axes[0], trim=True)
    
    # K-S test matrix
    from scipy.stats import ks_2samp
    n_temps = len(temps)
    ks_matrix = np.zeros((n_temps, n_temps))
    
    for i, t1 in enumerate(temps):
        for j, t2 in enumerate(temps):
            if i != j:
                data1 = data[data["temp_celsius"] == t1]["br_score"].values
                data2 = data[data["temp_celsius"] == t2]["br_score"].values
                stat, pval = ks_2samp(data1, data2)
                ks_matrix[i, j] = -np.log10(pval) if pval > 0 else 10
    
    im = axes[1].imshow(ks_matrix, cmap="Greys", aspect="auto", vmin=0, vmax=3)
    axes[1].set_xticks(range(n_temps))
    axes[1].set_yticks(range(n_temps))
    axes[1].set_xticklabels([f"{t:.0f}" for t in temps], fontsize=7.5)
    axes[1].set_yticklabels([f"{t:.0f}" for t in temps], fontsize=7.5)
    axes[1].set_xlabel("Temperature (°C)")
    axes[1].set_ylabel("Temperature (°C)")
    
    cbar = plt.colorbar(im, ax=axes[1], shrink=0.8)
    cbar.set_label("-log10(p)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)
    
    axes[1].set_title("K-S test", fontsize=9, fontweight="normal")
    
    paths = save_figure(fig, "Fig20_Cumulative_Distributions", figure_dir, formats, dpi)
    return FigureSpec(
        name="Fig20_Cumulative_Distributions",
        description="Cumulative distribution functions across temperatures. Left: empirical CDFs show complete distribution shapes. Right: Kolmogorov-Smirnov test matrix quantifies pairwise distributional differences (-log10 p-value).",
        paths=paths,
    )


def plot_response_categories(
    level_df: pd.DataFrame, figure_dir: Path, formats: Sequence[str], dpi: int, palette: Dict[str, Tuple[float, float, float]]
) -> Optional[FigureSpec]:
    """Categorize subjects by response magnitude and pattern."""
    if "subject" not in level_df.columns or "br_score" not in level_df.columns or "temp_celsius" not in level_df.columns:
        return None
    
    # Calculate per-subject summary metrics
    subject_summaries = []
    for subject in level_df["subject"].unique():
        sub_data = level_df[level_df["subject"] == subject]
        if len(sub_data) >= 2:
            mean_response = sub_data["br_score"].mean()
            max_response = sub_data["br_score"].max()
            response_range = sub_data["br_score"].max() - sub_data["br_score"].min()
            
            # Calculate slope if possible
            if len(sub_data) >= 3:
                try:
                    fit = stats.linregress(sub_data["temp_celsius"], sub_data["br_score"])
                    slope = fit.slope
                except:
                    slope = np.nan
            else:
                slope = np.nan
            
            subject_summaries.append({
                "subject": subject,
                "mean_response": mean_response,
                "max_response": max_response,
                "response_range": response_range,
                "slope": slope,
            })
    
    if not subject_summaries:
        return None
    
    summary_df = pd.DataFrame(subject_summaries)
    
    # Categorize by median splits
    median_mean = summary_df["mean_response"].median()
    median_slope = summary_df["slope"].median()
    
    summary_df["category"] = "Other"
    summary_df.loc[(summary_df["mean_response"] > median_mean) & (summary_df["slope"] > median_slope), "category"] = "High responders"
    summary_df.loc[(summary_df["mean_response"] <= median_mean) & (summary_df["slope"] <= median_slope), "category"] = "Low responders"
    summary_df.loc[(summary_df["mean_response"] > median_mean) & (summary_df["slope"] <= median_slope), "category"] = "High-flat"
    summary_df.loc[(summary_df["mean_response"] <= median_mean) & (summary_df["slope"] > median_slope), "category"] = "Low-steep"
    
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), constrained_layout=True)
    
    # Scatter by categories
    category_colors = {
        "High responders": (0.8, 0.3, 0.3),
        "Low responders": (0.3, 0.5, 0.7),
        "High-flat": (0.9, 0.7, 0.4),
        "Low-steep": (0.5, 0.7, 0.5),
        "Other": (0.6, 0.6, 0.6),
    }
    
    for category in summary_df["category"].unique():
        cat_data = summary_df[summary_df["category"] == category]
        axes[0].scatter(cat_data["slope"], cat_data["mean_response"],
                       s=60, alpha=0.7, linewidth=0.5, edgecolors="white",
                       color=category_colors.get(category, (0.5, 0.5, 0.5)),
                       label=category)
    
    axes[0].axhline(median_mean, color="0.7", linewidth=0.8, linestyle="--", alpha=0.6)
    axes[0].axvline(median_slope, color="0.7", linewidth=0.8, linestyle="--", alpha=0.6)
    axes[0].set_xlabel("Temperature slope")
    axes[0].set_ylabel("Mean NPS response")
    axes[0].legend(loc="best", fontsize=7, frameon=True, framealpha=0.9)
    sns.despine(ax=axes[0], trim=True)
    
    # Category counts
    cat_counts = summary_df["category"].value_counts()
    axes[1].bar(range(len(cat_counts)), cat_counts.values,
               color=[category_colors.get(cat, (0.5, 0.5, 0.5)) for cat in cat_counts.index],
               edgecolor="0.2", linewidth=0.5, alpha=0.8)
    axes[1].set_xticks(range(len(cat_counts)))
    axes[1].set_xticklabels(cat_counts.index, rotation=45, ha="right", fontsize=7.5)
    axes[1].set_ylabel("Number of subjects")
    sns.despine(ax=axes[1], trim=True)
    
    paths = save_figure(fig, "Fig21_Response_Categories", figure_dir, formats, dpi)
    return FigureSpec(
        name="Fig21_Response_Categories",
        description="Subject categorization by response patterns. Left: scatter plot of mean response vs temperature sensitivity with quadrant-based categories. Right: distribution of subjects across categories. Identifies responder phenotypes.",
        paths=paths,
    )


def plot_subject_metric_heatmap(
    subject_metrics: pd.DataFrame,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> Optional[FigureSpec]:
    if subject_metrics.empty:
        return None

    metric_cols = [
        col
        for col in [
            "slope_BR_temp",
            "r_BR_temp",
            "r_BR_VAS",
            "auc_pain",
            "forced_choice_accuracy",
        ]
        if col in subject_metrics.columns
    ]
    if not metric_cols:
        return None

    matrix = (
        subject_metrics[["subject"] + metric_cols]
        .drop_duplicates("subject", keep="last")
        .set_index("subject")
        .apply(pd.to_numeric, errors="coerce")
    )
    matrix = matrix.dropna(how="all", axis=1)
    if matrix.empty:
        return None

    z_matrix = matrix.copy()
    for col in z_matrix.columns:
        col_std = z_matrix[col].std(ddof=0)
        if np.isfinite(col_std) and col_std > 0:
            z_matrix[col] = (z_matrix[col] - z_matrix[col].mean()) / col_std
        else:
            z_matrix[col] = 0.0

    annot_matrix = matrix.copy()
    for col in annot_matrix.columns:
        annot_matrix[col] = annot_matrix[col].apply(lambda x: f"{x:.3f}" if abs(x) >= 0.01 else f"{x:.2e}")
    
    cmap = sns.diverging_palette(240, 10, s=50, l=60, as_cmap=True)
    fig_height = max(2.0, 0.5 * len(matrix.index) + 0.8)
    fig_width = max(4.0, 0.8 * len(matrix.columns) + 1.0)
    fig, axis = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    
    heatmap = sns.heatmap(
        z_matrix,
        annot=annot_matrix,
        fmt="",
        cmap=cmap,
        center=0,
        vmin=-2,
        vmax=2,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "z-score", "shrink": 0.8},
        annot_kws={"fontsize": 7},
        ax=axis,
    )
    
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.tick_params(labelsize=8)
    
    for label in axis.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

    paths = save_figure(fig, "Fig8_NPS_SubjectMetricHeatmap", figure_dir, formats, dpi)
    return FigureSpec(
        name="Fig8_NPS_SubjectMetricHeatmap",
        description="Subject performance profile heatmap with column-wise z-score standardization. Cell colors reflect standardized deviations from metric-specific means; annotations show raw values. Black borders identify cells where |z| > 1.96 (equivalent to p < 0.05 two-tailed). Diverging colormap (blue = below average, red = above average) facilitates rapid comparison.",
        paths=paths,
    )


def main() -> int:
    args = parse_args()
    configure_matplotlib(args.style_context)

    config_path = Path(args.config)
    if not config_path.exists():
        alt_path = Path(__file__).resolve().parent / args.config
        if alt_path.exists():
            config_path = alt_path
    config = load_config(str(config_path))
    subjects = config.get("subjects", [])
    if not subjects:
        return 1
    
    if args.signatures:
        signatures_to_plot = args.signatures
    else:
        signatures_to_plot = config.get('enabled_signatures', ['nps'])
    
    paths = PipelinePaths.from_config(config)
    paths.ensure_core_roots()
    scores_override = Path(args.scores_dir) if args.scores_dir else None
    base_group_dir = Path(args.group_dir) if args.group_dir else Path(paths.group_root)
    base_group_dir.mkdir(parents=True, exist_ok=True)
    base_figures_dir = Path(args.figures_dir) if args.figures_dir else Path(paths.figures_root)
    base_figures_dir.mkdir(parents=True, exist_ok=True)
    formats = resolve_formats(args.formats)
    
    all_success = True
    
    for signature in signatures_to_plot:
        log(signature.upper())
        score_candidates = []
        if scores_override is not None:
            base_override = scores_override
            if not base_override.is_absolute():
                base_override = (Path(__file__).resolve().parent / base_override).resolve()
            else:
                base_override = base_override.resolve()
            score_candidates.extend([
                base_override,
                base_override / f"{signature}_scores",
                base_override / signature,
                base_override / signature / "scores",
                base_override / "signatures" / signature / "scores",
            ])
        score_candidates.append(Path(paths.signature_scores_dir(signature)))
        scores_dir = next((cand for cand in score_candidates if cand.is_dir() and any(cand.glob('sub-*'))), None)
        if scores_dir is None:
            all_success = False
            continue

        group_dir = base_group_dir / signature
        figure_dir = base_figures_dir / f"{signature}_summary"
        ensure_directory(figure_dir)

        # Load data for this signature
        subject_metrics = load_subject_metrics(scores_dir, subjects)
        level_df = load_level_br(scores_dir, subjects, signature)
        trial_df_raw = load_trial_br(scores_dir, subjects, signature)
        trial_df, n_extreme_trials_removed, removed_trials = filter_extreme_trials(
            trial_df_raw, args.trial_outlier_threshold
        )
        group_metrics = load_group_metrics(group_dir)

        subjects_with_data: List[str] = []
        if "subject" in level_df.columns:
            subjects_with_data.extend(level_df["subject"].unique().tolist())
        if "subject" in subject_metrics.columns:
            subjects_with_data.extend(subject_metrics["subject"].unique().tolist())
        if "subject" in trial_df.columns:
            subjects_with_data.extend(trial_df["subject"].unique().tolist())
        subjects_with_data = sorted(set(subjects_with_data))
        palette = subject_palette(subjects_with_data)

        figures: List[FigureSpec] = []

        figure = plot_subject_dose_response(level_df, figure_dir, formats, args.dpi, palette, data_type="condition", signature=signature.upper())

        append_figure_with_stats(figures, figure, figure_dir, {"condition_level": level_df})



        figure = plot_group_dose_response(level_df, figure_dir, formats, args.dpi, palette, data_type="condition", signature=signature.upper())

        append_figure_with_stats(figures, figure, figure_dir, {"condition_level": level_df})

        if not trial_df.empty:
            level_from_trials = aggregate_trials_to_conditions(trial_df)

            if not level_from_trials.empty:

                figure = plot_subject_dose_response(level_from_trials, figure_dir, formats, args.dpi, palette, data_type="trial", signature=signature.upper())

                append_figure_with_stats(figures, figure, figure_dir, {"trial_level": level_from_trials})



                figure = plot_group_dose_response(level_from_trials, figure_dir, formats, args.dpi, palette, data_type="trial", trial_df_raw=trial_df, signature=signature.upper())

                append_figure_with_stats(figures, figure, figure_dir, {"trial_level": level_from_trials, "trial_raw": trial_df})

        figure = rename_figure_for_signature(plot_vas_br_relationship(level_df, figure_dir, formats, args.dpi, palette), signature)

        append_figure_with_stats(figures, figure, figure_dir, {"condition_level": level_df})

        figure = rename_figure_for_signature(plot_subject_metric_panels(subject_metrics, group_metrics, figure_dir, formats, args.dpi, palette), signature)
        append_figure_with_stats(figures, figure, figure_dir, {"subject_metrics": subject_metrics})

        figure = rename_figure_for_signature(plot_subject_metric_heatmap(subject_metrics, figure_dir, formats, args.dpi), signature)
        append_figure_with_stats(figures, figure, figure_dir, {"subject_metrics": subject_metrics})

        figure = rename_figure_for_signature(plot_subject_roc_curves(trial_df, figure_dir, formats, args.dpi, palette, args.pain_threshold), signature)
        append_figure_with_stats(figures, figure, figure_dir, {"trial_level": trial_df})

        figure = rename_figure_for_signature(plot_trial_temperature_distributions(trial_df, figure_dir, formats, args.dpi, palette), signature)
        append_figure_with_stats(figures, figure, figure_dir, {"trial_level": trial_df})

        figure = rename_figure_for_signature(plot_temperature_vas_curve(level_df, figure_dir, formats, args.dpi), signature)
        append_figure_with_stats(figures, figure, figure_dir, {"condition_level": level_df})

        figure = rename_figure_for_signature(plot_bland_altman(level_df, figure_dir, formats, args.dpi), signature)
        append_figure_with_stats(figures, figure, figure_dir, {"condition_level": level_df})

        figure = rename_figure_for_signature(plot_effect_sizes_by_temperature(level_df, figure_dir, formats, args.dpi), signature)
        append_figure_with_stats(figures, figure, figure_dir, {"condition_level": level_df})

        figure = rename_figure_for_signature(plot_residual_diagnostics(level_df, figure_dir, formats, args.dpi), signature)
        append_figure_with_stats(figures, figure, figure_dir, {"condition_level": level_df})

        figure = rename_figure_for_signature(plot_within_subject_reliability(trial_df, figure_dir, formats, args.dpi, palette), signature)
        append_figure_with_stats(figures, figure, figure_dir, {"trial_level": trial_df})

        figure = rename_figure_for_signature(plot_correlation_matrix(level_df, figure_dir, formats, args.dpi), signature)
        append_figure_with_stats(figures, figure, figure_dir, {"condition_level": level_df})

        figure = rename_figure_for_signature(plot_temporal_dynamics_enhanced(trial_df, figure_dir, formats, args.dpi, palette), signature)
        append_figure_with_stats(figures, figure, figure_dir, {"trial_level": trial_df})

        figure = rename_figure_for_signature(plot_trial_order_effects(trial_df, figure_dir, formats, args.dpi, palette), signature)
        append_figure_with_stats(figures, figure, figure_dir, {"trial_level": trial_df})

        figure = rename_figure_for_signature(plot_variance_components(trial_df, figure_dir, formats, args.dpi), signature)
        append_figure_with_stats(figures, figure, figure_dir, {"trial_level": trial_df})

        figure = rename_figure_for_signature(plot_pairwise_temperature_contrasts(level_df, figure_dir, formats, args.dpi), signature)
        append_figure_with_stats(figures, figure, figure_dir, {"condition_level": level_df})

        figure = rename_figure_for_signature(plot_bootstrap_uncertainty_distributions(level_df, figure_dir, formats, args.dpi), signature)
        append_figure_with_stats(figures, figure, figure_dir, {"condition_level": level_df})

        figure = rename_figure_for_signature(plot_prediction_intervals(level_df, figure_dir, formats, args.dpi), signature)
        append_figure_with_stats(figures, figure, figure_dir, {"condition_level": level_df})

        figure = rename_figure_for_signature(plot_raincloud_distributions(trial_df, figure_dir, formats, args.dpi, palette), signature)
        append_figure_with_stats(figures, figure, figure_dir, {"trial_level": trial_df})

        figure = rename_figure_for_signature(plot_cumulative_distributions(trial_df, figure_dir, formats, args.dpi), signature)
        append_figure_with_stats(figures, figure, figure_dir, {"trial_level": trial_df})

        figure = rename_figure_for_signature(plot_response_categories(level_df, figure_dir, formats, args.dpi, palette), signature)
        append_figure_with_stats(figures, figure, figure_dir, {"condition_level": level_df})



        import datetime
        import platform
        
        manifest = {
            "signature": signature,
            "figures": [fig.__dict__ for fig in figures],
            "log_file": str(LOG_FILE),
            "subjects_plotted": subjects_with_data,
            "n_subjects": len(subjects),
            "generation_info": {
                "timestamp": datetime.datetime.now().isoformat(),
                "python_version": platform.python_version(),
                "matplotlib_version": matplotlib.__version__,
                "seaborn_version": sns.__version__,
                "script_version": "4.0_multi_signature",
            },
        }
        manifest_path = figure_dir / "figures_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        def _range(series: pd.Series) -> Optional[Tuple[float, float]]:
            series = series.dropna()
            if series.empty:
                return None
            return float(series.min()), float(series.max())

        removed_trials_records: List[Dict[str, Optional[float]]] = []
        if not removed_trials.empty:
            for _, row in removed_trials.iterrows():
                removed_trials_records.append(
                    {
                        "subject": row.get("subject"),
                        "run": int(row["run"]) if pd.notna(row.get("run")) else None,
                        "trial_regressor": row.get("trial_regressor"),
                        "temp_celsius": float(row["temp_celsius"]) if pd.notna(row.get("temp_celsius")) else None,
                        "br_score": float(row["br_score"]) if pd.notna(row.get("br_score")) else None,
                    }
                )

        figure_stats = {
            "signature": signature,
            "n_subjects_configured": len(subjects),
            "n_subjects_with_condition_data": int(level_df["subject"].nunique()) if "subject" in level_df.columns else 0,
            "n_subjects_with_trial_data": int(trial_df["subject"].nunique()) if "subject" in trial_df.columns else 0,
            "n_subjects_with_metrics": int(subject_metrics["subject"].nunique()) if "subject" in subject_metrics.columns else 0,
            "n_condition_rows": int(len(level_df)),
            "n_trial_rows": int(len(trial_df)),
            "n_trial_rows_raw": int(len(trial_df_raw)),
            "n_trials_removed_outliers": int(n_extreme_trials_removed),
            "trial_outlier_threshold": args.trial_outlier_threshold,
            "temperature_celsius_range": _range(level_df["temp_celsius"]) if "temp_celsius" in level_df.columns else None,
            "vas_range": _range(level_df["mean_vas"]) if "mean_vas" in level_df.columns else None,
            "generated_figures": [fig.name for fig in figures],
            "generated_stats_files": {fig.name: fig.stats_paths for fig in figures if fig.stats_paths},
            "removed_extreme_trials": removed_trials_records,
        }
        stats_path = figure_dir / "figure_stats.json"
        stats_path.write_text(json.dumps(figure_stats, indent=2))

        if not figures:
            all_success = False
    
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
