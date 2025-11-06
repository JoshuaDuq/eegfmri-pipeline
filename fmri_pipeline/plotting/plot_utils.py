from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import seaborn as sns

from utils import get_log_function

log, _ = get_log_function(Path(__file__).stem)

EXTREME_TRIAL_THRESHOLD = 1000.0


@dataclass(frozen=True)
class FigureSpec:
    name: str
    description: str
    paths: List[str]
    stats_paths: List[str] = field(default_factory=list)


###################################################################
# Matplotlib Configuration
###################################################################

def configure_matplotlib(context: str = "paper") -> None:
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


###################################################################
# Figure Saving
###################################################################

def save_figure(fig: plt.Figure, base_name: str, figure_dir: Path, formats: Sequence[str], dpi: int) -> List[str]:
    fig.patch.set_facecolor("white")
    for axis in fig.axes:
        axis.set_facecolor("white")
    paths = []
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


###################################################################
# Color Palette
###################################################################

def subject_palette(subjects: Sequence[str]) -> Dict[str, Tuple[float, float, float]]:
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


###################################################################
# Panel Labels
###################################################################

def _add_panel_label(axis: plt.Axes, label: str, x: float = -0.12, y: float = 1.05) -> None:
    axis.text(
        x, y, label,
        transform=axis.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right"
    )


###################################################################
# Formatting
###################################################################

def _format_pvalue(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)


def _compute_correlation_with_ci(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> dict:
    corr_func = stats.pearsonr if method == "pearson" else stats.spearmanr
    r, p = corr_func(x, y)
    
    n_boot = 1000
    boot_rs = []
    n = len(x)
    for _ in range(n_boot):
        boot_idx = np.random.choice(n, n, replace=True)
        boot_r, _ = corr_func(x[boot_idx], y[boot_idx])
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
    data = data.dropna()
    if len(data) < 2:
        return np.nan
    return stats.sem(data)


###################################################################
# Data Loading
###################################################################

def load_subject_metrics(scores_dir: Path, subjects: Iterable[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for subject in subjects:
        metrics_path = scores_dir / subject / "subject_metrics.tsv"
        if not metrics_path.exists():
            continue
        try:
            df = pd.read_csv(metrics_path, sep="\t")
            df["subject"] = subject
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    metrics = pd.concat(frames, ignore_index=True)
    object_cols = metrics.select_dtypes(include="object").columns.difference(["subject", "notes"])
    for col in object_cols:
        metrics[col] = pd.to_numeric(metrics[col], errors="coerce")
    return metrics


def load_level_br(scores_dir: Path, subjects: Iterable[str], signature: str = "nps") -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    filename = f"level_{signature}.tsv"
    for subject in subjects:
        level_path = scores_dir / subject / filename
        if not level_path.exists():
            if signature == "nps":
                legacy_path = scores_dir / subject / "level_br.tsv"
                if legacy_path.exists():
                    level_path = legacy_path
                else:
                    continue
            else:
                continue
        try:
            df = pd.read_csv(level_path, sep="\t")
            df["subject"] = subject
            score_col = f"{signature}_score"
            if score_col in df.columns and "br_score" not in df.columns:
                df["br_score"] = df[score_col]
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_trial_br(scores_dir: Path, subjects: Iterable[str], signature: str = "nps") -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    filename = f"trial_{signature}.tsv"
    for subject in subjects:
        trial_path = scores_dir / subject / filename
        if not trial_path.exists():
            if signature == "nps":
                legacy_path = scores_dir / subject / "trial_br.tsv"
                if legacy_path.exists():
                    trial_path = legacy_path
                else:
                    continue
            else:
                continue
        try:
            df = pd.read_csv(trial_path, sep="\t")
            df["subject"] = subject
            score_col = f"{signature}_score"
            if score_col in df.columns and "br_score" not in df.columns:
                df["br_score"] = df[score_col]
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_group_metrics(group_dir: Path) -> Dict[str, Dict[str, float]]:
    import json
    verbose_path = group_dir / "group_stats_verbose.json"
    if not verbose_path.exists():
        return {}
    try:
        data = json.loads(verbose_path.read_text())
        return data.get("metrics", {})
    except Exception:
        return {}


###################################################################
# Data Processing
###################################################################

def aggregate_trials_to_conditions(trial_df: pd.DataFrame) -> pd.DataFrame:
    if trial_df.empty:
        return pd.DataFrame()
    
    required_cols = {"subject", "temp_celsius", "br_score"}
    if not required_cols.issubset(trial_df.columns):
        return pd.DataFrame()
    
    agg_dict = {
        "br_score": "mean",
        "n_trials_removed_run": "sum",
    }
    
    if "vas_rating" in trial_df.columns:
        agg_dict["vas_rating"] = "mean"
    if "pain_binary" in trial_df.columns:
        agg_dict["pain_binary"] = "mean"
    
    level_from_trials = trial_df.groupby(["subject", "temp_celsius"]).agg(agg_dict).reset_index()
    
    if "vas_rating" in level_from_trials.columns:
        level_from_trials.rename(columns={"vas_rating": "mean_vas"}, inplace=True)
    
    trial_counts = trial_df.groupby(["subject", "temp_celsius"]).size().reset_index(name="n_trials")
    level_from_trials = level_from_trials.merge(trial_counts, on=["subject", "temp_celsius"], how="left")
    
    return level_from_trials


def filter_extreme_trials(trial_df: pd.DataFrame, threshold: float = EXTREME_TRIAL_THRESHOLD) -> pd.DataFrame:
    if trial_df.empty or "br_score" not in trial_df.columns:
        return trial_df
    mask = trial_df["br_score"].abs() > threshold
    removed = trial_df.loc[mask].copy()
    if removed.empty:
        return trial_df
    cleaned = trial_df.loc[~mask].copy()
    log(
        f"Excluded {len(removed)} trial(s) with |NPS| > {threshold:.1f} (extreme outliers).",
        "WARNING",
    )
    return cleaned


###################################################################
# Statistics CSV
###################################################################

def save_stats_csv(stats_dict: Dict[str, Any], output_path: Path) -> None:
    if isinstance(stats_dict, pd.DataFrame):
        df = stats_dict.copy()
    else:
        try:
            df = pd.DataFrame([stats_dict] if not isinstance(stats_dict, list) else stats_dict)
        except Exception:
            df = pd.DataFrame()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def attach_basic_stats(fig_spec: Optional[FigureSpec], figure_dir: Path, stats_sources: Dict[str, Optional[pd.DataFrame]], extra_rows: Optional[pd.DataFrame] = None) -> Optional[FigureSpec]:
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
    stats_path = figure_dir / f"{fig_spec.name}_stats.csv"
    save_stats_csv(stats_df, stats_path)

    return FigureSpec(
        name=fig_spec.name,
        description=fig_spec.description,
        paths=fig_spec.paths,
        stats_paths=[str(stats_path)],
    )


def append_figure_with_stats(figures: List[FigureSpec], fig_spec: Optional[FigureSpec], figure_dir: Path, stats_sources: Dict[str, Optional[pd.DataFrame]], extra_rows: Optional[pd.DataFrame] = None) -> None:
    if fig_spec is None:
        return
    enriched = attach_basic_stats(fig_spec, figure_dir, stats_sources, extra_rows)
    figures.append(enriched if enriched is not None else fig_spec)

