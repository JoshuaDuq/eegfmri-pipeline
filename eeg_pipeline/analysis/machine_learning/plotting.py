"""Machine-learning result plotting (publication-style, mode-aware)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MLPlottingOptions:
    formats: tuple[str, ...] = ("png",)
    dpi: int = 300
    top_n_features: int = 20
    include_diagnostics: bool = True


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _normalize_formats(formats: Optional[Iterable[str]]) -> tuple[str, ...]:
    if formats is None:
        return ("png",)
    seen: List[str] = []
    for fmt in formats:
        f = str(fmt).strip().lower()
        if f in {"png", "pdf", "svg"} and f not in seen:
            seen.append(f)
    return tuple(seen) if seen else ("png",)


def _make_options(
    *,
    formats: Optional[Iterable[str]],
    dpi: Optional[int],
    top_n_features: Optional[int],
    include_diagnostics: Optional[bool],
) -> MLPlottingOptions:
    safe_dpi = 300 if dpi is None else max(72, int(dpi))
    safe_top_n = 20 if top_n_features is None else max(1, int(top_n_features))
    return MLPlottingOptions(
        formats=_normalize_formats(formats),
        dpi=safe_dpi,
        top_n_features=safe_top_n,
        include_diagnostics=True if include_diagnostics is None else bool(include_diagnostics),
    )


def _apply_publication_style(plt: Any, *, dpi: int) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.3,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.5,
        }
    )


def _safe_read_tsv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return None


def _save(fig: Any, base_path: Path, *, opts: MLPlottingOptions) -> List[Path]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    stem = str(base_path.with_suffix(""))
    out: List[Path] = []
    for fmt in opts.formats:
        path = Path(f"{stem}.{fmt}")
        fig.savefig(path, bbox_inches="tight", dpi=opts.dpi)
        out.append(path)
    fig.clf()
    return out


def _finite_xy(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    x = pd.to_numeric(df.get(x_col), errors="coerce")
    y = pd.to_numeric(df.get(y_col), errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y)
    return pd.DataFrame({x_col: x[mask], y_col: y[mask]})


def _plot_regression(results_dir: Path, opts: MLPlottingOptions) -> List[Path]:
    plt = _import_pyplot()
    _apply_publication_style(plt, dpi=opts.dpi)
    pred = _safe_read_tsv(results_dir / "data" / "loso_predictions.tsv")
    if pred is None:
        pred = _safe_read_tsv(results_dir / "data" / "cv_predictions.tsv")
    if pred is None:
        return []

    out: List[Path] = []
    clean = _finite_xy(pred, "y_true", "y_pred")
    if clean.empty:
        return out

    y_true = clean["y_true"].to_numpy(dtype=float)
    y_pred = clean["y_pred"].to_numpy(dtype=float)
    resid = y_true - y_pred
    r = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else np.nan
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    if opts.include_diagnostics:
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
        ax = axes[0]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.8))

    ax.scatter(y_true, y_pred, s=18, color="#1f77b4", alpha=0.8, edgecolors="white", linewidths=0.4)
    vmin = float(np.nanmin([y_true.min(), y_pred.min()]))
    vmax = float(np.nanmax([y_true.max(), y_pred.max()]))
    ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", color="#444444", linewidth=1.0, label="Identity")
    if len(y_true) >= 2:
        try:
            m, b = np.polyfit(y_true, y_pred, 1)
            xx = np.linspace(vmin, vmax, 100)
            ax.plot(xx, m * xx + b, color="#d62728", linewidth=1.3, label="Fit")
        except Exception as exc:
            logger.debug("Skipping regression fit overlay in prediction agreement plot: %s", exc)
    ax.set_xlabel("Observed target")
    ax.set_ylabel("Predicted target")
    ax.set_title("Prediction Agreement")
    ax.grid(True, axis="both")
    ax.legend(frameon=False, loc="best")
    ax.text(
        0.02,
        0.98,
        f"r={r:.2f}\nRMSE={rmse:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "#cccccc"},
    )

    if opts.include_diagnostics:
        ax2 = axes[1]
        ax2.axhline(0, color="#555555", linewidth=1.0, linestyle="--")
        ax2.scatter(y_pred, resid, s=16, color="#2ca02c", alpha=0.8, edgecolors="white", linewidths=0.3)
        ax2.set_xlabel("Predicted target")
        ax2.set_ylabel("Residual (observed - predicted)")
        ax2.set_title("Residual Diagnostics")
        ax2.grid(True, axis="both")

    out.extend(_save(fig, results_dir / "plots" / "regression_diagnostics", opts=opts))
    plt.close(fig)
    return out


def _plot_classification(results_dir: Path, opts: MLPlottingOptions) -> List[Path]:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

    plt = _import_pyplot()
    _apply_publication_style(plt, dpi=opts.dpi)
    pred = _safe_read_tsv(results_dir / "data" / "loso_predictions.tsv")
    if pred is None:
        pred = _safe_read_tsv(results_dir / "data" / "cv_predictions.tsv")
    if pred is None:
        return []
    if "y_true" not in pred.columns or "y_pred" not in pred.columns:
        return []

    y_true = pd.to_numeric(pred["y_true"], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(pred["y_pred"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_i = y_true[valid].astype(int)
    y_pred_i = y_pred[valid].astype(int)
    if len(y_true_i) == 0:
        return []

    out: List[Path] = []
    cm = confusion_matrix(y_true_i, y_pred_i, labels=[0, 1])
    fig_cm, ax_cm = plt.subplots(1, 1, figsize=(3.8, 3.6))
    im = ax_cm.imshow(cm, cmap="Blues")
    fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Pred 0", "Pred 1"])
    ax_cm.set_yticklabels(["True 0", "True 1"])
    ax_cm.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="#222222")
    out.extend(_save(fig_cm, results_dir / "plots" / "classification_confusion_matrix", opts=opts))
    plt.close(fig_cm)

    if "y_prob" in pred.columns and opts.include_diagnostics:
        y_prob = pd.to_numeric(pred["y_prob"], errors="coerce").to_numpy(dtype=float)
        p_mask = np.isfinite(y_prob) & np.isfinite(y_true)
        y_tp = y_true[p_mask].astype(int)
        y_pp = y_prob[p_mask]
        if len(y_tp) > 2 and len(np.unique(y_tp)) == 2:
            fpr, tpr, _ = roc_curve(y_tp, y_pp)
            roc_auc = float(auc(fpr, tpr))
            prec, rec, _ = precision_recall_curve(y_tp, y_pp)
            fig_curve, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
            axes[0].plot(fpr, tpr, color="#1f77b4", label=f"AUC={roc_auc:.2f}")
            axes[0].plot([0, 1], [0, 1], "--", color="#888888", linewidth=1.0)
            axes[0].set_xlabel("False Positive Rate")
            axes[0].set_ylabel("True Positive Rate")
            axes[0].set_title("ROC Curve")
            axes[0].legend(frameon=False, loc="lower right")
            axes[0].grid(True)

            axes[1].plot(rec, prec, color="#d62728")
            axes[1].set_xlabel("Recall")
            axes[1].set_ylabel("Precision")
            axes[1].set_title("Precision-Recall Curve")
            axes[1].grid(True)
            out.extend(_save(fig_curve, results_dir / "plots" / "classification_roc_pr", opts=opts))
            plt.close(fig_curve)

            prob_true, prob_pred = calibration_curve(y_tp, y_pp, n_bins=min(10, max(3, len(y_tp) // 5)))
            fig_cal, ax_cal = plt.subplots(1, 1, figsize=(4.0, 3.6))
            ax_cal.plot([0, 1], [0, 1], "--", color="#666666", linewidth=1.0, label="Ideal")
            ax_cal.plot(prob_pred, prob_true, marker="o", color="#2ca02c", label="Model")
            ax_cal.set_xlabel("Predicted probability")
            ax_cal.set_ylabel("Observed frequency")
            ax_cal.set_title("Calibration")
            ax_cal.legend(frameon=False, loc="best")
            ax_cal.grid(True)
            out.extend(_save(fig_cal, results_dir / "plots" / "classification_calibration", opts=opts))
            plt.close(fig_cal)

    return out


def _plot_time_generalization(results_dir: Path, opts: MLPlottingOptions) -> List[Path]:
    plt = _import_pyplot()
    _apply_publication_style(plt, dpi=opts.dpi)
    npz_path = results_dir / "time_generalization_regression.npz"
    if not npz_path.exists():
        return []
    with np.load(npz_path, allow_pickle=False) as data:
        r = data.get("r_matrix")
        r2 = data.get("r2_matrix")
        centers = data.get("window_centers")
        sig = data.get("sig_fdr")

    if r is None or r2 is None or np.size(r) == 0:
        return []
    r = np.asarray(r, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    n = r.shape[0]
    ticks = np.arange(n)
    labels = [str(i) for i in ticks]
    if centers is not None and len(centers) == n:
        labels = [f"{float(c):.2f}" for c in centers]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    lim = max(abs(np.nanmin(r)), abs(np.nanmax(r)))
    im0 = axes[0].imshow(r, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[0].set_title("Time Generalization (r)")
    axes[0].set_xlabel("Test window")
    axes[0].set_ylabel("Train window")
    axes[0].set_xticks(ticks[:: max(1, n // 6)])
    axes[0].set_yticks(ticks[:: max(1, n // 6)])
    axes[0].set_xticklabels(labels[:: max(1, n // 6)], rotation=45, ha="right")
    axes[0].set_yticklabels(labels[:: max(1, n // 6)])
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    if sig is not None and np.shape(sig) == np.shape(r):
        sig_mask = np.asarray(sig, dtype=bool)
        ys, xs = np.where(sig_mask)
        if len(xs) > 0:
            axes[0].scatter(xs, ys, s=7, facecolors="none", edgecolors="black", linewidths=0.4)

    im1 = axes[1].imshow(r2, cmap="viridis")
    axes[1].set_title("Time Generalization (R2)")
    axes[1].set_xlabel("Test window")
    axes[1].set_ylabel("Train window")
    axes[1].set_xticks(ticks[:: max(1, n // 6)])
    axes[1].set_yticks(ticks[:: max(1, n // 6)])
    axes[1].set_xticklabels(labels[:: max(1, n // 6)], rotation=45, ha="right")
    axes[1].set_yticklabels(labels[:: max(1, n // 6)])
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    out = _save(fig, results_dir / "plots" / "time_generalization_matrices", opts=opts)
    plt.close(fig)
    return out


def _plot_model_comparison(results_dir: Path, opts: MLPlottingOptions) -> List[Path]:
    plt = _import_pyplot()
    _apply_publication_style(plt, dpi=opts.dpi)
    df = _safe_read_tsv(results_dir / "metrics" / "model_comparison.tsv")
    if df is None or df.empty:
        return []
    if "model" not in df.columns:
        return []

    metrics = []
    for metric in ("r2", "mae"):
        if metric in df.columns:
            vals = pd.to_numeric(df[metric], errors="coerce")
            if np.isfinite(vals).any():
                metrics.append(metric)
    if not metrics:
        return []

    models = sorted(df["model"].astype(str).unique().tolist())
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.1 * len(metrics), 3.7))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        series_by_model = [
            pd.to_numeric(df.loc[df["model"].astype(str) == m, metric], errors="coerce").dropna().to_numpy()
            for m in models
        ]
        ax.boxplot(
            series_by_model,
            tick_labels=models,
            patch_artist=True,
            boxprops={"facecolor": "#cfe2f3", "alpha": 0.8},
        )
        for idx, vals in enumerate(series_by_model, start=1):
            if len(vals) == 0:
                continue
            x = np.full(len(vals), idx, dtype=float) + np.linspace(-0.07, 0.07, len(vals))
            ax.scatter(x, vals, s=14, color="#2f5597", alpha=0.7, linewidths=0)
        ax.set_title(f"Model comparison: {metric.upper()}")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric.upper())
        ax.grid(True, axis="y")

    out = _save(fig, results_dir / "plots" / "model_comparison_metrics", opts=opts)
    plt.close(fig)
    return out


def _plot_incremental_validity(results_dir: Path, opts: MLPlottingOptions) -> List[Path]:
    plt = _import_pyplot()
    _apply_publication_style(plt, dpi=opts.dpi)
    df = _safe_read_tsv(results_dir / "metrics" / "incremental_validity.tsv")
    if df is None or df.empty:
        return []
    if not {"r2_baseline", "r2_full", "delta_r2"}.issubset(set(df.columns)):
        return []

    base = pd.to_numeric(df["r2_baseline"], errors="coerce").to_numpy(dtype=float)
    full = pd.to_numeric(df["r2_full"], errors="coerce").to_numpy(dtype=float)
    delta = pd.to_numeric(df["delta_r2"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(base) & np.isfinite(full) & np.isfinite(delta)
    if not np.any(mask):
        return []
    base = base[mask]
    full = full[mask]
    delta = delta[mask]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for b, f in zip(base, full):
        axes[0].plot([0, 1], [b, f], color="#8a8a8a", alpha=0.7, linewidth=1.0)
        axes[0].scatter([0, 1], [b, f], color=["#d62728", "#1f77b4"], s=18, zorder=3)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Baseline", "Baseline + EEG"])
    axes[0].set_ylabel("R2")
    axes[0].set_title("Per-fold paired performance")
    axes[0].grid(True, axis="y")

    axes[1].hist(delta, bins=min(12, max(4, len(delta))), color="#2ca02c", alpha=0.85, edgecolor="white")
    axes[1].axvline(np.mean(delta), color="#111111", linestyle="--", linewidth=1.1, label=f"mean={np.mean(delta):.3f}")
    axes[1].set_xlabel("Delta R2 (full - baseline)")
    axes[1].set_ylabel("Fold count")
    axes[1].set_title("Incremental validity distribution")
    axes[1].legend(frameon=False, loc="best")
    axes[1].grid(True, axis="y")

    out = _save(fig, results_dir / "plots" / "incremental_validity_summary", opts=opts)
    plt.close(fig)
    return out


def _plot_uncertainty(results_dir: Path, opts: MLPlottingOptions) -> List[Path]:
    plt = _import_pyplot()
    _apply_publication_style(plt, dpi=opts.dpi)
    df = _safe_read_tsv(results_dir / "data" / "prediction_intervals.tsv")
    if df is None or df.empty:
        return []
    needed = {"y_pred", "lower", "upper", "y_true"}
    if not needed.issubset(set(df.columns)):
        return []

    pred = pd.to_numeric(df["y_pred"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(df["lower"], errors="coerce").to_numpy(dtype=float)
    up = pd.to_numeric(df["upper"], errors="coerce").to_numpy(dtype=float)
    truth = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(pred) & np.isfinite(low) & np.isfinite(up) & np.isfinite(truth)
    if not np.any(mask):
        return []
    pred, low, up, truth = pred[mask], low[mask], up[mask], truth[mask]
    order = np.argsort(pred)
    pred, low, up, truth = pred[order], low[order], up[order], truth[order]
    cov = float(np.mean((truth >= low) & (truth <= up)))

    fig, ax = plt.subplots(1, 1, figsize=(8.4, 3.6))
    x = np.arange(len(pred))
    ax.fill_between(x, low, up, color="#a6cee3", alpha=0.45, label="Prediction interval")
    ax.plot(x, pred, color="#1f78b4", linewidth=1.2, label="Prediction")
    ax.scatter(x, truth, s=12, color="#e31a1c", alpha=0.8, label="Observed")
    ax.set_xlabel("Samples (sorted by prediction)")
    ax.set_ylabel("Target")
    ax.set_title(f"Conformal intervals (empirical coverage={cov:.1%})")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="best")

    out = _save(fig, results_dir / "plots" / "uncertainty_intervals", opts=opts)
    plt.close(fig)
    return out


def _plot_feature_importance(results_dir: Path, opts: MLPlottingOptions, *, mode: str) -> List[Path]:
    plt = _import_pyplot()
    _apply_publication_style(plt, dpi=opts.dpi)
    if mode == "shap":
        df = _safe_read_tsv(results_dir / "importance" / "shap_importance.tsv")
        value_col = "shap_importance"
        err_col = "shap_std_across_folds" if df is not None and "shap_std_across_folds" in df.columns else "shap_std"
        out_name = "shap_importance_top_features"
    else:
        df = _safe_read_tsv(results_dir / "importance" / "permutation_importance.tsv")
        value_col = "importance_mean"
        err_col = "importance_std"
        out_name = "permutation_importance_top_features"
    if df is None or df.empty:
        return []
    if "feature" not in df.columns or value_col not in df.columns:
        return []

    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df[np.isfinite(df[value_col])]
    if df.empty:
        return []
    df = df.sort_values(value_col, ascending=False).head(opts.top_n_features)
    vals = df[value_col].to_numpy(dtype=float)
    feats = df["feature"].astype(str).tolist()
    errs = None
    if err_col in df.columns:
        err_vals = pd.to_numeric(df[err_col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(err_vals).any():
            errs = np.where(np.isfinite(err_vals), err_vals, 0.0)

    fig_h = max(3.6, 0.24 * len(feats) + 1.2)
    fig, ax = plt.subplots(1, 1, figsize=(7.4, fig_h))
    y = np.arange(len(feats))
    ax.barh(y, vals, xerr=errs, color="#4c78a8", alpha=0.9, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(feats)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top feature importance")
    ax.grid(True, axis="x")

    out = _save(fig, results_dir / "plots" / out_name, opts=opts)
    plt.close(fig)
    return out


def generate_ml_mode_plots(
    mode: str,
    results_dir: Path,
    logger: Any,
    *,
    formats: Optional[Sequence[str]] = None,
    dpi: Optional[int] = None,
    top_n_features: Optional[int] = None,
    include_diagnostics: Optional[bool] = None,
) -> List[str]:
    """Generate best-effort publication plots for one ML mode."""
    opts = _make_options(
        formats=formats,
        dpi=dpi,
        top_n_features=top_n_features,
        include_diagnostics=include_diagnostics,
    )
    mode_key = str(mode).strip().lower()
    plotters: Dict[str, Callable[[Path, MLPlottingOptions], List[Path]]] = {
        "regression": _plot_regression,
        "classify": _plot_classification,
        "timegen": _plot_time_generalization,
        "model_comparison": _plot_model_comparison,
        "incremental_validity": _plot_incremental_validity,
        "uncertainty": _plot_uncertainty,
        "shap": lambda d, o: _plot_feature_importance(d, o, mode="shap"),
        "permutation": lambda d, o: _plot_feature_importance(d, o, mode="permutation"),
    }
    if mode_key not in plotters:
        return []

    try:
        outputs = plotters[mode_key](results_dir, opts)
        return [str(p) for p in outputs]
    except Exception as exc:
        if logger is not None:
            logger.warning("ML plotting failed for mode=%s: %s", mode_key, exc)
        return []


__all__ = ["generate_ml_mode_plots", "MLPlottingOptions"]
