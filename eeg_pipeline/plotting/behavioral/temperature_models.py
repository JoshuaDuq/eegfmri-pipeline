from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from eeg_pipeline.infra.paths import deriv_stats_path, ensure_dir
from eeg_pipeline.plotting.io.figures import save_fig


def _find_first(stats_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(stats_dir.glob(pattern))
    return matches[0] if matches else None


def plot_temperature_models(
    *,
    subject: str,
    task: str,
    deriv_root: Path,
    config: Any,
    logger: Any,
    plots_dir: Path,
) -> Dict[str, Path]:
    import matplotlib.pyplot as plt

    stats_dir = deriv_stats_path(deriv_root, subject)
    if not stats_dir.exists():
        return {}

    out_dir = plots_dir / "temperature_models"
    ensure_dir(out_dir)

    trials_path = _find_first(stats_dir, "trials*.tsv") or _find_first(stats_dir, "trials*.parquet")
    if trials_path is None or not trials_path.exists():
        return {}

    from eeg_pipeline.infra.tsv import read_table
    df = read_table(trials_path)

    if "temperature" not in df.columns or "rating" not in df.columns:
        return {}

    t = pd.to_numeric(df["temperature"], errors="coerce")
    y = pd.to_numeric(df["rating"], errors="coerce")
    ok = t.notna() & y.notna()
    if int(ok.sum()) < 5:
        return {}

    # Optional metadata (for annotation only).
    cmp_meta_path = _find_first(stats_dir, "temperature_model_comparison*.metadata.json")
    bp_meta_path = _find_first(stats_dir, "temperature_breakpoint_test*.metadata.json")
    best_model = None
    best_break = None
    try:
        if cmp_meta_path and cmp_meta_path.exists():
            best_model = json.loads(cmp_meta_path.read_text()).get("best_model")
    except Exception:
        pass
    try:
        if bp_meta_path and bp_meta_path.exists():
            best_break = json.loads(bp_meta_path.read_text()).get("best_breakpoint")
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    ax.scatter(t[ok], y[ok], s=16, alpha=0.65)

    # Binned means for visual stability
    try:
        bins = np.unique(np.quantile(t[ok], np.linspace(0, 1, 9)))
        if bins.size >= 4:
            df_b = pd.DataFrame({"t": t[ok].to_numpy(), "y": y[ok].to_numpy()})
            df_b["bin"] = pd.cut(df_b["t"], bins=bins, include_lowest=True, duplicates="drop")
            m = df_b.groupby("bin", observed=True).agg(t_mean=("t", "mean"), y_mean=("y", "mean")).dropna()
            ax.plot(m["t_mean"], m["y_mean"], linewidth=2.0)
    except Exception:
        pass

    if best_break is not None and np.isfinite(best_break):
        ax.axvline(float(best_break), linestyle="--", linewidth=1.5)

    title = f"sub-{subject} {task}: rating vs temperature"
    if best_model:
        title += f" (best: {best_model})"
    ax.set_title(title)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Pain rating")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    out_path = out_dir / f"sub-{subject}_temperature_models.png"
    save_fig(fig, out_path, logger=logger)
    plt.close(fig)
    logger.info("Saved temperature model plot: %s", out_path.name)
    return {"temperature_models": out_path}


__all__ = ["plot_temperature_models"]
