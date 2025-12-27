from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from eeg_pipeline.infra.paths import deriv_stats_path, ensure_dir
from eeg_pipeline.plotting.io.figures import save_fig


def _find_first(stats_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(stats_dir.glob(pattern))
    return matches[0] if matches else None


def plot_stability_groupwise(
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

    stab_path = _find_first(stats_dir, "stability_groupwise*.tsv")
    if stab_path is None or not stab_path.exists():
        return {}

    df = pd.read_csv(stab_path, sep="\t")
    if df.empty or "r_overall" not in df.columns or "r_group_std" not in df.columns:
        return {}

    out_dir = plots_dir / "stability"
    ensure_dir(out_dir)

    x = pd.to_numeric(df["r_overall"], errors="coerce")
    y = pd.to_numeric(df["r_group_std"], errors="coerce")
    ok = x.notna() & y.notna()
    if int(ok.sum()) < 3:
        return {}

    # Highlight the most stable strong effects
    score = (x.abs() / (y + 1e-6)).where(ok, np.nan)
    top_idx = score.sort_values(ascending=False).head(10).index.tolist()

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=150)
    ax.scatter(x[ok], y[ok], s=18, alpha=0.65)
    ax.set_xlabel("Overall association (r)")
    ax.set_ylabel("Across-group variability (std r)")
    ax.set_title(f"sub-{subject} {task}: stability across {df.get('group_column', ['groups'])[0]}")
    ax.grid(True, alpha=0.2)

    for i in top_idx:
        try:
            ax.scatter([x.loc[i]], [y.loc[i]], s=40)
            feat = str(df.loc[i, "feature"]) if "feature" in df.columns else ""
            ax.annotate(feat, (x.loc[i], y.loc[i]), fontsize=7, alpha=0.85)
        except Exception:
            continue

    fig.tight_layout()
    out_path = out_dir / f"sub-{subject}_stability_groupwise.png"
    save_fig(fig, out_path, logger=logger)
    plt.close(fig)
    logger.info("Saved stability plot: %s", out_path.name)
    return {"stability_groupwise": out_path}


__all__ = ["plot_stability_groupwise"]

