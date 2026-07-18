"""Non-interactive Matplotlib initialization for pipeline processes."""

from __future__ import annotations

import logging
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def setup_matplotlib(config: Any = None) -> None:
    """Configure the non-interactive plotting backend and shared render defaults."""
    backend_key = "_backend_set_for_pipeline"
    if not getattr(matplotlib, backend_key, False):
        try:
            matplotlib.use("Agg", force=False)
            setattr(matplotlib, backend_key, True)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Failed to configure matplotlib Agg backend: %s", exc
            )

    sns.set_theme(context="paper", style="white", font_scale=1.05)
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.color": "0.85",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
        }
    )


__all__ = ["setup_matplotlib"]
