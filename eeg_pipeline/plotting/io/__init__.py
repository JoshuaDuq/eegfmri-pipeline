from __future__ import annotations

from .collections import collect_significant_plots
from .figures import (
    build_footer,
    unwrap_figure,
    get_behavior_footer,
    get_band_color,
    logratio_to_pct,
    pct_to_logratio,
    get_viz_params,
    plot_topomap_on_ax,
    robust_sym_vlim,
    setup_matplotlib,
    extract_eeg_picks,
    log_if_present,
    get_default_config,
    save_fig,
)

__all__ = [
    "build_footer",
    "unwrap_figure",
    "get_behavior_footer",
    "get_band_color",
    "logratio_to_pct",
    "pct_to_logratio",
    "get_viz_params",
    "plot_topomap_on_ax",
    "robust_sym_vlim",
    "setup_matplotlib",
    "extract_eeg_picks",
    "log_if_present",
    "get_default_config",
    "save_fig",
    "collect_significant_plots",
]
