from __future__ import annotations

import numpy as np
from typing import Any


def sliding_window_centers(config: Any, n_windows: int) -> np.ndarray:
    """
    Compute centers of sliding windows for connectivity features based on config.
    Uses plateau start/end and window length/step to cap window count.
    """
    feat_cfg = config.get("feature_engineering.features", {})
    plateau_default = config.get("time_frequency_analysis.plateau_window", [0.0, 0.0])
    
    plateau_window = feat_cfg.get("plateau_window", plateau_default)
    if isinstance(plateau_window, (list, tuple)) and len(plateau_window) >= 2:
        plateau_start = float(plateau_window[0])
        plateau_end = float(plateau_window[1])
    else:
        plateau_start = float(plateau_default[0])
        plateau_end = float(plateau_default[1])

    conn_cfg = config.get("feature_engineering.connectivity", {})
    win_len = float(conn_cfg.get("sliding_window_len", 1.0))
    win_step = float(conn_cfg.get("sliding_window_step", 0.5))

    if plateau_end <= plateau_start:
        return np.array([])

    max_windows = int(np.floor((plateau_end - plateau_start - win_len) / win_step) + 1)
    max_windows = max(0, max_windows)
    n_use = min(n_windows, max_windows)

    centers = plateau_start + np.arange(n_use) * win_step + (win_len / 2.0)
    return centers
