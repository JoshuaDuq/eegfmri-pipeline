from __future__ import annotations

import numpy as np
from typing import Any


def sliding_window_centers(config: Any, n_windows: int) -> np.ndarray:
    """
    Compute centers of sliding windows for connectivity features based on config.
    Uses plateau start/end and window length/step to cap window count.
    """
    feat_cfg = config.get("feature_engineering", {}).get("features", {})
    tf_cfg = config.get("time_frequency_analysis", {})
    plateau_default = tf_cfg.get("plateau_window", [0.0, 0.0])

    plateau_start = feat_cfg.get("plateau_start", plateau_default[0])
    plateau_end = feat_cfg.get("plateau_end", plateau_default[1])
    plateau_start = float(plateau_start)
    plateau_end = float(plateau_end)

    conn_cfg = config.get("feature_engineering", {}).get("connectivity", {})
    win_len = float(conn_cfg.get("sliding_window_len", 1.0))
    win_step = float(conn_cfg.get("sliding_window_step", 0.5))

    if plateau_end <= plateau_start:
        return np.array([])

    max_windows = int(np.floor((plateau_end - plateau_start - win_len) / win_step) + 1)
    max_windows = max(0, max_windows)
    n_use = min(n_windows, max_windows)

    centers = plateau_start + np.arange(n_use) * win_step + (win_len / 2.0)
    return centers
