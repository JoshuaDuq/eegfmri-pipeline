from __future__ import annotations

import numpy as np
from typing import Any


def sliding_window_centers(config: Any, n_windows: int) -> np.ndarray:
    """
    Compute centers of sliding windows for connectivity features based on config.
    Uses plateau start/end and window length/step to cap window count.
    """
    feat_cfg = config.get("feature_engineering", {}).get("features", {})
    plateau_start = float(feat_cfg.get("plateau_start", 0.0))
    plateau_end = float(feat_cfg.get("plateau_end", 0.0))
    conn_cfg = config.get("feature_engineering", {}).get("connectivity", {})
    win_len = float(conn_cfg.get("sliding_window_len", 1.0))
    win_step = float(conn_cfg.get("sliding_window_step", 0.5))
    centers = plateau_start + np.arange(n_windows) * win_step + (win_len / 2.0)
    if plateau_end > plateau_start:
        max_windows = int(np.floor((plateau_end - plateau_start - win_len) / win_step) + 1)
        centers = centers[:max_windows]
    return centers
