from __future__ import annotations

from typing import Dict, Tuple

import mne
import numpy as np


def prepare_topomap_correlation_data(band_data: Dict, info: mne.Info) -> Tuple[np.ndarray, np.ndarray]:
    n_info_chs = len(info["ch_names"])
    topo_data = np.zeros(n_info_chs)
    topo_mask = np.zeros(n_info_chs, dtype=bool)

    for j, info_ch in enumerate(info["ch_names"]):
        if info_ch in band_data["channels"]:
            ch_idx = band_data["channels"].index(info_ch)
            if np.isfinite(band_data["correlations"][ch_idx]):
                topo_data[j] = band_data["correlations"][ch_idx]
            topo_mask[j] = bool(band_data["significant_mask"][ch_idx])

    return topo_data, topo_mask


__all__ = ["prepare_topomap_correlation_data"]
