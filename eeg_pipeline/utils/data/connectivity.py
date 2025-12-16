from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def flatten_lower_triangles(
    connectivity_trials: np.ndarray,
    labels: Optional[np.ndarray],
    prefix: str,
) -> Tuple[pd.DataFrame, List[str]]:
    if connectivity_trials.ndim != 3:
        raise ValueError("Connectivity array must be 3D (trials, nodes, nodes)")

    n_trials, n_nodes, _ = connectivity_trials.shape
    lower_tri_i, lower_tri_j = np.tril_indices(n_nodes, k=-1)
    flattened_data = connectivity_trials[:, lower_tri_i, lower_tri_j]

    if labels is not None and len(labels) == n_nodes:
        pair_names = [f"{labels[i]}__{labels[j]}" for i, j in zip(lower_tri_i, lower_tri_j)]
    else:
        pair_names = [f"n{i}_n{j}" for i, j in zip(lower_tri_i, lower_tri_j)]

    column_names = [f"{prefix}_{pair}" for pair in pair_names]
    return pd.DataFrame(flattened_data), column_names


__all__ = ["flatten_lower_triangles"]
