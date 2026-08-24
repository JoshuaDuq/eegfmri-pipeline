"""Strict validation shared by report figure computations."""

from __future__ import annotations

from typing import Any

import numpy as np


def validated_binary_mask(mask_image: Any) -> np.ndarray:
    """Return a non-empty 3D mask after validating every stored value."""
    values = np.asanyarray(mask_image.dataobj)
    if values.ndim != 3:
        raise ValueError(f"The fitted analysis mask must be 3D, got {values.shape}.")
    if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
        raise ValueError("The fitted analysis mask must contain finite binary values.")
    mask = values.astype(bool)
    if not mask.any():
        raise ValueError("The fitted analysis mask contains no voxels.")
    return mask


__all__ = ["validated_binary_mask"]
