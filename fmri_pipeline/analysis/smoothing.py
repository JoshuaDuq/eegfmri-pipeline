from __future__ import annotations

import math
from typing import Optional


def normalize_smoothing_fwhm(value: Optional[float]) -> Optional[float]:
    """
    Normalize smoothing FWHM value (mm).

    - None or <= 0 disables smoothing (returns None)
    - Positive values return as float
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"smoothing_fwhm must be numeric or null, got {value!r}.") from exc
    if not math.isfinite(v):
        raise ValueError(f"smoothing_fwhm must be finite when provided, got {value!r}.")
    if v <= 0:
        return None
    return v
