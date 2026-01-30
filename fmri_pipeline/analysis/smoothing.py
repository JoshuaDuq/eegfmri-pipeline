from __future__ import annotations

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
    except Exception:
        return None
    if v <= 0:
        return None
    return v

