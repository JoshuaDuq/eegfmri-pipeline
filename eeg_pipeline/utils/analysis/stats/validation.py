"""
Data Validation
===============

Validation for baseline window (pre-stimulus) used in TFR and plotting.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union


###################################################################
# Baseline Window Validation
###################################################################


def validate_baseline_window_pre_stimulus(
    baseline_window: Union[Tuple[float, float], List[float]],
    logger: Optional[logging.Logger] = None,
    *,
    strict: bool = False,
) -> Tuple[float, float]:
    """Check baseline window ends before stimulus onset.
    
    Parameters
    ----------
    baseline_window : tuple or list
        Baseline window as (tmin, baseline_end)
    logger : Logger, optional
        Logger for warnings
    strict : bool, optional
        If True, raise ValueError when baseline extends past stimulus onset (t=0).
        If False (default), only log a warning. For scientifically valid baseline
        normalization, strict=True is recommended.
        
    Returns
    -------
    tuple
        Validated tuple (tmin, baseline_end)
        
    Raises
    ------
    ValueError
        If strict=True and baseline extends past stimulus onset, or if baseline_window
        is not a tuple/list with at least 2 elements.
    """
    STIMULUS_ONSET = 0.0
    
    if not isinstance(baseline_window, (tuple, list)) or len(baseline_window) < 2:
        raise ValueError(
            f"baseline_window must be a tuple or list with at least 2 elements, "
            f"got {type(baseline_window)}"
        )
    
    tmin = float(baseline_window[0])
    baseline_end_value = float(baseline_window[1])
    
    if baseline_end_value > STIMULUS_ONSET:
        msg = (
            f"Baseline window extends past stimulus onset: baseline_end={baseline_end_value:.3f}s > 0. "
            "This contaminates baseline with stimulus-evoked activity, invalidating "
            "baseline normalization. Adjust baseline_window to end at or before t=0."
        )
        if strict:
            raise ValueError(msg)
        elif logger:
            logger.warning(msg)
    
    return (tmin, baseline_end_value)

