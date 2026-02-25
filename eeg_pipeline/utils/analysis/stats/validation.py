"""
Data Validation
===============

Validation utilities for analysis assumptions: baseline window integrity
and predictor variable type compatibility.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple, Union

import pandas as pd

from .base import get_config_value as _get_config_value


###################################################################
# Predictor Type Validation
###################################################################

_VALID_PREDICTOR_TYPES = ("continuous", "binary", "categorical")
_MIN_UNIQUE_FOR_CURVE_FITTING = 5


def assert_predictor_type_continuous(config: Any, *, context: str) -> None:
    """Raise ValueError if behavior_analysis.predictor_type is not 'continuous'.

    Use this at analysis entry points where predictor data is not yet available
    (e.g., before loading events). For a full check including data uniqueness,
    use :func:`assert_continuous_predictor`.

    Parameters
    ----------
    config : Any
        Pipeline configuration object.
    context : str
        Analysis name for error messages (e.g., "psychometrics").

    Raises
    ------
    ValueError
        If predictor_type is not a recognised value or is not 'continuous'.
    """
    predictor_type = str(
        _get_config_value(config, "behavior_analysis.predictor_type", "continuous") or "continuous"
    ).strip().lower()

    if predictor_type not in _VALID_PREDICTOR_TYPES:
        raise ValueError(
            f"behavior_analysis.predictor_type must be one of {_VALID_PREDICTOR_TYPES}; "
            f"got '{predictor_type}'."
        )

    if predictor_type != "continuous":
        raise ValueError(
            f"{context} requires behavior_analysis.predictor_type='continuous' "
            f"(configured as '{predictor_type}'). "
            f"Disable this analysis via its enabled=false config key for "
            f"binary or categorical predictors."
        )


def assert_continuous_predictor(
    predictor: pd.Series,
    config: Any,
    *,
    context: str,
) -> None:
    """Raise ValueError if the predictor is not suitable for curve fitting.

    Performs both the config-level type check and a data-level uniqueness check.
    Curve-fitting analyses (predictor_residual, predictor_models) require a
    continuous predictor with sufficient unique values to identify a dose-response
    function. This guard prevents silently meaningless results for binary or
    categorical predictors.

    Parameters
    ----------
    predictor : pd.Series
        The predictor series as used in the analysis.
    config : Any
        Pipeline configuration object.
    context : str
        Analysis name for error messages (e.g., "predictor_residual").

    Raises
    ------
    ValueError
        If predictor_type is not 'continuous', or if the predictor has fewer
        than ``_MIN_UNIQUE_FOR_CURVE_FITTING`` unique values.
    """
    assert_predictor_type_continuous(config, context=context)

    n_unique = int(predictor.dropna().nunique())
    if n_unique < _MIN_UNIQUE_FOR_CURVE_FITTING:
        raise ValueError(
            f"{context} requires ≥{_MIN_UNIQUE_FOR_CURVE_FITTING} unique predictor values "
            f"for stable curve fitting; found {n_unique}. "
            f"Use a continuous predictor or set the analysis enabled=false."
        )


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

