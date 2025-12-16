"""
Base Statistics Utilities
=========================

Constants, config helpers, and core data structures for statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from eeg_pipeline.utils.config.loader import (
    ensure_config,
    get_config_value,
    get_constants,
    load_settings,
)


@dataclass
class CorrelationStats:
    """Container for correlation statistics."""

    correlation: float = np.nan
    p_value: float = np.nan
    ci_low: float = np.nan
    ci_high: float = np.nan
    r_partial: float = np.nan
    p_partial: float = np.nan
    n_partial: int = 0
    r_partial_temp: float = np.nan
    p_partial_temp: float = np.nan
    n_partial_temp: int = 0
    p_perm: float = np.nan
    p_partial_perm: float = np.nan
    p_partial_temp_perm: float = np.nan


def get_statistics_constants(config=None):
    """Load statistics constants from config."""
    if get_constants is not None:
        constants = get_constants("statistics", config)
        constants["cluster_structure_2d"] = np.array(
            constants["cluster_structure_2d"], dtype=int
        )
        return constants

    config = ensure_config(config)
    if config is None:
        raise ValueError("Config required for statistics constants.")

    constants = config.get("statistics.constants")
    if constants is None:
        raise ValueError("statistics.constants not found in config.")

    result = dict(constants)
    result["cluster_structure_2d"] = np.array(constants["cluster_structure_2d"], dtype=int)
    return result


def get_fdr_alpha(config: Optional[Any] = None) -> float:
    """Get FDR alpha from config."""
    return float(get_config_value(config, "statistics.fdr_alpha", get_config_value(config, "statistics.sig_alpha", 0.05)))


def get_ci_level(config: Optional[Any] = None) -> float:
    """Get confidence interval level from config."""
    return float(get_config_value(config, "statistics.ci_level", 0.95))


def get_n_permutations(config: Optional[Any] = None) -> int:
    """Get number of permutations from config."""
    return int(get_config_value(config, "statistics.n_permutations", 1000))


def get_n_bootstrap(config: Optional[Any] = None) -> int:
    """Get number of bootstrap samples from config."""
    return int(get_config_value(config, "statistics.n_bootstrap", 1000))

