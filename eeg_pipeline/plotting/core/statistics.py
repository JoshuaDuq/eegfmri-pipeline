"""
Core statistics utilities.

Statistical masking, cluster significance computation, and T-test helpers.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import mne

from eeg_pipeline.utils.config.loader import ensure_config, get_config_value
from .utils import log
from eeg_pipeline.plotting.io.figures import get_viz_params
from ...utils.analysis.stats import cluster_test_epochs


MIN_SUBJECTS_REQUIRED = 2
EMPTY_CLUSTER_RESULT = (None, None, None, None)
DEFAULT_ALPHA = 0.05
DEFAULT_N_PERMUTATIONS = 100


def get_strict_mode(config) -> bool:
    """Get strict mode setting from config.
    
    Args:
        config: Optional config dictionary
    
    Returns:
        Strict mode boolean (default: True)
    """
    config = ensure_config(config)
    return get_config_value(config, "analysis.strict_mode", True)


def _validate_mask_length(
    sig_mask: Optional[np.ndarray],
    expected_len: Optional[int],
    logger=None,
) -> bool:
    """Validate significance mask length matches expected length.
    
    Args:
        sig_mask: Significance mask array
        expected_len: Expected length for validation
        logger: Optional logger instance
    
    Returns:
        True if valid or no validation needed, False if mismatch
    """
    if sig_mask is None or expected_len is None:
        return True
    
    if len(sig_mask) != expected_len:
        if logger:
            log(
                f"Cluster significance mask length mismatch: "
                f"sig_mask length={len(sig_mask)}, diff_data_len={expected_len}. "
                f"Discarding results.",
                logger,
                "warning"
            )
        return False
    return True


def _handle_cluster_test_error(
    error: Exception,
    config,
    logger=None,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[int], Optional[float]]:
    """Handle cluster test errors with appropriate logging and strict mode.
    
    Args:
        error: Exception that occurred
        config: Config dictionary
        logger: Optional logger instance
    
    Returns:
        Empty cluster result tuple
    """
    if logger:
        log(
            f"Cluster test failed: {type(error).__name__}: {error}",
            logger,
            "warning"
        )
    strict_mode = get_strict_mode(config)
    if strict_mode:
        raise error
    return EMPTY_CLUSTER_RESULT


def compute_cluster_significance(
    tfr: mne.time_frequency.EpochsTFR,
    mask1: np.ndarray,
    mask2: np.ndarray,
    fmin: float,
    fmax_eff: float,
    tmin: float,
    tmax: float,
    config=None,
    diff_data_len: Optional[int] = None,
    logger=None,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[int], Optional[float]]:
    """Compute cluster significance for TFR data.
    
    Args:
        tfr: EpochsTFR object
        mask1: Boolean mask for first condition
        mask2: Boolean mask for second condition
        fmin: Minimum frequency
        fmax_eff: Effective maximum frequency
        tmin: Minimum time
        tmax: Maximum time
        config: Optional config dictionary
        diff_data_len: Optional expected length of difference data for validation
        logger: Optional logger instance
    
    Returns:
        Tuple of (significance_mask, cluster_p_min, cluster_k, cluster_mass)
        Returns (None, None, None, None) on failure
    """
    try:
        sig_mask, cluster_p_min, cluster_k, cluster_mass = cluster_test_epochs(
            tfr,
            mask1,
            mask2,
            fmin=fmin,
            fmax=fmax_eff,
            tmin=tmin,
            tmax=tmax,
            paired=False,
            config=config
        )
        
        if not _validate_mask_length(sig_mask, diff_data_len, logger):
            return EMPTY_CLUSTER_RESULT
        
        return sig_mask, cluster_p_min, cluster_k, cluster_mass
    except (ValueError, RuntimeError) as e:
        return _handle_cluster_test_error(e, config, logger)


def _get_test_type_description(paired: bool) -> str:
    """Get description string for test type.
    
    Args:
        paired: Whether test is paired
    
    Returns:
        Test type description string
    """
    if paired:
        return "paired cluster-based permutation test"
    return "cluster-based permutation test (two-sample, unpaired)"


def _format_baseline_correction(
    baseline_used: Optional[Tuple[Optional[float], Optional[float]]],
) -> str:
    """Format baseline correction string for title.
    
    Args:
        baseline_used: Optional tuple of (baseline_start, baseline_end)
    
    Returns:
        Formatted baseline correction string
    """
    base_str = "baseline correction: logratio"
    if baseline_used is None:
        return base_str
    
    bl_start, bl_end = baseline_used
    if bl_start is not None and bl_end is not None:
        return f"{base_str} [{bl_start:.2f}, {bl_end:.2f}]s"
    return base_str


def build_statistical_title(
    config,
    baseline_used: Optional[Tuple[Optional[float], Optional[float]]],
    paired: bool = False,
    n_trials_condition_2: Optional[int] = None,
    n_trials_condition_1: Optional[int] = None,
    n_subjects: Optional[int] = None,
    is_group: bool = False,
) -> str:
    """Build statistical title string with test parameters.
    
    Args:
        config: Config dictionary
        baseline_used: Optional tuple of (baseline_start, baseline_end)
        paired: Whether paired test was used
        n_trials_condition_2: Optional number of condition 2 trials
        n_trials_condition_1: Optional number of condition 1 trials
        n_subjects: Optional number of subjects
        is_group: Whether this is a group-level analysis
    
    Returns:
        Formatted statistical title string, or empty string if diff_annotation_enabled is False
    """
    viz_params = get_viz_params(config)
    if not viz_params["diff_annotation_enabled"]:
        return ""
    
    config = ensure_config(config)
    alpha = get_config_value(config, "statistics.sig_alpha", DEFAULT_ALPHA)
    n_perm = get_config_value(
        config, "statistics.cluster_n_perm", DEFAULT_N_PERMUTATIONS
    )
    
    parts = []
    test_type = _get_test_type_description(paired)
    parts.append(f"Statistical test: {test_type}")
    
    if is_group:
        if n_subjects is not None:
            parts.append(f"N={n_subjects} subjects")
    else:
        if n_trials_condition_2 is not None and n_trials_condition_1 is not None:
            parts.append(
                f"n_condition_2={n_trials_condition_2}, n_condition_1={n_trials_condition_1} trials"
            )
    
    parts.append(f"n_permutations={n_perm}")
    parts.append(f"alpha={alpha:.3f}")
    parts.append(_format_baseline_correction(baseline_used))
    parts.append("data transformation: log10(power/baseline)")
    parts.append("adjacency: channel spatial adjacency matrix")
    parts.append("cluster threshold: mass-based")
    
    return " | ".join(parts)
