"""
Core statistics utilities.

Statistical masking, cluster significance computation, and T-test helpers.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import mne

from eeg_pipeline.utils.config.loader import ensure_config, get_config_value
from .utils import log
from eeg_pipeline.plotting.io.figures import get_viz_params
from ...utils.analysis.tfr import extract_trial_band_power
from ...utils.analysis.stats import (
    cluster_test_epochs,
    cluster_test_two_sample_arrays,
)


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


def _extract_subject_means(
    tfr_list: List[mne.time_frequency.EpochsTFR],
    fmin: float,
    fmax: float,
    tmin: float,
    tmax: float,
) -> Tuple[List[np.ndarray], List[mne.time_frequency.EpochsTFR], Optional[mne.Info]]:
    """Extract mean band power for each subject in TFR list.
    
    Args:
        tfr_list: List of EpochsTFR objects
        fmin: Minimum frequency
        fmax: Maximum frequency
        tmin: Minimum time
        tmax: Maximum time
    
    Returns:
        Tuple of (mean_arrays, valid_tfr_list, reference_info)
    """
    mean_arrays = []
    valid_tfr_list = []
    reference_info = None
    
    for tfr in tfr_list:
        if not isinstance(tfr, mne.time_frequency.EpochsTFR):
            continue
        
        data = extract_trial_band_power(tfr, fmin, fmax, tmin, tmax)
        if data is None or data.shape[0] == 0:
            continue
        
        mean = np.nanmean(data, axis=0)
        if mean.ndim == 0:
            mean = mean.reshape(1)
        
        if reference_info is None:
            reference_info = tfr.info
        
        mean_arrays.append(mean)
        valid_tfr_list.append(tfr)
    
    return mean_arrays, valid_tfr_list, reference_info


def _find_common_channels(
    tfr_list: List[mne.time_frequency.EpochsTFR],
) -> List[str]:
    """Find common channels across all TFR objects.
    
    Args:
        tfr_list: List of EpochsTFR objects
    
    Returns:
        Sorted list of common channel names
    """
    if not tfr_list:
        return []
    
    channel_sets = [set(tfr.info["ch_names"]) for tfr in tfr_list]
    common_channels = set.intersection(*channel_sets) if channel_sets else set()
    return sorted(common_channels)


def _align_channels_to_common(
    mean_arrays: List[np.ndarray],
    tfr_list: List[mne.time_frequency.EpochsTFR],
    common_channels: List[str],
) -> List[np.ndarray]:
    """Align subject means to common channel set.
    
    Args:
        mean_arrays: List of mean arrays per subject
        tfr_list: List of corresponding TFR objects
        common_channels: List of common channel names
    
    Returns:
        List of aligned arrays
    """
    aligned_arrays = []
    
    for mean_array, tfr in zip(mean_arrays, tfr_list):
        try:
            channel_indices = [
                tfr.info["ch_names"].index(ch) for ch in common_channels
            ]
        except ValueError:
            continue
        
        aligned_arrays.append(mean_array[channel_indices])
    
    return aligned_arrays


def compute_cluster_significance_from_combined(
    tfr1_list: List[mne.time_frequency.EpochsTFR],
    tfr2_list: List[mne.time_frequency.EpochsTFR],
    fmin: float,
    fmax_eff: float,
    tmin: float,
    tmax: float,
    config=None,
    diff_data_len: Optional[int] = None,
    logger=None,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[int], Optional[float]]:
    """Compute cluster significance from combined subject data.
    
    Args:
        tfr1_list: List of EpochsTFR objects for first condition
        tfr2_list: List of EpochsTFR objects for second condition
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
    if not tfr1_list or not tfr2_list:
        return EMPTY_CLUSTER_RESULT
    
    if len(tfr1_list) < MIN_SUBJECTS_REQUIRED or len(tfr2_list) < MIN_SUBJECTS_REQUIRED:
        if logger:
            log(
                f"Cluster test requires at least {MIN_SUBJECTS_REQUIRED} subjects "
                f"per condition; skipping.",
                logger,
                "warning"
            )
        return EMPTY_CLUSTER_RESULT
    
    subject_a_means, tfr_a_valid, reference_info = _extract_subject_means(
        tfr1_list, fmin, fmax_eff, tmin, tmax
    )
    subject_b_means, tfr_b_valid, _ = _extract_subject_means(
        tfr2_list, fmin, fmax_eff, tmin, tmax
    )
    
    if len(subject_a_means) < MIN_SUBJECTS_REQUIRED or len(subject_b_means) < MIN_SUBJECTS_REQUIRED:
        if logger:
            log(
                "Insufficient subjects with valid data for cluster test; skipping.",
                logger,
                "warning"
            )
        return EMPTY_CLUSTER_RESULT
    
    all_valid_tfr = tfr_a_valid + tfr_b_valid
    common_channels = _find_common_channels(all_valid_tfr)
    
    if not common_channels:
        if logger:
            log(
                "No common channels across subjects for cluster test; skipping.",
                logger,
                "warning"
            )
        return EMPTY_CLUSTER_RESULT
    
    group_a_aligned = _align_channels_to_common(
        subject_a_means, tfr_a_valid, common_channels
    )
    group_b_aligned = _align_channels_to_common(
        subject_b_means, tfr_b_valid, common_channels
    )
    
    if len(group_a_aligned) < MIN_SUBJECTS_REQUIRED or len(group_b_aligned) < MIN_SUBJECTS_REQUIRED:
        if logger:
            log(
                "Insufficient subjects after channel alignment for cluster test; skipping.",
                logger,
                "warning"
            )
        return EMPTY_CLUSTER_RESULT
    
    group_a_subjects = np.stack(group_a_aligned, axis=0)
    group_b_subjects = np.stack(group_b_aligned, axis=0)
    
    channel_indices = [
        reference_info["ch_names"].index(ch) for ch in common_channels
    ]
    info_common = mne.pick_info(reference_info, channel_indices)
    
    try:
        config = ensure_config(config)
        alpha = get_config_value(config, "statistics.sig_alpha", DEFAULT_ALPHA)
        n_permutations = get_config_value(
            config, "statistics.cluster_n_perm", DEFAULT_N_PERMUTATIONS
        )
        
        sig_mask_full, cluster_p_min, cluster_k, cluster_mass = (
            cluster_test_two_sample_arrays(
                group_a_subjects,
                group_b_subjects,
                info_common,
                alpha=alpha,
                paired=False,
                n_permutations=n_permutations,
                config=config
            )
        )
        
        if sig_mask_full is None:
            return EMPTY_CLUSTER_RESULT
        
        if not _validate_mask_length(sig_mask_full, diff_data_len, logger):
            return EMPTY_CLUSTER_RESULT
        
        return sig_mask_full, cluster_p_min, cluster_k, cluster_mass
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
    n_trials_pain: Optional[int] = None,
    n_trials_non: Optional[int] = None,
    n_subjects: Optional[int] = None,
    is_group: bool = False,
) -> str:
    """Build statistical title string with test parameters.
    
    Args:
        config: Config dictionary
        baseline_used: Optional tuple of (baseline_start, baseline_end)
        paired: Whether paired test was used
        n_trials_pain: Optional number of pain trials
        n_trials_non: Optional number of non-pain trials
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
        if n_trials_pain is not None and n_trials_non is not None:
            parts.append(
                f"n_pain={n_trials_pain}, n_non={n_trials_non} trials"
            )
    
    parts.append(f"n_permutations={n_perm}")
    parts.append(f"alpha={alpha:.3f}")
    parts.append(_format_baseline_correction(baseline_used))
    parts.append("data transformation: log10(power/baseline)")
    parts.append("adjacency: channel spatial adjacency matrix")
    parts.append("cluster threshold: mass-based")
    
    return " | ".join(parts)
