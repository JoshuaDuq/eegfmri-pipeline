"""
Core statistics utilities.

Statistical masking, cluster significance computation, and T-test helpers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import mne
from scipy.stats import ttest_rel, ttest_ind, t as t_dist

from ..config import get_plot_config
from .utils import log
from ...utils.io.general import get_viz_params
from ...utils.analysis.tfr import extract_trial_band_power
from ...utils.analysis.stats import (
    cluster_test_epochs,
    cluster_test_two_sample_arrays,
)


###################################################################
# Configuration Helpers
###################################################################


def get_strict_mode(config) -> bool:
    """Get strict mode setting from config.
    
    Args:
        config: Optional config dictionary
    
    Returns:
        Strict mode boolean (default: True)
    """
    return config.get("analysis.strict_mode", True) if config else True


###################################################################
# Cluster Significance Computation
###################################################################


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
            tfr, mask1, mask2, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax,
            paired=False, config=config
        )
        if sig_mask is not None and diff_data_len is not None and len(sig_mask) != diff_data_len:
            if logger:
                log(
                    f"Cluster significance mask length mismatch: sig_mask length={len(sig_mask)}, "
                    f"diff_data_len={diff_data_len}. Discarding results.",
                    logger,
                    "warning"
                )
            return None, None, None, None
        return sig_mask, cluster_p_min, cluster_k, cluster_mass
    except (ValueError, RuntimeError) as e:
        if logger:
            log(f"Cluster test failed: {type(e).__name__}: {e}", logger, "warning")
        strict_mode = get_strict_mode(config)
        if strict_mode:
            raise
        return None, None, None, None


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
        return None, None, None, None
    
    min_subjects = 2
    if len(tfr1_list) < min_subjects or len(tfr2_list) < min_subjects:
        if logger:
            log("Cluster test requires at least 2 subjects per condition; skipping.", logger, "warning")
        return None, None, None, None
    
    subject_a_means = []
    subject_b_means = []
    tfr_a_list_valid = []
    tfr_b_list_valid = []
    reference_info = None
    
    for tfr_a, tfr_b in zip(tfr1_list, tfr2_list):
        if not isinstance(tfr_a, mne.time_frequency.EpochsTFR) or not isinstance(tfr_b, mne.time_frequency.EpochsTFR):
            continue
        
        data_a = extract_trial_band_power(tfr_a, fmin, fmax_eff, tmin, tmax)
        data_b = extract_trial_band_power(tfr_b, fmin, fmax_eff, tmin, tmax)
        
        if data_a is None or data_b is None:
            continue
        
        if data_a.shape[0] == 0 or data_b.shape[0] == 0:
            continue
        
        mean_a = np.nanmean(data_a, axis=0)
        mean_b = np.nanmean(data_b, axis=0)
        
        if mean_a.ndim == 0:
            mean_a = mean_a.reshape(1)
        if mean_b.ndim == 0:
            mean_b = mean_b.reshape(1)
        
        if reference_info is None:
            reference_info = tfr_a.info
        
        subject_a_means.append(mean_a)
        subject_b_means.append(mean_b)
        tfr_a_list_valid.append(tfr_a)
        tfr_b_list_valid.append(tfr_b)
    
    if len(subject_a_means) < min_subjects:
        if logger:
            log("Insufficient subjects with valid data for cluster test; skipping.", logger, "warning")
        return None, None, None, None
    
    ch_sets = [set(tfr.info["ch_names"]) for tfr in tfr_a_list_valid]
    ch_sets.extend([set(tfr.info["ch_names"]) for tfr in tfr_b_list_valid])
    common_chs = list(sorted(set.intersection(*ch_sets))) if ch_sets else []
    
    if len(common_chs) == 0:
        if logger:
            log("No common channels across subjects for cluster test; skipping.", logger, "warning")
        return None, None, None, None
    
    group_a_array = []
    group_b_array = []
    
    for mean_a, mean_b, tfr_a, tfr_b in zip(subject_a_means, subject_b_means, tfr_a_list_valid, tfr_b_list_valid):
        ch_indices_a = [tfr_a.info["ch_names"].index(ch) for ch in common_chs if ch in tfr_a.info["ch_names"]]
        ch_indices_b = [tfr_b.info["ch_names"].index(ch) for ch in common_chs if ch in tfr_b.info["ch_names"]]
        
        if len(ch_indices_a) != len(common_chs) or len(ch_indices_b) != len(common_chs):
            continue
        
        group_a_array.append(mean_a[ch_indices_a])
        group_b_array.append(mean_b[ch_indices_b])
    
    if len(group_a_array) < min_subjects or len(group_b_array) < min_subjects:
        if logger:
            log("Insufficient subjects after channel alignment for cluster test; skipping.", logger, "warning")
        return None, None, None, None
    
    group_a_subjects = np.stack(group_a_array, axis=0)
    group_b_subjects = np.stack(group_b_array, axis=0)
    
    info_common = mne.pick_info(reference_info, [reference_info["ch_names"].index(ch) for ch in common_chs])
    
    try:
        sig_mask_full, cluster_p_min, cluster_k, cluster_mass = cluster_test_two_sample_arrays(
            group_a_subjects, group_b_subjects, info_common,
            alpha=config.get("statistics.sig_alpha", 0.05) if config else 0.05,
            paired=False,
            n_permutations=config.get("statistics.cluster_n_perm", 1024) if config else 1024,
            config=config
        )
        
        if sig_mask_full is None:
            return None, None, None, None
        
        if diff_data_len is not None and len(sig_mask_full) != diff_data_len:
            if logger:
                log(
                    f"Cluster significance mask length mismatch: sig_mask length={len(sig_mask_full)}, "
                    f"diff_data_len={diff_data_len}. Discarding results.",
                    logger,
                    "warning"
                )
            return None, None, None, None
        
        return sig_mask_full, cluster_p_min, cluster_k, cluster_mass
    except (ValueError, RuntimeError) as e:
        if logger:
            log(f"Cluster test failed: {type(e).__name__}: {e}", logger, "warning")
        strict_mode = get_strict_mode(config)
        if strict_mode:
            raise
        return None, None, None, None


###################################################################
# Significance Mask Computation
###################################################################


def compute_significance_mask(
    tfr_sub: mne.time_frequency.EpochsTFR,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    fmin: float,
    fmax: float,
    tmin: float,
    tmax: float,
    config=None,
) -> Tuple[Optional[np.ndarray], Dict[str, Optional[float]]]:
    """Compute significance mask for TFR data.
    
    Args:
        tfr_sub: EpochsTFR object
        mask_a: Boolean mask for condition A
        mask_b: Boolean mask for condition B
        fmin: Minimum frequency
        fmax: Maximum frequency
        tmin: Minimum time
        tmax: Maximum time
        config: Optional config dictionary
    
    Returns:
        Tuple of (significance_mask, cluster_info_dict)
        Returns (None, {}) if diff_annotation_enabled is False
    """
    viz_params = get_viz_params(config)
    if not viz_params["diff_annotation_enabled"]:
        return None, {}
    
    sig_mask, cluster_p_min, cluster_k, cluster_mass = cluster_test_epochs(
        tfr_sub, mask_a, mask_b, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax,
        paired=False, config=config
    )
    return sig_mask, {
        "cluster_p_min": cluster_p_min,
        "cluster_k": cluster_k,
        "cluster_mass": cluster_mass
    }


###################################################################
# Statistical Title Building
###################################################################


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
    
    alpha = config.get("statistics.sig_alpha", 0.05) if config else 0.05
    n_perm = config.get("statistics.cluster_n_perm", 5000) if config else 5000
    
    parts = []
    
    if is_group:
        test_type = "paired cluster-based permutation test" if paired else "cluster-based permutation test (two-sample, unpaired)"
        parts.append(f"Statistical test: {test_type}")
        if n_subjects is not None:
            parts.append(f"N={n_subjects} subjects")
    else:
        test_type = "paired cluster-based permutation test" if paired else "cluster-based permutation test (two-sample, unpaired)"
        parts.append(f"Statistical test: {test_type}")
        if n_trials_pain is not None and n_trials_non is not None:
            parts.append(f"n_pain={n_trials_pain}, n_non={n_trials_non} trials")
    
    parts.append(f"n_permutations={n_perm}")
    parts.append(f"alpha={alpha:.3f}")
    
    if baseline_used is not None:
        bl_start, bl_end = baseline_used
        if bl_start is not None and bl_end is not None:
            parts.append(f"baseline correction: logratio [{bl_start:.2f}, {bl_end:.2f}]s")
        else:
            parts.append("baseline correction: logratio")
    else:
        parts.append("baseline correction: logratio")
    
    parts.append("data transformation: log10(power/baseline)")
    parts.append("adjacency: channel spatial adjacency matrix")
    parts.append("cluster threshold: mass-based")
    
    return " | ".join(parts)

