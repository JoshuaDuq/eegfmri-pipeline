"""
Core topomap utilities.

Generic topomap plotting helpers for label building and formatting.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import mne

from ..config import get_plot_config
from eeg_pipeline.plotting.io.figures import logratio_to_pct
from ...utils.analysis.stats import format_cluster_ann
from ...utils.config.loader import ensure_config, get_config_value


###################################################################
# Topomap Label Building
###################################################################


def _get_cluster_n_permutations(
    config: Optional[dict],
    tfr_config: Optional[dict],
) -> int:
    """Get number of cluster permutations from config.
    
    Args:
        config: Optional config dictionary
        tfr_config: Optional TFR-specific config dictionary
    
    Returns:
        Number of permutations (default: 100)
    """
    config = ensure_config(config)
    default_value = 100
    
    if tfr_config:
        tfr_default = tfr_config.get("default_cluster_n_perm", default_value)
        return get_config_value(config, "statistics.cluster_n_perm", tfr_default)
    
    return get_config_value(config, "statistics.cluster_n_perm", default_value)


def _build_cluster_annotation_text(
    cluster_p_min: float,
    cluster_k: Optional[int],
    cluster_mass: Optional[float],
    config: Optional[dict],
    paired: bool,
    n_permutations: int,
) -> str:
    """Build cluster annotation text with test type and permutation count.
    
    Args:
        cluster_p_min: Minimum cluster p-value
        cluster_k: Optional cluster size
        cluster_mass: Optional cluster mass
        config: Optional config dictionary
        paired: Whether paired test was used
        n_permutations: Number of permutations used
    
    Returns:
        Formatted cluster annotation string, or empty string if no annotation
    """
    cluster_text = format_cluster_ann(
        cluster_p_min, cluster_k, cluster_mass, config=config
    )
    
    if not cluster_text:
        return ""
    
    test_type = "paired cluster perm" if paired else "cluster perm"
    return f"{test_type} (n={n_permutations}): {cluster_text}"


def build_topomap_diff_label(
    diff_data: np.ndarray,
    cluster_p_min: Optional[float],
    cluster_k: Optional[int],
    cluster_mass: Optional[float],
    config: Optional[dict] = None,
    viz_params: Optional[dict] = None,
    paired: bool = False,
) -> str:
    """Build label text for difference topomap.
    
    Args:
        diff_data: Difference data array
        cluster_p_min: Optional minimum cluster p-value
        cluster_k: Optional cluster size
        cluster_mass: Optional cluster mass
        config: Optional config dictionary
        viz_params: Optional visualization parameters dictionary
        paired: Whether paired test was used
    
    Returns:
        Formatted label string with percentage change and cluster info
    """
    mean_difference = float(np.nanmean(diff_data))
    percentage_change = logratio_to_pct(mean_difference)
    percentage_label = f"Δ%={percentage_change:+.1f}%"
    
    if not viz_params or not viz_params.get("diff_annotation_enabled"):
        return percentage_label
    
    if cluster_p_min is None:
        return percentage_label
    
    tfr_config = None
    if config:
        plot_config = get_plot_config(config)
        tfr_config = plot_config.plot_type_configs.get("tfr", {})
    
    n_permutations = _get_cluster_n_permutations(config, tfr_config)
    cluster_text = _build_cluster_annotation_text(
        cluster_p_min,
        cluster_k,
        cluster_mass,
        config,
        paired,
        n_permutations,
    )
    
    if cluster_text:
        return f"{percentage_label} | {cluster_text}"
    
    return percentage_label


def build_topomap_percentage_label(data: np.ndarray) -> str:
    """Build percentage change label for topomap.
    
    Args:
        data: Data array to compute mean from
    
    Returns:
        Formatted percentage label string
    """
    mean_value = float(np.nanmean(data))
    percentage_change = logratio_to_pct(mean_value)
    return f"%Δ={percentage_change:+.1f}%"


###################################################################
# Topomap Data Preparation
###################################################################


def create_scalpmean_tfr_from_existing(
    tfr_avg: mne.time_frequency.AverageTFR,
    eeg_picks: list,
) -> mne.time_frequency.AverageTFR:
    """Create scalp-averaged TFR from existing TFR object.
    
    Args:
        tfr_avg: AverageTFR object
        eeg_picks: List of EEG channel picks
    
    Returns:
        New AverageTFR object with scalp-averaged data
    """
    tfr_scalpmean = tfr_avg.copy().pick(eeg_picks)
    scalpmean_data = np.asarray(tfr_scalpmean.data).mean(axis=0, keepdims=True)
    tfr_scalpmean = tfr_scalpmean.pick([0])
    tfr_scalpmean.data = scalpmean_data
    tfr_scalpmean.comment = "Scalp-averaged"
    return tfr_scalpmean

