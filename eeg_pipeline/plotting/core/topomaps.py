"""
Core topomap utilities.

Generic topomap plotting helpers for label building and formatting.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import mne

from ..config import get_plot_config
from ...utils.io.general import logratio_to_pct
from ...utils.analysis.stats import format_cluster_ann


###################################################################
# Topomap Label Building
###################################################################


def build_topomap_diff_label(
    diff_data: np.ndarray,
    cluster_p_min: Optional[float],
    cluster_k: Optional[int],
    cluster_mass: Optional[float],
    config=None,
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
    diff_mu = float(np.nanmean(diff_data))
    pct = logratio_to_pct(diff_mu)
    
    cl_txt = ""
    if viz_params and viz_params.get("diff_annotation_enabled") and cluster_p_min is not None:
        test_type = "paired cluster perm" if paired else "cluster perm"
        tfr_config = None
        if config:
            plot_cfg = get_plot_config(config)
            tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
        default_cluster_n_perm = tfr_config.get("default_cluster_n_perm", 100) if tfr_config else 100
        n_perm = config.get("statistics.cluster_n_perm", default_cluster_n_perm) if config else default_cluster_n_perm
        cl_txt = format_cluster_ann(cluster_p_min, cluster_k, cluster_mass, config=config)
        if cl_txt:
            cl_txt = f"{test_type} (n={n_perm}): {cl_txt}"
    
    return f"Δ%={pct:+.1f}%" + (f" | {cl_txt}" if cl_txt else "")


def build_topomap_percentage_label(data: np.ndarray) -> str:
    """Build percentage change label for topomap.
    
    Args:
        data: Data array to compute mean from
    
    Returns:
        Formatted percentage label string
    """
    mu = float(np.nanmean(data))
    pct = logratio_to_pct(mu)
    return f"%Δ={pct:+.1f}%"


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
    tfr_sm = tfr_avg.copy().pick(eeg_picks)
    data_sm = np.asarray(tfr_sm.data).mean(axis=0, keepdims=True)
    tfr_sm = tfr_sm.pick([0])
    tfr_sm.data = data_sm
    tfr_sm.comment = "Scalp-averaged"
    return tfr_sm

