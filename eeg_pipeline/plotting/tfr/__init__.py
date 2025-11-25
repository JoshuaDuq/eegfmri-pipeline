"""
Time-frequency representation (TFR) plotting module.

This module provides functions for creating time-frequency representation visualizations
at multiple levels: channel-level, scalp-mean, ROI-level, topomaps, group analysis,
and correlation analysis.
"""

from __future__ import annotations

# Channel-level TFR plotting
from .channels import (
    plot_cz_all_trials_raw,
    plot_cz_all_trials,
    plot_channels_all_trials,
    contrast_channels_pain_nonpain,
)

# Scalp-mean TFR plotting
from .scalpmean import (
    plot_scalpmean_all_trials,
    contrast_scalpmean_pain_nonpain,
)

# Contrast plotting
from .contrasts import (
    contrast_maxmin_temperature,
    contrast_pain_nonpain,
    plot_bands_pain_temp_contrasts,
)

# ROI TFR plotting
from .rois import (
    compute_roi_tfrs,
    plot_rois_all_trials,
    contrast_pain_nonpain_rois,
)

# Topomap plotting
from .topomaps import (
    plot_topomap_grid_baseline_temps,
    plot_pain_nonpain_temporal_topomaps_diff_allbands,
    plot_temporal_topomaps_allbands_plateau,
)

# Group-level TFR plotting
from .group import (
    group_contrast_maxmin_temperature,
    group_rois_all_trials,
    group_contrast_pain_nonpain_rois,
    group_contrast_pain_nonpain_scalpmean,
    group_plot_bands_pain_temp_contrasts,
    group_plot_topomap_grid_baseline_temps,
    group_plot_pain_nonpain_temporal_topomaps_diff_allbands,
    group_plot_temporal_topomaps_allbands_plateau,
)

# Time-frequency correlation
from .correlation import (
    group_tf_correlation,
)

# Quality control
from .qc import (
    qc_baseline_plateau_power,
)

# Visualization orchestration
from .viz import (
    visualize_subject_tfr,
    visualize_group_tfr,
    visualize_tfr_for_subjects,
)


__all__ = [
    # Channel-level TFR plotting
    "plot_cz_all_trials_raw",
    "plot_cz_all_trials",
    "plot_channels_all_trials",
    "contrast_channels_pain_nonpain",
    # Scalp-mean TFR plotting
    "plot_scalpmean_all_trials",
    "contrast_scalpmean_pain_nonpain",
    # Contrast plotting
    "contrast_maxmin_temperature",
    "contrast_pain_nonpain",
    "plot_bands_pain_temp_contrasts",
    # ROI TFR plotting
    "compute_roi_tfrs",
    "plot_rois_all_trials",
    "contrast_pain_nonpain_rois",
    # Topomap plotting
    "plot_topomap_grid_baseline_temps",
    "plot_pain_nonpain_temporal_topomaps_diff_allbands",
    "plot_temporal_topomaps_allbands_plateau",
    # Group-level TFR plotting
    "group_contrast_maxmin_temperature",
    "group_rois_all_trials",
    "group_contrast_pain_nonpain_rois",
    "group_contrast_pain_nonpain_scalpmean",
    "group_plot_bands_pain_temp_contrasts",
    "group_plot_topomap_grid_baseline_temps",
    "group_plot_pain_nonpain_temporal_topomaps_diff_allbands",
    "group_plot_temporal_topomaps_allbands_plateau",
    # Time-frequency correlation
    "group_tf_correlation",
    # Quality control
    "qc_baseline_plateau_power",
    # Visualization orchestration
    "visualize_subject_tfr",
    "visualize_group_tfr",
    "visualize_tfr_for_subjects",
]

