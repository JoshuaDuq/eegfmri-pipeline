"""
Time-frequency representation (TFR) plotting module.

Low-level plotting primitives live here. High-level orchestration/IO is defined in
`pipelines.viz.tfr` to keep responsibilities separated.
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


# Quality control
from .qc import (
    qc_baseline_plateau_power,
)

# Band power evolution
from .band_evolution import (
    visualize_band_evolution,
    plot_band_power_evolution_all_conditions,
    plot_band_power_by_roi,
    plot_condition_comparison_per_band,
    plot_roi_condition_comparison,
    plot_band_power_summary,
)

###################################################################
# Visualization orchestration wrappers (pipeline layer)
###################################################################
# These lightweight wrappers avoid an import-time dependency on
# `eeg_pipeline.pipelines.viz.tfr`, preventing circular imports while
# preserving the public API.


def visualize_subject_tfr(*args, **kwargs):
    from eeg_pipeline.pipelines.viz.tfr import visualize_subject_tfr as _impl

    return _impl(*args, **kwargs)


def visualize_tfr_for_subjects(*args, **kwargs):
    from eeg_pipeline.pipelines.viz.tfr import visualize_tfr_for_subjects as _impl

    return _impl(*args, **kwargs)


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
    # Quality control
    "qc_baseline_plateau_power",
    # Band power evolution
    "visualize_band_evolution",
    "plot_band_power_evolution_all_conditions",
    "plot_band_power_by_roi",
    "plot_condition_comparison_per_band",
    "plot_roi_condition_comparison",
    "plot_band_power_summary",
    # Visualization orchestration (pipeline)
    "visualize_subject_tfr",
    "visualize_tfr_for_subjects",
]
