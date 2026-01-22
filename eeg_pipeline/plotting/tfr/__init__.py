"""
Time-frequency representation (TFR) plotting module.

Low-level plotting primitives live here. High-level orchestration is defined in
`plotting.orchestration.tfr` to keep responsibilities separated.
"""

from __future__ import annotations

import importlib


def visualize_subject_tfr(*args, **kwargs):
    """Wrapper to avoid circular imports."""
    from eeg_pipeline.plotting.orchestration.tfr import visualize_subject_tfr as _impl

    return _impl(*args, **kwargs)


def visualize_tfr_for_subjects(*args, **kwargs):
    """Wrapper to avoid circular imports."""
    from eeg_pipeline.plotting.orchestration.tfr import visualize_tfr_for_subjects as _impl

    return _impl(*args, **kwargs)


__all__ = [
    # Channel-level TFR plotting
    "plot_channels_all_trials",
    "contrast_channels_pain_nonpain",
    # Scalp-mean TFR plotting
    "plot_scalpmean_all_trials",
    "contrast_scalpmean_pain_nonpain",
    # Contrast plotting
    "contrast_maxmin_temperature",
    "contrast_pain_nonpain",
    # ROI TFR plotting
    "compute_roi_tfrs",
    "plot_rois_all_trials",
    "contrast_pain_nonpain_rois",
    # Topomap plotting
    "plot_pain_nonpain_temporal_topomaps_diff_allbands",
    "plot_temporal_topomaps_allbands_active",
    # Quality control
    "qc_baseline_active_power",
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


def __getattr__(name: str):
    _module_map = {
        # Channel-level TFR plotting
        "plot_channels_all_trials": "channels",
        "contrast_channels_pain_nonpain": "channels",
        # Scalp-mean TFR plotting
        "plot_scalpmean_all_trials": "scalpmean",
        "contrast_scalpmean_pain_nonpain": "scalpmean",
        # Contrast plotting
        "contrast_maxmin_temperature": "contrasts",
        "contrast_pain_nonpain": "contrasts",
        # ROI TFR plotting
        "compute_roi_tfrs": "rois",
        "plot_rois_all_trials": "rois",
        "contrast_pain_nonpain_rois": "rois",
        # Topomap plotting
        "plot_pain_nonpain_temporal_topomaps_diff_allbands": "topomaps",
        "plot_temporal_topomaps_allbands_active": "topomaps",
        # Quality control
        "qc_baseline_active_power": "qc",
        # Band power evolution
        "visualize_band_evolution": "band_evolution",
        "plot_band_power_evolution_all_conditions": "band_evolution",
        "plot_band_power_by_roi": "band_evolution",
        "plot_condition_comparison_per_band": "band_evolution",
        "plot_roi_condition_comparison": "band_evolution",
        "plot_band_power_summary": "band_evolution",
    }

    module_name = _module_map.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(mod, name)
