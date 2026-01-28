"""
Core plotting utilities module.

Provides generic utilities for font sizes, logging, annotations, colorbars, and other common
plotting operations that have no dependencies on sibling plotting modules.
"""

from __future__ import annotations

import importlib

__all__ = [
    "get_font_sizes",
    "log",
    "format_cluster_significance_info",
    "format_ttest_significance_info",
    "build_significance_info",
    "extract_valid_pvalues",
    "apply_fdr_correction_to_roi_pvalues",
    "find_annotation_x_position",
    "build_roi_annotation_label",
    "render_roi_annotations",
    "add_roi_annotations",
    "get_sig_marker_text",
    "add_normalized_colorbar",
    "create_difference_colorbar",
    "add_diff_colorbar",
    "build_topomap_diff_label",
    "build_topomap_percentage_label",
    "create_scalpmean_tfr_from_existing",
    "get_strict_mode",
    "compute_cluster_significance",
    "build_statistical_title",
]

_MODULE_MAP = {
    "get_font_sizes": "utils",
    "log": "utils",
    "format_cluster_significance_info": "annotations",
    "format_ttest_significance_info": "annotations",
    "build_significance_info": "annotations",
    "extract_valid_pvalues": "annotations",
    "apply_fdr_correction_to_roi_pvalues": "annotations",
    "find_annotation_x_position": "annotations",
    "build_roi_annotation_label": "annotations",
    "render_roi_annotations": "annotations",
    "add_roi_annotations": "annotations",
    "get_sig_marker_text": "annotations",
    "add_normalized_colorbar": "colorbars",
    "create_difference_colorbar": "colorbars",
    "add_diff_colorbar": "colorbars",
    "build_topomap_diff_label": "topomaps",
    "build_topomap_percentage_label": "topomaps",
    "create_scalpmean_tfr_from_existing": "topomaps",
    "get_strict_mode": "statistics",
    "compute_cluster_significance": "statistics",
    "build_statistical_title": "statistics",
}


def __getattr__(name: str):
    module_name = _MODULE_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(mod, name)

