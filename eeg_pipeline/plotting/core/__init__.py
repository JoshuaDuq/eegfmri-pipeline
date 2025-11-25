"""
Core plotting utilities module.

Provides generic utilities for font sizes, logging, annotations, colorbars, and other common
plotting operations that have no dependencies on sibling plotting modules.
"""

from .utils import get_font_sizes, log
from .annotations import (
    format_cluster_significance_info,
    format_ttest_significance_info,
    build_significance_info,
    extract_valid_pvalues,
    apply_fdr_correction_to_roi_pvalues,
    find_annotation_x_position,
    build_roi_annotation_label,
    render_roi_annotations,
    get_sig_marker_text,
)
from .colorbars import (
    add_normalized_colorbar,
    create_difference_colorbar,
    add_diff_colorbar,
    create_colorbar_for_topomaps,
    add_colorbar,
)
from .topomaps import (
    build_topomap_diff_label,
    build_topomap_percentage_label,
    create_scalpmean_tfr_from_existing,
)
from .statistics import (
    get_strict_mode,
    compute_cluster_significance,
    compute_cluster_significance_from_combined,
    compute_significance_mask,
    build_statistical_title,
)

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
    "get_sig_marker_text",
    "add_normalized_colorbar",
    "create_difference_colorbar",
    "add_diff_colorbar",
    "create_colorbar_for_topomaps",
    "add_colorbar",
    "build_topomap_diff_label",
    "build_topomap_percentage_label",
    "create_scalpmean_tfr_from_existing",
    "get_strict_mode",
    "compute_cluster_significance",
    "compute_cluster_significance_from_combined",
    "compute_significance_mask",
    "build_statistical_title",
]

