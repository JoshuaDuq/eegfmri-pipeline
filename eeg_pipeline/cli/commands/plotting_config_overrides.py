"""Config override helpers for plotting CLI command."""

from __future__ import annotations

import argparse
from typing import Any

from eeg_pipeline.utils.parsing import (
    parse_frequency_band_definitions,
    parse_roi_definitions,
)

def _apply_config_override(config: Any, path: str, value: Any) -> None:
    """Set a config value at the given dot-separated path."""
    config[path] = value


def _get_arg_value(args: argparse.Namespace, attr_name: str) -> Any:
    """Get argument value if present, otherwise None."""
    return getattr(args, attr_name, None)


def _apply_defaults_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply plotting defaults overrides."""
    if args.formats:
        _apply_config_override(config, "plotting.defaults.formats", list(args.formats))
    if args.dpi is not None:
        _apply_config_override(config, "plotting.defaults.dpi", int(args.dpi))
    if args.savefig_dpi is not None:
        _apply_config_override(config, "plotting.defaults.savefig_dpi", int(args.savefig_dpi))
    if _get_arg_value(args, "bbox_inches"):
        _apply_config_override(config, "plotting.defaults.bbox_inches", str(args.bbox_inches))
    if _get_arg_value(args, "pad_inches") is not None:
        _apply_config_override(config, "plotting.defaults.pad_inches", float(args.pad_inches))


def _apply_font_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply font-related config overrides."""
    if _get_arg_value(args, "font_family"):
        _apply_config_override(config, "plotting.defaults.font.family", str(args.font_family))
    if _get_arg_value(args, "font_weight"):
        _apply_config_override(config, "plotting.defaults.font.weight", str(args.font_weight))
    if _get_arg_value(args, "font_size_small") is not None:
        _apply_config_override(config, "plotting.defaults.font.sizes.small", int(args.font_size_small))
    if _get_arg_value(args, "font_size_medium") is not None:
        _apply_config_override(config, "plotting.defaults.font.sizes.medium", int(args.font_size_medium))
    if _get_arg_value(args, "font_size_large") is not None:
        _apply_config_override(config, "plotting.defaults.font.sizes.large", int(args.font_size_large))
    if _get_arg_value(args, "font_size_title") is not None:
        _apply_config_override(config, "plotting.defaults.font.sizes.title", int(args.font_size_title))
    if _get_arg_value(args, "font_size_annotation") is not None:
        _apply_config_override(config, "plotting.defaults.font.sizes.annotation", int(args.font_size_annotation))
    if _get_arg_value(args, "font_size_label") is not None:
        _apply_config_override(config, "plotting.defaults.font.sizes.label", int(args.font_size_label))
    if _get_arg_value(args, "font_size_ylabel") is not None:
        _apply_config_override(config, "plotting.defaults.font.sizes.ylabel", int(args.font_size_ylabel))
    if _get_arg_value(args, "font_size_suptitle") is not None:
        _apply_config_override(config, "plotting.defaults.font.sizes.suptitle", int(args.font_size_suptitle))
    if _get_arg_value(args, "font_size_figure_title") is not None:
        _apply_config_override(config, "plotting.defaults.font.sizes.figure_title", int(args.font_size_figure_title))


def _apply_layout_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply layout-related config overrides."""
    if _get_arg_value(args, "layout_tight_rect"):
        _apply_config_override(config, "plotting.defaults.layout.tight_rect", list(args.layout_tight_rect))
    if _get_arg_value(args, "layout_tight_rect_microstate"):
        _apply_config_override(config, "plotting.defaults.layout.tight_rect_microstate", list(args.layout_tight_rect_microstate))
    if _get_arg_value(args, "gridspec_width_ratios"):
        _apply_config_override(config, "plotting.defaults.layout.gridspec.width_ratios", list(args.gridspec_width_ratios))
    if _get_arg_value(args, "gridspec_height_ratios"):
        _apply_config_override(config, "plotting.defaults.layout.gridspec.height_ratios", list(args.gridspec_height_ratios))
    if _get_arg_value(args, "gridspec_hspace") is not None:
        _apply_config_override(config, "plotting.defaults.layout.gridspec.hspace", float(args.gridspec_hspace))
    if _get_arg_value(args, "gridspec_wspace") is not None:
        _apply_config_override(config, "plotting.defaults.layout.gridspec.wspace", float(args.gridspec_wspace))
    if _get_arg_value(args, "gridspec_left") is not None:
        _apply_config_override(config, "plotting.defaults.layout.gridspec.left", float(args.gridspec_left))
    if _get_arg_value(args, "gridspec_right") is not None:
        _apply_config_override(config, "plotting.defaults.layout.gridspec.right", float(args.gridspec_right))
    if _get_arg_value(args, "gridspec_top") is not None:
        _apply_config_override(config, "plotting.defaults.layout.gridspec.top", float(args.gridspec_top))
    if _get_arg_value(args, "gridspec_bottom") is not None:
        _apply_config_override(config, "plotting.defaults.layout.gridspec.bottom", float(args.gridspec_bottom))


def _apply_figure_size_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply figure size config overrides."""
    if _get_arg_value(args, "figure_size_standard"):
        _apply_config_override(config, "plotting.figure_sizes.standard", list(args.figure_size_standard))
    if _get_arg_value(args, "figure_size_medium"):
        _apply_config_override(config, "plotting.figure_sizes.medium", list(args.figure_size_medium))
    if _get_arg_value(args, "figure_size_small"):
        _apply_config_override(config, "plotting.figure_sizes.small", list(args.figure_size_small))
    if _get_arg_value(args, "figure_size_square"):
        _apply_config_override(config, "plotting.figure_sizes.square", list(args.figure_size_square))
    if _get_arg_value(args, "figure_size_wide"):
        _apply_config_override(config, "plotting.figure_sizes.wide", list(args.figure_size_wide))
    if _get_arg_value(args, "figure_size_tfr"):
        _apply_config_override(config, "plotting.figure_sizes.tfr", list(args.figure_size_tfr))
    if _get_arg_value(args, "figure_size_topomap"):
        _apply_config_override(config, "plotting.figure_sizes.topomap", list(args.figure_size_topomap))


def _apply_color_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply color-related config overrides."""
    if _get_arg_value(args, "color_condition_2"):
        _apply_config_override(config, "plotting.styling.colors.condition_2", str(args.color_condition_2))
    if _get_arg_value(args, "color_condition_1"):
        _apply_config_override(config, "plotting.styling.colors.condition_1", str(args.color_condition_1))
    if _get_arg_value(args, "color_significant"):
        _apply_config_override(config, "plotting.styling.colors.significant", str(args.color_significant))
    if _get_arg_value(args, "color_nonsignificant"):
        _apply_config_override(config, "plotting.styling.colors.nonsignificant", str(args.color_nonsignificant))
    if _get_arg_value(args, "color_gray"):
        _apply_config_override(config, "plotting.styling.colors.gray", str(args.color_gray))
    if _get_arg_value(args, "color_light_gray"):
        _apply_config_override(config, "plotting.styling.colors.light_gray", str(args.color_light_gray))
    if _get_arg_value(args, "color_black"):
        _apply_config_override(config, "plotting.styling.colors.black", str(args.color_black))
    if _get_arg_value(args, "color_blue"):
        _apply_config_override(config, "plotting.styling.colors.blue", str(args.color_blue))
    if _get_arg_value(args, "color_red"):
        _apply_config_override(config, "plotting.styling.colors.red", str(args.color_red))
    if _get_arg_value(args, "color_network_node"):
        _apply_config_override(config, "plotting.styling.colors.network_node", str(args.color_network_node))


def _apply_alpha_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply alpha-related config overrides."""
    if _get_arg_value(args, "alpha_grid") is not None:
        _apply_config_override(config, "plotting.styling.alpha.grid", float(args.alpha_grid))
    if _get_arg_value(args, "alpha_fill") is not None:
        _apply_config_override(config, "plotting.styling.alpha.fill", float(args.alpha_fill))
    if _get_arg_value(args, "alpha_ci") is not None:
        _apply_config_override(config, "plotting.styling.alpha.ci", float(args.alpha_ci))
    if _get_arg_value(args, "alpha_ci_line") is not None:
        _apply_config_override(config, "plotting.styling.alpha.ci_line", float(args.alpha_ci_line))
    if _get_arg_value(args, "alpha_text_box") is not None:
        _apply_config_override(config, "plotting.styling.alpha.text_box", float(args.alpha_text_box))
    if _get_arg_value(args, "alpha_violin_body") is not None:
        _apply_config_override(config, "plotting.styling.alpha.violin_body", float(args.alpha_violin_body))
    if _get_arg_value(args, "alpha_ridge_fill") is not None:
        _apply_config_override(config, "plotting.styling.alpha.ridge_fill", float(args.alpha_ridge_fill))


def _apply_scatter_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply scatter plot styling overrides."""
    if _get_arg_value(args, "scatter_marker_size_small") is not None:
        _apply_config_override(config, "plotting.styling.scatter.marker_size.small", int(args.scatter_marker_size_small))
    if _get_arg_value(args, "scatter_marker_size_large") is not None:
        _apply_config_override(config, "plotting.styling.scatter.marker_size.large", int(args.scatter_marker_size_large))
    if _get_arg_value(args, "scatter_marker_size_default") is not None:
        _apply_config_override(config, "plotting.styling.scatter.marker_size.default", int(args.scatter_marker_size_default))
    if _get_arg_value(args, "scatter_alpha") is not None:
        _apply_config_override(config, "plotting.styling.scatter.alpha", float(args.scatter_alpha))
    if _get_arg_value(args, "scatter_edgecolor"):
        _apply_config_override(config, "plotting.styling.scatter.edgecolor", str(args.scatter_edgecolor))
    if _get_arg_value(args, "scatter_edgewidth") is not None:
        _apply_config_override(config, "plotting.styling.scatter.edgewidth", float(args.scatter_edgewidth))


def _apply_bar_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply bar plot styling overrides."""
    if _get_arg_value(args, "bar_alpha") is not None:
        _apply_config_override(config, "plotting.styling.bar.alpha", float(args.bar_alpha))
    if _get_arg_value(args, "bar_width") is not None:
        _apply_config_override(config, "plotting.styling.bar.width", float(args.bar_width))
    if _get_arg_value(args, "bar_capsize") is not None:
        _apply_config_override(config, "plotting.styling.bar.capsize", int(args.bar_capsize))
    if _get_arg_value(args, "bar_capsize_large") is not None:
        _apply_config_override(config, "plotting.styling.bar.capsize_large", int(args.bar_capsize_large))


def _apply_line_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply line plot styling overrides."""
    if _get_arg_value(args, "line_width_thin") is not None:
        _apply_config_override(config, "plotting.styling.line.width.thin", float(args.line_width_thin))
    if _get_arg_value(args, "line_width_standard") is not None:
        _apply_config_override(config, "plotting.styling.line.width.standard", float(args.line_width_standard))
    if _get_arg_value(args, "line_width_thick") is not None:
        _apply_config_override(config, "plotting.styling.line.width.thick", float(args.line_width_thick))
    if _get_arg_value(args, "line_width_bold") is not None:
        _apply_config_override(config, "plotting.styling.line.width.bold", float(args.line_width_bold))
    if _get_arg_value(args, "line_alpha_standard") is not None:
        _apply_config_override(config, "plotting.styling.line.alpha.standard", float(args.line_alpha_standard))
    if _get_arg_value(args, "line_alpha_dim") is not None:
        _apply_config_override(config, "plotting.styling.line.alpha.dim", float(args.line_alpha_dim))
    if _get_arg_value(args, "line_alpha_zero_line") is not None:
        _apply_config_override(config, "plotting.styling.line.alpha.zero_line", float(args.line_alpha_zero_line))
    if _get_arg_value(args, "line_alpha_fit_line") is not None:
        _apply_config_override(config, "plotting.styling.line.alpha.fit_line", float(args.line_alpha_fit_line))
    if _get_arg_value(args, "line_alpha_diagonal") is not None:
        _apply_config_override(config, "plotting.styling.line.alpha.diagonal", float(args.line_alpha_diagonal))
    if _get_arg_value(args, "line_alpha_reference") is not None:
        _apply_config_override(config, "plotting.styling.line.alpha.reference", float(args.line_alpha_reference))
    if _get_arg_value(args, "line_regression_width") is not None:
        _apply_config_override(config, "plotting.styling.line.regression_width", float(args.line_regression_width))
    if _get_arg_value(args, "line_residual_width") is not None:
        _apply_config_override(config, "plotting.styling.line.residual_width", float(args.line_residual_width))
    if _get_arg_value(args, "line_qq_width") is not None:
        _apply_config_override(config, "plotting.styling.line.qq_width", float(args.line_qq_width))


def _apply_histogram_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply histogram styling overrides."""
    if _get_arg_value(args, "hist_bins") is not None:
        _apply_config_override(config, "plotting.styling.histogram.bins", int(args.hist_bins))
    if _get_arg_value(args, "hist_bins_behavioral") is not None:
        _apply_config_override(config, "plotting.styling.histogram.bins_behavioral", int(args.hist_bins_behavioral))
    if _get_arg_value(args, "hist_bins_residual") is not None:
        _apply_config_override(config, "plotting.styling.histogram.bins_residual", int(args.hist_bins_residual))
    if _get_arg_value(args, "hist_bins_tfr") is not None:
        _apply_config_override(config, "plotting.styling.histogram.bins_tfr", int(args.hist_bins_tfr))
    if _get_arg_value(args, "hist_edgecolor"):
        _apply_config_override(config, "plotting.styling.histogram.edgecolor", str(args.hist_edgecolor))
    if _get_arg_value(args, "hist_edgewidth") is not None:
        _apply_config_override(config, "plotting.styling.histogram.edgewidth", float(args.hist_edgewidth))
    if _get_arg_value(args, "hist_alpha") is not None:
        _apply_config_override(config, "plotting.styling.histogram.alpha", float(args.hist_alpha))
    if _get_arg_value(args, "hist_alpha_residual") is not None:
        _apply_config_override(config, "plotting.styling.histogram.alpha_residual", float(args.hist_alpha_residual))
    if _get_arg_value(args, "hist_alpha_tfr") is not None:
        _apply_config_override(config, "plotting.styling.histogram.alpha_tfr", float(args.hist_alpha_tfr))


def _apply_kde_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply KDE styling overrides."""
    if _get_arg_value(args, "kde_points") is not None:
        _apply_config_override(config, "plotting.styling.kde.points", int(args.kde_points))
    if _get_arg_value(args, "kde_color"):
        _apply_config_override(config, "plotting.styling.kde.color", str(args.kde_color))
    if _get_arg_value(args, "kde_linewidth") is not None:
        _apply_config_override(config, "plotting.styling.kde.linewidth", float(args.kde_linewidth))
    if _get_arg_value(args, "kde_alpha") is not None:
        _apply_config_override(config, "plotting.styling.kde.alpha", float(args.kde_alpha))


def _apply_errorbar_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply errorbar styling overrides."""
    if _get_arg_value(args, "errorbar_markersize") is not None:
        _apply_config_override(config, "plotting.styling.errorbar.markersize", int(args.errorbar_markersize))
    if _get_arg_value(args, "errorbar_capsize") is not None:
        _apply_config_override(config, "plotting.styling.errorbar.capsize", int(args.errorbar_capsize))
    if _get_arg_value(args, "errorbar_capsize_large") is not None:
        _apply_config_override(config, "plotting.styling.errorbar.capsize_large", int(args.errorbar_capsize_large))


def _apply_text_position_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply text position overrides."""
    if _get_arg_value(args, "text_stats_x") is not None:
        _apply_config_override(config, "plotting.styling.text_position.stats_x", float(args.text_stats_x))
    if _get_arg_value(args, "text_stats_y") is not None:
        _apply_config_override(config, "plotting.styling.text_position.stats_y", float(args.text_stats_y))
    if _get_arg_value(args, "text_pvalue_x") is not None:
        _apply_config_override(config, "plotting.styling.text_position.p_value_x", float(args.text_pvalue_x))
    if _get_arg_value(args, "text_pvalue_y") is not None:
        _apply_config_override(config, "plotting.styling.text_position.p_value_y", float(args.text_pvalue_y))
    if _get_arg_value(args, "text_bootstrap_x") is not None:
        _apply_config_override(config, "plotting.styling.text_position.bootstrap_x", float(args.text_bootstrap_x))
    if _get_arg_value(args, "text_bootstrap_y") is not None:
        _apply_config_override(config, "plotting.styling.text_position.bootstrap_y", float(args.text_bootstrap_y))
    if _get_arg_value(args, "text_channel_annotation_x") is not None:
        _apply_config_override(config, "plotting.styling.text_position.channel_annotation_x", float(args.text_channel_annotation_x))
    if _get_arg_value(args, "text_channel_annotation_y") is not None:
        _apply_config_override(config, "plotting.styling.text_position.channel_annotation_y", float(args.text_channel_annotation_y))
    if _get_arg_value(args, "text_title_y") is not None:
        _apply_config_override(config, "plotting.styling.text_position.title_y", float(args.text_title_y))
    if _get_arg_value(args, "text_residual_qc_title_y") is not None:
        _apply_config_override(config, "plotting.styling.text_position.residual_qc_title_y", float(args.text_residual_qc_title_y))


def _apply_validation_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply validation threshold overrides."""
    if _get_arg_value(args, "validation_min_bins_for_calibration") is not None:
        _apply_config_override(config, "plotting.validation.min_bins_for_calibration", int(args.validation_min_bins_for_calibration))
    if _get_arg_value(args, "validation_max_bins_for_calibration") is not None:
        _apply_config_override(config, "plotting.validation.max_bins_for_calibration", int(args.validation_max_bins_for_calibration))
    if _get_arg_value(args, "validation_samples_per_bin") is not None:
        _apply_config_override(config, "plotting.validation.samples_per_bin", int(args.validation_samples_per_bin))
    if _get_arg_value(args, "validation_min_rois_for_fdr") is not None:
        _apply_config_override(config, "plotting.validation.min_rois_for_fdr", int(args.validation_min_rois_for_fdr))
    if _get_arg_value(args, "validation_min_pvalues_for_fdr") is not None:
        _apply_config_override(config, "plotting.validation.min_pvalues_for_fdr", int(args.validation_min_pvalues_for_fdr))


def _apply_tfr_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply TFR-related overrides."""
    if _get_arg_value(args, "tfr_topomap_window_size_ms") is not None:
        _apply_config_override(
            config,
            "time_frequency_analysis.topomap.temporal.window_size_ms",
            float(args.tfr_topomap_window_size_ms),
        )
    if _get_arg_value(args, "tfr_topomap_window_count") is not None:
        _apply_config_override(
            config,
            "time_frequency_analysis.topomap.temporal.window_count",
            int(args.tfr_topomap_window_count),
        )
    if _get_arg_value(args, "tfr_topomap_label_x_position") is not None:
        _apply_config_override(
            config,
            "plotting.plots.tfr.topomap.label_x_position",
            float(args.tfr_topomap_label_x_position),
        )
    if _get_arg_value(args, "tfr_topomap_label_y_position_bottom") is not None:
        _apply_config_override(
            config,
            "plotting.plots.tfr.topomap.label_y_position_bottom",
            float(args.tfr_topomap_label_y_position_bottom),
        )
    if _get_arg_value(args, "tfr_topomap_label_y_position") is not None:
        _apply_config_override(
            config,
            "plotting.plots.tfr.topomap.label_y_position",
            float(args.tfr_topomap_label_y_position),
        )
    if _get_arg_value(args, "tfr_topomap_title_y") is not None:
        _apply_config_override(
            config,
            "plotting.plots.tfr.topomap.title_y",
            float(args.tfr_topomap_title_y),
        )
    if _get_arg_value(args, "tfr_topomap_title_pad") is not None:
        _apply_config_override(
            config,
            "plotting.plots.tfr.topomap.title_pad",
            int(args.tfr_topomap_title_pad),
        )
    if _get_arg_value(args, "tfr_topomap_subplots_right") is not None:
        _apply_config_override(
            config,
            "plotting.plots.tfr.topomap.subplots_right",
            float(args.tfr_topomap_subplots_right),
        )
    if _get_arg_value(args, "tfr_topomap_temporal_hspace") is not None:
        _apply_config_override(
            config,
            "time_frequency_analysis.topomap.temporal.single_subject.hspace",
            float(args.tfr_topomap_temporal_hspace),
        )
    if _get_arg_value(args, "tfr_topomap_temporal_wspace") is not None:
        _apply_config_override(
            config,
            "time_frequency_analysis.topomap.temporal.single_subject.wspace",
            float(args.tfr_topomap_temporal_wspace),
        )
    if _get_arg_value(args, "shared_colorbar") is not None:
        config.setdefault("plotting", {}).setdefault("plots", {}).setdefault("itpc", {})["shared_colorbar"] = args.shared_colorbar
    if _get_arg_value(args, "tfr_log_base") is not None:
        _apply_config_override(config, "plotting.plots.tfr.log_base", float(args.tfr_log_base))
    if _get_arg_value(args, "tfr_percentage_multiplier") is not None:
        _apply_config_override(config, "plotting.plots.tfr.percentage_multiplier", float(args.tfr_percentage_multiplier))


def _apply_topomap_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply topomap plot overrides."""
    if _get_arg_value(args, "topomap_contours") is not None:
        _apply_config_override(config, "plotting.plots.topomap.contours", int(args.topomap_contours))
    if _get_arg_value(args, "topomap_colormap"):
        _apply_config_override(config, "plotting.plots.topomap.colormap", str(args.topomap_colormap))
    if _get_arg_value(args, "topomap_colorbar_fraction") is not None:
        _apply_config_override(config, "plotting.plots.topomap.colorbar_fraction", float(args.topomap_colorbar_fraction))
    if _get_arg_value(args, "topomap_colorbar_pad") is not None:
        _apply_config_override(config, "plotting.plots.topomap.colorbar_pad", float(args.topomap_colorbar_pad))
    if _get_arg_value(args, "topomap_diff_annotation_enabled") is not None:
        _apply_config_override(config, "plotting.plots.topomap.diff_annotation_enabled", bool(args.topomap_diff_annotation_enabled))
    if _get_arg_value(args, "topomap_annotate_descriptive") is not None:
        _apply_config_override(config, "plotting.plots.topomap.annotate_descriptive", bool(args.topomap_annotate_descriptive))
    if _get_arg_value(args, "topomap_sig_mask_marker"):
        _apply_config_override(config, "plotting.plots.topomap.sig_mask_params.marker", str(args.topomap_sig_mask_marker))
    if _get_arg_value(args, "topomap_sig_mask_markerfacecolor"):
        _apply_config_override(config, "plotting.plots.topomap.sig_mask_params.markerfacecolor", str(args.topomap_sig_mask_markerfacecolor))
    if _get_arg_value(args, "topomap_sig_mask_markeredgecolor"):
        _apply_config_override(config, "plotting.plots.topomap.sig_mask_params.markeredgecolor", str(args.topomap_sig_mask_markeredgecolor))
    if _get_arg_value(args, "topomap_sig_mask_linewidth") is not None:
        _apply_config_override(config, "plotting.plots.topomap.sig_mask_params.linewidth", float(args.topomap_sig_mask_linewidth))
    if _get_arg_value(args, "topomap_sig_mask_markersize") is not None:
        _apply_config_override(config, "plotting.plots.topomap.sig_mask_params.markersize", float(args.topomap_sig_mask_markersize))


def _apply_plot_sizing_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply plot type sizing overrides."""
    if _get_arg_value(args, "roi_width_per_band") is not None:
        _apply_config_override(config, "plotting.plots.roi.width_per_band", float(args.roi_width_per_band))
    if _get_arg_value(args, "roi_width_per_metric") is not None:
        _apply_config_override(config, "plotting.plots.roi.width_per_metric", float(args.roi_width_per_metric))
    if _get_arg_value(args, "roi_height_per_roi") is not None:
        _apply_config_override(config, "plotting.plots.roi.height_per_roi", float(args.roi_height_per_roi))
    if _get_arg_value(args, "power_width_per_band") is not None:
        _apply_config_override(config, "plotting.plots.power.width_per_band", float(args.power_width_per_band))
    if _get_arg_value(args, "power_height_per_segment") is not None:
        _apply_config_override(config, "plotting.plots.power.height_per_segment", float(args.power_height_per_segment))
    if _get_arg_value(args, "itpc_width_per_bin") is not None:
        _apply_config_override(config, "plotting.plots.itpc.width_per_bin", float(args.itpc_width_per_bin))
    if _get_arg_value(args, "itpc_height_per_band") is not None:
        _apply_config_override(config, "plotting.plots.itpc.height_per_band", float(args.itpc_height_per_band))
    if _get_arg_value(args, "itpc_width_per_band_box") is not None:
        _apply_config_override(config, "plotting.plots.itpc.width_per_band_box", float(args.itpc_width_per_band_box))
    if _get_arg_value(args, "itpc_height_box") is not None:
        _apply_config_override(config, "plotting.plots.itpc.height_box", float(args.itpc_height_box))
    if _get_arg_value(args, "pac_cmap"):
        _apply_config_override(config, "plotting.plots.pac.cmap", str(args.pac_cmap))
    if _get_arg_value(args, "pac_width_per_roi") is not None:
        _apply_config_override(config, "plotting.plots.pac.width_per_roi", float(args.pac_width_per_roi))
    if _get_arg_value(args, "pac_height_box") is not None:
        _apply_config_override(config, "plotting.plots.pac.height_box", float(args.pac_height_box))
    if _get_arg_value(args, "aperiodic_width_per_column") is not None:
        _apply_config_override(config, "plotting.plots.aperiodic.width_per_column", float(args.aperiodic_width_per_column))
    if _get_arg_value(args, "aperiodic_height_per_row") is not None:
        _apply_config_override(config, "plotting.plots.aperiodic.height_per_row", float(args.aperiodic_height_per_row))
    if _get_arg_value(args, "aperiodic_n_perm") is not None:
        _apply_config_override(config, "plotting.plots.aperiodic.n_perm", int(args.aperiodic_n_perm))
    if _get_arg_value(args, "complexity_width_per_measure") is not None:
        _apply_config_override(config, "plotting.plots.complexity.width_per_measure", float(args.complexity_width_per_measure))
    if _get_arg_value(args, "complexity_height_per_segment") is not None:
        _apply_config_override(config, "plotting.plots.complexity.height_per_segment", float(args.complexity_height_per_segment))
    if _get_arg_value(args, "connectivity_width_per_circle") is not None:
        _apply_config_override(config, "plotting.plots.connectivity.width_per_circle", float(args.connectivity_width_per_circle))
    if _get_arg_value(args, "connectivity_width_per_band") is not None:
        _apply_config_override(config, "plotting.plots.connectivity.width_per_band", float(args.connectivity_width_per_band))
    if _get_arg_value(args, "connectivity_height_per_measure") is not None:
        _apply_config_override(config, "plotting.plots.connectivity.height_per_measure", float(args.connectivity_height_per_measure))
    if _get_arg_value(args, "connectivity_circle_top_fraction") is not None:
        _apply_config_override(config, "plotting.plots.features.connectivity.circle_top_fraction", float(args.connectivity_circle_top_fraction))
    if _get_arg_value(args, "connectivity_circle_min_lines") is not None:
        _apply_config_override(config, "plotting.plots.features.connectivity.circle_min_lines", int(args.connectivity_circle_min_lines))
    if _get_arg_value(args, "connectivity_network_top_fraction") is not None:
        _apply_config_override(config, "plotting.plots.features.connectivity.network_top_fraction", float(args.connectivity_network_top_fraction))


def _apply_feature_selection_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply feature plot ordering/selection overrides."""
    if _get_arg_value(args, "pac_pairs"):
        _apply_config_override(config, "plotting.plots.features.pac_pairs", list(args.pac_pairs))
    if _get_arg_value(args, "connectivity_measures"):
        _apply_config_override(config, "plotting.plots.features.connectivity.measures", list(args.connectivity_measures))
    if _get_arg_value(args, "spectral_metrics"):
        _apply_config_override(config, "plotting.plots.features.spectral.metrics", list(args.spectral_metrics))
    if _get_arg_value(args, "bursts_metrics"):
        _apply_config_override(config, "plotting.plots.features.bursts.metrics", list(args.bursts_metrics))
    if _get_arg_value(args, "asymmetry_stat"):
        _apply_config_override(config, "plotting.plots.features.asymmetry.stat", str(args.asymmetry_stat))
    if _get_arg_value(args, "temporal_time_bins"):
        _apply_config_override(config, "plotting.plots.features.temporal.time_bins", list(args.temporal_time_bins))
    if _get_arg_value(args, "temporal_time_labels"):
        _apply_config_override(config, "plotting.plots.features.temporal.time_labels", list(args.temporal_time_labels))


def _apply_source_localization_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply source localization plot overrides."""
    if _get_arg_value(args, "source_plot_hemi"):
        _apply_config_override(config, "plotting.plots.features.sourcelocalization.hemi", str(args.source_plot_hemi))
    if _get_arg_value(args, "source_plot_views"):
        _apply_config_override(config, "plotting.plots.features.sourcelocalization.views", list(args.source_plot_views))
    if _get_arg_value(args, "source_plot_cortex"):
        _apply_config_override(config, "plotting.plots.features.sourcelocalization.cortex", str(args.source_plot_cortex))
    if _get_arg_value(args, "source_subjects_dir"):
        _apply_config_override(config, "feature_engineering.sourcelocalization.subjects_dir", str(args.source_subjects_dir))


def _apply_comparison_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply comparison-related overrides."""
    if _get_arg_value(args, "compare_windows") is not None:
        _apply_config_override(config, "plotting.comparisons.compare_windows", bool(args.compare_windows))
    if _get_arg_value(args, "comparison_windows"):
        _apply_config_override(config, "plotting.comparisons.comparison_windows", list(args.comparison_windows))
    if _get_arg_value(args, "compare_columns") is not None:
        _apply_config_override(config, "plotting.comparisons.compare_columns", bool(args.compare_columns))
    if _get_arg_value(args, "comparison_segment"):
        _apply_config_override(config, "plotting.comparisons.comparison_segment", str(args.comparison_segment))
    if _get_arg_value(args, "comparison_column"):
        _apply_config_override(config, "plotting.comparisons.comparison_column", str(args.comparison_column))
    if _get_arg_value(args, "comparison_values"):
        _apply_config_override(config, "plotting.comparisons.comparison_values", list(args.comparison_values))
    if _get_arg_value(args, "comparison_labels"):
        _apply_config_override(config, "plotting.comparisons.comparison_labels", list(args.comparison_labels))
    if _get_arg_value(args, "comparison_rois"):
        _apply_config_override(config, "plotting.comparisons.comparison_rois", list(args.comparison_rois))


def _apply_output_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply output-related overrides."""
    if _get_arg_value(args, "overwrite") is not None:
        _apply_config_override(config, "plotting.overwrite", bool(args.overwrite))


def _apply_roi_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply custom ROI definitions to config for plotting.

    Sets ROIs in both locations used by different plotting subsystems:
    - Top-level 'rois': Used by get_roi_definitions() in feature plots
    - 'time_frequency_analysis.rois': Used by get_rois() in TFR plots
    """
    if _get_arg_value(args, "rois"):
        custom_rois = parse_roi_definitions(args.rois)
        # Apply to top-level rois (used by get_roi_definitions in plotting/features)
        config["rois"] = custom_rois
        # Apply to TFR-specific rois (used by get_rois in TFR extraction)
        config["time_frequency_analysis.rois"] = custom_rois


def _apply_band_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply custom frequency band definitions to config for plotting.

    If --frequency-bands is specified, uses only those definitions.
    If --bands is also specified, filters to only selected bands.
    
    Sets bands in both locations used by different plotting subsystems:
    - Top-level 'frequency_bands': Used by get_frequency_bands() in feature plots
    - 'time_frequency_analysis.bands': Used by TFR analysis
    """
    if _get_arg_value(args, "frequency_bands"):
        custom_bands = parse_frequency_band_definitions(args.frequency_bands)
        
        selected_bands = getattr(args, "bands", None)
        if selected_bands:
            selected_lower = [b.lower() for b in selected_bands]
            filtered_bands = {k: v for k, v in custom_bands.items() if k.lower() in selected_lower}
            custom_bands = filtered_bands
        
        # Apply to top-level frequency_bands (used by get_frequency_bands)
        config["frequency_bands"] = custom_bands
        # Apply to TFR-specific bands (used by TFR analysis)
        config["time_frequency_analysis.bands"] = custom_bands


def apply_all_config_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply all config overrides from CLI arguments."""
    _apply_defaults_overrides(args, config)
    _apply_font_overrides(args, config)
    _apply_layout_overrides(args, config)
    _apply_figure_size_overrides(args, config)
    _apply_color_overrides(args, config)
    _apply_alpha_overrides(args, config)
    _apply_scatter_overrides(args, config)
    _apply_bar_overrides(args, config)
    _apply_line_overrides(args, config)
    _apply_histogram_overrides(args, config)
    _apply_kde_overrides(args, config)
    _apply_errorbar_overrides(args, config)
    _apply_text_position_overrides(args, config)
    _apply_validation_overrides(args, config)
    _apply_tfr_overrides(args, config)
    _apply_topomap_overrides(args, config)
    _apply_plot_sizing_overrides(args, config)
    _apply_feature_selection_overrides(args, config)
    _apply_source_localization_overrides(args, config)
    _apply_comparison_overrides(args, config)
    _apply_roi_overrides(args, config)
    _apply_band_overrides(args, config)
    _apply_output_overrides(args, config)
