"""Parser construction for plotting CLI command."""

from __future__ import annotations

import argparse

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_task_arg,
    add_output_format_args,
    add_path_args,
)
from eeg_pipeline.cli.commands.plotting_catalog import PLOT_BY_ID, PLOT_GROUPS


def setup_plotting(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the plotting command parser."""
    parser = subparsers.add_parser(
        "plotting",
        help="Plotting pipeline: curate visualization suites",
        description="Plotting pipeline: select and render visualization suites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=["visualize", "tfr"], help="Pipeline mode (visualize or tfr)")
    add_common_subject_args(parser)
    add_task_arg(parser)
    add_output_format_args(parser)
    parser.add_argument(
        "--analysis-scope",
        choices=["subject", "group"],
        default="subject",
        help="Compute plots per subject (default) or as a group aggregate (limited support)",
    )

    parser.add_argument(
        "--plots",
        nargs="+",
        choices=sorted(PLOT_BY_ID.keys()),
        default=None,
        metavar="PLOT_ID",
        help="Specific plot IDs to render",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=sorted(PLOT_GROUPS.keys()),
        default=None,
        metavar="GROUP",
        help="Plot groups to render (features, behavior, tfr, erp)",
    )
    parser.add_argument(
        "--all-plots",
        action="store_true",
        help="Render all available plots",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "svg", "pdf"],
        default=None,
        metavar="FMT",
        help="Output formats (default from config)",
    )
    parser.add_argument("--dpi", type=int, default=None, help="Figure DPI (default from config)")
    parser.add_argument("--savefig-dpi", type=int, default=None, help="Savefig DPI (default from config)")
    parser.add_argument("--shared-colorbar", action="store_true", default=None, help="Use shared colorbar across subplots")
    parser.add_argument("--no-shared-colorbar", action="store_false", dest="shared_colorbar", help="Disable shared colorbar")

    defaults_overrides = parser.add_argument_group("Plot defaults & styling overrides")
    defaults_overrides.add_argument("--bbox-inches", type=str, default=None, help="Matplotlib bbox_inches for savefig (default from config)")
    defaults_overrides.add_argument("--pad-inches", type=float, default=None, help="Matplotlib pad_inches for savefig (default from config)")

    # Fonts (plotting.defaults.font.*)
    defaults_overrides.add_argument("--font-family", type=str, default=None, help="Font family (default from config)")
    defaults_overrides.add_argument("--font-weight", type=str, default=None, help="Font weight (default from config)")
    defaults_overrides.add_argument("--font-size-small", type=int, default=None, help="Small font size (default from config)")
    defaults_overrides.add_argument("--font-size-medium", type=int, default=None, help="Medium font size (default from config)")
    defaults_overrides.add_argument("--font-size-large", type=int, default=None, help="Large font size (default from config)")
    defaults_overrides.add_argument("--font-size-title", type=int, default=None, help="Title font size (default from config)")
    defaults_overrides.add_argument("--font-size-annotation", type=int, default=None, help="Annotation font size (default from config)")
    defaults_overrides.add_argument("--font-size-label", type=int, default=None, help="Label font size (default from config)")
    defaults_overrides.add_argument("--font-size-ylabel", type=int, default=None, help="Y-label font size (default from config)")
    defaults_overrides.add_argument("--font-size-suptitle", type=int, default=None, help="Suptitle font size (default from config)")
    defaults_overrides.add_argument("--font-size-figure-title", type=int, default=None, help="Figure title font size (default from config)")

    # Layout (plotting.defaults.layout.*)
    defaults_overrides.add_argument(
        "--layout-tight-rect",
        nargs=4,
        type=float,
        default=None,
        metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"),
        help="tight_layout rect: left bottom right top (default from config)",
    )
    defaults_overrides.add_argument(
        "--gridspec-width-ratios",
        nargs="+",
        type=float,
        default=None,
        metavar="RATIO",
        help="gridspec width_ratios (default from config)",
    )
    defaults_overrides.add_argument(
        "--gridspec-height-ratios",
        nargs="+",
        type=float,
        default=None,
        metavar="RATIO",
        help="gridspec height_ratios (default from config)",
    )
    defaults_overrides.add_argument("--gridspec-hspace", type=float, default=None, help="gridspec hspace (default from config)")
    defaults_overrides.add_argument("--gridspec-wspace", type=float, default=None, help="gridspec wspace (default from config)")
    defaults_overrides.add_argument("--gridspec-left", type=float, default=None, help="gridspec left (default from config)")
    defaults_overrides.add_argument("--gridspec-right", type=float, default=None, help="gridspec right (default from config)")
    defaults_overrides.add_argument("--gridspec-top", type=float, default=None, help="gridspec top (default from config)")
    defaults_overrides.add_argument("--gridspec-bottom", type=float, default=None, help="gridspec bottom (default from config)")

    # Figure sizes (plotting.figure_sizes.*)
    defaults_overrides.add_argument("--figure-size-standard", nargs=2, type=float, default=None, metavar=("W", "H"), help="Standard figure size (default from config)")
    defaults_overrides.add_argument("--figure-size-medium", nargs=2, type=float, default=None, metavar=("W", "H"), help="Medium figure size (default from config)")
    defaults_overrides.add_argument("--figure-size-small", nargs=2, type=float, default=None, metavar=("W", "H"), help="Small figure size (default from config)")
    defaults_overrides.add_argument("--figure-size-square", nargs=2, type=float, default=None, metavar=("W", "H"), help="Square figure size (default from config)")
    defaults_overrides.add_argument("--figure-size-wide", nargs=2, type=float, default=None, metavar=("W", "H"), help="Wide figure size (default from config)")
    defaults_overrides.add_argument("--figure-size-tfr", nargs=2, type=float, default=None, metavar=("W", "H"), help="TFR figure size (default from config)")
    defaults_overrides.add_argument("--figure-size-topomap", nargs=2, type=float, default=None, metavar=("W", "H"), help="Topomap figure size (default from config)")

    # Styling colors (plotting.styling.colors.*)
    defaults_overrides.add_argument(
        "--color-condition-1",
        type=str,
        default=None,
        dest="color_condition_1",
        help="Color for comparison condition 1 (value1/label1) (default from config)",
    )
    defaults_overrides.add_argument(
        "--color-condition-2",
        type=str,
        default=None,
        dest="color_condition_2",
        help="Color for comparison condition 2 (value2/label2) (default from config)",
    )
    defaults_overrides.add_argument("--color-significant", type=str, default=None, help="Color for significant items (default from config)")
    defaults_overrides.add_argument("--color-nonsignificant", type=str, default=None, help="Color for non-significant items (default from config)")
    defaults_overrides.add_argument("--color-gray", type=str, default=None, help="Gray color (default from config)")
    defaults_overrides.add_argument("--color-light-gray", type=str, default=None, help="Light gray color (default from config)")
    defaults_overrides.add_argument("--color-black", type=str, default=None, help="Black color (default from config)")
    defaults_overrides.add_argument("--color-blue", type=str, default=None, help="Blue color (default from config)")
    defaults_overrides.add_argument("--color-red", type=str, default=None, help="Red color (default from config)")
    defaults_overrides.add_argument("--color-network-node", type=str, default=None, help="Network node color (default from config)")

    # Styling alpha (plotting.styling.alpha.*)
    defaults_overrides.add_argument("--alpha-grid", type=float, default=None, help="Grid alpha (default from config)")
    defaults_overrides.add_argument("--alpha-fill", type=float, default=None, help="Fill alpha (default from config)")
    defaults_overrides.add_argument("--alpha-ci", type=float, default=None, help="Confidence interval fill alpha (default from config)")
    defaults_overrides.add_argument("--alpha-ci-line", type=float, default=None, help="Confidence interval line alpha (default from config)")
    defaults_overrides.add_argument("--alpha-text-box", type=float, default=None, help="Text box alpha (default from config)")
    defaults_overrides.add_argument("--alpha-violin-body", type=float, default=None, help="Violin body alpha (default from config)")
    defaults_overrides.add_argument("--alpha-ridge-fill", type=float, default=None, help="Ridge fill alpha (default from config)")

    # Scatter styling (plotting.styling.scatter.*)
    defaults_overrides.add_argument("--scatter-marker-size-small", type=int, default=None, help="Scatter marker size (small) (default from config)")
    defaults_overrides.add_argument("--scatter-marker-size-large", type=int, default=None, help="Scatter marker size (large) (default from config)")
    defaults_overrides.add_argument("--scatter-marker-size-default", type=int, default=None, help="Scatter marker size (default) (default from config)")
    defaults_overrides.add_argument("--scatter-alpha", type=float, default=None, help="Scatter alpha (default from config)")
    defaults_overrides.add_argument("--scatter-edgecolor", type=str, default=None, help="Scatter edgecolor (default from config)")
    defaults_overrides.add_argument("--scatter-edgewidth", type=float, default=None, help="Scatter edgewidth (default from config)")

    # Bar styling (plotting.styling.bar.*)
    defaults_overrides.add_argument("--bar-alpha", type=float, default=None, help="Bar alpha (default from config)")
    defaults_overrides.add_argument("--bar-width", type=float, default=None, help="Bar width (default from config)")
    defaults_overrides.add_argument("--bar-capsize", type=int, default=None, help="Bar capsize (default from config)")
    defaults_overrides.add_argument("--bar-capsize-large", type=int, default=None, help="Bar capsize (large) (default from config)")

    # Line styling (plotting.styling.line.*)
    defaults_overrides.add_argument("--line-width-thin", type=float, default=None, help="Line width thin (default from config)")
    defaults_overrides.add_argument("--line-width-standard", type=float, default=None, help="Line width standard (default from config)")
    defaults_overrides.add_argument("--line-width-thick", type=float, default=None, help="Line width thick (default from config)")
    defaults_overrides.add_argument("--line-width-bold", type=float, default=None, help="Line width bold (default from config)")
    defaults_overrides.add_argument("--line-alpha-standard", type=float, default=None, help="Line alpha standard (default from config)")
    defaults_overrides.add_argument("--line-alpha-dim", type=float, default=None, help="Line alpha dim (default from config)")
    defaults_overrides.add_argument("--line-alpha-zero-line", type=float, default=None, help="Zero-line alpha (default from config)")
    defaults_overrides.add_argument("--line-alpha-fit-line", type=float, default=None, help="Fit line alpha (default from config)")
    defaults_overrides.add_argument("--line-alpha-diagonal", type=float, default=None, help="Diagonal alpha (default from config)")
    defaults_overrides.add_argument("--line-alpha-reference", type=float, default=None, help="Reference alpha (default from config)")
    defaults_overrides.add_argument("--line-regression-width", type=float, default=None, help="Regression line width (default from config)")
    defaults_overrides.add_argument("--line-residual-width", type=float, default=None, help="Residual line width (default from config)")
    defaults_overrides.add_argument("--line-qq-width", type=float, default=None, help="QQ line width (default from config)")

    # Histogram styling (plotting.styling.histogram.*)
    defaults_overrides.add_argument("--hist-bins", type=int, default=None, help="Histogram bins (default from config)")
    defaults_overrides.add_argument("--hist-bins-behavioral", type=int, default=None, help="Behavior histogram bins (default from config)")
    defaults_overrides.add_argument("--hist-bins-residual", type=int, default=None, help="Residual histogram bins (default from config)")
    defaults_overrides.add_argument("--hist-bins-tfr", type=int, default=None, help="TFR histogram bins (default from config)")
    defaults_overrides.add_argument("--hist-edgecolor", type=str, default=None, help="Histogram edgecolor (default from config)")
    defaults_overrides.add_argument("--hist-edgewidth", type=float, default=None, help="Histogram edgewidth (default from config)")
    defaults_overrides.add_argument("--hist-alpha", type=float, default=None, help="Histogram alpha (default from config)")
    defaults_overrides.add_argument("--hist-alpha-residual", type=float, default=None, help="Residual histogram alpha (default from config)")
    defaults_overrides.add_argument("--hist-alpha-tfr", type=float, default=None, help="TFR histogram alpha (default from config)")

    # KDE styling (plotting.styling.kde.*)
    defaults_overrides.add_argument("--kde-points", type=int, default=None, help="KDE points (default from config)")
    defaults_overrides.add_argument("--kde-color", type=str, default=None, help="KDE color (default from config)")
    defaults_overrides.add_argument("--kde-linewidth", type=float, default=None, help="KDE linewidth (default from config)")
    defaults_overrides.add_argument("--kde-alpha", type=float, default=None, help="KDE alpha (default from config)")

    # Errorbar styling (plotting.styling.errorbar.*)
    defaults_overrides.add_argument("--errorbar-markersize", type=int, default=None, help="Errorbar markersize (default from config)")
    defaults_overrides.add_argument("--errorbar-capsize", type=int, default=None, help="Errorbar capsize (default from config)")
    defaults_overrides.add_argument("--errorbar-capsize-large", type=int, default=None, help="Errorbar capsize (large) (default from config)")

    # Text positions (plotting.styling.text_position.*)
    defaults_overrides.add_argument("--text-stats-x", type=float, default=None, help="Stats text x position (default from config)")
    defaults_overrides.add_argument("--text-stats-y", type=float, default=None, help="Stats text y position (default from config)")
    defaults_overrides.add_argument("--text-pvalue-x", type=float, default=None, help="P-value text x position (default from config)")
    defaults_overrides.add_argument("--text-pvalue-y", type=float, default=None, help="P-value text y position (default from config)")
    defaults_overrides.add_argument("--text-bootstrap-x", type=float, default=None, help="Bootstrap text x position (default from config)")
    defaults_overrides.add_argument("--text-bootstrap-y", type=float, default=None, help="Bootstrap text y position (default from config)")
    defaults_overrides.add_argument("--text-channel-annotation-x", type=float, default=None, help="Channel annotation x position (default from config)")
    defaults_overrides.add_argument("--text-channel-annotation-y", type=float, default=None, help="Channel annotation y position (default from config)")
    defaults_overrides.add_argument("--text-title-y", type=float, default=None, help="Title y position (default from config)")

    # Validation thresholds (plotting.validation.*)
    defaults_overrides.add_argument("--validation-min-bins-for-calibration", type=int, default=None, help="Min bins for calibration (default from config)")
    defaults_overrides.add_argument("--validation-max-bins-for-calibration", type=int, default=None, help="Max bins for calibration (default from config)")
    defaults_overrides.add_argument("--validation-samples-per-bin", type=int, default=None, help="Samples per bin for calibration (default from config)")
    defaults_overrides.add_argument("--validation-min-rois-for-fdr", type=int, default=None, help="Min ROIs for FDR correction (default from config)")
    defaults_overrides.add_argument("--validation-min-pvalues-for-fdr", type=int, default=None, help="Min p-values for FDR correction (default from config)")


    overrides = parser.add_argument_group("Plot overrides")

    # Topomap controls (plotting.plots.topomap.*)
    overrides.add_argument("--topomap-contours", type=int, default=None, help="Topomap contour lines (default from config)")
    overrides.add_argument("--topomap-colormap", type=str, default=None, help="Topomap colormap (default from config)")
    overrides.add_argument("--topomap-colorbar-fraction", type=float, default=None, help="Topomap colorbar fraction (default from config)")
    overrides.add_argument("--topomap-colorbar-pad", type=float, default=None, help="Topomap colorbar pad (default from config)")
    overrides.add_argument("--topomap-diff-annotation-enabled", action="store_true", default=None, help="Enable topomap diff annotation")
    overrides.add_argument("--no-topomap-diff-annotation-enabled", action="store_false", dest="topomap_diff_annotation_enabled", help="Disable topomap diff annotation")
    overrides.add_argument("--topomap-annotate-descriptive", action="store_true", default=None, help="Annotate descriptive topomaps with a note")
    overrides.add_argument("--no-topomap-annotate-descriptive", action="store_false", dest="topomap_annotate_descriptive", help="Disable descriptive topomap note")
    overrides.add_argument("--topomap-sig-mask-marker", type=str, default=None, help="Significance mask marker (default from config)")
    overrides.add_argument("--topomap-sig-mask-markerfacecolor", type=str, default=None, help="Significance mask marker facecolor (default from config)")
    overrides.add_argument("--topomap-sig-mask-markeredgecolor", type=str, default=None, help="Significance mask marker edgecolor (default from config)")
    overrides.add_argument("--topomap-sig-mask-linewidth", type=float, default=None, help="Significance mask marker linewidth (default from config)")
    overrides.add_argument("--topomap-sig-mask-markersize", type=float, default=None, help="Significance mask marker size (default from config)")

    # TFR controls (plotting.plots.tfr.*)
    overrides.add_argument("--tfr-log-base", type=float, default=None, help="Log base for log-ratio conversions (default from config)")
    overrides.add_argument("--tfr-percentage-multiplier", type=float, default=None, help="Percent multiplier for log-ratio conversions (default from config)")

    # Figure sizing controls (plotting.plots.<plot_type>.*)
    overrides.add_argument("--roi-width-per-band", type=float, default=None, help="ROI plot width per band (default from config)")
    overrides.add_argument("--roi-width-per-metric", type=float, default=None, help="ROI plot width per metric (default from config)")
    overrides.add_argument("--roi-height-per-roi", type=float, default=None, help="ROI plot height per ROI (default from config)")

    overrides.add_argument("--power-width-per-band", type=float, default=None, help="Power plot width per band (default from config)")
    overrides.add_argument("--power-height-per-segment", type=float, default=None, help="Power plot height per segment (default from config)")

    overrides.add_argument("--itpc-width-per-bin", type=float, default=None, help="ITPC plot width per bin (default from config)")
    overrides.add_argument("--itpc-height-per-band", type=float, default=None, help="ITPC plot height per band (default from config)")
    overrides.add_argument("--itpc-width-per-band-box", type=float, default=None, help="ITPC boxplot width per band (default from config)")
    overrides.add_argument("--itpc-height-box", type=float, default=None, help="ITPC boxplot height (default from config)")

    overrides.add_argument("--pac-cmap", type=str, default=None, help="PAC colormap (default from config)")
    overrides.add_argument("--pac-width-per-roi", type=float, default=None, help="PAC plot width per ROI/pair (default from config)")
    overrides.add_argument("--pac-height-box", type=float, default=None, help="PAC boxplot height (default from config)")

    overrides.add_argument("--aperiodic-width-per-column", type=float, default=None, help="Aperiodic plot width per column (default from config)")
    overrides.add_argument("--aperiodic-height-per-row", type=float, default=None, help="Aperiodic plot height per row (default from config)")
    overrides.add_argument("--aperiodic-n-perm", type=int, default=None, help="Aperiodic permutation count for comparisons (default from config)")


    overrides.add_argument("--complexity-width-per-measure", type=float, default=None, help="Complexity plot width per measure (default from config)")
    overrides.add_argument("--complexity-height-per-segment", type=float, default=None, help="Complexity plot height per segment (default from config)")

    overrides.add_argument("--connectivity-width-per-circle", type=float, default=None, help="Connectivity circle plot width (default from config)")
    overrides.add_argument("--connectivity-width-per-band", type=float, default=None, help="Connectivity plot width per band (default from config)")
    overrides.add_argument("--connectivity-height-per-measure", type=float, default=None, help="Connectivity plot height per measure (default from config)")
    overrides.add_argument("--connectivity-circle-top-fraction", type=float, default=None, help="Connectivity circle top fraction (default from config)")
    overrides.add_argument("--connectivity-circle-min-lines", type=int, default=None, help="Connectivity circle min lines (default from config)")

    overrides.add_argument("--source-subjects-dir", type=str, default=None, help="FreeSurfer subjects directory for 3D plotting")

    # Ordering / selection controls (plotting.plots.features.*)
    overrides.add_argument("--pac-pairs", nargs="+", default=None, metavar="PAIR", help="PAC pair ordering tokens (e.g. theta_gamma alpha_beta)")
    overrides.add_argument("--connectivity-measures", nargs="+", default=None, metavar="MEASURE", help="Connectivity measures to visualize (e.g. aec wpli)")
    overrides.add_argument("--spectral-metrics", nargs="+", default=None, metavar="METRIC", help="Spectral metrics to plot (default from config)")
    overrides.add_argument("--bursts-metrics", nargs="+", default=None, metavar="METRIC", help="Burst metrics to plot (default from config)")
    overrides.add_argument("--asymmetry-stat", type=str, default=None, help="Asymmetry stat to plot (default from config)")
    overrides.add_argument("--temporal-time-bins", nargs="+", default=None, metavar="BIN", help="Temporal bin segment names (default from config)")
    overrides.add_argument("--temporal-time-labels", nargs="+", default=None, metavar="LABEL", help="Temporal bin labels (default from config)")
    overrides.add_argument(
        "--feature-plotters",
        nargs="+",
        default=None,
        metavar="PLOTTER_ID",
        help="Feature plotters to run (IDs like power.plot_tfr); default runs all plotters for selected feature suites",
    )

    comparisons = parser.add_argument_group("Plotting comparisons")
    comparisons.add_argument("--compare-windows", action="store_true", default=None, help="Enable paired comparisons between time windows")
    comparisons.add_argument("--no-compare-windows", action="store_false", dest="compare_windows", help="Disable time window comparisons")
    comparisons.add_argument("--comparison-windows", nargs="+", default=None, metavar="WINDOW", help="Time windows for paired comparison (e.g. baseline active)")
    comparisons.add_argument("--compare-columns", action="store_true", default=None, help="Enable paired comparisons between event columns")
    comparisons.add_argument("--no-compare-columns", action="store_false", dest="compare_columns", help="Disable column comparisons")
    comparisons.add_argument("--comparison-segment", type=str, default=None, help="Segment name for column comparisons (default from config)")
    comparisons.add_argument("--comparison-column", type=str, default=None, help="Column name from events.tsv for comparisons (e.g. condition)")
    comparisons.add_argument("--comparison-values", nargs="+", default=None, metavar="VALUE", help="Values in the comparison column for paired plots (e.g. 0 1)")
    comparisons.add_argument(
        "--comparison-labels",
        nargs=2,
        default=None,
        metavar=("LABEL1", "LABEL2"),
        help="Display labels for the comparison values (e.g. condition_A condition_B)",
    )
    comparisons.add_argument("--comparison-rois", nargs="+", default=None, metavar="ROI", help="ROIs/Channels to include in paired comparison plots (default: all)")

    per_plot = parser.add_argument_group("Per-plot overrides")
    per_plot.add_argument(
        "--plot-item-config",
        nargs="+",
        action="append",
        default=None,
        metavar="ITEM",
        help=(
            "Per-plot overrides; can be repeated. "
            "Example: --plot-item-config tfr_scalpmean compare_windows true. "
            "Keys: compare_windows, comparison_windows, compare_columns, "
            "comparison_segment, comparison_column, comparison_values, comparison_labels, comparison_rois, "
            "source_segment, source_condition, source_subjects_dir, source_condition_a, source_condition_b, "
            "topomap_windows (or topomap_window for single value), tfr_topomap_active_window, "
            "tfr_topomap_window_size_ms, tfr_topomap_window_count, tfr_topomap_label_x_position, tfr_topomap_label_y_position_bottom, "
            "tfr_topomap_label_y_position, tfr_topomap_title_y, tfr_topomap_title_pad, "
            "tfr_topomap_subplots_right, tfr_topomap_temporal_hspace, tfr_topomap_temporal_wspace, "
            "connectivity_circle_top_fraction, connectivity_circle_min_lines, "
            "connectivity_network_top_fraction, itpc_shared_colorbar, "
            "scatter_features, scatter_columns, scatter_aggregation_modes, scatter_segment, "
            "temporal_stats_feature_folder."
        ),
    )

    add_path_args(parser)

    output_group = parser.add_argument_group("Output options")
    output_group.add_argument("--overwrite", action="store_true", default=None, help="Overwrite existing plot files (default from config)")
    output_group.add_argument("--no-overwrite", dest="overwrite", action="store_false", help="Skip existing plot files")

    tfr_group = parser.add_argument_group("TFR visualization options (mode: tfr)")
    tfr_group.add_argument(
        "--tfr-roi",
        action="store_true",
        help="ROI-only visualization",
    )
    tfr_group.add_argument(
        "--tfr-topomaps-only",
        action="store_true",
        help="Topomaps only",
    )
    tfr_group.add_argument(
        "--tmin",
        type=float,
        default=None,
        help="Start time in seconds",
    )
    tfr_group.add_argument(
        "--tmax",
        type=float,
        default=None,
        help="End time in seconds",
    )
    tfr_group.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel jobs (default: from config)",
    )

    roi_group = parser.add_argument_group("ROI configuration")
    roi_group.add_argument(
        "--rois",
        nargs="+",
        default=None,
        metavar="ROI_DEF",
        help="Custom ROI definitions in format 'name:ch1,ch2,...' (e.g., 'Frontal:Fp1,Fp2,F3,F4'). These ROIs will be used for all ROI-based visualizations.",
    )

    band_group = parser.add_argument_group("Band configuration")
    band_group.add_argument(
        "--bands",
        nargs="+",
        default=None,
        metavar="BAND",
        help="Select specific frequency bands to use (e.g., 'theta alpha beta'). Default: all bands.",
    )
    band_group.add_argument(
        "--frequency-bands",
        nargs="+",
        default=None,
        metavar="BAND_DEF",
        help="Custom frequency band definitions in format 'name:low:high' (e.g., 'theta:4.0:8.0 alpha:8.0:13.0'). Overrides default band frequencies.",
    )

    return parser
