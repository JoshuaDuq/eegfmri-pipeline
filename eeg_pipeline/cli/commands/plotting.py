"""Plotting orchestration CLI command."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_task_arg,
    add_output_format_args,
    add_path_args,
    create_progress_reporter,
    resolve_task,
    get_deriv_root,
)
from eeg_pipeline.utils.config.loader import ConfigDict


@dataclass(frozen=True)
class PlotDefinition:
    plot_id: str
    group: str
    label: str
    description: str
    required_files: List[str]
    feature_categories: Optional[List[str]] = None
    feature_plot_patterns: Optional[List[str]] = None
    behavior_plots: Optional[List[str]] = None
    tfr_plots: Optional[List[str]] = None
    erp_plots: Optional[List[str]] = None
    ml_mode: Optional[str] = None
    requires_epochs: bool = False
    requires_features: bool = False
    requires_stats: bool = False


def _load_plot_catalog() -> List[PlotDefinition]:
    catalog_path = Path(__file__).resolve().parents[2] / "plotting" / "plot_catalog.json"
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    plots: List[PlotDefinition] = []
    for entry in payload.get("plots", []):
        plots.append(
            PlotDefinition(
                plot_id=str(entry["id"]),
                group=str(entry["group"]),
                label=str(entry.get("label", "")),
                description=str(entry.get("description", "")),
                required_files=list(entry.get("required_files", [])),
                feature_categories=entry.get("feature_categories"),
                feature_plot_patterns=entry.get("feature_plot_patterns"),
                behavior_plots=entry.get("behavior_plots"),
                tfr_plots=entry.get("tfr_plots"),
                erp_plots=entry.get("erp_plots"),
                ml_mode=entry.get("ml_mode"),
                requires_epochs=bool(entry.get("requires_epochs", False)),
                requires_features=bool(entry.get("requires_features", False)),
                requires_stats=bool(entry.get("requires_stats", False)),
            )
        )
    return plots


PLOT_CATALOG: List[PlotDefinition] = _load_plot_catalog()

PLOT_BY_ID: Dict[str, PlotDefinition] = {plot.plot_id: plot for plot in PLOT_CATALOG}
PLOT_GROUPS: Dict[str, List[str]] = {}
for plot in PLOT_CATALOG:
    PLOT_GROUPS.setdefault(plot.group, []).append(plot.plot_id)


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
        help="Plot groups to render (features, behavior, tfr, erp, machine_learning)",
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
    defaults_overrides.add_argument("--text-residual-qc-title-y", type=float, default=None, help="Residual QC title y position (default from config)")

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

    overrides.add_argument("--quality-width-per-plot", type=float, default=None, help="Quality plot width per subplot (default from config)")
    overrides.add_argument("--quality-height-per-plot", type=float, default=None, help="Quality plot height per subplot (default from config)")
    overrides.add_argument("--quality-distribution-n-cols", type=int, default=None, help="Quality distribution grid columns (default from config)")
    overrides.add_argument("--quality-distribution-max-features", type=int, default=None, help="Max features for quality distribution grid (default from config)")
    overrides.add_argument("--quality-outlier-z-threshold", type=float, default=None, help="Outlier robust z-threshold (default from config)")
    overrides.add_argument("--quality-outlier-max-features", type=int, default=None, help="Max features in outlier heatmap (default from config)")
    overrides.add_argument("--quality-outlier-max-trials", type=int, default=None, help="Max trials in outlier heatmap (default from config)")
    overrides.add_argument("--quality-snr-threshold-db", type=float, default=None, help="SNR threshold in dB (default from config)")

    overrides.add_argument("--complexity-width-per-measure", type=float, default=None, help="Complexity plot width per measure (default from config)")
    overrides.add_argument("--complexity-height-per-segment", type=float, default=None, help="Complexity plot height per segment (default from config)")

    overrides.add_argument("--connectivity-width-per-circle", type=float, default=None, help="Connectivity circle plot width (default from config)")
    overrides.add_argument("--connectivity-width-per-band", type=float, default=None, help="Connectivity plot width per band (default from config)")
    overrides.add_argument("--connectivity-height-per-measure", type=float, default=None, help="Connectivity plot height per measure (default from config)")
    overrides.add_argument("--connectivity-circle-top-fraction", type=float, default=None, help="Connectivity circle top fraction (default from config)")
    overrides.add_argument("--connectivity-circle-min-lines", type=int, default=None, help="Connectivity circle min lines (default from config)")

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
        metavar=("PLOT_ID", "KEY", "VALUE"),
        help=(
            "Per-plot overrides; can be repeated. "
            "Example: --plot-item-config tfr_scalpmean compare_windows true. "
            "Keys: compare_windows, comparison_windows, compare_columns, "
            "comparison_segment, comparison_column, comparison_values, comparison_labels, comparison_rois, "
            "topomap_windows (or topomap_window for single value), tfr_topomap_active_window, "
            "tfr_topomap_window_size_ms, tfr_topomap_window_count, tfr_topomap_label_x_position, tfr_topomap_label_y_position_bottom, "
            "tfr_topomap_label_y_position, tfr_topomap_title_y, tfr_topomap_title_pad, "
            "tfr_topomap_subplots_right, tfr_topomap_temporal_hspace, tfr_topomap_temporal_wspace, "
            "connectivity_circle_top_fraction, connectivity_circle_min_lines, "
            "connectivity_network_top_fraction, itpc_shared_colorbar."
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


def _resolve_plot_ids(args: argparse.Namespace) -> List[str]:
    selected: Set[str] = set()
    if args.plots:
        selected.update(args.plots)
    if args.groups:
        for group in args.groups:
            selected.update(PLOT_GROUPS.get(group, []))
    if args.all_plots or not selected:
        selected.update(PLOT_BY_ID.keys())
    return sorted(selected)


def _unique_in_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for item in items:
        if item in seen:
            continue
        result.append(item)
        seen.add(item)
    return result


def _apply_config_override(config: Any, path: str, value: Any) -> None:
    """Set a config value at the given dot-separated path."""
    config[path] = value


def _get_arg_value(args: argparse.Namespace, attr_name: str) -> Any:
    """Get argument value if present, otherwise None."""
    return getattr(args, attr_name, None)


def _parse_bool(value: str) -> Optional[bool]:
    value_lower = str(value).strip().lower()
    if value_lower in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value_lower in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None


def _parse_plot_item_configs(raw: Optional[List[List[str]]]) -> Dict[str, Dict[str, List[str]]]:
    configs: Dict[str, Dict[str, List[str]]] = {}
    if not raw:
        return configs
    for entry in raw:
        if not entry or len(entry) < 3:
            continue
        plot_id = entry[0]
        key = entry[1]
        values = entry[2:]
        configs.setdefault(plot_id, {})[key] = values
    return configs


def _apply_plot_item_overrides(config: Any, overrides: Dict[str, List[str]]) -> None:
    for key, values in overrides.items():
        if key == "compare_windows" and values:
            parsed = _parse_bool(values[0])
            if parsed is not None:
                _apply_config_override(config, "plotting.comparisons.compare_windows", parsed)
        elif key == "comparison_windows" and values:
            _apply_config_override(config, "plotting.comparisons.comparison_windows", list(values))
        elif key == "compare_columns" and values:
            parsed = _parse_bool(values[0])
            if parsed is not None:
                _apply_config_override(config, "plotting.comparisons.compare_columns", parsed)
        elif key == "comparison_segment" and values:
            _apply_config_override(config, "plotting.comparisons.comparison_segment", values[0])
        elif key == "comparison_column" and values:
            _apply_config_override(config, "plotting.comparisons.comparison_column", values[0])
        elif key == "comparison_values" and values:
            _apply_config_override(config, "plotting.comparisons.comparison_values", list(values))
        elif key == "comparison_labels" and len(values) >= 2:
            _apply_config_override(config, "plotting.comparisons.comparison_labels", [values[0], values[1]])
        elif key == "comparison_rois" and values:
            _apply_config_override(config, "plotting.comparisons.comparison_rois", list(values))
        elif key == "topomap_windows" and values:
            _apply_config_override(config, "plotting.plots.features.power.topomap_windows", list(values))
        elif key == "topomap_window" and values:
            _apply_config_override(config, "plotting.plots.features.power.topomap_windows", [values[0]])
        elif key == "connectivity_circle_top_fraction" and values:
            try:
                _apply_config_override(config, "plotting.plots.features.connectivity.circle_top_fraction", float(values[0]))
            except ValueError:
                pass
        elif key == "connectivity_circle_min_lines" and values:
            try:
                _apply_config_override(config, "plotting.plots.features.connectivity.circle_min_lines", int(values[0]))
            except ValueError:
                pass
        elif key == "connectivity_network_top_fraction" and values:
            try:
                _apply_config_override(config, "plotting.plots.features.connectivity.network_top_fraction", float(values[0]))
            except ValueError:
                pass
        elif key == "itpc_shared_colorbar" and values:
            parsed = _parse_bool(values[0])
            if parsed is not None:
                _apply_config_override(config, "plotting.plots.itpc.shared_colorbar", parsed)
        elif key == "tfr_topomap_active_window" and values:
            try:
                parts = values[0].split()
                if len(parts) == 2:
                    tmin = float(parts[0])
                    tmax = float(parts[1])
                    _apply_config_override(
                        config,
                        "time_frequency_analysis.active_window",
                        [tmin, tmax]
                    )
            except (ValueError, IndexError):
                pass
        elif key == "tfr_topomap_window_size_ms" and values:
            try:
                _apply_config_override(
                    config,
                    "time_frequency_analysis.topomap.temporal.window_size_ms",
                    float(values[0])
                )
            except ValueError:
                pass
        elif key == "tfr_topomap_window_count" and values:
            try:
                _apply_config_override(
                    config,
                    "time_frequency_analysis.topomap.temporal.window_count",
                    int(values[0])
                )
            except ValueError:
                pass
        elif key == "tfr_topomap_label_x_position" and values:
            try:
                _apply_config_override(
                    config,
                    "plotting.plots.tfr.topomap.label_x_position",
                    float(values[0])
                )
            except ValueError:
                pass
        elif key == "tfr_topomap_label_y_position_bottom" and values:
            try:
                _apply_config_override(
                    config,
                    "plotting.plots.tfr.topomap.label_y_position_bottom",
                    float(values[0])
                )
            except ValueError:
                pass
        elif key == "tfr_topomap_label_y_position" and values:
            try:
                _apply_config_override(
                    config,
                    "plotting.plots.tfr.topomap.label_y_position",
                    float(values[0])
                )
            except ValueError:
                pass
        elif key == "tfr_topomap_title_y" and values:
            try:
                _apply_config_override(
                    config,
                    "plotting.plots.tfr.topomap.title_y",
                    float(values[0])
                )
            except ValueError:
                pass
        elif key == "tfr_topomap_title_pad" and values:
            try:
                _apply_config_override(
                    config,
                    "plotting.plots.tfr.topomap.title_pad",
                    int(values[0])
                )
            except ValueError:
                pass
        elif key == "tfr_topomap_subplots_right" and values:
            try:
                _apply_config_override(
                    config,
                    "plotting.plots.tfr.topomap.subplots_right",
                    float(values[0])
                )
            except ValueError:
                pass
        elif key == "tfr_topomap_temporal_hspace" and values:
            try:
                _apply_config_override(
                    config,
                    "time_frequency_analysis.topomap.temporal.single_subject.hspace",
                    float(values[0])
                )
            except ValueError:
                pass
        elif key == "tfr_topomap_temporal_wspace" and values:
            try:
                _apply_config_override(
                    config,
                    "time_frequency_analysis.topomap.temporal.single_subject.wspace",
                    float(values[0])
                )
            except ValueError:
                pass


_PLOT_ITEM_CONFIG_KEYS: Dict[str, str] = {
    "compare_windows": "plotting.comparisons.compare_windows",
    "comparison_windows": "plotting.comparisons.comparison_windows",
    "compare_columns": "plotting.comparisons.compare_columns",
    "comparison_segment": "plotting.comparisons.comparison_segment",
    "comparison_column": "plotting.comparisons.comparison_column",
    "comparison_values": "plotting.comparisons.comparison_values",
    "comparison_labels": "plotting.comparisons.comparison_labels",
    "comparison_rois": "plotting.comparisons.comparison_rois",
    "topomap_windows": "plotting.plots.features.power.topomap_windows",
    "topomap_window": "plotting.plots.features.power.topomap_windows",
    "tfr_topomap_active_window": "time_frequency_analysis.active_window",
    "tfr_topomap_window_size_ms": "time_frequency_analysis.topomap.temporal.window_size_ms",
    "tfr_topomap_window_count": "time_frequency_analysis.topomap.temporal.window_count",
    "tfr_topomap_label_x_position": "plotting.plots.tfr.topomap.label_x_position",
    "tfr_topomap_label_y_position_bottom": "plotting.plots.tfr.topomap.label_y_position_bottom",
    "tfr_topomap_label_y_position": "plotting.plots.tfr.topomap.label_y_position",
    "tfr_topomap_title_y": "plotting.plots.tfr.topomap.title_y",
    "tfr_topomap_title_pad": "plotting.plots.tfr.topomap.title_pad",
    "tfr_topomap_subplots_right": "plotting.plots.tfr.topomap.subplots_right",
    "tfr_topomap_temporal_hspace": "time_frequency_analysis.topomap.temporal.single_subject.hspace",
    "tfr_topomap_temporal_wspace": "time_frequency_analysis.topomap.temporal.single_subject.wspace",
    "connectivity_circle_top_fraction": "plotting.plots.features.connectivity.circle_top_fraction",
    "connectivity_circle_min_lines": "plotting.plots.features.connectivity.circle_min_lines",
    "connectivity_network_top_fraction": "plotting.plots.features.connectivity.network_top_fraction",
    "itpc_shared_colorbar": "plotting.plots.itpc.shared_colorbar",
}


def _validate_plot_item_configs(configs: Dict[str, Dict[str, List[str]]]) -> None:
    errors: List[str] = []
    for plot_id, overrides in configs.items():
        if plot_id not in PLOT_BY_ID:
            errors.append(f"Unknown plot_id '{plot_id}'. See --plots choices or use --all-plots.")
            continue

        for key, values in overrides.items():
            if key not in _PLOT_ITEM_CONFIG_KEYS:
                allowed = ", ".join(sorted(_PLOT_ITEM_CONFIG_KEYS.keys()))
                errors.append(f"Unknown key '{key}' for plot_id '{plot_id}'. Allowed keys: {allowed}.")
                continue

            if key in {"compare_windows", "compare_columns"}:
                if not values:
                    errors.append(f"plot_id '{plot_id}': {key} expects a boolean value (true/false).")
                    continue
                parsed = _parse_bool(values[0])
                if parsed is None:
                    errors.append(
                        f"plot_id '{plot_id}': {key} expects true/false, got: {values[0]!r}."
                    )
                continue

            if key in {"comparison_segment", "comparison_column", "connectivity_circle_top_fraction", "connectivity_circle_min_lines", "connectivity_network_top_fraction"}:
                if not values or not str(values[0]).strip():
                    errors.append(f"plot_id '{plot_id}': {key} expects a non-empty value.")
                continue

            if key == "topomap_windows":
                if not values:
                    errors.append(f"plot_id '{plot_id}': {key} expects one or more values.")
                continue

            if key == "topomap_window":
                if not values or not str(values[0]).strip():
                    errors.append(f"plot_id '{plot_id}': {key} expects a non-empty value.")
                continue

            if key == "connectivity_circle_top_fraction":
                try:
                    val = float(values[0])
                    if not (0.0 <= val <= 1.0):
                        errors.append(f"plot_id '{plot_id}': {key} must be between 0.0 and 1.0.")
                except (ValueError, IndexError):
                    errors.append(f"plot_id '{plot_id}': {key} must be a number between 0.0 and 1.0.")
                continue

            if key == "connectivity_circle_min_lines":
                try:
                    val = int(values[0])
                    if val < 0:
                        errors.append(f"plot_id '{plot_id}': {key} must be a non-negative integer.")
                except (ValueError, IndexError):
                    errors.append(f"plot_id '{plot_id}': {key} must be a non-negative integer.")
                continue

            if key == "connectivity_network_top_fraction":
                try:
                    val = float(values[0])
                    if not (0.0 <= val <= 1.0):
                        errors.append(f"plot_id '{plot_id}': {key} must be between 0.0 and 1.0.")
                except (ValueError, IndexError):
                    errors.append(f"plot_id '{plot_id}': {key} must be a number between 0.0 and 1.0.")
                continue

            if key == "itpc_shared_colorbar":
                if not values:
                    errors.append(f"plot_id '{plot_id}': {key} expects a boolean value (true/false).")
                else:
                    val_str = str(values[0]).lower()
                    if val_str not in {"true", "false"}:
                        errors.append(f"plot_id '{plot_id}': {key} must be 'true' or 'false'.")
                continue

            if key in {"comparison_windows", "comparison_values", "comparison_rois"}:
                if not values:
                    errors.append(f"plot_id '{plot_id}': {key} expects one or more values.")

            if key == "comparison_labels":
                if len(values) != 2:
                    errors.append(f"plot_id '{plot_id}': {key} expects exactly 2 values (label1 label2).")

    if errors:
        joined = "\n  - ".join(errors)
        raise ValueError(f"Invalid --plot-item-config overrides:\n  - {joined}")


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
    if _get_arg_value(args, "quality_width_per_plot") is not None:
        _apply_config_override(config, "plotting.plots.quality.width_per_plot", float(args.quality_width_per_plot))
    if _get_arg_value(args, "quality_height_per_plot") is not None:
        _apply_config_override(config, "plotting.plots.quality.height_per_plot", float(args.quality_height_per_plot))
    if _get_arg_value(args, "quality_distribution_n_cols") is not None:
        _apply_config_override(config, "plotting.plots.features.quality.distribution.n_cols", int(args.quality_distribution_n_cols))
    if _get_arg_value(args, "quality_distribution_max_features") is not None:
        _apply_config_override(config, "plotting.plots.features.quality.distribution.max_features", int(args.quality_distribution_max_features))
    if _get_arg_value(args, "quality_outlier_z_threshold") is not None:
        _apply_config_override(config, "plotting.plots.features.quality.outlier.z_threshold", float(args.quality_outlier_z_threshold))
    if _get_arg_value(args, "quality_outlier_max_features") is not None:
        _apply_config_override(config, "plotting.plots.features.quality.outlier.max_features", int(args.quality_outlier_max_features))
    if _get_arg_value(args, "quality_outlier_max_trials") is not None:
        _apply_config_override(config, "plotting.plots.features.quality.outlier.max_trials", int(args.quality_outlier_max_trials))
    if _get_arg_value(args, "quality_snr_threshold_db") is not None:
        _apply_config_override(config, "plotting.plots.features.quality.snr.threshold_db", float(args.quality_snr_threshold_db))
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


def _parse_roi_definitions(roi_defs: List[str]) -> Dict[str, List[str]]:
    """Parse ROI definitions from CLI format 'name:ch1,ch2,...'.
    
    Returns a dict mapping ROI name to list of regex patterns for matching channels.
    """
    rois: Dict[str, List[str]] = {}
    for roi_def in roi_defs:
        if ":" not in roi_def:
            raise ValueError(f"Invalid ROI definition '{roi_def}'; expected 'name:ch1,ch2,...'")
        name, channels_str = roi_def.split(":", 1)
        name = name.strip()
        channels = [ch.strip() for ch in channels_str.split(",") if ch.strip()]
        if not channels:
            raise ValueError(f"Invalid ROI definition '{roi_def}'; no channels specified")
        # Convert channel list to regex patterns (matches exact channel names)
        rois[name] = [f"^({'|'.join(channels)})$"]
    return rois


def _apply_roi_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply custom ROI definitions to config for plotting.

    Sets ROIs in both locations used by different plotting subsystems:
    - Top-level 'rois': Used by get_roi_definitions() in feature plots
    - 'time_frequency_analysis.rois': Used by get_rois() in TFR plots
    """
    if _get_arg_value(args, "rois"):
        custom_rois = _parse_roi_definitions(args.rois)
        # Apply to top-level rois (used by get_roi_definitions in plotting/features)
        config["rois"] = custom_rois
        # Apply to TFR-specific rois (used by get_rois in TFR extraction)
        config["time_frequency_analysis.rois"] = custom_rois


def _parse_frequency_band_definitions(band_defs: List[str]) -> Dict[str, List[float]]:
    """Parse frequency band definitions from CLI format 'name:low:high'."""
    bands: Dict[str, List[float]] = {}
    for band_def in band_defs:
        parts = band_def.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid frequency band definition '{band_def}'; expected 'name:low:high'")
        name = parts[0].strip().lower()
        try:
            low = float(parts[1].strip())
            high = float(parts[2].strip())
        except ValueError:
            raise ValueError(f"Invalid frequency values in '{band_def}'; expected numeric low:high")
        if low >= high:
            raise ValueError(f"Invalid frequency range in '{band_def}'; low must be < high")
        bands[name] = [low, high]
    return bands


def _apply_band_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply custom frequency band definitions to config for plotting.

    If --frequency-bands is specified, uses only those definitions.
    If --bands is also specified, filters to only selected bands.
    
    Sets bands in both locations used by different plotting subsystems:
    - Top-level 'frequency_bands': Used by get_frequency_bands() in feature plots
    - 'time_frequency_analysis.bands': Used by TFR analysis
    """
    if _get_arg_value(args, "frequency_bands"):
        custom_bands = _parse_frequency_band_definitions(args.frequency_bands)
        
        selected_bands = getattr(args, "bands", None)
        if selected_bands:
            selected_lower = [b.lower() for b in selected_bands]
            filtered_bands = {k: v for k, v in custom_bands.items() if k.lower() in selected_lower}
            custom_bands = filtered_bands
        
        # Apply to top-level frequency_bands (used by get_frequency_bands)
        config["frequency_bands"] = custom_bands
        # Apply to TFR-specific bands (used by TFR analysis)
        config["time_frequency_analysis.bands"] = custom_bands


def _apply_all_config_overrides(args: argparse.Namespace, config: Any) -> None:
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
    _apply_comparison_overrides(args, config)
    _apply_roi_overrides(args, config)
    _apply_band_overrides(args, config)
    _apply_output_overrides(args, config)


def _map_plot_id_to_plotters(plot_id: str, feature_categories: List[str]) -> Optional[List[str]]:
    """Map plot ID to specific plotter names for feature categories.
    
    When a specific plot is selected, only run the relevant plotters instead of all
    plotters for the category.
    
    Parameters
    ----------
    plot_id : str
        Plot ID from catalog (e.g., "power_by_condition")
    feature_categories : list of str
        Feature categories this plot belongs to
        
    Returns
    -------
    list of str or None
        List of plotter tokens in "category.name" format, or None to run all plotters
    """
    # Mapping from plot IDs to plotter function names
    # These MUST match the exact registered plotter function names in registrations.py
    PLOT_ID_TO_PLOTTER = {
        # Power plots -> 4 registered: plot_tfr_visualization, plot_psd_visualization, 
        #                              plot_power_condition_comparison, plot_power_summary
        "power_by_condition": "plot_power_condition_comparison",
        "power_spectral_density": "plot_psd_visualization",
        "band_power_topomaps": "plot_power_summary",
        "cross_frequency_power_correlation": "plot_power_condition_comparison",
        
        # Connectivity plots -> 2 registered: plot_connectivity_mne_suite, plot_connectivity_condition
        "connectivity_by_condition": "plot_connectivity_condition",
        "connectivity_circle_condition": "plot_connectivity_mne_suite",
        "connectivity_heatmap": "plot_connectivity_mne_suite",
        "connectivity_network": "plot_connectivity_mne_suite",
        
        # Aperiodic plots -> 1 registered: aperiodic_suite
        "aperiodic_topomaps": "aperiodic_suite",
        "aperiodic_by_condition": "aperiodic_suite",
        "aperiodic_temporal_evolution": "aperiodic_suite",
        
        # ITPC plots -> 1 registered: itpc_suite
        "itpc_heatmap": "itpc_suite",
        "itpc_topomaps": "itpc_suite",
        "itpc_by_condition": "itpc_suite",
        "itpc_temporal_evolution": "itpc_suite",
        
        # PAC plots -> 2 registered: pac_summary, pac_suite
        "pac_summary": "pac_summary",
        "pac_comodulograms": "pac_suite",
        "pac_by_condition": "pac_suite",
        "pac_time_ribbons": "pac_suite",
        
        # ERDS plots -> 1 registered: plot_erds
        "erds_temporal_evolution": "plot_erds",
        "erds_latency_distribution": "plot_erds",
        "erds_erd_ers_separation": "plot_erds",
        "erds_global_summary": "plot_erds",
        "erds_by_condition": "plot_erds",
        
        # Complexity plots -> 1 registered: plot_complexity
        "complexity_by_band": "plot_complexity",
        "complexity_by_condition": "plot_complexity",
        "complexity_temporal_evolution": "plot_complexity",
        
        # Spectral plots -> 1 registered: plot_spectral
        "spectral_summary": "plot_spectral",
        "spectral_edge_frequency": "plot_spectral",
        "spectral_by_condition": "plot_spectral",
        "spectral_temporal_evolution": "plot_spectral",
        
        # Ratios plots -> 1 registered: plot_ratios
        "ratios_by_pair": "plot_ratios",
        "ratios_by_condition": "plot_ratios",
        "ratios_temporal_evolution": "plot_ratios",
        
        # Asymmetry plots -> 1 registered: plot_asymmetry
        "asymmetry_by_band": "plot_asymmetry",
        "asymmetry_by_condition": "plot_asymmetry",
        "asymmetry_temporal_evolution": "plot_asymmetry",
        
        # Bursts plots -> 1 registered: plot_bursts
        "bursts_by_band": "plot_bursts",
        "bursts_by_condition": "plot_bursts",
        "burst_temporal_evolution": "plot_bursts",
        
        # Quality plots -> 1 registered: quality_suite
        "quality_feature_distributions": "quality_suite",
        "quality_outlier_heatmap": "quality_suite",
        "quality_snr_distribution": "quality_suite",
        
        # ERP plots -> 1 registered: erp_suite
        "erp_butterfly": "erp_suite",
        "erp_roi": "erp_suite",
        "erp_contrast": "erp_suite",
        "erp_topomaps": "erp_suite",
        
        # Temporal plots -> 1 registered: plot_temporal
        "temporal_evolution": "plot_temporal",
    }
    
    plotter_name = PLOT_ID_TO_PLOTTER.get(plot_id)
    if plotter_name is None:
        return None  # Run all plotters for the category
    
    # Convert to "category.name" format
    result = []
    for category in feature_categories:
        result.append(f"{category}.{plotter_name}")
    
    return result if result else None


def _collect_plot_definitions(
    plot_ids: List[str],
) -> tuple[Set[str], Set[str], List[str], List[str], List[str], List[str], Set[str]]:
    """Collect plot definitions and extract categories, plots, and modes."""
    feature_categories: Set[str] = set()
    feature_plot_patterns: Set[str] = set()
    feature_plotters: Set[str] = set()
    behavior_plots: List[str] = []
    tfr_plots: List[str] = []
    erp_plots: List[Any] = []
    ml_modes: Set[str] = set()

    for plot_id in plot_ids:
        definition = PLOT_BY_ID.get(plot_id)
        if definition is None:
            continue
        if definition.feature_categories:
            feature_categories.update(definition.feature_categories)
            if definition.feature_plot_patterns:
                feature_plot_patterns.update(str(p) for p in definition.feature_plot_patterns)
            else:
                feature_plot_patterns.add(plot_id)
            # Map plot ID to specific plotter function names
            plotter_names = _map_plot_id_to_plotters(plot_id, list(definition.feature_categories))
            if plotter_names:
                feature_plotters.update(plotter_names)
        if definition.behavior_plots:
            behavior_plots.extend(definition.behavior_plots)
        if definition.tfr_plots:
            tfr_plots.extend(definition.tfr_plots)
        if definition.erp_plots:
            erp_plots.append(definition.erp_plots)
        if definition.ml_mode:
            ml_modes.add(definition.ml_mode)

    flat_erp_plots = []
    for plot in erp_plots:
        if isinstance(plot, list):
            flat_erp_plots.extend(plot)
        else:
            flat_erp_plots.append(plot)

    return (
        feature_categories,
        feature_plot_patterns,
        _unique_in_order(behavior_plots),
        _unique_in_order(tfr_plots),
        _unique_in_order(flat_erp_plots),
        sorted(feature_plotters) if feature_plotters else [],
        ml_modes,
    )


def _render_plots_with_per_plot_config(
    plot_ids: List[str],
    plot_item_configs: Dict[str, Dict[str, List[str]]],
    subjects: List[str],
    task: str,
    config: Any,
    selected_feature_plotters: Optional[List[str]],
    progress: Any,
) -> None:
    """Render plots with per-plot configuration overrides."""
    from eeg_pipeline.plotting.orchestration.features import visualize_features_for_subjects
    from eeg_pipeline.plotting.orchestration.behavior import visualize_behavior_for_subjects
    from eeg_pipeline.plotting.orchestration.tfr import visualize_tfr_for_subjects
    from eeg_pipeline.plotting.orchestration.erp import visualize_erp_for_subjects
    from eeg_pipeline.plotting.orchestration.machine_learning import (
        visualize_regression_from_disk,
        visualize_time_generalization_from_disk,
        visualize_classification_from_disk,
    )

    total = len(plot_ids)
    for idx, plot_id in enumerate(plot_ids, start=1):
        definition = PLOT_BY_ID.get(plot_id)
        if definition is None:
            continue

        plot_config = ConfigDict(copy.deepcopy(dict(config)))
        overrides = plot_item_configs.get(plot_id, {})
        if overrides:
            _apply_plot_item_overrides(plot_config, overrides)

        progress.step(f"Rendering {definition.label or plot_id}", current=idx, total=total)

        if definition.feature_categories:
            patterns = (
                list(definition.feature_plot_patterns)
                if definition.feature_plot_patterns
                else [plot_id]
            )
            
            # Map plot ID to plotter names if feature_plotters not explicitly provided
            plotter_names = selected_feature_plotters
            if plotter_names is None:
                plotter_names = _map_plot_id_to_plotters(plot_id, definition.feature_categories)
            
            visualize_features_for_subjects(
                subjects=subjects,
                task=task,
                config=plot_config,
                visualize_categories=sorted(definition.feature_categories),
                feature_plotters=plotter_names,
                plot_name_patterns=patterns,
            )
        if definition.behavior_plots:
            visualize_behavior_for_subjects(
                subjects=subjects,
                task=task,
                config=plot_config,
                plots=_unique_in_order(list(definition.behavior_plots)),
            )
        if definition.tfr_plots:
            visualize_tfr_for_subjects(
                subjects=subjects,
                task=task,
                config=plot_config,
                plots=_unique_in_order(list(definition.tfr_plots)),
            )
        if definition.erp_plots:
            plots_list = definition.erp_plots if isinstance(definition.erp_plots, list) else [definition.erp_plots]
            visualize_erp_for_subjects(
                subjects=subjects,
                task=task,
                config=plot_config,
                plots=_unique_in_order(list(plots_list)),
            )
        if definition.ml_mode:
            deriv_root = get_deriv_root(plot_config)
            if definition.ml_mode == "regression":
                results_dir = deriv_root / "machine_learning" / "regression"
                visualize_regression_from_disk(results_dir=results_dir, config=plot_config)
            elif definition.ml_mode == "timegen":
                results_dir = deriv_root / "machine_learning" / "time_generalization"
                visualize_time_generalization_from_disk(results_dir=results_dir, config=plot_config)
            elif definition.ml_mode == "classify":
                results_dir = deriv_root / "machine_learning" / "classification"
                visualize_classification_from_disk(results_dir=results_dir, config=plot_config)


def _render_plots_without_per_plot_config(
    feature_categories: Set[str],
    feature_plot_patterns: Set[str],
    behavior_plots: List[str],
    tfr_plots: List[str],
    erp_plots: List[str],
    ml_modes: Set[str],
    subjects: List[str],
    task: str,
    config: Any,
    selected_feature_plotters: Optional[List[str]],
    progress: Any,
) -> None:
    """Render plots without per-plot configuration overrides."""
    from eeg_pipeline.plotting.orchestration.features import visualize_features_for_subjects
    from eeg_pipeline.plotting.orchestration.behavior import visualize_behavior_for_subjects
    from eeg_pipeline.plotting.orchestration.tfr import visualize_tfr_for_subjects
    from eeg_pipeline.plotting.orchestration.erp import visualize_erp_for_subjects
    from eeg_pipeline.plotting.orchestration.machine_learning import (
        visualize_regression_from_disk,
        visualize_time_generalization_from_disk,
        visualize_classification_from_disk,
    )

    steps = sum([
        bool(feature_categories),
        bool(behavior_plots),
        bool(tfr_plots),
        bool(erp_plots),
        bool(ml_modes),
    ])
    step_idx = 0

    if feature_categories:
        step_idx += 1
        progress.step("Rendering feature plots", current=step_idx, total=steps)
        visualize_features_for_subjects(
            subjects=subjects,
            task=task,
            config=config,
            visualize_categories=sorted(feature_categories),
            feature_plotters=selected_feature_plotters,
            plot_name_patterns=sorted(feature_plot_patterns) if feature_plot_patterns else None,
        )

    if behavior_plots:
        step_idx += 1
        progress.step("Rendering behavior plots", current=step_idx, total=steps)
        visualize_behavior_for_subjects(
            subjects=subjects,
            task=task,
            config=config,
            plots=behavior_plots,
        )

    if tfr_plots:
        step_idx += 1
        progress.step("Rendering TFR plots", current=step_idx, total=steps)
        visualize_tfr_for_subjects(
            subjects=subjects,
            task=task,
            config=config,
            plots=tfr_plots,
        )

    if erp_plots:
        step_idx += 1
        progress.step("Rendering ERP plots", current=step_idx, total=steps)
        visualize_erp_for_subjects(
            subjects=subjects,
            task=task,
            config=config,
            plots=erp_plots,
        )

    if ml_modes:
        step_idx += 1
        progress.step("Rendering machine learning plots", current=step_idx, total=steps)
        deriv_root = get_deriv_root(config)
        if "regression" in ml_modes:
            results_dir = deriv_root / "machine_learning" / "regression"
            visualize_regression_from_disk(results_dir=results_dir, config=config)
        if "timegen" in ml_modes:
            results_dir = deriv_root / "machine_learning" / "time_generalization"
            visualize_time_generalization_from_disk(results_dir=results_dir, config=config)
        if "classify" in ml_modes:
            results_dir = deriv_root / "machine_learning" / "classification"
            visualize_classification_from_disk(results_dir=results_dir, config=config)


def run_plotting(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the plotting command."""
    if args.mode == "tfr":
        return _run_tfr_mode(args, subjects, config)

    if getattr(args, "bids_root", None):
        config.setdefault("paths", {})["bids_root"] = args.bids_root
    if getattr(args, "deriv_root", None):
        config.setdefault("paths", {})["deriv_root"] = args.deriv_root

    plot_ids = _resolve_plot_ids(args)
    if not plot_ids:
        raise ValueError("No plots selected")

    plot_item_configs = _parse_plot_item_configs(getattr(args, "plot_item_config", None))
    if plot_item_configs:
        _validate_plot_item_configs(plot_item_configs)

    selected_feature_plotters = _get_arg_value(args, "feature_plotters")
    _apply_all_config_overrides(args, config)

    task = resolve_task(args.task, config)
    progress = create_progress_reporter(args)

    if plot_item_configs:
        progress.start("plotting", subjects)
        _render_plots_with_per_plot_config(
            plot_ids=plot_ids,
            plot_item_configs=plot_item_configs,
            subjects=subjects,
            task=task,
            config=config,
            selected_feature_plotters=selected_feature_plotters,
            progress=progress,
        )
        progress.complete(success=True)
        return

    (
        feature_categories,
        feature_plot_patterns,
        behavior_plots,
        tfr_plots,
        erp_plots,
        computed_feature_plotters,
        ml_modes,
    ) = _collect_plot_definitions(plot_ids)

    if not any([feature_categories, behavior_plots, tfr_plots, erp_plots, ml_modes]):
        raise ValueError("No plots resolved from selection")

    # Use computed plotters if available, otherwise fall back to CLI argument
    effective_feature_plotters = (
        computed_feature_plotters if computed_feature_plotters 
        else selected_feature_plotters
    )

    progress.start("plotting", subjects)
    _render_plots_without_per_plot_config(
        feature_categories=feature_categories,
        feature_plot_patterns=feature_plot_patterns,
        behavior_plots=behavior_plots,
        tfr_plots=tfr_plots,
        erp_plots=erp_plots,
        ml_modes=ml_modes,
        subjects=subjects,
        task=task,
        config=config,
        selected_feature_plotters=effective_feature_plotters,
        progress=progress,
    )
    progress.complete(success=True)


def _update_tfr_config(
    config: Any,
    bands: Optional[List[str]] = None,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
) -> None:
    """Update config with TFR analysis parameters."""
    tfr_section = config.setdefault("time_frequency_analysis", {})
    if bands is not None:
        tfr_section["selected_bands"] = bands
    if tmin is not None:
        tfr_section["tmin"] = tmin
    if tmax is not None:
        tfr_section["tmax"] = tmax


def _validate_time_range(tmin: Optional[float], tmax: Optional[float]) -> None:
    """Validate that time range is logically consistent."""
    if tmin is not None and tmax is not None and tmin >= tmax:
        raise ValueError(f"tmin ({tmin}) must be less than tmax ({tmax})")


def _run_tfr_mode(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute TFR visualization mode."""
    from eeg_pipeline.plotting.orchestration.tfr import visualize_tfr_for_subjects

    _validate_time_range(args.tmin, args.tmax)
    _update_tfr_config(config, args.bands, args.tmin, args.tmax)

    # Apply all config overrides (including formats, dpi, etc.)
    _apply_all_config_overrides(args, config)

    # Apply per-plot config overrides for TFR mode
    plot_item_configs = _parse_plot_item_configs(args.plot_item_config)
    if plot_item_configs:
        _validate_plot_item_configs(plot_item_configs)
        # Apply all overrides from matched plot IDs to the config
        for plot_id, overrides in plot_item_configs.items():
            # Only apply if this is a TFR-related plot or if --plots matches
            definition = PLOT_BY_ID.get(plot_id)
            if definition and definition.tfr_plots:
                _apply_plot_item_overrides(config, overrides)

    progress = create_progress_reporter(args)
    progress.start("tfr_visualize", subjects)
    progress.step("Rendering TFR plots", current=1, total=2)

    # Extract TFR-specific plots from --plots argument if provided
    tfr_plots = None
    if args.plots:
        _, _, _, tfr_plots, _, _, _ = _collect_plot_definitions(args.plots)
        tfr_plots = tfr_plots if tfr_plots else None

    visualize_tfr_for_subjects(
        subjects=subjects,
        task=args.task,
        tfr_roi_only=args.tfr_roi,
        tfr_topomaps_only=args.tfr_topomaps_only,
        plots=tfr_plots,
        n_jobs=args.n_jobs,
        config=config,
    )

    progress.step("Finalizing", current=2, total=2)
    progress.complete(success=True)


__all__ = [
    "setup_plotting",
    "run_plotting",
]
