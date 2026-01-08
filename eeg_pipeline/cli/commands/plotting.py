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


@dataclass(frozen=True)
class PlotDefinition:
    plot_id: str
    group: str
    label: str
    description: str
    required_files: List[str]
    feature_categories: Optional[List[str]] = None
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
    parser.add_argument("mode", choices=["visualize"], help="Pipeline mode (visualize)")
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
    # Backwards-compatible aliases (ITPC-specific naming).
    parser.add_argument("--itpc-shared-colorbar", action="store_true", default=None, dest="shared_colorbar", help=argparse.SUPPRESS)
    parser.add_argument("--no-itpc-shared-colorbar", action="store_false", dest="shared_colorbar", help=argparse.SUPPRESS)

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
        "--layout-tight-rect-microstate",
        nargs=4,
        type=float,
        default=None,
        metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"),
        help="tight_layout rect for microstate plots (default from config)",
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
    # Backwards-compatible aliases (suppressed from help).
    defaults_overrides.add_argument("--color-nonpain", type=str, default=None, dest="color_condition_1", help=argparse.SUPPRESS)
    defaults_overrides.add_argument("--color-pain", type=str, default=None, dest="color_condition_2", help=argparse.SUPPRESS)
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

    # TFR misc (plotting.tfr.*)
    defaults_overrides.add_argument(
        "--tfr-default-baseline-window",
        nargs=2,
        type=float,
        default=None,
        metavar=("TMIN", "TMAX"),
        help="Default baseline window used by TF plotting when missing (default from config)",
    )

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
            "Example: --plot-item-config tfr_scalpmean tfr_default_baseline_window -5.0 -0.01. "
            "Keys: tfr_default_baseline_window, compare_windows, comparison_windows, compare_columns, "
            "comparison_segment, comparison_column, comparison_values, comparison_labels, comparison_rois."
        ),
    )

    add_path_args(parser)
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
        if key == "tfr_default_baseline_window":
            if len(values) == 2:
                try:
                    config["plotting.tfr.default_baseline_window"] = [float(values[0]), float(values[1])]
                except ValueError:
                    pass
            continue

        if key == "compare_windows":
            if values:
                parsed = _parse_bool(values[0])
                if parsed is not None:
                    config["plotting.comparisons.compare_windows"] = parsed
            continue
        if key == "comparison_windows":
            if values:
                config["plotting.comparisons.comparison_windows"] = list(values)
            continue

        if key == "compare_columns":
            if values:
                parsed = _parse_bool(values[0])
                if parsed is not None:
                    config["plotting.comparisons.compare_columns"] = parsed
            continue
        if key == "comparison_segment":
            if values:
                config["plotting.comparisons.comparison_segment"] = values[0]
            continue
        if key == "comparison_column":
            if values:
                config["plotting.comparisons.comparison_column"] = values[0]
            continue
        if key == "comparison_values":
            if values:
                config["plotting.comparisons.comparison_values"] = list(values)
            continue
        if key == "comparison_labels":
            if len(values) >= 2:
                config["plotting.comparisons.comparison_labels"] = [values[0], values[1]]
            continue
        if key == "comparison_rois":
            if values:
                config["plotting.comparisons.comparison_rois"] = list(values)
            continue


_PLOT_ITEM_CONFIG_KEYS: Dict[str, str] = {
    "tfr_default_baseline_window": "plotting.tfr.default_baseline_window",
    "compare_windows": "plotting.comparisons.compare_windows",
    "comparison_windows": "plotting.comparisons.comparison_windows",
    "compare_columns": "plotting.comparisons.compare_columns",
    "comparison_segment": "plotting.comparisons.comparison_segment",
    "comparison_column": "plotting.comparisons.comparison_column",
    "comparison_values": "plotting.comparisons.comparison_values",
    "comparison_labels": "plotting.comparisons.comparison_labels",
    "comparison_rois": "plotting.comparisons.comparison_rois",
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

            if key == "tfr_default_baseline_window":
                if len(values) != 2:
                    errors.append(
                        f"plot_id '{plot_id}': {key} expects 2 values (tmin tmax), got {len(values)}."
                    )
                    continue
                try:
                    float(values[0])
                    float(values[1])
                except ValueError:
                    errors.append(
                        f"plot_id '{plot_id}': {key} values must be numeric (tmin tmax), got: {values!r}."
                    )
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

            if key in {"comparison_segment", "comparison_column"}:
                if not values or not str(values[0]).strip():
                    errors.append(f"plot_id '{plot_id}': {key} expects a non-empty value.")
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


def run_plotting(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the plotting command."""
    from eeg_pipeline.plotting.orchestration.features import visualize_features_for_subjects
    from eeg_pipeline.plotting.orchestration.behavior import visualize_behavior_for_subjects
    from eeg_pipeline.plotting.orchestration.tfr import visualize_tfr_for_subjects
    from eeg_pipeline.plotting.orchestration.erp import visualize_erp_for_subjects
    from eeg_pipeline.plotting.orchestration.machine_learning import (
        visualize_regression_from_disk,
        visualize_time_generalization_from_disk,
        visualize_classification_from_disk,
    )

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

    selected_feature_plotters = getattr(args, "feature_plotters", None)

    if args.formats:
        config["plotting.defaults.formats"] = list(args.formats)
    if args.dpi is not None:
        config["plotting.defaults.dpi"] = int(args.dpi)
    if args.savefig_dpi is not None:
        config["plotting.defaults.savefig_dpi"] = int(args.savefig_dpi)
    if getattr(args, "bbox_inches", None):
        config["plotting.defaults.bbox_inches"] = str(args.bbox_inches)
    if getattr(args, "pad_inches", None) is not None:
        config["plotting.defaults.pad_inches"] = float(args.pad_inches)

    if getattr(args, "font_family", None):
        config["plotting.defaults.font.family"] = str(args.font_family)
    if getattr(args, "font_weight", None):
        config["plotting.defaults.font.weight"] = str(args.font_weight)
    if getattr(args, "font_size_small", None) is not None:
        config["plotting.defaults.font.sizes.small"] = int(args.font_size_small)
    if getattr(args, "font_size_medium", None) is not None:
        config["plotting.defaults.font.sizes.medium"] = int(args.font_size_medium)
    if getattr(args, "font_size_large", None) is not None:
        config["plotting.defaults.font.sizes.large"] = int(args.font_size_large)
    if getattr(args, "font_size_title", None) is not None:
        config["plotting.defaults.font.sizes.title"] = int(args.font_size_title)
    if getattr(args, "font_size_annotation", None) is not None:
        config["plotting.defaults.font.sizes.annotation"] = int(args.font_size_annotation)
    if getattr(args, "font_size_label", None) is not None:
        config["plotting.defaults.font.sizes.label"] = int(args.font_size_label)
    if getattr(args, "font_size_ylabel", None) is not None:
        config["plotting.defaults.font.sizes.ylabel"] = int(args.font_size_ylabel)
    if getattr(args, "font_size_suptitle", None) is not None:
        config["plotting.defaults.font.sizes.suptitle"] = int(args.font_size_suptitle)
    if getattr(args, "font_size_figure_title", None) is not None:
        config["plotting.defaults.font.sizes.figure_title"] = int(args.font_size_figure_title)

    if getattr(args, "layout_tight_rect", None):
        config["plotting.defaults.layout.tight_rect"] = list(args.layout_tight_rect)
    if getattr(args, "layout_tight_rect_microstate", None):
        config["plotting.defaults.layout.tight_rect_microstate"] = list(args.layout_tight_rect_microstate)
    if getattr(args, "gridspec_width_ratios", None):
        config["plotting.defaults.layout.gridspec.width_ratios"] = list(args.gridspec_width_ratios)
    if getattr(args, "gridspec_height_ratios", None):
        config["plotting.defaults.layout.gridspec.height_ratios"] = list(args.gridspec_height_ratios)
    if getattr(args, "gridspec_hspace", None) is not None:
        config["plotting.defaults.layout.gridspec.hspace"] = float(args.gridspec_hspace)
    if getattr(args, "gridspec_wspace", None) is not None:
        config["plotting.defaults.layout.gridspec.wspace"] = float(args.gridspec_wspace)
    if getattr(args, "gridspec_left", None) is not None:
        config["plotting.defaults.layout.gridspec.left"] = float(args.gridspec_left)
    if getattr(args, "gridspec_right", None) is not None:
        config["plotting.defaults.layout.gridspec.right"] = float(args.gridspec_right)
    if getattr(args, "gridspec_top", None) is not None:
        config["plotting.defaults.layout.gridspec.top"] = float(args.gridspec_top)
    if getattr(args, "gridspec_bottom", None) is not None:
        config["plotting.defaults.layout.gridspec.bottom"] = float(args.gridspec_bottom)

    if getattr(args, "figure_size_standard", None):
        config["plotting.figure_sizes.standard"] = list(args.figure_size_standard)
    if getattr(args, "figure_size_medium", None):
        config["plotting.figure_sizes.medium"] = list(args.figure_size_medium)
    if getattr(args, "figure_size_small", None):
        config["plotting.figure_sizes.small"] = list(args.figure_size_small)
    if getattr(args, "figure_size_square", None):
        config["plotting.figure_sizes.square"] = list(args.figure_size_square)
    if getattr(args, "figure_size_wide", None):
        config["plotting.figure_sizes.wide"] = list(args.figure_size_wide)
    if getattr(args, "figure_size_tfr", None):
        config["plotting.figure_sizes.tfr"] = list(args.figure_size_tfr)
    if getattr(args, "figure_size_topomap", None):
        config["plotting.figure_sizes.topomap"] = list(args.figure_size_topomap)

    if getattr(args, "color_condition_2", None):
        config["plotting.styling.colors.condition_2"] = str(args.color_condition_2)
    if getattr(args, "color_condition_1", None):
        config["plotting.styling.colors.condition_1"] = str(args.color_condition_1)
    if getattr(args, "color_significant", None):
        config["plotting.styling.colors.significant"] = str(args.color_significant)
    if getattr(args, "color_nonsignificant", None):
        config["plotting.styling.colors.nonsignificant"] = str(args.color_nonsignificant)
    if getattr(args, "color_gray", None):
        config["plotting.styling.colors.gray"] = str(args.color_gray)
    if getattr(args, "color_light_gray", None):
        config["plotting.styling.colors.light_gray"] = str(args.color_light_gray)
    if getattr(args, "color_black", None):
        config["plotting.styling.colors.black"] = str(args.color_black)
    if getattr(args, "color_blue", None):
        config["plotting.styling.colors.blue"] = str(args.color_blue)
    if getattr(args, "color_red", None):
        config["plotting.styling.colors.red"] = str(args.color_red)
    if getattr(args, "color_network_node", None):
        config["plotting.styling.colors.network_node"] = str(args.color_network_node)

    if getattr(args, "alpha_grid", None) is not None:
        config["plotting.styling.alpha.grid"] = float(args.alpha_grid)
    if getattr(args, "alpha_fill", None) is not None:
        config["plotting.styling.alpha.fill"] = float(args.alpha_fill)
    if getattr(args, "alpha_ci", None) is not None:
        config["plotting.styling.alpha.ci"] = float(args.alpha_ci)
    if getattr(args, "alpha_ci_line", None) is not None:
        config["plotting.styling.alpha.ci_line"] = float(args.alpha_ci_line)
    if getattr(args, "alpha_text_box", None) is not None:
        config["plotting.styling.alpha.text_box"] = float(args.alpha_text_box)
    if getattr(args, "alpha_violin_body", None) is not None:
        config["plotting.styling.alpha.violin_body"] = float(args.alpha_violin_body)
    if getattr(args, "alpha_ridge_fill", None) is not None:
        config["plotting.styling.alpha.ridge_fill"] = float(args.alpha_ridge_fill)

    if getattr(args, "scatter_marker_size_small", None) is not None:
        config["plotting.styling.scatter.marker_size.small"] = int(args.scatter_marker_size_small)
    if getattr(args, "scatter_marker_size_large", None) is not None:
        config["plotting.styling.scatter.marker_size.large"] = int(args.scatter_marker_size_large)
    if getattr(args, "scatter_marker_size_default", None) is not None:
        config["plotting.styling.scatter.marker_size.default"] = int(args.scatter_marker_size_default)
    if getattr(args, "scatter_alpha", None) is not None:
        config["plotting.styling.scatter.alpha"] = float(args.scatter_alpha)
    if getattr(args, "scatter_edgecolor", None):
        config["plotting.styling.scatter.edgecolor"] = str(args.scatter_edgecolor)
    if getattr(args, "scatter_edgewidth", None) is not None:
        config["plotting.styling.scatter.edgewidth"] = float(args.scatter_edgewidth)

    if getattr(args, "bar_alpha", None) is not None:
        config["plotting.styling.bar.alpha"] = float(args.bar_alpha)
    if getattr(args, "bar_width", None) is not None:
        config["plotting.styling.bar.width"] = float(args.bar_width)
    if getattr(args, "bar_capsize", None) is not None:
        config["plotting.styling.bar.capsize"] = int(args.bar_capsize)
    if getattr(args, "bar_capsize_large", None) is not None:
        config["plotting.styling.bar.capsize_large"] = int(args.bar_capsize_large)

    if getattr(args, "line_width_thin", None) is not None:
        config["plotting.styling.line.width.thin"] = float(args.line_width_thin)
    if getattr(args, "line_width_standard", None) is not None:
        config["plotting.styling.line.width.standard"] = float(args.line_width_standard)
    if getattr(args, "line_width_thick", None) is not None:
        config["plotting.styling.line.width.thick"] = float(args.line_width_thick)
    if getattr(args, "line_width_bold", None) is not None:
        config["plotting.styling.line.width.bold"] = float(args.line_width_bold)
    if getattr(args, "line_alpha_standard", None) is not None:
        config["plotting.styling.line.alpha.standard"] = float(args.line_alpha_standard)
    if getattr(args, "line_alpha_dim", None) is not None:
        config["plotting.styling.line.alpha.dim"] = float(args.line_alpha_dim)
    if getattr(args, "line_alpha_zero_line", None) is not None:
        config["plotting.styling.line.alpha.zero_line"] = float(args.line_alpha_zero_line)
    if getattr(args, "line_alpha_fit_line", None) is not None:
        config["plotting.styling.line.alpha.fit_line"] = float(args.line_alpha_fit_line)
    if getattr(args, "line_alpha_diagonal", None) is not None:
        config["plotting.styling.line.alpha.diagonal"] = float(args.line_alpha_diagonal)
    if getattr(args, "line_alpha_reference", None) is not None:
        config["plotting.styling.line.alpha.reference"] = float(args.line_alpha_reference)
    if getattr(args, "line_regression_width", None) is not None:
        config["plotting.styling.line.regression_width"] = float(args.line_regression_width)
    if getattr(args, "line_residual_width", None) is not None:
        config["plotting.styling.line.residual_width"] = float(args.line_residual_width)
    if getattr(args, "line_qq_width", None) is not None:
        config["plotting.styling.line.qq_width"] = float(args.line_qq_width)

    if getattr(args, "hist_bins", None) is not None:
        config["plotting.styling.histogram.bins"] = int(args.hist_bins)
    if getattr(args, "hist_bins_behavioral", None) is not None:
        config["plotting.styling.histogram.bins_behavioral"] = int(args.hist_bins_behavioral)
    if getattr(args, "hist_bins_residual", None) is not None:
        config["plotting.styling.histogram.bins_residual"] = int(args.hist_bins_residual)
    if getattr(args, "hist_bins_tfr", None) is not None:
        config["plotting.styling.histogram.bins_tfr"] = int(args.hist_bins_tfr)
    if getattr(args, "hist_edgecolor", None):
        config["plotting.styling.histogram.edgecolor"] = str(args.hist_edgecolor)
    if getattr(args, "hist_edgewidth", None) is not None:
        config["plotting.styling.histogram.edgewidth"] = float(args.hist_edgewidth)
    if getattr(args, "hist_alpha", None) is not None:
        config["plotting.styling.histogram.alpha"] = float(args.hist_alpha)
    if getattr(args, "hist_alpha_residual", None) is not None:
        config["plotting.styling.histogram.alpha_residual"] = float(args.hist_alpha_residual)
    if getattr(args, "hist_alpha_tfr", None) is not None:
        config["plotting.styling.histogram.alpha_tfr"] = float(args.hist_alpha_tfr)

    if getattr(args, "kde_points", None) is not None:
        config["plotting.styling.kde.points"] = int(args.kde_points)
    if getattr(args, "kde_color", None):
        config["plotting.styling.kde.color"] = str(args.kde_color)
    if getattr(args, "kde_linewidth", None) is not None:
        config["plotting.styling.kde.linewidth"] = float(args.kde_linewidth)
    if getattr(args, "kde_alpha", None) is not None:
        config["plotting.styling.kde.alpha"] = float(args.kde_alpha)

    if getattr(args, "errorbar_markersize", None) is not None:
        config["plotting.styling.errorbar.markersize"] = int(args.errorbar_markersize)
    if getattr(args, "errorbar_capsize", None) is not None:
        config["plotting.styling.errorbar.capsize"] = int(args.errorbar_capsize)
    if getattr(args, "errorbar_capsize_large", None) is not None:
        config["plotting.styling.errorbar.capsize_large"] = int(args.errorbar_capsize_large)

    if getattr(args, "text_stats_x", None) is not None:
        config["plotting.styling.text_position.stats_x"] = float(args.text_stats_x)
    if getattr(args, "text_stats_y", None) is not None:
        config["plotting.styling.text_position.stats_y"] = float(args.text_stats_y)
    if getattr(args, "text_pvalue_x", None) is not None:
        config["plotting.styling.text_position.p_value_x"] = float(args.text_pvalue_x)
    if getattr(args, "text_pvalue_y", None) is not None:
        config["plotting.styling.text_position.p_value_y"] = float(args.text_pvalue_y)
    if getattr(args, "text_bootstrap_x", None) is not None:
        config["plotting.styling.text_position.bootstrap_x"] = float(args.text_bootstrap_x)
    if getattr(args, "text_bootstrap_y", None) is not None:
        config["plotting.styling.text_position.bootstrap_y"] = float(args.text_bootstrap_y)
    if getattr(args, "text_channel_annotation_x", None) is not None:
        config["plotting.styling.text_position.channel_annotation_x"] = float(args.text_channel_annotation_x)
    if getattr(args, "text_channel_annotation_y", None) is not None:
        config["plotting.styling.text_position.channel_annotation_y"] = float(args.text_channel_annotation_y)
    if getattr(args, "text_title_y", None) is not None:
        config["plotting.styling.text_position.title_y"] = float(args.text_title_y)
    if getattr(args, "text_residual_qc_title_y", None) is not None:
        config["plotting.styling.text_position.residual_qc_title_y"] = float(args.text_residual_qc_title_y)

    if getattr(args, "validation_min_bins_for_calibration", None) is not None:
        config["plotting.validation.min_bins_for_calibration"] = int(args.validation_min_bins_for_calibration)
    if getattr(args, "validation_max_bins_for_calibration", None) is not None:
        config["plotting.validation.max_bins_for_calibration"] = int(args.validation_max_bins_for_calibration)
    if getattr(args, "validation_samples_per_bin", None) is not None:
        config["plotting.validation.samples_per_bin"] = int(args.validation_samples_per_bin)
    if getattr(args, "validation_min_rois_for_fdr", None) is not None:
        config["plotting.validation.min_rois_for_fdr"] = int(args.validation_min_rois_for_fdr)
    if getattr(args, "validation_min_pvalues_for_fdr", None) is not None:
        config["plotting.validation.min_pvalues_for_fdr"] = int(args.validation_min_pvalues_for_fdr)

    if getattr(args, "tfr_default_baseline_window", None):
        config["plotting.tfr.default_baseline_window"] = list(args.tfr_default_baseline_window)
    if getattr(args, "shared_colorbar", None) is not None:
        config.setdefault("plotting", {}).setdefault("plots", {}).setdefault("itpc", {})["shared_colorbar"] = args.shared_colorbar

    # Topomap overrides
    if getattr(args, "topomap_contours", None) is not None:
        config["plotting.plots.topomap.contours"] = int(args.topomap_contours)
    if getattr(args, "topomap_colormap", None):
        config["plotting.plots.topomap.colormap"] = str(args.topomap_colormap)
    if getattr(args, "topomap_colorbar_fraction", None) is not None:
        config["plotting.plots.topomap.colorbar_fraction"] = float(args.topomap_colorbar_fraction)
    if getattr(args, "topomap_colorbar_pad", None) is not None:
        config["plotting.plots.topomap.colorbar_pad"] = float(args.topomap_colorbar_pad)
    if getattr(args, "topomap_diff_annotation_enabled", None) is not None:
        config["plotting.plots.topomap.diff_annotation_enabled"] = bool(args.topomap_diff_annotation_enabled)
    if getattr(args, "topomap_annotate_descriptive", None) is not None:
        config["plotting.plots.topomap.annotate_descriptive"] = bool(args.topomap_annotate_descriptive)
    if getattr(args, "topomap_sig_mask_marker", None):
        config["plotting.plots.topomap.sig_mask_params.marker"] = str(args.topomap_sig_mask_marker)
    if getattr(args, "topomap_sig_mask_markerfacecolor", None):
        config["plotting.plots.topomap.sig_mask_params.markerfacecolor"] = str(args.topomap_sig_mask_markerfacecolor)
    if getattr(args, "topomap_sig_mask_markeredgecolor", None):
        config["plotting.plots.topomap.sig_mask_params.markeredgecolor"] = str(args.topomap_sig_mask_markeredgecolor)
    if getattr(args, "topomap_sig_mask_linewidth", None) is not None:
        config["plotting.plots.topomap.sig_mask_params.linewidth"] = float(args.topomap_sig_mask_linewidth)
    if getattr(args, "topomap_sig_mask_markersize", None) is not None:
        config["plotting.plots.topomap.sig_mask_params.markersize"] = float(args.topomap_sig_mask_markersize)

    # TFR overrides
    if getattr(args, "tfr_log_base", None) is not None:
        config["plotting.plots.tfr.log_base"] = float(args.tfr_log_base)
    if getattr(args, "tfr_percentage_multiplier", None) is not None:
        config["plotting.plots.tfr.percentage_multiplier"] = float(args.tfr_percentage_multiplier)

    # Plot type sizing overrides
    if getattr(args, "roi_width_per_band", None) is not None:
        config["plotting.plots.roi.width_per_band"] = float(args.roi_width_per_band)
    if getattr(args, "roi_width_per_metric", None) is not None:
        config["plotting.plots.roi.width_per_metric"] = float(args.roi_width_per_metric)
    if getattr(args, "roi_height_per_roi", None) is not None:
        config["plotting.plots.roi.height_per_roi"] = float(args.roi_height_per_roi)

    if getattr(args, "power_width_per_band", None) is not None:
        config["plotting.plots.power.width_per_band"] = float(args.power_width_per_band)
    if getattr(args, "power_height_per_segment", None) is not None:
        config["plotting.plots.power.height_per_segment"] = float(args.power_height_per_segment)

    if getattr(args, "itpc_width_per_bin", None) is not None:
        config["plotting.plots.itpc.width_per_bin"] = float(args.itpc_width_per_bin)
    if getattr(args, "itpc_height_per_band", None) is not None:
        config["plotting.plots.itpc.height_per_band"] = float(args.itpc_height_per_band)
    if getattr(args, "itpc_width_per_band_box", None) is not None:
        config["plotting.plots.itpc.width_per_band_box"] = float(args.itpc_width_per_band_box)
    if getattr(args, "itpc_height_box", None) is not None:
        config["plotting.plots.itpc.height_box"] = float(args.itpc_height_box)

    if getattr(args, "pac_cmap", None):
        config["plotting.plots.pac.cmap"] = str(args.pac_cmap)
    if getattr(args, "pac_width_per_roi", None) is not None:
        config["plotting.plots.pac.width_per_roi"] = float(args.pac_width_per_roi)
    if getattr(args, "pac_height_box", None) is not None:
        config["plotting.plots.pac.height_box"] = float(args.pac_height_box)

    if getattr(args, "aperiodic_width_per_column", None) is not None:
        config["plotting.plots.aperiodic.width_per_column"] = float(args.aperiodic_width_per_column)
    if getattr(args, "aperiodic_height_per_row", None) is not None:
        config["plotting.plots.aperiodic.height_per_row"] = float(args.aperiodic_height_per_row)
    if getattr(args, "aperiodic_n_perm", None) is not None:
        config["plotting.plots.aperiodic.n_perm"] = int(args.aperiodic_n_perm)

    if getattr(args, "quality_width_per_plot", None) is not None:
        config["plotting.plots.quality.width_per_plot"] = float(args.quality_width_per_plot)
    if getattr(args, "quality_height_per_plot", None) is not None:
        config["plotting.plots.quality.height_per_plot"] = float(args.quality_height_per_plot)
    if getattr(args, "quality_distribution_n_cols", None) is not None:
        config["plotting.plots.features.quality.distribution.n_cols"] = int(args.quality_distribution_n_cols)
    if getattr(args, "quality_distribution_max_features", None) is not None:
        config["plotting.plots.features.quality.distribution.max_features"] = int(args.quality_distribution_max_features)
    if getattr(args, "quality_outlier_z_threshold", None) is not None:
        config["plotting.plots.features.quality.outlier.z_threshold"] = float(args.quality_outlier_z_threshold)
    if getattr(args, "quality_outlier_max_features", None) is not None:
        config["plotting.plots.features.quality.outlier.max_features"] = int(args.quality_outlier_max_features)
    if getattr(args, "quality_outlier_max_trials", None) is not None:
        config["plotting.plots.features.quality.outlier.max_trials"] = int(args.quality_outlier_max_trials)
    if getattr(args, "quality_snr_threshold_db", None) is not None:
        config["plotting.plots.features.quality.snr.threshold_db"] = float(args.quality_snr_threshold_db)

    if getattr(args, "complexity_width_per_measure", None) is not None:
        config["plotting.plots.complexity.width_per_measure"] = float(args.complexity_width_per_measure)
    if getattr(args, "complexity_height_per_segment", None) is not None:
        config["plotting.plots.complexity.height_per_segment"] = float(args.complexity_height_per_segment)

    if getattr(args, "connectivity_width_per_circle", None) is not None:
        config["plotting.plots.connectivity.width_per_circle"] = float(args.connectivity_width_per_circle)
    if getattr(args, "connectivity_width_per_band", None) is not None:
        config["plotting.plots.connectivity.width_per_band"] = float(args.connectivity_width_per_band)
    if getattr(args, "connectivity_height_per_measure", None) is not None:
        config["plotting.plots.connectivity.height_per_measure"] = float(args.connectivity_height_per_measure)
    if getattr(args, "connectivity_circle_top_fraction", None) is not None:
        config["plotting.plots.features.connectivity.circle_top_fraction"] = float(args.connectivity_circle_top_fraction)
    if getattr(args, "connectivity_circle_min_lines", None) is not None:
        config["plotting.plots.features.connectivity.circle_min_lines"] = int(args.connectivity_circle_min_lines)

    # Feature plot ordering / selection overrides
    if getattr(args, "pac_pairs", None):
        config["plotting.plots.features.pac_pairs"] = list(args.pac_pairs)
    if getattr(args, "connectivity_measures", None):
        config["plotting.plots.features.connectivity.measures"] = list(args.connectivity_measures)
    if getattr(args, "spectral_metrics", None):
        config["plotting.plots.features.spectral.metrics"] = list(args.spectral_metrics)
    if getattr(args, "bursts_metrics", None):
        config["plotting.plots.features.bursts.metrics"] = list(args.bursts_metrics)
    if getattr(args, "asymmetry_stat", None):
        config["plotting.plots.features.asymmetry.stat"] = str(args.asymmetry_stat)
    if getattr(args, "temporal_time_bins", None):
        config["plotting.plots.features.temporal.time_bins"] = list(args.temporal_time_bins)
    if getattr(args, "temporal_time_labels", None):
        config["plotting.plots.features.temporal.time_labels"] = list(args.temporal_time_labels)

    # Comparison overrides
    if getattr(args, "compare_windows", None) is not None:
        config["plotting.comparisons.compare_windows"] = bool(args.compare_windows)
    if getattr(args, "comparison_windows", None):
        config["plotting.comparisons.comparison_windows"] = list(args.comparison_windows)
    if getattr(args, "compare_columns", None) is not None:
        config["plotting.comparisons.compare_columns"] = bool(args.compare_columns)
    if getattr(args, "comparison_segment", None):
        config["plotting.comparisons.comparison_segment"] = str(args.comparison_segment)
    if getattr(args, "comparison_column", None):
        config["plotting.comparisons.comparison_column"] = str(args.comparison_column)
    if getattr(args, "comparison_values", None):
        config["plotting.comparisons.comparison_values"] = list(args.comparison_values)
    if getattr(args, "comparison_labels", None):
        config["plotting.comparisons.comparison_labels"] = list(args.comparison_labels)
    if getattr(args, "comparison_rois", None):
        config["plotting.comparisons.comparison_rois"] = list(args.comparison_rois)

    task = resolve_task(args.task, config)
    progress = create_progress_reporter(args)

    # If per-plot overrides are present, dispatch each selected plot ID with its
    # own config copy so overrides don't leak across plot types.
    if plot_item_configs:
        progress.start("plotting", subjects)
        total = len(plot_ids)
        for idx, plot_id in enumerate(plot_ids, start=1):
            definition = PLOT_BY_ID.get(plot_id)
            if definition is None:
                continue

            plot_config = copy.deepcopy(config)
            overrides = plot_item_configs.get(plot_id, {})
            if overrides:
                _apply_plot_item_overrides(plot_config, overrides)

            progress.step(f"Rendering {definition.label or plot_id}", current=idx, total=total)

            if definition.feature_categories:
                visualize_features_for_subjects(
                    subjects=subjects,
                    task=task,
                    config=plot_config,
                    visualize_categories=sorted(definition.feature_categories),
                    feature_plotters=selected_feature_plotters,
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
                if definition.ml_mode == "timegen":
                    results_dir = deriv_root / "machine_learning" / "time_generalization"
                    visualize_time_generalization_from_disk(results_dir=results_dir, config=plot_config)
                if definition.ml_mode == "classify":
                    results_dir = deriv_root / "machine_learning" / "classification"
                    visualize_classification_from_disk(results_dir=results_dir, config=plot_config)

        progress.complete(success=True)
        return


    feature_categories: Set[str] = set()
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
        if definition.behavior_plots:
            behavior_plots.extend(definition.behavior_plots)
        if definition.tfr_plots:
            tfr_plots.extend(definition.tfr_plots)
        if definition.erp_plots:
            erp_plots.append(definition.erp_plots) # erp_plots is just used as a list of strings
        if definition.ml_mode:
            ml_modes.add(definition.ml_mode)

    # Flatten and unique erp_plots
    flat_erp_plots = []
    for p in erp_plots:
        if isinstance(p, list):
            flat_erp_plots.extend(p)
        else:
            flat_erp_plots.append(p)
    erp_plots = _unique_in_order(flat_erp_plots)

    behavior_plots = _unique_in_order(behavior_plots)
    tfr_plots = _unique_in_order(tfr_plots)

    steps = 0
    if feature_categories:
        steps += 1
    if behavior_plots:
        steps += 1
    if tfr_plots:
        steps += 1
    if erp_plots:
        steps += 1
    if ml_modes:
        steps += 1

    if steps == 0:
        raise ValueError("No plots resolved from selection")

    progress.start("plotting", subjects)
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

    progress.complete(success=True)


__all__ = [
    "setup_plotting",
    "run_plotting",
]
