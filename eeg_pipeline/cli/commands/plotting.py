"""Plotting orchestration CLI command."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_task_arg,
    add_output_format_args,
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
    decoding_mode: Optional[str] = None


PLOT_CATALOG: List[PlotDefinition] = [
    # Features
    PlotDefinition(
        "features_power",
        "features",
        "Power",
        "Band power summaries and topomaps",
        ["features_power.tsv"],
        feature_categories=["power"],
    ),
    PlotDefinition(
        "features_connectivity",
        "features",
        "Connectivity",
        "Connectivity heatmaps and networks",
        ["features_connectivity.parquet"],
        feature_categories=["connectivity"],
    ),
    PlotDefinition(
        "features_aperiodic",
        "features",
        "Aperiodic",
        "1/f spectral slope diagnostics",
        ["features_aperiodic.tsv"],
        feature_categories=["aperiodic"],
    ),
    PlotDefinition(
        "features_itpc",
        "features",
        "ITPC",
        "Inter-trial phase coherence plots",
        ["features_itpc.tsv"],
        feature_categories=["itpc"],
    ),
    PlotDefinition(
        "features_pac",
        "features",
        "PAC",
        "Phase-amplitude coupling plots",
        ["features_pac_trials.tsv"],
        feature_categories=["pac"],
    ),
    PlotDefinition(
        "features_erds",
        "features",
        "ERDS",
        "Event-related desync/sync plots",
        ["features_erds.tsv"],
        feature_categories=["erds"],
    ),
    # Behavior
    PlotDefinition(
        "behavior_psychometrics",
        "behavior",
        "Psychometrics",
        "Rating distributions and psychometrics",
        ["events.tsv", "features_power.tsv"],
        behavior_plots=["psychometrics"],
    ),
    PlotDefinition(
        "behavior_power_scatter",
        "behavior",
        "Power ROI Scatter",
        "Power vs behavior scatter plots",
        ["features_power.tsv", "stats/corr_stats_power_*"],
        behavior_plots=["power_roi_scatter"],
    ),
    PlotDefinition(
        "behavior_complexity_scatter",
        "behavior",
        "Complexity Scatter",
        "Complexity vs behavior scatter plots",
        ["features_complexity.tsv"],
        behavior_plots=["complexity_scatter"],
    ),
    PlotDefinition(
        "behavior_aperiodic_scatter",
        "behavior",
        "Aperiodic Scatter",
        "Aperiodic vs behavior scatter plots",
        ["features_aperiodic.tsv"],
        behavior_plots=["aperiodic_scatter"],
    ),
    PlotDefinition(
        "behavior_connectivity_scatter",
        "behavior",
        "Connectivity Scatter",
        "Connectivity vs behavior scatter plots",
        ["features_connectivity.parquet"],
        behavior_plots=["connectivity_scatter"],
    ),
    PlotDefinition(
        "behavior_itpc_scatter",
        "behavior",
        "ITPC Scatter",
        "ITPC vs behavior scatter plots",
        ["features_itpc.tsv"],
        behavior_plots=["itpc_scatter"],
    ),
    PlotDefinition(
        "behavior_temporal_topomaps",
        "behavior",
        "Temporal Topomaps",
        "Temporal correlation topomaps",
        ["stats/temporal_*"],
        behavior_plots=["temporal_topomaps"],
    ),
    PlotDefinition(
        "behavior_pain_clusters",
        "behavior",
        "Pain Clusters",
        "Cluster-based temporal contrasts",
        ["stats/cluster_*"],
        behavior_plots=["pain_clusters"],
    ),
    PlotDefinition(
        "behavior_dose_response",
        "behavior",
        "Dose Response",
        "Dose-response curves and contrasts",
        ["events.tsv", "features_power.tsv"],
        behavior_plots=["dose_response"],
    ),
    PlotDefinition(
        "behavior_mediation",
        "behavior",
        "Mediation",
        "Mediation path diagrams",
        ["stats/mediation.tsv"],
        behavior_plots=["mediation"],
    ),
    PlotDefinition(
        "behavior_top_predictors",
        "behavior",
        "Top Predictors",
        "Top predictors summary",
        ["stats/corr_stats_*"],
        behavior_plots=["top_predictors"],
    ),
    # TFR
    PlotDefinition(
        "tfr_scalpmean",
        "tfr",
        "Scalp-Mean",
        "Scalp-mean TFR plots",
        ["epochs/*.fif"],
        tfr_plots=["scalpmean"],
    ),
    PlotDefinition(
        "tfr_scalpmean_contrast",
        "tfr",
        "Scalp-Mean Contrast",
        "Pain vs non-pain contrasts",
        ["epochs/*.fif", "events.tsv"],
        tfr_plots=["scalpmean_contrast"],
    ),
    PlotDefinition(
        "tfr_channels",
        "tfr",
        "Channels",
        "Channel-level TFR plots",
        ["epochs/*.fif"],
        tfr_plots=["channels"],
    ),
    PlotDefinition(
        "tfr_channels_contrast",
        "tfr",
        "Channels Contrast",
        "Channel-level contrast plots",
        ["epochs/*.fif", "events.tsv"],
        tfr_plots=["channels_contrast"],
    ),
    PlotDefinition(
        "tfr_rois",
        "tfr",
        "ROIs",
        "ROI-level TFR plots",
        ["epochs/*.fif"],
        tfr_plots=["rois"],
    ),
    PlotDefinition(
        "tfr_rois_contrast",
        "tfr",
        "ROI Contrast",
        "ROI-level contrast plots",
        ["epochs/*.fif", "events.tsv"],
        tfr_plots=["rois_contrast"],
    ),
    PlotDefinition(
        "tfr_topomaps",
        "tfr",
        "Topomaps",
        "Time-frequency topomaps",
        ["epochs/*.fif", "events.tsv"],
        tfr_plots=["topomaps"],
    ),
    PlotDefinition(
        "tfr_band_evolution",
        "tfr",
        "Band Evolution",
        "Band evolution over time",
        ["epochs/*.fif"],
        tfr_plots=["band_evolution"],
    ),
    # Decoding
    PlotDefinition(
        "decoding_regression_plots",
        "decoding",
        "Regression Plots",
        "LOSO regression diagnostics",
        ["decoding/regression/loso_predictions.tsv"],
        decoding_mode="regression",
    ),
    PlotDefinition(
        "decoding_timegen_plots",
        "decoding",
        "Time-Generalization",
        "Time-generalization matrices",
        ["decoding/time_generalization/time_generalization_regression.npz"],
        decoding_mode="timegen",
    ),
    # ERP
    PlotDefinition(
        "erp_butterfly",
        "erp",
        "Butterfly",
        "Butterfly ERP plots (all channels)",
        ["epochs/*.fif"],
        erp_plots=["butterfly"],
    ),
    PlotDefinition(
        "erp_roi",
        "erp",
        "ROI Waveforms",
        "ROI-based ERP waveforms with error bars",
        ["epochs/*.fif"],
        erp_plots=["roi"],
    ),
    PlotDefinition(
        "erp_contrast",
        "erp",
        "Contrast",
        "ERP condition contrasts (Pain vs No-Pain)",
        ["epochs/*.fif", "events.tsv"],
        erp_plots=["contrast"],
    ),
    PlotDefinition(
        "erp_topomaps",
        "erp",
        "Topomaps",
        "ERP spatial distributions",
        ["epochs/*.fif"],
        erp_plots=["topomaps"],
    ),
]

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
        help="Plot groups to render (features, behavior, tfr, decoding)",
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


def run_plotting(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the plotting command."""
    from eeg_pipeline.plotting.orchestration.features import visualize_features_for_subjects
    from eeg_pipeline.plotting.orchestration.behavior import visualize_behavior_for_subjects
    from eeg_pipeline.plotting.orchestration.tfr import visualize_tfr_for_subjects
    from eeg_pipeline.plotting.orchestration.erp import visualize_erp_for_subjects
    from eeg_pipeline.plotting.orchestration.decoding import (
        visualize_regression_from_disk,
        visualize_time_generalization_from_disk,
    )

    plot_ids = _resolve_plot_ids(args)
    if not plot_ids:
        raise ValueError("No plots selected")

    if args.formats:
        config["plotting.defaults.formats"] = list(args.formats)
    if args.dpi is not None:
        config["plotting.defaults.dpi"] = int(args.dpi)
    if args.savefig_dpi is not None:
        config["plotting.defaults.savefig_dpi"] = int(args.savefig_dpi)
    if getattr(args, "shared_colorbar", None) is not None:
        config.setdefault("plotting", {}).setdefault("plots", {}).setdefault("itpc", {})["shared_colorbar"] = args.shared_colorbar


    feature_categories: Set[str] = set()
    behavior_plots: List[str] = []
    tfr_plots: List[str] = []
    erp_plots: List[Any] = []
    decoding_modes: Set[str] = set()

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
        if definition.decoding_mode:
            decoding_modes.add(definition.decoding_mode)

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

    task = resolve_task(args.task, config)
    progress = create_progress_reporter(args)

    steps = 0
    if feature_categories:
        steps += 1
    if behavior_plots:
        steps += 1
    if tfr_plots:
        steps += 1
    if erp_plots:
        steps += 1
    if decoding_modes:
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

    if decoding_modes:
        step_idx += 1
        progress.step("Rendering decoding plots", current=step_idx, total=steps)
        deriv_root = get_deriv_root(config)
        if "regression" in decoding_modes:
            results_dir = deriv_root / "decoding" / "regression"
            visualize_regression_from_disk(results_dir=results_dir, config=config)
        if "timegen" in decoding_modes:
            results_dir = deriv_root / "decoding" / "time_generalization"
            visualize_time_generalization_from_disk(results_dir=results_dir, config=config)

    progress.complete(success=True)


__all__ = [
    "setup_plotting",
    "run_plotting",
]
