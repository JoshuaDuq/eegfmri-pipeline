"""
Feature Visualization Pipeline - Descriptive Analysis
======================================================

Creates publication-quality visualizations DESCRIBING extracted features.
NO correlations with pain ratings - those belong in behavioral pipeline.

Each plot answers ONE specific question about feature characteristics.

DIRECTORY STRUCTURE:
====================
plots/features/
├── power/           - Spectral power characteristics
├── microstates/     - Brain state dynamics  
├── connectivity/    - Functional connectivity patterns
├── pac/             - Phase-amplitude coupling
├── complexity/      - Nonlinear dynamics (PE, Hjorth, LZC)
├── erds/            - Event-related desynchronization/synchronization
├── dynamics/        - Burst detection and GFP dynamics
├── itpc/            - Inter-trial phase coherence
└── summary/         - Data quality and overview
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Callable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import mne
from mne.viz import plot_topomap

from eeg_pipeline.utils.io.paths import ensure_dir, deriv_plots_path, deriv_features_path, deriv_stats_path
from eeg_pipeline.utils.io.tsv import read_tsv as _read_tsv
from eeg_pipeline.utils.io.plotting import save_fig as _save_fig, setup_matplotlib
from eeg_pipeline.utils.io.logging import get_logger
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.utils.analysis.features.metadata import NamingSchema
from eeg_pipeline.plotting.features.context import FeaturePlotContext, VisualizationRegistry, VisualizationManager
from eeg_pipeline.plotting.features.report import generate_feature_report

from eeg_pipeline.plotting.features.burst import (
    plot_burst_summary_by_band,
    plot_dynamics_by_condition,
)


from eeg_pipeline.plotting.features.connectivity import (
    plot_connectivity_by_condition,
)
from eeg_pipeline.plotting.features.microstates import (
    plot_microstate_by_condition,
    plot_microstate_templates,
    plot_microstate_transition_matrix,
)
from eeg_pipeline.plotting.features.power import (
    plot_channel_power_heatmap,
    plot_power_by_condition,
    plot_power_variability_comprehensive,
    plot_cross_frequency_power_correlation,
    plot_feature_stability_heatmap,
    plot_temporal_autocorrelation,
    plot_feature_redundancy_matrix,
    plot_band_power_topomaps,
    plot_spectral_slope_topomap,
    plot_power_trial_variability,
    plot_power_topomaps_from_df,
)
from eeg_pipeline.plotting.features.erds import (
    plot_erds_temporal_evolution,
)
from eeg_pipeline.plotting.features.dynamics import (
    plot_autocorrelation_decay,
)
from eeg_pipeline.plotting.features.phase import (
    plot_pac_summary,
    plot_pac_by_condition,
)
from eeg_pipeline.plotting.features.roi_condition import (
    plot_power_by_roi_band_condition,
    plot_dynamics_by_roi_band_condition,
    plot_connectivity_by_roi_band_condition,
    plot_aperiodic_by_roi_condition,
    plot_itpc_by_roi_band_condition,
    plot_itpc_plateau_vs_baseline,
    plot_pac_by_roi_condition,
    plot_band_segment_condition,
    plot_power_plateau_vs_baseline,
    plot_temporal_evolution,
    plot_feature_correlation_heatmap,
)

# MNE-based utilities being imported for registered wrappers
from eeg_pipeline.utils.analysis.tfr import compute_tfr_for_visualization
from eeg_pipeline.utils.analysis.windowing import sliding_window_centers
from eeg_pipeline.utils.config.loader import get_frequency_band_names, get_config_value
from eeg_pipeline.utils.data.loading import load_epochs_for_analysis, load_feature_bundle_for_subject

# MNE-based plotters
from eeg_pipeline.plotting.features.connectivity import (
    plot_connectivity_circle_for_band,
    plot_sliding_connectivity_trajectories,
    plot_connectivity_heatmap,
    plot_connectivity_network,
)


# =============================================================================
# Main Entry Point
# =============================================================================

def visualize_features(
    subject: str,
    deriv_root: Path,
    config: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    epochs_info: Optional[mne.Info] = None,
    aligned_events: Optional[pd.DataFrame] = None,
    epochs: Optional[mne.Epochs] = None,
    tfr: Optional[mne.time_frequency.EpochsTFR] = None,
) -> Dict[str, Path]:
    """Generate descriptive feature visualizations using registered plotters."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    features_dir = deriv_features_path(deriv_root, subject)
    plots_dir = deriv_plots_path(deriv_root, subject, subdir="features")
    ensure_dir(plots_dir)
    
    ctx = FeaturePlotContext(
        subject=subject,
        plots_dir=plots_dir,
        features_dir=features_dir,
        config=config,
        logger=logger,
        epochs_info=epochs_info,
        aligned_events=aligned_events,
        epochs=epochs,
        tfr=tfr,
    )
    ctx.load_data()
    
    if ctx.power_df is None and ctx.connectivity_df is None:
        logger.warning(f"No feature data found for subject {subject}")
        # Even if dataframes are missing, we might have MNE objects to plot something?
        # But usually features rely on extracted TSVs.
        # Check if we should return empty or try anyway.
        # Let's verify MNE objects availability at least
        if ctx.epochs is None:
             logger.warning("No epochs data found either.")
             return {}
    
    manager = VisualizationManager(ctx)
    saved_plots = manager.run_all()
    
    # Generate plot manifest
    _save_plot_manifest(saved_plots, plots_dir, subject, logger)
    
    return saved_plots


def _save_plot_manifest(
    saved_plots: Dict[str, Path],
    plots_dir: Path,
    subject: str,
    logger: logging.Logger,
) -> None:
    """Save a JSON manifest of all generated plots.
    
    Scans the plots directory to find all generated image files.
    The manifest includes plot paths and metadata.
    """
    import json
    from datetime import datetime
    
    # Scan directory for all image files (more reliable than tracking return values)
    plot_files = []
    for ext in ["png", "svg", "pdf"]:
        plot_files.extend(plots_dir.rglob(f"*.{ext}"))
    
    manifest = {
        "subject": subject,
        "generated_at": datetime.now().isoformat(),
        "plots_directory": str(plots_dir),
        "total_plots": len(plot_files),
        "plots": []
    }
    
    for path in sorted(plot_files):
        # Derive feature type from subdirectory
        path_str = str(path)
        feature_type = "unknown"
        for ftype in ["power", "connectivity", "microstates", "pac", "itpc", 
                      "complexity", "burst", "aperiodic", "dynamics", "summary"]:
            if f"/{ftype}/" in path_str or f"\\{ftype}\\" in path_str:
                feature_type = ftype
                break
        
        manifest["plots"].append({
            "name": path.stem,
            "path": str(path.relative_to(plots_dir)),
            "feature_type": feature_type,
            "format": path.suffix[1:],  # Remove leading dot
        })
    
    manifest_path = plots_dir / f"sub-{subject}_plot_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Saved plot manifest ({len(plot_files)} plots)")


# =============================================================================
# Helper Functions
# =============================================================================

def _safe_plot(
    ctx: FeaturePlotContext,
    saved_files: Dict[str, Path],
    name: str,
    subdir: str,
    filename: Optional[str],
    plot_func: Callable[..., Any],
    *args, 
    **kwargs
) -> None:
    """Helper to safely execute a plotting function and handle errors/logging."""
    try:
        if filename:
            path = ctx.subdir(subdir) / filename
             # If function takes path as first arg
            plot_func(path, *args, **kwargs)
            saved_files[name] = path
            ctx.logger.info(f"Created: {name}")
        else:
             # Function handles saving or returns path(s)
             # If it returns path, store it.
             # We assume plot_func uses ctx info if filename not provided, or args contain dir
             res = plot_func(*args, **kwargs)
             if res: # If it returns path or dict
                  if isinstance(res, (str, Path)):
                       saved_files[name] = Path(res)
                  elif isinstance(res, dict):
                       saved_files.update(res)
             ctx.logger.info(f"Executed: {name}")

    except Exception as e:
        ctx.logger.warning(f"Failed to create {name}: {e}")
        # ctx.logger.debug(f"Traceback: ", exc_info=True)


# =============================================================================
# MNE-BASED PLOTTERS (Migrated from viz.py)
# =============================================================================

@VisualizationRegistry.register("power")
def _plot_tfr_visualization(ctx: FeaturePlotContext, saved_files: Dict[str, Path]) -> None:
    """Compute and plot TFR."""
    if ctx.epochs is None:
        return
        
    _safe_plot(
        ctx, saved_files, "TFR", "power",
        None,
        compute_tfr_for_visualization, ctx.epochs, ctx.config, ctx.logger
    )
    
    # Power Spectral Density (requires TFR)
    try:
        from eeg_pipeline.plotting.features.power import plot_power_spectral_density
        from eeg_pipeline.utils.analysis.tfr import compute_tfr
        
        tfr = compute_tfr(ctx.epochs, ctx.config, ctx.logger)
        if tfr is not None:
            _safe_plot(
                ctx, saved_files, "power_spectral_density", "power",
                None,
                plot_power_spectral_density,
                tfr, ctx.subject, ctx.subdir("power"), ctx.logger, ctx.aligned_events, ctx.config
            )
    except Exception as e:
        ctx.logger.debug(f"Could not generate PSD plot: {e}")

@VisualizationRegistry.register("microstates")
def _plot_microstate_templates_mne(ctx: FeaturePlotContext, saved_files: Dict[str, Path]) -> None:
    """Plot microstate templates."""
    if ctx.epochs is None or ctx.microstate_df is None:
        return

    n_microstates = int(ctx.config.get("feature_engineering.microstates.n_states", 4))
    # Try to find stats/templates file
    stats_dir = deriv_stats_path(ctx.config.deriv_root, ctx.subject)
    template_path = stats_dir / f"microstates_templates_K{n_microstates}.npz"
    
    if not template_path.exists():
        ctx.logger.debug("No microstate templates file found.")
        return

    try:
        data = np.load(template_path)
        ms_templates = data['templates']
        picks = mne.pick_types(ctx.epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
        info_eeg = mne.pick_info(ctx.epochs.info, picks)
        
        def _do(save_dir):
            plot_microstate_templates(
                ms_templates, info_eeg, ctx.subject, save_dir, n_microstates, ctx.logger, ctx.config
            )
            
        _safe_plot(
            ctx, saved_files, "microstate_templates", "microstates",
            None,
            _do, ctx.subdir("microstates")
        )
    except Exception as e:
        ctx.logger.error(f"Error loading/plotting microstates: {e}")

@VisualizationRegistry.register("connectivity")
def _plot_connectivity_mne_suite(ctx: FeaturePlotContext, saved_files: Dict[str, Path]) -> None:
    """Plot comprehensive connectivity visualizations (Circle, Heatmap, Network)."""
    if ctx.connectivity_df is None or ctx.epochs is None:
        return

    power_bands = get_frequency_band_names(ctx.config)
    conn_measures = get_config_value(ctx.config, "plotting.plots.features.connectivity.measures", ["wpli", "aec"])
    
    for measure in conn_measures:
        for band in power_bands:
            prefix = f"{measure}_{band}"
            conn_dir = ctx.subdir("connectivity")
            
            # 1. Circle Plot by Condition
            if ctx.aligned_events is not None:
                from eeg_pipeline.plotting.features.connectivity import plot_connectivity_circle_by_condition
                _safe_plot(
                    ctx, saved_files, f"{measure}_{band}_circle_condition", "connectivity",
                    None,
                    plot_connectivity_circle_by_condition,
                    ctx.connectivity_df, ctx.aligned_events, ctx.epochs.info,
                    ctx.subject, conn_dir, ctx.logger, ctx.config,
                    measure=measure, band=band
                )
            
            # 2. Heatmap
            _safe_plot(
                ctx, saved_files, f"{measure}_{band}_heatmap", "connectivity",
                None,
                plot_connectivity_heatmap,
                ctx.connectivity_df, ctx.epochs.info, ctx.subject, conn_dir, ctx.logger, ctx.config,
                prefix=prefix, events_df=None
            )
            
            # 3. Network Graph
            _safe_plot(
                ctx, saved_files, f"{measure}_{band}_network", "connectivity",
                None,
                plot_connectivity_network,
                ctx.connectivity_df, ctx.epochs.info, ctx.subject, conn_dir, ctx.logger, ctx.config,
                prefix=prefix, events_df=None
            )

@VisualizationRegistry.register("connectivity")
def _plot_connectivity_dynamics(ctx: FeaturePlotContext, saved_files: Dict[str, Path]) -> None:
    """Plot sliding window connectivity dynamics."""
    if ctx.connectivity_df is None:
        return

    # Identify sliding windows from columns
    # Pattern: sw{N}corr_all_...
    sw_labels = sorted(
        {match.group(1) for col in ctx.connectivity_df.columns for match in [re.match(r"^sw(\d+)corr_all_", col)] if match}
    )
    
    if not sw_labels:
        return
        
    window_indices = [int(lbl) for lbl in sw_labels if lbl.isdigit()]
    if not window_indices:
        return
        
    window_centers = sliding_window_centers(ctx.config, max(window_indices) + 1)
    
    _safe_plot(
        ctx, saved_files, "connectivity_dynamics", "connectivity",
        None,
        plot_sliding_connectivity_trajectories,
        ctx.connectivity_df, window_indices, window_centers, None,
        ctx.subject, ctx.subdir("connectivity"), ctx.logger, ctx.config
    )




# =============================================================================
# POWER PLOTS - Descriptive
# =============================================================================


@VisualizationRegistry.register("power")
def _plot_power_condition_comparison(ctx: FeaturePlotContext, saved_files: Dict[str, Path]) -> None:
    """Compare power between conditions."""
    if ctx.power_df is None or ctx.aligned_events is None:
        return
        
    _safe_plot(
        ctx, saved_files, "power_by_condition", "power",
        None,
        plot_power_by_condition,
        power_df=ctx.power_df,
        events_df=ctx.aligned_events,
        subject=ctx.subject,
        save_dir=ctx.subdir("power"),
        logger=ctx.logger,
        config=ctx.config
    )
    
    _safe_plot(
        ctx, saved_files, "power_roi_band_condition", "power",
        None,
        plot_power_by_roi_band_condition,
        features_df=ctx.power_df,
        events_df=ctx.aligned_events,
        subject=ctx.subject,
        save_dir=ctx.subdir("power"),
        logger=ctx.logger,
        config=ctx.config
    )
    
    # Unified Power Band × Segment × Condition
    if ctx.all_features is not None and ctx.aligned_events is not None:
        _safe_plot(
            ctx, saved_files, "power_band_segment_condition", "power",
            None,
            plot_band_segment_condition,
            ctx.all_features, ctx.aligned_events, ctx.subject, ctx.subdir("power"), 
            ctx.logger, ctx.config, "power", "Band Power", ["baseline", "plateau"]
        )
    
    # Power Plateau vs Baseline (Paired Wilcoxon)
    if ctx.all_features is not None:
        _safe_plot(
            ctx, saved_files, "power_plateau_vs_baseline", "power",
            None,
            plot_power_plateau_vs_baseline,
            ctx.all_features, ctx.subject, ctx.subdir("power"), ctx.logger, ctx.config
        )
    
    # Power Temporal Evolution (Early/Mid/Late)
    if ctx.all_features is not None and ctx.aligned_events is not None:
        _safe_plot(
            ctx, saved_files, "power_temporal_evolution", "power",
            None,
            plot_temporal_evolution,
            ctx.all_features, ctx.aligned_events, ctx.subject, ctx.subdir("power"), 
            ctx.logger, ctx.config, "power", "Band Power"
        )


@VisualizationRegistry.register("power")
def _plot_power_variability_and_cross_freq(ctx: FeaturePlotContext, saved_files: Dict[str, Path]) -> None:
    """Power variability (CV, Fano) and cross-frequency correlation plots."""
    if ctx.power_df is None:
        return
    
    power_bands = get_frequency_band_names(ctx.config)
    
    _safe_plot(
        ctx, saved_files, "power_variability_comprehensive", "power",
        None,
        plot_power_variability_comprehensive,
        pow_df=ctx.power_df,
        bands=power_bands,
        subject=ctx.subject,
        save_dir=ctx.subdir("power"),
        logger=ctx.logger,
        config=ctx.config
    )
    
    _safe_plot(
        ctx, saved_files, "cross_frequency_power_correlation", "power",
        None,
        plot_cross_frequency_power_correlation,
        pow_df=ctx.power_df,
        bands=power_bands,
        subject=ctx.subject,
        save_dir=ctx.subdir("power"),
        logger=ctx.logger,
        config=ctx.config
    )
    
    _safe_plot(
        ctx, saved_files, "power_trial_variability", "power",
        None,
        plot_power_trial_variability,
        pow_df=ctx.power_df,
        bands=power_bands,
        subject=ctx.subject,
        save_dir=ctx.subdir("power"),
        logger=ctx.logger,
        config=ctx.config
    )


@VisualizationRegistry.register("power")
def _plot_power_descriptive_summary(ctx: FeaturePlotContext, saved_files: Dict[str, Path]) -> None:
    """Feature stability, temporal autocorrelation, redundancy, and topomaps."""
    power_bands = get_frequency_band_names(ctx.config)
    
    if ctx.all_features is not None:
        _safe_plot(
            ctx, saved_files, "feature_stability_heatmap", "summary",
            None,
            plot_feature_stability_heatmap,
            features_df=ctx.all_features,
            subject=ctx.subject,
            save_dir=ctx.subdir("summary"),
            logger=ctx.logger,
            config=ctx.config
        )
        
        _safe_plot(
            ctx, saved_files, "temporal_autocorrelation", "summary",
            None,
            plot_temporal_autocorrelation,
            features_df=ctx.all_features,
            subject=ctx.subject,
            save_dir=ctx.subdir("summary"),
            logger=ctx.logger,
            config=ctx.config
        )
        
        _safe_plot(
            ctx, saved_files, "feature_redundancy_matrix", "summary",
            None,
            plot_feature_redundancy_matrix,
            features_df=ctx.all_features,
            subject=ctx.subject,
            save_dir=ctx.subdir("summary"),
            logger=ctx.logger,
            config=ctx.config
        )
    
    if ctx.power_df is not None and ctx.epochs_info is not None:
        _safe_plot(
            ctx, saved_files, "band_power_topomaps_plateau", "power",
            None,
            plot_band_power_topomaps,
            pow_df=ctx.power_df,
            epochs_info=ctx.epochs_info,
            bands=power_bands,
            subject=ctx.subject,
            save_dir=ctx.subdir("power"),
            logger=ctx.logger,
            config=ctx.config,
            segment="plateau"
        )
        
        _safe_plot(
            ctx, saved_files, "power_topomaps_from_df", "power",
            None,
            plot_power_topomaps_from_df,
            pow_df=ctx.power_df,
            epochs_info=ctx.epochs_info,
            bands=power_bands,
            subject=ctx.subject,
            save_dir=ctx.subdir("power"),
            logger=ctx.logger,
            config=ctx.config,
        )
        
        _safe_plot(
            ctx, saved_files, "band_power_topomaps_baseline", "power",
            None,
            plot_band_power_topomaps,
            pow_df=ctx.power_df,
            epochs_info=ctx.epochs_info,
            bands=power_bands,
            subject=ctx.subject,
            save_dir=ctx.subdir("power"),
            logger=ctx.logger,
            config=ctx.config,
            segment="baseline"
        )
    
    if ctx.aperiodic_df is not None and ctx.epochs_info is not None:
        _safe_plot(
            ctx, saved_files, "spectral_slope_topomap", "aperiodic",
            None,
            plot_spectral_slope_topomap,
            aperiodic_df=ctx.aperiodic_df,
            epochs_info=ctx.epochs_info,
            subject=ctx.subject,
            save_dir=ctx.subdir("aperiodic"),
            logger=ctx.logger,
            config=ctx.config
        )
    # feature_importance_ranking intentionally disabled per request


# =============================================================================
# ERDS PLOTS - Event-Related Desynchronization/Synchronization
# =============================================================================

@VisualizationRegistry.register("erds")
def _plot_erds_descriptive(ctx: FeaturePlotContext, saved_files: Dict[str, Path]) -> None:
    """ERDS distribution and temporal evolution plots."""
    if ctx.dynamics_df is None:
        return
    
    erds_dir = ctx.subdir("erds")
    
    _safe_plot(
        ctx, saved_files, "erds_temporal_evolution", "erds",
        None,
        plot_erds_temporal_evolution,
        features_df=ctx.dynamics_df,
        save_path=erds_dir / f"sub-{ctx.subject}_erds_temporal_evolution",
        config=ctx.config,
    )


# =============================================================================
# DYNAMICS PLOTS - DFA, Autocorrelation, MSE
# =============================================================================

@VisualizationRegistry.register("dynamics")
def _plot_dynamics_descriptive(ctx: FeaturePlotContext, saved_files: Dict[str, Path]) -> None:
    """Temporal dynamics plots: autocorrelation decay, DFA scaling."""
    if ctx.dynamics_df is None:
        return
    
    dynamics_dir = ctx.subdir("dynamics")
    
    _safe_plot(
        ctx, saved_files, "autocorrelation_decay", "dynamics",
        None,
        plot_autocorrelation_decay,
        dynamics_df=ctx.dynamics_df,
        save_path=dynamics_dir / f"sub-{ctx.subject}_autocorrelation_decay"
    )


# =============================================================================
# MICROSTATE PLOTS - Descriptive
# =============================================================================

@VisualizationRegistry.register("microstates")
def _plot_microstate_condition_comparison(ctx: FeaturePlotContext, saved_files: Dict[str, Path]) -> None:
    """Compare microstate metrics between conditions."""
    if ctx.microstate_df is None or ctx.aligned_events is None:
        return
    try:
        plot_microstate_by_condition(
            microstate_df=ctx.microstate_df,
            events_df=ctx.aligned_events,
            subject=ctx.subject,
            save_dir=ctx.subdir("microstates"),
            logger=ctx.logger,
            config=ctx.config
        )
        ctx.logger.info("Created: microstate_condition_comparisons")
    except Exception as e:
        ctx.logger.warning(f"Failed microstate condition comparison: {e}")
    
    try:
        plot_microstate_transition_matrix(
            microstate_df=ctx.microstate_df,
            subject=ctx.subject,
            save_dir=ctx.subdir("microstates"),
            logger=ctx.logger,
            config=ctx.config
        )
    except Exception as e:
        ctx.logger.warning(f"Failed microstate transition matrix: {e}")
    
    # Unified Band × Segment × Condition for Microstates
    if ctx.microstate_df is not None and ctx.aligned_events is not None:
        _safe_plot(
            ctx, saved_files, "microstates_band_segment_condition", "microstates",
            None,
            plot_band_segment_condition,
            ctx.microstate_df, ctx.aligned_events, ctx.subject, ctx.subdir("microstates"), 
            ctx.logger, ctx.config, "microstates", "Microstates", ["baseline", "plateau"]
        )


@VisualizationRegistry.register("pac")
def _plot_pac_summary(ctx: FeaturePlotContext, saved_files: Dict[str, Path]) -> None:
    """PAC summary plot."""
    if ctx.pac_df is None:
        return
    
    _safe_plot(
        ctx, saved_files, "pac_summary", "pac",
        None,
        plot_pac_summary,
        pac_df=ctx.pac_df,
        subject=ctx.subject,
        save_dir=ctx.subdir("pac"),
        logger=ctx.logger,
        config=ctx.config
    )


@VisualizationRegistry.register("connectivity")
def _plot_connectivity_condition(ctx: FeaturePlotContext, saved_files: Dict[str, Path]) -> None:
    """Compare connectivity between Pain and Non-pain conditions."""
    if ctx.connectivity_df is None or ctx.aligned_events is None:
        return
    
    _safe_plot(
        ctx, saved_files, "connectivity_by_condition", "connectivity",
        None,
        plot_connectivity_by_condition,
        features_df=ctx.connectivity_df,
        events_df=ctx.aligned_events,
        subject=ctx.subject,
        save_dir=ctx.subdir("connectivity"),
        logger=ctx.logger,
        config=ctx.config
    )
    
    conn_measures = get_config_value(ctx.config, "plotting.plots.features.connectivity.measures", ["wpli", "aec"])
    for measure in conn_measures:
        _safe_plot(
            ctx, saved_files, f"conn_{measure}_roi_band_condition", "connectivity",
            None,
            plot_connectivity_by_roi_band_condition,
            features_df=ctx.connectivity_df,
            events_df=ctx.aligned_events,
            subject=ctx.subject,
            save_dir=ctx.subdir("connectivity"),
            logger=ctx.logger,
            config=ctx.config,
            measure=measure
        )
    
    # Unified Band × Segment × Condition for Connectivity (wPLI global mean)
    if ctx.connectivity_df is not None and ctx.aligned_events is not None:
        _safe_plot(
            ctx, saved_files, "conn_band_segment_condition", "connectivity",
            None,
            plot_band_segment_condition,
            ctx.connectivity_df, ctx.aligned_events, ctx.subject, ctx.subdir("connectivity"), 
            ctx.logger, ctx.config, "conn_plv", "Connectivity (PLV)", ["baseline", "plateau"]
        )


# =============================================================================
# COMPLEXITY PLOTS
# =============================================================================

# =============================================================================
# BURST / DYNAMICS PLOTS
# =============================================================================

@VisualizationRegistry.register("burst")
def _plot_burst_features(ctx: FeaturePlotContext, saved_files: Dict[str, Path]) -> None:
    """Generate burst dynamics visualizations."""
    df = ctx.dynamics_df if ctx.dynamics_df is not None else ctx.all_features
    if df is None:
        return
    
    save_dir = ctx.subdir("dynamics")
    
    _safe_plot(
        ctx, saved_files, "burst_summary", "dynamics",
        None,
        plot_burst_summary_by_band,
        df, save_dir / f"sub-{ctx.subject}_burst_summary.png",
        config=ctx.config
    )
    
    if ctx.aligned_events is not None:
        _safe_plot(
            ctx, saved_files, "dynamics_by_condition", "dynamics",
            None,
            plot_dynamics_by_condition,
            df, ctx.aligned_events, ctx.subject, save_dir, ctx.config
        )
    
    # Unified Band × Segment × Condition for Dynamics
    if ctx.dynamics_df is not None and ctx.aligned_events is not None:
        _safe_plot(
            ctx, saved_files, "dynamics_band_segment_condition", "dynamics",
            None,
            plot_band_segment_condition,
            ctx.dynamics_df, ctx.aligned_events, ctx.subject, save_dir, 
            ctx.logger, ctx.config, "dynamics", "Dynamics", ["baseline", "plateau"]
        )


# =============================================================================
# PAC PLOTS
# =============================================================================

@VisualizationRegistry.register("pac")
def _plot_pac_features(ctx: FeaturePlotContext, saved_files: Dict[str, Path]) -> None:
    """Generate phase-amplitude coupling visualizations."""
    from eeg_pipeline.plotting.features.phase import (
        plot_pac_comodulograms, plot_pac_time_ribbons
    )
    
    pac_dir = ctx.subdir("pac")
    
    if ctx.pac_df is not None:
        _safe_plot(
            ctx, saved_files, "pac_comodulograms", "pac",
            None,
            plot_pac_comodulograms,
            ctx.pac_df, ctx.subject, pac_dir, ctx.logger, ctx.config
        )
    
    if ctx.pac_time_df is not None:
        _safe_plot(
            ctx, saved_files, "pac_time_ribbons", "pac",
            None,
            plot_pac_time_ribbons,
            ctx.pac_time_df, ctx.subject, pac_dir, ctx.logger, ctx.config
        )
    
    if ctx.pac_trials_df is not None and ctx.aligned_events is not None:
        _safe_plot(
            ctx, saved_files, "pac_by_condition", "pac",
            None,
            plot_pac_by_condition,
            ctx.pac_trials_df,
            ctx.aligned_events,
            ctx.subject,
            pac_dir,
            ctx.logger,
            ctx.config,
        )
    
    # Note: plot_pac_by_roi_condition in _plot_itpc_features handles PAC condition comparison
    # using wide-format pac_trials_df


# =============================================================================
# APERIODIC PLOTS
# =============================================================================

@VisualizationRegistry.register("aperiodic")
def _plot_aperiodic_features(ctx: FeaturePlotContext, saved_files: Dict[str, Path]) -> None:
    """Generate aperiodic (1/f) feature visualizations."""
    from eeg_pipeline.plotting.features.aperiodic import (
        plot_aperiodic_topomaps,
        plot_aperiodic_by_condition,
    )
    
    aper_dir = ctx.subdir("aperiodic")
    
    if ctx.aperiodic_df is not None and ctx.epochs_info is not None:
        _safe_plot(
            ctx, saved_files, "aperiodic_topomaps", "aperiodic",
            None,
            plot_aperiodic_topomaps,
            ctx.aperiodic_df, ctx.aligned_events, ctx.epochs_info,
            ctx.subject, aper_dir, ctx.logger, ctx.config
        )
    
    if ctx.aperiodic_df is not None and ctx.aligned_events is not None:
        _safe_plot(
            ctx, saved_files, "aperiodic_by_condition", "aperiodic",
            None,
            plot_aperiodic_by_condition,
            ctx.aperiodic_df, ctx.aligned_events,
            ctx.subject, aper_dir, ctx.logger, ctx.config
        )
        
        _safe_plot(
            ctx, saved_files, "aperiodic_roi_condition", "aperiodic",
            None,
            plot_aperiodic_by_roi_condition,
            features_df=ctx.all_features,
            events_df=ctx.aligned_events,
            subject=ctx.subject,
            save_dir=aper_dir,
            logger=ctx.logger,
            config=ctx.config
        )
    
    # Unified Band × Segment × Condition for Aperiodic
    if ctx.aperiodic_df is not None and ctx.aligned_events is not None:
        _safe_plot(
            ctx, saved_files, "aperiodic_band_segment_condition", "aperiodic",
            None,
            plot_band_segment_condition,
            ctx.aperiodic_df, ctx.aligned_events, ctx.subject, aper_dir, 
            ctx.logger, ctx.config, "aperiodic", "Aperiodic (1/f)", ["baseline", "plateau"]
        )


# =============================================================================
# ITPC PLOTS
# =============================================================================

@VisualizationRegistry.register("itpc")
def _plot_itpc_features(ctx: FeaturePlotContext, saved_files: Dict[str, Path]) -> None:
    """Generate inter-trial phase coherence visualizations."""
    from eeg_pipeline.plotting.features.phase import plot_itpc_heatmap, plot_itpc_topomaps, plot_itpc_by_condition
    
    itpc_dir = ctx.subdir("itpc")
    
    stats_dir = deriv_stats_path(ctx.config.deriv_root, ctx.subject)
    itpc_path = stats_dir / "itpc_data.npz"
    
    if itpc_path.exists():
        try:
            data = np.load(itpc_path, allow_pickle=True)
            itpc_map = data.get('itpc_map')
            freqs = data.get('freqs')
            times = data.get('times')
            
            if itpc_map is not None and freqs is not None and times is not None:
                _safe_plot(
                    ctx, saved_files, "itpc_heatmap", "itpc",
                    None,
                    plot_itpc_heatmap,
                    itpc_map, freqs, times, ctx.subject, itpc_dir, ctx.logger, ctx.config
                )
        except Exception as e:
            ctx.logger.debug(f"Could not load ITPC data: {e}")
    
    if ctx.itpc_df is not None and ctx.epochs_info is not None:
        _safe_plot(
            ctx, saved_files, "itpc_topomaps", "itpc",
            None,
            plot_itpc_topomaps,
            ctx.itpc_df, ctx.epochs_info, ctx.subject, itpc_dir, ctx.logger, ctx.config
        )
    
    if ctx.itpc_df is not None and ctx.aligned_events is not None:
        _safe_plot(
            ctx, saved_files, "itpc_by_condition", "itpc",
            None,
            plot_itpc_by_condition,
            ctx.itpc_df, ctx.aligned_events, ctx.subject, itpc_dir, ctx.logger, ctx.config
        )
    
    # ITPC by ROI × Band × Condition
    if ctx.itpc_df is not None and ctx.aligned_events is not None:
        _safe_plot(
            ctx, saved_files, "itpc_roi_band_condition", "itpc",
            None,  # Function handles saving internally
            plot_itpc_by_roi_band_condition,
            ctx.itpc_df, ctx.aligned_events, ctx.subject, itpc_dir, ctx.logger, ctx.config
        )
    
    # ITPC Plateau vs Baseline comparison
    if ctx.itpc_df is not None:
        _safe_plot(
            ctx, saved_files, "itpc_plateau_vs_baseline", "itpc",
            None,  # Function handles saving internally
            plot_itpc_plateau_vs_baseline,
            ctx.itpc_df, ctx.subject, itpc_dir, ctx.logger, ctx.config
        )
    
    # PAC by ROI × Condition
    if ctx.pac_trials_df is not None and ctx.aligned_events is not None:
        pac_dir = ctx.subdir("pac")
        _safe_plot(
            ctx, saved_files, "pac_roi_condition", "pac",
            None,  # Function handles saving internally
            plot_pac_by_roi_condition,
            ctx.pac_trials_df, ctx.aligned_events, ctx.subject, pac_dir, ctx.logger, ctx.config
        )
    
    # Unified Band × Segment × Condition for ITPC
    if ctx.itpc_df is not None and ctx.aligned_events is not None:
        _safe_plot(
            ctx, saved_files, "itpc_band_segment_condition", "itpc",
            None,
            plot_band_segment_condition,
            ctx.itpc_df, ctx.aligned_events, ctx.subject, itpc_dir, ctx.logger, ctx.config,
            "itpc", "ITPC", ["baseline", "plateau"]
        )
    
    # Feature Correlation Heatmap (All Features)
    if ctx.all_features is not None:
        summary_dir = ctx.subdir("summary")
        _safe_plot(
            ctx, saved_files, "feature_correlation_heatmap", "summary",
            None,
            plot_feature_correlation_heatmap,
            ctx.all_features, ctx.subject, summary_dir, ctx.logger, ctx.config
        )
    
    # QC: Missing Data Matrix
    if ctx.all_features is not None:
        from eeg_pipeline.plotting.features.quality import plot_missing_data_matrix
        summary_dir = ctx.subdir("summary")
        _safe_plot(
            ctx, saved_files, "missing_data_matrix", "summary",
            None,
            plot_missing_data_matrix,
            ctx.all_features, summary_dir / f"sub-{ctx.subject}_missing_data_matrix",
            config=ctx.config
        )
    
    # QC: Feature Distribution Grid


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def visualize_features_for_subjects(
    subjects: List[str],
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    config: Any = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Visualize features for multiple subjects."""
    if not subjects:
        raise ValueError("No subjects specified")
    
    if config is None:
        from eeg_pipeline.utils.config.loader import load_settings
        config = load_settings()
    
    setup_matplotlib(config)
    
    task = task or config.get("project.task", "thermalactive")
    
    if logger is None:
        logger = get_logger(__name__)
    
    logger.info(f"Starting feature visualization: {len(subjects)} subject(s), task={task}")
    
    for idx, subject in enumerate(subjects, 1):
        logger.info(f"[{idx}/{len(subjects)}] Visualizing sub-{subject}")
        
        # Load Epochs (needed for MNE plots and aligned events)
        # Note: Feature DataFrames are loaded inside visualize_features via ctx.load_data()
        # to avoid redundant loading (previously load_feature_bundle_for_subject was called
        # here AND ctx.load_data() was called inside visualize_features)
        from eeg_pipeline.utils.data.loading import load_epochs_for_analysis
        epochs, aligned_events = load_epochs_for_analysis(
            subject, task, align="strict", preload=False,
            deriv_root=config.deriv_root, bids_root=config.bids_root,
            config=config, logger=logger
        )
        
        # Trigger visualization (DataFrames loaded inside via ctx.load_data())
        visualize_features(
            subject=subject,
            deriv_root=config.deriv_root,
            config=config,
            logger=logger,
            epochs_info=epochs.info if epochs else None,
            aligned_events=aligned_events,
            epochs=epochs  # Passing epochs enables MNE plots; TFR cached on first use
        )
    
    logger.info("Feature visualization complete")
