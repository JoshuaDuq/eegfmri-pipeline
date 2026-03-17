"""
Feature Visualization Registry

All visualization registration functions for feature plotting.
Each function registers with VisualizationRegistry to be called during feature visualization.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import mne

from eeg_pipeline.plotting.features.context import FeaturePlotContext, VisualizationRegistry
from eeg_pipeline.plotting.core.runner import safe_plot
from eeg_pipeline.utils.analysis.tfr import compute_tfr_for_visualization
from eeg_pipeline.utils.analysis.events import resolve_comparison_spec
from eeg_pipeline.utils.config.loader import get_frequency_band_names, get_config_value
from eeg_pipeline.utils.data.source_localization_paths import (
    resolve_source_localization_method,
    source_localization_estimates_dir,
)

from eeg_pipeline.plotting.features.aperiodic import (
    plot_aperiodic_topomaps,
    plot_aperiodic_by_condition,
)
from eeg_pipeline.plotting.features.connectivity import (
    plot_connectivity_by_condition,
    plot_connectivity_circle_by_condition,
    plot_connectivity_heatmap,
    plot_connectivity_network,
)
from eeg_pipeline.plotting.features.erds import (
    plot_erds_by_condition,
)
from eeg_pipeline.plotting.features.complexity import (
    plot_complexity_by_condition,
)
from eeg_pipeline.plotting.features.spectral import (
    plot_spectral_by_condition,
)
from eeg_pipeline.plotting.features.ratios import (
    plot_ratios_by_condition,
)
from eeg_pipeline.plotting.features.asymmetry import (
    plot_asymmetry_by_condition,
)
from eeg_pipeline.plotting.features.microstates import (
    plot_microstates_by_condition,
)
from eeg_pipeline.plotting.features.phase import (
    plot_itpc_topomaps,
    plot_itpc_by_condition,
    plot_pac_by_condition,
)
from eeg_pipeline.plotting.features.power import (
    plot_power_by_condition,
    plot_band_power_topomaps,
    plot_band_power_topomaps_window_contrast,
    plot_power_spectral_density,
    plot_power_timecourse_by_condition,
    plot_band_power_evolution,
)

from eeg_pipeline.plotting.features.source_localization import (
    plot_source_stc_3d,
    plot_source_glass_brain,
    plot_source_band_panel,
    plot_source_cluster_timecourse,
    plot_source_cluster_composition,
    plot_source_atlas_roi_heatmap,
    plot_source_atlas_surface,
    plot_source_cluster_raincloud,
    plot_source_cluster_tfr,
)

from eeg_pipeline.plotting.erp import (
    plot_butterfly_erp,
    plot_roi_erp,
    plot_erp_contrast,
)


###################################################################
# Aperiodic
###################################################################


@VisualizationRegistry.register("aperiodic")
def aperiodic_suite(ctx: FeaturePlotContext, saved_files):
    aper_dir = ctx.subdir("aperiodic")

    if ctx.aperiodic_df is None:
        ctx.logger.warning("aperiodic_df is None; skipping aperiodic plots")
        return
    if ctx.aperiodic_df.empty:
        ctx.logger.warning("aperiodic_df is empty; skipping aperiodic plots")
        return

    if ctx.aperiodic_df is not None and ctx.epochs_info is not None:
        safe_plot(
            ctx,
            saved_files,
            "aperiodic_topomaps",
            "aperiodic",
            None,
            plot_aperiodic_topomaps,
            ctx.aperiodic_df,
            ctx.aligned_events,
            ctx.epochs_info,
            ctx.subject,
            aper_dir,
            ctx.logger,
            ctx.config,
        )

    if ctx.aperiodic_df is not None and ctx.aligned_events is not None:
        safe_plot(
            ctx,
            saved_files,
            "aperiodic_by_condition",
            "aperiodic",
            None,
            plot_aperiodic_by_condition,
            ctx.aperiodic_df,
            ctx.aligned_events,
            ctx.subject,
            aper_dir,
            ctx.logger,
            ctx.config,
            stats_dir=ctx.stats_dir,
        )


###################################################################
# Connectivity
###################################################################


@VisualizationRegistry.register("connectivity")
def plot_connectivity_mne_suite(ctx: FeaturePlotContext, saved_files):
    if ctx.connectivity_df is None or ctx.epochs is None:
        return

    power_bands = get_frequency_band_names(ctx.config)
    conn_measures = get_config_value(
        ctx.config, "plotting.plots.features.connectivity.measures", ["wpli", "aec"]
    )

    for measure in conn_measures:
        for band in power_bands:
            conn_dir = ctx.subdir("connectivity")

            if ctx.aligned_events is not None:
                safe_plot(
                    ctx,
                    saved_files,
                    f"{measure}_{band}_circle_condition",
                    "connectivity",
                    None,
                    plot_connectivity_circle_by_condition,
                    ctx.connectivity_df,
                    ctx.aligned_events,
                    ctx.epochs.info,
                    ctx.subject,
                    conn_dir,
                    ctx.logger,
                    ctx.config,
                    measure=measure,
                    band=band,
                )

            safe_plot(
                ctx,
                saved_files,
                f"{measure}_{band}_heatmap",
                "connectivity",
                None,
                plot_connectivity_heatmap,
                ctx.connectivity_df,
                ctx.epochs.info,
                ctx.subject,
                conn_dir,
                ctx.logger,
                ctx.config,
                measure=measure,
                band=band,
                events_df=None,
            )

            safe_plot(
                ctx,
                saved_files,
                f"{measure}_{band}_network",
                "connectivity",
                None,
                plot_connectivity_network,
                ctx.connectivity_df,
                ctx.epochs.info,
                ctx.subject,
                conn_dir,
                ctx.logger,
                ctx.config,
                measure=measure,
                band=band,
                events_df=ctx.aligned_events if ctx.aligned_events is not None else None,
            )


@VisualizationRegistry.register("connectivity")
def plot_connectivity_condition(ctx: FeaturePlotContext, saved_files):
    if ctx.connectivity_df is None or ctx.aligned_events is None:
        return

    safe_plot(
        ctx,
        saved_files,
        "connectivity_by_condition",
        "connectivity",
        None,
        plot_connectivity_by_condition,
        features_df=ctx.connectivity_df,
        events_df=ctx.aligned_events,
        subject=ctx.subject,
        save_dir=ctx.subdir("connectivity"),
        logger=ctx.logger,
        config=ctx.config,
        stats_dir=ctx.stats_dir,
    )


###################################################################
# ERDS
###################################################################


@VisualizationRegistry.register("erds")
def plot_erds(ctx: FeaturePlotContext, saved_files):
    if ctx.erds_df is None:
        return

    erds_dir = ctx.subdir("erds")

    if ctx.aligned_events is not None:
        safe_plot(
            ctx,
            saved_files,
            "erds_by_condition",
            "erds",
            None,
            plot_erds_by_condition,
            features_df=ctx.erds_df,
            events_df=ctx.aligned_events,
            subject=ctx.subject,
            save_dir=erds_dir,
            logger=ctx.logger,
            config=ctx.config,
            stats_dir=ctx.stats_dir,
        )


###################################################################
# Complexity
###################################################################


@VisualizationRegistry.register("complexity")
def plot_complexity(ctx: FeaturePlotContext, saved_files):
    if ctx.complexity_df is None:
        return

    comp_dir = ctx.subdir("complexity")

    if ctx.aligned_events is not None:
        safe_plot(
            ctx,
            saved_files,
            "complexity_by_condition",
            "complexity",
            None,
            plot_complexity_by_condition,
            features_df=ctx.complexity_df,
            events_df=ctx.aligned_events,
            subject=ctx.subject,
            save_dir=comp_dir,
            logger=ctx.logger,
            config=ctx.config,
            stats_dir=ctx.stats_dir,
        )

###################################################################
# Spectral
###################################################################


@VisualizationRegistry.register("spectral")
def plot_spectral(ctx: FeaturePlotContext, saved_files):
    if ctx.spectral_df is None:
        return

    spectral_dir = ctx.subdir("spectral")

    if ctx.aligned_events is not None:
        safe_plot(
            ctx,
            saved_files,
            "spectral_by_condition",
            "spectral",
            None,
            plot_spectral_by_condition,
            features_df=ctx.spectral_df,
            events_df=ctx.aligned_events,
            subject=ctx.subject,
            save_dir=spectral_dir,
            logger=ctx.logger,
            config=ctx.config,
            stats_dir=ctx.stats_dir,
        )

###################################################################
# Ratios
###################################################################


@VisualizationRegistry.register("ratios")
def plot_ratios(ctx: FeaturePlotContext, saved_files):
    if ctx.ratios_df is None:
        return

    ratios_dir = ctx.subdir("ratios")

    if ctx.aligned_events is not None:
        safe_plot(
            ctx,
            saved_files,
            "ratios_by_condition",
            "ratios",
            None,
            plot_ratios_by_condition,
            features_df=ctx.ratios_df,
            events_df=ctx.aligned_events,
            subject=ctx.subject,
            save_dir=ratios_dir,
            logger=ctx.logger,
            config=ctx.config,
            stats_dir=ctx.stats_dir,
        )

###################################################################
# Asymmetry
###################################################################


@VisualizationRegistry.register("asymmetry")
def plot_asymmetry(ctx: FeaturePlotContext, saved_files):
    if ctx.asymmetry_df is None:
        return

    asym_dir = ctx.subdir("asymmetry")

    if ctx.aligned_events is not None:
        safe_plot(
            ctx,
            saved_files,
            "asymmetry_by_condition",
            "asymmetry",
            None,
            plot_asymmetry_by_condition,
            features_df=ctx.asymmetry_df,
            events_df=ctx.aligned_events,
            subject=ctx.subject,
            save_dir=asym_dir,
            logger=ctx.logger,
            config=ctx.config,
            stats_dir=ctx.stats_dir,
        )


###################################################################
# Microstates
###################################################################


@VisualizationRegistry.register("microstates")
def plot_microstates(ctx: FeaturePlotContext, saved_files):
    if getattr(ctx, "microstates_df", None) is None:
        return

    micro_dir = ctx.subdir("microstates")

    if ctx.aligned_events is not None:
        safe_plot(
            ctx,
            saved_files,
            "microstates_by_condition",
            "microstates",
            None,
            plot_microstates_by_condition,
            features_df=ctx.microstates_df,
            events_df=ctx.aligned_events,
            subject=ctx.subject,
            save_dir=micro_dir,
            logger=ctx.logger,
            config=ctx.config,
            stats_dir=ctx.stats_dir,
        )

###################################################################
# Temporal
###################################################################


###################################################################
# ITPC
###################################################################


@VisualizationRegistry.register("itpc")
def itpc_suite(ctx: FeaturePlotContext, saved_files):
    itpc_dir = ctx.subdir("itpc")

    if ctx.itpc_df is not None and ctx.epochs_info is not None:
        safe_plot(
            ctx,
            saved_files,
            "itpc_topomaps",
            "itpc",
            None,
            plot_itpc_topomaps,
            ctx.itpc_df,
            ctx.epochs_info,
            ctx.subject,
            itpc_dir,
            ctx.logger,
            ctx.config,
        )

    if ctx.itpc_df is not None and ctx.aligned_events is not None:
        safe_plot(
            ctx,
            saved_files,
            "itpc_by_condition",
            "itpc",
            None,
            plot_itpc_by_condition,
            ctx.itpc_df,
            ctx.aligned_events,
            ctx.subject,
            itpc_dir,
            ctx.logger,
            ctx.config,
            stats_dir=ctx.stats_dir,
        )


###################################################################
# PAC
###################################################################


@VisualizationRegistry.register("pac")
def pac_suite(ctx: FeaturePlotContext, saved_files):
    pac_dir = ctx.subdir("pac")

    # Prefer pac_trials_df if available, otherwise fall back to pac_df
    pac_df = ctx.pac_trials_df if ctx.pac_trials_df is not None and not ctx.pac_trials_df.empty else ctx.pac_df
    
    if pac_df is None or pac_df.empty:
        ctx.logger.warning(
            f"PAC suite: No PAC data available (pac_trials_df={ctx.pac_trials_df is not None}, "
            f"pac_df={ctx.pac_df is not None}), skipping pac_by_condition plot"
        )
        return
    
    if ctx.aligned_events is None:
        ctx.logger.warning("PAC suite: aligned_events is None, skipping pac_by_condition plot")
        return
    
    ctx.logger.info(
        f"PAC suite: Using {'pac_trials_df' if ctx.pac_trials_df is not None and not ctx.pac_trials_df.empty else 'pac_df'}, "
        f"shape={pac_df.shape}, aligned_events shape={ctx.aligned_events.shape}"
    )
    
    safe_plot(
        ctx,
        saved_files,
        "pac_by_condition",
        "pac",
        None,
        plot_pac_by_condition,
        pac_trials_df=pac_df,
        events_df=ctx.aligned_events,
        subject=ctx.subject,
        save_dir=pac_dir,
        logger=ctx.logger,
        config=ctx.config,
        stats_dir=ctx.stats_dir,
    )


###################################################################
# Power
###################################################################


@VisualizationRegistry.register("power")
def plot_tfr_visualization(ctx: FeaturePlotContext, saved_files):
    """Generate time-frequency representation visualization."""
    if ctx.epochs is None:
        return

    safe_plot(
        ctx,
        saved_files,
        "TFR",
        "power",
        None,
        compute_tfr_for_visualization,
        ctx.epochs,
        ctx.config,
        ctx.logger,
    )


@VisualizationRegistry.register("power")
def plot_psd_visualization(ctx: FeaturePlotContext, saved_files):
    """Generate power spectral density visualization."""
    if ctx.epochs is None:
        return

    tfr = ctx.get_or_compute_tfr()
    if tfr is None:
        return

    safe_plot(
        ctx,
        saved_files,
        "power_spectral_density",
        "power",
        None,
        plot_power_spectral_density,
        tfr,
        ctx.subject,
        ctx.subdir("power"),
        ctx.logger,
        ctx.aligned_events,
        ctx.config,
    )


@VisualizationRegistry.register("power")
def plot_power_timecourse_visualization(ctx: FeaturePlotContext, saved_files):
    """Generate time-resolved power trajectories by condition."""
    if ctx.epochs is None:
        return

    tfr = ctx.get_or_compute_tfr()
    if tfr is None:
        return

    safe_plot(
        ctx,
        saved_files,
        "power_timecourse",
        "power",
        None,
        plot_power_timecourse_by_condition,
        tfr,
        ctx.subject,
        ctx.subdir("power"),
        ctx.logger,
        ctx.aligned_events,
        ctx.config,
    )


###################################################################
# TFR
###################################################################


@VisualizationRegistry.register("tfr")
def tfr_suite(ctx: FeaturePlotContext, saved_files):
    if ctx.epochs is None:
        return

    tfr = ctx.get_or_compute_tfr()
    if tfr is None:
        return

    conditions = [("All", np.ones(len(ctx.epochs), dtype=bool))]
    if ctx.aligned_events is not None:
        from eeg_pipeline.utils.analysis.events import extract_multi_group_masks
        spec = extract_multi_group_masks(ctx.aligned_events, ctx.config, require_enabled=True)
        if spec:
            masks_dict, group_labels = spec
            conditions = [(label, masks_dict[label]) for label in group_labels]
            
    # Always try to plot if tfr is available
    safe_plot(
        ctx,
        saved_files,
        "tfr_band_evolution",
        "tfr",
        None,
        plot_band_power_evolution,
        tfr_epochs=tfr,
        conditions=conditions,
        subject=ctx.subject,
        save_dir=ctx.subdir("tfr"),
        logger=ctx.logger,
        config=ctx.config,
        roi_name="all",
    )


@VisualizationRegistry.register("power")
def plot_power_condition_comparison(ctx: FeaturePlotContext, saved_files):
    if ctx.power_df is None or ctx.aligned_events is None:
        return

    power_bands = get_frequency_band_names(ctx.config)

    safe_plot(
        ctx,
        saved_files,
        "power_by_condition",
        "power",
        None,
        plot_power_by_condition,
        power_df=ctx.power_df,
        events_df=ctx.aligned_events,
        subject=ctx.subject,
        save_dir=ctx.subdir("power"),
        logger=ctx.logger,
        config=ctx.config,
        stats_dir=ctx.stats_dir,
    )


@VisualizationRegistry.register("power")
def plot_power_summary(ctx: FeaturePlotContext, saved_files):
    power_bands = get_frequency_band_names(ctx.config)

    if ctx.power_df is not None and ctx.epochs_info is not None:
        topomap_windows = get_config_value(
            ctx.config, "plotting.plots.features.power.topomap_windows", None
        )
        
        ctx.logger.info(f"topomap_windows config value: {topomap_windows} (type: {type(topomap_windows)})")
        
        if topomap_windows:
            if isinstance(topomap_windows, str):
                windows_list = [w.strip() for w in topomap_windows.split() if w.strip()]
            elif isinstance(topomap_windows, list):
                windows_list = [str(w).strip() for w in topomap_windows if str(w).strip()]
            else:
                windows_list = []
        else:
            windows_list = []
        
        ctx.logger.info(f"windows_list after parsing: {windows_list}")
        
        if windows_list:
            for window in windows_list:
                plot_name = f"band_power_topomaps_{window}"
                ctx.logger.info(f"Generating topomap plot for window: {window} (plot_name: {plot_name})")
                safe_plot(
                    ctx,
                    saved_files,
                    plot_name,
                    "power",
                    None,
                    plot_band_power_topomaps,
                    pow_df=ctx.power_df,
                    epochs_info=ctx.epochs_info,
                    bands=power_bands,
                    subject=ctx.subject,
                    save_dir=ctx.subdir("power"),
                    logger=ctx.logger,
                    config=ctx.config,
                    segment=window,
                    events_df=ctx.aligned_events if ctx.aligned_events is not None else None,
                )

            compare_windows = bool(get_config_value(ctx.config, "plotting.comparisons.compare_windows", True))
            if compare_windows and len(windows_list) == 2:
                window1, window2 = windows_list[0], windows_list[1]
                plot_name = f"band_power_topomaps_contrast_{window2}_minus_{window1}"
                ctx.logger.info(
                    f"Generating topomap window contrast: {window2} - {window1} (plot_name: {plot_name})"
                )
                safe_plot(
                    ctx,
                    saved_files,
                    plot_name,
                    "power",
                    None,
                    plot_band_power_topomaps_window_contrast,
                    pow_df=ctx.power_df,
                    epochs_info=ctx.epochs_info,
                    bands=power_bands,
                    subject=ctx.subject,
                    save_dir=ctx.subdir("power"),
                    logger=ctx.logger,
                    config=ctx.config,
                    window1=window1,
                    window2=window2,
                )
        else:
            patterns = getattr(ctx, "plot_name_patterns", None)
            if patterns and any(fnmatch.fnmatch("band_power_topomaps", str(pat)) for pat in patterns):
                ctx.logger.error(
                    "band_power_topomaps requires plotting.plots.features.power.topomap_windows to be set. "
                    "No fallback will be used."
                )

###################################################################
# ERP
###################################################################


def _normalize_condition_key(label: str) -> str:
    """Normalize condition label to a valid key format."""
    key = str(label).strip().lower().replace(" ", "_").replace("-", "_")
    return key or "condition"


def _build_epoch_query(col: str, value: Any) -> str:
    """Build an epoch query string for a column and value."""
    col_expr = f"`{col}`"
    numeric_value = pd.to_numeric(str(value), errors="coerce")
    if not np.isnan(numeric_value):
        if float(numeric_value).is_integer():
            return f"{col_expr} == {int(numeric_value)}"
        return f"{col_expr} == {float(numeric_value)}"
    return f"{col_expr} == {repr(str(value))}"


def _resolve_erp_conditions(
    epochs: mne.Epochs,
    events_df: pd.DataFrame,
    config: Any,
) -> dict[str, str] | None:
    """Resolve ERP condition queries from comparison spec."""
    spec = resolve_comparison_spec(events_df, config, require_enabled=False)
    if spec is None:
        return None

    col, v1, v2, label1, label2 = spec
    candidates = {
        _normalize_condition_key(label1): _build_epoch_query(col, v1),
        _normalize_condition_key(label2): _build_epoch_query(col, v2),
    }

    available_conditions = {}
    for name, query in candidates.items():
        try:
            if len(epochs[query]) > 0:
                available_conditions[name] = query
        except (ValueError, KeyError, AttributeError):
            continue

    return available_conditions if available_conditions else None


@VisualizationRegistry.register("erp")
def erp_suite(ctx: FeaturePlotContext, saved_files):
    if ctx.epochs is None:
        return

    erp_dir = ctx.subdir("erp")

    conditions = None
    if ctx.aligned_events is not None:
        conditions = _resolve_erp_conditions(
            ctx.epochs,
            ctx.aligned_events,
            ctx.config,
        )

    safe_plot(
        ctx,
        saved_files,
        "erp_butterfly",
        "erp",
        None,
        plot_butterfly_erp,
        ctx.epochs,
        ctx.subject,
        erp_dir,
        ctx.config,
        ctx.logger,
        conditions=conditions,
    )

    safe_plot(
        ctx,
        saved_files,
        "erp_roi",
        "erp",
        None,
        plot_roi_erp,
        ctx.epochs,
        ctx.subject,
        erp_dir,
        ctx.config,
        ctx.logger,
        conditions=conditions,
    )

    if conditions and len(conditions) >= 2:
        safe_plot(
            ctx,
            saved_files,
            "erp_contrast",
            "erp",
            None,
            plot_erp_contrast,
            ctx.epochs,
            ctx.subject,
            erp_dir,
            ctx.config,
            ctx.logger,
        )

###################################################################
# Source Localization
###################################################################


def _find_stc_files(out_dir: Path, subject: str, source_method: str) -> list[Path]:
    """Glob STC files matching both segmented and legacy (no-seg) naming conventions."""
    patterns = [
        f"sub-{subject}_*_seg-*_cond-*_band-*_{source_method}-vl.stc",
        f"sub-{subject}_*_seg-*_cond-*_band-*_{source_method}-lh.stc",
        f"sub-{subject}_*_cond-*_band-*_{source_method}-vl.stc",
        f"sub-{subject}_*_cond-*_band-*_{source_method}-lh.stc",
    ]
    seen: set[Path] = set()
    files: list[Path] = []
    for pattern in patterns:
        for path in sorted(out_dir.glob(pattern)):
            if path not in seen and not path.name.startswith("._"):
                seen.add(path)
                files.append(path)
    return files


# Plots that read parquet/metadata rather than STC files.
_PARQUET_BASED_PLOT_IDS = frozenset({
    "source_cluster_composition",
    "source_atlas_roi_heatmap",
    "source_atlas_surface",
    "source_cluster_raincloud",
    "source_cluster_tfr",
})


@VisualizationRegistry.register("sourcelocalization")
def sourcelocalization_suite(ctx: FeaturePlotContext, saved_files):
    source_dir = ctx.subdir("sourcelocalization")
    source_dir.mkdir(parents=True, exist_ok=True)

    stc_plot_enabled = get_config_value(
        ctx.config, "plotting.plots.features.sourcelocalization.plot_stc", True
    )
    if not stc_plot_enabled:
        return

    source_method = resolve_source_localization_method(ctx.config)

    for mode in ("eeg_only", "fmri_informed"):
        # All catalog_id → plot_func pairs for this mode.
        plot_specs: list[tuple[str, Any, str]] = []
        if mode == "eeg_only":
            plot_specs = [
                ("source_localization_3d_eeg_only", plot_source_stc_3d, "3D Source Brains"),
                ("source_glass_brain_eeg_only", plot_source_glass_brain, "Glass Brain"),
                ("source_band_panel_eeg_only", plot_source_band_panel, "Band Comparison Panel"),
            ]
        else:
            plot_specs = [
                ("source_localization_3d_fmri_informed", plot_source_stc_3d, "3D Source Brains"),
                ("source_glass_brain_fmri_informed", plot_source_glass_brain, "Glass Brain"),
                ("source_band_panel_fmri_informed", plot_source_band_panel, "Band Comparison Panel"),
                ("source_cluster_timecourse", plot_source_cluster_timecourse, "Cluster Time Course"),
                ("source_cluster_composition", plot_source_cluster_composition, "Cluster Composition"),
                ("source_atlas_roi_heatmap", plot_source_atlas_roi_heatmap, "Atlas ROI Heatmap"),
                ("source_atlas_surface", plot_source_atlas_surface, "Atlas Surface Topography"),
                ("source_cluster_raincloud", plot_source_cluster_raincloud, "Cluster Epoch Distribution"),
                ("source_cluster_tfr", plot_source_cluster_tfr, "Cluster Source TFR"),
            ]

        # Filter to only the catalog IDs requested by the user.
        patterns = getattr(ctx, "plot_name_patterns", None)
        if patterns:
            plot_specs = [
                (cid, func, label)
                for cid, func, label in plot_specs
                if any(fnmatch.fnmatch(cid, str(p)) for p in patterns)
            ]
        if not plot_specs:
            continue

        # Check mode-keyed subdir first (new layout).
        mode_dir = source_localization_estimates_dir(
            features_dir=ctx.features_dir,
            method=source_method,
            mode=mode,
        )
        stc_files = _find_stc_files(mode_dir, ctx.subject, source_method) if mode_dir.exists() else []

        # For eeg_only, also check the legacy flat dir as a fallback.
        if not stc_files and mode == "eeg_only":
            legacy_dir = source_localization_estimates_dir(
                features_dir=ctx.features_dir,
                method=source_method,
            )
            stc_files = _find_stc_files(legacy_dir, ctx.subject, source_method) if legacy_dir.exists() else []
            if stc_files:
                ctx.logger.info(
                    "Source localization: using legacy flat source_estimates/ dir for eeg_only plotting."
                )

        # STC-based plots require source estimate files.
        stc_plot_specs = [
            (cid, fn, lb) for cid, fn, lb in plot_specs
            if cid not in _PARQUET_BASED_PLOT_IDS
        ]

        if stc_files:
            mode_save_dir = source_dir / mode
            for catalog_id, plot_func, label in stc_plot_specs:
                ctx.logger.info("  → %s (%s, %d files)", label, mode, len(stc_files))
                safe_plot(
                    ctx,
                    saved_files,
                    catalog_id,
                    "sourcelocalization",
                    None,
                    plot_func,
                    subject=ctx.subject,
                    stc_files=stc_files,
                    save_dir=mode_save_dir,
                    config=ctx.config,
                    logger=ctx.logger,
                )
        elif stc_plot_specs:
            expected_dir = mode_dir
            ctx.logger.warning(
                "No %s source estimate files found for sub-%s "
                "(expected STC files in %s). "
                "Run 'features compute --source-save-stc' with the appropriate mode first.",
                mode,
                ctx.subject,
                expected_dir,
            )

    # Parquet/metadata-based plots: dispatched once, independently of STC files.
    parquet_plot_specs = [
        (cid, fn, lb) for cid, fn, lb in plot_specs
        if cid in _PARQUET_BASED_PLOT_IDS
    ]
    if parquet_plot_specs:
        parquet_save_dir = source_dir / "fmri_informed"
        for catalog_id, plot_func, label in parquet_plot_specs:
            ctx.logger.info("  → %s (fmri_informed)", label)
            safe_plot(
                ctx,
                saved_files,
                catalog_id,
                "sourcelocalization",
                None,
                plot_func,
                subject=ctx.subject,
                features_dir=ctx.features_dir,
                save_dir=parquet_save_dir,
                config=ctx.config,
                logger=ctx.logger,
            )
