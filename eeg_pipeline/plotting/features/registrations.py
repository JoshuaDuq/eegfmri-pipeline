"""
Feature Visualization Registry

All visualization registration functions for feature plotting.
Each function registers with VisualizationRegistry to be called during feature visualization.
"""
from __future__ import annotations

import re

import numpy as np
import mne

from eeg_pipeline.plotting.features.context import FeaturePlotContext, VisualizationRegistry
from eeg_pipeline.plotting.core.runner import safe_plot
from eeg_pipeline.infra.paths import deriv_stats_path, ensure_dir
from eeg_pipeline.utils.analysis.tfr import compute_tfr_for_visualization
from eeg_pipeline.utils.analysis.windowing import sliding_window_centers
from eeg_pipeline.utils.config.loader import get_frequency_band_names, get_config_value
from eeg_pipeline.domain.features.naming import NamingSchema

from eeg_pipeline.plotting.features.aperiodic import (
    plot_aperiodic_topomaps,
    plot_aperiodic_by_condition,
)
from eeg_pipeline.plotting.features.connectivity import (
    plot_connectivity_by_condition,
    plot_connectivity_circle_by_condition,
    plot_connectivity_heatmap,
    plot_connectivity_network,
    plot_sliding_connectivity_trajectories,
)
from eeg_pipeline.plotting.features.erds import (
    plot_erds_temporal_evolution,
    plot_erds_latency_distribution,
    plot_erds_erd_ers_separation,
    plot_erds_global_summary,
    plot_erds_by_condition,
)
from eeg_pipeline.plotting.features.quality import (
    plot_feature_distribution_grid,
    plot_outlier_trials_heatmap,
    plot_snr_distribution,
)
from eeg_pipeline.plotting.features.complexity import (
    plot_complexity_by_condition,
    plot_complexity_by_band,
)
from eeg_pipeline.plotting.features.spectral import (
    plot_spectral_summary,
    plot_spectral_edge_frequency,
    plot_spectral_by_condition,
)
from eeg_pipeline.plotting.features.ratios import (
    plot_ratios_by_pair,
    plot_ratios_by_condition,
)
from eeg_pipeline.plotting.features.asymmetry import (
    plot_asymmetry_by_band,
    plot_asymmetry_by_condition,
)
from eeg_pipeline.plotting.features.bursts import (
    plot_bursts_by_band,
    plot_bursts_by_condition,
)
from eeg_pipeline.plotting.features.phase import (
    plot_itpc_heatmap,
    plot_itpc_topomaps,
    plot_itpc_by_condition,
    plot_pac_summary,
    plot_pac_by_condition,
    plot_pac_comodulograms,
    plot_pac_time_ribbons,
    convert_pac_wide_to_long,
)
from eeg_pipeline.plotting.features.power import (
    plot_power_by_condition,
    plot_power_variability_comprehensive,
    plot_cross_frequency_power_correlation,
    plot_band_power_topomaps,
    plot_spectral_slope_topomap,
    plot_power_topomaps_from_df,
)
from eeg_pipeline.plotting.features.roi import (
    plot_band_segment_condition,
    plot_temporal_evolution,
)

from eeg_pipeline.plotting.erp import (
    plot_butterfly_erp,
    plot_roi_erp,
    plot_erp_contrast,
    plot_erp_topomaps,
)


###################################################################
# Aperiodic
###################################################################


@VisualizationRegistry.register("aperiodic")
def aperiodic_suite(ctx: FeaturePlotContext, saved_files):
    aper_dir = ctx.subdir("aperiodic")

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
        )

        if ctx.all_features is not None:


            safe_plot(
                ctx,
                saved_files,
                "aperiodic_band_segment_condition",
                "aperiodic",
                None,
                plot_band_segment_condition,
                ctx.all_features,
                ctx.aligned_events,
                ctx.subject,
                aper_dir,
                ctx.logger,
                ctx.config,
                "aperiodic",
                "Aperiodic",
                ["baseline", "active"],
            )

        if ctx.temporal_df is None and ctx.all_features is not None and ctx.aligned_events is not None:
            safe_plot(
                ctx,
                saved_files,
                "aperiodic_temporal_evolution",
                "aperiodic",
                None,
                plot_temporal_evolution,
                ctx.all_features,
                ctx.aligned_events,
                ctx.subject,
                aper_dir,
                ctx.logger,
                ctx.config,
                "aperiodic",
                "Aperiodic",
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
            prefix = f"{measure}_{band}"
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
                prefix=prefix,
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
                prefix=prefix,
                events_df=None,
            )


@VisualizationRegistry.register("connectivity")
def plot_connectivity_dynamics(ctx: FeaturePlotContext, saved_files):
    if ctx.connectivity_df is None:
        return

    sw_labels = sorted(
        {
            match.group(1)
            for col in ctx.connectivity_df.columns
            for match in [re.match(r"^sw(\d+)corr_all_", col)]
            if match
        }
    )
    if not sw_labels:
        return

    window_indices = [int(lbl) for lbl in sw_labels if lbl.isdigit()]
    if not window_indices:
        return

    window_centers = sliding_window_centers(ctx.config, max(window_indices) + 1)

    safe_plot(
        ctx,
        saved_files,
        "connectivity_dynamics",
        "connectivity",
        None,
        plot_sliding_connectivity_trajectories,
        ctx.connectivity_df,
        window_indices,
        window_centers,
        None,
        ctx.subject,
        ctx.subdir("connectivity"),
        ctx.logger,
        ctx.config,
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
    )


###################################################################
# ERDS
###################################################################


@VisualizationRegistry.register("erds")
def plot_erds(ctx: FeaturePlotContext, saved_files):
    if ctx.erds_df is None:
        return

    erds_dir = ctx.subdir("erds")

    safe_plot(
        ctx,
        saved_files,
        "erds_temporal_evolution",
        "erds",
        None,
        plot_erds_temporal_evolution,
        features_df=ctx.erds_df,
        save_path=erds_dir / f"sub-{ctx.subject}_erds_temporal_evolution",
        config=ctx.config,
    )

    safe_plot(
        ctx,
        saved_files,
        "erds_latency_distribution",
        "erds",
        None,
        plot_erds_latency_distribution,
        features_df=ctx.erds_df,
        save_path=erds_dir / f"sub-{ctx.subject}_erds_latency_distribution",
        config=ctx.config,
    )

    safe_plot(
        ctx,
        saved_files,
        "erds_erd_ers_separation",
        "erds",
        None,
        plot_erds_erd_ers_separation,
        features_df=ctx.erds_df,
        save_path=erds_dir / f"sub-{ctx.subject}_erds_erd_ers_separation",
        config=ctx.config,
    )

    safe_plot(
        ctx,
        saved_files,
        "erds_global_summary",
        "erds",
        None,
        plot_erds_global_summary,
        features_df=ctx.erds_df,
        save_path=erds_dir / f"sub-{ctx.subject}_erds_global_summary",
        config=ctx.config,
    )

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
        )


###################################################################
# Complexity
###################################################################


@VisualizationRegistry.register("complexity")
def plot_complexity(ctx: FeaturePlotContext, saved_files):
    if ctx.complexity_df is None:
        return

    comp_dir = ctx.subdir("complexity")

    safe_plot(
        ctx,
        saved_files,
        "complexity_by_band",
        "complexity",
        None,
        plot_complexity_by_band,
        features_df=ctx.complexity_df,
        save_path=comp_dir / f"sub-{ctx.subject}_complexity_by_band",
        config=ctx.config,
    )

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
        )

    if ctx.all_features is not None and ctx.aligned_events is not None:
        safe_plot(
            ctx,
            saved_files,
            "complexity_band_segment_condition",
            "complexity",
            None,
            plot_band_segment_condition,
            ctx.all_features,
            ctx.aligned_events,
            ctx.subject,
            comp_dir,
            ctx.logger,
            ctx.config,
            "comp",
            "Complexity",
            ["baseline", "active"],
        )

    if ctx.temporal_df is None and ctx.all_features is not None and ctx.aligned_events is not None:
        safe_plot(
            ctx,
            saved_files,
            "complexity_temporal_evolution",
            "complexity",
            None,
            plot_temporal_evolution,
            ctx.all_features,
            ctx.aligned_events,
            ctx.subject,
            comp_dir,
            ctx.logger,
            ctx.config,
            "comp",
            "Complexity",
        )

###################################################################
# Spectral
###################################################################


@VisualizationRegistry.register("spectral")
def plot_spectral(ctx: FeaturePlotContext, saved_files):
    if ctx.spectral_df is None:
        return

    spectral_dir = ctx.subdir("spectral")

    safe_plot(
        ctx,
        saved_files,
        "spectral_summary",
        "spectral",
        None,
        plot_spectral_summary,
        features_df=ctx.spectral_df,
        save_path=spectral_dir / f"sub-{ctx.subject}_spectral_summary",
        config=ctx.config,
    )

    safe_plot(
        ctx,
        saved_files,
        "spectral_edge_frequency",
        "spectral",
        None,
        plot_spectral_edge_frequency,
        features_df=ctx.spectral_df,
        save_path=spectral_dir / f"sub-{ctx.subject}_spectral_edge_frequency",
        config=ctx.config,
    )

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
        )

    if ctx.all_features is not None and ctx.aligned_events is not None:
        safe_plot(
            ctx,
            saved_files,
            "spectral_band_segment_condition",
            "spectral",
            None,
            plot_band_segment_condition,
            ctx.all_features,
            ctx.aligned_events,
            ctx.subject,
            spectral_dir,
            ctx.logger,
            ctx.config,
            "spectral",
            "Spectral Peak",
            ["baseline", "active"],
        )

    if ctx.temporal_df is None and ctx.all_features is not None and ctx.aligned_events is not None:
        safe_plot(
            ctx,
            saved_files,
            "spectral_temporal_evolution",
            "spectral",
            None,
            plot_temporal_evolution,
            ctx.all_features,
            ctx.aligned_events,
            ctx.subject,
            spectral_dir,
            ctx.logger,
            ctx.config,
            "spectral",
            "Spectral Peak",
        )


###################################################################
# Ratios
###################################################################


@VisualizationRegistry.register("ratios")
def plot_ratios(ctx: FeaturePlotContext, saved_files):
    if ctx.ratios_df is None:
        return

    ratios_dir = ctx.subdir("ratios")

    safe_plot(
        ctx,
        saved_files,
        "ratios_by_pair",
        "ratios",
        None,
        plot_ratios_by_pair,
        features_df=ctx.ratios_df,
        save_path=ratios_dir / f"sub-{ctx.subject}_ratios_by_pair",
        config=ctx.config,
    )

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
        )

    if ctx.all_features is not None and ctx.aligned_events is not None:
        safe_plot(
            ctx,
            saved_files,
            "ratios_band_segment_condition",
            "ratios",
            None,
            plot_band_segment_condition,
            ctx.all_features,
            ctx.aligned_events,
            ctx.subject,
            ratios_dir,
            ctx.logger,
            ctx.config,
            "ratios",
            "Band Ratios",
            ["baseline", "active"],
        )

    if ctx.temporal_df is None and ctx.all_features is not None and ctx.aligned_events is not None:
        safe_plot(
            ctx,
            saved_files,
            "ratios_temporal_evolution",
            "ratios",
            None,
            plot_temporal_evolution,
            ctx.all_features,
            ctx.aligned_events,
            ctx.subject,
            ratios_dir,
            ctx.logger,
            ctx.config,
            "ratios",
            "Band Ratios",
        )


###################################################################
# Asymmetry
###################################################################


@VisualizationRegistry.register("asymmetry")
def plot_asymmetry(ctx: FeaturePlotContext, saved_files):
    if ctx.asymmetry_df is None:
        return

    asym_dir = ctx.subdir("asymmetry")

    safe_plot(
        ctx,
        saved_files,
        "asymmetry_by_band",
        "asymmetry",
        None,
        plot_asymmetry_by_band,
        features_df=ctx.asymmetry_df,
        save_path=asym_dir / f"sub-{ctx.subject}_asymmetry_by_band",
        config=ctx.config,
    )

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
        )

    if ctx.all_features is not None and ctx.aligned_events is not None:
        safe_plot(
            ctx,
            saved_files,
            "asymmetry_band_segment_condition",
            "asymmetry",
            None,
            plot_band_segment_condition,
            ctx.all_features,
            ctx.aligned_events,
            ctx.subject,
            asym_dir,
            ctx.logger,
            ctx.config,
            "asymmetry",
            "Hemispheric Asymmetry",
            ["baseline", "active"],
        )

    if ctx.temporal_df is None and ctx.all_features is not None and ctx.aligned_events is not None:
        safe_plot(
            ctx,
            saved_files,
            "asymmetry_temporal_evolution",
            "asymmetry",
            None,
            plot_temporal_evolution,
            ctx.all_features,
            ctx.aligned_events,
            ctx.subject,
            asym_dir,
            ctx.logger,
            ctx.config,
            "asymmetry",
            "Hemispheric Asymmetry",
        )


###################################################################
# Bursts
###################################################################


@VisualizationRegistry.register("bursts")
def plot_bursts(ctx: FeaturePlotContext, saved_files):
    if ctx.bursts_df is None:
        return

    bursts_dir = ctx.subdir("bursts")

    safe_plot(
        ctx,
        saved_files,
        "bursts_by_band",
        "bursts",
        None,
        plot_bursts_by_band,
        features_df=ctx.bursts_df,
        save_path=bursts_dir / f"sub-{ctx.subject}_bursts_by_band",
        config=ctx.config,
    )

    if ctx.aligned_events is not None:
        safe_plot(
            ctx,
            saved_files,
            "bursts_by_condition",
            "bursts",
            None,
            plot_bursts_by_condition,
            features_df=ctx.bursts_df,
            events_df=ctx.aligned_events,
            subject=ctx.subject,
            save_dir=bursts_dir,
            logger=ctx.logger,
            config=ctx.config,
        )

    if ctx.all_features is not None and ctx.aligned_events is not None:
        safe_plot(
            ctx,
            saved_files,
            "burst_band_segment_condition",
            "bursts",
            None,
            plot_band_segment_condition,
            ctx.all_features,
            ctx.aligned_events,
            ctx.subject,
            bursts_dir,
            ctx.logger,
            ctx.config,
            "bursts",
            "Burst Dynamics",
            ["baseline", "active"],
        )

    if ctx.temporal_df is None and ctx.all_features is not None and ctx.aligned_events is not None:
        safe_plot(
            ctx,
            saved_files,
            "burst_temporal_evolution",
            "bursts",
            None,
            plot_temporal_evolution,
            ctx.all_features,
            ctx.aligned_events,
            ctx.subject,
            bursts_dir,
            ctx.logger,
            ctx.config,
            "bursts",
            "Burst Dynamics",
        )


###################################################################
# Temporal
###################################################################


@VisualizationRegistry.register("temporal")
def plot_temporal(ctx: FeaturePlotContext, saved_files):
    temporal_df = ctx.temporal_df
    if temporal_df is None or ctx.aligned_events is None:
        return

    temporal_dir = ctx.subdir("temporal")
    safe_plot(
        ctx,
        saved_files,
        "temporal_evolution",
        "temporal",
        None,
        plot_temporal_evolution,
        temporal_df,
        ctx.aligned_events,
        ctx.subject,
        temporal_dir,
        ctx.logger,
        ctx.config,
        feature_prefix="power",
        feature_label="Band Power",
    )


###################################################################
# ITPC
###################################################################


@VisualizationRegistry.register("itpc")
def itpc_suite(ctx: FeaturePlotContext, saved_files):
    itpc_dir = ctx.subdir("itpc")

    deriv_root = getattr(ctx.config, "deriv_root", None)
    if deriv_root is not None:
        stats_dir = deriv_stats_path(deriv_root, ctx.subject)
        itpc_path = stats_dir / "itpc_data.npz"

        if itpc_path.exists():
            try:
                data = np.load(itpc_path, allow_pickle=True)
                itpc_map = data.get("itpc_map")
                freqs = data.get("freqs")
                times = data.get("times")
                if itpc_map is not None and freqs is not None and times is not None:
                    safe_plot(
                        ctx,
                        saved_files,
                        "itpc_heatmap",
                        "itpc",
                        None,
                        plot_itpc_heatmap,
                        itpc_map,
                        freqs,
                        times,
                        ctx.subject,
                        itpc_dir,
                        ctx.logger,
                        ctx.config,
                    )
            except Exception as e:
                ctx.logger.debug(f"Could not load ITPC data: {e}")

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
        )


        safe_plot(
            ctx,
            saved_files,
            "itpc_band_segment_condition",
            "itpc",
            None,
            plot_band_segment_condition,
            ctx.itpc_df,
            ctx.aligned_events,
            ctx.subject,
            itpc_dir,
            ctx.logger,
            ctx.config,
            "itpc",
            "ITPC",
            ["baseline", "active"],
        )

    if ctx.temporal_df is None and ctx.all_features is not None and ctx.aligned_events is not None:
        safe_plot(
            ctx,
            saved_files,
            "itpc_temporal_evolution",
            "itpc",
            None,
            plot_temporal_evolution,
            ctx.all_features,
            ctx.aligned_events,
            ctx.subject,
            itpc_dir,
            ctx.logger,
            ctx.config,
            "itpc",
            "ITPC",
        )



###################################################################
# PAC
###################################################################


@VisualizationRegistry.register("pac")
def pac_summary(ctx: FeaturePlotContext, saved_files):
    pac_source = ctx.pac_df if ctx.pac_df is not None else ctx.pac_trials_df
    if pac_source is None:
        return

    safe_plot(
        ctx,
        saved_files,
        "pac_summary",
        "pac",
        None,
        plot_pac_summary,
        pac_df=pac_source,
        subject=ctx.subject,
        save_dir=ctx.subdir("pac"),
        logger=ctx.logger,
        config=ctx.config,
    )


@VisualizationRegistry.register("pac")
def pac_suite(ctx: FeaturePlotContext, saved_files):
    pac_dir = ctx.subdir("pac")
    pac_long = ctx.pac_df
    if pac_long is None and ctx.pac_trials_df is not None:
        pac_long = convert_pac_wide_to_long(ctx.pac_trials_df, logger=ctx.logger, config=ctx.config)
    if pac_long is not None:
        safe_plot(
            ctx,
            saved_files,
            "pac_comodulograms",
            "pac",
            None,
            plot_pac_comodulograms,
            pac_long,
            ctx.subject,
            pac_dir,
            ctx.logger,
            ctx.config,
        )

    if ctx.pac_time_df is not None:
        safe_plot(
            ctx,
            saved_files,
            "pac_time_ribbons",
            "pac",
            None,
            plot_pac_time_ribbons,
            ctx.pac_time_df,
            ctx.subject,
            pac_dir,
            ctx.logger,
            ctx.config,
        )

    if ctx.pac_trials_df is not None and ctx.aligned_events is not None:
        safe_plot(
            ctx,
            saved_files,
            "pac_by_condition",
            "pac",
            None,
            plot_pac_by_condition,
            ctx.pac_trials_df,
            ctx.aligned_events,
            ctx.subject,
            pac_dir,
            ctx.logger,
            ctx.config,
        )


###################################################################
# Power
###################################################################


@VisualizationRegistry.register("power")
def plot_tfr(ctx: FeaturePlotContext, saved_files):
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

    from eeg_pipeline.plotting.features.power import plot_power_spectral_density
    from eeg_pipeline.utils.analysis.tfr import compute_tfr_morlet

    tfr = compute_tfr_morlet(ctx.epochs, ctx.config, ctx.logger)
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
def plot_power_condition_comparison(ctx: FeaturePlotContext, saved_files):
    if ctx.power_df is None or ctx.aligned_events is None:
        return

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
    )


    if ctx.all_features is not None and ctx.aligned_events is not None:
        safe_plot(
            ctx,
            saved_files,
            "power_band_segment_condition",
            "power",
            None,
            plot_band_segment_condition,
            ctx.all_features,
            ctx.aligned_events,
            ctx.subject,
            ctx.subdir("power"),
            ctx.logger,
            ctx.config,
            "power",
            "Band Power",
            ["baseline", "active"],
        )


@VisualizationRegistry.register("power")
def plot_power_variability(ctx: FeaturePlotContext, saved_files):
    if ctx.power_df is None:
        return

    power_bands = get_frequency_band_names(ctx.config)

    safe_plot(
        ctx,
        saved_files,
        "power_variability_comprehensive",
        "power",
        None,
        plot_power_variability_comprehensive,
        pow_df=ctx.power_df,
        bands=power_bands,
        subject=ctx.subject,
        save_dir=ctx.subdir("power"),
        logger=ctx.logger,
        config=ctx.config,
    )

    safe_plot(
        ctx,
        saved_files,
        "cross_frequency_power_correlation",
        "power",
        None,
        plot_cross_frequency_power_correlation,
        pow_df=ctx.power_df,
        bands=power_bands,
        subject=ctx.subject,
        save_dir=ctx.subdir("power"),
        logger=ctx.logger,
        config=ctx.config,
    )

@VisualizationRegistry.register("power")
def plot_power_summary(ctx: FeaturePlotContext, saved_files):
    power_bands = get_frequency_band_names(ctx.config)

    if ctx.power_df is not None and ctx.epochs_info is not None:
        safe_plot(
            ctx,
            saved_files,
            "band_power_topomaps_active",
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
            segment="active",
        )

        safe_plot(
            ctx,
            saved_files,
            "power_topomaps_from_df",
            "power",
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

        safe_plot(
            ctx,
            saved_files,
            "band_power_topomaps_baseline",
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
            segment="baseline",
        )

    if ctx.aperiodic_df is not None and ctx.epochs_info is not None:
        safe_plot(
            ctx,
            saved_files,
            "spectral_slope_topomap",
            "aperiodic",
            None,
            plot_spectral_slope_topomap,
            aperiodic_df=ctx.aperiodic_df,
            epochs_info=ctx.epochs_info,
            subject=ctx.subject,
            save_dir=ctx.subdir("aperiodic"),
            logger=ctx.logger,
            config=ctx.config,
        )


###################################################################
# ERP
###################################################################


@VisualizationRegistry.register("erp")
def erp_suite(ctx: FeaturePlotContext, saved_files):
    if ctx.epochs is None:
        return

    erp_dir = ctx.subdir("erp")
    
    # Conditions for contrast
    conditions = None
    if ctx.aligned_events is not None:
        from eeg_pipeline.utils.analysis.events import resolve_comparison_spec
        import numpy as np
        import pandas as pd

        def _condition_key(label: str) -> str:
            key = str(label).strip().lower().replace(" ", "_").replace("-", "_")
            if key in {"non_pain", "nopain", "no_pain"}:
                return "nonpain"
            return key or "condition"

        def _query_for_value(col: str, value: Any) -> str:
            col_expr = f"`{col}`"
            try:
                v_num = pd.to_numeric(str(value), errors="coerce")
                if not np.isnan(v_num):
                    if float(v_num).is_integer():
                        return f"{col_expr} == {int(v_num)}"
                    return f"{col_expr} == {float(v_num)}"
            except Exception:
                pass
            return f"{col_expr} == {repr(str(value))}"

        spec = resolve_comparison_spec(ctx.aligned_events, ctx.config, require_enabled=False)
        if spec is not None:
            col, v1, v2, label1, label2 = spec
            candidates = {
                _condition_key(label1): _query_for_value(col, v1),
                _condition_key(label2): _query_for_value(col, v2),
            }
            available_conditions = {}
            for name, query in candidates.items():
                try:
                    if len(ctx.epochs[query]) > 0:
                        available_conditions[name] = query
                except Exception:
                    continue
            conditions = available_conditions if available_conditions else None

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

    safe_plot(
        ctx,
        saved_files,
        "erp_topomaps",
        "erp",
        None,
        plot_erp_topomaps,
        ctx.epochs,
        ctx.subject,
        erp_dir,
        ctx.config,
        ctx.logger,
        conditions=conditions,
    )


###################################################################
# Summary / Quality
###################################################################


@VisualizationRegistry.register("quality")
def quality_suite(ctx: FeaturePlotContext, saved_files):
    if ctx.quality_df is None:
        return

    quality_dir = ctx.subdir("quality")
    ensure_dir(quality_dir)

    safe_plot(
        ctx,
        saved_files,
        "quality_feature_distributions",
        "quality",
        None,
        plot_feature_distribution_grid,
        ctx.quality_df,
        quality_dir / f"sub-{ctx.subject}_quality_feature_distributions",
        config=ctx.config,
    )

    safe_plot(
        ctx,
        saved_files,
        "quality_outlier_heatmap",
        "quality",
        None,
        plot_outlier_trials_heatmap,
        ctx.quality_df,
        quality_dir / f"sub-{ctx.subject}_quality_outlier_heatmap",
        config=ctx.config,
    )

    snr_col = None
    for col in ctx.quality_df.columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid"):
            continue
        if parsed.get("group") != "quality":
            continue
        if parsed.get("scope") != "global":
            continue
        if parsed.get("stat") == "snr":
            snr_col = str(col)
            if parsed.get("segment") == "active":
                break

    if snr_col is not None:
        safe_plot(
            ctx,
            saved_files,
            "quality_snr_distribution",
            "quality",
            None,
            plot_snr_distribution,
            ctx.quality_df,
            quality_dir / f"sub-{ctx.subject}_quality_snr_distribution",
            snr_col=snr_col,
            config=ctx.config,
        )
