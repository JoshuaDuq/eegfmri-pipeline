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
from eeg_pipeline.infra.paths import deriv_stats_path
from eeg_pipeline.utils.analysis.tfr import compute_tfr_for_visualization
from eeg_pipeline.utils.analysis.windowing import sliding_window_centers
from eeg_pipeline.utils.config.loader import get_frequency_band_names, get_config_value

from eeg_pipeline.plotting.features.aperiodic import (
    plot_aperiodic_topomaps,
    plot_aperiodic_by_condition,
)
from eeg_pipeline.plotting.features.burst import (
    plot_burst_summary_by_band,
    plot_dynamics_by_condition,
)
from eeg_pipeline.plotting.features.connectivity import (
    plot_connectivity_by_condition,
    plot_connectivity_circle_by_condition,
    plot_connectivity_heatmap,
    plot_connectivity_network,
    plot_sliding_connectivity_trajectories,
)
from eeg_pipeline.plotting.features.dynamics import plot_autocorrelation_decay
from eeg_pipeline.plotting.features.erds import plot_erds_temporal_evolution
from eeg_pipeline.plotting.features.microstates import (
    plot_microstate_by_condition,
    plot_microstate_templates,
    plot_microstate_transition_matrix,
)
from eeg_pipeline.plotting.features.phase import (
    plot_itpc_heatmap,
    plot_itpc_topomaps,
    plot_itpc_by_condition,
    plot_pac_summary,
    plot_pac_by_condition,
    plot_pac_comodulograms,
    plot_pac_time_ribbons,
)
from eeg_pipeline.plotting.features.power import (
    plot_power_by_condition,
    plot_power_variability_comprehensive,
    plot_cross_frequency_power_correlation,
    plot_band_power_topomaps,
    plot_spectral_slope_topomap,
    plot_power_trial_variability,
    plot_power_topomaps_from_df,
)
from eeg_pipeline.plotting.features.roi import (
    plot_aperiodic_by_roi_condition,
    plot_band_segment_condition,
    plot_connectivity_by_roi_band_condition,
    plot_itpc_by_roi_band_condition,
    plot_itpc_plateau_vs_baseline,
    plot_pac_by_roi_condition,
    plot_power_by_roi_band_condition,
    plot_power_plateau_vs_baseline,
    plot_temporal_evolution,
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
                "aperiodic_roi_condition",
                "aperiodic",
                None,
                plot_aperiodic_by_roi_condition,
                features_df=ctx.all_features,
                events_df=ctx.aligned_events,
                subject=ctx.subject,
                save_dir=aper_dir,
                logger=ctx.logger,
                config=ctx.config,
            )

        safe_plot(
            ctx,
            saved_files,
            "aperiodic_band_segment_condition",
            "aperiodic",
            None,
            plot_band_segment_condition,
            ctx.aperiodic_df,
            ctx.aligned_events,
            ctx.subject,
            aper_dir,
            ctx.logger,
            ctx.config,
            "aperiodic",
            "Aperiodic (1/f)",
            ["baseline", "plateau"],
        )


###################################################################
# Burst / Dynamics
###################################################################


@VisualizationRegistry.register("burst")
def plot_burst_suite(ctx: FeaturePlotContext, saved_files):
    df = ctx.dynamics_df if ctx.dynamics_df is not None else ctx.all_features
    if df is None:
        return

    save_dir = ctx.subdir("dynamics")

    safe_plot(
        ctx,
        saved_files,
        "burst_summary",
        "dynamics",
        None,
        plot_burst_summary_by_band,
        df,
        save_dir / f"sub-{ctx.subject}_burst_summary.png",
        config=ctx.config,
    )

    if ctx.aligned_events is not None:
        safe_plot(
            ctx,
            saved_files,
            "dynamics_by_condition",
            "dynamics",
            None,
            plot_dynamics_by_condition,
            df,
            ctx.aligned_events,
            ctx.subject,
            save_dir,
            ctx.config,
        )

    if ctx.dynamics_df is not None and ctx.aligned_events is not None:
        safe_plot(
            ctx,
            saved_files,
            "dynamics_band_segment_condition",
            "dynamics",
            None,
            plot_band_segment_condition,
            ctx.dynamics_df,
            ctx.aligned_events,
            ctx.subject,
            save_dir,
            ctx.logger,
            ctx.config,
            "dynamics",
            "Dynamics",
            ["baseline", "plateau"],
        )


@VisualizationRegistry.register("dynamics")
def plot_dynamics(ctx: FeaturePlotContext, saved_files):
    if ctx.dynamics_df is None:
        return

    dynamics_dir = ctx.subdir("dynamics")

    safe_plot(
        ctx,
        saved_files,
        "autocorrelation_decay",
        "dynamics",
        None,
        plot_autocorrelation_decay,
        dynamics_df=ctx.dynamics_df,
        save_path=dynamics_dir / f"sub-{ctx.subject}_autocorrelation_decay",
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

    conn_measures = get_config_value(
        ctx.config, "plotting.plots.features.connectivity.measures", ["wpli", "aec"]
    )
    for measure in conn_measures:
        safe_plot(
            ctx,
            saved_files,
            f"conn_{measure}_roi_band_condition",
            "connectivity",
            None,
            plot_connectivity_by_roi_band_condition,
            features_df=ctx.connectivity_df,
            events_df=ctx.aligned_events,
            subject=ctx.subject,
            save_dir=ctx.subdir("connectivity"),
            logger=ctx.logger,
            config=ctx.config,
            measure=measure,
        )

    safe_plot(
        ctx,
        saved_files,
        "conn_band_segment_condition",
        "connectivity",
        None,
        plot_band_segment_condition,
        ctx.connectivity_df,
        ctx.aligned_events,
        ctx.subject,
        ctx.subdir("connectivity"),
        ctx.logger,
        ctx.config,
        "conn_plv",
        "Connectivity (PLV)",
        ["baseline", "plateau"],
    )


###################################################################
# ERDS
###################################################################


@VisualizationRegistry.register("erds")
def plot_erds(ctx: FeaturePlotContext, saved_files):
    if ctx.dynamics_df is None:
        return

    erds_dir = ctx.subdir("erds")

    safe_plot(
        ctx,
        saved_files,
        "erds_temporal_evolution",
        "erds",
        None,
        plot_erds_temporal_evolution,
        features_df=ctx.dynamics_df,
        save_path=erds_dir / f"sub-{ctx.subject}_erds_temporal_evolution",
        config=ctx.config,
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
            "itpc_roi_band_condition",
            "itpc",
            None,
            plot_itpc_by_roi_band_condition,
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
            ["baseline", "plateau"],
        )

    if ctx.itpc_df is not None:
        safe_plot(
            ctx,
            saved_files,
            "itpc_plateau_vs_baseline",
            "itpc",
            None,
            plot_itpc_plateau_vs_baseline,
            ctx.itpc_df,
            ctx.subject,
            itpc_dir,
            ctx.logger,
            ctx.config,
        )

    if ctx.pac_trials_df is not None and ctx.aligned_events is not None:
        pac_dir = ctx.subdir("pac")
        safe_plot(
            ctx,
            saved_files,
            "pac_roi_condition",
            "pac",
            None,
            plot_pac_by_roi_condition,
            ctx.pac_trials_df,
            ctx.aligned_events,
            ctx.subject,
            pac_dir,
            ctx.logger,
            ctx.config,
        )


###################################################################
# Microstates
###################################################################


@VisualizationRegistry.register("microstates")
def plot_microstate_templates_from_stats(ctx: FeaturePlotContext, saved_files):
    if ctx.epochs is None or ctx.microstate_df is None:
        return

    n_microstates = int(ctx.config.get("feature_engineering.microstates.n_states", 4))

    deriv_root = getattr(ctx.config, "deriv_root", None)
    if deriv_root is None:
        return

    stats_dir = deriv_stats_path(deriv_root, ctx.subject)
    template_path = stats_dir / f"microstates_templates_K{n_microstates}.npz"
    if not template_path.exists():
        return

    try:
        data = np.load(template_path)
        ms_templates = data["templates"]
        picks = mne.pick_types(
            ctx.epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads"
        )
        info_eeg = mne.pick_info(ctx.epochs.info, picks)

        def _do(save_dir):
            plot_microstate_templates(
                ms_templates,
                info_eeg,
                ctx.subject,
                save_dir,
                n_microstates,
                ctx.logger,
                ctx.config,
            )

        safe_plot(
            ctx,
            saved_files,
            "microstate_templates",
            "microstates",
            None,
            _do,
            ctx.subdir("microstates"),
        )
    except Exception as e:
        ctx.logger.error(f"Error loading/plotting microstates: {e}")


@VisualizationRegistry.register("microstates")
def plot_microstate_condition(ctx: FeaturePlotContext, saved_files):
    if ctx.microstate_df is None or ctx.aligned_events is None:
        return

    safe_plot(
        ctx,
        saved_files,
        "microstate_by_condition",
        "microstates",
        None,
        plot_microstate_by_condition,
        microstate_df=ctx.microstate_df,
        events_df=ctx.aligned_events,
        subject=ctx.subject,
        save_dir=ctx.subdir("microstates"),
        logger=ctx.logger,
        config=ctx.config,
    )

    safe_plot(
        ctx,
        saved_files,
        "microstate_transition_matrix",
        "microstates",
        None,
        plot_microstate_transition_matrix,
        microstate_df=ctx.microstate_df,
        subject=ctx.subject,
        save_dir=ctx.subdir("microstates"),
        logger=ctx.logger,
        config=ctx.config,
    )

    safe_plot(
        ctx,
        saved_files,
        "microstates_band_segment_condition",
        "microstates",
        None,
        plot_band_segment_condition,
        ctx.microstate_df,
        ctx.aligned_events,
        ctx.subject,
        ctx.subdir("microstates"),
        ctx.logger,
        ctx.config,
        "microstates",
        "Microstates",
        ["baseline", "plateau"],
    )


###################################################################
# PAC
###################################################################


@VisualizationRegistry.register("pac")
def pac_summary(ctx: FeaturePlotContext, saved_files):
    if ctx.pac_df is None:
        return

    safe_plot(
        ctx,
        saved_files,
        "pac_summary",
        "pac",
        None,
        plot_pac_summary,
        pac_df=ctx.pac_df,
        subject=ctx.subject,
        save_dir=ctx.subdir("pac"),
        logger=ctx.logger,
        config=ctx.config,
    )


@VisualizationRegistry.register("pac")
def pac_suite(ctx: FeaturePlotContext, saved_files):
    pac_dir = ctx.subdir("pac")

    if ctx.pac_df is not None:
        safe_plot(
            ctx,
            saved_files,
            "pac_comodulograms",
            "pac",
            None,
            plot_pac_comodulograms,
            ctx.pac_df,
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

    safe_plot(
        ctx,
        saved_files,
        "power_roi_band_condition",
        "power",
        None,
        plot_power_by_roi_band_condition,
        features_df=ctx.power_df,
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
            ["baseline", "plateau"],
        )

    if ctx.all_features is not None:
        safe_plot(
            ctx,
            saved_files,
            "power_plateau_vs_baseline",
            "power",
            None,
            plot_power_plateau_vs_baseline,
            ctx.all_features,
            ctx.subject,
            ctx.subdir("power"),
            ctx.logger,
            ctx.config,
        )

    if ctx.all_features is not None and ctx.aligned_events is not None:
        safe_plot(
            ctx,
            saved_files,
            "power_temporal_evolution",
            "power",
            None,
            plot_temporal_evolution,
            ctx.all_features,
            ctx.aligned_events,
            ctx.subject,
            ctx.subdir("power"),
            ctx.logger,
            ctx.config,
            "power",
            "Band Power",
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

    safe_plot(
        ctx,
        saved_files,
        "power_trial_variability",
        "power",
        None,
        plot_power_trial_variability,
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
            "band_power_topomaps_plateau",
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
            segment="plateau",
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
# Summary / Quality
###################################################################



