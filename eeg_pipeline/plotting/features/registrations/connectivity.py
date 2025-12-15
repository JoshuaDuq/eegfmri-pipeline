from __future__ import annotations

import re

from eeg_pipeline.plotting.features.context import FeaturePlotContext, VisualizationRegistry
from eeg_pipeline.plotting.core.runner import safe_plot
from eeg_pipeline.utils.analysis.windowing import sliding_window_centers
from eeg_pipeline.utils.config.loader import get_frequency_band_names, get_config_value

from eeg_pipeline.plotting.features.connectivity import (
    plot_connectivity_by_condition,
    plot_connectivity_circle_by_condition,
    plot_connectivity_heatmap,
    plot_connectivity_network,
    plot_sliding_connectivity_trajectories,
)
from eeg_pipeline.plotting.features.roi import (
    plot_connectivity_by_roi_band_condition,
    plot_band_segment_condition,
)


@VisualizationRegistry.register("connectivity")
def plot_connectivity_mne_suite(ctx: FeaturePlotContext, saved_files):
    if ctx.connectivity_df is None or ctx.epochs is None:
        return

    power_bands = get_frequency_band_names(ctx.config)
    conn_measures = get_config_value(ctx.config, "plotting.plots.features.connectivity.measures", ["wpli", "aec"])

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

    conn_measures = get_config_value(ctx.config, "plotting.plots.features.connectivity.measures", ["wpli", "aec"])
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
