from __future__ import annotations

from eeg_pipeline.plotting.features.context import FeaturePlotContext, VisualizationRegistry
from eeg_pipeline.plotting.core.runner import safe_plot

from eeg_pipeline.plotting.features.aperiodic import plot_aperiodic_topomaps, plot_aperiodic_by_condition
from eeg_pipeline.plotting.features.roi import (
    plot_aperiodic_by_roi_condition,
    plot_band_segment_condition,
)


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
