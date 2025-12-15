from __future__ import annotations

from eeg_pipeline.plotting.features.context import FeaturePlotContext, VisualizationRegistry
from eeg_pipeline.plotting.core.runner import safe_plot

from eeg_pipeline.plotting.features.phase import (
    plot_pac_summary,
    plot_pac_by_condition,
    plot_pac_comodulograms,
    plot_pac_time_ribbons,
)


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
