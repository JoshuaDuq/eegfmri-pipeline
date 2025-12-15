from __future__ import annotations

from eeg_pipeline.plotting.features.context import FeaturePlotContext, VisualizationRegistry
from eeg_pipeline.plotting.core.runner import safe_plot

from eeg_pipeline.plotting.features.burst import plot_burst_summary_by_band, plot_dynamics_by_condition
from eeg_pipeline.plotting.features.roi import plot_band_segment_condition


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
