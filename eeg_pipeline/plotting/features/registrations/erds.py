from __future__ import annotations

from eeg_pipeline.plotting.features.context import FeaturePlotContext, VisualizationRegistry
from eeg_pipeline.plotting.core.runner import safe_plot

from eeg_pipeline.plotting.features.erds import plot_erds_temporal_evolution


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
