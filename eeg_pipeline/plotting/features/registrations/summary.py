from __future__ import annotations

from eeg_pipeline.plotting.features.context import FeaturePlotContext, VisualizationRegistry
from eeg_pipeline.plotting.core.runner import safe_plot


@VisualizationRegistry.register("summary")
def plot_missing_data(ctx: FeaturePlotContext, saved_files):
    if ctx.all_features is None:
        return

    try:
        from eeg_pipeline.plotting.features.quality import plot_missing_data_matrix
    except Exception:
        return

    summary_dir = ctx.subdir("summary")

    safe_plot(
        ctx,
        saved_files,
        "missing_data_matrix",
        "summary",
        None,
        plot_missing_data_matrix,
        ctx.all_features,
        summary_dir / f"sub-{ctx.subject}_missing_data_matrix",
        config=ctx.config,
    )
