from __future__ import annotations

from eeg_pipeline.plotting.features.context import FeaturePlotContext, VisualizationRegistry
from eeg_pipeline.plotting.core.runner import safe_plot

from eeg_pipeline.plotting.features.dynamics import plot_autocorrelation_decay


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
