from __future__ import annotations

import numpy as np

from eeg_pipeline.plotting.features.context import FeaturePlotContext, VisualizationRegistry
from eeg_pipeline.plotting.core.runner import safe_plot
from eeg_pipeline.io.paths import deriv_stats_path

from eeg_pipeline.plotting.features.phase import (
    plot_itpc_heatmap,
    plot_itpc_topomaps,
    plot_itpc_by_condition,
)
from eeg_pipeline.plotting.features.roi import (
    plot_itpc_by_roi_band_condition,
    plot_itpc_plateau_vs_baseline,
    plot_pac_by_roi_condition,
    plot_band_segment_condition,
)


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
