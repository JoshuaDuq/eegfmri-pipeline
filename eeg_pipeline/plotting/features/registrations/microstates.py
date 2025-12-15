from __future__ import annotations

import numpy as np
import mne

from eeg_pipeline.plotting.features.context import FeaturePlotContext, VisualizationRegistry
from eeg_pipeline.plotting.core.runner import safe_plot
from eeg_pipeline.io.paths import deriv_stats_path

from eeg_pipeline.plotting.features.microstates import (
    plot_microstate_by_condition,
    plot_microstate_templates,
    plot_microstate_transition_matrix,
)
from eeg_pipeline.plotting.features.roi import plot_band_segment_condition


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
        picks = mne.pick_types(ctx.epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
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

    try:
        plot_microstate_by_condition(
            microstate_df=ctx.microstate_df,
            events_df=ctx.aligned_events,
            subject=ctx.subject,
            save_dir=ctx.subdir("microstates"),
            logger=ctx.logger,
            config=ctx.config,
        )
    except Exception as e:
        ctx.logger.warning(f"Failed microstate condition comparison: {e}")

    try:
        plot_microstate_transition_matrix(
            microstate_df=ctx.microstate_df,
            subject=ctx.subject,
            save_dir=ctx.subdir("microstates"),
            logger=ctx.logger,
            config=ctx.config,
        )
    except Exception as e:
        ctx.logger.warning(f"Failed microstate transition matrix: {e}")

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
