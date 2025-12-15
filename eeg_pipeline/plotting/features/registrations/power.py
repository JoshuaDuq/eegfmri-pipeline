from __future__ import annotations

from eeg_pipeline.plotting.features.context import FeaturePlotContext, VisualizationRegistry
from eeg_pipeline.plotting.core.runner import safe_plot
from eeg_pipeline.utils.analysis.tfr import compute_tfr_for_visualization
from eeg_pipeline.utils.config.loader import get_frequency_band_names

from eeg_pipeline.plotting.features.power import (
    plot_power_by_condition,
    plot_power_variability_comprehensive,
    plot_cross_frequency_power_correlation,
    plot_feature_stability_heatmap,
    plot_temporal_autocorrelation,
    plot_feature_redundancy_matrix,
    plot_band_power_topomaps,
    plot_spectral_slope_topomap,
    plot_power_trial_variability,
    plot_power_topomaps_from_df,
)
from eeg_pipeline.plotting.features.roi import (
    plot_power_by_roi_band_condition,
    plot_band_segment_condition,
    plot_power_plateau_vs_baseline,
    plot_temporal_evolution,
    plot_feature_correlation_heatmap,
)


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

    try:
        from eeg_pipeline.plotting.features.power import plot_power_spectral_density
        from eeg_pipeline.utils.analysis.tfr import compute_tfr

        tfr = compute_tfr(ctx.epochs, ctx.config, ctx.logger)
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
    except Exception as e:
        ctx.logger.debug(f"Could not generate PSD plot: {e}")


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

    if ctx.all_features is not None:
        safe_plot(
            ctx,
            saved_files,
            "feature_stability_heatmap",
            "summary",
            None,
            plot_feature_stability_heatmap,
            features_df=ctx.all_features,
            subject=ctx.subject,
            save_dir=ctx.subdir("summary"),
            logger=ctx.logger,
            config=ctx.config,
        )

        safe_plot(
            ctx,
            saved_files,
            "temporal_autocorrelation",
            "summary",
            None,
            plot_temporal_autocorrelation,
            features_df=ctx.all_features,
            subject=ctx.subject,
            save_dir=ctx.subdir("summary"),
            logger=ctx.logger,
            config=ctx.config,
        )

        safe_plot(
            ctx,
            saved_files,
            "feature_redundancy_matrix",
            "summary",
            None,
            plot_feature_redundancy_matrix,
            features_df=ctx.all_features,
            subject=ctx.subject,
            save_dir=ctx.subdir("summary"),
            logger=ctx.logger,
            config=ctx.config,
        )

        safe_plot(
            ctx,
            saved_files,
            "feature_correlation_heatmap",
            "summary",
            None,
            plot_feature_correlation_heatmap,
            ctx.all_features,
            ctx.subject,
            ctx.subdir("summary"),
            ctx.logger,
            ctx.config,
        )

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
