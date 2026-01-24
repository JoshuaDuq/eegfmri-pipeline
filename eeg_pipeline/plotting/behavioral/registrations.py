from __future__ import annotations

from typing import Any

from eeg_pipeline.plotting.behavioral.dose_response import (
    visualize_dose_response,
    visualize_pain_probability,
)
from eeg_pipeline.plotting.behavioral.registry import BehaviorPlotContext, BehaviorPlotRegistry
from eeg_pipeline.plotting.behavioral.scatter.behavior_scatter import plot_behavior_scatter
from eeg_pipeline.plotting.behavioral.scatter.psychometrics import plot_psychometrics
from eeg_pipeline.plotting.behavioral.temporal.topomaps import plot_temporal_correlation_topomaps_by_pain
from eeg_pipeline.utils.config.loader import get_config_value


def _record_results(ctx: BehaviorPlotContext, result: Any) -> None:
    """Record plot results in context if they contain 'all' key."""
    if isinstance(result, dict) and "all" in result:
        ctx.all_results.append(result)


@BehaviorPlotRegistry.register("psychometrics", name="psychometrics")
def run_psychometrics(ctx: BehaviorPlotContext, saved_plots: dict[str, Any]) -> None:
    plot_psychometrics(ctx.subject, ctx.deriv_root, ctx.task, ctx.config)
    saved_plots["psychometrics"] = ctx.plots_dir


@BehaviorPlotRegistry.register("scatter", name="behavior_scatter")
def run_behavior_scatter(ctx: BehaviorPlotContext, saved_plots: dict[str, Any]) -> None:
    """Unified behavior scatter plot supporting multiple features, columns, and aggregation modes."""
    scatter_config = get_config_value(ctx.config, "plotting.plots.behavior.scatter", {})

    result = plot_behavior_scatter(
        subject=ctx.subject,
        deriv_root=ctx.deriv_root,
        task=ctx.task,
        features=scatter_config.get("features"),
        columns=scatter_config.get("columns"),
        aggregation_modes=scatter_config.get("aggregation_modes"),
        use_spearman=ctx.use_spearman,
        plots_dir=ctx.plots_dir,
        config=ctx.config,
    )
    _record_results(ctx, result)
    saved_plots["behavior_scatter"] = ctx.plots_dir


@BehaviorPlotRegistry.register("temporal", name="temporal_topomaps")
def run_temporal_topomaps(ctx: BehaviorPlotContext, saved_plots: dict[str, Any]) -> None:
    plot_temporal_correlation_topomaps_by_pain(
        subject=ctx.subject,
        task=ctx.task,
        plots_dir=ctx.plots_dir,
        stats_dir=ctx.stats_dir,
        config=ctx.config,
        logger=ctx.logger,
        use_spearman=ctx.use_spearman,
    )
    saved_plots["temporal_topomaps"] = ctx.plots_dir / "topomaps"


@BehaviorPlotRegistry.register("dose_response", name="dose_response")
def run_dose_response(ctx: BehaviorPlotContext, saved_plots: dict[str, Any]) -> None:
    result = visualize_dose_response(
        subject=ctx.subject,
        deriv_root=ctx.deriv_root,
        task=ctx.task,
        config=ctx.config,
        logger=ctx.logger,
    )
    saved_plots.update(result)


@BehaviorPlotRegistry.register("dose_response", name="pain_probability")
def run_pain_probability(ctx: BehaviorPlotContext, saved_plots: dict[str, Any]) -> None:
    result = visualize_pain_probability(
        subject=ctx.subject,
        deriv_root=ctx.deriv_root,
        task=ctx.task,
        config=ctx.config,
        logger=ctx.logger,
    )
    saved_plots.update(result)
