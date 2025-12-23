from __future__ import annotations

from eeg_pipeline.plotting.behavioral.dose_response import visualize_dose_response
from eeg_pipeline.plotting.behavioral.registry import BehaviorPlotRegistry
from eeg_pipeline.plotting.behavioral.scatter.aperiodic import plot_aperiodic_roi_scatter
from eeg_pipeline.plotting.behavioral.scatter.connectivity import plot_connectivity_roi_scatter
from eeg_pipeline.plotting.behavioral.scatter.complexity import plot_complexity_roi_scatter
from eeg_pipeline.plotting.behavioral.scatter.itpc import plot_itpc_roi_scatter
from eeg_pipeline.plotting.behavioral.scatter.power import plot_power_roi_scatter
from eeg_pipeline.plotting.behavioral.scatter.psychometrics import plot_psychometrics
from eeg_pipeline.plotting.behavioral.scatter.summary import plot_top_behavioral_predictors
from eeg_pipeline.plotting.behavioral.temporal.clusters import plot_pain_nonpain_clusters
from eeg_pipeline.plotting.behavioral.temporal.topomaps import plot_temporal_correlation_topomaps_by_pain
from eeg_pipeline.plotting.behavioral.mediation_plots import plot_mediation_path_diagram
from eeg_pipeline.utils.analysis.stats.mediation import MediationResult


def _record_results(ctx, result) -> None:
    if isinstance(result, dict) and "all" in result:
        ctx.all_results.append(result)


@BehaviorPlotRegistry.register("psychometrics", name="psychometrics")
def run_psychometrics(ctx, saved_plots):
    plot_psychometrics(ctx.subject, ctx.deriv_root, ctx.task, ctx.config)
    saved_plots["psychometrics"] = ctx.plots_dir


@BehaviorPlotRegistry.register("scatter", name="power_roi_scatter")
def run_power_scatter(ctx, saved_plots):
    result = plot_power_roi_scatter(
        subject=ctx.subject,
        deriv_root=ctx.deriv_root,
        task=ctx.task,
        use_spearman=ctx.use_spearman,
        plots_dir=ctx.plots_dir,
        config=ctx.config,
        rating_stats=ctx.rating_stats,
        temp_stats=ctx.temp_stats,
    )
    _record_results(ctx, result)
    saved_plots["power_roi_scatter"] = ctx.plots_dir


@BehaviorPlotRegistry.register("scatter", name="complexity_scatter")
def run_complexity_scatter(ctx, saved_plots):
    result = plot_complexity_roi_scatter(
        subject=ctx.subject,
        deriv_root=ctx.deriv_root,
        task=ctx.task,
        use_spearman=ctx.use_spearman,
        plots_dir=ctx.plots_dir,
        config=ctx.config,
    )
    _record_results(ctx, result)
    saved_plots["complexity_scatter"] = ctx.plots_dir


@BehaviorPlotRegistry.register("scatter", name="aperiodic_scatter")
def run_aperiodic_scatter(ctx, saved_plots):
    result = plot_aperiodic_roi_scatter(
        subject=ctx.subject,
        deriv_root=ctx.deriv_root,
        task=ctx.task,
        use_spearman=ctx.use_spearman,
        plots_dir=ctx.plots_dir,
        config=ctx.config,
    )
    _record_results(ctx, result)
    saved_plots["aperiodic_scatter"] = ctx.plots_dir


@BehaviorPlotRegistry.register("scatter", name="connectivity_scatter")
def run_connectivity_scatter(ctx, saved_plots):
    result = plot_connectivity_roi_scatter(
        subject=ctx.subject,
        deriv_root=ctx.deriv_root,
        task=ctx.task,
        use_spearman=ctx.use_spearman,
        plots_dir=ctx.plots_dir,
        config=ctx.config,
    )
    _record_results(ctx, result)
    saved_plots["connectivity_scatter"] = ctx.plots_dir


@BehaviorPlotRegistry.register("scatter", name="itpc_scatter")
def run_itpc_scatter(ctx, saved_plots):
    result = plot_itpc_roi_scatter(
        subject=ctx.subject,
        deriv_root=ctx.deriv_root,
        task=ctx.task,
        use_spearman=ctx.use_spearman,
        plots_dir=ctx.plots_dir,
        config=ctx.config,
    )
    _record_results(ctx, result)
    saved_plots["itpc_scatter"] = ctx.plots_dir


@BehaviorPlotRegistry.register("temporal", name="temporal_topomaps")
def run_temporal_topomaps(ctx, saved_plots):
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


@BehaviorPlotRegistry.register("temporal", name="pain_clusters")
def run_pain_clusters(ctx, saved_plots):
    plot_pain_nonpain_clusters(
        subject=ctx.subject,
        stats_dir=ctx.stats_dir,
        plots_dir=ctx.plots_dir,
        config=ctx.config,
        logger=ctx.logger,
    )
    saved_plots["pain_clusters"] = ctx.plots_dir / "topomaps"


@BehaviorPlotRegistry.register("dose_response", name="dose_response")
def run_dose_response(ctx, saved_plots):
    results = visualize_dose_response(
        subject=ctx.subject,
        deriv_root=ctx.deriv_root,
        task=ctx.task,
        config=ctx.config,
        logger=ctx.logger,
    )
    saved_plots.update(results)


@BehaviorPlotRegistry.register("summary", name="top_predictors")
def run_top_predictors(ctx, saved_plots):
    plot_top_behavioral_predictors(
        subject=ctx.subject,
        task=ctx.task,
        plots_dir=ctx.plots_dir,
        config=ctx.config,
    )
    saved_plots["top_predictors"] = ctx.plots_dir


@BehaviorPlotRegistry.register("advanced", name="mediation")
def run_mediation_plots(ctx, saved_plots):
    import pandas as pd
    med_path = ctx.stats_dir / "mediation.tsv"
    if not med_path.exists():
        return
    
    try:
        df = pd.read_csv(med_path, sep="\t")
        if df.empty:
            return
            
        med_dir = ctx.plots_dir / "mediation"
        med_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        for _, row in df.iterrows():
            # Reconstruct MediationResult from row
            res = MediationResult(
                a=row.get("path_a", row.get("a", 0)),
                b=row.get("path_b", row.get("b", 0)),
                c=row.get("path_c_total", row.get("c", 0)),
                c_prime=row.get("path_c_prime_direct", row.get("c_prime", 0)),
                ab=row.get("indirect_effect_ab", row.get("ab", 0)),
                p_a=row.get("p_a", 1.0),
                p_b=row.get("p_b", 1.0),
                p_c=row.get("p_c", 1.0),
                p_c_prime=row.get("p_cp", row.get("p_c_prime", 1.0)),
                p_ab=row.get("sobel_p", row.get("p_ab", 1.0)),
                ci_ab_low=row.get("ci_ab_low", 0),
                ci_ab_high=row.get("ci_ab_high", 0),
                proportion_mediated=row.get("proportion_mediated", 0),
                n=int(row.get("n", 0)),
                x_label="Temperature",
                m_label=str(row.get("mediator", "Mediator")),
                y_label="Pain Rating"
            )
            results.append(res)
            
            # Save individual diagram
            safe_name = str(res.m_label).replace("/", "_").replace(" ", "_")
            plot_mediation_path_diagram(
                res, 
                med_dir / f"sub-{ctx.subject}_mediation_{safe_name}.png",
                config=ctx.config,
                logger=ctx.logger
            )
            
        saved_plots["mediation"] = med_dir
    except Exception as e:
        ctx.logger.warning(f"Failed to generate mediation plots: {e}")
