"""Connectivity behavioral correlations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any, TYPE_CHECKING
import logging

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from eeg_pipeline.analysis.behavior.core import BehaviorContext


def compute_sliding_state_metrics(
    subject: str,
    task: str,
    conn_df: Optional[pd.DataFrame],
    aligned_events: Optional[pd.DataFrame],
    deriv_root: Path,
    config: Any,
    logger: logging.Logger,
) -> None:
    """Compute sliding connectivity state metrics and correlate with behavior."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from eeg_pipeline.utils.io.general import deriv_stats_path, ensure_dir, write_tsv, get_column_from_config
    from eeg_pipeline.analysis.behavior.core import safe_correlation, build_correlation_record, save_correlation_results

    if conn_df is None or conn_df.empty or aligned_events is None or aligned_events.empty:
        return

    sw_cols = [c for c in conn_df.columns if c.startswith("sw") and "corr_all" in c]
    if not sw_cols:
        return

    metric_cols = [c for c in sw_cols if any(m in c for m in ["_geff", "_clust", "modularity"])]
    if not metric_cols:
        return

    X = conn_df[metric_cols].values
    valid_mask = ~np.any(np.isnan(X), axis=1)
    n_clusters = int(config.get("behavior_analysis.sliding_states.n_clusters", 3))

    if valid_mask.sum() < n_clusters * 2:
        logger.warning(f"Insufficient valid trials for state clustering ({valid_mask.sum()})")
        return

    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[valid_mask])
        kmeans = KMeans(n_clusters=n_clusters, random_state=config.get("project.random_state", 42), n_init=10)
        labels = kmeans.fit_predict(X_scaled)
    except Exception as e:
        logger.warning(f"State clustering failed: {e}")
        return

    rating_col = get_column_from_config(config, "event_columns.rating", aligned_events)
    if rating_col and rating_col in aligned_events.columns:
        rating = pd.to_numeric(aligned_events.loc[valid_mask, rating_col], errors="coerce")
        records = []
        for state_id in range(n_clusters):
            state_mask = (labels == state_id)
            if state_mask.sum() < 5:
                continue
            r, p, n = safe_correlation(state_mask.astype(float), rating.values, method="spearman", min_samples=5)
            if np.isfinite(r):
                records.append(build_correlation_record(
                    f"state_{state_id}", "connectivity", r, p, n, "spearman",
                    identifier_type="state", analysis_type="sliding_state"
                ))
        if records:
            stats_dir = deriv_stats_path(deriv_root, subject)
            ensure_dir(stats_dir)
            save_correlation_results(records, stats_dir / "corr_stats_sliding_states_vs_rating.tsv",
                                    apply_fdr=True, config=config, logger=logger)

    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(stats_dir)
    write_tsv(pd.DataFrame({"trial": range(len(labels)), "state": labels}),
              stats_dir / "connectivity_state_assignments.tsv")


def correlate_connectivity_roi_from_context(ctx: "BehaviorContext") -> None:
    """Correlate connectivity features with behavioral measures using core function."""
    from eeg_pipeline.analysis.behavior.core import (
        correlate_features_loop, save_correlation_results, get_min_samples,
    )
    from eeg_pipeline.utils.io.general import ensure_dir

    logger = ctx.logger
    config = ctx.config

    if ctx.connectivity_df is None or ctx.connectivity_df.empty:
        logger.debug("No connectivity features")
        return
    if ctx.targets is None:
        logger.debug("No target variable")
        return

    logger.info("Computing connectivity-behavior correlations...")
    conn_df = ctx.connectivity_df
    min_samples = get_min_samples(config, "roi")
    ensure_dir(ctx.stats_dir)

    # Graph-level correlations
    graph_cols = [c for c in conn_df.columns if any(m in c for m in ["_geff", "_clust", "_pc", "_smallworld"])]
    records = []
    if graph_cols:
        graph_df = conn_df[graph_cols]
        records, _ = correlate_features_loop(
            feature_df=graph_df,
            target_values=ctx.targets,
            method=ctx.method,
            min_samples=min_samples,
            logger=None,
            identifier_type="graph_metric",
            analysis_type="connectivity",
        )
    if records:
        save_correlation_results(records, ctx.stats_dir / f"corr_stats_conn_graph_vs_rating_{ctx.method}.tsv",
                                apply_fdr=True, config=config, logger=logger)
        logger.info(f"  Graph metrics: {len(records)} correlations")

    # Edge-level (sampled, moderate+ only)
    edge_cols = [c for c in conn_df.columns if c not in graph_cols and not c.startswith("sw")]
    max_edges = int(config.get("behavior_analysis.statistics.max_edges_to_correlate", 500))
    if len(edge_cols) > max_edges:
        rng = np.random.default_rng(config.get("project.random_state", 42))
        edge_cols = list(rng.choice(edge_cols, size=max_edges, replace=False))

    if edge_cols:
        edge_df = conn_df[edge_cols]
        edge_records, _ = correlate_features_loop(
            feature_df=edge_df,
            target_values=ctx.targets,
            method=ctx.method,
            min_samples=min_samples,
            logger=None,
            identifier_type="edge",
            analysis_type="connectivity",
        )
        # Filter to moderate+ effect sizes only
        edge_records = [r for r in edge_records if abs(r.correlation) > 0.2]
    else:
        edge_records = []
    if edge_records:
        save_correlation_results(edge_records, ctx.stats_dir / f"corr_stats_conn_edges_vs_rating_{ctx.method}.tsv",
                                apply_fdr=True, config=config, logger=logger)
        logger.info(f"  Edges: {len(edge_records)} correlations (|r| > 0.2)")

    # Temperature correlations
    if ctx.temperature is not None and len(ctx.temperature.dropna()) >= min_samples:
        temp_records = []
        if graph_cols:
            graph_df = conn_df[graph_cols]
            temp_records, _ = correlate_features_loop(
                feature_df=graph_df,
                target_values=ctx.temperature,
                method=ctx.method,
                min_samples=min_samples,
                logger=None,
                identifier_type="graph_metric",
                analysis_type="connectivity",
            )
        if temp_records:
            save_correlation_results(temp_records, ctx.stats_dir / f"corr_stats_conn_graph_vs_temp_{ctx.method}.tsv",
                                    apply_fdr=True, config=config, logger=logger)
            logger.info(f"  Graph vs temp: {len(temp_records)} correlations")
