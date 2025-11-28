import logging
import pickle
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

from eeg_pipeline.utils.data.loading import (
    build_covariate_matrix,
    build_covariates_without_temp,
    extract_measure_prefixes,
    extract_node_names_from_prefix,
    build_summary_map_for_prefix,
)
from eeg_pipeline.utils.io.general import (
    deriv_stats_path,
    deriv_plots_path,
    ensure_dir,
    sanitize_label,
    get_column_from_config,
    get_pain_column_from_config,
    get_temperature_column_from_config,
    write_tsv,
)
from eeg_pipeline.utils.analysis.stats import (
    should_apply_fisher_transform,
    get_correlation_method,
    compute_correlation,
    compute_bootstrap_ci,
    compute_correlation_pvalue,
    compute_partial_correlation_for_roi_pair,
    compute_permutation_pvalues_for_roi_pair,
    compute_temp_correlation_for_roi_pair,
    build_correlation_matrices_for_prefix,
    compute_fdr_rejections_for_heatmap,
    joint_valid_mask,
    compute_fisher_transformed_mean,
    _safe_float,
    fdr_bh,
    perm_pval_simple,
)
from eeg_pipeline.analysis.behavior.core import save_correlation_results
from eeg_pipeline.utils.analysis.tfr import build_rois_from_info as _build_rois, get_summary_type
from eeg_pipeline.utils.io.general import build_partial_covars_string
from eeg_pipeline.analysis.behavior.core import build_correlation_record
from eeg_pipeline.analysis.behavior.core import MIN_SAMPLES_DEFAULT, MIN_SAMPLES_EDGE
from eeg_pipeline.utils.analysis.windowing import sliding_window_centers
from eeg_pipeline.plotting.features import (
    plot_sliding_state_centroids,
    plot_sliding_state_sequences,
    plot_sliding_state_occupancy_boxplot,
    plot_sliding_state_occupancy_ribbons,
    plot_sliding_state_lagged_correlation_surfaces,
    plot_edge_significance_circle_from_stats,
    plot_graph_metric_distributions,
)
from eeg_pipeline.utils.io.general import build_connectivity_heatmap_records


def _build_roi_pair_rating_record(
    measure_band: str,
    roi_i: str,
    roi_j: str,
    n_edges: int,
    correlation: float,
    p_value: float,
    n_eff: int,
    method: str,
    ci_low: float,
    ci_high: float,
    r_partial: float,
    p_partial: float,
    n_partial: int,
    covariates_df: Optional[pd.DataFrame],
    p_perm: float,
    p_partial_perm: float,
    n_perm: int,
) -> Dict[str, Any]:
    """Build correlation record for ROI pair connectivity analysis."""
    partial_covars_str = build_partial_covars_string(covariates_df)
    record = build_correlation_record(
        identifier=f"{roi_i}_{roi_j}",
        band=measure_band,
        r=correlation,
        p=p_value,
        n=n_eff,
        method=method,
        ci_low=ci_low,
        ci_high=ci_high,
        r_partial=r_partial,
        p_partial=p_partial,
        n_partial=n_partial,
        p_perm=p_perm,
        p_partial_perm=p_partial_perm,
        identifier_type="roi_pair",
        analysis_type="connectivity",
        measure_band=measure_band,
        roi_i=roi_i,
        roi_j=roi_j,
        summary_type=get_summary_type(roi_i, roi_j),
        n_edges=n_edges,
        partial_covars=partial_covars_str,
        n_perm=n_perm,
    )
    return record.to_dict()




def _extract_sliding_windows(conn_df: Optional[pd.DataFrame]) -> Tuple[Dict[int, List[str]], List[Tuple[str, str]]]:
    if conn_df is None or conn_df.empty:
        return {}, []
    window_cols: Dict[int, List[str]] = {}
    edge_pairs: List[Tuple[str, str]] = []
    for col in conn_df.columns:
        if not isinstance(col, str) or not col.startswith("sw"):
            continue
        tokens = col.split("corr_all__", 1)
        if len(tokens) != 2:
            continue
        win_part = tokens[0]
        try:
            win_idx = int(win_part.replace("sw", "").replace("corr_all", ""))
        except ValueError:
            continue
        window_cols.setdefault(win_idx, []).append(col)
    if window_cols:
        first_cols = window_cols[sorted(window_cols.keys())[0]]
        for col in first_cols:
            if "__" in col:
                _, pair = col.split("corr_all__", 1)
                parts = pair.split("__")
                if len(parts) == 2:
                    edge_pairs.append((parts[0], parts[1]))
    return window_cols, edge_pairs


def compute_sliding_state_metrics(
    *,
    subject: str,
    task: str,
    conn_df: Optional[pd.DataFrame],
    aligned_events: Optional[pd.DataFrame],
    deriv_root: Path,
    config: Any,
    logger: logging.Logger,
) -> None:
    window_cols, edge_pairs = _extract_sliding_windows(conn_df)
    if not window_cols or not edge_pairs:
        logger.info("No sliding connectivity windows found; skipping sliding state metrics")
        return

    n_windows = len(window_cols)
    n_edges = len(edge_pairs)
    window_indices = sorted(window_cols.keys())
    data_blocks = []
    sample_trial_idx: List[int] = []
    sample_win_idx: List[int] = []

    for win_idx in window_indices:
        cols = window_cols[win_idx]
        block = conn_df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
        data_blocks.append(block)
        sample_trial_idx.extend(list(range(block.shape[0])))
        sample_win_idx.extend([win_idx] * block.shape[0])

    samples = np.vstack(data_blocks)
    if samples.shape[0] < 2:
        logger.warning("Not enough samples for sliding state clustering; skipping")
        return

    n_clusters = int(config.get("behavior_analysis.sliding_states.n_clusters", 3))
    random_state = int(config.get("random.seed", 42))
    if samples.shape[0] < n_clusters:
        logger.warning("Samples (%d) fewer than clusters (%d); reducing clusters to samples", samples.shape[0], n_clusters)
        n_clusters = samples.shape[0]

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels_all = kmeans.fit_predict(samples)
    centroids = kmeans.cluster_centers_

    n_trials = conn_df.shape[0]
    state_matrix = np.full((n_trials, n_windows), -1, dtype=int)
    idx = 0
    for win_pos, win_idx in enumerate(window_indices):
        block = conn_df[window_cols[win_idx]]
        block_len = len(block)
        state_matrix[:block_len, win_pos] = labels_all[idx:idx + block_len]
        idx += block_len

    occupancy = []
    for row in state_matrix:
        valid = row[row >= 0]
        if valid.size == 0:
            occupancy.append([np.nan] * n_clusters)
        else:
            counts = np.bincount(valid, minlength=n_clusters).astype(float)
            occupancy.append((counts / counts.sum()).tolist())
    occupancy = np.asarray(occupancy)

    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(stats_dir)
    np.savez_compressed(
        stats_dir / "sliding_states.npz",
        state_matrix=state_matrix,
        occupancy=occupancy,
        centroids=centroids,
        window_indices=np.array(window_indices),
        edge_pairs=np.array(edge_pairs, dtype=object),
    )

    results = []
    temp_results = []
    rating_col = get_column_from_config(config, "event_columns.rating", aligned_events) if aligned_events is not None else None
    rating_vals = pd.to_numeric(aligned_events[rating_col], errors="coerce") if rating_col else None
    temp_col = get_column_from_config(config, "event_columns.temperature", aligned_events) if aligned_events is not None else None
    temp_vals = pd.to_numeric(aligned_events[temp_col], errors="coerce") if temp_col else None

    for state_idx in range(n_clusters):
        occ = occupancy[:, state_idx]
        valid_mask = np.isfinite(occ)
        if rating_vals is not None:
            r, p = spearmanr(occ[valid_mask], rating_vals[valid_mask], nan_policy="omit") if valid_mask.sum() > 2 else (np.nan, np.nan)
            results.append({"state": state_idx, "r": _safe_float(r), "p": _safe_float(p), "n": int(valid_mask.sum())})
        if temp_vals is not None:
            r, p = spearmanr(occ[valid_mask], temp_vals[valid_mask], nan_policy="omit") if valid_mask.sum() > 2 else (np.nan, np.nan)
            temp_results.append({"state": state_idx, "r": _safe_float(r), "p": _safe_float(p), "n": int(valid_mask.sum())})

    if results:
        df_r = pd.DataFrame(results)
        df_r.to_csv(stats_dir / "corr_stats_sliding_states_vs_rating.tsv", sep="\t", index=False)
    if temp_results:
        df_t = pd.DataFrame(temp_results)
        df_t.to_csv(stats_dir / "corr_stats_sliding_states_vs_temp.tsv", sep="\t", index=False)

    plots_dir = deriv_plots_path(deriv_root, subject, subdir="sliding_states")
    plot_sliding_state_sequences(state_matrix, window_indices, plots_dir, logger, config)
    plot_sliding_state_occupancy_boxplot(occupancy, aligned_events, plots_dir, logger, config)

    labels_path = deriv_root / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_connectivity_labels.npy"
    ch_names = None
    if labels_path.exists():
        try:
            ch_names = np.load(labels_path, allow_pickle=True).tolist()
        except (OSError, ValueError, pickle.UnpicklingError):
            ch_names = None
    plot_sliding_state_centroids(centroids, edge_pairs, ch_names, plots_dir, logger, config)
    all_centers = sliding_window_centers(config, max(window_indices) + 1)
    window_centers = np.array([all_centers[idx] for idx in window_indices], dtype=float)

    occupancy_mean = np.full((n_clusters, n_windows), np.nan, dtype=float)
    occupancy_sem = np.full((n_clusters, n_windows), np.nan, dtype=float)
    corr_rating_r = np.full((n_clusters, n_windows), np.nan, dtype=float)
    corr_rating_p = np.full((n_clusters, n_windows), np.nan, dtype=float)
    corr_temp_r = np.full((n_clusters, n_windows), np.nan, dtype=float)
    corr_temp_p = np.full((n_clusters, n_windows), np.nan, dtype=float)

    for w_idx, win_pos in enumerate(range(n_windows)):
        win_states = state_matrix[:, win_pos]
        valid_mask = win_states >= 0
        if not np.any(valid_mask):
            continue
        for s_idx in range(n_clusters):
            occ_vec = (win_states == s_idx).astype(float)
            occ_valid = occ_vec[valid_mask]
            n_valid = occ_valid.size
            if n_valid > 0:
                occupancy_mean[s_idx, w_idx] = np.nanmean(occ_valid)
                if n_valid > 1:
                    occupancy_sem[s_idx, w_idx] = np.nanstd(occ_valid, ddof=1) / np.sqrt(n_valid)
            if rating_vals is not None:
                r, p = spearmanr(occ_vec[valid_mask], rating_vals[valid_mask], nan_policy="omit") if n_valid > 2 else (np.nan, np.nan)
                corr_rating_r[s_idx, w_idx] = _safe_float(r)
                corr_rating_p[s_idx, w_idx] = _safe_float(p)
            if temp_vals is not None:
                r, p = spearmanr(occ_vec[valid_mask], temp_vals[valid_mask], nan_policy="omit") if n_valid > 2 else (np.nan, np.nan)
                corr_temp_r[s_idx, w_idx] = _safe_float(r)
                corr_temp_p[s_idx, w_idx] = _safe_float(p)

    np.savez_compressed(
        stats_dir / "sliding_state_dynamics.npz",
        occupancy_mean=occupancy_mean,
        occupancy_sem=occupancy_sem,
        window_centers=window_centers,
        corr_rating_r=corr_rating_r,
        corr_rating_p=corr_rating_p,
        corr_temp_r=corr_temp_r,
        corr_temp_p=corr_temp_p,
        window_indices=np.array(window_indices),
    )

    plot_sliding_state_occupancy_ribbons(
        occupancy_mean,
        occupancy_sem,
        window_centers,
        plots_dir,
        logger,
        config,
    )
    if np.isfinite(corr_rating_r).any():
        plot_sliding_state_lagged_correlation_surfaces(
            window_centers,
            corr_rating_r,
            corr_rating_p,
            plots_dir,
            logger,
            config,
            target_label="VAS",
        )
    if np.isfinite(corr_temp_r).any():
        plot_sliding_state_lagged_correlation_surfaces(
            window_centers,
            corr_temp_r,
            corr_temp_p,
            plots_dir,
            logger,
            config,
            target_label="Temperature",
        )


def _correlate_sliding_connectivity(
    conn_df: Optional[pd.DataFrame],
    ratings: Optional[pd.Series],
    config,
    stats_dir: Path,
    logger: logging.Logger,
    use_spearman: bool,
) -> None:
    if conn_df is None or conn_df.empty or ratings is None or ratings.empty:
        return

    window_labels = sorted(
        {int(m.group(1)) for col in conn_df.columns for m in [re.match(r"^sw(\d+)corr_all_", str(col))] if m}
    )
    if not window_labels:
        return

    centers = sliding_window_centers(config, max(window_labels) + 1)
    records: List[Dict[str, Any]] = []
    for win in window_labels:
        prefix = f"sw{win}corr_all_"
        win_cols = [c for c in conn_df.columns if str(c).startswith(prefix) and "__" in str(c)]
        if not win_cols:
            continue
        mean_conn = conn_df[win_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        corr_res = compute_correlation(mean_conn, ratings, use_spearman=use_spearman)
        if corr_res is None:
            continue
        r_val, p_val = corr_res
        valid_mask = pd.notna(mean_conn) & pd.notna(ratings)
        n_val = int(valid_mask.sum())
        records.append(
            {
                "window": int(win),
                "window_center": float(centers[win]) if win < len(centers) else np.nan,
                "r": float(r_val),
                "p": float(p_val),
                "n": int(n_val),
                "method": "spearman" if use_spearman else "pearson",
            }
        )

    if not records:
        return

    df = pd.DataFrame(records)
    save_correlation_results(
        df,
        stats_dir / "corr_stats_sliding_conn_vs_rating.tsv",
        apply_fdr=True,
        config=config,
        logger=logger,
        use_permutation_p=False,
        add_fdr_reject=True,
    )


def correlate_connectivity_roi_summaries(
    subject: str,
    task: Optional[str] = None,
    use_spearman: bool = True,
    partial_covars: Optional[List[str]] = None,
    bootstrap: int = 0,
    n_perm: int = 0,
    rng: Optional[np.random.Generator] = None
) -> None:
    from eeg_pipeline.pipelines.behavior import initialize_analysis_context
    from eeg_pipeline.analysis.behavior.core import BehaviorContext
    
    try:
        config, task, deriv_root, stats_dir, logger = initialize_analysis_context(
            subject, task, None
        )
    except ValueError:
        return

    rng_seed = config.get("random.seed", 42)
    rng = rng or np.random.default_rng(rng_seed)

    ctx = BehaviorContext(
        subject=subject,
        task=task,
        config=config,
        logger=logger,
        deriv_root=deriv_root,
        stats_dir=stats_dir,
        use_spearman=use_spearman,
        bootstrap=bootstrap,
        n_perm=n_perm,
        rng=rng,
        partial_covars=partial_covars,
    )
    
    if not ctx.load_data():
        return
    
    X = ctx.connectivity_df
    y = ctx.targets
    if X is None or X.empty or y is None:
        return
    
    info = ctx.epochs_info
    roi_map = _build_rois(info, config=config) if info else {}
    
    temp_series = ctx.temperature
    temp_col = ctx.temperature_column
    covariates_df = ctx.covariates_df
    covariates_without_temp_df = ctx.covariates_without_temp_df

    prefixes = extract_measure_prefixes(X.columns)
    
    for prefix in prefixes:
        prefix_columns = [c for c in X.columns if c.startswith(prefix + "_")]
        if not prefix_columns:
            continue
        
        summary_map = build_summary_map_for_prefix(
            prefix, prefix_columns, roi_map
        )
        if not summary_map:
            continue
        
        apply_fisher_transform = should_apply_fisher_transform(prefix)

        recs: List[Dict[str, object]] = []
        recs_temp: List[Dict[str, object]] = []
        method = get_correlation_method(use_spearman)
        
        for (roi_i, roi_j), cols_list in summary_map.items():
            edge_df = X[cols_list].apply(pd.to_numeric, errors="coerce")
            xi = (
                compute_fisher_transformed_mean(edge_df)
                if apply_fisher_transform
                else edge_df.mean(axis=1)
            )
            
            mask = joint_valid_mask(xi, y)
            n_eff = int(mask.sum())
            min_samples_roi = config.get("behavior_analysis.statistics.min_samples_roi", 5)
            if n_eff < min_samples_roi:
                continue
            
            xi_masked = xi.iloc[mask]
            y_masked = y.iloc[mask]

            edge_rs: List[float] = []
            for edge_col in cols_list:
                edge_series = pd.to_numeric(edge_df[edge_col], errors="coerce")
                edge_mask = joint_valid_mask(edge_series, y)
                if edge_mask.sum() < min_samples_roi:
                    continue
                r_edge, _ = compute_correlation(edge_series[edge_mask], y[edge_mask], use_spearman)
                if np.isfinite(r_edge):
                    edge_rs.append(float(r_edge))

            if edge_rs:
                edge_rs_arr = np.asarray(edge_rs, dtype=float)
                edge_rs_arr = np.clip(edge_rs_arr, -0.999999, 0.999999)
                z_mean = np.nanmean(np.arctanh(edge_rs_arr))
                correlation = float(np.tanh(z_mean))
                n_edge_boot = int(config.get("behavior_analysis.statistics.edge_bootstrap_n", 1000))
                if n_edge_boot > 0 and len(edge_rs_arr) > 1:
                    boot_stats = []
                    for _ in range(n_edge_boot):
                        resample = rng.choice(edge_rs_arr, size=edge_rs_arr.shape[0], replace=True)
                        boot_stats.append(np.nanmean(np.arctanh(resample)))
                    boot_stats = np.asarray(boot_stats, dtype=float)
                    if np.isfinite(z_mean):
                        p_value = float((np.sum(np.abs(boot_stats) >= np.abs(z_mean)) + 1) / (len(boot_stats) + 1))
                    else:
                        p_value = np.nan
                else:
                    p_value = compute_correlation_pvalue(edge_rs.tolist(), config=config)
            else:
                correlation, p_value = compute_correlation(xi_masked, y_masked, use_spearman)
            
            r_partial, p_partial, n_partial = compute_partial_correlation_for_roi_pair(
                xi_masked, y_masked, covariates_df, mask, method
            )

            ci_low, ci_high = compute_bootstrap_ci(
                xi_masked, y_masked, bootstrap, use_spearman, rng,
                min_samples_roi, logger=logger, config=config
            )

            p_perm, p_partial_perm = compute_permutation_pvalues_for_roi_pair(
                xi_masked, y_masked, covariates_df, mask, method, n_perm, n_eff, rng,
            )

            rating_record = _build_roi_pair_rating_record(
                prefix,
                roi_i,
                roi_j,
                len(cols_list),
                correlation,
                p_value,
                n_eff,
                method,
                ci_low,
                ci_high,
                r_partial,
                p_partial,
                n_partial,
                covariates_df,
                p_perm,
                p_partial_perm,
                n_perm,
            )
            recs.append(rating_record)

            if temp_series is not None and not temp_series.empty:
                temp_record = compute_temp_correlation_for_roi_pair(
                    xi, temp_series, covariates_without_temp_df,
                    bootstrap, n_perm, use_spearman, prefix,
                    roi_i, roi_j, len(cols_list), rng, logger, config=config,
                )
                if temp_record is not None:
                    recs_temp.append(temp_record)

        if recs:
            df = pd.DataFrame(recs)
            save_correlation_results(
                df,
                stats_dir / f"corr_stats_conn_roi_summary_{sanitize_label(prefix)}_vs_rating.tsv",
                apply_fdr=True,
                config=config,
                logger=logger,
                use_permutation_p=True,
                add_fdr_reject=True,
            )

        if recs_temp:
            df_t = pd.DataFrame(recs_temp)
            save_correlation_results(
                df_t,
                stats_dir / f"corr_stats_conn_roi_summary_{sanitize_label(prefix)}_vs_temp.tsv",
                apply_fdr=True,
                config=config,
                logger=logger,
                use_permutation_p=False,
                add_fdr_reject=True,
            )


def correlate_connectivity_heatmaps(subject: str, task: Optional[str] = None, use_spearman: bool = True) -> None:
    from eeg_pipeline.pipelines.behavior import initialize_analysis_context
    
    try:
        config, task, deriv_root, stats_dir, logger = initialize_analysis_context(
            subject, task, None
        )
    except ValueError:
        return
    
    logger.info(f"Starting connectivity correlation analysis for sub-{subject}")
    plot_subdir = config.get("plotting.behavioral.plot_subdir", "04_behavior_correlations")
    plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)
    ensure_dir(plots_dir)

    from eeg_pipeline.analysis.behavior.core import BehaviorContext
    
    ctx = BehaviorContext(
        subject=subject,
        task=task,
        config=config,
        logger=logger,
        deriv_root=deriv_root,
        stats_dir=stats_dir,
        use_spearman=use_spearman,
    )
    
    if not ctx.load_data():
        logger.warning(
            f"Failed to load data for sub-{subject}; "
            f"skipping connectivity correlations."
        )
        return
    
    connectivity_dataframe = ctx.connectivity_df
    target_values = ctx.targets
    if connectivity_dataframe is None or connectivity_dataframe.empty or target_values is None:
        logger.warning(
            f"Connectivity features or targets missing for sub-{subject}; "
            f"skipping connectivity correlations."
        )
        return

    column_names = list(connectivity_dataframe.columns)
    measure_prefixes = sorted({"_".join(col.split("_")[:2]) for col in column_names})

    for prefix in measure_prefixes:
        _process_connectivity_prefix(
            prefix,
            column_names,
            connectivity_dataframe,
            target_values,
            use_spearman,
            stats_dir,
            config,
            logger,
        )

    _correlate_connectivity_graph_metrics(
        connectivity_dataframe,
        target_values,
        stats_dir,
        config,
        logger,
        use_spearman,
    )
    
    try:
        compute_connectivity_condition_effects(
            subject=subject,
            task=task,
            deriv_root=deriv_root,
            config=config,
            logger=logger,
            connectivity_df=connectivity_dataframe,
            plots_dir=plots_dir,
        )
    except (ValueError, KeyError, RuntimeError) as exc:
        logger.warning("Connectivity condition effects failed for sub-%s: %s", subject, exc)


def _process_connectivity_prefix(
    prefix: str,
    column_names: List[str],
    connectivity_dataframe: pd.DataFrame,
    target_values: pd.Series,
    use_spearman: bool,
    stats_dir: Path,
    config: Any,
    logger: Any,
) -> None:
    prefix_columns = [col for col in column_names if col.startswith(prefix + "_")]
    if not prefix_columns:
        return
    
    node_info = extract_node_names_from_prefix(prefix, prefix_columns)
    if node_info is None:
        logger.warning(f"Could not infer nodes for {prefix}; skipping heatmap.")
        return
    
    node_names, node_to_index = node_info
    
    correlation_matrix, p_value_matrix = build_correlation_matrices_for_prefix(
        prefix,
        prefix_columns,
        connectivity_dataframe,
        target_values,
        node_to_index,
        use_spearman,
    )
    
    rejection_map, critical_value = compute_fdr_rejections_for_heatmap(
        p_value_matrix, len(node_names), config
    )

    records = build_connectivity_heatmap_records(
        len(node_names),
        node_names,
        correlation_matrix,
        p_value_matrix,
        rejection_map,
        critical_value,
    )
    
    if records:
        results_dataframe = pd.DataFrame(records)
        output_path = stats_dir / f"corr_stats_edges_{sanitize_label(prefix)}_vs_rating.tsv"
        write_tsv(results_dataframe, output_path)
        logger.info(f"Saved connectivity heatmap correlations for {prefix}")


def _correlate_connectivity_graph_metrics(
    connectivity_dataframe: pd.DataFrame,
    target_values: pd.Series,
    stats_dir: Path,
    config: Any,
    logger: Any,
    use_spearman: bool,
) -> None:
    metric_cols = [c for c in connectivity_dataframe.columns if "_deg_" in c or "modularity" in c]
    if not metric_cols:
        return

    stats_cfg = config.get("behavior_analysis", {}).get("statistics", {})
    n_perm = int(stats_cfg.get("n_permutations", 100))
    rng_seed = config.get("random.seed", 42)
    rng = np.random.default_rng(rng_seed)
    method = "spearman" if use_spearman else "pearson"

    records: List[Dict[str, Any]] = []
    for col in metric_cols:
        vals = pd.to_numeric(connectivity_dataframe[col], errors="coerce")
        corr_result = compute_correlation(vals, target_values, use_spearman=use_spearman)
        if corr_result is None:
            continue
        r, p = corr_result
        valid_mask = pd.notna(vals) & pd.notna(target_values)
        n = int(valid_mask.sum())
        
        vals_aligned = vals[valid_mask]
        target_aligned = target_values[valid_mask]
        
        p_perm = perm_pval_simple(
            vals_aligned,
            target_aligned,
            method,
            n_perm,
            rng,
            config=config,
        ) if n_perm > 0 else np.nan
        
        records.append(
            {
                "predictor": col,
                "r": r,
                "p": p,
                "p_perm": p_perm,
                "n": n,
                "method": method,
            }
        )

    if not records:
        return

    df = pd.DataFrame(records)
    save_correlation_results(
        df,
        stats_dir / "corr_stats_conn_graph_vs_rating.tsv",
        apply_fdr=True,
        config=config,
        logger=logger,
        use_permutation_p=True,
        add_fdr_reject=True,
    )


def _parse_edge_columns(prefix_cols: List[str], prefix: str) -> List[Tuple[str, str, str]]:
    parsed = []
    for col in prefix_cols:
        if not col.startswith(prefix + "_"):
            continue
        remainder = col[len(prefix) + 1 :]
        if "__" in remainder:
            ch1, ch2 = remainder.split("__", 1)
            parsed.append((col, ch1, ch2))
    return parsed


def compute_connectivity_condition_effects(
    *,
    subject: str,
    task: str,
    deriv_root: Path,
    config: Any,
    logger: logging.Logger,
    connectivity_df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    if connectivity_df is None or connectivity_df.empty:
        return

    epochs, events = load_epochs_for_analysis(
        subject,
        task,
        align="strict",
        preload=False,
        deriv_root=deriv_root,
        bids_root=config.bids_root,
        config=config,
        logger=logger,
    )
    if events is None or events.empty:
        logger.warning("No aligned events found; skipping connectivity condition effects")
        return

    pain_col = get_pain_column_from_config(config, events)
    temp_col = get_temperature_column_from_config(config, events)
    rating_col = get_column_from_config(config, "event_columns.rating", events)

    pain_vals = pd.to_numeric(events[pain_col], errors="coerce") if pain_col and pain_col in events.columns else None
    temp_vals = pd.to_numeric(events[temp_col], errors="coerce") if temp_col and temp_col in events.columns else None
    rating_vals = pd.to_numeric(events[rating_col], errors="coerce") if rating_col and rating_col in events.columns else None

    column_names = list(connectivity_df.columns)
    measure_prefixes = sorted({"_".join(col.split("_")[:2]) for col in column_names})
    alpha = config.get("behavior_analysis.statistics.fdr_alpha", 0.05)
    stats_dir = deriv_stats_path(deriv_root, subject)

    for prefix in measure_prefixes:
        prefix_cols = [c for c in column_names if c.startswith(prefix + "_") and "__" in c]
        edges = _parse_edge_columns(prefix_cols, prefix)
        if not edges:
            continue

        stats_records = []
        temp_records = []
        for col, ch1, ch2 in edges:
            vals = pd.to_numeric(connectivity_df[col], errors="coerce").to_numpy(dtype=float)
            valid_mask = np.isfinite(vals)
            if valid_mask.sum() < MIN_SAMPLES_DEFAULT:
                continue

            if pain_vals is not None:
                pain_mask = (pain_vals == 1) & valid_mask
                nonpain_mask = (pain_vals == 0) & valid_mask
                if pain_mask.sum() >= MIN_SAMPLES_DEFAULT and nonpain_mask.sum() >= MIN_SAMPLES_DEFAULT:
                    try:
                        _, p_val = stats.mannwhitneyu(vals[nonpain_mask], vals[pain_mask], alternative="two-sided")
                    except ValueError:
                        p_val = np.nan
                    effect = float(np.nanmean(vals[pain_mask]) - np.nanmean(vals[nonpain_mask]))
                    stats_records.append(
                        {
                            "edge": f"{ch1}__{ch2}",
                            "p": p_val,
                            "effect": effect,
                            "n_nonpain": int(nonpain_mask.sum()),
                            "n_pain": int(pain_mask.sum()),
                        }
                    )

            if temp_vals is not None and temp_vals.notna().any():
                try:
                    r_temp, p_temp = spearmanr(vals[valid_mask], temp_vals[valid_mask], nan_policy="omit")
                except (ValueError, RuntimeWarning, FloatingPointError):
                    r_temp, p_temp = np.nan, np.nan
                temp_records.append(
                    {
                        "edge": f"{ch1}__{ch2}",
                        "r": _safe_float(r_temp),
                        "p": _safe_float(p_temp),
                        "n": int(valid_mask.sum()),
                    }
                )

        from eeg_pipeline.analysis.behavior.core import save_correlation_results
        
        if stats_records:
            df_pain = pd.DataFrame(stats_records)
            out_path = stats_dir / f"corr_stats_edges_{sanitize_label(prefix)}_pain.tsv"
            save_correlation_results(df_pain, out_path, apply_fdr=True, config=config, logger=logger)
            sig_edges = set(df_pain.loc[df_pain["q"] < alpha, "edge"]) if "q" in df_pain.columns else set()
            plot_edge_significance_circle_from_stats(df_pain, prefix, plots_dir, logger, config, sig_edges)

        if temp_records:
            df_temp = pd.DataFrame(temp_records)
            out_temp = stats_dir / f"corr_stats_edges_{sanitize_label(prefix)}_temp.tsv"
            save_correlation_results(df_temp, out_temp, apply_fdr=True, config=config, logger=logger)

    plot_graph_metric_distributions(connectivity_df, events, plots_dir, logger, config)


def correlate_connectivity_roi_from_context(ctx: "BehaviorContext") -> None:
    """Compute connectivity ROI correlations using pre-loaded data from context."""
    subject = ctx.subject
    task = ctx.task
    logger = ctx.logger
    config = ctx.config
    deriv_root = ctx.deriv_root
    stats_dir = ctx.stats_dir
    
    logger.info(f"Computing connectivity ROI correlations for sub-{subject}")
    
    X = ctx.connectivity_df
    y = ctx.targets
    
    if X is None or X.empty or y is None:
        logger.warning("No connectivity features or targets in context")
        return
    
    info = ctx.epochs_info
    roi_map = _build_rois(info, config=config) if info else {}
    
    temp_series = ctx.temperature
    temp_col = ctx.temperature_column
    covariates_df = ctx.covariates_df
    covariates_without_temp_df = ctx.covariates_without_temp_df
    
    prefixes = extract_measure_prefixes(X.columns)
    
    for prefix in prefixes:
        prefix_columns = [c for c in X.columns if c.startswith(prefix + "_")]
        if not prefix_columns:
            continue
        
        summary_map = build_summary_map_for_prefix(prefix, prefix_columns, roi_map)
        if not summary_map:
            continue
        
        apply_fisher_transform = should_apply_fisher_transform(prefix)
        
        recs: List[Dict[str, object]] = []
        recs_temp: List[Dict[str, object]] = []
        method = get_correlation_method(ctx.use_spearman)
        
        for (roi_i, roi_j), cols_list in summary_map.items():
            edge_df = X[cols_list].apply(pd.to_numeric, errors="coerce")
            xi = (
                compute_fisher_transformed_mean(edge_df)
                if apply_fisher_transform
                else edge_df.mean(axis=1)
            )
            
            mask = joint_valid_mask(xi, y)
            n_eff = int(mask.sum())
            min_samples_roi = config.get("behavior_analysis.statistics.min_samples_roi", 5)
            if n_eff < min_samples_roi:
                continue
            
            xi_masked = xi.iloc[mask]
            y_masked = y.iloc[mask]
            
            edge_rs: List[float] = []
            for edge_col in cols_list:
                edge_series = pd.to_numeric(edge_df[edge_col], errors="coerce")
                edge_mask = joint_valid_mask(edge_series, y)
                if edge_mask.sum() < min_samples_roi:
                    continue
                r_edge, _ = compute_correlation(edge_series[edge_mask], y[edge_mask], ctx.use_spearman)
                if np.isfinite(r_edge):
                    edge_rs.append(float(r_edge))
            
            if edge_rs:
                edge_rs_arr = np.asarray(edge_rs, dtype=float)
                edge_rs_arr = np.clip(edge_rs_arr, -0.999999, 0.999999)
                z_mean = np.nanmean(np.arctanh(edge_rs_arr))
                correlation = float(np.tanh(z_mean))
            else:
                correlation, _ = compute_correlation(xi_masked, y_masked, ctx.use_spearman)
            
            p_value = compute_correlation_pvalue(correlation, n_eff, ctx.use_spearman)
            
            ci_low, ci_high = (np.nan, np.nan)
            if ctx.bootstrap > 0:
                ci_low, ci_high = compute_bootstrap_ci(xi_masked, y_masked, ctx.bootstrap, ctx.use_spearman)
            
            r_partial, p_partial, n_partial = compute_partial_correlation_for_roi_pair(
                xi_masked, y_masked, covariates_df, ctx.use_spearman, min_samples_roi
            )
            
            p_perm, p_partial_perm = compute_permutation_pvalues_for_roi_pair(
                xi_masked, y_masked, covariates_df, ctx.use_spearman, min_samples_roi, ctx.n_perm, ctx.rng
            )
            
            r_temp, p_temp, n_temp, p_temp_perm = compute_temp_correlation_for_roi_pair(
                xi_masked, y_masked, temp_series, covariates_without_temp_df, ctx.use_spearman, min_samples_roi, ctx.n_perm, ctx.rng
            )
            
            rec = _build_roi_pair_rating_record(
                prefix, roi_i, roi_j, len(cols_list), correlation, p_value, n_eff, method,
                ci_low, ci_high, r_partial, p_partial, n_partial, r_temp, p_temp, n_temp,
                p_perm, p_partial_perm, p_temp_perm, ctx.n_perm,
            )
            recs.append(rec)
            
            if temp_series is not None:
                rec_temp = _build_roi_pair_temp_record(
                    prefix, roi_i, roi_j, len(cols_list), xi_masked, temp_series, 
                    covariates_without_temp_df, ctx.use_spearman, min_samples_roi
                )
                if rec_temp:
                    recs_temp.append(rec_temp)
        
        from eeg_pipeline.analysis.behavior.core import save_correlation_results
        
        if recs:
            df_rating = pd.DataFrame(recs)
            out_path = stats_dir / f"corr_stats_conn_roi_summary_{prefix}_vs_rating.tsv"
            save_correlation_results(df_rating, out_path, apply_fdr=True, config=config, logger=logger)
            logger.info(f"Saved {len(recs)} ROI pair records for {prefix}")
        
        if recs_temp:
            df_temp = pd.DataFrame(recs_temp)
            out_temp = stats_dir / f"corr_stats_conn_roi_summary_{prefix}_vs_temp.tsv"
            save_correlation_results(df_temp, out_temp, apply_fdr=True, config=config, logger=logger)

