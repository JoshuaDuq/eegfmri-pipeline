from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from eeg_pipeline.utils.config.loader import get_config_bool, get_config_float, get_config_value


def stage_temporal_tfr_impl(ctx: Any) -> Optional[Dict[str, Any]]:
    """Compute time-frequency representation correlations."""
    from eeg_pipeline.analysis.behavior.api import compute_time_frequency_from_context

    ctx.logger.info("Computing time-frequency correlations...")
    return compute_time_frequency_from_context(ctx)


def normalize_temporal_feature_name(name: str) -> Optional[str]:
    key = str(name).strip().lower()
    alias_map = {
        "power": "power",
        "spectral": "power",
        "temporal": "power",
        "time_frequency": "power",
        "timefrequency": "power",
        "itpc": "itpc",
        "erds": "erds",
    }
    return alias_map.get(key, None)


def resolve_temporal_feature_selection_impl(
    ctx: Any,
    selected_features: Optional[List[str]] = None,
) -> List[str]:
    """Resolve effective temporal features from config toggles and user filters."""
    default_cfg = {"power": True, "itpc": False, "erds": False}
    cfg_raw = get_config_value(ctx.config, "behavior_analysis.temporal.features", {}) or {}
    cfg_enabled: Dict[str, bool] = {
        feature: bool((cfg_raw or {}).get(feature, default_value))
        for feature, default_value in default_cfg.items()
    }

    raw_filters: List[str] = []
    if selected_features is not None:
        raw_filters.extend([str(x) for x in selected_features if str(x).strip()])
    else:
        if ctx.selected_feature_files:
            raw_filters.extend([str(x) for x in ctx.selected_feature_files if str(x).strip()])
        if ctx.feature_categories:
            raw_filters.extend([str(x) for x in ctx.feature_categories if str(x).strip()])
        if ctx.computation_features and "temporal" in ctx.computation_features:
            raw_filters.extend([str(x) for x in (ctx.computation_features.get("temporal") or []) if str(x).strip()])

    explicit_filter = False
    requested: set[str] = set()
    for item in raw_filters:
        item_norm = str(item).strip().lower()
        if not item_norm:
            continue
        if item_norm == "all":
            explicit_filter = True
            requested.update(default_cfg.keys())
            continue
        normalized = normalize_temporal_feature_name(item_norm)
        if normalized is not None:
            explicit_filter = True
            requested.add(normalized)

    if not explicit_filter:
        requested = set(default_cfg.keys())

    enabled = [feat for feat in ["power", "itpc", "erds"] if cfg_enabled.get(feat, False) and feat in requested]
    return enabled


def stage_temporal_stats_impl(
    ctx: Any,
    selected_features: Optional[List[str]] = None,
    *,
    resolve_temporal_feature_selection_fn: Callable[[Any, Optional[List[str]]], List[str]],
    sanitize_permutation_groups_fn: Callable[[Any, Any, str], Any],
    get_stats_subfolder_fn: Callable[[Any, str], Any],
    write_stats_table_fn: Callable[[Any, pd.DataFrame, Any], Any],
    format_correlation_method_label_fn: Callable[[str, Optional[str]], str],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Compute temporal statistics (power, ITPC, ERDS correlations)."""
    from eeg_pipeline.analysis.behavior.api import compute_temporal_from_context
    from eeg_pipeline.utils.analysis.stats.temporal import compute_itpc_temporal_from_context

    selected_temporal_features = resolve_temporal_feature_selection_fn(ctx, selected_features)

    correction_method = str(get_config_value(ctx.config, "behavior_analysis.temporal.correction_method", "fdr")).strip().lower()
    fdr_alpha = get_config_float(ctx.config, "behavior_analysis.statistics.fdr_alpha", 0.05)
    allow_iid_trials = get_config_bool(ctx.config, "behavior_analysis.statistics.allow_iid_trials", False)
    if not allow_iid_trials:
        if correction_method != "cluster":
            raise ValueError(
                "Temporal trial-level inference is non-i.i.d by default. "
                "Set behavior_analysis.temporal.correction_method='cluster' for grouped permutation inference, "
                "or set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
            )
        run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
        events = getattr(ctx, "aligned_events", None)
        if not isinstance(events, pd.DataFrame) or events.empty or run_col not in events.columns:
            raise ValueError(
                "Temporal cluster inference requires grouped labels under non-i.i.d mode. "
                f"Missing run grouping column '{run_col}' in aligned events."
            )
        groups = sanitize_permutation_groups_fn(events[run_col].to_numpy(), ctx.logger, "Temporal")
        if groups is None:
            raise ValueError(
                "Temporal cluster inference requires valid grouped permutation labels "
                f"in '{run_col}' (at least 2 samples per group)."
            )

    ctx.logger.info(
        "Temporal: using %s correction (alpha=%.3f), features=%s",
        correction_method,
        fdr_alpha,
        selected_temporal_features if selected_temporal_features else [],
    )

    results: Dict[str, Optional[Dict[str, Any]]] = {"power": None, "itpc": None, "erds": None}

    if not selected_temporal_features:
        ctx.logger.warning(
            "Temporal: no enabled features after applying temporal feature toggles and selection filters; skipping."
        )
        return results

    if "power" in selected_temporal_features:
        ctx.logger.info("Computing temporal correlations by condition...")
        results["power"] = compute_temporal_from_context(ctx)
    else:
        ctx.logger.info("Temporal: skipping power feature by configuration/filter.")

    if "itpc" in selected_temporal_features:
        ctx.logger.info("Computing ITPC temporal correlations...")
        itpc_results = compute_itpc_temporal_from_context(ctx)
        results["itpc"] = itpc_results
        if itpc_results:
            ctx.logger.info(
                "ITPC temporal: %s tests, %s significant",
                itpc_results.get("n_tests", 0),
                itpc_results.get("n_sig_raw", 0),
            )

    if "erds" in selected_temporal_features:
        from eeg_pipeline.utils.analysis.stats.temporal import compute_erds_temporal_from_context

        ctx.logger.info("Computing ERDS temporal correlations...")
        erds_results = compute_erds_temporal_from_context(ctx)
        results["erds"] = erds_results
        if erds_results:
            ctx.logger.info(
                "ERDS temporal: %s tests, %s significant",
                erds_results.get("n_tests", 0),
                erds_results.get("n_sig_raw", 0),
            )

    all_temporal_records = []
    for res in results.values():
        if res and "records" in res:
            all_temporal_records.extend(res["records"])

    if all_temporal_records:
        out_dir = get_stats_subfolder_fn(ctx, "temporal_correlations")
        method_suffix = "_spearman" if ctx.use_spearman else "_pearson"

        df_temporal = pd.DataFrame(all_temporal_records)
        if "p" in df_temporal.columns and "p_raw" not in df_temporal.columns:
            df_temporal["p_raw"] = df_temporal["p"]

        if "p_raw" in df_temporal.columns:
            p_vals = pd.to_numeric(df_temporal["p_raw"], errors="coerce").fillna(1.0).to_numpy()
            effective_correction_method = correction_method

            if correction_method == "fdr":
                reject, p_corrected, _, _ = multipletests(p_vals, alpha=fdr_alpha, method="fdr_bh")
                df_temporal["p_fdr"] = p_corrected
                df_temporal["sig_fdr"] = reject
                df_temporal["p_primary"] = df_temporal["p_fdr"]
                n_sig = int(reject.sum())
                ctx.logger.info("Temporal FDR: %d/%d significant at alpha=%s", n_sig, len(p_vals), fdr_alpha)

            elif correction_method == "bonferroni":
                reject, p_corrected, _, _ = multipletests(p_vals, alpha=fdr_alpha, method="bonferroni")
                df_temporal["p_bonferroni"] = p_corrected
                df_temporal["sig_bonferroni"] = reject
                df_temporal["p_primary"] = df_temporal["p_bonferroni"]
                n_sig = int(reject.sum())
                ctx.logger.info("Temporal Bonferroni: %d/%d significant at alpha=%s", n_sig, len(p_vals), fdr_alpha)

            elif correction_method == "cluster":
                if "p_cluster" in df_temporal.columns:
                    p_cluster_series = pd.to_numeric(df_temporal["p_cluster"], errors="coerce")
                    p_cluster_vals = p_cluster_series.to_numpy(dtype=float)
                    p_raw_series = pd.to_numeric(df_temporal.get("p_raw", np.nan), errors="coerce")
                    expected_rows = p_raw_series.notna().to_numpy()
                    missing_cluster_rows = expected_rows & ~np.isfinite(p_cluster_vals)
                    if missing_cluster_rows.any():
                        raise ValueError(
                            "Temporal cluster correction requested, but cluster-corrected p-values were missing for "
                            f"{int(missing_cluster_rows.sum())} temporal tests. "
                            "Do not fallback to asymptotic/FDR under non-i.i.d settings."
                        )
                    df_temporal["p_primary"] = p_cluster_vals
                    if "cluster_significant" in df_temporal.columns:
                        df_temporal["sig_cluster"] = df_temporal["cluster_significant"].fillna(False).astype(bool)
                    else:
                        df_temporal["sig_cluster"] = np.isfinite(p_cluster_vals) & (p_cluster_vals < fdr_alpha)
                    n_sig = int(df_temporal["sig_cluster"].sum())
                    ctx.logger.info("Temporal cluster: %d/%d significant at alpha=%s", n_sig, len(p_cluster_vals), fdr_alpha)
                else:
                    raise ValueError(
                        "Temporal cluster correction requested but no p_cluster column is present in temporal outputs. "
                        "Refusing silent fallback to asymptotic/FDR inference."
                    )

            elif correction_method == "none":
                ctx.logger.warning("Temporal: no multiple comparison correction applied (use with caution)")
                df_temporal["sig_raw"] = p_vals < fdr_alpha
                df_temporal["p_primary"] = df_temporal["p_raw"]

            df_temporal["correction_method_requested"] = correction_method
            df_temporal["correction_method"] = effective_correction_method

        combined_path = out_dir / f"temporal_correlations{method_suffix}.parquet"
        write_stats_table_fn(ctx, df_temporal, combined_path)
        ctx.logger.info(
            "Saved combined temporal correlations: %d tests -> %s",
            len(all_temporal_records),
            combined_path.name,
        )

        normalized_records = []
        method = "spearman" if ctx.use_spearman else "pearson"
        method_label = format_correlation_method_label_fn(method, None)
        target_label = str(get_config_value(ctx.config, "behavior_analysis.temporal.target_column", "") or "").strip() or "rating"
        for _, row in df_temporal.iterrows():
            normalized_records.append(
                {
                    "analysis_type": "temporal_correlations",
                    "feature_id": row.get("channel", ""),
                    "feature_type": row.get("feature", "temporal"),
                    "target": target_label,
                    "method": method,
                    "robust_method": None,
                    "method_label": method_label,
                    "n": row.get("n", np.nan),
                    "r": row.get("r", np.nan),
                    "p_raw": row.get("p_raw", row.get("p", np.nan)),
                    "p_primary": row.get("p_primary", row.get("p_raw", row.get("p", np.nan))),
                    "p_fdr": row.get("p_fdr", np.nan),
                    "notes": f"band={row.get('band', '')}, time={row.get('time_start', '')}–{row.get('time_end', '')}s, condition={row.get('condition', '')}",
                }
            )

        if normalized_records:
            df_normalized = pd.DataFrame(normalized_records)
            normalized_path = out_dir / f"normalized_results{method_suffix}.parquet"
            write_stats_table_fn(ctx, df_normalized, normalized_path)
            ctx.logger.debug(
                "Temporal normalized results: %d records -> %s",
                len(normalized_records),
                normalized_path.name,
            )

    return results


def stage_cluster_impl(ctx: Any, config: Any) -> Dict[str, Any]:
    from eeg_pipeline.analysis.behavior.api import run_cluster_test_from_context

    ctx.logger.info("Running cluster permutation tests...")
    ctx.n_perm = config.n_permutations
    results = run_cluster_test_from_context(ctx)
    return results if results else {"status": "completed"}
