from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import get_config_bool, get_config_value


def resolve_condition_compare_column(df_trials: pd.DataFrame, config: Any) -> str:
    """
    Resolve the condition column from config.

    Resolution order:
    1. ``behavior_analysis.condition.compare_column`` (explicit override)
    2. ``event_columns.binary_outcome`` (discovered from column aliases)

    Raises ``ValueError`` if neither resolves to a column present in the trial table.
    """
    from eeg_pipeline.utils.data.columns import get_binary_outcome_column_from_config

    compare_col = str(get_config_value(config, "behavior_analysis.condition.compare_column", "") or "").strip()
    if compare_col and compare_col in df_trials.columns:
        return compare_col

    fallback_col = get_binary_outcome_column_from_config(config, df_trials)
    if fallback_col and fallback_col in df_trials.columns:
        return str(fallback_col)

    raise ValueError(
        "Could not resolve a condition comparison column. "
        "Set 'behavior_analysis.condition.compare_column' or configure "
        "'event_columns.binary_outcome' to match a column in the trial table. "
        f"Available columns: {sorted(df_trials.columns.tolist())}"
    )


def stage_condition_column_impl(
    ctx: Any,
    config: Any,
    df_trials: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
    *,
    stage_condition_multigroup_fn: Callable[..., pd.DataFrame],
    load_trial_table_df_fn: Callable[[Any], Optional[pd.DataFrame]],
    is_dataframe_valid_fn: Callable[[Optional[pd.DataFrame]], bool],
    get_feature_columns_fn: Callable[[pd.DataFrame, Any, Optional[str]], List[str]],
    check_early_exit_conditions_fn: Callable[..., tuple[bool, Optional[str]]],
    feature_suffix_from_context_fn: Callable[[Any], str],
    get_stats_subfolder_fn: Callable[[Any, str], Path],
    sanitize_permutation_groups_fn: Callable[[Any, Any, str], Any],
    compute_unified_fdr_fn: Callable[..., pd.DataFrame],
    write_parquet_with_optional_csv_fn: Callable[[pd.DataFrame, Path, bool], None],
    resolve_condition_compare_column_fn: Callable[[pd.DataFrame, Any], str],
    unified_fdr_family_columns: Sequence[str],
) -> pd.DataFrame:
    """Run column-based condition comparison."""
    from eeg_pipeline.analysis.behavior.api import compute_condition_effects, split_by_condition

    compare_values = get_config_value(ctx.config, "behavior_analysis.condition.compare_values", [])
    use_multigroup = isinstance(compare_values, (list, tuple)) and len(compare_values) > 2
    if use_multigroup:
        ctx.logger.info(
            "Condition column: %d values specified, delegating to multigroup comparison",
            len(compare_values),
        )
        return stage_condition_multigroup_fn(ctx, config, df_trials=df_trials, feature_cols=feature_cols)

    fail_fast = get_config_value(ctx.config, "behavior_analysis.condition.fail_fast", True)
    primary_unit = str(get_config_value(ctx.config, "behavior_analysis.condition.primary_unit", "trial")).strip().lower()
    allow_iid_trials = get_config_bool(ctx.config, "behavior_analysis.statistics.allow_iid_trials", False)
    perm_enabled = get_config_bool(ctx.config, "behavior_analysis.condition.permutation.enabled", False)
    if primary_unit in {"trial", "trialwise"} and not perm_enabled and not allow_iid_trials:
        raise ValueError(
            "Trial-level condition comparisons require a valid non-i.i.d inference method. "
            "Enable permutation testing (behavior_analysis.condition.permutation.enabled=true) "
            "or use run-level aggregation (behavior_analysis.condition.primary_unit=run_mean). "
            "Set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
        )

    use_run_unit = primary_unit in {"run", "run_mean", "runmean", "run_level"}
    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()

    if df_trials is None:
        df_trials = load_trial_table_df_fn(ctx)
    if not is_dataframe_valid_fn(df_trials):
        ctx.logger.warning("Condition column: trial table missing; skipping.")
        return pd.DataFrame()

    if feature_cols is None:
        feature_cols = get_feature_columns_fn(df_trials, ctx, "condition")

    compare_col = resolve_condition_compare_column_fn(df_trials, ctx.config)
    _ = get_config_bool(ctx.config, "behavior_analysis.condition.overwrite", True)
    if use_run_unit and run_col in df_trials.columns and compare_col in df_trials.columns:
        ctx.logger.info("Condition: aggregating to run×condition level (primary_unit=%s)", primary_unit)
        group_keys = [run_col, compare_col]
        df_agg = df_trials.groupby(group_keys, dropna=True)[feature_cols].mean(numeric_only=True).reset_index()
        cell_counts = df_trials.groupby(group_keys, dropna=True).size().rename("n_trials_cell").reset_index()
        df_trials = df_agg.merge(cell_counts, on=group_keys, how="left")
        ctx.logger.info("  Run×condition level: %d observations", len(df_trials))

    if not feature_cols:
        ctx.logger.info("Condition column: no feature columns found; skipping.")
        return pd.DataFrame()

    min_trials_required = 2 if use_run_unit else 10
    should_skip, skip_reason = check_early_exit_conditions_fn(
        df_trials,
        feature_cols,
        min_features=1,
        min_trials=min_trials_required,
    )
    if should_skip:
        ctx.logger.info("Condition column: skipping due to %s", skip_reason)
        return pd.DataFrame()

    suffix = feature_suffix_from_context_fn(ctx)
    out_dir = get_stats_subfolder_fn(ctx, "condition_effects")

    try:
        cond_a_mask, cond_b_mask, n_condition_a, n_condition_b = split_by_condition(df_trials, ctx.config, ctx.logger)

        if n_condition_a == 0 and n_condition_b == 0:
            msg = (
                "Condition split produced zero trials; check "
                "behavior_analysis.condition.compare_column / behavior_analysis.condition.compare_values "
                "and/or config event_columns.binary_outcome"
            )
            if fail_fast:
                raise ValueError(msg)
            ctx.logger.warning(msg)
            return pd.DataFrame()

        compare_values = get_config_value(ctx.config, "behavior_analysis.condition.compare_values", None)
        if compare_values and len(compare_values) >= 2:
            condition_value1, condition_value2 = str(compare_values[0]), str(compare_values[1])
        else:
            if compare_col in df_trials.columns:
                condition_series = df_trials[compare_col]
                unique_vals = condition_series.dropna().unique()
                if len(unique_vals) >= 2:
                    condition_value1, condition_value2 = str(unique_vals[0]), str(unique_vals[1])
                else:
                    condition_value1, condition_value2 = "1", "0"
            else:
                condition_value1, condition_value2 = "1", "0"

        features = df_trials[feature_cols].copy()
        groups = None
        if getattr(ctx, "group_ids", None) is not None:
            groups_candidate = np.asarray(ctx.group_ids)
            if len(groups_candidate) == len(df_trials):
                groups = groups_candidate
            else:
                ctx.logger.warning(
                    "Condition column: ignoring ctx.group_ids length=%d because current data has %d rows.",
                    len(groups_candidate),
                    len(df_trials),
                )

        if groups is None:
            run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
            if run_col and run_col in df_trials.columns:
                groups = df_trials[run_col].to_numpy()

        groups = sanitize_permutation_groups_fn(groups, ctx.logger, "Condition column")
        if primary_unit in {"trial", "trialwise"} and not allow_iid_trials and perm_enabled and groups is None:
            raise ValueError(
                "Trial-level condition comparison requires grouped permutation labels for non-i.i.d trials. "
                "Provide behavior_analysis.run_adjustment.column in the trial table (or ctx.group_ids), "
                "or set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
            )

        strict_non_iid_trial = primary_unit in {"trial", "trialwise"} and not allow_iid_trials
        p_primary_mode = "perm" if strict_non_iid_trial else None

        column_df = compute_condition_effects(
            features,
            cond_a_mask,
            cond_b_mask,
            min_samples=max(int(getattr(config, "min_samples", 10)), 2),
            fdr_alpha=config.fdr_alpha,
            logger=ctx.logger,
            n_jobs=config.n_jobs,
            config=ctx.config,
            groups=groups,
            paired=bool(use_run_unit),
            pair_ids=df_trials[run_col].to_numpy() if bool(use_run_unit and run_col in df_trials.columns) else None,
            p_primary_mode=p_primary_mode,
        )

        if column_df is not None and not column_df.empty:
            column_df = column_df.copy()
            column_df["comparison_type"] = "column"
            column_df["analysis_kind"] = "condition_column"
            column_df["condition_column"] = compare_col
            column_df["condition_value1"] = condition_value1
            column_df["condition_value2"] = condition_value2

            if "p_value" in column_df.columns and "p_raw" not in column_df.columns:
                column_df["p_raw"] = pd.to_numeric(column_df["p_value"], errors="coerce")
            if "p_value" in column_df.columns and "p_primary" not in column_df.columns:
                column_df["p_primary"] = pd.to_numeric(column_df["p_value"], errors="coerce")
            if "q_value" in column_df.columns and "p_fdr" not in column_df.columns:
                column_df["p_fdr"] = pd.to_numeric(column_df["q_value"], errors="coerce")

            column_df = compute_unified_fdr_fn(
                ctx,
                config,
                column_df,
                p_col="p_primary",
                family_cols=unified_fdr_family_columns,
                analysis_type="condition_column",
            )

            col_path = out_dir / f"condition_effects_column{suffix}_{compare_col}.parquet"
            write_parquet_with_optional_csv_fn(column_df, col_path, also_save_csv=ctx.also_save_csv)
            ctx.logger.info("Condition column comparison: %d features saved to %s", len(column_df), col_path)
            return column_df

    except Exception as exc:
        ctx.logger.error("Condition column comparison failed: %s", exc)
        raise

    return pd.DataFrame()


def stage_condition_impl(
    ctx: Any,
    config: Any,
    *,
    load_trial_table_df_fn: Callable[[Any], Optional[pd.DataFrame]],
    is_dataframe_valid_fn: Callable[[Optional[pd.DataFrame]], bool],
    get_filtered_feature_cols_fn: Callable[[pd.DataFrame, Any], List[str]],
    stage_condition_multigroup_fn: Callable[..., pd.DataFrame],
    stage_condition_column_fn: Callable[..., pd.DataFrame],
    stage_condition_window_fn: Callable[..., pd.DataFrame],
) -> pd.DataFrame:
    """Backward-compatible condition stage (column + optional window + optional multigroup)."""
    df_trials = load_trial_table_df_fn(ctx)
    if not is_dataframe_valid_fn(df_trials):
        ctx.logger.warning("Condition: trial table missing; skipping.")
        return pd.DataFrame()

    feature_cols = get_filtered_feature_cols_fn(df_trials, ctx)
    if not feature_cols:
        ctx.logger.info("Condition: no feature columns found; skipping.")
        return pd.DataFrame()

    result_dfs: List[pd.DataFrame] = []

    compare_values = get_config_value(ctx.config, "behavior_analysis.condition.compare_values", [])
    use_multigroup = isinstance(compare_values, (list, tuple)) and len(compare_values) > 2

    if use_multigroup:
        multigroup_df = stage_condition_multigroup_fn(ctx, config, df_trials=df_trials, feature_cols=feature_cols)
        if multigroup_df is not None and not multigroup_df.empty:
            result_dfs.append(multigroup_df)
    else:
        col_df = stage_condition_column_fn(ctx, config, df_trials=df_trials, feature_cols=feature_cols)
        if col_df is not None and not col_df.empty:
            result_dfs.append(col_df)

    compare_windows = get_config_value(ctx.config, "behavior_analysis.condition.compare_windows", [])
    win_df = stage_condition_window_fn(
        ctx,
        config,
        df_trials=df_trials,
        feature_cols=feature_cols,
        compare_windows=compare_windows if isinstance(compare_windows, list) else None,
    )
    if win_df is not None and not win_df.empty:
        result_dfs.append(win_df)

    if result_dfs:
        return pd.concat(result_dfs, ignore_index=True)
    return pd.DataFrame()


def stage_condition_multigroup_impl(
    ctx: Any,
    config: Any,
    df_trials: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
    *,
    load_trial_table_df_fn: Callable[[Any], Optional[pd.DataFrame]],
    is_dataframe_valid_fn: Callable[[Optional[pd.DataFrame]], bool],
    get_feature_columns_fn: Callable[[pd.DataFrame, Any, Optional[str]], List[str]],
    resolve_condition_compare_column_fn: Callable[[pd.DataFrame, Any], str],
    compute_unified_fdr_fn: Callable[..., pd.DataFrame],
    feature_suffix_from_context_fn: Callable[[Any], str],
    get_stats_subfolder_fn: Callable[[Any, str], Path],
    write_stats_table_fn: Callable[[Any, pd.DataFrame, Path], Path],
    unified_fdr_family_columns: Sequence[str],
) -> pd.DataFrame:
    """Run multi-group condition comparison (3+ groups)."""
    from eeg_pipeline.utils.analysis.stats.effect_size import compute_multigroup_condition_effects

    if df_trials is None:
        df_trials = load_trial_table_df_fn(ctx)
    if not is_dataframe_valid_fn(df_trials):
        ctx.logger.warning("Condition multigroup: trial table missing; skipping.")
        return pd.DataFrame()

    if feature_cols is None:
        feature_cols = get_feature_columns_fn(df_trials, ctx, "condition")
    if not feature_cols:
        ctx.logger.info("Condition multigroup: no feature columns found; skipping.")
        return pd.DataFrame()

    primary_unit = str(
        get_config_value(ctx.config, "behavior_analysis.condition.primary_unit", "trial") or "trial"
    ).strip().lower()
    allow_iid_trials = get_config_bool(ctx.config, "behavior_analysis.statistics.allow_iid_trials", False)
    if primary_unit in {"trial", "trialwise"} and not allow_iid_trials:
        raise ValueError(
            "Trial-level multigroup condition comparisons assume i.i.d trials. "
            "Use run-level aggregation (behavior_analysis.condition.primary_unit=run_mean) "
            "or set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
        )

    compare_column = resolve_condition_compare_column_fn(df_trials, ctx.config)
    compare_values = get_config_value(ctx.config, "behavior_analysis.condition.compare_values", [])
    _ = get_config_bool(ctx.config, "behavior_analysis.condition.overwrite", True)
    compare_labels = get_config_value(ctx.config, "behavior_analysis.condition.compare_labels", None)

    if not isinstance(compare_values, (list, tuple)) or len(compare_values) < 3:
        ctx.logger.info("Condition multigroup: requires 3+ compare_values; skipping.")
        return pd.DataFrame()

    if compare_column not in df_trials.columns:
        ctx.logger.warning("Condition multigroup: column '%s' not found; skipping.", compare_column)
        return pd.DataFrame()

    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
    use_run_unit = primary_unit in {"run", "run_mean", "runmean", "run_level"}
    if use_run_unit:
        if run_col not in df_trials.columns:
            raise ValueError(
                f"Run-level multigroup condition comparisons requested but run column '{run_col}' is missing."
            )
        ctx.logger.info(
            "Condition multigroup: aggregating to run×condition level (primary_unit=%s)",
            primary_unit,
        )
        group_keys = [run_col, compare_column]
        df_agg = (
            df_trials.groupby(group_keys, dropna=True)[feature_cols]
            .mean(numeric_only=True)
            .reset_index()
        )
        cell_counts = (
            df_trials.groupby(group_keys, dropna=True)
            .size()
            .rename("n_trials_cell")
            .reset_index()
        )
        df_trials = df_agg.merge(cell_counts, on=group_keys, how="left")

    if isinstance(compare_labels, (list, tuple)) and len(compare_labels) >= len(compare_values):
        group_labels = [str(l).strip() for l in compare_labels[: len(compare_values)]]
    else:
        group_labels = [str(v) for v in compare_values]

    column_values = df_trials[compare_column]
    group_masks = {}
    for val, label in zip(compare_values, group_labels):
        try:
            numeric_val = float(val)
            mask = (pd.to_numeric(column_values, errors="coerce") == numeric_val).values
        except (ValueError, TypeError):
            val_str = str(val).strip().lower()
            mask = (column_values.astype(str).str.strip().str.lower() == val_str).values

        if np.any(mask):
            group_masks[label] = mask
            ctx.logger.debug("  Group '%s': %d trials", label, int(np.sum(mask)))

    if len(group_masks) < 2:
        ctx.logger.warning("Condition multigroup: fewer than 2 groups have data; skipping.")
        return pd.DataFrame()

    ctx.logger.info(
        "Condition multigroup (%s): %d groups, %d features",
        compare_column,
        len(group_masks),
        len(feature_cols),
    )

    features_df = df_trials[feature_cols].copy()
    multigroup_df = compute_multigroup_condition_effects(
        features_df=features_df,
        group_masks=group_masks,
        group_labels=group_labels,
        fdr_alpha=config.fdr_alpha,
        logger=ctx.logger,
        config=ctx.config,
        paired=bool(use_run_unit),
        pair_ids=df_trials[run_col].to_numpy() if bool(use_run_unit and run_col in df_trials.columns) else None,
    )

    if multigroup_df is not None and not multigroup_df.empty:
        if "p_raw" not in multigroup_df.columns and "p_value" in multigroup_df.columns:
            multigroup_df["p_raw"] = pd.to_numeric(multigroup_df["p_value"], errors="coerce")
        if "p_primary" not in multigroup_df.columns:
            multigroup_df["p_primary"] = pd.to_numeric(multigroup_df.get("p_raw", np.nan), errors="coerce")

        multigroup_df = compute_unified_fdr_fn(
            ctx,
            config,
            multigroup_df,
            p_col="p_primary",
            family_cols=unified_fdr_family_columns,
            analysis_type="condition_multigroup",
        )
        if "p_fdr" in multigroup_df.columns:
            q_vals = pd.to_numeric(multigroup_df["p_fdr"], errors="coerce")
            multigroup_df["q_value"] = q_vals
            multigroup_df["significant_fdr"] = q_vals < float(getattr(config, "fdr_alpha", 0.05))

        multigroup_df["compare_column"] = compare_column
        suffix = feature_suffix_from_context_fn(ctx)
        out_dir = get_stats_subfolder_fn(ctx, "condition_effects")
        filename = f"condition_effects_multigroup{suffix}_{compare_column}.tsv"
        path = out_dir / filename
        write_stats_table_fn(ctx, multigroup_df, path)
        ctx.logger.info("Saved multi-group condition effects to %s", path)

    return multigroup_df


def stage_condition_window_impl(
    ctx: Any,
    config: Any,
    df_trials: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
    compare_windows: Optional[List[str]] = None,
    *,
    load_trial_table_df_fn: Callable[[Any], Optional[pd.DataFrame]],
    is_dataframe_valid_fn: Callable[[Optional[pd.DataFrame]], bool],
    get_feature_columns_fn: Callable[[pd.DataFrame, Any, Optional[str]], List[str]],
    check_early_exit_conditions_fn: Callable[..., tuple[bool, Optional[str]]],
    feature_suffix_from_context_fn: Callable[[Any], str],
    resolve_condition_compare_column_fn: Callable[[pd.DataFrame, Any], str],
    get_stats_subfolder_fn: Callable[[Any, str], Path],
    run_window_comparison_fn: Callable[[Any, pd.DataFrame, List[str], List[str], int, float, str], pd.DataFrame],
    write_parquet_with_optional_csv_fn: Callable[[pd.DataFrame, Path, bool], None],
) -> pd.DataFrame:
    """Run window-based condition comparison (e.g., baseline vs active)."""
    if compare_windows is None:
        compare_windows = get_config_value(ctx.config, "behavior_analysis.condition.compare_windows", [])

    min_windows_required = 2
    if not compare_windows or len(compare_windows) < min_windows_required:
        if compare_windows:
            ctx.logger.warning(
                "Window comparison requires at least %d windows, got: %s",
                min_windows_required,
                compare_windows,
            )
        return pd.DataFrame()

    if df_trials is None:
        df_trials = load_trial_table_df_fn(ctx)
    if not is_dataframe_valid_fn(df_trials):
        ctx.logger.warning("Condition window: trial table missing; skipping.")
        return pd.DataFrame()

    if feature_cols is None:
        feature_cols = get_feature_columns_fn(df_trials, ctx, "condition")
    if not feature_cols:
        ctx.logger.info("Condition window: no feature columns found; skipping.")
        return pd.DataFrame()

    should_skip, skip_reason = check_early_exit_conditions_fn(df_trials, feature_cols, min_features=1, min_trials=10)
    if should_skip:
        ctx.logger.info("Condition window: skipping due to %s", skip_reason)
        return pd.DataFrame()

    suffix = feature_suffix_from_context_fn(ctx)
    compare_col = resolve_condition_compare_column_fn(df_trials, ctx.config)
    out_dir = get_stats_subfolder_fn(ctx, "condition_effects")
    window_min_samples = int(
        get_config_value(
            ctx.config,
            "behavior_analysis.condition.window_comparison.min_samples",
            getattr(config, "min_samples", 10),
        )
        or 0
    )

    ctx.logger.info("Running window comparison: %s", compare_windows)
    window_df = run_window_comparison_fn(
        ctx,
        df_trials,
        feature_cols,
        compare_windows,
        window_min_samples,
        config.fdr_alpha,
        suffix,
    )

    if not window_df.empty:
        window_df["condition_column"] = compare_col
        win_path = out_dir / f"condition_effects_window{suffix}.parquet"
        write_parquet_with_optional_csv_fn(window_df, win_path, also_save_csv=ctx.also_save_csv)
        ctx.logger.info("Condition window comparison: %d features saved to %s", len(window_df), win_path)

    return window_df


def run_window_comparison_impl(
    ctx: Any,
    df_trials: pd.DataFrame,
    feature_cols: List[str],
    windows: List[str],
    min_samples: int,
    fdr_alpha: float,
    suffix: str,
    *,
    feature_prefixes: Sequence[str],
    compute_pairwise_effect_sizes_fn: Callable[[np.ndarray, np.ndarray], Tuple[float, float, float, float, float]],
    feature_type_resolver_fn: Callable[[str, Any], str],
    compute_unified_fdr_fn: Callable[..., pd.DataFrame],
    unified_fdr_family_columns: Sequence[str],
) -> pd.DataFrame:
    """Run paired window comparison on feature columns."""
    from scipy import stats as sp_stats
    from eeg_pipeline.domain.features.naming import NamingSchema

    _ = fdr_alpha
    _ = suffix

    if len(windows) < 2:
        ctx.logger.warning("Window comparison requires at least 2 windows")
        return pd.DataFrame()

    window1, window2 = windows[0], windows[1]

    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
    wc_primary_unit = str(
        get_config_value(ctx.config, "behavior_analysis.condition.window_comparison.primary_unit", "trial") or "trial"
    ).strip().lower()
    allow_iid_trials = get_config_bool(ctx.config, "behavior_analysis.statistics.allow_iid_trials", False)
    if wc_primary_unit in {"trial", "trialwise"} and not allow_iid_trials:
        raise ValueError(
            "Trial-level window comparisons assume i.i.d trials. "
            "Use run-level aggregation (behavior_analysis.condition.window_comparison.primary_unit=run_mean) "
            "or set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
        )
    use_run_unit = wc_primary_unit in {"run", "run_mean", "runmean", "run_level"}
    if use_run_unit and run_col not in df_trials.columns:
        raise ValueError(
            f"Run-level window comparison requested but run column '{run_col}' is missing from trial table."
        )

    window1_features: Dict[Tuple[str, str, str, str, str], str] = {}
    window2_features: Dict[Tuple[str, str, str, str, str], str] = {}

    prefixes = sorted(feature_prefixes, key=len, reverse=True)
    w1 = str(window1).strip().lower()
    w2 = str(window2).strip().lower()

    n_unparseable = 0
    for col in feature_cols:
        col_str = str(col)
        matched_prefix = next((p for p in prefixes if col_str.startswith(p)), None)
        raw_name = col_str[len(matched_prefix) :] if matched_prefix else col_str

        parsed = NamingSchema.parse(raw_name)
        if not parsed.get("valid"):
            n_unparseable += 1
            continue

        parsed_group = str(parsed.get("group") or "").strip().lower()
        parsed_segment = str(parsed.get("segment") or "").strip().lower()
        parsed_band = str(parsed.get("band") or "").strip()

        if matched_prefix and parsed_group in {w1, w2}:
            seg = parsed_group
            band = parsed_segment if parsed_segment else parsed_band
        else:
            seg = parsed_segment if parsed_segment else parsed_group
            band = parsed_band if parsed_band else parsed_segment

        seg = seg.strip().lower()
        if seg not in {w1, w2}:
            continue

        group_name = matched_prefix.rstrip("_") if matched_prefix else str(parsed.get("group") or "")
        key = (
            group_name,
            band,
            str(parsed.get("scope") or ""),
            str(parsed.get("identifier") or ""),
            str(parsed.get("stat") or ""),
        )
        if seg == w1:
            window1_features[key] = col_str
        else:
            window2_features[key] = col_str

    if n_unparseable and ctx.logger is not None:
        ctx.logger.debug(
            "Window comparison: skipped %d unparseable feature columns (NamingSchema invalid).",
            int(n_unparseable),
        )

    common_bases = sorted(set(window1_features.keys()) & set(window2_features.keys()))
    if not common_bases:
        w1_stats = {k[4] for k in window1_features.keys()}
        w2_stats = {k[4] for k in window2_features.keys()}
        common_stats = w1_stats & w2_stats

        if not common_stats:
            ctx.logger.warning(
                "No matching feature pairs found for windows %s and %s. "
                "Reason: different stat types (%s has %s, %s has %s). "
                "Window comparisons require features with matching (group, band, scope, identifier, stat).",
                window1,
                window2,
                window1,
                sorted(w1_stats),
                window2,
                sorted(w2_stats),
            )
        else:
            ctx.logger.warning(
                "No matching feature pairs found for windows %s and %s. "
                "Found %d %s features and %d %s features, "
                "but none share the same (group, band, scope, identifier, stat) combination.",
                window1,
                window2,
                len(window1_features),
                window1,
                len(window2_features),
                window2,
            )
        return pd.DataFrame()

    n_pairs = len(common_bases)
    ctx.logger.info("Window comparison: %d feature pairs for %s vs %s (vectorized)", n_pairs, window1, window2)

    cols1 = [window1_features[base] for base in common_bases]
    cols2 = [window2_features[base] for base in common_bases]

    v1_matrix = df_trials[cols1].to_numpy(dtype=np.float64, na_value=np.nan)
    v2_matrix = df_trials[cols2].to_numpy(dtype=np.float64, na_value=np.nan)

    valid_mask = np.isfinite(v1_matrix) & np.isfinite(v2_matrix)
    n_valid_per_pair = valid_mask.sum(axis=0)

    records: List[Dict[str, Any]] = []
    for i, base_name in enumerate(common_bases):
        col1 = cols1[i]
        col2 = cols2[i]

        stat_run = np.nan
        p_val_run = np.nan
        n_runs = np.nan
        if use_run_unit and run_col in df_trials.columns:
            df_run = pd.DataFrame({"run": df_trials[run_col], "v1": v1_matrix[:, i], "v2": v2_matrix[:, i]}).dropna()
            run_means = df_run.groupby("run", dropna=True)[["v1", "v2"]].mean(numeric_only=True)
            n_runs = int(len(run_means))
            if n_runs < max(min_samples, 2):
                continue
            v1_valid = run_means["v1"].to_numpy(dtype=float)
            v2_valid = run_means["v2"].to_numpy(dtype=float)
            try:
                stat_run, p_val_run = sp_stats.wilcoxon(v1_valid, v2_valid)
            except (ValueError, TypeError, sp_stats.Error, KeyError) as exc:
                raise RuntimeError(
                    f"Run-level Wilcoxon failed for window comparison '{window1}' vs '{window2}' "
                    f"(col1={col1}, col2={col2}, run_col={run_col})"
                ) from exc
            stat, p_val = stat_run, p_val_run
            n_primary = int(n_runs)
        else:
            n_valid = int(n_valid_per_pair[i])
            if n_valid < max(min_samples, 2):
                continue
            v1_valid = v1_matrix[valid_mask[:, i], i]
            v2_valid = v2_matrix[valid_mask[:, i], i]
            try:
                stat, p_val = sp_stats.wilcoxon(v1_valid, v2_valid)
            except (ValueError, TypeError, sp_stats.Error) as exc:
                raise RuntimeError(
                    f"Wilcoxon failed for window comparison '{window1}' vs '{window2}' " f"(col1={col1}, col2={col2})"
                ) from exc
            n_primary = n_valid

        mean_diff_i, std_diff_i, cohens_d_i, hedges_g_i, _ = compute_pairwise_effect_sizes_fn(v1_valid, v2_valid)

        col1_prefix = next((p for p in prefixes if str(col1).startswith(p)), "")
        col1_raw = str(col1)[len(col1_prefix) :] if col1_prefix else str(col1)

        records.append(
            {
                "feature": "::".join(str(x) for x in base_name),
                "feature_col_window1": col1,
                "feature_col_window2": col2,
                "feature_type": feature_type_resolver_fn(col1_raw, ctx.config),
                "analysis_kind": "condition_window",
                "comparison_type": "window",
                "window1": window1,
                "window2": window2,
                "n_pairs": n_primary,
                "n_runs": n_runs,
                "mean_window1": float(np.nanmean(v1_valid)),
                "mean_window2": float(np.nanmean(v2_valid)),
                "std_window1": float(np.nanstd(v1_valid, ddof=1)),
                "std_window2": float(np.nanstd(v2_valid, ddof=1)),
                "mean_diff": float(mean_diff_i),
                "std_diff": float(std_diff_i) if np.isfinite(std_diff_i) else np.nan,
                "statistic": float(stat),
                "p_raw": float(p_val),
                "statistic_run": float(stat_run) if np.isfinite(stat_run) else np.nan,
                "p_value_run": float(p_val_run) if np.isfinite(p_val_run) else np.nan,
                "cohens_d": float(cohens_d_i),
                "hedges_g": float(hedges_g_i),
            }
        )

    if not records:
        ctx.logger.info("Window comparison: no valid feature pairs with sufficient samples")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    if use_run_unit and "p_value_run" in df.columns:
        df["p_primary"] = pd.to_numeric(df["p_value_run"], errors="coerce")
        df["p_primary_source"] = "run_mean"
    else:
        df["p_primary"] = pd.to_numeric(df["p_raw"], errors="coerce")
        df["p_primary_source"] = "trial"

    df = compute_unified_fdr_fn(
        ctx,
        ctx.config,
        df,
        p_col="p_primary",
        family_cols=unified_fdr_family_columns,
        analysis_type="condition_window",
    )
    return df
