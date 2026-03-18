from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.behavior_loader import ensure_behavior_config
from eeg_pipeline.utils.config.loader import get_config_value, require_config_value


def _resolve_condition_permutation_count(config: Any, *, perm_enabled: bool) -> int:
    """Resolve the effective condition permutation count from canonical config."""
    if not perm_enabled:
        return 0

    scoped = get_config_value(
        config,
        "behavior_analysis.condition.permutation.n_permutations",
        None,
    )
    if scoped is not None:
        return int(scoped)

    statistics = require_config_value(
        config,
        "behavior_analysis.statistics.n_permutations",
    )
    return int(statistics)


def resolve_condition_compare_column(df_trials: pd.DataFrame, config: Any) -> str:
    """
    Resolve the condition column from config.

    Resolution order:
    1. ``behavior_analysis.condition.compare_column`` (explicit override)
    2. ``event_columns.condition`` (generic condition aliases)
    3. ``event_columns.binary_outcome`` (binary condition aliases)

    Raises ``ValueError`` if neither resolves to a column present in the trial table.
    """
    from eeg_pipeline.utils.data.columns import (
        get_binary_outcome_column_from_config,
        get_condition_column_from_config,
    )

    compare_col_value = get_config_value(
        config, "behavior_analysis.condition.compare_column", None
    )
    compare_col = str(compare_col_value or "").strip()
    if compare_col and compare_col in df_trials.columns:
        return compare_col

    fallback_col = get_condition_column_from_config(config, df_trials)
    if fallback_col and fallback_col in df_trials.columns:
        return str(fallback_col)

    fallback_col = get_binary_outcome_column_from_config(config, df_trials)
    if fallback_col and fallback_col in df_trials.columns:
        return str(fallback_col)

    raise ValueError(
        "Could not resolve a condition comparison column. "
        "Set 'behavior_analysis.condition.compare_column' or configure "
        "'event_columns.condition' / 'event_columns.binary_outcome' to match "
        "a column in the trial table. "
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
    from eeg_pipeline.utils.analysis.stats.effect_size import resolve_binary_condition_values

    ctx.config = ensure_behavior_config(ctx.config)

    compare_values = require_config_value(
        ctx.config, "behavior_analysis.condition.compare_values"
    )
    use_multigroup = isinstance(compare_values, (list, tuple)) and len(compare_values) > 2
    if use_multigroup:
        ctx.logger.info(
            "Condition column: %d values specified, delegating to multigroup comparison",
            len(compare_values),
        )
        return stage_condition_multigroup_fn(ctx, config, df_trials=df_trials, feature_cols=feature_cols)

    fail_fast = bool(
        require_config_value(ctx.config, "behavior_analysis.condition.fail_fast")
    )
    primary_unit = str(
        require_config_value(ctx.config, "behavior_analysis.condition.primary_unit")
    ).strip().lower()
    allow_iid_trials = bool(
        require_config_value(ctx.config, "behavior_analysis.statistics.allow_iid_trials")
    )
    perm_enabled = bool(
        require_config_value(
            ctx.config, "behavior_analysis.condition.permutation.enabled"
        )
    )
    n_perm = _resolve_condition_permutation_count(ctx.config, perm_enabled=perm_enabled)
    if primary_unit in {"trial", "trialwise"} and (not perm_enabled or n_perm <= 0) and not allow_iid_trials:
        raise ValueError(
            "Trial-level condition comparisons require a valid non-i.i.d inference method. "
            "Enable permutation testing with a positive permutation count "
            "(behavior_analysis.condition.permutation.enabled=true and "
            "behavior_analysis.condition.permutation.n_permutations>0) "
            "or use run-level aggregation (behavior_analysis.condition.primary_unit=run_mean). "
            "Set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
        )

    use_run_unit = primary_unit in {"run", "run_mean", "runmean", "run_level"}
    run_col = str(
        require_config_value(ctx.config, "behavior_analysis.run_adjustment.column")
    ).strip()

    if df_trials is None:
        df_trials = load_trial_table_df_fn(ctx)
    if not is_dataframe_valid_fn(df_trials):
        ctx.logger.warning("Condition column: trial table missing; skipping.")
        return pd.DataFrame()

    if feature_cols is None:
        feature_cols = get_feature_columns_fn(df_trials, ctx, "condition")

    compare_col = resolve_condition_compare_column_fn(df_trials, ctx.config)
    _ = bool(
        require_config_value(ctx.config, "behavior_analysis.condition.overwrite")
    )
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

    min_trials_required = 2
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
                "and/or config event_columns.condition"
            )
            if fail_fast:
                raise ValueError(msg)
            ctx.logger.warning(msg)
            return pd.DataFrame()

        condition_value1, condition_value2 = resolve_binary_condition_values(
            df_trials,
            ctx.config,
        )

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
            run_col = str(
                require_config_value(ctx.config, "behavior_analysis.run_adjustment.column")
            ).strip()
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
            min_samples=max(int(getattr(config, "min_samples", None)), 2),
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
) -> pd.DataFrame:
    """Backward-compatible condition stage (column or multigroup)."""
    ctx.config = ensure_behavior_config(ctx.config)

    df_trials = load_trial_table_df_fn(ctx)
    if not is_dataframe_valid_fn(df_trials):
        ctx.logger.warning("Condition: trial table missing; skipping.")
        return pd.DataFrame()

    feature_cols = get_filtered_feature_cols_fn(df_trials, ctx)
    if not feature_cols:
        ctx.logger.info("Condition: no feature columns found; skipping.")
        return pd.DataFrame()

    result_dfs: List[pd.DataFrame] = []

    compare_values = require_config_value(
        ctx.config, "behavior_analysis.condition.compare_values"
    )
    use_multigroup = isinstance(compare_values, (list, tuple)) and len(compare_values) > 2

    if use_multigroup:
        multigroup_df = stage_condition_multigroup_fn(ctx, config, df_trials=df_trials, feature_cols=feature_cols)
        if multigroup_df is not None and not multigroup_df.empty:
            result_dfs.append(multigroup_df)
    else:
        col_df = stage_condition_column_fn(ctx, config, df_trials=df_trials, feature_cols=feature_cols)
        if col_df is not None and not col_df.empty:
            result_dfs.append(col_df)

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

    ctx.config = ensure_behavior_config(ctx.config)

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
        require_config_value(ctx.config, "behavior_analysis.condition.primary_unit")
    ).strip().lower()
    allow_iid_trials = bool(
        require_config_value(ctx.config, "behavior_analysis.statistics.allow_iid_trials")
    )
    if primary_unit in {"trial", "trialwise"} and not allow_iid_trials:
        raise ValueError(
            "Trial-level multigroup condition comparisons assume i.i.d trials. "
            "Use run-level aggregation (behavior_analysis.condition.primary_unit=run_mean) "
            "or set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
        )

    compare_column = resolve_condition_compare_column_fn(df_trials, ctx.config)
    compare_values = require_config_value(
        ctx.config, "behavior_analysis.condition.compare_values"
    )
    _ = bool(
        require_config_value(ctx.config, "behavior_analysis.condition.overwrite")
    )
    compare_labels = get_config_value(
        ctx.config, "behavior_analysis.condition.compare_labels", None
    )

    if not isinstance(compare_values, (list, tuple)) or len(compare_values) < 3:
        ctx.logger.info("Condition multigroup: requires 3+ compare_values; skipping.")
        return pd.DataFrame()

    if compare_column not in df_trials.columns:
        ctx.logger.warning("Condition multigroup: column '%s' not found; skipping.", compare_column)
        return pd.DataFrame()

    run_col = str(
        require_config_value(ctx.config, "behavior_analysis.run_adjustment.column")
    ).strip()
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
