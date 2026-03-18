from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.behavior_loader import ensure_behavior_config
from eeg_pipeline.utils.config.loader import require_config_value


def stage_regression_impl(
    ctx: Any,
    config: Any,
    *,
    feature_suffix_from_context_fn: Callable[[Any], str],
    load_trial_table_df_fn: Callable[[Any], Optional[pd.DataFrame]],
    is_dataframe_valid_fn: Callable[[Optional[pd.DataFrame]], bool],
    get_feature_columns_fn: Callable[[pd.DataFrame, Any], List[str]],
    check_early_exit_conditions_fn: Callable[..., tuple[bool, Optional[str]]],
    sanitize_permutation_groups_fn: Callable[[Any, Any, str], Any],
    attach_predictor_metadata_fn: Callable[[pd.DataFrame, Dict[str, Any], str], pd.DataFrame],
    get_stats_subfolder_fn: Callable[[Any, str], Path],
    write_stats_table_fn: Callable[[Any, pd.DataFrame, Path], Path],
) -> pd.DataFrame:
    """Trialwise regression stage with optional run-level aggregation."""
    from eeg_pipeline.utils.analysis.stats.trialwise_regression import run_trialwise_feature_regressions

    ctx.config = ensure_behavior_config(ctx.config)

    suffix = feature_suffix_from_context_fn(ctx)
    method_label = getattr(config, "method_label", "")
    method_suffix = f"_{method_label}" if method_label else ""

    df_trials = load_trial_table_df_fn(ctx)
    if not is_dataframe_valid_fn(df_trials):
        ctx.logger.warning("Regression: trial table missing; skipping.")
        return pd.DataFrame()

    primary_unit = str(
        require_config_value(ctx.config, "behavior_analysis.regression.primary_unit")
    ).strip().lower()
    use_run_unit = primary_unit in {"run", "run_mean", "runmean", "run_level"}
    run_col = str(
        require_config_value(ctx.config, "behavior_analysis.run_adjustment.column")
    ).strip()
    allow_iid_trials = bool(
        require_config_value(ctx.config, "behavior_analysis.statistics.allow_iid_trials")
    )
    n_perm = int(
        require_config_value(ctx.config, "behavior_analysis.regression.n_permutations")
    )
    include_run_block = bool(
        require_config_value(ctx.config, "behavior_analysis.regression.include_run_block")
    )

    feature_cols = get_feature_columns_fn(df_trials, ctx)
    min_observations = 2 if use_run_unit else 10
    should_skip, skip_reason = check_early_exit_conditions_fn(
        df_trials,
        feature_cols,
        min_features=1,
        min_trials=min_observations,
    )
    if should_skip:
        ctx.logger.info("Regression: skipping due to %s", skip_reason)
        return pd.DataFrame()

    if use_run_unit and run_col not in df_trials.columns:
        raise ValueError(
            f"Run-level regression requested (primary_unit={primary_unit!r}) "
            f"but run column '{run_col}' is missing from trial table."
        )
    if use_run_unit and include_run_block:
        raise ValueError(
            "Run-level regression cannot include run/block adjustment because run is the analysis unit. "
            "Disable behavior_analysis.regression.include_run_block for run-level regression."
        )
    if primary_unit in {"trial", "trialwise"} and not allow_iid_trials and n_perm <= 0:
        raise ValueError(
            "Trial-level regression requires a valid non-i.i.d inference method. "
            "Set behavior_analysis.regression.n_permutations > 0, "
            "use run-level aggregation (behavior_analysis.regression.primary_unit=run_mean), "
            "or set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
        )

    if use_run_unit and run_col in df_trials.columns:
        ctx.logger.info(
            "Regression: using feature-specific run-level aggregation (primary_unit=%s)",
            primary_unit,
        )

    groups = None
    if use_run_unit:
        if n_perm > 0:
            ctx.logger.info(
                "Regression: run-level aggregation uses ungrouped permutation across runs."
            )
    else:
        if getattr(ctx, "group_ids", None) is not None:
            groups_candidate = np.asarray(ctx.group_ids)
            if len(groups_candidate) == len(df_trials):
                groups = groups_candidate
            else:
                ctx.logger.warning(
                    "Regression: ignoring ctx.group_ids length=%d because current data has %d rows.",
                    len(groups_candidate),
                    len(df_trials),
                )
        if groups is None:
            if run_col in df_trials.columns:
                groups = df_trials[run_col].to_numpy()
            elif "block" in df_trials.columns:
                groups = df_trials["block"].to_numpy()
            elif "run" in df_trials.columns:
                groups = df_trials["run"].to_numpy()
        groups = sanitize_permutation_groups_fn(groups, ctx.logger, "Regression")
    if primary_unit in {"trial", "trialwise"} and not allow_iid_trials and groups is None:
        raise ValueError(
            "Trial-level regression with permutation inference requires grouped labels. "
            "Provide behavior_analysis.run_adjustment.column in the trial table (or ctx.group_ids), "
            "or set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
        )
    strict_permutation_primary = bool(
        primary_unit in {"trial", "trialwise"} and not allow_iid_trials and n_perm > 0
    )

    reg_df, reg_meta = run_trialwise_feature_regressions(
        df_trials,
        feature_cols=feature_cols,
        config=ctx.config,
        groups_for_permutation=groups,
        strict_permutation_primary=strict_permutation_primary,
    )
    reg_meta["primary_unit"] = primary_unit
    ctx.data_qc["trialwise_regression"] = reg_meta
    reg_df = attach_predictor_metadata_fn(reg_df, reg_meta)

    out_dir = get_stats_subfolder_fn(ctx, "trialwise_regression")
    out_path = out_dir / f"regression_feature_effects{suffix}{method_suffix}.parquet"
    if not reg_df.empty:
        actual_path = write_stats_table_fn(ctx, reg_df, out_path)
        ctx.logger.info("Regression results saved: %s (%d features)", actual_path.name, len(reg_df))
    return reg_df
