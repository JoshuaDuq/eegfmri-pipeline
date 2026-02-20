from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import get_config_bool, get_config_float, get_config_int, get_config_value


def stage_mediation_impl(
    ctx: Any,
    config: Any,
    *,
    load_trial_table_df_fn: Callable[[Any], Optional[pd.DataFrame]],
    is_dataframe_valid_fn: Callable[[Optional[pd.DataFrame]], bool],
    get_feature_columns_fn: Callable[[pd.DataFrame, Any, Optional[str]], List[str]],
    check_early_exit_conditions_fn: Callable[..., tuple[bool, Optional[str]]],
    sanitize_permutation_groups_fn: Callable[[Any, Any, str], Any],
    feature_type_resolver_fn: Callable[[str, Any], str],
    compute_unified_fdr_fn: Callable[..., pd.DataFrame],
    unified_fdr_family_columns: List[str],
) -> pd.DataFrame:
    """Run mediation analysis: test if neural features mediate temperature→rating."""
    from eeg_pipeline.analysis.behavior.api import run_mediation_analysis

    df_trials = load_trial_table_df_fn(ctx)
    if not is_dataframe_valid_fn(df_trials):
        ctx.logger.info("Mediation: trial table missing; skipping.")
        return pd.DataFrame()

    required_columns = {"temperature", "rating"}
    missing_columns = required_columns - set(df_trials.columns)
    if missing_columns:
        ctx.logger.warning("Mediation: requires %s columns; missing: %s. Skipping.", required_columns, missing_columns)
        return pd.DataFrame()

    feature_cols = get_feature_columns_fn(df_trials, ctx, "mediation")
    should_skip, skip_reason = check_early_exit_conditions_fn(df_trials, feature_cols, min_features=1, min_trials=10)
    if should_skip:
        ctx.logger.info("Mediation: skipping due to %s", skip_reason)
        return pd.DataFrame()

    if not feature_cols:
        ctx.logger.info("Mediation: no feature columns found; skipping.")
        return pd.DataFrame()

    ctx.logger.info("Running mediation analysis...")
    n_bootstrap = get_config_int(ctx.config, "behavior_analysis.mediation.n_bootstrap", 1000)
    n_permutations = get_config_int(ctx.config, "behavior_analysis.mediation.n_permutations", 0)
    allow_iid_trials = get_config_bool(ctx.config, "behavior_analysis.statistics.allow_iid_trials", False)
    p_primary_mode = str(
        get_config_value(ctx.config, "behavior_analysis.mediation.p_primary_mode", "perm_if_available") or "perm_if_available"
    ).strip().lower()
    min_effect_size = get_config_float(ctx.config, "behavior_analysis.mediation.min_effect_size", 0.05)
    max_mediators = get_config_value(ctx.config, "behavior_analysis.mediation.max_mediators", None)
    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
    perm_scheme = str(get_config_value(ctx.config, "behavior_analysis.permutation.scheme", "shuffle") or "shuffle").strip().lower()

    if max_mediators is not None:
        max_mediators = int(max_mediators)
        variances = df_trials[feature_cols].var()
        mediators = variances.nlargest(max(1, max_mediators)).index.tolist()
        ctx.logger.info("Limiting to top %d mediators by variance", max_mediators)
    else:
        mediators = feature_cols
        ctx.logger.info("Testing all %d features as mediators (no limit)", len(mediators))

    groups_for_resampling = None
    if getattr(ctx, "group_ids", None) is not None:
        groups_candidate = np.asarray(ctx.group_ids)
        if len(groups_candidate) == len(df_trials):
            groups_for_resampling = groups_candidate
        else:
            ctx.logger.warning(
                "Mediation: ignoring ctx.group_ids length=%d because trial table has %d rows.",
                len(groups_candidate),
                len(df_trials),
            )
    if groups_for_resampling is None and run_col in df_trials.columns:
        groups_for_resampling = df_trials[run_col].to_numpy()
    groups_for_resampling = sanitize_permutation_groups_fn(groups_for_resampling, ctx.logger, "Mediation")

    if not allow_iid_trials:
        if n_permutations <= 0:
            raise ValueError(
                "Mediation requires non-i.i.d inference under repeated measures. "
                "Set behavior_analysis.mediation.n_permutations > 0 or set "
                "behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
            )
        if groups_for_resampling is None:
            raise ValueError(
                "Mediation requires grouped resampling labels for non-i.i.d trials. "
                "Provide behavior_analysis.run_adjustment.column in the trial table (or ctx.group_ids), "
                "or set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
            )

    result = run_mediation_analysis(
        df_trials,
        "temperature",
        mediators,
        "rating",
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        groups=groups_for_resampling,
        permutation_scheme=perm_scheme,
        min_effect_size=min_effect_size,
    )
    if result is None or result.empty:
        return pd.DataFrame()

    med_df = result.copy()
    if "analysis_kind" not in med_df.columns:
        med_df["analysis_kind"] = "mediation"

    if "p_raw" not in med_df.columns:
        med_df["p_raw"] = pd.to_numeric(med_df.get("sobel_p", med_df.get("p_value", np.nan)), errors="coerce")
    if "p_ab_perm" not in med_df.columns:
        med_df["p_ab_perm"] = np.nan

    use_perm = p_primary_mode in {"perm", "permutation", "perm_if_available", "permutation_if_available"}
    if use_perm:
        perm_p = pd.to_numeric(med_df["p_ab_perm"], errors="coerce")
        raw_p = pd.to_numeric(med_df["p_raw"], errors="coerce")
        if not allow_iid_trials:
            med_df["p_primary"] = perm_p.where(perm_p.notna(), np.nan)
            med_df["p_primary_source"] = np.where(perm_p.notna(), "perm", "perm_missing_required")
        else:
            med_df["p_primary"] = perm_p.where(perm_p.notna(), raw_p)
            med_df["p_primary_source"] = np.where(perm_p.notna(), "perm", "sobel")
    else:
        med_df["p_primary"] = pd.to_numeric(med_df["p_raw"], errors="coerce")
        med_df["p_primary_source"] = "sobel"

    mediator_col = "mediator" if "mediator" in med_df.columns else ("feature" if "feature" in med_df.columns else None)
    if mediator_col is not None:
        try:
            med_df["feature_type"] = [feature_type_resolver_fn(str(m), ctx.config) for m in med_df[mediator_col].astype(str).tolist()]
        except Exception:
            med_df["feature_type"] = "unknown"
    else:
        med_df["feature_type"] = "unknown"

    med_df = compute_unified_fdr_fn(
        ctx,
        config,
        med_df,
        p_col="p_primary",
        family_cols=unified_fdr_family_columns,
        analysis_type="mediation",
    )
    med_df["significant_mediation"] = pd.to_numeric(med_df.get("p_fdr", np.nan), errors="coerce") < float(
        getattr(config, "fdr_alpha", 0.05)
    )
    return med_df


def stage_mixed_effects_impl(ctx: Any, config: Any) -> pd.DataFrame:
    _ = config
    ctx.logger.warning(
        "Skipping mixed-effects analysis in subject-level mode; "
        "run this at group level with multiple subjects via run_group_level()."
    )
    return pd.DataFrame()


def stage_moderation_impl(
    ctx: Any,
    config: Any,
    *,
    load_trial_table_df_fn: Callable[[Any], Optional[pd.DataFrame]],
    is_dataframe_valid_fn: Callable[[Optional[pd.DataFrame]], bool],
    get_feature_columns_fn: Callable[[pd.DataFrame, Any, Optional[str]], List[str]],
    check_early_exit_conditions_fn: Callable[..., tuple[bool, Optional[str]]],
    sanitize_permutation_groups_fn: Callable[[Any, Any, str], Any],
    feature_type_resolver_fn: Callable[[str, Any], str],
    compute_unified_fdr_fn: Callable[..., pd.DataFrame],
    get_stats_subfolder_fn: Callable[[Any, str], Any],
    feature_suffix_from_context_fn: Callable[[Any], str],
    write_parquet_with_optional_csv_fn: Callable[[pd.DataFrame, Any, bool], None],
    unified_fdr_family_columns: List[str],
) -> pd.DataFrame:
    """Run moderation analysis: feature moderates temperature→rating relationship."""
    from eeg_pipeline.utils.analysis.stats.moderation import run_moderation_analysis

    suffix = feature_suffix_from_context_fn(ctx)
    method_label = getattr(config, "method_label", "")
    method_suffix = f"_{method_label}" if method_label else ""

    df_trials = load_trial_table_df_fn(ctx)
    if not is_dataframe_valid_fn(df_trials):
        ctx.logger.warning("Moderation: trial table missing; skipping.")
        return pd.DataFrame()

    required_columns = {"temperature", "rating"}
    missing_columns = required_columns - set(df_trials.columns)
    if missing_columns:
        ctx.logger.warning("Moderation: requires %s columns; missing: %s. Skipping.", required_columns, missing_columns)
        return pd.DataFrame()

    feature_cols = get_feature_columns_fn(df_trials, ctx, "moderation")
    should_skip, skip_reason = check_early_exit_conditions_fn(df_trials, feature_cols, min_features=1, min_trials=10)
    if should_skip:
        ctx.logger.info("Moderation: skipping due to %s", skip_reason)
        return pd.DataFrame()

    if not feature_cols:
        ctx.logger.info("Moderation: no feature columns found; skipping.")
        return pd.DataFrame()

    max_features = getattr(config, "moderation_max_features", None)
    fdr_alpha = float(getattr(config, "fdr_alpha", 0.05))
    n_permutations = get_config_int(ctx.config, "behavior_analysis.moderation.n_permutations", 0)
    min_samples = max(int(getattr(config, "moderation_min_samples", 15)), 2)
    allow_iid_trials = get_config_bool(ctx.config, "behavior_analysis.statistics.allow_iid_trials", False)
    p_primary_mode = str(
        get_config_value(ctx.config, "behavior_analysis.moderation.p_primary_mode", "perm_if_available") or "perm_if_available"
    ).strip().lower()
    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
    perm_scheme = str(get_config_value(ctx.config, "behavior_analysis.permutation.scheme", "shuffle") or "shuffle").strip().lower()

    groups_for_resampling = None
    if getattr(ctx, "group_ids", None) is not None:
        groups_candidate = np.asarray(ctx.group_ids)
        if len(groups_candidate) == len(df_trials):
            groups_for_resampling = groups_candidate
        else:
            ctx.logger.warning(
                "Moderation: ignoring ctx.group_ids length=%d because trial table has %d rows.",
                len(groups_candidate),
                len(df_trials),
            )
    if groups_for_resampling is None and run_col in df_trials.columns:
        groups_for_resampling = df_trials[run_col].to_numpy()
    groups_for_resampling = sanitize_permutation_groups_fn(groups_for_resampling, ctx.logger, "Moderation")

    if not allow_iid_trials:
        if n_permutations <= 0:
            raise ValueError(
                "Moderation requires non-i.i.d inference under repeated measures. "
                "Set behavior_analysis.moderation.n_permutations > 0 or set "
                "behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
            )
        if groups_for_resampling is None:
            raise ValueError(
                "Moderation requires grouped resampling labels for non-i.i.d trials. "
                "Provide behavior_analysis.run_adjustment.column in the trial table (or ctx.group_ids), "
                "or set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
            )

    if max_features is not None and len(feature_cols) > max_features:
        variances = df_trials[feature_cols].var()
        feature_cols = variances.nlargest(max_features).index.tolist()
        ctx.logger.info("Moderation: limited to top %d features by variance", max_features)
    else:
        ctx.logger.info("Moderation: testing all %d features (no limit)", len(feature_cols))

    temperature = pd.to_numeric(df_trials["temperature"], errors="coerce").to_numpy()
    rating = pd.to_numeric(df_trials["rating"], errors="coerce").to_numpy()

    records: List[Dict[str, Any]] = []
    for feat in feature_cols:
        feature_values = pd.to_numeric(df_trials[feat], errors="coerce").to_numpy()

        valid_mask = np.isfinite(temperature) & np.isfinite(rating) & np.isfinite(feature_values)
        n_valid = int(valid_mask.sum())
        if n_valid < min_samples:
            continue
        groups_valid = None
        if groups_for_resampling is not None and len(groups_for_resampling) == len(valid_mask):
            groups_valid = np.asarray(groups_for_resampling)[valid_mask]
        groups_valid = sanitize_permutation_groups_fn(groups_valid, ctx.logger, f"Moderation[{feat}]")
        if not allow_iid_trials and groups_valid is None:
            continue

        result = run_moderation_analysis(
            X=temperature[valid_mask],
            W=feature_values[valid_mask],
            Y=rating[valid_mask],
            n_perm=n_permutations,
            x_label="temperature",
            w_label=str(feat),
            y_label="rating",
            center_predictors=True,
            rng=getattr(ctx, "rng", None),
            groups=groups_valid,
            permutation_scheme=perm_scheme,
        )

        records.append(
            {
                "feature": str(feat),
                "feature_type": feature_type_resolver_fn(str(feat), ctx.config),
                "n": result.n,
                "b1_temperature": result.b1,
                "b2_feature": result.b2,
                "b3_interaction": result.b3,
                "se_b3": result.se_b3,
                "p_interaction": result.p_b3,
                "p_interaction_perm": result.p_b3_perm,
                "n_permutations": int(getattr(result, "n_permutations", n_permutations) or 0),
                "slope_low_w": result.slope_low_w,
                "slope_mean_w": result.slope_mean_w,
                "slope_high_w": result.slope_high_w,
                "p_slope_low": result.p_slope_low,
                "p_slope_mean": result.p_slope_mean,
                "p_slope_high": result.p_slope_high,
                "r_squared": result.r_squared,
                "r_squared_change": result.r_squared_change,
                "f_interaction": result.f_interaction,
                "p_f_interaction": result.p_f_interaction,
                "jn_low": result.jn_low,
                "jn_high": result.jn_high,
                "jn_type": result.jn_type,
                "significant_moderation_raw": result.is_significant_moderation(fdr_alpha),
            }
        )

    mod_df = pd.DataFrame(records) if records else pd.DataFrame()

    if not mod_df.empty:
        if "analysis_kind" not in mod_df.columns:
            mod_df["analysis_kind"] = "moderation"

        mod_df["p_raw"] = pd.to_numeric(mod_df["p_interaction"], errors="coerce")
        use_perm = p_primary_mode in {"perm", "permutation", "perm_if_available", "permutation_if_available"}
        if use_perm and "p_interaction_perm" in mod_df.columns:
            p_perm = pd.to_numeric(mod_df["p_interaction_perm"], errors="coerce")
            if not allow_iid_trials:
                mod_df["p_primary"] = p_perm.where(p_perm.notna(), np.nan)
                mod_df["p_primary_source"] = np.where(p_perm.notna(), "perm", "perm_missing_required")
            else:
                mod_df["p_primary"] = p_perm.where(p_perm.notna(), mod_df["p_raw"])
                mod_df["p_primary_source"] = np.where(p_perm.notna(), "perm", "asymptotic")
        else:
            mod_df["p_primary"] = mod_df["p_raw"]
            mod_df["p_primary_source"] = "asymptotic"

        mod_df = compute_unified_fdr_fn(
            ctx,
            config,
            mod_df,
            p_col="p_primary",
            family_cols=unified_fdr_family_columns,
            analysis_type="moderation",
        )
        if "p_fdr" in mod_df.columns:
            p_fdr = pd.to_numeric(mod_df["p_fdr"], errors="coerce")
        else:
            p_fdr = pd.Series(np.nan, index=mod_df.index, dtype=float)
        mod_df["p_fdr"] = p_fdr
        mod_df["significant_moderation"] = p_fdr < fdr_alpha

    out_dir = get_stats_subfolder_fn(ctx, "moderation")
    out_path = out_dir / f"moderation_results{suffix}{method_suffix}.parquet"
    if not mod_df.empty:
        write_parquet_with_optional_csv_fn(mod_df, out_path, also_save_csv=ctx.also_save_csv)
        n_sig = int((pd.to_numeric(mod_df.get("p_fdr", np.nan), errors="coerce") < fdr_alpha).sum())
        ctx.logger.info("Moderation: %d features tested, %d significant (FDR < %.2f)", len(mod_df), n_sig, fdr_alpha)
    else:
        ctx.logger.info("Moderation: no valid results.")

    return mod_df
