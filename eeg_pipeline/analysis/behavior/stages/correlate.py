from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from eeg_pipeline.analysis.behavior.config_resolver import resolve_correlation_targets
from eeg_pipeline.utils.analysis.stats.correlation import normalize_robust_correlation_method
from eeg_pipeline.utils.config.loader import get_config_bool, get_config_int, get_config_value


@dataclass
class CorrelateDesign:
    """Design matrix components for correlation analysis."""

    df_trials: pd.DataFrame
    feature_cols: List[str]
    targets: List[str]
    cov_df: Optional[pd.DataFrame]
    temperature_series: Optional[pd.Series]
    predictor_column: str
    run_col: str
    run_adjust_in_correlations: bool
    groups_for_perm: Optional[pd.Series]


def stage_correlate_design_impl(
    ctx: Any,
    config: Any,
    *,
    load_trial_table_df_fn: Callable[[Any], Optional[pd.DataFrame]],
    is_dataframe_valid_fn: Callable[[Optional[pd.DataFrame]], bool],
    get_feature_columns_fn: Callable[[pd.DataFrame, Any, Optional[str]], List[str]],
    sanitize_permutation_groups_fn: Callable[[Any, Any, str], Any],
) -> Optional[CorrelateDesign]:
    """Assemble design matrix: targets, covariates, feature columns."""
    from eeg_pipeline.utils.data.columns import resolve_predictor_column

    df_trials = load_trial_table_df_fn(ctx)
    if is_dataframe_valid_fn(df_trials):
        ctx.logger.info(
            "Correlations design: loaded trial table (%d rows, %d cols)",
            df_trials.shape[0],
            df_trials.shape[1],
        )
    else:
        ctx.logger.warning("Correlations design: trial table missing; skipping.")
        return None

    primary_unit = str(
        get_config_value(ctx.config, "behavior_analysis.correlations.primary_unit", "trial") or "trial"
    ).strip().lower()
    allow_iid_trials = get_config_bool(ctx.config, "behavior_analysis.statistics.allow_iid_trials", False)
    perm_enabled = get_config_bool(ctx.config, "behavior_analysis.correlations.permutation.enabled", False)
    n_perm = get_config_int(
        ctx.config,
        "behavior_analysis.correlations.permutation.n_permutations",
        get_config_int(ctx.config, "behavior_analysis.statistics.n_permutations", 0),
    )
    if primary_unit in {"trial", "trialwise"} and (not perm_enabled or n_perm <= 0) and not allow_iid_trials:
        raise ValueError(
            "Trial-level correlations require a valid non-i.i.d inference method. "
            "Enable permutation testing with a positive permutation count "
            "(behavior_analysis.correlations.permutation.enabled=true and "
            "behavior_analysis.correlations.permutation.n_permutations>0) "
            "or use run-level aggregation (behavior_analysis.correlations.primary_unit=run_mean). "
            "Set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
        )

    feature_cols = get_feature_columns_fn(df_trials, ctx, "correlations")
    if not feature_cols:
        ctx.logger.warning("Correlations design: no feature columns after filtering.")
        return None

    explicit_target_column = str(
        get_config_value(ctx.config, "behavior_analysis.correlations.target_column", "") or ""
    ).strip()
    if explicit_target_column:
        targets = [explicit_target_column]
    else:
        targets = resolve_correlation_targets(
            ctx.config,
            logger=ctx.logger,
            default_targets=[],
        )
        use_cv_resid = get_config_bool(
            ctx.config,
            "behavior_analysis.correlations.use_crossfit_pain_residual",
            True,
        )
        if (not explicit_target_column) and use_cv_resid and "pain_residual_cv" in df_trials.columns:
            targets = ["pain_residual_cv", *[t for t in targets if t != "pain_residual_cv"]]
        prefer_pain_residual = get_config_bool(
            ctx.config,
            "behavior_analysis.correlations.prefer_pain_residual",
            False,
        )
        if prefer_pain_residual:
            preferred_target: Optional[str] = None
            if use_cv_resid and "pain_residual_cv" in df_trials.columns:
                preferred_target = "pain_residual_cv"
            elif "pain_residual" in df_trials.columns:
                preferred_target = "pain_residual"
            if preferred_target is not None:
                updated_targets: List[str] = []
                for target_name in targets:
                    if target_name == "pain_residual" and preferred_target == "pain_residual_cv":
                        if preferred_target not in updated_targets:
                            updated_targets.append(preferred_target)
                    elif target_name != preferred_target:
                        updated_targets.append(target_name)
                if (
                    preferred_target in updated_targets
                    or "pain_residual" in targets
                    or "pain_residual_cv" in targets
                ):
                    targets = [preferred_target, *[t for t in updated_targets if t != preferred_target]]
    targets = [t for t in targets if t in df_trials.columns]

    if not targets:
        ctx.logger.warning("Correlations design: no valid target columns found.")
        return None

    run_adjust_enabled = get_config_bool(ctx.config, "behavior_analysis.run_adjustment.enabled", False)
    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
    if not run_col:
        run_col = "run_id"
    if primary_unit in {"run", "run_mean", "runmean", "run_level"} and run_col not in df_trials.columns:
        raise ValueError(
            f"Run-level correlations requested (primary_unit={primary_unit!r}) "
            f"but run column '{run_col}' is missing from the trial table."
        )
    run_adjust_in_correlations = bool(
        get_config_value(ctx.config, "behavior_analysis.run_adjustment.include_in_correlations", run_adjust_enabled)
    )
    max_run_dummies = get_config_int(ctx.config, "behavior_analysis.run_adjustment.max_dummies", 20)

    cov_parts = []
    if bool(getattr(config, "control_trial_order", True)):
        for c in ["trial_index_within_group", "trial_index"]:
            if c in df_trials.columns:
                cov_parts.append(pd.DataFrame({c: pd.to_numeric(df_trials[c], errors="coerce")}, index=df_trials.index))
                break
    if run_adjust_in_correlations and run_col in df_trials.columns:
        run_s = df_trials[run_col]
        n_levels = int(pd.Series(run_s).nunique(dropna=True))
        if n_levels > 1 and n_levels <= max(1, max_run_dummies + 1):
            run_dum = pd.get_dummies(run_s.astype("category"), prefix=run_col, drop_first=True)
            cov_parts.append(run_dum)
        elif n_levels > max_run_dummies + 1:
            ctx.logger.warning(
                "Correlations design: run adjustment requested but %s has %d levels (> max %d dummies); skipping.",
                run_col,
                n_levels,
                max_run_dummies,
            )
    cov_df = pd.concat(cov_parts, axis=1) if cov_parts else None

    predictor_column = resolve_predictor_column(df_trials, ctx.config) or "temperature"
    temperature_series = None
    if bool(getattr(config, "control_temperature", True)) and predictor_column in df_trials.columns:
        temperature_series = pd.to_numeric(df_trials[predictor_column], errors="coerce")

    groups_for_perm = None
    if getattr(ctx, "group_ids", None) is not None:
        groups_candidate = np.asarray(ctx.group_ids)
        if len(groups_candidate) == len(df_trials):
            groups_for_perm = groups_candidate
        else:
            ctx.logger.warning(
                "Correlations design: ignoring ctx.group_ids length=%d because trial table has %d rows.",
                len(groups_candidate),
                len(df_trials),
            )
    if groups_for_perm is None and run_col in df_trials.columns:
        groups_for_perm = df_trials[run_col].to_numpy()
    groups_for_perm = sanitize_permutation_groups_fn(
        groups_for_perm,
        ctx.logger,
        "Correlations",
    )
    if primary_unit in {"trial", "trialwise"} and not allow_iid_trials and perm_enabled and groups_for_perm is None:
        raise ValueError(
            "Trial-level correlations require grouped permutation labels for non-i.i.d trials. "
            "Provide behavior_analysis.run_adjustment.column in the trial table (or ctx.group_ids), "
            "or set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
        )
    groups_for_perm_series = (
        pd.Series(groups_for_perm, index=df_trials.index, name=run_col)
        if groups_for_perm is not None
        else None
    )

    ctx.logger.info(
        "Correlations design: %d features x %d targets, covariates=%s, temp_control=%s",
        len(feature_cols),
        len(targets),
        cov_df.shape[1] if cov_df is not None else 0,
        temperature_series is not None,
    )

    return CorrelateDesign(
        df_trials=df_trials,
        feature_cols=feature_cols,
        targets=targets,
        cov_df=cov_df,
        temperature_series=temperature_series,
        predictor_column=predictor_column,
        run_col=run_col,
        run_adjust_in_correlations=run_adjust_in_correlations,
        groups_for_perm=groups_for_perm_series,
    )


def _compute_single_effect_size(
    feat: str,
    target: str,
    df_trials: pd.DataFrame,
    cov_df: Optional[pd.DataFrame],
    temperature_series: Optional[pd.Series],
    predictor_column: str,
    run_col: str,
    run_adjust_in_correlations: bool,
    method: str,
    robust_method: Optional[str],
    method_label: str,
    min_samples: int,
    run_min_samples: int,
    want_raw: bool,
    want_partial_cov: bool,
    want_partial_temp: bool,
    want_partial_cov_temp: bool,
    want_run_mean: bool,
    config: Any,
    feature_type_resolver_fn: Callable[[str, Any], str],
    feature_band_resolver_fn: Callable[[str, Any], str],
) -> Dict[str, Any]:
    """Compute effect size for a single feature-target pair."""
    from eeg_pipeline.utils.analysis.stats import compute_partial_correlations_with_cov_temp
    from eeg_pipeline.utils.analysis.stats.correlation import safe_correlation

    constant_variance_threshold = 1e-12

    x = pd.to_numeric(df_trials[feat], errors="coerce")
    y = pd.to_numeric(df_trials[target], errors="coerce")
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    finite_xy = np.isfinite(x_arr) & np.isfinite(y_arr)

    skip_reason = None
    if int(finite_xy.sum()) >= min_samples:
        feature_std = float(np.nanstd(x_arr[finite_xy], ddof=1))
        target_std = float(np.nanstd(y_arr[finite_xy], ddof=1))
        if feature_std <= constant_variance_threshold:
            skip_reason = "feature_constant"
        elif target_std <= constant_variance_threshold:
            skip_reason = "target_constant"

    r_raw, p_raw, n = safe_correlation(x_arr, y_arr, method, min_samples, robust_method=robust_method)

    rec: Dict[str, Any] = {
        "feature": str(feat),
        "feature_type": feature_type_resolver_fn(str(feat), config),
        "band": feature_band_resolver_fn(str(feat), config),
        "target": str(target),
        "method": method,
        "robust_method": robust_method,
        "method_label": method_label,
        "n": int(n),
        "r_raw": float(r_raw) if np.isfinite(r_raw) else np.nan if want_raw else np.nan,
        "p_raw": float(p_raw) if np.isfinite(p_raw) else np.nan if want_raw else np.nan,
        "r": float(r_raw) if np.isfinite(r_raw) else np.nan if want_raw else np.nan,
        "p": float(p_raw) if np.isfinite(p_raw) else np.nan if want_raw else np.nan,
        "p_value": float(p_raw) if np.isfinite(p_raw) else np.nan if want_raw else np.nan,
        "skip_reason": skip_reason,
        "run_adjustment_enabled": bool(run_adjust_in_correlations and run_col in df_trials.columns),
        "run_column": run_col if run_col in df_trials.columns else None,
    }

    temp_for_partial = (
        temperature_series
        if (temperature_series is not None and target != predictor_column)
        else None
    )

    if want_partial_cov or want_partial_temp or want_partial_cov_temp:
        r_pc, p_pc, n_pc, r_pt, p_pt, n_pt, r_pct, p_pct, n_pct = compute_partial_correlations_with_cov_temp(
            roi_values=x,
            target_values=y,
            covariates_df=cov_df,
            temperature_series=temp_for_partial,
            method=method,
            context="trial_table",
            logger=None,
            min_samples=min_samples,
            config=config,
        )

        if want_partial_cov or want_partial_cov_temp:
            rec.update(
                {
                    "r_partial_cov": r_pc,
                    "p_partial_cov": p_pc,
                    "n_partial_cov": n_pc,
                }
            )

        if want_partial_temp or want_partial_cov_temp:
            rec.update(
                {
                    "r_partial_temp": r_pt,
                    "p_partial_temp": p_pt,
                    "n_partial_temp": n_pt,
                }
            )

        if want_partial_cov_temp:
            rec.update(
                {
                    "r_partial_cov_temp": r_pct,
                    "p_partial_cov_temp": p_pct,
                    "n_partial_cov_temp": n_pct,
                }
            )

    if want_run_mean and run_col in df_trials.columns:
        df_run = pd.DataFrame({run_col: df_trials[run_col], "x": x, "y": y})
        run_means = df_run.groupby(run_col, dropna=True)[["x", "y"]].mean(numeric_only=True)
        r_run, p_run, n_run = safe_correlation(
            run_means["x"].to_numpy(dtype=float),
            run_means["y"].to_numpy(dtype=float),
            method,
            min_samples=max(int(run_min_samples), 3),
            robust_method=None,
        )
        rec.update(
            {
                "n_runs": int(n_run),
                "r_run_mean": float(r_run) if np.isfinite(r_run) else np.nan,
                "p_run_mean": float(p_run) if np.isfinite(p_run) else np.nan,
            }
        )

    return rec


def stage_correlate_effect_sizes_impl(
    ctx: Any,
    config: Any,
    design: CorrelateDesign,
    *,
    feature_type_resolver_fn: Callable[[str, Any], str],
    feature_band_resolver_fn: Callable[[str, Any], str],
) -> List[Dict[str, Any]]:
    """Compute raw and partial correlation effect sizes."""
    if design is None:
        ctx.logger.warning("Correlations effect sizes: design missing; skipping.")
        return []

    from joblib import Parallel, delayed
    from eeg_pipeline.utils.parallel import _normalize_n_jobs, get_n_jobs

    method = getattr(config, "method", "spearman")
    robust_method = getattr(config, "robust_method", None)
    method_label = getattr(config, "method_label", "")
    min_samples = int(getattr(config, "min_samples", 10))
    run_min_samples = get_config_int(
        ctx.config,
        "behavior_analysis.correlations.min_runs",
        max(int(min_samples), 3),
    )

    correlation_types = get_config_value(
        ctx.config,
        "behavior_analysis.correlations.types",
        ["partial_cov_temp"],
    )
    if not isinstance(correlation_types, (list, tuple)):
        correlation_types = [correlation_types]

    want_raw = "raw" in correlation_types
    want_partial_cov = "partial_cov" in correlation_types
    want_partial_temp = "partial_temp" in correlation_types
    want_partial_cov_temp = "partial_cov_temp" in correlation_types
    primary_unit = str(
        get_config_value(ctx.config, "behavior_analysis.correlations.primary_unit", "trial") or "trial"
    ).strip().lower()
    want_run_mean = ("run_mean" in correlation_types) or (
        primary_unit in {"run", "run_mean", "runmean", "run_level"}
    )

    has_covariate_controls = design.cov_df is not None and not design.cov_df.empty
    has_temperature_control = bool(
        design.temperature_series is not None
        and any(str(t) != design.predictor_column for t in design.targets)
    )

    if robust_method not in (None, "", False):
        if has_covariate_controls or has_temperature_control:
            raise ValueError(
                "Correlations: robust correlation with covariate/temperature controls is not supported. "
                "Disable robust correlation or run without partial controls to avoid confounded primary effects."
            )
        if want_partial_cov or want_partial_temp or want_partial_cov_temp:
            ctx.logger.info(
                "Correlations: robust_method=%s disables partial correlations; using raw only.",
                robust_method,
            )
        want_raw = True
        want_partial_cov = False
        want_partial_temp = False
        want_partial_cov_temp = False

    tasks = [(feat, target) for target in design.targets for feat in design.feature_cols]
    n_tasks = len(tasks)
    n_jobs = get_n_jobs(ctx.config, default=-1, config_path="behavior_analysis.n_jobs")
    n_jobs_actual = _normalize_n_jobs(n_jobs)

    ctx.logger.info(
        "Correlations effect sizes: %d feature-target pairs, n_jobs=%d",
        n_tasks,
        n_jobs_actual,
    )

    if n_tasks == 0:
        return []

    if n_jobs_actual > 1 and n_tasks >= 100:
        records = Parallel(n_jobs=n_jobs_actual, backend="loky")(
            delayed(_compute_single_effect_size)(
                feat=feat,
                target=target,
                df_trials=design.df_trials,
                cov_df=design.cov_df,
                temperature_series=design.temperature_series,
                predictor_column=design.predictor_column,
                run_col=design.run_col,
                run_adjust_in_correlations=design.run_adjust_in_correlations,
                method=method,
                robust_method=robust_method,
                method_label=method_label,
                min_samples=min_samples,
                run_min_samples=run_min_samples,
                want_raw=want_raw,
                want_partial_cov=want_partial_cov,
                want_partial_temp=want_partial_temp,
                want_partial_cov_temp=want_partial_cov_temp,
                want_run_mean=want_run_mean,
                config=ctx.config,
                feature_type_resolver_fn=feature_type_resolver_fn,
                feature_band_resolver_fn=feature_band_resolver_fn,
            )
            for feat, target in tasks
        )
    else:
        records = [
            _compute_single_effect_size(
                feat=feat,
                target=target,
                df_trials=design.df_trials,
                cov_df=design.cov_df,
                temperature_series=design.temperature_series,
                predictor_column=design.predictor_column,
                run_col=design.run_col,
                run_adjust_in_correlations=design.run_adjust_in_correlations,
                method=method,
                robust_method=robust_method,
                method_label=method_label,
                min_samples=min_samples,
                run_min_samples=run_min_samples,
                want_raw=want_raw,
                want_partial_cov=want_partial_cov,
                want_partial_temp=want_partial_temp,
                want_partial_cov_temp=want_partial_cov_temp,
                want_run_mean=want_run_mean,
                config=ctx.config,
                feature_type_resolver_fn=feature_type_resolver_fn,
                feature_band_resolver_fn=feature_band_resolver_fn,
            )
            for feat, target in tasks
        ]

    ctx.logger.info("Correlations effect sizes: computed %d feature-target pairs", len(records))
    return records


def _compute_single_pvalue(
    rec: Dict[str, Any],
    df_trials: pd.DataFrame,
    df_index: pd.Index,
    cov_df: Optional[pd.DataFrame],
    temperature_series: Optional[pd.Series],
    predictor_column: str,
    groups_for_perm: Optional[pd.Series],
    method: str,
    robust_method: Optional[str],
    n_perm: int,
    perm_scheme: str,
    rng_seed: int,
    config: Any,
    perm_ok_robust: bool,
) -> Dict[str, Any]:
    """Compute permutation p-values for a single record."""
    from eeg_pipeline.utils.analysis.stats.permutation import compute_permutation_pvalues_with_cov_temp

    feat = rec["feature"]
    target = rec["target"]
    r_raw = rec.get("r_raw", np.nan)
    n = rec.get("n", 0)

    result = rec.copy()

    if not (np.isfinite(r_raw) and int(n) > 0):
        result.update(
            {
                "n_permutations": int(n_perm),
                "p_perm_raw": np.nan,
                "p_perm_partial_cov": np.nan,
                "p_perm_partial_temp": np.nan,
                "p_perm_partial_cov_temp": np.nan,
            }
        )
        return result

    rng = np.random.default_rng(rng_seed)
    x = pd.to_numeric(df_trials[feat], errors="coerce")
    y = pd.to_numeric(df_trials[target], errors="coerce")

    if perm_ok_robust:
        from eeg_pipeline.utils.analysis.stats.correlation import compute_robust_correlation
        from eeg_pipeline.utils.analysis.stats.permutation import permute_within_groups

        x_vec = x.to_numpy(dtype=float)
        y_vec = y.to_numpy(dtype=float)
        valid = np.isfinite(x_vec) & np.isfinite(y_vec)

        if valid.sum() < 4:
            p_perm_raw = np.nan
        else:
            x_v = x_vec[valid]
            y_v = y_vec[valid]
            groups_v = np.asarray(groups_for_perm)[valid] if groups_for_perm is not None else None

            r_obs, _ = compute_robust_correlation(x_v, y_v, method=str(robust_method).strip().lower())
            if not np.isfinite(r_obs):
                p_perm_raw = np.nan
            else:
                extreme = 0
                for _ in range(int(n_perm)):
                    try:
                        perm_idx = permute_within_groups(
                            len(y_v),
                            rng,
                            groups_v,
                            scheme=perm_scheme,
                            strict=True,
                        )
                    except ValueError:
                        p_perm_raw = np.nan
                        break
                    y_perm = y_v[perm_idx]
                    r_perm, _ = compute_robust_correlation(x_v, y_perm, method=str(robust_method).strip().lower())
                    if np.isfinite(r_perm) and abs(r_perm) >= abs(r_obs):
                        extreme += 1
                else:
                    p_perm_raw = float((extreme + 1) / (int(n_perm) + 1))

        result.update(
            {
                "n_permutations": int(n_perm),
                "p_perm_raw": float(p_perm_raw) if np.isfinite(p_perm_raw) else np.nan,
                "p_perm_partial_cov": np.nan,
                "p_perm_partial_temp": np.nan,
                "p_perm_partial_cov_temp": np.nan,
            }
        )
    else:
        temp_for_partial = (
            temperature_series
            if (temperature_series is not None and target != predictor_column)
            else None
        )
        p_perm, p_perm_cov, p_perm_temp, p_perm_cov_temp = compute_permutation_pvalues_with_cov_temp(
            x_aligned=pd.Series(x.to_numpy(dtype=float), index=df_index),
            y_aligned=pd.Series(y.to_numpy(dtype=float), index=df_index),
            covariates_df=cov_df,
            temp_series=temp_for_partial,
            method=method.strip().lower(),
            n_perm=n_perm,
            n_eff=int(n),
            rng=rng,
            config=config,
            groups=groups_for_perm,
        )
        result.update(
            {
                "n_permutations": int(n_perm),
                "p_perm_raw": float(p_perm) if np.isfinite(p_perm) else np.nan,
                "p_perm_partial_cov": float(p_perm_cov) if np.isfinite(p_perm_cov) else np.nan,
                "p_perm_partial_temp": float(p_perm_temp) if np.isfinite(p_perm_temp) else np.nan,
                "p_perm_partial_cov_temp": float(p_perm_cov_temp) if np.isfinite(p_perm_cov_temp) else np.nan,
            }
        )

    return result


def stage_correlate_pvalues_impl(
    ctx: Any,
    config: Any,
    design: CorrelateDesign,
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compute permutation p-values for correlations."""
    if not records or not isinstance(records, list):
        ctx.logger.warning("Correlations pvalues: no valid records; skipping.")
        return []

    from joblib import Parallel, delayed
    from eeg_pipeline.utils.parallel import _normalize_n_jobs, get_n_jobs

    method = getattr(config, "method", "spearman")
    robust_method = getattr(config, "robust_method", None)
    perm_enabled = get_config_bool(ctx.config, "behavior_analysis.correlations.permutation.enabled", False)
    n_perm = get_config_int(ctx.config, "behavior_analysis.correlations.permutation.n_permutations", ctx.n_perm or 0)
    perm_scheme = str(get_config_value(ctx.config, "behavior_analysis.permutation.scheme", "shuffle") or "shuffle").strip().lower()

    perm_ok_standard = (
        perm_enabled
        and n_perm > 0
        and (robust_method in (None, "", False))
        and isinstance(method, str)
        and method.strip().lower() in {"spearman", "pearson"}
    )
    perm_ok_robust = (
        perm_enabled
        and n_perm > 0
        and (robust_method not in (None, "", False))
        and isinstance(robust_method, str)
        and robust_method.strip().lower() in {"percentage_bend", "winsorized", "shepherd"}
    )

    if not (perm_ok_standard or perm_ok_robust):
        if perm_enabled and robust_method not in (None, "", False):
            ctx.logger.debug("Correlations pvalues: permutation disabled for robust_method=%s", robust_method)
        for rec in records:
            rec.update(
                {
                    "n_permutations": int(n_perm) if perm_enabled else 0,
                    "p_perm_raw": np.nan,
                    "p_perm_partial_cov": np.nan,
                    "p_perm_partial_temp": np.nan,
                    "p_perm_partial_cov_temp": np.nan,
                }
            )
        return records

    base_seed = 42 if ctx.rng is None else int(ctx.rng.integers(0, 2**31))
    n_records = len(records)
    n_jobs = get_n_jobs(ctx.config, default=-1, config_path="behavior_analysis.n_jobs")
    n_jobs_actual = _normalize_n_jobs(n_jobs)

    ctx.logger.info(
        "Correlations pvalues: %d records, n_perm=%d, n_jobs=%d",
        n_records,
        n_perm,
        n_jobs_actual,
    )

    if n_jobs_actual > 1 and n_records >= 100:
        updated_records = Parallel(n_jobs=n_jobs_actual, backend="loky")(
            delayed(_compute_single_pvalue)(
                rec=rec,
                df_trials=design.df_trials,
                df_index=design.df_trials.index,
                cov_df=design.cov_df,
                temperature_series=design.temperature_series,
                predictor_column=design.predictor_column,
                groups_for_perm=design.groups_for_perm,
                method=method,
                robust_method=robust_method,
                n_perm=n_perm,
                perm_scheme=perm_scheme,
                rng_seed=base_seed + i,
                config=ctx.config,
                perm_ok_robust=perm_ok_robust,
            )
            for i, rec in enumerate(records)
        )
    else:
        updated_records = [
            _compute_single_pvalue(
                rec=rec,
                df_trials=design.df_trials,
                df_index=design.df_trials.index,
                cov_df=design.cov_df,
                temperature_series=design.temperature_series,
                predictor_column=design.predictor_column,
                groups_for_perm=design.groups_for_perm,
                method=method,
                robust_method=robust_method,
                n_perm=n_perm,
                perm_scheme=perm_scheme,
                rng_seed=base_seed + i,
                config=ctx.config,
                perm_ok_robust=perm_ok_robust,
            )
            for i, rec in enumerate(records)
        ]

    n_computed = sum(1 for r in updated_records if np.isfinite(r.get("p_perm_raw", np.nan)))
    ctx.logger.info("Correlations pvalues: computed %d permutation tests (n_perm=%d)", n_computed, n_perm)
    return updated_records


def stage_correlate_primary_selection_impl(
    ctx: Any,
    config: Any,
    design: CorrelateDesign,
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Select primary p-value and effect size for each test."""
    if not records or not isinstance(records, list):
        ctx.logger.warning("Correlations primary selection: no valid records; skipping.")
        return []

    p_primary_mode = str(get_config_value(ctx.config, "behavior_analysis.correlations.p_primary_mode", "perm_if_available")).strip().lower()
    primary_unit = str(get_config_value(ctx.config, "behavior_analysis.correlations.primary_unit", "trial")).strip().lower()
    use_run_unit = primary_unit in {"run", "run_mean", "runmean", "run_level"}
    allow_iid_trials = get_config_bool(ctx.config, "behavior_analysis.statistics.allow_iid_trials", False)
    if (not use_run_unit) and (not allow_iid_trials):
        if p_primary_mode not in {"perm", "permutation"}:
            ctx.logger.warning(
                "Correlations: overriding p_primary_mode=%r to 'perm' under non-i.i.d trial-level mode.",
                p_primary_mode,
            )
        p_primary_mode = "perm"
    strict_perm_mode = p_primary_mode in {"perm", "permutation"}

    for rec in records:
        target = rec.get("target", "")
        p_kind = "p_raw"
        p_primary = rec["p_raw"]
        r_primary = rec["r_raw"]
        src = "raw"

        robust_method = rec.get("robust_method", None)
        if use_run_unit:
            p_kind = "p_run_mean"
            p_primary = rec.get("p_run_mean", np.nan)
            r_primary = rec.get("r_run_mean", np.nan)
            src = "run_mean"
            if not (pd.notna(p_primary) and np.isfinite(float(p_primary))):
                p_primary = np.nan
                r_primary = np.nan
                src = "run_mean_missing"
        elif robust_method not in (None, "", False):
            p_kind = "p_raw"
            p_primary = rec.get("p_raw", np.nan)
            r_primary = rec.get("r_raw", np.nan)
            src = "raw_robust"
            if p_primary_mode in {"perm", "permutation", "perm_if_available", "permutation_if_available"}:
                p_perm_raw = rec.get("p_perm_raw", np.nan)
                if pd.notna(p_perm_raw):
                    p_kind = "p_perm_raw"
                    p_primary = p_perm_raw
                    src = "raw_robust_perm"
                elif strict_perm_mode:
                    p_kind = "perm_missing_required"
                    p_primary = np.nan
                    src = "perm_missing_required"
        else:
            want_partial_cov = design.cov_df is not None and not design.cov_df.empty
            want_partial_temp = (
                bool(getattr(config, "control_temperature", True))
                and target != design.predictor_column
                and design.temperature_series is not None
            )

            if want_partial_temp and want_partial_cov:
                p_kind = "p_partial_cov_temp"
                p_primary = rec.get("p_partial_cov_temp", np.nan)
                r_primary = rec.get("r_partial_cov_temp", np.nan)
                src = "partial_cov_temp"
            elif want_partial_temp:
                p_kind = "p_partial_temp"
                p_primary = rec.get("p_partial_temp", np.nan)
                r_primary = rec.get("r_partial_temp", np.nan)
                src = "partial_temp"
            elif want_partial_cov:
                p_kind = "p_partial_cov"
                p_primary = rec.get("p_partial_cov", np.nan)
                r_primary = rec.get("r_partial_cov", np.nan)
                src = "partial_cov"

            if p_primary_mode in {"perm", "permutation", "perm_if_available", "permutation_if_available"}:
                perm_map = {
                    "p_raw": "p_perm_raw",
                    "p_partial_cov": "p_perm_partial_cov",
                    "p_partial_temp": "p_perm_partial_temp",
                    "p_partial_cov_temp": "p_perm_partial_cov_temp",
                }
                perm_key = perm_map.get(p_kind)
                if perm_key and pd.notna(rec.get(perm_key, np.nan)):
                    p_kind = perm_key
                    p_primary = rec.get(perm_key, np.nan)
                    src = f"{src}_perm"
                elif strict_perm_mode:
                    p_kind = "perm_missing_required"
                    p_primary = np.nan
                    src = "perm_missing_required"

            if not (pd.notna(p_primary) and np.isfinite(float(p_primary))):
                p_primary = np.nan
                if p_kind == "perm_missing_required":
                    src = "perm_missing_required"
                else:
                    r_primary = np.nan
                    src = f"{src}_missing"

        rec["p_kind_primary"] = p_kind
        rec["p_primary"] = p_primary
        rec["r_primary"] = r_primary
        rec["p_primary_source"] = src

    return records


def stage_correlate_fdr_impl(
    ctx: Any,
    config: Any,
    records: List[Dict[str, Any]],
    *,
    compute_unified_fdr_fn: Callable[..., pd.DataFrame],
    unified_fdr_family_columns: Sequence[str],
) -> pd.DataFrame:
    """Apply unified FDR correction with explicit family structure."""
    if records is None:
        raise ValueError("Correlations FDR: records is None")
    if not isinstance(records, list):
        raise TypeError(f"Correlations FDR: records must be a list, got {type(records)!r}")
    if not records:
        ctx.logger.info("Correlations FDR: no records; skipping.")
        return pd.DataFrame()

    try:
        corr_df = pd.DataFrame(records)
    except Exception as exc:
        raise ValueError("Correlations FDR: failed to build DataFrame from records") from exc

    if corr_df.empty:
        return corr_df

    if "p_primary" not in corr_df.columns:
        ctx.logger.error("Missing 'p_primary' column. Ensure 'correlate_primary_selection' stage runs before 'correlate_fdr'.")
        raise KeyError("p_primary")

    if "analysis_kind" not in corr_df.columns:
        corr_df["analysis_kind"] = "correlation"

    return compute_unified_fdr_fn(
        ctx,
        config,
        corr_df,
        p_col="p_primary",
        family_cols=list(unified_fdr_family_columns),
        analysis_type="correlations",
    )


def stage_correlate_impl(
    ctx: Any,
    config: Any,
    *,
    stage_correlate_design_fn: Callable[[Any, Any], Optional[CorrelateDesign]],
    stage_correlate_effect_sizes_fn: Callable[[Any, Any, CorrelateDesign], List[Dict[str, Any]]],
    stage_correlate_pvalues_fn: Callable[[Any, Any, CorrelateDesign, List[Dict[str, Any]]], List[Dict[str, Any]]],
    stage_correlate_primary_selection_fn: Callable[[Any, Any, CorrelateDesign, List[Dict[str, Any]]], List[Dict[str, Any]]],
    stage_correlate_fdr_fn: Callable[[Any, Any, List[Dict[str, Any]]], pd.DataFrame],
) -> pd.DataFrame:
    """Composed correlations stage."""
    design = stage_correlate_design_fn(ctx, config)
    if design is None:
        return pd.DataFrame()

    records = stage_correlate_effect_sizes_fn(ctx, config, design)
    records = stage_correlate_pvalues_fn(ctx, config, design, records)
    records = stage_correlate_primary_selection_fn(ctx, config, design, records)
    return stage_correlate_fdr_fn(ctx, config, records)


def stage_pain_sensitivity_impl(
    ctx: Any,
    config: Any,
    *,
    load_trial_table_df_fn: Callable[[Any], Optional[pd.DataFrame]],
    is_dataframe_valid_fn: Callable[[Optional[pd.DataFrame]], bool],
    get_feature_columns_fn: Callable[[pd.DataFrame, Any, Optional[str]], List[str]],
    sanitize_permutation_groups_fn: Callable[[Any, Any, str], Any],
    compute_unified_fdr_fn: Callable[..., pd.DataFrame],
    unified_fdr_family_columns: Sequence[str],
) -> pd.DataFrame:
    """Compute pain sensitivity correlations (independent stage)."""
    from eeg_pipeline.analysis.behavior.api import run_pain_sensitivity_correlations
    from eeg_pipeline.utils.data.columns import (
        resolve_outcome_column,
        resolve_predictor_column,
    )

    df_trials = load_trial_table_df_fn(ctx)
    if not is_dataframe_valid_fn(df_trials):
        ctx.logger.warning("Pain sensitivity: trial table missing; skipping.")
        return pd.DataFrame()

    predictor_column = resolve_predictor_column(df_trials, ctx.config) or "temperature"
    outcome_column = resolve_outcome_column(df_trials, ctx.config) or "rating"

    required_columns = {predictor_column, outcome_column}
    missing_columns = required_columns - set(df_trials.columns)
    if missing_columns:
        ctx.logger.warning(
            "Sensitivity stage requires predictor/outcome columns %s; missing: %s. Skipping.",
            required_columns,
            missing_columns,
        )
        return pd.DataFrame()

    method = getattr(config, "method", "spearman")
    primary_unit = str(
        get_config_value(ctx.config, "behavior_analysis.pain_sensitivity.primary_unit", "trial") or "trial"
    ).strip().lower()
    use_run_unit = primary_unit in {"run", "run_mean", "runmean", "run_level"}
    allow_iid_trials = get_config_bool(ctx.config, "behavior_analysis.statistics.allow_iid_trials", False)
    n_perm = get_config_int(
        ctx.config,
        "behavior_analysis.pain_sensitivity.n_permutations",
        get_config_int(ctx.config, "behavior_analysis.statistics.n_permutations", 0),
    )
    p_primary_mode = str(
        get_config_value(ctx.config, "behavior_analysis.pain_sensitivity.p_primary_mode", "perm_if_available")
        or "perm_if_available"
    ).strip().lower()
    if primary_unit in {"trial", "trialwise"} and (not allow_iid_trials):
        if p_primary_mode not in {"perm", "permutation"}:
            ctx.logger.warning(
                "Pain sensitivity: overriding p_primary_mode=%r to 'perm' under non-i.i.d trial-level mode.",
                p_primary_mode,
            )
        p_primary_mode = "perm"
    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
    perm_scheme = str(get_config_value(ctx.config, "behavior_analysis.permutation.scheme", "shuffle") or "shuffle").strip().lower()
    if use_run_unit and run_col not in df_trials.columns:
        raise ValueError(
            f"Run-level pain sensitivity requested but run column '{run_col}' is missing from trial table."
        )
    if primary_unit in {"trial", "trialwise"} and not allow_iid_trials and n_perm <= 0:
        raise ValueError(
            "Trial-level pain sensitivity requires a valid non-i.i.d inference method. "
            "Set behavior_analysis.pain_sensitivity.n_permutations > 0, "
            "use run-level aggregation (behavior_analysis.pain_sensitivity.primary_unit=run_mean), "
            "or set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
        )

    if use_run_unit:
        ctx.logger.info("Pain sensitivity: aggregating to run-level (primary_unit=%s)", primary_unit)
        psi_feature_cols_run = get_feature_columns_fn(df_trials, ctx, "pain_sensitivity")
        agg_cols = [
            c
            for c in (
                psi_feature_cols_run
                + [outcome_column, predictor_column]
            )
            if c in df_trials.columns
        ]
        if not agg_cols:
            ctx.logger.warning("Pain sensitivity: no aggregatable columns found; skipping.")
            return pd.DataFrame()
        df_trials = df_trials.groupby(run_col)[agg_cols].mean(numeric_only=True).reset_index()

    robust_method_cfg = get_config_value(ctx.config, "behavior_analysis.robust_correlation", None)
    robust_method_cfg = normalize_robust_correlation_method(
        robust_method_cfg,
        default=None,
        strict=True,
    )

    psi_feature_cols = get_feature_columns_fn(df_trials, ctx, "pain_sensitivity")

    if not psi_feature_cols:
        ctx.logger.warning("Pain sensitivity: no feature columns found; skipping.")
        return pd.DataFrame()

    ctx.logger.info("Pain sensitivity: analyzing %d features", len(psi_feature_cols))
    psi_features = df_trials[psi_feature_cols].copy()
    groups_for_perm = None
    if getattr(ctx, "group_ids", None) is not None:
        groups_candidate = np.asarray(ctx.group_ids)
        if len(groups_candidate) == len(df_trials):
            groups_for_perm = groups_candidate
        else:
            ctx.logger.warning(
                "Pain sensitivity: ignoring ctx.group_ids length=%d because trial table has %d rows.",
                len(groups_candidate),
                len(df_trials),
            )
    if groups_for_perm is None and run_col in df_trials.columns:
        groups_for_perm = df_trials[run_col].to_numpy()
    groups_for_perm = sanitize_permutation_groups_fn(
        groups_for_perm,
        ctx.logger,
        "Pain sensitivity",
    )
    if primary_unit in {"trial", "trialwise"} and not allow_iid_trials and groups_for_perm is None:
        raise ValueError(
            "Trial-level pain sensitivity requires grouped permutation labels for non-i.i.d trials. "
            "Provide behavior_analysis.run_adjustment.column in the trial table (or ctx.group_ids), "
            "or set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
        )
    psi_df = run_pain_sensitivity_correlations(
        features_df=psi_features,
        ratings=pd.to_numeric(df_trials[outcome_column], errors="coerce"),
        temperatures=pd.to_numeric(df_trials[predictor_column], errors="coerce"),
        method=method,
        robust_method=robust_method_cfg,
        min_samples=int(getattr(config, "min_samples", 10)),
        logger=ctx.logger,
        config=ctx.config,
        n_perm=n_perm,
        groups=groups_for_perm,
        permutation_scheme=perm_scheme,
        p_primary_mode=p_primary_mode,
        rng=getattr(ctx, "rng", None),
    )

    if is_dataframe_valid_fn(psi_df):
        if "analysis_kind" not in psi_df.columns:
            psi_df["analysis_kind"] = "pain_sensitivity"

        if "p_primary" in psi_df.columns:
            psi_df["p_primary"] = pd.to_numeric(psi_df["p_primary"], errors="coerce")
        else:
            p_column = next((col for col in ["p_psi", "p_value", "p"] if col in psi_df.columns), None)
            if p_column:
                psi_df["p_primary"] = pd.to_numeric(psi_df[p_column], errors="coerce")

        if "p_raw" not in psi_df.columns:
            p_raw_col = next((col for col in ["p_psi", "p_value", "p"] if col in psi_df.columns), None)
            if p_raw_col:
                psi_df["p_raw"] = pd.to_numeric(psi_df[p_raw_col], errors="coerce")

        if "p_primary" in psi_df.columns:
            psi_df = compute_unified_fdr_fn(
                ctx,
                config,
                psi_df,
                p_col="p_primary",
                family_cols=list(unified_fdr_family_columns),
                analysis_type="pain_sensitivity",
            )

    return psi_df if is_dataframe_valid_fn(psi_df) else pd.DataFrame()
