from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from eeg_pipeline.analysis.behavior.result_types import GroupLevelResult
from eeg_pipeline.analysis.behavior.config_resolver import resolve_correlation_method
from eeg_pipeline.utils.analysis.stats.correlation import compute_correlation
from eeg_pipeline.utils.config.loader import get_config_bool, get_config_float, get_config_int, get_config_value
from eeg_pipeline.utils.data.columns import resolve_outcome_column, resolve_predictor_column


_PARTIAL_DESIGN_CONDITION_THRESHOLD = 1e10


def _resolve_group_level_block_column(df: pd.DataFrame, config: Any) -> Optional[str]:
    configured_run_col = str(
        get_config_value(config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id"
    ).strip()
    candidates: List[str] = []
    for candidate in (configured_run_col, "block", "run_id", "run", "session"):
        normalized = str(candidate).strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _build_subject_partial_permutation_state(
    x: pd.Series,
    y: pd.Series,
    cov_df: pd.DataFrame,
    method: str,
) -> Optional[Dict[str, Any]]:
    aligned = pd.concat([x.rename("x"), y.rename("y"), cov_df], axis=1).dropna()
    if len(aligned) < max(3, cov_df.shape[1] + 3):
        return None

    x_values = aligned["x"].to_numpy(dtype=float)
    y_values = aligned["y"].to_numpy(dtype=float)
    z_values = aligned[cov_df.columns].to_numpy(dtype=float)
    method_normalized = str(method or "spearman").strip().lower()
    if method_normalized == "spearman":
        x_values = stats.rankdata(x_values)
        y_values = stats.rankdata(y_values)
        z_values = np.column_stack(
            [stats.rankdata(z_values[:, idx]) for idx in range(z_values.shape[1])]
        )

    design = np.column_stack([np.ones(len(aligned), dtype=float), z_values])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return None

    condition_number = np.linalg.cond(design)
    if not np.isfinite(condition_number) or condition_number > _PARTIAL_DESIGN_CONDITION_THRESHOLD:
        return None

    beta_x = np.linalg.lstsq(design, x_values, rcond=None)[0]
    beta_y = np.linalg.lstsq(design, y_values, rcond=None)[0]
    x_residuals = x_values - design @ beta_x
    y_residuals = y_values - design @ beta_y
    y_fitted = design @ beta_y
    r_observed, _ = compute_correlation(x_residuals, y_residuals, method="pearson")
    if not np.isfinite(r_observed):
        return None

    return {
        "index": aligned.index,
        "design": design,
        "x_residuals": x_residuals,
        "y_residuals": y_residuals,
        "y_fitted": y_fitted,
        "r_observed": float(r_observed),
    }


def _compute_permuted_subject_partial_r(
    partial_state: Dict[str, Any],
    permuted_indices: np.ndarray,
) -> float:
    design = np.asarray(partial_state["design"], dtype=float)
    x_residuals = np.asarray(partial_state["x_residuals"], dtype=float)
    y_residuals = np.asarray(partial_state["y_residuals"], dtype=float)
    y_fitted = np.asarray(partial_state["y_fitted"], dtype=float)
    y_permuted = y_fitted + y_residuals[np.asarray(permuted_indices, dtype=int)]
    beta_y_perm = np.linalg.lstsq(design, y_permuted, rcond=None)[0]
    y_permuted_residuals = y_permuted - design @ beta_y_perm
    r_permuted, _ = compute_correlation(x_residuals, y_permuted_residuals, method="pearson")
    return float(r_permuted) if np.isfinite(r_permuted) else np.nan


def run_group_level_correlations_impl(
    subjects: List[str],
    deriv_root: Path,
    config: Any,
    logger: Any,
    use_block_permutation: bool = True,
    n_perm: int = 1000,
    fdr_alpha: float = 0.05,
    target_col: str = "outcome",
    control_predictor: bool = False,
    control_trial_order: bool = False,
    control_run_effects: bool = False,
    max_run_dummies: int = 20,
    random_state: Optional[int] = None,
    *,
    find_trial_table_path_fn: Callable[[Path, Optional[List[str]]], Optional[Path]],
    feature_prefixes: Sequence[str],
    feature_type_resolver: Callable[[str, Any], str],
    constant_variance_threshold: float,
) -> pd.DataFrame:
    """Run multilevel correlations across subjects with block-aware permutations."""
    from eeg_pipeline.infra.paths import deriv_stats_path
    from eeg_pipeline.infra.tsv import read_table
    from eeg_pipeline.utils.analysis.stats.fdr import hierarchical_fdr
    from eeg_pipeline.utils.analysis.stats.permutation import permute_within_groups

    feature_files = get_config_value(config, "behavior_analysis.feature_files", None)
    if isinstance(feature_files, str):
        feature_files = [feature_files]
    elif feature_files is None:
        feature_files = get_config_value(config, "behavior_analysis.feature_categories", None)
        if isinstance(feature_files, str):
            feature_files = [feature_files]

    all_trials: List[pd.DataFrame] = []

    for sub in subjects:
        stats_dir = deriv_stats_path(deriv_root, sub)
        trial_path = find_trial_table_path_fn(stats_dir, feature_files=feature_files)

        if trial_path is None:
            continue

        df = read_table(trial_path)
        if df is None or df.empty:
            continue

        df["subject_id"] = sub
        all_trials.append(df)

    if len(all_trials) < 2:
        logger.warning("Multilevel correlations require >=2 subjects.")
        return pd.DataFrame()

    combined = pd.concat(all_trials, ignore_index=True)
    correlation_method = resolve_correlation_method(config, logger=logger, default="spearman")

    resolved_target = resolve_outcome_column(combined, config) or "outcome"
    target_column = str(target_col or resolved_target).strip() or resolved_target
    if target_column not in combined.columns:
        logger.warning("Multilevel correlations: target column '%s' not found.", target_column)
        return pd.DataFrame()
    predictor_column = resolve_predictor_column(combined, config) or "predictor"
    block_col = _resolve_group_level_block_column(combined, config)
    if control_run_effects and block_col is None:
        raise ValueError(
            "Group-level correlations requested run adjustment, but no run/block column "
            "could be resolved from the combined trial tables."
        )

    feature_cols = [c for c in combined.columns if str(c).startswith(tuple(feature_prefixes))]
    outcome = pd.to_numeric(combined[target_column], errors="coerce").to_numpy(dtype=float)
    subject_all = combined["subject_id"].astype(str).to_numpy(dtype=object)
    block_all = combined[block_col].to_numpy() if block_col is not None else None

    unique_subjects = np.unique(subject_all)
    subject_indices = {subj: np.where(subject_all == subj)[0] for subj in unique_subjects}

    def _aggregate_subject_rs(r_values: List[float]) -> float:
        arr = np.asarray(r_values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan
        clipped = np.clip(arr, -0.999999, 0.999999)
        z_vals = np.arctanh(clipped)
        return float(np.tanh(np.nanmean(z_vals)))

    if random_state is None:
        seed = get_config_value(config, "project.random_state", 42)
    else:
        seed = random_state
    try:
        seed_int = int(seed) if seed is not None else None
    except (TypeError, ValueError):
        seed_int = None
    rng = np.random.default_rng(seed_int)
    records: List[Dict[str, Any]] = []
    for feat in feature_cols:
        feat_type = feature_type_resolver(str(feat), config)
        family_id = f"corr_{feat_type}"

        feature_vals = pd.to_numeric(combined[feat], errors="coerce").to_numpy(dtype=float)
        if int((np.isfinite(feature_vals) & np.isfinite(outcome)).sum()) < 10:
            continue

        subject_payloads: List[Dict[str, Any]] = []
        used_partial = False
        for subj in unique_subjects:
            idx = subject_indices[subj]
            subj_df = combined.iloc[idx]
            x_sub_s = pd.to_numeric(subj_df[str(feat)], errors="coerce")
            y_sub_s = pd.to_numeric(subj_df[target_column], errors="coerce")

            valid_xy = np.isfinite(x_sub_s.to_numpy(dtype=float)) & np.isfinite(y_sub_s.to_numpy(dtype=float))
            if int(valid_xy.sum()) < 3:
                continue

            x_valid = x_sub_s.loc[valid_xy]
            y_valid = y_sub_s.loc[valid_xy]

            cov_df = pd.DataFrame(index=x_valid.index)
            if (
                control_predictor
                and predictor_column in subj_df.columns
                and target_column != predictor_column
            ):
                cov_df[predictor_column] = pd.to_numeric(
                    subj_df.loc[x_valid.index, predictor_column],
                    errors="coerce",
                )

            if control_trial_order:
                for trial_col in (
                    "trial_index_within_group",
                    "trial_index",
                    "trial_in_run",
                    "trial",
                    "trial_number",
                ):
                    if trial_col in subj_df.columns:
                        cov_df["trial_index"] = pd.to_numeric(subj_df.loc[x_valid.index, trial_col], errors="coerce")
                        break

            if control_run_effects:
                if block_col not in subj_df.columns:
                    raise ValueError(
                        f"Group-level correlations requested run adjustment, but subject '{subj}' "
                        f"is missing run column '{block_col}'."
                    )
                block_sub_s = subj_df.loc[x_valid.index, block_col]
                n_levels = int(pd.Series(block_sub_s).nunique(dropna=True))
                max_levels = max(1, int(max_run_dummies)) + 1
                if n_levels > 1 and n_levels <= max_levels:
                    run_dummies = pd.get_dummies(block_sub_s.astype("category"), prefix=str(block_col), drop_first=True)
                    cov_df = pd.concat([cov_df, run_dummies], axis=1)
                elif n_levels > max_levels:
                    raise ValueError(
                        "Group-level correlations requested run adjustment, but "
                        f"subject '{subj}' has {n_levels} levels in '{block_col}' "
                        f"(> max {max_levels} supported levels for dummy encoding with "
                        f"max_run_dummies={max_run_dummies})."
                    )

            if cov_df is not None and not cov_df.empty:
                cov_df = cov_df.apply(pd.to_numeric, errors="coerce")
                cov_df = cov_df.replace([np.inf, -np.inf], np.nan)
                cov_df = cov_df.dropna(axis=1, how="all")
                if not cov_df.empty:
                    nonconstant = [
                        c
                        for c in cov_df.columns
                        if int(pd.to_numeric(cov_df[c], errors="coerce").nunique(dropna=True)) > 1
                    ]
                    cov_df = cov_df[nonconstant]
                if cov_df.empty:
                    cov_df = None
            else:
                cov_df = None

            if cov_df is not None and not cov_df.empty:
                finite_cov = np.all(np.isfinite(cov_df.to_numpy(dtype=float)), axis=1)
                finite_xy_sub = np.isfinite(x_valid.to_numpy(dtype=float)) & np.isfinite(y_valid.to_numpy(dtype=float))
                valid_final = finite_xy_sub & finite_cov
            else:
                valid_final = np.isfinite(x_valid.to_numpy(dtype=float)) & np.isfinite(y_valid.to_numpy(dtype=float))

            if int(valid_final.sum()) < 3:
                continue

            x_final = pd.to_numeric(x_valid.loc[valid_final], errors="coerce")
            y_final = pd.to_numeric(y_valid.loc[valid_final], errors="coerce")
            cov_final = cov_df.loc[x_final.index] if cov_df is not None else None

            if int(len(x_final)) < 3 or float(np.nanstd(x_final.to_numpy(dtype=float))) <= constant_variance_threshold:
                continue
            if float(np.nanstd(y_final.to_numpy(dtype=float))) <= constant_variance_threshold:
                continue

            if cov_final is not None and not cov_final.empty:
                partial_state = _build_subject_partial_permutation_state(
                    x_final,
                    y_final,
                    cov_final,
                    correlation_method,
                )
                if partial_state is None:
                    continue
                r_sub = float(partial_state["r_observed"])
                uses_partial = True
                analysis_index = partial_state["index"]
            else:
                r_sub, _ = compute_correlation(
                    x_final.to_numpy(dtype=float),
                    y_final.to_numpy(dtype=float),
                    method=correlation_method,
                )
                uses_partial = False
                partial_state = None
                analysis_index = x_final.index

            if not np.isfinite(r_sub):
                continue
            used_partial = used_partial or uses_partial

            block_sub = None
            can_block_permute = False
            if block_all is not None:
                block_sub = np.asarray(subj_df.loc[analysis_index, block_col], dtype=object)
                if use_block_permutation and block_sub.size > 0:
                    _, counts = np.unique(block_sub, return_counts=True)
                    can_block_permute = bool(np.all(counts >= 2))

            subject_payloads.append(
                {
                    "subject": subj,
                    "x_vals": x_final.to_numpy(dtype=float),
                    "y_vals": y_final.to_numpy(dtype=float),
                    "cov_df": cov_final.reset_index(drop=True) if cov_final is not None else None,
                    "partial_state": partial_state,
                    "uses_partial": uses_partial,
                    "r_obs": float(r_sub),
                    "n_obs": int(len(x_final)),
                    "block_sub": block_sub,
                    "can_block_permute": can_block_permute,
                }
            )

        if len(subject_payloads) < 2:
            continue

        r_obs = _aggregate_subject_rs([float(p["r_obs"]) for p in subject_payloads])
        if not np.isfinite(r_obs):
            continue

        n_block_ready = int(sum(1 for p in subject_payloads if p.get("can_block_permute", False)))
        block_permutation_requested = bool(use_block_permutation and block_col is not None)
        perm_method = "subject_block_restricted" if block_permutation_requested else "subject_restricted"

        null_rs: List[float] = []
        permutation_failed = False
        if int(n_perm) > 0:
            if block_permutation_requested and n_block_ready < len(subject_payloads):
                perm_method = "subject_block_restricted_unavailable"
            else:
                for _ in range(int(n_perm)):
                    perm_subject_rs: List[float] = []
                    for payload in subject_payloads:
                        y_vals = np.asarray(payload["y_vals"], dtype=float)
                        x_vals = np.asarray(payload["x_vals"], dtype=float)

                        if block_permutation_requested:
                            if not payload.get("can_block_permute", False) or payload.get("block_sub") is None:
                                permutation_failed = True
                                break
                            try:
                                perm_idx = permute_within_groups(
                                    len(y_vals),
                                    rng,
                                    np.asarray(payload["block_sub"], dtype=object),
                                    scheme="shuffle",
                                    strict=True,
                                )
                            except ValueError:
                                permutation_failed = True
                                break
                        else:
                            perm_idx = rng.permutation(len(y_vals))

                        if payload.get("uses_partial", False):
                            partial_state = payload.get("partial_state")
                            if not isinstance(partial_state, dict):
                                permutation_failed = True
                                break
                            r_perm_sub = _compute_permuted_subject_partial_r(
                                partial_state,
                                perm_idx,
                            )
                        else:
                            y_perm = y_vals[perm_idx]
                            r_perm_sub, _ = compute_correlation(
                                x_vals,
                                y_perm,
                                method=correlation_method,
                            )

                        if np.isfinite(r_perm_sub):
                            perm_subject_rs.append(float(r_perm_sub))

                    if permutation_failed:
                        break
                    if len(perm_subject_rs) < 2:
                        continue
                    r_perm = _aggregate_subject_rs(perm_subject_rs)
                    if np.isfinite(r_perm):
                        null_rs.append(float(r_perm))

        if permutation_failed:
            null_rs = []
            perm_method = "subject_block_restricted_failed" if block_permutation_requested else "subject_restricted_failed"

        n_perm_effective = int(len(null_rs))
        if null_rs:
            p_perm = (np.sum(np.abs(null_rs) >= np.abs(r_obs)) + 1) / (n_perm_effective + 1)
            null_rs_sorted = np.sort(null_rs)
            ci_lower = float(np.percentile(null_rs_sorted, 2.5))
            ci_upper = float(np.percentile(null_rs_sorted, 97.5))
        else:
            p_perm = np.nan
            ci_lower = np.nan
            ci_upper = np.nan
            
        # Parametric fallback using 1-sample t-test on Fisher Z-transformed correlations
        p_parametric = np.nan
        r_observed_list = [float(p["r_obs"]) for p in subject_payloads if np.isfinite(float(p["r_obs"]))]
        if len(r_observed_list) > 1:
            z_obs = np.arctanh(np.clip(np.asarray(r_observed_list), -0.999999, 0.999999))
            if np.std(z_obs) > 1e-10:
                from scipy import stats
                _, p_parametric = stats.ttest_1samp(z_obs, popmean=0.0)
            
        estimator = (
            f"subject_balanced_partial_{correlation_method}"
            if used_partial
            else f"subject_balanced_within_subject_centered_{correlation_method}"
        )

        records.append(
            {
                "feature": str(feat),
                "target": target_column,
                "feature_type": feat_type,
                "family_id": family_id,
                "family_kind": "feature_type",
                "r": float(r_obs),
                "n": int(sum(int(p["n_obs"]) for p in subject_payloads)),
                "n_subjects": int(len(subject_payloads)),
                "estimator": estimator,
                "p_parametric": float(p_parametric),
                "p_perm": p_perm,
                "ci_lower_2_5": ci_lower,
                "ci_upper_97_5": ci_upper,
                "permutation_method": perm_method,
                "n_perm_requested": int(n_perm),
                "n_perm_effective": n_perm_effective,
                "n_perm": n_perm_effective,
            }
        )

    if not records:
        return pd.DataFrame()

    results_df = pd.DataFrame(records)
    
    results_df["p_primary"] = results_df["p_perm"].copy()
    results_df["p_primary_kind"] = "p_perm"
    missing_perm = results_df["p_primary"].isna()
    if missing_perm.any():
        results_df.loc[missing_perm, "p_primary_kind"] = "perm_missing_required"

    return hierarchical_fdr(
        results_df,
        p_col="p_primary",
        family_col="family_id",
        alpha=fdr_alpha,
        config=config,
    )


def run_group_level_analysis_impl(
    subjects: List[str],
    deriv_root: Path,
    config: Any,
    logger: Any,
    run_multilevel_correlations: bool = False,
    output_dir: Optional[Path] = None,
    *,
    run_multilevel_correlations_fn: Callable[..., pd.DataFrame],
    write_parquet_with_optional_csv_fn: Callable[[pd.DataFrame, Path, bool], None],
    also_save_csv_from_config_fn: Callable[[Any], bool],
) -> GroupLevelResult:
    """Run all group-level analyses."""
    from eeg_pipeline.infra.paths import ensure_dir

    logger.info("=" * 60)
    logger.info("Group-Level Behavior Analysis")
    logger.info("=" * 60)
    logger.info("Subjects: %s", ", ".join(subjects))

    multilevel_df = None

    if run_multilevel_correlations:
        logger.info("Running multilevel correlations with block-restricted permutations...")
        gl_corr_cfg = get_config_value(config, "behavior_analysis.group_level.multilevel_correlations", {})
        if not isinstance(gl_corr_cfg, dict):
            gl_corr_cfg = {}

        default_target = str(get_config_value(config, "behavior_analysis.outcome_column", "") or "outcome").strip() or "outcome"
        target_col = str(gl_corr_cfg.get("target", default_target) or default_target).strip()
        control_predictor = bool(
            gl_corr_cfg.get("control_predictor", get_config_bool(config, "behavior_analysis.predictor_control_enabled", True))
        )
        control_trial_order = bool(
            gl_corr_cfg.get("control_trial_order", get_config_bool(config, "behavior_analysis.control_trial_order", True))
        )
        run_adjust_enabled = get_config_bool(config, "behavior_analysis.run_adjustment.enabled", False)
        control_run_effects = bool(
            gl_corr_cfg.get(
                "control_run_effects",
                run_adjust_enabled
                and get_config_bool(config, "behavior_analysis.run_adjustment.include_in_correlations", True),
            )
        )
        max_run_dummies = int(
            gl_corr_cfg.get("max_run_dummies", get_config_int(config, "behavior_analysis.run_adjustment.max_dummies", 20))
        )
        random_state = gl_corr_cfg.get("random_state", get_config_value(config, "project.random_state", None))

        multilevel_df = run_multilevel_correlations_fn(
            subjects=subjects,
            deriv_root=deriv_root,
            config=config,
            logger=logger,
            use_block_permutation=bool(
                gl_corr_cfg.get(
                    "block_permutation",
                    get_config_bool(config, "behavior_analysis.group_level.block_permutation", True),
                )
            ),
            n_perm=get_config_int(config, "behavior_analysis.statistics.n_permutations", 1000),
            fdr_alpha=get_config_float(config, "behavior_analysis.statistics.fdr_alpha", 0.05),
            target_col=target_col,
            control_predictor=control_predictor,
            control_trial_order=control_trial_order,
            control_run_effects=control_run_effects,
            max_run_dummies=max_run_dummies,
            random_state=random_state,
        )

        if output_dir and multilevel_df is not None and not multilevel_df.empty:
            ensure_dir(output_dir)
            out_path = output_dir / "group_multilevel_correlations.parquet"
            write_parquet_with_optional_csv_fn(
                multilevel_df,
                out_path,
                also_save_csv=also_save_csv_from_config_fn(config),
            )
            logger.info("Saved multilevel correlations: %s", out_path)

    return GroupLevelResult(
        multilevel_correlations=multilevel_df,
        n_subjects=len(subjects),
        subjects=subjects,
        metadata={"status": "ok"},
    )
