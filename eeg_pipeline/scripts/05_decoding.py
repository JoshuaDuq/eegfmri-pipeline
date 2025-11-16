from __future__ import annotations

# Standard library
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import logging
import warnings
import threading
import random as pyrandom

# Third-party
import numpy as np
import pandas as pd
import mne
import scipy
from scipy.stats import pearsonr, spearmanr, ConstantInputWarning, kendalltau, rankdata, norm
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, GridSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.metrics import r2_score, make_scorer, explained_variance_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import VarianceThreshold
from joblib import Parallel, delayed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Local - config and data
from eeg_pipeline.utils.config_loader import load_settings
from eeg_pipeline.utils.data_loading import (
    load_multiple_subjects_decoding_data,
    load_epochs_with_targets,
    resolve_columns,
    validate_data_not_empty,
    validate_data_lengths,
    validate_trial_ids,
    validate_sufficient_subjects,
)
from eeg_pipeline.utils.io_utils import (
    _find_clean_epochs_path,
    ensure_dir,
    fdr_bh,
    deriv_stats_path,
    save_fig as _central_save_fig,
    build_footer as _build_footer,
)
from eeg_pipeline.utils.decoding_utils import (
    create_base_preprocessing_pipeline,
    create_elasticnet_pipeline,
    create_rf_pipeline,
    make_pearsonr_scorer,
    fit_with_warning_logging,
    grid_search_with_warning_logging,
    create_stratified_cv_by_binned_targets,
    create_block_aware_cv_for_within_subject,
    compute_within_subject_trial_index,
    log_cv_adjacency_info,
    blocks_constant_per_subject,
    partial_corr_xy_given_z,
    cluster_bootstrap_subjects,
    compute_calibration_metrics,
    safe_pearsonr,
    compute_metrics,
    bootstrap_pooled_metrics_by_subject,
    per_subject_pearson_and_spearman,
)
from eeg_pipeline.utils.decoding_io import (
    create_run_manifest,
    setup_file_logging,
    prepare_best_params_path,
    read_best_params_jsonl,
    read_best_params_jsonl_combined,
    export_best_params_long_table,
    export_predictions,
    export_indices,
    parse_pow_feature,
    write_feature_importance_tsv,
    aggregate_group_feature_topomaps,
    build_all_metrics_wide,
    find_bids_electrodes_tsv,
    make_montage_from_bids_electrodes,
    resolve_montage,
    prepare_config_dict,
)
from eeg_pipeline.utils.stats_utils import (
    check_pyriemann,
    align_epochs_to_pivot_chs,
)

# Local - analysis
from eeg_pipeline.analysis.decoding import (
    nested_loso_predictions,
    loso_predictions_with_fixed_params,
    within_subject_kfold_predictions,
    loso_baseline_predictions,
    compute_enet_coefs_per_fold,
    compute_rf_block_permutation_importance_per_fold,
    run_shap_rf_loso,
    subject_id_decodability_auc,
    aggregate_temperature_trial_and_block,
    loso_riemann_regression,
    riemann_export_cov_bins,
    riemann_export_cov_bins_per_fold,
    run_riemann_band_limited_decoding,
    run_riemann_sliding_window,
)

# Local - plotting
from eeg_pipeline.plotting.plot_decoding import (
    configure_plotting,
    despine,
    plot_prediction_scatter,
    plot_per_subject_performance,
    plot_residual_diagnostics,
    plot_model_comparison,
    plot_calibration_curve,
    plot_bootstrap_distributions,
    plot_permutation_null,
    plot_feature_importance_top_n,
    plot_feature_importance_stability,
    plot_riemann_band_comparison,
    plot_riemann_sliding_window,
    plot_incremental_validity,
)


###################################################################
# Logging Setup
###################################################################

logger = logging.getLogger("decode_pain")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(stream=sys.stdout)
_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
if not logger.handlers:
    logger.addHandler(_handler)

# Optional file handler will be attached per run
_FILE_LOG_HANDLER: Optional[logging.Handler] = None


configure_plotting()



BEST_PARAMS_MODE: str = os.environ.get("BEST_PARAMS_MODE", "truncate")
RUN_ID: Optional[str] = None




###################################################################
# Analysis Functions
###################################################################

def _setup_models_and_config(
    config_dict: Dict[str, Any],
    seed: int,
    config: Optional[Any] = None,
) -> Tuple[Any, Dict[str, Any], Any, Dict[str, Any], int]:
    en_cfg = config_dict["models"]["elasticnet"]
    enet_pipe = create_elasticnet_pipeline(seed=seed, config=config)
    enet_pipe.named_steps["regressor"].regressor.max_iter = en_cfg["max_iter"]
    enet_pipe.named_steps["regressor"].regressor.tol = en_cfg["tol"]
    enet_pipe.named_steps["regressor"].regressor.selection = en_cfg["selection"]
    enet_grid = {
        "regressor__regressor__alpha": en_cfg["grid"]["alpha"],
        "regressor__regressor__l1_ratio": en_cfg["grid"]["l1_ratio"],
    }

    rf_cfg = config_dict["models"]["random_forest"]
    rf_pipe = create_rf_pipeline(n_estimators=rf_cfg["n_estimators"], n_jobs=1, seed=seed, config=config)
    rf_pipe.named_steps["rf"].bootstrap = rf_cfg["bootstrap"]
    
    rf_grid = {
        "rf__max_depth": rf_cfg["grid"]["max_depth"],
        "rf__max_features": rf_cfg["grid"]["max_features"],
        "rf__min_samples_leaf": rf_cfg["grid"]["min_samples_leaf"],
    }

    inner_splits = int(config_dict["cv"]["inner_splits"])
    
    return enet_pipe, enet_grid, rf_pipe, rf_grid, inner_splits


def _run_loso_analysis(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    groups: np.ndarray,
    enet_pipe: Any,
    enet_grid: Dict[str, Any],
    rf_pipe: Any,
    rf_grid: Dict[str, Any],
    inner_splits: int,
    n_jobs: int,
    seed: int,
    outer_n_jobs: int,
    results_dir: Path,
    config_dict: Dict[str, Any],
    meta: pd.DataFrame,
    blocks_source: str,
) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    if X_all is None or X_all.empty:
        raise ValueError("X_all must be a non-empty DataFrame")
    if y_all is None or len(y_all) == 0:
        raise ValueError("y_all must be a non-empty Series")
    if groups is None or len(groups) == 0:
        raise ValueError("groups must be a non-empty array")
    if len(X_all) != len(y_all) or len(X_all) != len(groups) or len(X_all) != len(meta):
        raise ValueError(
            f"Length mismatch: X_all={len(X_all)}, y_all={len(y_all)}, "
            f"groups={len(groups)}, meta={len(meta)}"
        )
    if "trial_id" not in meta.columns:
        raise ValueError(f"meta missing required column 'trial_id'. Available columns: {list(meta.columns)}")
    if len(meta["trial_id"]) != len(meta):
        raise ValueError(f"meta.trial_id has length {len(meta['trial_id'])}, expected {len(meta)}")
    
    best_params_en_path = prepare_best_params_path(results_dir / config_dict["paths"]["best_params"]["elasticnet_loso"], BEST_PARAMS_MODE, RUN_ID)
    best_params_rf_path = prepare_best_params_path(results_dir / config_dict["paths"]["best_params"]["rf_loso"], BEST_PARAMS_MODE, RUN_ID)

    y_true_en, y_pred_en, groups_ordered_en, test_indices_en, fold_ids_en = nested_loso_predictions(
        X=X_all, y=y_all, groups=groups,
        pipe=enet_pipe, param_grid=enet_grid,
        inner_cv_splits=inner_splits, n_jobs=n_jobs, seed=seed,
        best_params_log_path=best_params_en_path, model_name="ElasticNet",
        outer_n_jobs=outer_n_jobs,
    )
    pooled_en, per_subj_en = compute_metrics(y_true_en, y_pred_en, np.asarray(groups_ordered_en))
    logger.info(f"ElasticNet pooled: r={pooled_en['pearson_r']:.3f}, R2={pooled_en['r2']:.3f}, EVS={pooled_en['explained_variance']:.3f}, avg_r_Fz={pooled_en['avg_subject_r_fisher_z']:.3f}")

    results_dir.mkdir(parents=True, exist_ok=True)
    pred_en = export_predictions(
        y_true=y_true_en,
        y_pred=y_pred_en,
        groups_ordered=groups_ordered_en,
        test_indices=test_indices_en,
        fold_ids=fold_ids_en,
        model_name="ElasticNet",
        meta=meta,
        save_path=results_dir / config_dict["paths"]["predictions"]["elasticnet_loso"],
    )
    ensure_dir((results_dir / config_dict["paths"]["per_subject_metrics"]["elasticnet_loso"]).parent)
    per_subj_en.to_csv(results_dir / config_dict["paths"]["per_subject_metrics"]["elasticnet_loso"], sep="\t", index=False)
    
    export_indices(
        groups_ordered=groups_ordered_en,
        test_indices=test_indices_en,
        fold_ids=fold_ids_en,
        meta=meta,
        save_path=results_dir / config_dict["paths"]["indices"]["elasticnet_loso"],
        blocks_source=blocks_source,
        add_heldout_subject_id=True,
    )

    y_true_rf, y_pred_rf, groups_ordered_rf, test_indices_rf, fold_ids_rf = nested_loso_predictions(
        X=X_all, y=y_all, groups=groups,
        pipe=rf_pipe, param_grid=rf_grid,
        inner_cv_splits=inner_splits, n_jobs=n_jobs, seed=seed,
        best_params_log_path=best_params_rf_path, model_name="RandomForest",
        outer_n_jobs=outer_n_jobs,
    )
    pooled_rf, per_subj_rf = compute_metrics(y_true_rf, y_pred_rf, np.asarray(groups_ordered_rf))
    logger.info(f"RandomForest pooled: r={pooled_rf['pearson_r']:.3f}, R2={pooled_rf['r2']:.3f}, EVS={pooled_rf['explained_variance']:.3f}, avg_r_Fz={pooled_rf['avg_subject_r_fisher_z']:.3f}")

    pred_rf = export_predictions(
        y_true=y_true_rf,
        y_pred=y_pred_rf,
        groups_ordered=groups_ordered_rf,
        test_indices=test_indices_rf,
        fold_ids=fold_ids_rf,
        model_name="RandomForest",
        meta=meta,
        save_path=results_dir / config_dict["paths"]["predictions"]["rf_loso"],
    )
    ensure_dir((results_dir / config_dict["paths"]["per_subject_metrics"]["rf_loso"]).parent)
    per_subj_rf.to_csv(results_dir / config_dict["paths"]["per_subject_metrics"]["rf_loso"], sep="\t", index=False)

    export_indices(
        groups_ordered=groups_ordered_rf,
        test_indices=test_indices_rf,
        fold_ids=fold_ids_rf,
        meta=meta,
        save_path=results_dir / config_dict["paths"]["indices"]["rf_loso"],
        blocks_source=blocks_source,
        add_heldout_subject_id=True,
    )

    return (
        y_true_en, y_pred_en, groups_ordered_en, test_indices_en, fold_ids_en,
        pooled_en, per_subj_en, pred_en,
        y_true_rf, y_pred_rf, groups_ordered_rf, test_indices_rf, fold_ids_rf,
        pooled_rf, per_subj_rf, pred_rf,
        best_params_en_path, best_params_rf_path
    )


def _run_within_subject_analysis(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    groups: np.ndarray,
    enet_pipe: Any,
    enet_grid: Dict[str, Any],
    rf_pipe: Any,
    rf_grid: Dict[str, Any],
    inner_splits: int,
    n_jobs: int,
    seed: int,
    outer_n_jobs: int,
    results_dir: Path,
    config_dict: Dict[str, Any],
    meta: pd.DataFrame,
    deriv_root: Path,
    task: str,
    blocks_all: Optional[np.ndarray],
) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
    if not config_dict["flags"]["run_within_subject_kfold"]:
        return None, None, None, None, None, None, None, None, None, None, None, None

    best_params_en_within_path = prepare_best_params_path(results_dir / config_dict["paths"]["best_params"]["elasticnet_within"], BEST_PARAMS_MODE, RUN_ID)
    y_true_wen, y_pred_wen, groups_ordered_wen, test_indices_wen, fold_ids_wen = within_subject_kfold_predictions(
        X=X_all, y=y_all, groups=groups,
        pipe=enet_pipe, param_grid=enet_grid,
        inner_cv_splits=inner_splits, n_jobs=n_jobs, seed=seed,
        best_params_log_path=best_params_en_within_path, model_name="ElasticNet",
        outer_n_jobs=outer_n_jobs, deriv_root=deriv_root, task=task,
        blocks_all=blocks_all,
    )
    
    pred_wen = None
    pooled_wen = None
    per_subj_wen = None
    
    if len(y_true_wen) > 0:
        pooled_wen, per_subj_wen = compute_metrics(y_true_wen, y_pred_wen, np.asarray(groups_ordered_wen))
        logger.info(f"ElasticNet Within-Subject KFold pooled: r={pooled_wen['pearson_r']:.3f}, R2={pooled_wen['r2']:.3f}, EVS={pooled_wen['explained_variance']:.3f}, avg_r_Fz={pooled_wen['avg_subject_r_fisher_z']:.3f}")

        pred_wen = export_predictions(
            y_true=y_true_wen,
            y_pred=y_pred_wen,
            groups_ordered=groups_ordered_wen,
            test_indices=test_indices_wen,
            fold_ids=fold_ids_wen,
            model_name="ElasticNet_WithinKFold",
            meta=meta,
            save_path=results_dir / config_dict["paths"]["predictions"]["elasticnet_within"],
        )
        ensure_dir((results_dir / config_dict["paths"]["per_subject_metrics"]["elasticnet_within"]).parent)
        per_subj_wen.to_csv(results_dir / config_dict["paths"]["per_subject_metrics"]["elasticnet_within"], sep="\t", index=False)
    else:
        logger.warning("ElasticNet Within-Subject KFold: No predictions generated (all subjects skipped due to missing run_id).")

    export_indices(
        groups_ordered=groups_ordered_wen,
        test_indices=test_indices_wen,
        fold_ids=fold_ids_wen,
        meta=meta,
        save_path=results_dir / config_dict["paths"]["indices"]["elasticnet_within"],
    )

    best_params_rf_within_path = prepare_best_params_path(results_dir / config_dict["paths"]["best_params"]["rf_within"], BEST_PARAMS_MODE, RUN_ID)
    y_true_wrf, y_pred_wrf, groups_ordered_wrf, test_indices_wrf, fold_ids_wrf = within_subject_kfold_predictions(
        X=X_all, y=y_all, groups=groups,
        pipe=rf_pipe, param_grid=rf_grid,
        inner_cv_splits=inner_splits, n_jobs=n_jobs, seed=seed,
        best_params_log_path=best_params_rf_within_path, model_name="RandomForest",
        outer_n_jobs=outer_n_jobs, deriv_root=deriv_root, task=task,
        blocks_all=blocks_all,
    )
    
    pred_wrf = None
    pooled_wrf = None
    per_subj_wrf = None
    
    if len(y_true_wrf) > 0:
        pooled_wrf, per_subj_wrf = compute_metrics(y_true_wrf, y_pred_wrf, np.asarray(groups_ordered_wrf))
        logger.info(f"RF Within-Subject KFold pooled: r={pooled_wrf['pearson_r']:.3f}, R2={pooled_wrf['r2']:.3f}, EVS={pooled_wrf['explained_variance']:.3f}, avg_r_Fz={pooled_wrf['avg_subject_r_fisher_z']:.3f}")

        pred_wrf = export_predictions(
            y_true=y_true_wrf,
            y_pred=y_pred_wrf,
            groups_ordered=groups_ordered_wrf,
            test_indices=test_indices_wrf,
            fold_ids=fold_ids_wrf,
            model_name="RandomForest_WithinKFold",
            meta=meta,
            save_path=results_dir / config_dict["paths"]["predictions"]["rf_within"],
        )
        ensure_dir((results_dir / config_dict["paths"]["per_subject_metrics"]["rf_within"]).parent)
        per_subj_wrf.to_csv(results_dir / config_dict["paths"]["per_subject_metrics"]["rf_within"], sep="\t", index=False)
    else:
        logger.warning("RandomForest Within-Subject KFold: No predictions generated (all subjects skipped due to missing run_id).")

    export_indices(
        groups_ordered=groups_ordered_wrf,
        test_indices=test_indices_wrf,
        fold_ids=fold_ids_wrf,
        meta=meta,
        save_path=results_dir / config_dict["paths"]["indices"]["rf_within"],
    )

    return pred_wen, pooled_wen, per_subj_wen, pred_wrf, pooled_wrf, per_subj_wrf, y_true_wen, y_true_wrf, groups_ordered_wen, groups_ordered_wrf, test_indices_wen, test_indices_wrf


def _run_permutation_testing(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    groups: np.ndarray,
    enet_pipe: Any,
    enet_grid: Dict[str, Any],
    rf_pipe: Any,
    rf_grid: Dict[str, Any],
    inner_splits: int,
    n_jobs: int,
    seed: int,
    outer_n_jobs: int,
    config_dict: Dict[str, Any],
    results_dir: Path,
    pooled_en: Optional[Dict[str, float]],
    pooled_rf: Optional[Dict[str, float]],
    best_params_en_path: Path,
    best_params_rf_path: Path,
) -> None:
    have_en = pooled_en is not None
    have_rf = pooled_rf is not None
    
    if not (have_en or have_rf):
        return

    n_perm_refit = int(config_dict["analysis"]["n_perm_refit"])
    perm_jobs = int(config_dict["analysis"].get("perm_refit_n_jobs", 1))

    obs_candidates = []
    if have_en:
        obs_candidates.append(float(pooled_en.get("pearson_r", np.nan)))
    if have_rf:
        obs_candidates.append(float(pooled_rf.get("pearson_r", np.nan)))
    obs_r_max = float(np.nanmax(np.abs(np.asarray(obs_candidates, float)))) if len(obs_candidates) > 0 else float('nan')
    models_used = [m for m in ["ElasticNet" if have_en else None, "RandomForest" if have_rf else None] if m is not None]
    logger.info(f"Permutation null will evaluate models: {', '.join(models_used) if models_used else 'none'}")

    best_params_en_map = read_best_params_jsonl_combined(best_params_en_path) if have_en else {}
    best_params_rf_map = read_best_params_jsonl_combined(best_params_rf_path) if have_rf else {}

    use_nested = bool(config_dict["analysis"].get("perm_refit_use_nested_cv", True))
    
    # CRITICAL FIX: Validate sufficient trials per subject for valid permutation testing
    # Permutation testing requires sufficient trials per subject to create valid null distribution
    min_trials_per_subject = int(config_dict["analysis"].get("min_trials_per_subject_for_permutation", 10))
    unique_groups = np.unique(groups)
    group_counts = {g: int(np.sum(groups == g)) for g in unique_groups}
    insufficient_subjects = [g for g, count in group_counts.items() if count < min_trials_per_subject]
    
    if insufficient_subjects:
        n_insufficient = len(insufficient_subjects)
        min_count = min(group_counts.values())
        error_msg = (
            f"Permutation testing validation failed: {n_insufficient} subject(s) have insufficient trials "
            f"for valid permutation testing. Minimum required: {min_trials_per_subject} trials per subject. "
            f"Found minimum: {min_count} trials. Subjects with insufficient trials: {insufficient_subjects[:5]}{'...' if n_insufficient > 5 else ''}. "
            f"Permutation testing with too few trials per subject produces unreliable p-values. "
            f"Increase minimum trials per subject or exclude subjects with insufficient data."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Validate that permutation will actually break the signal
    # Check that subjects have sufficient variability for meaningful permutation
    for g in unique_groups:
        idx = np.where(groups == g)[0]
        y_subj = y_all.iloc[idx].values
        if len(np.unique(y_subj)) < 3:
            logger.warning(
                f"Subject {g} has only {len(np.unique(y_subj))} unique target values. "
                f"Permutation may have limited effect on null distribution."
            )
    
    logger.info(
        f"Permutation testing validation passed: all {len(unique_groups)} subjects have >= {min_trials_per_subject} trials "
        f"(range: {min(group_counts.values())}-{max(group_counts.values())} trials per subject)"
    )

    def _one_perm_max(i: int) -> float:
        rng_i = np.random.default_rng(seed + 12345 + i)
        y_perm = y_all.to_numpy().copy()
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            if len(idx) > 1:
                y_perm[idx] = rng_i.permutation(y_perm[idx])
        y_perm_series = pd.Series(y_perm)

        vals = []
        if have_en:
            if use_nested:
                y_true_e, y_pred_e, _, _, _ = nested_loso_predictions(
                    X=X_all, y=y_perm_series, groups=groups,
                    pipe=enet_pipe, param_grid=enet_grid,
                    inner_cv_splits=inner_splits, n_jobs=n_jobs, seed=seed + 2000 + i,
                    best_params_log_path=None, model_name="ElasticNet", outer_n_jobs=outer_n_jobs,
                )
            else:
                y_true_e, y_pred_e, _, _, _ = loso_predictions_with_fixed_params(
                    X_all, y_perm_series, groups, enet_pipe, best_params_en_map, seed=seed + 2000 + i, outer_n_jobs=outer_n_jobs,
                )
            r_e, _ = safe_pearsonr(y_true_e, y_pred_e)
            vals.append(float(r_e) if np.isfinite(r_e) else 0.0)
        if have_rf:
            if use_nested:
                y_true_r, y_pred_r, _, _, _ = nested_loso_predictions(
                    X=X_all, y=y_perm_series, groups=groups,
                    pipe=rf_pipe, param_grid=rf_grid,
                    inner_cv_splits=inner_splits, n_jobs=n_jobs, seed=seed + 3000 + i,
                    best_params_log_path=None, model_name="RandomForest", outer_n_jobs=outer_n_jobs,
                )
            else:
                y_true_r, y_pred_r, _, _, _ = loso_predictions_with_fixed_params(
                    X_all, y_perm_series, groups, rf_pipe, best_params_rf_map, seed=seed + 3000 + i, outer_n_jobs=outer_n_jobs,
                )
            r_r, _ = safe_pearsonr(y_true_r, y_pred_r)
            vals.append(float(r_r) if np.isfinite(r_r) else 0.0)

        return float(np.max(np.abs(vals))) if len(vals) > 0 else 0.0

    if perm_jobs > 1:
        null_list = Parallel(n_jobs=perm_jobs, prefer="processes")(delayed(_one_perm_max)(i) for i in range(n_perm_refit))
        null_rs = np.asarray(null_list, dtype=float)
    else:
        null_rs = np.zeros(n_perm_refit, dtype=float)
        rng = np.random.default_rng(seed + 12345)
        for i in range(n_perm_refit):
            y_perm = y_all.to_numpy().copy()
            for g in np.unique(groups):
                idx = np.where(groups == g)[0]
                if len(idx) > 1:
                    y_perm[idx] = rng.permutation(y_perm[idx])
            y_perm_series = pd.Series(y_perm)
            vals = []
            if have_en:
                if use_nested:
                    y_true_e, y_pred_e, _, _, _ = nested_loso_predictions(
                        X=X_all,
                        y=y_perm_series,
                        groups=groups,
                        pipe=enet_pipe,
                        param_grid=enet_grid,
                        inner_cv_splits=inner_splits,
                        n_jobs=n_jobs,
                        seed=seed + 2000 + i,
                        best_params_log_path=None,
                        model_name="ElasticNet",
                        outer_n_jobs=outer_n_jobs,
                    )
                else:
                    y_true_e, y_pred_e, _, _, _ = loso_predictions_with_fixed_params(
                        X_all, y_perm_series, groups, enet_pipe, best_params_en_map, seed=seed + 2000 + i, outer_n_jobs=outer_n_jobs,
                    )
                r_e, _ = safe_pearsonr(y_true_e, y_pred_e)
                vals.append(float(r_e) if np.isfinite(r_e) else 0.0)
            if have_rf:
                if use_nested:
                    y_true_r, y_pred_r, _, _, _ = nested_loso_predictions(
                        X=X_all,
                        y=y_perm_series,
                        groups=groups,
                        pipe=rf_pipe,
                        param_grid=rf_grid,
                        inner_cv_splits=inner_splits,
                        n_jobs=n_jobs,
                        seed=seed + 3000 + i,
                        best_params_log_path=None,
                        model_name="RandomForest",
                        outer_n_jobs=outer_n_jobs,
                    )
                else:
                    y_true_r, y_pred_r, _, _, _ = loso_predictions_with_fixed_params(
                        X_all, y_perm_series, groups, rf_pipe, best_params_rf_map, seed=seed + 3000 + i, outer_n_jobs=outer_n_jobs,
                    )
                r_r, _ = safe_pearsonr(y_true_r, y_pred_r)
                vals.append(float(r_r) if np.isfinite(r_r) else 0.0)
            null_rs[i] = float(np.max(np.abs(vals))) if len(vals) > 0 else 0.0

    if len(null_rs) != n_perm_refit:
        error_msg = (
            f"Permutation testing failed: expected {n_perm_refit} permutations "
            f"but got {len(null_rs)}. Permutation null distribution is incomplete."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    if not np.all(np.isfinite(null_rs)):
        n_nonfinite = int(np.sum(~np.isfinite(null_rs)))
        error_msg = (
            f"Permutation testing failed: {n_nonfinite} non-finite values in null distribution. "
            f"Permutation null distribution is invalid."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    obs_r = obs_r_max
    p_refit_two_sided = float((np.sum(np.abs(null_rs) >= abs(obs_r)) + 1) / (len(null_rs) + 1))
    p_refit_one_sided_pos = float((np.sum(null_rs >= obs_r) + 1) / (len(null_rs) + 1))
    p_refit_one_sided_neg = float((np.sum(null_rs <= obs_r) + 1) / (len(null_rs) + 1))
    
    ensure_dir((results_dir / config_dict["paths"]["summaries"]["permutation_refit_null_rs"]).parent)
    np.savetxt(results_dir / config_dict["paths"]["summaries"]["permutation_refit_null_rs"], null_rs, fmt="%.6f")
    
    null_stats = {
        "mean": float(np.mean(null_rs)),
        "std": float(np.std(null_rs, ddof=1)) if n_perm_refit > 1 else 0.0,
        "median": float(np.median(null_rs)),
        "q25": float(np.percentile(null_rs, 25)),
        "q75": float(np.percentile(null_rs, 75)),
        "min": float(np.min(null_rs)),
        "max": float(np.max(null_rs)),
        "skewness": float(scipy.stats.skew(null_rs)) if len(null_rs) > 2 else 0.0,
        "kurtosis": float(scipy.stats.kurtosis(null_rs)) if len(null_rs) > 2 else 0.0,
    }
    
    ensure_dir((results_dir / config_dict["paths"]["summaries"]["permutation_refit_summary"]).parent)
    with open(results_dir / config_dict["paths"]["summaries"]["permutation_refit_summary"], "w", encoding="utf-8") as f:
        json.dump({
            "statistic": "max_abs_r_across_models",
            "observed_r": obs_r,
            "n_perm": int(n_perm_refit),
            "p_two_sided_abs_r": p_refit_two_sided,
            "p_one_sided_positive": p_refit_one_sided_pos,
            "p_one_sided_negative": p_refit_one_sided_neg,
            "null_distribution_stats": null_stats,
            "models_considered": [m for m in ["ElasticNet" if have_en else None, "RandomForest" if have_rf else None] if m is not None],
            "selection_correction": "max_statistic",
            "approximation": ("nested_cv" if use_nested else "fixed_params"),
        }, f, indent=2)
    logger.info(f"Saved refit-based permutation null (max-statistic across models) p_two_sided={p_refit_two_sided:.4g}, p_one_sided_pos={p_refit_one_sided_pos:.4g}")




def _setup_analysis_data(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config_dict: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    X_all, y_all, groups, meta = load_multiple_subjects_decoding_data(
        subjects=subjects, deriv_root=deriv_root, task=task, config=config_dict, logger=logger
    )
    
    validate_data_not_empty(X_all, y_all, groups, meta)
    validate_data_lengths(X_all, y_all, groups, meta)
    validate_trial_ids(meta, logger)
    validate_sufficient_subjects(groups)
    
    return X_all, y_all, groups, meta


def _run_interpretability_analyses(X_all, y_all, groups, meta, best_params_en_path, best_params_rf_path, feat_names, deriv_root, results_dir, export_dir, config_dict, seed, blocks_all, blocks_source, blocks_reliability, pooled_en, pooled_rf, logger):
    agg = str(config_dict.get("viz", {}).get("coef_agg", "abs")).lower()
    if agg not in ("abs", "signed"):
        agg = "abs"
    
    best_params_en_map = read_best_params_jsonl_combined(best_params_en_path)
    best_params_rf_map = read_best_params_jsonl_combined(best_params_rf_path)
    
    if pooled_en is not None:
        en_coefs = compute_enet_coefs_per_fold(X=X_all, y=y_all, groups=np.asarray(groups), best_params_map=best_params_en_map, seed=seed)
        logo = LeaveOneGroupOut()
        for fold_idx, (_, test_idx) in enumerate(logo.split(X_all, y_all, groups)):
            subject = str(np.asarray(groups)[test_idx][0])
            subj_coef = en_coefs[fold_idx:fold_idx+1, :]
            subj_stats_dir = deriv_stats_path(deriv_root, subject) / "05_decode_pain_experience"
            write_feature_importance_tsv(
                subject=subject,
                coef_matrix=subj_coef,
                feature_names=feat_names,
                stats_dir=subj_stats_dir,
                method="elasticnet",
                aggregate=agg,
                mode="regression",
                target="rating",
            )
        
        plot_feature_importance_stability(
            coef_matrix=en_coefs,
            feature_names=feat_names,
            model_name="ElasticNet",
            save_path=export_dir / "en_feature_importance_stability",
            plot_type="violin",
            top_n_channels=30,
        )
    
    if pooled_rf is not None:
        rf_imps, rf_perm_meta = compute_rf_block_permutation_importance_per_fold(
            X=X_all, y=y_all, groups=np.asarray(groups), best_params_map=best_params_rf_map, seed=seed,
            n_repeats=int(config_dict["analysis"]["rf_perm_importance_repeats"]), blocks_all=blocks_all,
            blocks_origin=blocks_source, config_dict=config_dict
        )
        mean_imps = np.nanmean(rf_imps, axis=0)
        
        if len(rf_perm_meta) > 0:
            perm_meta_df = pd.DataFrame(rf_perm_meta)
            perm_meta_df["blocks_origin"] = blocks_source
            perm_meta_df["blocks_reliability"] = blocks_reliability
            perm_meta_path = results_dir / "permutation" / "rf_block_strategy.tsv"
            ensure_dir(perm_meta_path.parent)
            perm_meta_df.to_csv(perm_meta_path, sep="\t", index=False)
        
        logo = LeaveOneGroupOut()
        for fold_idx, (_, test_idx) in enumerate(logo.split(X_all, y_all, groups)):
            subject = str(np.asarray(groups)[test_idx][0])
            subj_imp = rf_imps[fold_idx:fold_idx+1, :]
            subj_stats_dir = deriv_stats_path(deriv_root, subject) / "05_decode_pain_experience"
            extra_cols = None
            if fold_idx < len(rf_perm_meta):
                entry = rf_perm_meta[fold_idx]
                extra_cols = {
                    "block_strategy": entry.get("block_strategy"),
                    "block_notes": entry.get("notes"),
                    "n_unique_blocks": entry.get("n_unique_blocks"),
                }
            write_feature_importance_tsv(
                subject=subject,
                coef_matrix=subj_imp,
                feature_names=feat_names,
                stats_dir=subj_stats_dir,
                method="rfperm",
                aggregate="mean",
                mode="regression",
                target="rating",
                extra_columns=extra_cols,
            )
        
        if config_dict["flags"]["run_shap"]:
            run_shap_rf_loso(X_all, y_all, np.asarray(groups), feat_names, best_params_rf_map, export_dir, seed)
        
        plot_feature_importance_stability(
            coef_matrix=rf_imps,
            feature_names=feat_names,
            model_name="RandomForest",
            save_path=export_dir / "rf_feature_importance_stability",
            plot_type="violin",
            top_n_channels=30,
        )


def _generate_all_plots(
    export_dir: Path,
    results_dir: Path,
    config_dict: Dict[str, Any],
    summary_path: Path,
    pred_en: Optional[pd.DataFrame] = None,
    pooled_en: Optional[Dict[str, float]] = None,
    per_subj_en: Optional[pd.DataFrame] = None,
    pred_rf: Optional[pd.DataFrame] = None,
    pooled_rf: Optional[Dict[str, float]] = None,
    per_subj_rf: Optional[pd.DataFrame] = None,
    cal_metrics: Optional[Dict[str, Any]] = None,
    pred_wen: Optional[pd.DataFrame] = None,
    pooled_wen: Optional[Dict[str, float]] = None,
    per_subj_wen: Optional[pd.DataFrame] = None,
    pred_wrf: Optional[pd.DataFrame] = None,
    pooled_wrf: Optional[Dict[str, float]] = None,
    per_subj_wrf: Optional[pd.DataFrame] = None,
    pooled_bg: Optional[Dict[str, float]] = None,
    pooled_riem: Optional[Dict[str, float]] = None,
    pooled_t: Optional[Dict[str, float]] = None,
    bootstrap_results: Optional[Any] = None,
    inc_summary: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> None:
    logger.info("Generating publication-quality figures...")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    if pred_en is not None:
        plot_prediction_scatter(pred_en, 'ElasticNet', pooled_en, export_dir / 'en_prediction_scatter')
        plot_per_subject_performance(per_subj_en, 'ElasticNet', export_dir / 'en_per_subject')
        plot_residual_diagnostics(pred_en, 'ElasticNet', export_dir / 'en_residuals')
    
    if pred_rf is not None:
        plot_prediction_scatter(pred_rf, 'RandomForest', pooled_rf, export_dir / 'rf_prediction_scatter')
        plot_per_subject_performance(per_subj_rf, 'RandomForest', export_dir / 'rf_per_subject')
        plot_residual_diagnostics(pred_rf, 'RandomForest', export_dir / 'rf_residuals')
        if cal_metrics is not None:
            plot_calibration_curve(pred_rf, 'RandomForest', cal_metrics, export_dir / 'rf_calibration')
    
    if pred_wen is not None:
        plot_prediction_scatter(pred_wen, 'ElasticNet_Within', pooled_wen, export_dir / 'en_within_prediction_scatter')
        plot_per_subject_performance(per_subj_wen, 'ElasticNet_Within', export_dir / 'en_within_per_subject')
    
    if pred_wrf is not None:
        plot_prediction_scatter(pred_wrf, 'RandomForest_Within', pooled_wrf, export_dir / 'rf_within_prediction_scatter')
        plot_per_subject_performance(per_subj_wrf, 'RandomForest_Within', export_dir / 'rf_within_per_subject')
    
    models_comparison = {
        'Baseline': pooled_bg,
        'ElasticNet': pooled_en,
        'RandomForest': pooled_rf,
        'EN_Within': pooled_wen,
        'RF_Within': pooled_wrf,
        'Riemann': pooled_riem,
        'Temperature': pooled_t if isinstance(pooled_t, dict) else None,
    }
    models_comparison = {k: v for k, v in models_comparison.items() if v is not None}
    if models_comparison:
        plot_model_comparison(models_comparison, export_dir / 'model_comparison')
    
    if bootstrap_results:
        plot_bootstrap_distributions(bootstrap_results, export_dir / 'bootstrap_distributions')
    
    perm_null_path = results_dir / config_dict["paths"]["summaries"]["permutation_refit_null_rs"]
    perm_summary_path = results_dir / config_dict["paths"]["summaries"]["permutation_refit_summary"]
    if perm_null_path.exists() and perm_summary_path.exists():
        null_rs = np.loadtxt(perm_null_path)
        with open(perm_summary_path, 'r') as f:
            perm_summary = json.load(f)
        plot_permutation_null(null_rs, perm_summary['observed_r'], perm_summary['p_two_sided_abs_r'], 
                             export_dir / 'permutation_null')
    
    shap_path = export_dir / 'rf_shap_loso_feature_importance.tsv'
    if shap_path.exists():
        shap_df = pd.read_csv(shap_path, sep='\t')
        top_n = int(config_dict.get("decoding.visualization.top_n_features", 20))
        plot_feature_importance_top_n(shap_df, 'RF_SHAP', export_dir / 'rf_shap_top20', top_n=top_n)
    
    if inc_summary is not None:
        plot_incremental_validity(inc_summary, export_dir / 'incremental_validity')
    
    logger.info(f"Figures saved to: {export_dir}")
    logger.info(f"Artifacts: pred_en={results_dir / config_dict['paths']['predictions']['elasticnet_loso']} | pred_rf={results_dir / config_dict['paths']['predictions']['rf_loso']} | summary={summary_path}")


###################################################################
# Main Entry Point
###################################################################

def main(subjects: Optional[List[str]] = None, task: Optional[str] = None, n_jobs: int = 1, seed: Optional[int] = None, outer_n_jobs: int = 1, all_subjects: bool = False, no_plots: bool = False) -> None:
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be >= 1, got: {n_jobs}")
    if outer_n_jobs < 1:
        raise ValueError(f"outer_n_jobs must be >= 1, got: {outer_n_jobs}")
    if subjects is not None and not isinstance(subjects, list):
        raise ValueError(f"subjects must be None or a list, got: {type(subjects)}")
    if subjects is not None and len(subjects) == 0:
        raise ValueError("subjects list cannot be empty. Use None or --all-subjects instead.")
    
    cfg = load_settings()
    cfg.apply_thread_limits()
    

    config_dict, legacy = prepare_config_dict(cfg)
    
    if task is None:
        task = legacy["TASK"]
    if not task or not isinstance(task, str):
        raise ValueError(f"task must be non-empty string, got: {task}")
    if seed is None:
        seed = legacy["RANDOM_STATE"]
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be non-negative integer, got: {seed}")
    deriv_root = Path(legacy["DERIV_ROOT"])
    if not deriv_root.exists():
        raise ValueError(f"deriv_root does not exist: {deriv_root}")
    
    results_dir = deriv_root / config_dict["paths"]["results_subdir"]
    export_dir = results_dir / config_dict["paths"]["plots_subdir"] / "05_decode_pain_experience"
    export_dir.mkdir(parents=True, exist_ok=True)

    log_path = setup_file_logging(results_dir, RUN_ID)
    logger.info(f"File logging initialized: {log_path}")

    cli_args = {
        "subjects": subjects,
        "task": task,
        "n_jobs": n_jobs,
        "seed": seed,
        "outer_n_jobs": outer_n_jobs,
        "best_params_mode": BEST_PARAMS_MODE,
        "run_id": RUN_ID
    }
    create_run_manifest(results_dir, cli_args, config_dict, RUN_ID)
    logger.info(f"Created run manifest at {results_dir / 'run_manifest.json'}")

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    subs = "all" if subjects is None else ",".join([str(s) for s in subjects])
    logger.info(f"Run header | ts={ts} task={task} seed={seed} subjects={subs} best_params_mode={BEST_PARAMS_MODE} run_id={RUN_ID or ''} results_dir={results_dir} export_dir={export_dir}")

    if all_subjects:
        subjects = None
    elif not subjects:
        raise ValueError("No subjects specified. Use --subjects or --all-subjects.")
    
    X_all, y_all, groups, meta = _setup_analysis_data(subjects, task, deriv_root, config_dict, logger)
    feat_names = list(X_all.columns)

    enet_pipe, enet_grid, rf_pipe, rf_grid, inner_splits = _setup_models_and_config(config_dict, seed, config=cfg)

    temps_all, trials_all, blocks_events, blocks_provenance = aggregate_temperature_trial_and_block(meta, deriv_root, task, config=config_dict)
    blocks_all: Optional[np.ndarray] = blocks_events if blocks_provenance == "events" else None
    blocks_source = "events" if blocks_all is not None else "none"
    blocks_reliability = "none"

    if blocks_all is not None:
        n_valid = int(np.sum(np.isfinite(blocks_all)))
        if n_valid == 0:
            logger.warning("Run metadata present in events but all values are NaN; disabling block-aware operations.")
            blocks_all = None
            blocks_source = "none"
            blocks_reliability = "events_empty"
        elif n_valid == len(blocks_all):
            blocks_reliability = "events_complete"
            logger.info(f"Using run metadata from events ({n_valid}/{len(blocks_all)} trials).")
        else:
            blocks_reliability = "events_partial"
            logger.warning(f"Partial run metadata from events ({n_valid}/{len(blocks_all)} trials).")
    else:
        logger.info("Run metadata unavailable; block-aware operations will fall back to contiguous splits.")

    meta["run_id"] = (
        pd.Series(blocks_all, index=meta.index)
        if blocks_all is not None
        else pd.Series(np.full(len(meta), np.nan), index=meta.index)
    )

    (
        y_true_en, y_pred_en, groups_ordered_en, test_indices_en, fold_ids_en,
        pooled_en, per_subj_en, pred_en,
        y_true_rf, y_pred_rf, groups_ordered_rf, test_indices_rf, fold_ids_rf,
        pooled_rf, per_subj_rf, pred_rf,
        best_params_en_path, best_params_rf_path
    ) = _run_loso_analysis(
        X_all, y_all, groups,
        enet_pipe, enet_grid, rf_pipe, rf_grid,
        inner_splits, n_jobs, seed, outer_n_jobs,
        results_dir, config_dict, meta, blocks_source
    )

    (
        pred_wen, pooled_wen, per_subj_wen,
        pred_wrf, pooled_wrf, per_subj_wrf,
        y_true_wen, y_true_wrf, groups_ordered_wen, groups_ordered_wrf,
        test_indices_wen, test_indices_wrf
    ) = _run_within_subject_analysis(
        X_all, y_all, groups,
        enet_pipe, enet_grid, rf_pipe, rf_grid,
        inner_splits, n_jobs, seed, outer_n_jobs,
        results_dir, config_dict, meta, deriv_root, task, blocks_all
    )

    _run_permutation_testing(
        X_all, y_all, groups,
        enet_pipe, enet_grid, rf_pipe, rf_grid,
        inner_splits, n_jobs, seed, outer_n_jobs,
        config_dict, results_dir,
        pooled_en, pooled_rf,
        best_params_en_path, best_params_rf_path
    )
    
    # Calibration metrics
    cal_metrics = compute_calibration_metrics(y_true_rf, y_pred_rf)
    ensure_dir((results_dir / "calibration_metrics.json").parent)
    with open(results_dir / "calibration_metrics.json", "w", encoding="utf-8") as f:
        json.dump(cal_metrics, f, indent=2)
    logger.info(f"Calibration metrics: slope={cal_metrics['slope']:.3f}, intercept={cal_metrics['intercept']:.3f}")

    _run_interpretability_analyses(
        X_all, y_all, groups, meta, best_params_en_path, best_params_rf_path, feat_names,
        deriv_root, results_dir, export_dir, config_dict, seed, blocks_all, blocks_source,
        blocks_reliability, pooled_en, pooled_rf, logger
    )

    # Incremental validity analysis
    pred_rf["temperature"] = temps_all[np.asarray(test_indices_rf)]
    pred_rf["trial_number"] = trials_all[np.asarray(test_indices_rf)]
    if blocks_all is not None and blocks_source != "none":
        pred_rf["block_id"] = blocks_all[np.asarray(test_indices_rf)]
    else:
        pred_rf["block_id"] = np.nan

    # Temperature-only LOSO regression
    meta_subjects = meta["subject_id"].astype(str)
    temp_valid_mask = np.isfinite(temps_all)
    temp_counts = (
        pd.DataFrame({"subject_id": meta_subjects, "temp_valid": temp_valid_mask})
        .groupby("subject_id")["temp_valid"]
        .sum()
    )
    subjects_without_temp = temp_counts[temp_counts <= 0].index.tolist()
    if len(subjects_without_temp) > 0:
        logger.warning(
            "Temperature-only baseline: dropping %d subjects without temperature measurements: %s",
            len(subjects_without_temp),
            ",".join(subjects_without_temp),
        )
    baseline_subject_mask = ~meta_subjects.isin(subjects_without_temp)
    baseline_trial_mask = baseline_subject_mask & temp_valid_mask
    baseline_indices = np.where(baseline_trial_mask)[0]

    pred_t = pd.DataFrame()
    per_subj_t = pd.DataFrame()
    pooled_t: dict = {"pearson_r": np.nan, "r2": np.nan}

    if baseline_indices.size > 0:
        X_temp = pd.DataFrame({"temperature": temps_all[baseline_trial_mask]})
        
        if X_temp["temperature"].isna().any():
            n_nan_temp = int(X_temp["temperature"].isna().sum())
            error_msg = (
                f"Temperature baseline: {n_nan_temp} NaN temperature values found in baseline trials. "
                f"Temperature-only baseline requires valid temperature measurements for all trials."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        temp_pipe = create_base_preprocessing_pipeline(include_scaling=True)
        ridge_alpha_default = float(config_dict.get("decoding.models.temperature_only.ridge_alpha_default", 1.0))
        temp_pipe.steps.append(("ridge", Ridge(alpha=ridge_alpha_default)))
        temp_grid = {"ridge__alpha": config_dict["models"]["temperature_only"]["ridge_alpha_grid"]}
        y_baseline = y_all.iloc[baseline_indices].reset_index(drop=True)
        groups_baseline = groups[baseline_trial_mask]
        meta_baseline = meta.iloc[baseline_indices].reset_index(drop=True)
        
        if y_baseline.isna().any():
            n_nan_y = int(y_baseline.isna().sum())
            error_msg = (
                f"Temperature baseline: {n_nan_y} NaN target values found in baseline trials. "
                f"Temperature-only baseline requires valid target values for all trials."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        y_true_t, y_pred_t, groups_t, test_idx_t, fold_ids_t = nested_loso_predictions(
            X=X_temp,
            y=y_baseline,
            groups=groups_baseline,
            pipe=temp_pipe,
            param_grid=temp_grid,
            inner_cv_splits=int(config_dict["cv"]["inner_splits"]),
            n_jobs=n_jobs,
            seed=seed,
            best_params_log_path=results_dir / config_dict["paths"]["best_params"]["temperature_only"],
            model_name="TempOnly",
            outer_n_jobs=outer_n_jobs,
        )
        pooled_t, per_subj_t = compute_metrics(y_true_t, y_pred_t, np.asarray(groups_t))

        orig_indices = baseline_indices
        orig_test_indices = orig_indices[np.asarray(test_idx_t)]

        extra: dict = {}
        if "run_id" in meta.columns:
            extra["run_id"] = meta.loc[orig_test_indices, "run_id"].tolist()

        pred_t = pd.DataFrame({
            "subject_id": groups_t,
            "trial_id": meta_baseline.loc[test_idx_t, "trial_id"].values,
            "y_true": y_true_t,
            "y_pred": y_pred_t,
            "fold": fold_ids_t,
            "model": "TemperatureOnly",
            "temperature": temps_all[orig_test_indices],
            **extra,
        })
        ensure_dir((results_dir / config_dict["paths"]["predictions"]["temperature_only"]).parent)
        pred_t.to_csv(results_dir / config_dict["paths"]["predictions"]["temperature_only"], sep="\t", index=False)
        ensure_dir((results_dir / config_dict["paths"]["per_subject_metrics"]["temperature_only"]).parent)
        per_subj_t.to_csv(results_dir / config_dict["paths"]["per_subject_metrics"]["temperature_only"], sep="\t", index=False)

        idx_t = pd.DataFrame({
            "subject_id": groups_t,
            "trial_id": meta_baseline.loc[test_idx_t, "trial_id"].values,
            "fold": fold_ids_t,
        })
        ensure_dir((results_dir / config_dict["paths"]["indices"]["temperature_only"]).parent)
        idx_t.to_csv(results_dir / config_dict["paths"]["indices"]["temperature_only"], sep="\t", index=False)
    else:
        logger.warning("Temperature-only LOSO baseline skipped: no trials with valid temperature measurements found.")

    # Partial correlation analysis
    default_rf_metrics = {"pearson_r": float(pooled_rf.get("pearson_r", np.nan)), "r2": float(pooled_rf.get("r2", np.nan))}
    default_temp_metrics = {"pearson_r": float(pooled_t.get("pearson_r", np.nan)) if isinstance(pooled_t, dict) else np.nan, "r2": float(pooled_t.get("r2", np.nan)) if isinstance(pooled_t, dict) else np.nan}
    inc_summary = {
        "RandomForest": default_rf_metrics,
        "TemperatureOnly": default_temp_metrics,
        "delta_r": {"estimate": float("nan"), "ci95": [float("nan"), float("nan")]},
        "partial_r_given_temperature": {"estimate": float("nan"), "ci95": [float("nan"), float("nan")]},
        "partial_r_given_temp_trial_subjectmean": {"estimate": float("nan"), "ci95": [float("nan"), float("nan")]},
        "delta_r2_incremental": {"estimate": float("nan"), "note": "approx partial r^2 (unique variance beyond temperature)"},
        "delta_r2_incremental_multi": {"estimate": float("nan"), "note": "approx partial r^2 (unique variance beyond temp, trial, subj mean)"},
        "delta_r2_full_minus_temp": float("nan"),
    }
    if pred_t.empty:
        logger.warning("Incremental validity analysis skipped: temperature-only baseline unavailable.")
    else:
        key_cols = ["subject_id", "trial_id"]
        rf_temp = pred_rf.merge(pred_t[key_cols].drop_duplicates(), on=key_cols, how="inner")
        if rf_temp.empty:
            logger.warning("Incremental validity analysis skipped: no overlapping trials between RF predictions and temperature baseline.")
        else:
            r_partial = partial_corr_xy_given_z(
                x=rf_temp["y_pred"].to_numpy(),
                y=rf_temp["y_true"].to_numpy(),
                z=rf_temp["temperature"].to_numpy(),
            )
            r2_partial = float(r_partial ** 2) if np.isfinite(r_partial) else float("nan")

            block_vals = rf_temp["block_id"].to_numpy()
            covariates = [
                rf_temp["temperature"].to_numpy(),
                rf_temp["trial_number"].to_numpy(),
                rf_temp.groupby("subject_id")["y_true"].transform("mean").to_numpy(),
            ]
            if np.isfinite(block_vals).any():
                covariates.insert(2, block_vals)
            Z_multi = np.column_stack(covariates)
            r_partial_multi = partial_corr_xy_given_z(
                x=rf_temp["y_pred"].to_numpy(),
                y=rf_temp["y_true"].to_numpy(),
                Z=Z_multi,
            )
            r2_partial_multi = float(r_partial_multi ** 2) if np.isfinite(r_partial_multi) else float("nan")

            pooled_rf_subset, _ = compute_metrics(
                rf_temp["y_true"].to_numpy(),
                rf_temp["y_pred"].to_numpy(),
                rf_temp["subject_id"].to_numpy(),
            )
            r_rf = float(pooled_rf_subset.get("pearson_r", np.nan))
            r2_rf_subset = float(pooled_rf_subset.get("r2", np.nan))
            r_t = default_temp_metrics["pearson_r"]
            r2_t = default_temp_metrics["r2"]
            delta_r = float(r_rf - r_t) if np.isfinite(r_rf) and np.isfinite(r_t) else float("nan")
            delta_r2_full_minus_temp = float(r2_rf_subset - r2_t) if np.isfinite(r2_rf_subset) and np.isfinite(r2_t) else float("nan")
            delta_r2 = r2_partial

            def _metric_delta_r(d: pd.DataFrame) -> float:
                r_rf_i, _ = safe_pearsonr(d["y_true"].to_numpy(), d["y_pred_rf"].to_numpy())
                r_t_i, _ = safe_pearsonr(d["y_true"].to_numpy(), d["y_pred_temp"].to_numpy())
                return float(r_rf_i - r_t_i)

            def _metric_partial_r(d: pd.DataFrame) -> float:
                return float(partial_corr_xy_given_z(d["y_pred_rf"].to_numpy(), d["y_true"].to_numpy(), d["temperature"].to_numpy()))

            def _metric_partial_r_multi(d: pd.DataFrame) -> float:
                block_vals_local = d["block_id"].to_numpy()
                covs_local = [
                    d["temperature"].to_numpy(),
                    d["trial_number"].to_numpy(),
                    d.groupby("subject_id")["y_true"].transform("mean").to_numpy(),
                ]
                if np.isfinite(block_vals_local).any():
                    covs_local.insert(2, block_vals_local)
                Zm = np.column_stack(covs_local)
                return float(partial_corr_xy_given_z(d["y_pred_rf"].to_numpy(), d["y_true"].to_numpy(), Zm))

            df_boot = rf_temp.rename(columns={"y_pred": "y_pred_rf"})
            df_boot = df_boot.merge(
                pred_t[key_cols + ["y_pred", "temperature"]].rename(columns={"y_pred": "y_pred_temp"}),
                on=key_cols,
                how="inner",
            )
            n_before_drop = len(df_boot)
            df_boot = df_boot.dropna(subset=["y_pred_temp"])
            n_after_drop = len(df_boot)
            if n_before_drop > n_after_drop:
                n_dropped = n_before_drop - n_after_drop
                error_msg = (
                    f"Bootstrap analysis: dropped {n_dropped} rows with NaN y_pred_temp "
                    f"({100.0 * n_dropped / n_before_drop:.1f}% of data). "
                    f"This indicates upstream issues in temperature-only baseline predictions."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            delta_r_est = float("nan")
            r_partial_est = float("nan")
            r_partial_multi_est = float("nan")
            delta_r_ci = (float("nan"), float("nan"))
            r_partial_ci = (float("nan"), float("nan"))
            r_partial_multi_ci = (float("nan"), float("nan"))

            if not df_boot.empty:
                delta_r_est, delta_r_ci = cluster_bootstrap_subjects(df_boot, "subject_id", n_boot=int(config_dict["analysis"]["bootstrap_n"]), seed=seed, func=_metric_delta_r)
                r_partial_est, r_partial_ci = cluster_bootstrap_subjects(df_boot, "subject_id", n_boot=int(config_dict["analysis"]["bootstrap_n"]), seed=seed + 1, func=_metric_partial_r)
                r_partial_multi_est, r_partial_multi_ci = cluster_bootstrap_subjects(df_boot, "subject_id", n_boot=int(config_dict["analysis"]["bootstrap_n"]), seed=seed + 2, func=_metric_partial_r_multi)

            inc_summary = {
                "RandomForest": {"pearson_r": r_rf, "r2": r2_rf_subset},
                "TemperatureOnly": {"pearson_r": r_t, "r2": r2_t},
                "delta_r": {"estimate": float(delta_r_est), "ci95": [float(delta_r_ci[0]), float(delta_r_ci[1])]},
                "partial_r_given_temperature": {"estimate": float(r_partial_est), "ci95": [float(r_partial_ci[0]), float(r_partial_ci[1])]},
                "partial_r_given_temp_trial_subjectmean": {"estimate": float(r_partial_multi_est), "ci95": [float(r_partial_multi_ci[0]), float(r_partial_multi_ci[1])]},
                "delta_r2_incremental": {"estimate": float(r2_partial), "note": "approx partial r^2 (unique variance beyond temperature)"},
                "delta_r2_incremental_multi": {"estimate": float(r2_partial_multi), "note": "approx partial r^2 (unique variance beyond temp, trial, subj mean)"},
                "delta_r2_full_minus_temp": float(delta_r2_full_minus_temp),
            }

    ensure_dir((results_dir / config_dict["paths"]["summaries"]["incremental"]).parent)
    with open(results_dir / config_dict["paths"]["summaries"]["incremental"], "w", encoding="utf-8") as f:
        json.dump(inc_summary, f, indent=2)
    logger.info(f"Saved incremental validity summary at {results_dir / config_dict['paths']['summaries']['incremental']}")

# Baseline analysis
    y_true_bg, y_pred_bg, groups_bg, test_idx_bg, fold_bg = loso_baseline_predictions(y_all, groups, mode="global")
    pooled_bg, per_subj_bg = compute_metrics(y_true_bg, y_pred_bg, np.asarray(groups_bg))
    logger.info(f"Baseline (global mean) pooled: r={pooled_bg['pearson_r']:.3f}, R2={pooled_bg['r2']:.3f}")

    extra: dict = {}
    if "run_id" in meta.columns:
        extra["run_id"] = meta.loc[test_idx_bg, "run_id"].tolist()
    
    pred_bg = pd.DataFrame({
        "subject_id": groups_bg,
        "trial_id": meta.loc[test_idx_bg, "trial_id"].values,
        "y_true": y_true_bg,
        "y_pred": y_pred_bg,
        "fold": fold_bg,
        "model": "BaselineGlobal",
        **extra,
    })
    ensure_dir((results_dir / config_dict["paths"]["predictions"]["baseline_global"]).parent)
    pred_bg.to_csv(results_dir / config_dict["paths"]["predictions"]["baseline_global"], sep="\t", index=False)
    ensure_dir((results_dir / config_dict["paths"]["per_subject_metrics"]["baseline_global"]).parent)
    per_subj_bg.to_csv(results_dir / config_dict["paths"]["per_subject_metrics"]["baseline_global"], sep="\t", index=False)
    
    # Save indices
    idx_bg = pd.DataFrame({
        "subject_id": groups_bg,
        "trial_id": meta.loc[test_idx_bg, "trial_id"].values,
        "fold": fold_bg,
        **extra,
    })
    idx_bg["heldout_subject_id"] = idx_bg["subject_id"].astype(str)
    ensure_dir((results_dir / config_dict["paths"]["indices"]["baseline_global"]).parent)
    idx_bg.to_csv(results_dir / config_dict["paths"]["indices"]["baseline_global"], sep="\t", index=False)

    logger.info("Diagnostic subject-test baseline removed to prevent data leakage.")

    # Riemann analysis
    pooled_riem = None
    if check_pyriemann() and config_dict["flags"]["run_riemann"]:
        y_true_r, y_pred_r, pooled_riem, per_subj_riem = loso_riemann_regression(
            deriv_root=deriv_root,
            subjects=subjects,
            task=task,
            results_dir=results_dir,
            n_jobs=n_jobs,
            seed=seed,
            outer_n_jobs=outer_n_jobs,
            config_dict=config_dict,
        )
    else:
        logger.warning("pyriemann not installed; skipping Model 2. Install with `pip install pyriemann`.")

    # Riemannian covariance-based insights
    if check_pyriemann():
        riemann_export_cov_bins_per_fold(
            deriv_root=deriv_root,
            subjects=subjects,
            task=task,
            export_dir=export_dir,
            plateau_window=tuple(config_dict["analysis"]["riemann"]["plateau_window"]),
        )
        riemann_export_cov_bins(
            deriv_root=deriv_root,
            subjects=subjects,
            task=task,
            export_dir=export_dir,
            plateau_window=tuple(config_dict["analysis"]["riemann"]["plateau_window"]),
        )
        run_riemann_band_limited_decoding(
            deriv_root=deriv_root,
            subjects=subjects,
            task=task,
            results_dir=results_dir,
            export_dir=export_dir,
            bands=config_dict["analysis"]["riemann"]["bands"],
            n_jobs=n_jobs,
            seed=seed,
            outer_n_jobs=outer_n_jobs,
            config_dict=config_dict,
        )
        run_riemann_sliding_window(
            deriv_root=deriv_root,
            subjects=subjects,
            task=task,
            results_dir=results_dir,
            plateau_window=tuple(config_dict["analysis"]["riemann"]["plateau_window"]),
            window_len=float(config_dict["analysis"]["riemann"]["sliding_window"]["window_len"]),
            step=float(config_dict["analysis"]["riemann"]["sliding_window"]["step"]),
            n_jobs=n_jobs,
            seed=seed,
            outer_n_jobs=outer_n_jobs,
            config_dict=config_dict,
        )

    # Subject-ID decodability check
    subject_id_decodability_auc(X_all, groups, results_dir=results_dir, seed=seed)

    # Bootstrap CIs for pooled metrics
    bootstrap_results = {}
    model_preds = []
    if 'pred_en' in locals():
        model_preds.append(("ElasticNet", pred_en))
    if 'pred_wen' in locals():
        model_preds.append(("ElasticNetWithinKFold", pred_wen))
    if 'pred_rf' in locals():
        model_preds.append(("RandomForest", pred_rf))
    if 'pred_wrf' in locals():
        model_preds.append(("RandomForestWithinKFold", pred_wrf))
    if 'pred_bg' in locals():
        model_preds.append(("BaselineGlobal", pred_bg))
    
    for name, df_pred in model_preds:
        res = bootstrap_pooled_metrics_by_subject(df_pred[["subject_id", "y_true", "y_pred"]].copy(), seed=seed, config_dict=config_dict)
        bootstrap_results[name] = res
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "summary_bootstrap.json", "w", encoding="utf-8") as f:
        json.dump(bootstrap_results, f, indent=2)
    logger.info(f"Saved bootstrap CIs at {results_dir / 'summary_bootstrap.json'}")

    # Rank-based robustness: Spearman's rho vs Pearson's r
    models_stats = []
    if 'pred_en' in locals():
        models_stats.append((per_subject_pearson_and_spearman(pred_en), "ElasticNet", "#1f77b4"))
    if 'pred_rf' in locals():
        models_stats.append((per_subject_pearson_and_spearman(pred_rf), "RandomForest", "#ff7f0e"))
    if 'pred_wen' in locals():
        models_stats.append((per_subject_pearson_and_spearman(pred_wen), "EN-WithinKFold", "#2ca02c"))
    if 'pred_wrf' in locals():
        models_stats.append((per_subject_pearson_and_spearman(pred_wrf), "RF-WithinKFold", "#d62728"))

    # Results dashboard TSV
    ensure_dir((results_dir / config_dict["paths"]["summaries"]["all_metrics_wide"]).parent)
    build_all_metrics_wide(results_dir, results_dir / config_dict["paths"]["summaries"]["all_metrics_wide"])

    # Export combined best-params maps
    en_combined = read_best_params_jsonl_combined(best_params_en_path)
    rf_combined = read_best_params_jsonl_combined(best_params_rf_path)
    with open(results_dir / "best_params_elasticnet_combined.json", "w", encoding="utf-8") as f:
        json.dump(en_combined, f, indent=2)
    with open(results_dir / "best_params_rf_combined.json", "w", encoding="utf-8") as f:
        json.dump(rf_combined, f, indent=2)

    # Export long-form best params tables
    export_best_params_long_table(best_params_en_path, results_dir / "best_params_elasticnet_long.tsv")
    export_best_params_long_table(best_params_rf_path, results_dir / "best_params_rf_long.tsv")

    # Export blocks source provenance
    blk_info = {"source": blocks_source, "events_provenance": blocks_provenance, "reliability": blocks_reliability}
    if blocks_all is not None and hasattr(meta, "subject_id"):
        blk_counts = {}
        for sid in pd.unique(meta["subject_id"].astype(str)):
            rows = meta.index[meta["subject_id"].astype(str) == sid].to_numpy()
            if len(rows) > 0:
                vals = np.asarray(blocks_all)[rows]
                blk_counts[str(sid)] = int(len(pd.unique(vals)))
        blk_info["unique_blocks_per_subject"] = blk_counts
    with open(results_dir / "blocks_source.json", "w", encoding="utf-8") as f:
        json.dump(blk_info, f, indent=2)

    # Group-level feature importance aggregation
    if config_dict.get("flags", {}).get("run_group_topomaps", False) and len(subjects) >= 3:
        logger.info("Running group-level feature importance aggregation...")
        agg = str(config_dict.get("viz", {}).get("coef_agg", "abs")).lower()
        if agg not in ("abs", "signed"):
            agg = "abs"
        
        if 'pooled_en' in locals():
            min_subjects = int(config_dict.get("analysis.min_subjects_for_topomaps", 3))
            result = aggregate_group_feature_topomaps(
                subjects=subjects,
                deriv_root=deriv_root,
                task=task,
                method="elasticnet",
                aggregate=agg,
                target="rating",
                min_subjects=min_subjects,
                config=config_dict,
            )
            if result:
                logger.info(f"ElasticNet group feature aggregation saved at {result}")
        
        if 'pooled_rf' in locals():
            min_subjects = int(config_dict.get("analysis.min_subjects_for_topomaps", 3))
            result = aggregate_group_feature_topomaps(
                subjects=subjects,
                deriv_root=deriv_root,
                task=task,
                method="rfperm",
                aggregate="mean",
                target="rating",
                min_subjects=min_subjects,
                config=config_dict,
            )
            if result:
                logger.info(f"RF perm group feature aggregation saved at {result}")

    # Summary JSON
    versions = {
        "sklearn": getattr(__import__('sklearn'), '__version__', None),
        "mne": getattr(__import__('mne'), '__version__', None),
    }
    try:
        import pyriemann as _pr
        versions["pyriemann"] = getattr(_pr, "__version__", None)
    except ImportError:
        versions["pyriemann"] = None

    summary = {
        "BaselineGlobal": pooled_bg,
        "ElasticNet": pooled_en,
        "ElasticNetWithinKFold": (pooled_wen if 'pooled_wen' in locals() else None),
        "RandomForest": pooled_rf,
        "RandomForestWithinKFold": (pooled_wrf if 'pooled_wrf' in locals() else None),
        "Riemann": pooled_riem,
        "n_trials": int(len(X_all)),
        "n_subjects": int(len(np.unique(groups))),
        "n_features": int(X_all.shape[1]),
        "versions": versions,
    }
    summary_path = results_dir / config_dict["paths"]["summaries"]["summary"]
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary at {summary_path}")
    
    if not no_plots:
        plot_vars = {
            'pred_en': locals().get('pred_en'),
            'pooled_en': locals().get('pooled_en'),
            'per_subj_en': locals().get('per_subj_en'),
            'pred_rf': locals().get('pred_rf'),
            'pooled_rf': locals().get('pooled_rf'),
            'per_subj_rf': locals().get('per_subj_rf'),
            'cal_metrics': locals().get('cal_metrics'),
            'pred_wen': locals().get('pred_wen'),
            'pooled_wen': locals().get('pooled_wen'),
            'per_subj_wen': locals().get('per_subj_wen'),
            'pred_wrf': locals().get('pred_wrf'),
            'pooled_wrf': locals().get('pooled_wrf'),
            'per_subj_wrf': locals().get('per_subj_wrf'),
            'pooled_bg': locals().get('pooled_bg'),
            'pooled_riem': locals().get('pooled_riem'),
            'pooled_t': locals().get('pooled_t'),
            'bootstrap_results': locals().get('bootstrap_results'),
            'inc_summary': locals().get('inc_summary'),
        }
        _generate_all_plots(
            export_dir, results_dir, config_dict, summary_path,
            **plot_vars
        )
    else:
        logger.info("Skipping all plotting operations (--no-plots enabled)")


###################################################################
# Command Line Interface
###################################################################

if __name__ == "__main__":
    import argparse

    configure_plotting()
    
    cfg = load_settings()
    cfg.apply_thread_limits()
    config_dict, legacy = prepare_config_dict(cfg)
    
    default_task = legacy["TASK"]
    random_state = legacy["RANDOM_STATE"]

    parser = argparse.ArgumentParser(description="Behavioral psychometrics and EEG decoding (script 05)")
    subj_group = parser.add_mutually_exclusive_group(required=True)
    subj_group.add_argument("--subjects", nargs="*", help="Subject IDs to process (e.g., 001 002)")
    subj_group.add_argument("--all-subjects", action="store_true", help="Process all subjects with available features")
    parser.add_argument("--task", default=default_task, help="Task label (default from config)")
    # General compute controls
    parser.add_argument("--n_jobs", type=int, default=1, help="Inner CV parallelism (default: 1)")
    parser.add_argument("--outer_n_jobs", type=int, default=1, help="Parallel outer LOSO folds (default: 1)")
    # Analysis knobs
    parser.add_argument("--n_perm_quick", type=int, default=int(config_dict.get("analysis", {}).get("n_perm_quick", 1000)), help="Quick permutation iterations")
    parser.add_argument("--n_perm_refit", type=int, default=int(config_dict.get("analysis", {}).get("n_perm_refit", 500)), help="Refit-based permutation iterations")
    parser.add_argument("--rf_perm_repeats", type=int, default=int(config_dict.get("analysis", {}).get("rf_perm_importance_repeats", 20)), help="RF permutation importance repeats")
    parser.add_argument("--perm_refit_n_jobs", type=int, default=int(config_dict.get("analysis", {}).get("perm_refit_n_jobs", 1)), help="Refit permutation parallelism")
    parser.add_argument("--bootstrap_n", type=int, default=int(config_dict.get("analysis", {}).get("bootstrap_n", 1000)), help="Bootstrap iterations for CIs")
    parser.add_argument("--inner_splits", type=int, default=int(config_dict.get("cv", {}).get("inner_splits", 5)), help="Inner CV splits")
    # Interpretability controls
    parser.add_argument("--montage", default=str(config_dict.get("viz", {}).get("montage", "standard_1005")), help="Montage: standard name or bids:[electrodes.tsv] or bids_auto")
    parser.add_argument("--coef_agg", choices=["abs", "signed"], default=str(config_dict.get("viz", {}).get("coef_agg", "abs")).lower(), help="ElasticNet coef aggregation")
    # Feature flags
    parser.add_argument("--no-within", action="store_true", help="Disable within-subject KFold analysis")
    parser.add_argument("--no-riemann", action="store_true", help="Disable Riemann decoding analyses")
    parser.add_argument("--no-shap", action="store_true", help="Disable SHAP analysis")
    parser.add_argument("--group-topomaps", action="store_true", help="Enable group-level feature importance aggregation with FDR correction")
    # Best-params behavior and run info
    parser.add_argument("--best-params-mode", choices=["append", "truncate", "run_scoped"], default=str(BEST_PARAMS_MODE), help="How to record CV best params")
    parser.add_argument("--run-id", default=None, help="Optional run identifier to tag outputs")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip all plotting operations"
    )
    args = parser.parse_args()

    # seeds
    np.random.seed(random_state)
    pyrandom.seed(random_state)

    # Apply CLI configs
    run_id = args.run_id
    best_params_mode = str(args.best_params_mode)
    config_dict["analysis"]["n_perm_quick"] = int(args.n_perm_quick)
    config_dict["analysis"]["n_perm_refit"] = int(args.n_perm_refit)
    config_dict["analysis"]["rf_perm_importance_repeats"] = int(args.rf_perm_repeats)
    config_dict["analysis"]["perm_refit_n_jobs"] = int(args.perm_refit_n_jobs)
    config_dict["analysis"]["bootstrap_n"] = int(args.bootstrap_n)
    config_dict["cv"]["inner_splits"] = int(args.inner_splits)
    # Interpretability options
    config_dict["viz"]["montage"] = args.montage
    config_dict["viz"]["coef_agg"] = args.coef_agg
    if args.no_within:
        config_dict["flags"]["run_within_subject_kfold"] = False
    if args.no_riemann:
        config_dict["flags"]["run_riemann"] = False
    if args.no_shap:
        config_dict["flags"]["run_shap"] = False
    if args.group_topomaps:
        config_dict["flags"]["run_group_topomaps"] = True

    main(subjects=args.subjects, task=args.task, n_jobs=args.n_jobs, seed=random_state, outer_n_jobs=int(args.outer_n_jobs), all_subjects=args.all_subjects, no_plots=args.no_plots)
