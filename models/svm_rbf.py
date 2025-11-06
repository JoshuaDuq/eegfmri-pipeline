import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd

from ..utils import (
    config_loader,
    cv_evaluation,
    data_loading,
    feature_utils,
    io_utils,
    metrics,
    model_builders,
    permutation,
    target_signatures,
)

###################################################################
# Constants
###################################################################

MODEL_NAME = "svm_rbf"


###################################################################
# Gamma Grid Normalization
###################################################################

def normalize_gamma_grid(raw_grid: Sequence[Union[str, float]]) -> List[Union[str, float]]:
    seen: set[tuple] = set()
    normalized: List[Union[str, float]] = []

    for raw in raw_grid:
        item: Union[str, float]
        if isinstance(raw, str):
            candidate = raw.strip().lower()
            if candidate in {"scale", "auto"}:
                item = candidate
            else:
                try:
                    item = float(raw)
                except ValueError as exc:
                    raise ValueError(f"Invalid gamma value '{raw}'.") from exc
        elif isinstance(raw, (int, float)):
            item = float(raw)
        else:
            raise TypeError(f"Gamma grid entries must be str or float, got {type(raw)!r}.")

        if isinstance(item, float):
            if item <= 0:
                raise ValueError("Gamma grid numeric values must be positive.")
            key = ("float", item)
        else:
            key = ("str", item)

        if key not in seen:
            seen.add(key)
            normalized.append(item)

    if not normalized:
        raise ValueError("Gamma grid must include at least one value after normalization.")

    return normalized


###################################################################
# Argument Parsing
###################################################################

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict NPS beta responses from EEG oscillatory power with an RBF-kernel SVR."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for outputs. Defaults to machine_learning/outputs/eeg_to_signature_svm_<timestamp>.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Subject identifiers (with or without sub- prefix). Defaults to all available.",
    )
    parser.add_argument(
        "--bands",
        nargs="*",
        default=list(feature_utils.DEFAULT_BANDS),
        help="EEG frequency bands to include (subset of %s)." % (feature_utils.SUPPORTED_BANDS,),
    )
    parser.add_argument(
        "--target-signature",
        type=str,
        default=target_signatures.DEFAULT_TARGET_KEY,
        choices=sorted(target_signatures.TARGET_SIGNATURES),
        help="fMRI signature to decode (default: %(default)s).",
    )
    parser.add_argument(
        "--include-temperature",
        action="store_true",
        help="Include stimulus temperature as an additional predictor.",
    )
    parser.add_argument(
        "--c-grid",
        nargs="+",
        default=None,
        help="C hyperparameter grid (positive values).",
    )
    parser.add_argument(
        "--gamma-grid",
        nargs="+",
        default=None,
        help="Gamma hyperparameter grid (scale, auto, or positive floats).",
    )
    parser.add_argument(
        "--epsilon-grid",
        nargs="+",
        default=None,
        help="Epsilon hyperparameter grid (non-negative values).",
    )
    parser.add_argument(
        "--svm-cache-size",
        type=float,
        default=None,
        help="SVR cache size in MB.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for grid searches (-1 uses all cores).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splits and models.",
    )
    parser.add_argument(
        "--permutation-seed",
        type=int,
        default=None,
        help="Random seed for label permutations (defaults to --random-state when omitted).",
    )
    parser.add_argument(
        "--permutation-count",
        type=int,
        default=0,
        help="Number of label permutations to run for R^2 significance testing (0 disables).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file. Defaults to utils/ml_config.yaml if not specified.",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Disable loading configuration from YAML file.",
    )
    args = parser.parse_args(argv)
    
    if not args.no_config:
        config = config_loader.load_config(args.config)
        config_loader.apply_config_to_args(config, args)
        
        svm_config = config_loader.get_model_config(config, "svm_rbf")
        if svm_config:
            if not hasattr(args, "c_grid") or not args.c_grid:
                args.c_grid = [str(v) for v in svm_config.get("param_grid", {}).get("C", ["0.1", "1.0", "10.0", "100.0"])]
            if not hasattr(args, "gamma_grid") or not args.gamma_grid:
                gamma_vals = svm_config.get("param_grid", {}).get("gamma", ["scale", "0.001", "0.01", "0.1"])
                args.gamma_grid = [str(v) if isinstance(v, (int, float)) else v for v in gamma_vals]
            if not hasattr(args, "epsilon_grid") or not args.epsilon_grid:
                args.epsilon_grid = [str(v) for v in svm_config.get("param_grid", {}).get("epsilon", ["0.01", "0.1", "0.5"])]
            if not hasattr(args, "svm_cache_size"):
                args.svm_cache_size = svm_config.get("cache_size", 200.0)
    
    return args


###################################################################
# Main Function
###################################################################

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    target = target_signatures.get_target_signature(args.target_signature)

    bands = tuple(dict.fromkeys(band.lower() for band in args.bands))
    if not bands:
        raise ValueError("No bands specified.")
    for band in bands:
        if band not in feature_utils.SUPPORTED_BANDS:
            raise ValueError(f"Unsupported band '{band}'. Supported bands: {feature_utils.SUPPORTED_BANDS}.")

    if not args.c_grid:
        args.c_grid = ["0.1", "1.0", "10.0", "100.0"]
    if not args.gamma_grid:
        args.gamma_grid = ["scale", "0.001", "0.01", "0.1"]
    if not args.epsilon_grid:
        args.epsilon_grid = ["0.01", "0.1", "0.5"]
    if args.svm_cache_size is None:
        args.svm_cache_size = 200.0

    c_grid = list(dict.fromkeys(float(val) for val in args.c_grid))
    if not c_grid:
        raise ValueError("C grid must include at least one value.")
    if any(val <= 0 for val in c_grid):
        raise ValueError("C grid values must be positive.")

    epsilon_grid = list(dict.fromkeys(float(val) for val in args.epsilon_grid))
    if not epsilon_grid:
        raise ValueError("Epsilon grid must include at least one value.")
    if any(val < 0 for val in epsilon_grid):
        raise ValueError("Epsilon grid values must be non-negative.")

    gamma_grid = normalize_gamma_grid(args.gamma_grid)

    repo_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = (
        Path(__file__).resolve().parent.parent / "outputs" / f"eeg_to_{target.key}_svm_{timestamp}"
    )
    output_dir = Path(args.output_dir) if args.output_dir else default_output
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = io_utils.setup_logging(output_dir, logger_name=f"eeg_to_{target.key}_svm")
    logger.info("EEG -> %s SVM training pipeline started.", target.display_name)
    logger.info("Using bands: %s", ", ".join(bands))
    logger.info("Hyperparameter grid | C: %s | gamma: %s | epsilon: %s", c_grid, gamma_grid, epsilon_grid)
    logger.info("SVR cache size: %.1f MB", args.svm_cache_size)

    eeg_deriv_root = repo_root / "eeg_pipeline" / "bids_output" / "derivatives"
    try:
        fmri_outputs_root = target_signatures.resolve_fmri_outputs_root(repo_root, target)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise SystemExit(1)
    logger.info("Using fMRI outputs root: %s", fmri_outputs_root)

    if not eeg_deriv_root.exists():
        logger.error("EEG derivatives directory not found: %s", eeg_deriv_root)
        raise SystemExit(1)

    available_subjects = data_loading.discover_subjects(eeg_deriv_root, fmri_outputs_root, target)
    if not available_subjects:
        logger.error(
            "No subjects with both EEG features and fMRI %s scores were found.",
            target.short_name,
        )
        raise SystemExit(1)
    logger.info("Available subjects with aligned data: %s", ", ".join(available_subjects))

    if args.subjects:
        requested = [s if s.startswith("sub-") else f"sub-{s}" for s in args.subjects]
        invalid = sorted(set(requested) - set(available_subjects))
        if invalid:
            logger.error("Requested subjects missing required data: %s", ", ".join(invalid))
            raise SystemExit(1)
        subjects = sorted(requested)
    else:
        subjects = available_subjects

    subject_results = []
    drops_summary: Dict[str, List[Dict[str, float]]] = {}

    for subject in subjects:
        result = data_loading.load_subject_dataset(subject, eeg_deriv_root, fmri_outputs_root, bands, target, logger)
        subject_results.append(result)
        if result.dropped_trials:
            drops_summary[subject] = result.dropped_trials

    if not subject_results:
        logger.error("No feature columns detected.")
        raise SystemExit(1)

    target_column = subject_results[0].target_column

    all_feature_sets = [set(res.feature_columns) for res in subject_results]
    if not all_feature_sets:
        logger.error("No feature columns available after loading subjects.")
        raise SystemExit(1)
    master_features = sorted(set().union(*all_feature_sets))
    if not master_features:
        logger.error("Feature union across subjects is empty.")
        raise SystemExit(1)

    metadata_cols = [
        "subject",
        "run",
        "trial_idx_run",
        "trial_idx_global",
        "temp_celsius",
        "vas_rating",
        "pain_binary",
        target_column,
    ]
    if target_column != "br_score":
        metadata_cols.append("br_score")

    for subj_idx, res in enumerate(subject_results):
        missing = sorted(set(master_features) - set(res.feature_columns))
        if missing:
            preview = ", ".join(missing[:10])
            if len(missing) > 10:
                preview += ", ..."
            logger.info(
                "  %s: filling %d missing feature(s) with NaN: %s",
                subjects[subj_idx],
                len(missing),
                preview,
            )
        res.data = res.data.reindex(columns=metadata_cols + master_features)
        res.feature_columns = list(master_features)

    data = pd.concat([res.data for res in subject_results], ignore_index=True)
    feature_columns = list(master_features)
    if args.include_temperature and "temp_celsius" not in feature_columns:
        feature_columns.append("temp_celsius")

    X = data[feature_columns].copy()
    y = data[target_column].copy()

    if y.isna().any():
        n_na = y.isna().sum()
        logger.error("Target variable (%s) contains %d NaN values; cannot proceed.", target_column, n_na)
        raise SystemExit(1)
    if np.isinf(y).any():
        n_inf = np.isinf(y).sum()
        logger.error("Target variable (%s) contains %d infinite values; cannot proceed.", target_column, n_inf)
        raise SystemExit(1)
    
    logger.info(
        "Target (%s) statistics: min=%.3f, max=%.3f, mean=%.3f, std=%.3f",
        target_column,
        y.min(),
        y.max(),
        y.mean(),
        y.std(),
    )

    if len(subjects) > 1:
        outer_groups = data["subject"].to_numpy()
        outer_group_level = "subject"
    else:
        outer_groups = data["run"].to_numpy()
        outer_group_level = "run"
    composite_run_ids = (
        data["subject"].astype(str).str.strip()
        + "__run__"
        + data["run"].astype(str).str.strip()
    ).to_numpy()
    run_groups = composite_run_ids
    logger.info(
        "Inner CV grouping will use subject+run composites (%d unique groups).",
        len(np.unique(run_groups)),
    )

    logger.info(
        "Assembled dataset: %d trials, %d subjects, %d features.",
        len(data),
        len(subjects),
        len(feature_columns),
    )

    feature_ratio = len(feature_columns) / max(len(data), 1)
    logger.info(
        "Feature-to-sample ratio: %.2f (%d features / %d trials)",
        feature_ratio,
        len(feature_columns),
        len(data),
    )
    if feature_ratio > 0.1:
        logger.warning(
            "Feature-to-sample ratio %.2f exceeds 0.10; consider stronger regularisation, feature selection, or dimensionality reduction.",
            feature_ratio,
        )

    if "temp_celsius" in data.columns:
        temp_counts = data["temp_celsius"].value_counts().sort_index()
        logger.info("Temperature distribution (counts per condition):\n%s", temp_counts.to_string())
    else:
        temp_counts = pd.Series(dtype=int)

    logger.info(
        "%s betas reflect the delayed (~5-7 s) hemodynamic response; EEG features are interpreted with this lag in mind.",
        target.display_name,
    )

    plateau_window = None
    eeg_config_path = repo_root / "eeg_pipeline" / "utils" / "eeg_config.yaml"
    if eeg_config_path.exists():
        try:
            import yaml
            cfg = yaml.safe_load(eeg_config_path.read_text())
            plateau_window = cfg.get("time_frequency_analysis", {}).get("plateau_window")
            if plateau_window is not None:
                logger.info("EEG plateau window from config: %s seconds", plateau_window)
                if not isinstance(plateau_window, (int, float)) or float(plateau_window) <= 0:
                    raise ValueError(
                        "Invalid plateau_window in eeg_config.yaml; expected positive number of seconds."
                    )
                plateau_window = float(plateau_window)
                if plateau_window < 3 or plateau_window > 10:
                    logger.warning(
                        "Configured plateau window %.2f s is outside the typical 3-10 s range for HRF-aligned EEG features.",
                        plateau_window,
                    )
        except Exception as exc:
            logger.warning("Could not read EEG plateau window from %s: %s", eeg_config_path, exc)
    else:
        logger.warning("EEG config file not found at %s; plateau window not verified.", eeg_config_path)

    if plateau_window is None:
        raise ValueError(
            "Unable to determine EEG plateau window from configuration; verify EEG/fMRI alignment before training."
        )

    temperature_baseline_metrics: Optional[Dict[str, float]] = None
    temperature_baseline_desc: Optional[str] = None
    if args.include_temperature and "temp_celsius" in data.columns:
        temperature_baseline_metrics, temperature_baseline_desc = metrics.compute_temperature_baseline_cv(
            temp=data["temp_celsius"],
            target=y,
            outer_groups=outer_groups,
            random_state=args.random_state,
            logger=logger,
        )

    temperature_predictor_note = (
        "Temperature covariate included as predictor alongside EEG features."
        if args.include_temperature and "temp_celsius" in feature_columns
        else "Temperature covariate excluded from predictors; per-temperature diagnostics reflect distributional balance only."
    )
    logger.info(temperature_predictor_note)

    svm_builder = model_builders.make_svm_builder(cache_size=args.svm_cache_size)
    param_grid = {
        "svm__C": c_grid,
        "svm__gamma": gamma_grid,
        "svm__epsilon": epsilon_grid,
    }

    grid_size = int(np.prod([len(v) for v in param_grid.values()])) if param_grid else 1
    if grid_size * 5 > len(X):
        logger.warning(
            "SVM grid (%d combinations) is large relative to available trials (%d); consider trimming the search space.",
            grid_size,
            len(X),
        )

    result = cv_evaluation.nested_cv_evaluate(
        model_name=MODEL_NAME,
        builder=svm_builder,
        param_grid=param_grid,
        X=X,
        y=y,
        feature_names=feature_columns,
        meta=data,
        outer_groups=outer_groups,
        run_groups=run_groups,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        logger=logger,
    )

    pred_df = metrics.build_prediction_frame(
        data=data,
        y_true=y,
        y_pred=result["predictions"],
        model_name=MODEL_NAME,
        target_column=target_column,
        target_key=target.key,
        fold_assignments=result["fold_assignments"],
    )
    pred_path = output_dir / f"predictions_{MODEL_NAME}.tsv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_path, sep="\t", index=False)

    subj_metrics = metrics.compute_group_metrics(pred_df, ["subject"])
    subj_metrics_path = output_dir / f"per_subject_metrics_{MODEL_NAME}.tsv"
    subj_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    subj_metrics.to_csv(subj_metrics_path, sep="\t", index=False)
    temp_metrics = metrics.compute_group_metrics(pred_df, ["temp_celsius"])
    temp_metrics_path = output_dir / f"per_temperature_metrics_{MODEL_NAME}.tsv"
    temp_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    temp_metrics.to_csv(temp_metrics_path, sep="\t", index=False)

    fold_path = None
    fold_df = pd.DataFrame(result["fold_details"])
    if not fold_df.empty:
        fold_df["best_params"] = fold_df["best_params"].apply(lambda d: json.dumps(d))
        if "test_temp_counts" in fold_df.columns:
            fold_df["test_temp_counts"] = fold_df["test_temp_counts"].apply(lambda d: json.dumps(d))
        fold_path = output_dir / f"cv_folds_{MODEL_NAME}.tsv"
        fold_path.parent.mkdir(parents=True, exist_ok=True)
        fold_df.to_csv(fold_path, sep="\t", index=False)

    metrics_path = output_dir / f"metrics_{MODEL_NAME}.json"
    io_utils.write_json(metrics_path, result["summary_metrics"])

    best_params_path = output_dir / f"best_params_{MODEL_NAME}.json"
    io_utils.write_json(best_params_path, [fold["best_params"] for fold in result["fold_details"]])

    r2_values = [fold["test_r2"] for fold in result["fold_details"]]
    svm_entry: Dict[str, Any] = {
        "name": MODEL_NAME,
        "metrics": result["summary_metrics"],
        "fold_mean_r2": float(np.mean(r2_values)) if r2_values else None,
        "fold_std_r2": float(np.std(r2_values)) if r2_values else None,
        "prediction_file": pred_path.name,
        "per_subject_metrics_file": subj_metrics_path.name,
        "per_temperature_metrics_file": temp_metrics_path.name,
        "fold_details_file": fold_path.name if fold_path else None,
        "metrics_file": metrics_path.name,
        "best_params_file": best_params_path.name,
        "outer_cv": result.get("outer_cv_desc"),
        "inner_cv": result.get("inner_cv_desc"),
        "param_grid": param_grid,
    }

    if args.permutation_count > 0:
        perm_summary, perm_null = permutation.permutation_test_r2(
            model_name=MODEL_NAME,
            builder=svm_builder,
            param_grid=param_grid,
            X=X,
            y=y,
            feature_names=feature_columns,
            meta=data,
            outer_groups=outer_groups,
            run_groups=run_groups,
            n_permutations=args.permutation_count,
            true_r2=result["summary_metrics"]["r2"],
            random_state=args.permutation_seed if args.permutation_seed is not None else args.random_state,
            n_jobs=args.n_jobs,
            logger=logger,
        )
        perm_json_path = output_dir / f"permutation_{MODEL_NAME}.json"
        io_utils.write_json(perm_json_path, perm_summary)
        null_path = output_dir / f"permutation_{MODEL_NAME}_null.npy"
        np.save(null_path, perm_null)
        svm_entry["permutation_test"] = {
            "p_value": perm_summary["p_value"],
            "true_r2": perm_summary["true_r2"],
            "null_mean": perm_summary["null_mean"],
            "null_std": perm_summary["null_std"],
            "null_quantiles": perm_summary["null_quantiles"],
            "result_file": perm_json_path.name,
            "null_distribution_file": null_path.name,
        }

    final_stratify = data["temp_celsius"] if "temp_celsius" in data.columns else None
    final_estimator, final_best_params, final_cv_score, final_cv_desc = cv_evaluation.fit_final_estimator(
        model_name=MODEL_NAME,
        builder=svm_builder,
        param_grid=param_grid,
        X=X,
        y=y,
        run_groups=run_groups,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        stratify_labels=final_stratify,
        logger=logger,
    )
    logger.info("SVM refit using %s", final_cv_desc)
    final_predictions = final_estimator.predict(X)
    final_metrics = metrics.compute_metrics(y.to_numpy(), final_predictions)

    final_pred_df = metrics.build_prediction_frame(
        data=data,
        y_true=y,
        y_pred=final_predictions,
        model_name=MODEL_NAME,
        target_column=target_column,
        target_key=target.key,
    )
    final_pred_path = output_dir / f"final_model_predictions_{MODEL_NAME}.tsv"
    final_pred_path.parent.mkdir(parents=True, exist_ok=True)
    final_pred_df.to_csv(final_pred_path, sep="\t", index=False)

    final_subj_metrics = metrics.compute_group_metrics(final_pred_df, ["subject"])
    final_subj_metrics_path = output_dir / f"final_per_subject_metrics_{MODEL_NAME}.tsv"
    final_subj_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    final_subj_metrics.to_csv(final_subj_metrics_path, sep="\t", index=False)

    final_temp_metrics = metrics.compute_group_metrics(final_pred_df, ["temp_celsius"])
    final_temp_metrics_path = output_dir / f"final_per_temperature_metrics_{MODEL_NAME}.tsv"
    final_temp_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    final_temp_metrics.to_csv(final_temp_metrics_path, sep="\t", index=False)

    model_payload = {
        "model": final_estimator,
        "feature_names": feature_columns,
        "bands": list(bands),
        "subjects": subjects,
        "target": {
            "key": target.key,
            "display_name": target.display_name,
            "column": target_column,
            "description": target.description,
        },
        "trained_on": {
            "n_trials": int(len(data)),
            "outer_group_level": outer_group_level,
        },
    }
    joblib.dump(model_payload, output_dir / f"final_model_{MODEL_NAME}.joblib")

    temperature_distribution = temp_counts.to_dict() if isinstance(temp_counts, pd.Series) else {}
    summary = {
        "target": {
            "key": target.key,
            "display_name": target.display_name,
            "column": target_column,
            "description": target.description,
        },
        "bands": list(bands),
        "n_subjects": len(subjects),
        "subjects": subjects,
        "n_trials": int(len(data)),
        "feature_count": len(feature_columns),
        "feature_to_sample_ratio": feature_ratio,
        "temperature_distribution": temperature_distribution,
        "eeg_plateau_window": plateau_window,
        "temperature_only_r2": temperature_baseline_metrics["r2"] if temperature_baseline_metrics else None,
        "temperature_only_baseline": {
            "metrics": temperature_baseline_metrics,
            "cv_strategy": temperature_baseline_desc,
        }
        if temperature_baseline_metrics
        else None,
        "outer_cv_level": outer_group_level,
        "outer_cv_strategy": result.get("outer_cv_desc"),
        "inner_cv_strategy": result.get("inner_cv_desc"),
        "include_temperature": bool(args.include_temperature),
        "models": [svm_entry],
        "best_model": {
            "name": MODEL_NAME,
            "cv_best_score": final_cv_score,
            "final_best_params": final_best_params,
            "final_metrics": final_metrics,
            "prediction_file": final_pred_path.name,
            "per_subject_metrics_file": final_subj_metrics_path.name,
            "per_temperature_metrics_file": final_temp_metrics_path.name,
            "model_artifact_file": f"final_model_{MODEL_NAME}.joblib",
            "refit_cv_strategy": final_cv_desc,
            "permutation_test": svm_entry.get("permutation_test"),
        },
        "notes": [
            f"Target signature: {target.display_name} (column '{target_column}').",
            "EEG features derive from the stimulation plateau window and are aligned to delayed hemodynamic responses.",
        ],
    }
    summary["notes"].append(temperature_predictor_note)
    if temperature_baseline_metrics:
        summary["notes"].append(
            "Temperature-only baseline R² = %.3f (%s)."
            % (temperature_baseline_metrics["r2"], temperature_baseline_desc)
        )
    if drops_summary:
        summary["dropped_trials"] = drops_summary

    io_utils.write_json(output_dir / "summary.json", summary)
    logger.info("SVM final R2=%.3f", final_metrics["r2"])
    logger.info("All outputs written to %s", output_dir)


if __name__ == "__main__":
    main()

