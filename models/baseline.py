import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

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

IN_SAMPLE_METRIC_NOTE = (
    "Refit on the full dataset; metrics reflect in-sample (resubstitution) diagnostics only."
)


###################################################################
# Argument Parsing
###################################################################

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict fMRI signature beta responses (e.g., NPS, SIIPS1) from EEG oscillatory features."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store outputs. Defaults to machine_learning/outputs/eeg_to_<target>_<timestamp>.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Subject identifiers (with or without sub- prefix). Default uses all available.",
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
        help="fMRI signature to predict (default: %(default)s).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["elasticnet", "random_forest"],
        help="Model names to evaluate (elasticnet, random_forest).",
    )
    parser.add_argument(
        "--include-temperature",
        action="store_true",
        help="Include trial temperature as an additional predictor.",
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
        "--permutation-per-model",
        type=int,
        default=0,
        help="Number of label permutations to run per model for R^2 significance testing (0 disables).",
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
        
        baseline_config = config_loader.get_model_config(config, "baseline")
        if baseline_config and not args.models:
            args.models = baseline_config.get("models", ["elasticnet", "random_forest"])
    
    return args


###################################################################
# Main Function
###################################################################

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    target = target_signatures.get_target_signature(args.target_signature)

    bands = tuple(dict.fromkeys(band.lower() for band in args.bands))
    for band in bands:
        if band not in feature_utils.SUPPORTED_BANDS:
            raise ValueError(f"Unsupported band '{band}'. Supported bands: {feature_utils.SUPPORTED_BANDS}.")
    if not bands:
        raise ValueError("No bands specified.")

    repo_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = (
        Path(__file__).resolve().parent.parent / "outputs" / f"eeg_to_{target.key}_{timestamp}"
    )
    output_dir = Path(args.output_dir) if args.output_dir else default_output
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = io_utils.setup_logging(output_dir, logger_name=f"eeg_to_{target.key}")
    logger.info("EEG -> %s training pipeline started.", target.display_name)
    logger.info("Using bands: %s", ", ".join(bands))

    eeg_deriv_root = repo_root / "eeg_pipeline" / "bids_output" / "derivatives"
    try:
        fmri_outputs_root = target_signatures.resolve_fmri_outputs_root(repo_root, target)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    logger.info("Using fMRI outputs root: %s", fmri_outputs_root)

    if not eeg_deriv_root.exists():
        logger.error("EEG derivatives directory not found: %s", eeg_deriv_root)
        sys.exit(1)

    available_subjects = data_loading.discover_subjects(eeg_deriv_root, fmri_outputs_root, target)
    if not available_subjects:
        logger.error(
            "No subjects with both EEG features and fMRI %s scores were found.",
            target.short_name,
        )
        sys.exit(1)
    logger.info("Available subjects with aligned data: %s", ", ".join(available_subjects))

    if args.subjects:
        requested = [s if s.startswith("sub-") else f"sub-{s}" for s in args.subjects]
        invalid = sorted(set(requested) - set(available_subjects))
        if invalid:
            logger.error("Requested subjects missing required data: %s", ", ".join(invalid))
            sys.exit(1)
        subjects = sorted(requested)
    else:
        subjects = available_subjects

    subject_results: List[data_loading.SubjectDataset] = []
    drops_summary: Dict[str, List[Dict[str, float]]] = {}

    for subject in subjects:
        result = data_loading.load_subject_dataset(
            subject,
            eeg_deriv_root,
            fmri_outputs_root,
            bands,
            target,
            logger,
        )
        subject_results.append(result)
        if result.dropped_trials:
            drops_summary[subject] = result.dropped_trials

    if not subject_results:
        logger.error("No subjects loaded.")
        sys.exit(1)

    target_column = subject_results[0].target_column
    if any(res.target_column != target_column for res in subject_results):
        raise ValueError("Inconsistent target columns across subjects; expected uniform target.")

    all_feature_sets: List[set[str]] = [set(res.feature_columns) for res in subject_results]
    if not all_feature_sets:
        logger.error("No feature columns detected in loaded subjects.")
        sys.exit(1)

    master_feature_set = set().union(*all_feature_sets)
    if not master_feature_set:
        logger.error("Feature union across subjects is empty; check preprocessing outputs.")
        sys.exit(1)

    feature_columns = sorted(master_feature_set)
    logger.info(
        "Using unified feature list with %d columns (subject-specific gaps will be imputed).",
        len(feature_columns),
    )

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
        missing = sorted(master_feature_set - set(res.feature_columns))
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
        res.data = res.data.reindex(columns=metadata_cols + feature_columns)
        res.feature_columns = list(feature_columns)

    data = pd.concat([res.data for res in subject_results], ignore_index=True)
    if args.include_temperature and "temp_celsius" not in feature_columns:
        feature_columns.append("temp_celsius")

    X = data[feature_columns].copy()
    y = data[target_column].copy()

    if y.isna().any():
        n_na = y.isna().sum()
        logger.error("Target variable (%s) contains %d NaN values; cannot proceed.", target_column, n_na)
        sys.exit(1)
    if np.isinf(y).any():
        n_inf = np.isinf(y).sum()
        logger.error("Target variable (%s) contains %d infinite values; cannot proceed.", target_column, n_inf)
        sys.exit(1)
    
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
        data["subject"].astype(str) + "_run" + data["run"].astype(int).astype(str)
    ).to_numpy()
    run_groups = composite_run_ids
    logger.info(
        "Inner CV grouping will use subject+run composites (%d unique groups).",
        len(np.unique(run_groups)),
    )

    logger.info("Assembled dataset: %d trials, %d subjects, %d features.", len(data), len(subjects), len(feature_columns))

    feature_ratio = len(feature_columns) / max(len(data), 1)
    logger.info(
        "Feature-to-sample ratio: %.2f (%d features / %d trials)",
        feature_ratio,
        len(feature_columns),
        len(data),
    )
    high_dimensionality = feature_ratio > 0.1
    if high_dimensionality:
        logger.warning(
            "Feature-to-sample ratio %.2f exceeds 0.10; expanding model search to favour stronger shrinkage."
            % feature_ratio
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
                if isinstance(plateau_window, (list, tuple)) and len(plateau_window) == 2:
                    plateau_window = float(plateau_window[1]) - float(plateau_window[0])
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
    if "temp_celsius" in data.columns:
        try:
            temperature_baseline_metrics, temperature_baseline_desc = metrics.compute_temperature_baseline_cv(
                temp=data["temp_celsius"],
                target=y,
                outer_groups=outer_groups,
                random_state=args.random_state,
                logger=logger,
            )
        except Exception as exc:
            logger.warning(
                "Temperature-only baseline could not be computed: %s", exc
            )

    temperature_predictor_note = (
        "Temperature covariate included as predictor alongside EEG features."
        if args.include_temperature and "temp_celsius" in feature_columns
        else "Temperature covariate excluded from predictors; per-temperature diagnostics reflect distributional balance only."
    )
    logger.info(temperature_predictor_note)

    model_names = [name.lower() for name in args.models]
    for name in model_names:
        if name not in model_builders.MODEL_REGISTRY:
            logger.error("Unknown model '%s'. Available models: %s", name, ", ".join(model_builders.MODEL_REGISTRY.keys()))
            sys.exit(1)

    model_results: List[Dict[str, object]] = []
    summary_model_entries: List[Dict[str, object]] = []
    outer_cv_strategy_record: Optional[str] = None
    inner_cv_strategy_record: Optional[str] = None

    model_configs: Dict[str, Tuple] = {}
    for name in model_names:
        builder, base_grid = model_builders.MODEL_REGISTRY[name]
        param_grid = model_builders.adjust_param_grid_for_high_dimensionality(
            name,
            base_grid,
            high_dimensionality=high_dimensionality,
            logger=logger,
        )
        model_configs[name] = (builder, param_grid)

    for name in model_names:
        builder, param_grid = model_configs[name]
        grid_size = int(np.prod([len(v) for v in param_grid.values()])) if param_grid else 1
        if grid_size * 5 > len(X):
            logger.warning(
                "Model %s grid (%d combinations) is large relative to available trials (%d); consider simplifying the search space or switching to randomized search.",
                name,
                grid_size,
                len(X),
            )

        result = cv_evaluation.nested_cv_evaluate(
            model_name=name,
            builder=builder,
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
        model_results.append(result)

        if outer_cv_strategy_record is None:
            outer_cv_strategy_record = result.get("outer_cv_desc")
        if inner_cv_strategy_record is None:
            inner_cv_strategy_record = result.get("inner_cv_desc")

        pred_df = metrics.build_prediction_frame(
            data=data,
            y_true=y,
            y_pred=result["predictions"],
            model_name=name,
            target_column=target_column,
            target_key=target.key,
            fold_assignments=result["fold_assignments"],
        )
        pred_path = output_dir / f"predictions_{name}.tsv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(pred_path, sep="\t", index=False)

        subj_metrics = metrics.compute_group_metrics(pred_df, ["subject"])
        subj_metrics_path = output_dir / f"per_subject_metrics_{name}.tsv"
        subj_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        subj_metrics.to_csv(subj_metrics_path, sep="\t", index=False)

        temp_metrics = metrics.compute_group_metrics(pred_df, ["temp_celsius"])
        temp_metrics_path = output_dir / f"per_temperature_metrics_{name}.tsv"
        temp_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        temp_metrics.to_csv(temp_metrics_path, sep="\t", index=False)

        fold_path = None
        fold_df = pd.DataFrame(result["fold_details"])
        if not fold_df.empty:
            fold_df["best_params"] = fold_df["best_params"].apply(lambda d: json.dumps(d))
            if "test_temp_counts" in fold_df.columns:
                fold_df["test_temp_counts"] = fold_df["test_temp_counts"].apply(lambda d: json.dumps(d))
            fold_path = output_dir / f"cv_folds_{name}.tsv"
            fold_path.parent.mkdir(parents=True, exist_ok=True)
            fold_df.to_csv(fold_path, sep="\t", index=False)

        metrics_path = output_dir / f"metrics_{name}.json"
        io_utils.write_json(metrics_path, result["summary_metrics"])

        best_params_path = output_dir / f"best_params_{name}.json"
        io_utils.write_json(best_params_path, [fold["best_params"] for fold in result["fold_details"]])

        r2_values = [fold["test_r2"] for fold in result["fold_details"]]
        entry: Dict[str, Any] = {
            "name": name,
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
        }

        if args.permutation_per_model > 0:
            perm_summary, perm_null = permutation.permutation_test_r2(
                model_name=name,
                builder=builder,
                param_grid=param_grid,
                X=X,
                y=y,
                feature_names=feature_columns,
                meta=data,
                outer_groups=outer_groups,
                run_groups=run_groups,
                n_permutations=args.permutation_per_model,
                true_r2=result["summary_metrics"]["r2"],
                random_state=args.permutation_seed if args.permutation_seed is not None else args.random_state,
                n_jobs=args.n_jobs,
                logger=logger,
            )
            perm_json_path = output_dir / f"permutation_{name}.json"
            io_utils.write_json(perm_json_path, perm_summary)
            null_path = output_dir / f"permutation_{name}_null.npy"
            np.save(null_path, perm_null)
            entry["permutation_test"] = {
                "p_value": perm_summary["p_value"],
                "true_r2": perm_summary["true_r2"],
                "null_mean": perm_summary["null_mean"],
                "null_std": perm_summary["null_std"],
                "null_quantiles": perm_summary["null_quantiles"],
                "result_file": perm_json_path.name,
                "null_distribution_file": null_path.name,
            }

        summary_model_entries.append(entry)

    if not model_results:
        logger.error("No models were evaluated; aborting.")
        sys.exit(1)

    best_result = max(model_results, key=lambda res: res["summary_metrics"]["r2"])
    best_model_name = best_result["name"]
    builder, param_grid = model_configs[best_model_name]
    final_stratify = data["temp_celsius"] if "temp_celsius" in data.columns else None
    final_estimator, final_best_params, final_cv_score, final_cv_desc = cv_evaluation.fit_final_estimator(
        model_name=best_model_name,
        builder=builder,
        param_grid=param_grid,
        X=X,
        y=y,
        run_groups=run_groups,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        stratify_labels=final_stratify,
        logger=logger,
    )
    logger.info(
        "Best model %s refit on all trials using %s (diagnostic in-sample evaluation)",
        best_model_name,
        final_cv_desc,
    )
    final_predictions = final_estimator.predict(X)
    in_sample_metrics = metrics.compute_metrics(y.to_numpy(), final_predictions)

    final_pred_df = metrics.build_prediction_frame(
        data=data,
        y_true=y,
        y_pred=final_predictions,
        model_name=best_model_name,
        target_column=target_column,
        target_key=target.key,
    )
    final_pred_df["evaluation"] = "in_sample"
    final_pred_path = output_dir / f"final_model_predictions_{best_model_name}.tsv"
    final_pred_path.parent.mkdir(parents=True, exist_ok=True)
    final_pred_df.to_csv(final_pred_path, sep="\t", index=False)

    final_subj_metrics = metrics.compute_group_metrics(final_pred_df, ["subject"])
    final_subj_metrics["evaluation"] = "in_sample"
    final_subj_metrics_path = output_dir / f"final_per_subject_metrics_{best_model_name}.tsv"
    final_subj_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    final_subj_metrics.to_csv(final_subj_metrics_path, sep="\t", index=False)

    final_temp_metrics = metrics.compute_group_metrics(final_pred_df, ["temp_celsius"])
    final_temp_metrics["evaluation"] = "in_sample"
    final_temp_metrics_path = output_dir / f"final_per_temperature_metrics_{best_model_name}.tsv"
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
    joblib.dump(model_payload, output_dir / f"final_model_{best_model_name}.joblib")

    importance_path: Optional[Path] = None
    importance_df = model_builders.extract_feature_importance(final_estimator, feature_columns)
    if importance_df is not None:
        importance_path = output_dir / f"feature_importance_{best_model_name}.tsv"
        importance_path.parent.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(importance_path, sep="\t", index=False)

    best_entry = next((entry for entry in summary_model_entries if entry["name"] == best_model_name), None)
    if best_entry is None:
        logger.warning("Best model entry missing from summary list; reusing aggregate metrics.")
        best_entry = {
            "name": best_model_name,
            "metrics": best_result["summary_metrics"],
        }

    temperature_distribution = temp_counts.to_dict() if isinstance(temp_counts, pd.Series) else {}
    temporal_alignment_note = (
        "EEG features derived from the stimulation plateau window (see eeg_pipeline/eeg_config.yaml) and fMRI betas estimated with a canonical HRF at stimulus onset (~5-7 s peak)."
    )
    nonzero_coefficients = None
    reg_step = final_estimator.named_steps.get('reg') if isinstance(final_estimator, Pipeline) else None
    if reg_step is not None and hasattr(reg_step, 'coef_'):
        coef = np.asarray(reg_step.coef_)
        nonzero_coefficients = int(np.count_nonzero(coef))
        logger.info("Final %s non-zero coefficients: %d / %d", best_model_name, nonzero_coefficients, coef.size)

    temperature_baseline_entry = None
    if temperature_baseline_metrics is not None:
        temperature_baseline_entry = {
            "metrics": temperature_baseline_metrics,
            "cv_strategy": temperature_baseline_desc,
        }

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
        "temperature_only_baseline": temperature_baseline_entry,
        "outer_cv_level": outer_group_level,
        "outer_cv_strategy": outer_cv_strategy_record or "Not determined",
        "inner_cv_strategy": inner_cv_strategy_record or "Not determined",
        "include_temperature": bool(args.include_temperature),
        "models": summary_model_entries,
        "best_model": {
            "name": best_model_name,
            "cv_best_score": final_cv_score,
            "final_best_params": final_best_params,
            "final_metrics": in_sample_metrics,
            "final_metrics_label": "in-sample (resubstitution) diagnostics",
            "in_sample_metrics": in_sample_metrics,
            "in_sample_metrics_note": IN_SAMPLE_METRIC_NOTE,
            "prediction_file": final_pred_path.name,
            "per_subject_metrics_file": final_subj_metrics_path.name,
            "per_temperature_metrics_file": final_temp_metrics_path.name,
            "model_artifact_file": f"final_model_{best_model_name}.joblib",
            "feature_importance_file": importance_path.name if importance_path else None,
            "refit_cv_strategy": final_cv_desc,
            "nonzero_coefficients": nonzero_coefficients,
            "permutation_test": best_entry.get("permutation_test") if isinstance(best_entry, dict) else None,
        },
        "notes": [
            f"Target signature: {target.display_name} (column '{target_column}').",
            temporal_alignment_note,
            IN_SAMPLE_METRIC_NOTE,
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
    logger.info(
        "Best model: %s | CV R2=%.3f | In-sample R2=%.3f",
        best_model_name,
        final_cv_score,
        in_sample_metrics["r2"],
    )

    manifest_additional = {
        "dataset": {
            "n_trials": int(len(data)),
            "n_subjects": len(subjects),
            "bands": list(bands),
            "feature_count": len(feature_columns),
            "include_temperature": bool(args.include_temperature),
            "outer_groups": outer_group_level,
            "target_signature": target.key,
            "target_column": target_column,
            "target_display_name": target.display_name,
        },
        "temperature_baseline": temperature_baseline_entry,
        "models": summary_model_entries,
        "refit_in_sample_metrics_note": IN_SAMPLE_METRIC_NOTE,
    }
    io_utils.create_run_manifest(output_dir, args, repo_root, additional=manifest_additional)

    logger.info("All outputs written to %s", output_dir)


if __name__ == "__main__":
    main()

