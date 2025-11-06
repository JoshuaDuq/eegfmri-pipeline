import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..utils import (
    config_loader,
    cv_evaluation,
    data_loading,
    feature_utils,
    io_utils,
    metrics,
    permutation,
    target_signatures,
)

###################################################################
# Constants
###################################################################

MODEL_NAME = "covariance"


###################################################################
# Model-Specific Transformers
###################################################################

class TangentFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        *,
        channel_names: Sequence[str],
        band_names: Sequence[str],
        channel_band_indices: Dict[str, List[int]],
        global_feature_indices: Sequence[int],
        metric: str = "riemann",
        epsilon: float = 1e-6,
    ) -> None:
        self.channel_names = list(channel_names)
        self.band_names = list(band_names)
        self.channel_band_indices = channel_band_indices
        self.global_feature_indices = list(global_feature_indices)
        self.metric = metric
        self.epsilon = float(epsilon)

    def _compute_covariances(self, X: np.ndarray) -> np.ndarray:
        n_channels = len(self.channel_names)
        n_bands = len(self.band_names)
        covs = np.zeros((len(X), n_channels, n_channels), dtype=np.float64)
        for row_idx, row in enumerate(X):
            mat = np.zeros((n_channels, n_bands), dtype=np.float64)
            for ch_idx, channel in enumerate(self.channel_names):
                indices = self.channel_band_indices[channel]
                mat[ch_idx, :] = row[indices]
            cov = mat @ mat.T
            if self.epsilon > 0:
                cov += self.epsilon * np.eye(n_channels)
            covs[row_idx] = cov
        return covs

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        covs = self._compute_covariances(np.asarray(X, dtype=float))
        self._tangent = TangentSpace(metric=self.metric)
        self._tangent.fit(covs, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        covs = self._compute_covariances(X_arr)
        tangent = self._tangent.transform(covs)
        if self.global_feature_indices:
            globals_ = X_arr[:, self.global_feature_indices]
            return np.hstack([tangent, globals_.astype(float)])
        return tangent


###################################################################
# Helper Functions
###################################################################

def _normalize_positive(values: Sequence[float], *, strictly: bool = True) -> List[float]:
    unique: List[float] = []
    for val in values:
        cast = float(val)
        if strictly and cast <= 0:
            raise ValueError("All grid values must be positive.")
        if not strictly and cast < 0:
            raise ValueError("All grid values must be non-negative.")
        if cast not in unique:
            unique.append(cast)
    if not unique:
        raise ValueError("Grid must contain at least one value.")
    return unique


def make_covariance_builder(
    *,
    model_name: str,
    tangent_params: Dict[str, Any],
    channel_names: List[str],
    band_names: List[str],
    channel_band_indices: Dict[str, List[int]],
    global_feature_indices: List[int],
) -> Callable:
    def builder(random_state: int, _n_jobs: int) -> Pipeline:
        tangent = TangentFeatureExtractor(
            channel_names=channel_names,
            band_names=band_names,
            channel_band_indices=channel_band_indices,
            global_feature_indices=global_feature_indices,
            metric=tangent_params["metric"],
            epsilon=tangent_params["epsilon"],
        )
        if model_name == "elasticnet":
            reg = ElasticNet(max_iter=5000, random_state=random_state)
        elif model_name == "ridge":
            reg = Ridge()
        else:
            raise ValueError(f"Unsupported model {model_name}")
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("tangent", tangent),
                ("variance", VarianceThreshold()),
                ("scaler", StandardScaler()),
                ("reg", reg),
            ]
        )

    return builder


###################################################################
# Argument Parsing
###################################################################

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict fMRI signature beta responses (e.g., NPS, SIIPS1) from EEG power using covariance + tangent-space regressors."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for outputs. Defaults to machine_learning/outputs/eeg_to_<target>_covariance_<timestamp>.",
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
        help="Include stimulus temperature as an additional tangent-space covariate.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["elasticnet", "ridge"],
        help="Models to evaluate (elasticnet, ridge).",
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
        help="Random seed for cross-validation reproducibility.",
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
        help="Number of label permutations for R^2 significance testing (0 disables).",
    )
    parser.add_argument(
        "--elasticnet-alpha-grid",
        nargs="*",
        type=float,
        default=[0.001, 0.01, 0.1, 1.0],
        help="ElasticNet alpha grid.",
    )
    parser.add_argument(
        "--elasticnet-l1-grid",
        nargs="*",
        type=float,
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help="ElasticNet l1_ratio grid.",
    )
    parser.add_argument(
        "--ridge-alpha-grid",
        nargs="*",
        type=float,
        default=[0.1, 1.0, 10.0, 100.0],
        help="Ridge alpha grid.",
    )
    parser.add_argument(
        "--tangent-metric",
        type=str,
        default="riemann",
        help="Metric for pyriemann TangentSpace (riemann, logeuclid, etc).",
    )
    parser.add_argument(
        "--cov-epsilon",
        type=float,
        default=1e-6,
        help="Diagonal regularisation added to covariance matrices for SPD stability.",
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
        
        cov_config = config_loader.get_model_config(config, "covariance")
        if cov_config:
            if not args.models:
                args.models = cov_config.get("models", ["elasticnet", "ridge"])
            
            tangent_config = cov_config.get("tangent_space", {})
            if not hasattr(args, "tangent_metric") or args.tangent_metric == "riemann":
                args.tangent_metric = tangent_config.get("metric", "riemann")
            if not hasattr(args, "cov_epsilon") or args.cov_epsilon == 1e-6:
                args.cov_epsilon = tangent_config.get("epsilon", 1e-6)
            
            if not hasattr(args, "elasticnet_alpha_grid") or not args.elasticnet_alpha_grid:
                args.elasticnet_alpha_grid = cov_config.get("elasticnet", {}).get("param_grid", {}).get("alpha", [0.001, 0.01, 0.1, 1.0])
            if not hasattr(args, "elasticnet_l1_grid") or not args.elasticnet_l1_grid:
                args.elasticnet_l1_grid = cov_config.get("elasticnet", {}).get("param_grid", {}).get("l1_ratio", [0.1, 0.3, 0.5, 0.7, 0.9])
            if not hasattr(args, "ridge_alpha_grid") or not args.ridge_alpha_grid:
                args.ridge_alpha_grid = cov_config.get("ridge", {}).get("param_grid", {}).get("alpha", [0.1, 1.0, 10.0, 100.0])
    
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

    elasticnet_alpha_grid = _normalize_positive(args.elasticnet_alpha_grid)
    elasticnet_l1_grid = _normalize_positive(args.elasticnet_l1_grid, strictly=False)
    ridge_alpha_grid = _normalize_positive(args.ridge_alpha_grid)

    repo_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = (
        Path(__file__).resolve().parent.parent / "outputs" / f"eeg_to_{target.key}_covariance_{timestamp}"
    )
    output_dir = Path(args.output_dir) if args.output_dir else default_output
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = io_utils.setup_logging(output_dir, logger_name=f"eeg_to_{target.key}_covariance")
    logger.info("EEG -> %s covariance/tangent-space pipeline started.", target.display_name)
    logger.info("Using bands: %s", ", ".join(bands))
    logger.info(
        "ElasticNet grid | alpha: %s | l1_ratio: %s | Ridge alpha grid: %s",
        elasticnet_alpha_grid,
        elasticnet_l1_grid,
        ridge_alpha_grid,
    )

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

    channel_feature_columns = feature_utils.select_direct_power_columns(master_features, bands)
    if not channel_feature_columns:
        raise ValueError("No direct power features found for the requested bands; cannot build covariance matrices.")

    channel_names = sorted({col.split("_", 2)[-1] for col in channel_feature_columns})
    expected_columns = [f"pow_{band}_{channel}" for channel in channel_names for band in bands]
    missing_columns = [col for col in expected_columns if col not in channel_feature_columns]
    if missing_columns:
        preview = ", ".join(sorted(missing_columns)[:10])
        if len(missing_columns) > 10:
            preview += ", ..."
        raise ValueError(
            "Missing required power feature columns for covariance modelling: %s. "
            "Regenerate EEG features or adjust --bands to match available columns." % preview
        )

    global_feature_columns: List[str] = []
    if args.include_temperature:
        global_feature_columns.append("temp_celsius")

    feature_columns = channel_feature_columns + global_feature_columns

    data = pd.concat([res.data for res in subject_results], ignore_index=True)
    missing_globals = [col for col in global_feature_columns if col not in data.columns]
    if missing_globals:
        raise ValueError(f"Requested global covariates missing from dataset: {missing_globals}")

    channel_feature_columns = [col for col in channel_feature_columns if col in feature_columns]
    global_feature_columns = [col for col in global_feature_columns if col in feature_columns]
    if not channel_feature_columns:
        raise ValueError("All channel features were removed due to zero variance; cannot build covariance matrices.")
    channel_names = sorted({col.split("_", 2)[-1] for col in channel_feature_columns})

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
        "Assembled dataset: %d trials, %d subjects, %d features (channels=%d, globals=%d).",
        len(data),
        len(subjects),
        len(feature_columns),
        len(channel_feature_columns),
        len(global_feature_columns),
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
        "Covariance construction | channels=%d | bands=%s | tangent metric=%s | epsilon=%.2e",
        len(channel_names),
        ", ".join(bands),
        args.tangent_metric,
        args.cov_epsilon,
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
        "Temperature covariate included as predictor alongside EEG-derived covariance features."
        if args.include_temperature and "temp_celsius" in feature_columns
        else "Temperature covariate excluded from predictors; per-temperature diagnostics reflect distributional balance only."
    )
    logger.info(temperature_predictor_note)

    channel_band_indices: Dict[str, List[int]] = {
        channel: [feature_columns.index(f"pow_{band}_{channel}") for band in bands]
        for channel in channel_names
    }

    global_indices = [feature_columns.index(col) for col in global_feature_columns]

    tangent_params = {"metric": args.tangent_metric, "epsilon": args.cov_epsilon}

    model_names = [name.lower() for name in args.models]
    allowed_models = {"elasticnet", "ridge"}
    for name in model_names:
        if name not in allowed_models:
            raise ValueError(f"Unsupported model '{name}'. Allowed: {', '.join(sorted(allowed_models))}.")

    builder_cache: Dict[str, Callable] = {}
    param_grids: Dict[str, Dict[str, Sequence[Any]]] = {}

    for name in model_names:
        builder_cache[name] = make_covariance_builder(
            model_name=name,
            tangent_params=tangent_params,
            channel_names=channel_names,
            band_names=list(bands),
            channel_band_indices=channel_band_indices,
            global_feature_indices=global_indices,
        )
        if name == "elasticnet":
            param_grids[name] = {
                "reg__alpha": elasticnet_alpha_grid,
                "reg__l1_ratio": elasticnet_l1_grid,
            }
        elif name == "ridge":
            param_grids[name] = {
                "reg__alpha": ridge_alpha_grid,
            }

    results: List[Dict[str, Any]] = []
    summary_entries: List[Dict[str, Any]] = []
    outer_cv_desc: Optional[str] = None
    inner_cv_desc: Optional[str] = None

    for name in model_names:
        logger.info("Starting nested CV for %s.", name)
        result = cv_evaluation.nested_cv_evaluate(
            model_name=name,
            builder=builder_cache[name],
            param_grid=param_grids[name],
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
        results.append(result)

        if outer_cv_desc is None:
            outer_cv_desc = result.get("outer_cv_desc")
        if inner_cv_desc is None:
            inner_cv_desc = result.get("inner_cv_desc")

        pred_df = metrics.build_prediction_frame(
            data=data,
            y_true=y,
            y_pred=result["predictions"],
            model_name=name,
            target_column=target_column,
            target_key=target.key,
            fold_assignments=result["fold_assignments"],
        )
        pred_path = output_dir / f"predictions_{MODEL_NAME}_{name}.tsv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(pred_path, sep="\t", index=False)

        subj_metrics = metrics.compute_group_metrics(pred_df, ["subject"])
        subj_metrics_path = output_dir / f"per_subject_metrics_{MODEL_NAME}_{name}.tsv"
        subj_metrics.to_csv(subj_metrics_path, sep="\t", index=False)

        temp_metrics = metrics.compute_group_metrics(pred_df, ["temp_celsius"])
        temp_metrics_path = output_dir / f"per_temperature_metrics_{MODEL_NAME}_{name}.tsv"
        temp_metrics.to_csv(temp_metrics_path, sep="\t", index=False)

        fold_path = None
        fold_df = pd.DataFrame(result["fold_details"])
        if not fold_df.empty:
            fold_df["best_params"] = fold_df["best_params"].apply(lambda d: json.dumps(d))
            if "test_temp_counts" in fold_df.columns:
                fold_df["test_temp_counts"] = fold_df["test_temp_counts"].apply(lambda d: json.dumps(d))
            fold_path = output_dir / f"cv_folds_{MODEL_NAME}_{name}.tsv"
            fold_df.to_csv(fold_path, sep="\t", index=False)

        metrics_path = output_dir / f"metrics_{MODEL_NAME}_{name}.json"
        io_utils.write_json(metrics_path, result["summary_metrics"])

        best_params_path = output_dir / f"best_params_{MODEL_NAME}_{name}.json"
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
            "param_grid": param_grids[name],
            "target_signature": target.key,
            "target_column": target_column,
        }

        if args.permutation_count > 0:
            perm_summary, perm_null = permutation.permutation_test_r2(
                model_name=name,
                builder=builder_cache[name],
                param_grid=param_grids[name],
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
            perm_json_path = output_dir / f"permutation_{MODEL_NAME}_{name}.json"
            io_utils.write_json(perm_json_path, perm_summary)
            null_path = output_dir / f"permutation_{MODEL_NAME}_{name}_null.npy"
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

        summary_entries.append(entry)

    if not results:
        logger.error("No models were evaluated; aborting.")
        raise SystemExit(1)

    best_result = max(results, key=lambda res: res["summary_metrics"]["r2"])
    best_model_name = best_result["name"]
    builder = builder_cache[best_model_name]
    param_grid = param_grids[best_model_name]
    final_stratify = data["temp_celsius"] if "temp_celsius" in data.columns else None
    final_estimator, final_best_params, final_cv_score, final_cv_desc = cv_evaluation.fit_final_estimator(
        model_name=f"{MODEL_NAME}_{best_model_name}",
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
    logger.info("Covariance pipeline refit using %s", final_cv_desc)
    final_predictions = final_estimator.predict(X)
    final_metrics = metrics.compute_metrics(y.to_numpy(), final_predictions)

    final_pred_df = metrics.build_prediction_frame(
        data=data,
        y_true=y,
        y_pred=final_predictions,
        model_name=f"{MODEL_NAME}_{best_model_name}",
        target_column=target_column,
        target_key=target.key,
    )
    final_pred_path = output_dir / f"final_model_predictions_{MODEL_NAME}_{best_model_name}.tsv"
    final_pred_df.to_csv(final_pred_path, sep="\t", index=False)

    final_subj_metrics = metrics.compute_group_metrics(final_pred_df, ["subject"])
    final_subj_metrics_path = output_dir / f"final_per_subject_metrics_{MODEL_NAME}_{best_model_name}.tsv"
    final_subj_metrics.to_csv(final_subj_metrics_path, sep="\t", index=False)

    final_temp_metrics = metrics.compute_group_metrics(final_pred_df, ["temp_celsius"])
    final_temp_metrics_path = output_dir / f"final_per_temperature_metrics_{MODEL_NAME}_{best_model_name}.tsv"
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
        "tangent": {
            "metric": args.tangent_metric,
            "epsilon": args.cov_epsilon,
            "channel_names": channel_names,
            "band_names": list(bands),
            "global_features": global_feature_columns,
        },
    }
    joblib.dump(model_payload, output_dir / f"final_model_{MODEL_NAME}_{best_model_name}.joblib")

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
        "outer_cv_strategy": outer_cv_desc,
        "inner_cv_strategy": inner_cv_desc,
        "include_temperature": bool(args.include_temperature),
        "tangent_transform": {
            "metric": args.tangent_metric,
            "epsilon": args.cov_epsilon,
            "channels": channel_names,
            "bands": list(bands),
        },
        "models": summary_entries,
        "best_model": {
            "name": best_model_name,
            "cv_best_score": final_cv_score,
            "final_best_params": final_best_params,
            "final_metrics": final_metrics,
            "prediction_file": final_pred_path.name,
            "per_subject_metrics_file": final_subj_metrics_path.name,
            "per_temperature_metrics_file": final_temp_metrics_path.name,
            "model_artifact_file": f"final_model_{MODEL_NAME}_{best_model_name}.joblib",
            "refit_cv_strategy": final_cv_desc,
            "permutation_test": next((entry.get("permutation_test") for entry in summary_entries if entry["name"] == best_model_name), None),
        },
        "notes": [
            f"Target signature: {target.display_name} (column '{target_column}').",
            "Covariance matrices are constructed from band-limited sensor powers and projected to the tangent space of the SPD manifold.",
            "Global covariates (e.g., temperature) are appended after tangent-space transformation when requested.",
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
    logger.info("Covariance pipeline final R2=%.3f", final_metrics["r2"])
    logger.info("All outputs written to %s", output_dir)


if __name__ == "__main__":
    main()

