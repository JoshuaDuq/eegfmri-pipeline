from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

###################################################################
# Metrics Computation
###################################################################

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
    }
    try:
        metrics["pearson_r"] = float(pearsonr(y_true, y_pred)[0])
    except Exception:
        metrics["pearson_r"] = float("nan")
    try:
        metrics["spearman_r"] = float(spearmanr(y_true, y_pred)[0])
    except Exception:
        metrics["spearman_r"] = float("nan")

    try:
        slope, intercept, r_value, p_value, std_err = linregress(y_pred, y_true)
        metrics.update(
            {
                "calibration_slope": float(slope),
                "calibration_intercept": float(intercept),
                "calibration_r": float(r_value),
                "calibration_p": float(p_value),
                "calibration_std_err": float(std_err),
            }
        )
    except Exception:
        metrics.update(
            {
                "calibration_slope": float("nan"),
                "calibration_intercept": float("nan"),
                "calibration_r": float("nan"),
                "calibration_p": float("nan"),
                "calibration_std_err": float("nan"),
            }
        )

    return metrics


def compute_group_metrics(pred_df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if not group_cols:
        raise ValueError('group_cols must contain at least one column name')
    true_col = "target_true" if "target_true" in pred_df.columns else "br_true"
    pred_col = "target_pred" if "target_pred" in pred_df.columns else "br_pred"
    rows: List[Dict[str, float]] = []
    for keys, grp in pred_df.groupby(list(group_cols), dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        metrics = compute_metrics(grp[true_col].to_numpy(), grp[pred_col].to_numpy())
        row = {col: key_tuple[idx] for idx, col in enumerate(group_cols)}
        row['n_trials'] = int(len(grp))
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(group_cols)).reset_index(drop=True)


###################################################################
# Prediction Frame Construction
###################################################################

def build_prediction_frame(
    data: pd.DataFrame,
    y_true,
    y_pred: np.ndarray,
    *,
    model_name: str,
    target_column: str,
    target_key: str,
    fold_assignments: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    frame = data[
        ["subject", "run", "trial_idx_run", "trial_idx_global", "temp_celsius", "vas_rating", "pain_binary"]
    ].copy()
    values = y_true.to_numpy() if hasattr(y_true, 'to_numpy') else np.asarray(y_true)
    pred_values = np.asarray(y_pred, dtype=float)
    frame["target_true"] = values
    frame["target_pred"] = pred_values
    frame[f"{target_column}_true"] = values
    frame[f"{target_column}_pred"] = pred_values
    frame["br_true"] = values
    frame["br_pred"] = pred_values
    frame["target_key"] = target_key
    frame["model"] = model_name
    if fold_assignments is not None:
        frame["cv_fold"] = fold_assignments
    return frame


###################################################################
# Temperature Baseline
###################################################################

def compute_temperature_baseline_cv(
    temp,
    target,
    outer_groups: np.ndarray,
    random_state: int,
    logger,
) -> tuple[Dict[str, float], str]:
    from .cv_utils import (
        compute_cv_splits,
        enforce_temperature_coverage,
        make_outer_splitter,
        prepare_temperature_metadata,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    if temp.isna().all() if hasattr(temp, 'isna') else np.isnan(temp).all():
        raise ValueError("Temperature series contains only NaN values; cannot compute baseline")

    baseline_X = temp.to_numpy().reshape(-1, 1) if hasattr(temp, 'to_numpy') else np.asarray(temp).reshape(-1, 1)
    baseline_y = target.to_numpy() if hasattr(target, 'to_numpy') else np.asarray(target)

    stratify_labels, temp_levels = prepare_temperature_metadata(temp)
    splitter, groups_used, desc = make_outer_splitter(
        outer_groups,
        random_state,
        stratify_labels=stratify_labels,
        temperature_levels=temp_levels,
        logger=logger,
    )
    predictions = np.zeros_like(baseline_y, dtype=float)

    splits = compute_cv_splits(
        splitter,
        baseline_X,
        baseline_y,
        groups=groups_used,
        stratify_labels=stratify_labels,
    )
    if stratify_labels is not None and len(temp_levels) > 1:
        enforce_temperature_coverage(
            stratify_labels,
            splits,
            temp_levels,
            context="Temperature baseline outer CV",
            check_train=True,
        )

    for train_idx, test_idx in splits:
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("reg", LinearRegression()),
            ]
        )
        model.fit(baseline_X[train_idx], baseline_y[train_idx])
        predictions[test_idx] = model.predict(baseline_X[test_idx])

    metrics = compute_metrics(baseline_y, predictions)
    logger.info(
        "Temperature-only baseline via %s | R2=%.3f | MAE=%.3f | RMSE=%.3f",
        desc,
        metrics["r2"],
        metrics["mae"],
        metrics["rmse"],
    )
    return metrics, desc

