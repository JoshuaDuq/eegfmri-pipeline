from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .cv_evaluation import nested_cv_evaluate

###################################################################
# Permutation Testing
###################################################################

def permutation_test_r2(
    *,
    model_name: str,
    builder,
    param_grid: Dict[str, Sequence],
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: Sequence[str],
    meta: pd.DataFrame,
    outer_groups: np.ndarray,
    run_groups: np.ndarray,
    n_permutations: int,
    true_r2: float,
    random_state: Optional[int],
    n_jobs: int,
    fit_params_fn: Optional[Callable[[pd.DataFrame], Dict[str, Any]]] = None,
    logger: Any,
) -> tuple[Dict[str, Any], np.ndarray]:
    if n_permutations <= 0:
        raise ValueError("n_permutations must be a positive integer")

    if not isinstance(y, pd.Series):
        y_series = pd.Series(np.asarray(y), name="target")
    else:
        y_series = y.copy()

    rng = np.random.default_rng(random_state)
    null_scores = np.zeros(n_permutations, dtype=float)
    seed_display = random_state if random_state is not None else "random"
    logger.info(
        "Running permutation test with %d permutations for model %s (nested CV retraining, seed=%s)",
        n_permutations,
        model_name,
        seed_display,
    )

    log_check_interval = max(1, n_permutations // 10)

    if outer_groups is not None:
        perm_groups = np.asarray(outer_groups)
        group_source = "outer_groups"
    elif run_groups is not None:
        perm_groups = np.asarray(run_groups)
        group_source = "run_groups"
    else:
        raise ValueError(
            "Permutation testing requires group labels (outer or run) to preserve dependency structure."
        )
    if perm_groups.size == 0:
        raise ValueError("Permutation groups array is empty; cannot perform grouped permutations.")

    for idx in range(n_permutations):
        values = y_series.to_numpy(copy=True)
        if perm_groups is not None:
            for group in np.unique(perm_groups):
                group_idx = np.where(perm_groups == group)[0]
                if len(group_idx) > 1:
                    values[group_idx] = values[group_idx][rng.permutation(len(group_idx))]
        else:
            values = rng.permutation(values)
        permuted = pd.Series(values, index=y_series.index, name=y_series.name)
        perm_result = nested_cv_evaluate(
            model_name=model_name,
            builder=builder,
            param_grid=param_grid,
            X=X,
            y=permuted,
            feature_names=feature_names,
            meta=meta,
            outer_groups=outer_groups,
            run_groups=run_groups,
            random_state=random_state,
            n_jobs=n_jobs,
            logger=logger,
            log_progress=False,
            fit_params_fn=fit_params_fn,
        )
        null_scores[idx] = perm_result["summary_metrics"]["r2"]
        if (idx + 1) % log_check_interval == 0 or (idx + 1) == n_permutations:
            logger.info(
                "Permutation progress: %d/%d (null mean R2=%.4f)",
                idx + 1,
                n_permutations,
                float(np.mean(null_scores[: idx + 1])),
            )

    p_value = (np.sum(null_scores >= true_r2) + 1) / (n_permutations + 1)
    summary = {
        "true_r2": float(true_r2),
        "p_value": float(p_value),
        "null_mean": float(np.mean(null_scores)),
        "null_std": float(np.std(null_scores)),
        "permutation_group_field": group_source,
        "null_quantiles": {
            "05": float(np.quantile(null_scores, 0.05)),
            "50": float(np.quantile(null_scores, 0.5)),
            "95": float(np.quantile(null_scores, 0.95)),
        },
        "n_permutations": int(n_permutations),
        "random_state": random_state,
    }
    return summary, null_scores

