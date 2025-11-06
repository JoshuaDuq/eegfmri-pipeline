from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

###################################################################
# Parameter Grid Adjustments
###################################################################

def adjust_param_grid_for_high_dimensionality(
    model_name: str,
    param_grid: Dict[str, Sequence],
    *,
    high_dimensionality: bool,
    logger,
) -> Dict[str, Sequence]:
    if not param_grid:
        return {}

    adjusted: Dict[str, List[Any]] = {key: list(values) for key, values in param_grid.items()}
    if not high_dimensionality:
        return adjusted

    modifications: List[str] = []
    if model_name == "elasticnet":
        alpha_values = adjusted.setdefault("reg__alpha", [])
        extra_alphas = [30.0, 100.0]
        added_alphas = [val for val in extra_alphas if val not in alpha_values]
        if added_alphas:
            alpha_values.extend(added_alphas)
            modifications.append(f"added reg__alpha={added_alphas}")

        l1_values = adjusted.setdefault("reg__l1_ratio", [])
        if 0.95 not in l1_values:
            l1_values.append(0.95)
            modifications.append("added reg__l1_ratio=0.95")

    elif model_name == "random_forest":
        max_features = adjusted.setdefault("rf__max_features", [])
        if 0.1 not in max_features:
            max_features.append(0.1)
            modifications.append("added rf__max_features=0.1")

        leaf_values = adjusted.setdefault("rf__min_samples_leaf", [])
        added_leaves: List[int] = []
        for val in (6, 10):
            if val not in leaf_values:
                leaf_values.append(val)
                added_leaves.append(val)
        if added_leaves:
            modifications.append(
                "expanded rf__min_samples_leaf to include stronger regularisation levels"
            )

    if modifications:
        logger.info(
            "High feature-to-sample mitigation for %s: %s",
            model_name,
            "; ".join(modifications),
        )

    return adjusted


###################################################################
# Feature Importance Extraction
###################################################################

def extract_feature_importance(model: Pipeline, feature_names: Sequence[str]) -> Optional[pd.DataFrame]:
    if "reg" in model.named_steps:
        reg = model.named_steps["reg"]
        coef = getattr(reg, "coef_", None)
        if coef is None:
            return None
        coef = np.asarray(coef)
        scaler = model.named_steps.get("scaler")
        if scaler is not None and hasattr(scaler, "scale_"):
            scale = np.asarray(scaler.scale_)
            scale[scale == 0] = 1.0
            coef = coef / scale
        df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": coef,
                "importance_abs": np.abs(coef),
            }
        )
        return df.sort_values("importance_abs", ascending=False)
    if "rf" in model.named_steps:
        rf = model.named_steps["rf"]
        importances = np.asarray(rf.feature_importances_)
        df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
                "importance_abs": np.abs(importances),
            }
        )
        return df.sort_values("importance_abs", ascending=False)
    return None


###################################################################
# Baseline Model Builders
###################################################################

def build_elasticnet(random_state: int, _: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("scaler", StandardScaler()),
            ("reg", ElasticNet(max_iter=5000, random_state=random_state)),
        ]
    )


ELASTICNET_PARAM_GRID = {
    "reg__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
    "reg__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
}


def build_random_forest(random_state: int, _estimator_jobs: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=600,
                    random_state=random_state,
                    n_jobs=1,
                    bootstrap=True,
                ),
            ),
        ]
    )


RANDOM_FOREST_PARAM_GRID = {
    "rf__max_depth": [None, 12, 20],
    "rf__max_features": ["sqrt", "log2", 0.3],
    "rf__min_samples_leaf": [1, 2, 4],
}


MODEL_REGISTRY = {
    "elasticnet": (build_elasticnet, ELASTICNET_PARAM_GRID),
    "random_forest": (build_random_forest, RANDOM_FOREST_PARAM_GRID),
}


###################################################################
# SVM Model Builder
###################################################################

def make_svm_builder(cache_size: float):
    def builder(random_state: int, _n_jobs: int) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", VarianceThreshold()),
                ("scaler", StandardScaler()),
                ("svm", SVR(kernel="rbf", cache_size=cache_size)),
            ]
        )

    return builder

