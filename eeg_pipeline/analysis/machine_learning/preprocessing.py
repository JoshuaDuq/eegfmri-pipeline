"""
Shared preprocessing utilities for machine learning pipelines.

These transformers are designed to be CV-safe: they learn any column masks on the
training fold only (via ``fit``) and apply them to the corresponding test fold.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold as _SklearnVarianceThreshold

logger = logging.getLogger(__name__)


class VarianceThreshold(BaseEstimator, TransformerMixin):
    """VarianceThreshold that raises an actionable error when no features pass.

    When the threshold would remove all features (common with subject-level or
    within-subject CV and small train folds), re-raises with a clear message
    and suggested remedies instead of the generic sklearn error.
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = float(threshold)

    def fit(self, X, y=None):
        self._vt = _SklearnVarianceThreshold(threshold=self.threshold)
        try:
            self._vt.fit(X, y)
        except ValueError as e:
            if "No feature in X meets the variance threshold" in str(e):
                raise ValueError(
                    f"No feature has variance >= {self.threshold}. "
                    "This often happens with subject-level or within-subject CV when "
                    "train folds are small (e.g. one subject or few blocks). "
                    "Remedy: set variance_threshold=0.0 or variance_threshold_grid=[0.0] "
                    "in machine_learning.preprocessing, or use more subjects / --cv-scope group."
                ) from e
            raise
        return self

    def transform(self, X):
        return self._vt.transform(X)

    def get_support(self, indices: bool = False):
        return self._vt.get_support(indices=indices)

    def get_feature_names_out(self, input_features=None):
        return self._vt.get_feature_names_out(input_features=input_features)


class ReplaceInfWithNaN(BaseEstimator, TransformerMixin):
    """Replace +/-inf with NaN so imputers can handle them."""

    def __init__(self, copy: bool = True):
        self.copy = copy

    def fit(self, X, y=None):  # noqa: N803 (sklearn signature)
        return self

    def transform(self, X):  # noqa: N803 (sklearn signature)
        X_arr = np.asarray(X, dtype=float)
        if self.copy:
            X_arr = X_arr.copy()
        inf_mask = np.isinf(X_arr)
        if np.any(inf_mask):
            X_arr[inf_mask] = np.nan
        return X_arr


class DropAllNaNColumns(BaseEstimator, TransformerMixin):
    """Drop columns that are entirely NaN/inf in the training data.

    This is important when using "union_impute" feature harmonization where some
    families may be absent for entire subsets, and for preventing SimpleImputer
    from producing NaN statistics for all-missing columns.
    """

    def __init__(self, min_finite: int = 1):
        self.min_finite = int(min_finite)

    def fit(self, X, y=None):  # noqa: N803 (sklearn signature)
        X_arr = np.asarray(X, dtype=float)
        finite_counts = np.sum(np.isfinite(X_arr), axis=0)
        self.support_mask_ = finite_counts >= self.min_finite
        if not np.any(self.support_mask_):
            raise ValueError(
                "All feature columns are empty (all-NaN/inf) after filtering. "
                "Check feature extraction outputs and harmonization settings."
            )
        return self

    def transform(self, X):  # noqa: N803 (sklearn signature)
        X_arr = np.asarray(X, dtype=float)
        if not hasattr(self, "support_mask_"):
            raise RuntimeError("DropAllNaNColumns is not fitted yet.")
        return X_arr[:, self.support_mask_]

    def get_support(self, indices: bool = False):
        if not hasattr(self, "support_mask_"):
            raise RuntimeError("DropAllNaNColumns is not fitted yet.")
        if indices:
            return np.flatnonzero(self.support_mask_)
        return self.support_mask_

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> List[str]:
        if input_features is None:
            raise ValueError("input_features is required for get_feature_names_out.")
        support = self.get_support(indices=True)
        input_list = list(input_features)
        return [input_list[i] for i in support]


def transform_feature_names_through_steps(
    steps: Sequence[tuple[str, object]],
    feature_names: List[str],
) -> List[str]:
    """Transform feature names through sklearn pipeline steps.

    Supports common selectors via ``get_support`` and custom transformers via
    ``get_feature_names_out``. Steps that don't change dimensionality are ignored.
    """
    names = list(feature_names)
    for _name, step in steps:
        if hasattr(step, "get_feature_names_out"):
            try:
                names = list(step.get_feature_names_out(names))  # type: ignore[call-arg]
                continue
            except Exception as exc:
                logger.debug("Ignoring get_feature_names_out failure for step %r: %s", _name, exc)
        if hasattr(step, "get_support"):
            try:
                mask = step.get_support()  # type: ignore[call-arg]
                if mask is not None:
                    mask_arr = np.asarray(mask, dtype=bool)
                    if mask_arr.shape[0] == len(names):
                        names = [n for n, keep in zip(names, mask_arr) if bool(keep)]
                        continue
            except Exception as exc:
                logger.debug("Ignoring get_support failure for step %r: %s", _name, exc)
    return names


__all__ = [
    "ReplaceInfWithNaN",
    "DropAllNaNColumns",
    "VarianceThreshold",
    "transform_feature_names_through_steps",
]
