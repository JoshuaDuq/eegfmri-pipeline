from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    StratifiedKFold,
)

###################################################################
# Temperature Metadata Preparation
###################################################################

def prepare_temperature_metadata(values: Optional[Sequence[Any]]) -> Tuple[Optional[np.ndarray], np.ndarray]:
    if values is None:
        return None, np.array([], dtype=object)
    series = pd.Series(values, copy=True)
    if series.empty:
        return None, np.array([], dtype=object)
    labels = series.to_numpy(dtype=object)
    valid_mask = ~series.isna()
    if not valid_mask.any():
        return labels, np.array([], dtype=object)
    levels = series.loc[valid_mask].astype(object).unique()
    levels = np.array(sorted(levels, key=lambda v: str(v)), dtype=object)
    return labels, levels


###################################################################
# Temperature Coverage Enforcement
###################################################################

def enforce_temperature_coverage(
    stratify_labels: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    required_levels: Sequence[str],
    *,
    context: str,
    check_train: bool = True,
    check_validation: bool = True,
    logger: Optional[Any] = None,
) -> None:
    if stratify_labels is None or len(stratify_labels) == 0:
        return
    if required_levels is None:
        required_levels = []
    required = [level for level in required_levels if not pd.isna(level)]
    if not required:
        return
    required_set = set(required)
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        if check_train:
            train_levels = {
                label for label in np.asarray(stratify_labels, dtype=object)[train_idx] if not pd.isna(label)
            }
            missing_train = sorted(required_set - train_levels)
            if missing_train:
                raise ValueError(
                    f"{context} fold {fold_idx} training partition is missing temperatures {missing_train}."
                )
        test_levels = {
            label for label in np.asarray(stratify_labels, dtype=object)[test_idx] if not pd.isna(label)
        }
        missing_test = sorted(required_set - test_levels)
        if missing_test:
            message = (
                f"{context} fold {fold_idx} validation partition is missing temperatures {missing_test}."
            )
            if check_validation:
                raise ValueError(message)
            if logger is not None:
                logger.warning(message)


###################################################################
# CV Split Computation
###################################################################

def compute_cv_splits(
    splitter,
    X,
    y,
    *,
    groups: Optional[np.ndarray] = None,
    stratify_labels: Optional[np.ndarray] = None,
):
    if isinstance(splitter, StratifiedGroupKFold):
        if groups is None:
            raise ValueError("StratifiedGroupKFold requires group labels.")
        if stratify_labels is None:
            raise ValueError("StratifiedGroupKFold requires stratify_labels.")
        return list(splitter.split(X, stratify_labels, groups=groups))
    if isinstance(splitter, (GroupKFold, LeaveOneGroupOut)):
        if groups is None:
            raise ValueError("Group-based splitter requires group labels.")
        return list(splitter.split(X, y, groups=groups))
    if isinstance(splitter, StratifiedKFold):
        target = stratify_labels if stratify_labels is not None else y
        return list(splitter.split(X, target))
    return list(splitter.split(X, y))


###################################################################
# CV Splitter Builders with Temperature Coverage
###################################################################

def _build_groupkfold_with_temperature_coverage(
    X,
    y,
    groups: np.ndarray,
    temp_labels: np.ndarray,
    temp_levels: Sequence[Any],
    *,
    random_state: int,
    context: str,
    logger: Optional[Any] = None,
) -> Tuple[GroupKFold, List[Tuple[np.ndarray, np.ndarray]], str]:
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError(f"{context}: require at least two groups for GroupKFold fallback.")

    max_splits = min(5, unique_groups.size)
    last_error: Optional[Exception] = None
    for n_splits in range(max_splits, 1, -1):
        splitter = GroupKFold(n_splits=n_splits)
        splits = compute_cv_splits(splitter, X, y, groups=groups)
        try:
            enforce_temperature_coverage(
                temp_labels,
                splits,
                temp_levels,
                context=context,
                check_train=True,
                check_validation=True,
                logger=logger,
            )
        except ValueError as err:
            last_error = err
            if logger is not None:
                logger.debug(
                    "%s | GroupKFold(n_splits=%d) insufficient temperature coverage: %s",
                    context,
                    n_splits,
                    err,
                )
            continue
        desc = (
            f"GroupKFold(n_splits={n_splits}) on run labels (temperature-balanced fallback)"
        )
        return splitter, splits, desc

    last_msg = f"{last_error}" if last_error is not None else "coverage check failed"
    raise ValueError(
        f"{context}: unable to satisfy temperature coverage with GroupKFold fallback ({last_msg})."
    )


def _build_kfold_with_temperature_coverage(
    X,
    y,
    stratify_labels: np.ndarray,
    temp_levels: Sequence[Any],
    *,
    random_state: int,
    context: str,
    logger: Optional[Any] = None,
) -> Tuple[KFold, List[Tuple[np.ndarray, np.ndarray]], str]:
    n_samples = len(X)
    if n_samples < 2:
        raise ValueError(f"{context}: require >=2 samples for KFold fallback.")

    max_splits = min(5, n_samples)
    last_error: Optional[Exception] = None
    for n_splits in range(max_splits, 1, -1):
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = compute_cv_splits(splitter, X, y)
        try:
            enforce_temperature_coverage(
                stratify_labels,
                splits,
                temp_levels,
                context=context,
                check_train=True,
                check_validation=True,
                logger=logger,
            )
        except ValueError as err:
            last_error = err
            if logger is not None:
                logger.debug(
                    "%s | KFold(n_splits=%d) insufficient temperature coverage: %s",
                    context,
                    n_splits,
                    err,
                )
            continue
        desc = (
            f"KFold(n_splits={n_splits}, shuffle=True, random_state={random_state}) on pooled trials (temperature-balanced fallback)"
        )
        return splitter, splits, desc

    last_msg = f"{last_error}" if last_error is not None else "coverage check failed"
    raise ValueError(
        f"{context}: unable to satisfy temperature coverage with KFold fallback ({last_msg})."
    )


###################################################################
# Outer and Inner CV Splitter Creation
###################################################################

def make_outer_splitter(
    groups: np.ndarray,
    random_state: int,
    stratify_labels: Optional[np.ndarray] = None,
    temperature_levels: Optional[Sequence[Any]] = None,
    logger: Optional[Any] = None,
):
    unique_groups = np.unique(groups)
    levels = list(temperature_levels) if temperature_levels is not None else []
    levels = [level for level in levels if not pd.isna(level)]

    if stratify_labels is not None and len(stratify_labels) != len(groups):
        raise ValueError(
            "Length of stratify_labels must match number of samples in groups for make_outer_splitter."
        )

    if len(unique_groups) >= 2:
        if logger is not None and stratify_labels is not None and levels:
            for group in unique_groups:
                mask = groups == group
                group_levels = {
                    label
                    for label in np.asarray(stratify_labels, dtype=object)[mask]
                    if not pd.isna(label)
                }
                missing = sorted(set(levels) - group_levels)
                if missing:
                    logger.warning(
                        "Outer CV group %s is missing temperature levels %s; fold performance may be hard to interpret.",
                        group,
                        missing,
                    )
        desc = f"LeaveOneGroupOut (n_groups={len(unique_groups)})"
        return LeaveOneGroupOut(), groups, desc

    raise ValueError(
        "Outer CV requires at least two distinct outer groups; provide additional runs/subjects to enable group-held-out evaluation."
    )


def make_inner_cv(
    run_groups: np.ndarray,
    random_state: int,
    stratify_labels: Optional[np.ndarray] = None,
    temperature_levels: Optional[Sequence[Any]] = None,
    logger: Optional[Any] = None,
):
    levels = list(temperature_levels) if temperature_levels is not None else []
    levels = [level for level in levels if not pd.isna(level)]

    if stratify_labels is not None and len(stratify_labels) != len(run_groups):
        raise ValueError(
            "Length of stratify_labels must match number of samples in run_groups for make_inner_cv."
        )

    unique_runs = np.unique(run_groups)
    if len(unique_runs) >= 2:
        n_splits = min(5, len(unique_runs))
        if len(unique_runs) == 2:
            n_splits = 2
        if stratify_labels is not None and levels:
            try:
                splitter = StratifiedGroupKFold(n_splits=n_splits)
                desc = f"StratifiedGroupKFold(n_splits={n_splits}) on run labels"
                return splitter, run_groups, desc
            except ValueError as exc:
                if logger is not None:
                    logger.warning(
                        "StratifiedGroupKFold unavailable due to temperature/run distribution: %s. Falling back to GroupKFold.",
                        exc,
                    )
        desc = f"GroupKFold(n_splits={n_splits}) on run labels"
        return GroupKFold(n_splits=n_splits), run_groups, desc

    raise ValueError(
        "Inner CV requires at least two distinct run groups; cannot construct a leakage-free validation strategy."
    )


###################################################################
# Fold Label Utilities
###################################################################

def fold_group_label(groups: Optional[np.ndarray], test_idx: np.ndarray) -> str:
    if groups is None:
        return "pooled"
    values = np.unique(groups[test_idx])
    return ",".join(str(v) for v in values)

