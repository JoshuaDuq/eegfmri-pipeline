"""Shared resting-state helpers for feature extraction."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np

from eeg_pipeline.utils.config.loader import get_config_value


REST_INCOMPATIBLE_FEATURE_CATEGORIES = frozenset({"erp", "erds", "itpc", "phase"})
_MISSING = object()


def is_resting_state_feature_mode(config: Any) -> bool:
    """Return whether feature extraction is explicitly configured for rest."""
    return bool(get_config_value(config, "feature_engineering.task_is_rest", False))


def validate_rest_configuration(config: Any) -> None:
    """Require preprocessing and feature extraction rest flags to agree when both are set."""
    preprocessing_raw = get_config_value(
        config,
        "preprocessing.task_is_rest",
        _MISSING,
    )
    feature_raw = get_config_value(
        config,
        "feature_engineering.task_is_rest",
        _MISSING,
    )
    if preprocessing_raw is _MISSING or feature_raw is _MISSING:
        return

    preprocessing_task_is_rest = bool(preprocessing_raw)
    feature_task_is_rest = bool(feature_raw)
    if preprocessing_task_is_rest == feature_task_is_rest:
        return

    raise ValueError(
        "Resting-state feature extraction requires preprocessing.task_is_rest and "
        "feature_engineering.task_is_rest to match; these flags must match. "
        f"got preprocessing.task_is_rest={preprocessing_task_is_rest} and "
        f"feature_engineering.task_is_rest={feature_task_is_rest}."
    )


def validate_rest_feature_categories(
    feature_categories: Iterable[str],
    config: Any,
) -> None:
    """Raise if scientifically event-locked features are requested in rest mode."""
    if not is_resting_state_feature_mode(config):
        return

    incompatible = [
        str(category)
        for category in feature_categories
        if str(category) in REST_INCOMPATIBLE_FEATURE_CATEGORIES
    ]
    if incompatible:
        category_list = ", ".join(sorted(dict.fromkeys(incompatible)))
        raise ValueError(
            "Resting-state feature extraction is not scientifically valid for "
            f"event-locked categories: {category_list}."
        )


def validate_rest_analysis_mode(
    config: Any,
    analysis_mode: Any,
) -> None:
    """Raise when resting-state extraction is combined with trial-wise analysis mode."""
    if not is_resting_state_feature_mode(config):
        return

    normalized_mode = str(analysis_mode or "").strip().lower()
    if not normalized_mode or normalized_mode == "group_stats":
        return

    if normalized_mode == "trial_ml_safe":
        raise ValueError(
            "Resting-state feature extraction requires feature_engineering.analysis_mode=group_stats; "
            "trial_ml_safe is event/trial-oriented and not supported."
        )

    raise ValueError(
        "Resting-state feature extraction requires feature_engineering.analysis_mode=group_stats; "
        f"got {normalized_mode!r}."
    )


def raise_if_rest_incompatible(
    config: Any,
    *,
    feature_name: str,
) -> None:
    """Raise if a single extractor is invoked in resting-state mode."""
    if is_resting_state_feature_mode(config):
        raise ValueError(
            f"{feature_name} is not scientifically valid for resting-state feature extraction."
        )


def raise_if_rest_evoked_subtraction(
    config: Any,
    *,
    feature_name: str,
) -> None:
    """Raise if event-related evoked subtraction is enabled in resting-state mode."""
    if not is_resting_state_feature_mode(config):
        return

    raise ValueError(
        f"{feature_name} subtract_evoked is not scientifically valid for resting-state "
        "feature extraction."
    )


def valid_rest_analysis_segment_masks(
    masks: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Return valid, non-baseline masks that can be analyzed in rest mode."""
    return {
        name: np.asarray(mask, dtype=bool)
        for name, mask in masks.items()
        if mask is not None
        and np.any(mask)
        and str(name).strip().lower() != "baseline"
    }


def select_single_rest_analysis_segment(
    masks: Dict[str, np.ndarray],
    *,
    feature_name: str,
    target_name: str,
) -> Tuple[str, np.ndarray]:
    """Resolve a single unambiguous analysis segment for a rest-mode fallback."""
    valid_masks = valid_rest_analysis_segment_masks(masks)
    if not valid_masks:
        raise ValueError(
            f"{feature_name}: resting-state mode requires exactly one valid non-baseline "
            f"analysis segment when targeted window '{target_name}' is empty, but none were found."
        )
    if len(valid_masks) != 1:
        available = ", ".join(sorted(str(name) for name in valid_masks))
        raise ValueError(
            f"{feature_name}: resting-state mode requires exactly one valid non-baseline "
            f"analysis segment when targeted window '{target_name}' is empty; found {available}."
        )

    segment_name, segment_mask = next(iter(valid_masks.items()))
    return str(segment_name), np.asarray(segment_mask, dtype=bool)
