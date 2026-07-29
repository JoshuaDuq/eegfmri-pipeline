"""Shared EEG dataset root resolution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from eeg_pipeline.utils.config.loader import get_config_value

_MISSING = object()


def _get_optional_path(config: Any, dotted_key: str, legacy_key: str) -> Optional[Path]:
    raw_value = get_config_value(config, dotted_key, _MISSING)
    if raw_value is _MISSING or raw_value is None:
        raw_value = get_config_value(config, legacy_key, _MISSING)
    # An explicit ``null`` is how this config says "not set" — every optional path key
    # ships that way. Without this it reached str() below and became the literal path
    # "None", so a missing root produced a nonexistent directory rather than the error
    # that names the key.
    if raw_value is _MISSING or raw_value is None:
        return None

    value = str(raw_value).strip()
    if not value:
        return None
    return Path(value)


def _resolve_required_path(
    config: Any,
    *,
    dotted_key: str,
    legacy_key: str,
    error_message: str,
) -> Path:
    path_value = _get_optional_path(config, dotted_key, legacy_key)
    if path_value is None:
        raise ValueError(error_message)
    return path_value


def resolve_resting_state_eeg_mode(config: Any) -> bool:
    """Resolve whether EEG inputs should come from the resting-state dataset."""
    preprocessing_raw = get_config_value(config, "preprocessing.task_is_rest", _MISSING)
    feature_raw = get_config_value(config, "feature_engineering.task_is_rest", _MISSING)

    if preprocessing_raw is _MISSING and feature_raw is _MISSING:
        return False
    if preprocessing_raw is _MISSING:
        return bool(feature_raw)
    if feature_raw is _MISSING:
        return bool(preprocessing_raw)

    preprocessing_task_is_rest = bool(preprocessing_raw)
    feature_task_is_rest = bool(feature_raw)
    if preprocessing_task_is_rest != feature_task_is_rest:
        raise ValueError(
            "Resting-state EEG input selection requires preprocessing.task_is_rest "
            "and feature_engineering.task_is_rest to match."
        )
    return preprocessing_task_is_rest


def resolve_eeg_bids_root(config: Any, *, task_is_rest: Optional[bool] = None) -> Path:
    """Resolve the EEG BIDS input root for task or resting-state processing.

    The rest-specific root is where a study that acquires *both* keeps its resting-state
    dataset apart from its task dataset. A study that acquires only rest has one dataset
    and one root, so when the rest key is unset the primary root is used rather than
    demanding the same path be written twice under a second name.
    """
    use_rest_root = (
        resolve_resting_state_eeg_mode(config) if task_is_rest is None else bool(task_is_rest)
    )
    if use_rest_root and _get_optional_path(config, "paths.bids_rest_root", "bids_rest_root"):
        return _resolve_required_path(
            config,
            dotted_key="paths.bids_rest_root",
            legacy_key="bids_rest_root",
            error_message=(
                "Resting-state EEG input root is not configured. Set "
                "paths.bids_rest_root in eeg_config.yaml or pass --bids-rest-root."
            ),
        )
    return _resolve_required_path(
        config,
        dotted_key="paths.bids_root",
        legacy_key="bids_root",
        error_message=(
            "EEG BIDS root is not configured. Set paths.bids_root in eeg_config.yaml "
            "or pass --bids-root."
        ),
    )


def resolve_eeg_deriv_root(config: Any, *, task_is_rest: Optional[bool] = None) -> Path:
    """Resolve the EEG derivatives root for task or resting-state processing.

    Falls back to the primary derivatives root when no rest-specific one is set, for the
    same reason as the input root above: a rest-only study has one of each.
    """
    use_rest_root = (
        resolve_resting_state_eeg_mode(config) if task_is_rest is None else bool(task_is_rest)
    )
    if use_rest_root and _get_optional_path(config, "paths.deriv_rest_root", "deriv_rest_root"):
        return _resolve_required_path(
            config,
            dotted_key="paths.deriv_rest_root",
            legacy_key="deriv_rest_root",
            error_message=(
                "Resting-state EEG derivatives root is not configured. Set "
                "paths.deriv_rest_root in eeg_config.yaml or pass --deriv-rest-root."
            ),
        )
    return _resolve_required_path(
        config,
        dotted_key="paths.deriv_root",
        legacy_key="deriv_root",
        error_message=(
            "EEG derivatives root is not configured. Set paths.deriv_root in eeg_config.yaml "
            "or pass --deriv-root."
        ),
    )


def resolve_resting_state_fmri_mode(config: Any) -> bool:
    """Resolve whether fMRI inputs should come from the resting-state dataset."""
    preprocessing_raw = get_config_value(config, "fmri_preprocessing.task_is_rest", _MISSING)
    resting_state_raw = get_config_value(config, "fmri_resting_state.task_is_rest", _MISSING)

    if preprocessing_raw is _MISSING and resting_state_raw is _MISSING:
        return False
    if preprocessing_raw is _MISSING:
        return bool(resting_state_raw)
    if resting_state_raw is _MISSING:
        return bool(preprocessing_raw)

    preprocessing_task_is_rest = bool(preprocessing_raw)
    resting_state_task_is_rest = bool(resting_state_raw)
    if preprocessing_task_is_rest != resting_state_task_is_rest:
        raise ValueError(
            "Resting-state fMRI input selection requires fmri_preprocessing.task_is_rest "
            "and fmri_resting_state.task_is_rest to match."
        )
    return preprocessing_task_is_rest


def resolve_fmri_bids_root(config: Any, *, task_is_rest: Optional[bool] = None) -> Path:
    """Resolve the fMRI BIDS input root for task or resting-state processing."""
    use_rest_root = (
        resolve_resting_state_fmri_mode(config) if task_is_rest is None else bool(task_is_rest)
    )
    if use_rest_root and _get_optional_path(config, "paths.bids_rest_root", "bids_rest_root"):
        return _resolve_required_path(
            config,
            dotted_key="paths.bids_rest_root",
            legacy_key="bids_rest_root",
            error_message=(
                "Resting-state fMRI input root is not configured. Set paths.bids_rest_root "
                "in eeg_config.yaml or pass --bids-rest-root."
            ),
        )
    return _resolve_required_path(
        config,
        dotted_key="paths.bids_fmri_root",
        legacy_key="bids_fmri_root",
        error_message=(
            "fMRI BIDS root is not configured. Set paths.bids_fmri_root in eeg_config.yaml "
            "or pass --bids-fmri-root."
        ),
    )


def resolve_fmri_deriv_root(config: Any, *, task_is_rest: Optional[bool] = None) -> Path:
    """Resolve the fMRI derivatives root for task or resting-state processing."""
    use_rest_root = (
        resolve_resting_state_fmri_mode(config) if task_is_rest is None else bool(task_is_rest)
    )
    if use_rest_root and _get_optional_path(config, "paths.deriv_rest_root", "deriv_rest_root"):
        return _resolve_required_path(
            config,
            dotted_key="paths.deriv_rest_root",
            legacy_key="deriv_rest_root",
            error_message=(
                "Resting-state fMRI derivatives root is not configured. Set paths.deriv_rest_root "
                "in eeg_config.yaml or pass --deriv-rest-root."
            ),
        )
    return _resolve_required_path(
        config,
        dotted_key="paths.deriv_root",
        legacy_key="deriv_root",
        error_message=(
            "fMRI derivatives root is not configured. Set paths.deriv_root in eeg_config.yaml "
            "or pass --deriv-root."
        ),
    )
