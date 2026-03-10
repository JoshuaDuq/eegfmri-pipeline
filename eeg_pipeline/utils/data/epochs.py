from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mne
import numpy as np
import pandas as pd

from ..config.loader import ConfigDict
from eeg_pipeline.infra.paths import (
    find_clean_epochs_path,
    _find_clean_events_path,
)
from eeg_pipeline.utils.data.columns import (
    find_binary_outcome_column_in_events,
    resolve_outcome_column,
    resolve_predictor_column,
)
from eeg_pipeline.utils.data.feature_alignment import require_trial_id_column

EEGConfig = ConfigDict


def _validate_event_columns(
    events_df: pd.DataFrame,
    config: EEGConfig,
    logger: logging.Logger,
    *,
    required_groups: Optional[Any] = None,
) -> None:
    if events_df is None or events_df.empty:
        return

    event_cols_config = config.get("event_columns", {})
    if not event_cols_config:
        logger.warning("No event_columns found in config; skipping validation")
        return

    required_groups = (
        required_groups
        if required_groups is not None
        else config.get("event_columns.required", None)
    )
    missing_columns = _find_missing_event_columns(
        events_df,
        event_cols_config,
        required_groups=required_groups,
        config=config,
    )
    if missing_columns:
        available = list(events_df.columns)
        error_msg = (
            f"Required event columns not found in events DataFrame. "
            f"Missing: {', '.join(missing_columns)}. "
            f"Available columns: {available}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)


def _find_missing_event_columns(
    events_df: pd.DataFrame,
    event_cols_config: Dict[str, Any],
    *,
    required_groups: Optional[Any] = None,
    config: Optional[EEGConfig] = None,
) -> list[str]:
    required_set: Optional[set[str]] = None
    if isinstance(required_groups, (list, tuple, set)):
        required_set = {str(item).strip() for item in required_groups if str(item).strip()}
        if len(required_set) == 0:
            return []

    missing_columns = []
    for logical_name, candidates in event_cols_config.items():
        if str(logical_name) == "required":
            continue
        if required_set is not None and str(logical_name) not in required_set:
            continue
        explicit_key = None
        if logical_name == "outcome":
            explicit_key = "behavior_analysis.outcome_column"
        elif logical_name == "predictor":
            explicit_key = "behavior_analysis.predictor_column"
        if (
            explicit_key is not None
            and config is not None
            and hasattr(config, "get")
        ):
            explicit_col = str(config.get(explicit_key, "") or "").strip()
            if explicit_col and explicit_col in events_df.columns:
                continue

        # Reuse shared resolvers so validation matches downstream behavior.
        if logical_name == "outcome":
            if resolve_outcome_column(events_df, config) is not None:
                continue
        elif logical_name == "predictor":
            if resolve_predictor_column(events_df, config) is not None:
                continue
        elif logical_name == "binary_outcome":
            if config is not None and find_binary_outcome_column_in_events(events_df, config) is not None:
                continue

        if not isinstance(candidates, (list, tuple)):
            continue
        found = any(col in events_df.columns for col in candidates)
        if not found:
            missing_columns.append(
                f"event_columns.{logical_name} (tried: {candidates})"
            )
    return missing_columns


def _validate_align_mode(align: str) -> None:
    valid_align_modes = ("strict", "warn", "none")
    if align not in valid_align_modes:
        raise ValueError(
            f"align must be one of {valid_align_modes}, got '{align}'"
        )


def _resolve_task_is_rest(
    config: EEGConfig,
    task_is_rest: Optional[bool],
) -> bool:
    if task_is_rest is not None:
        return bool(task_is_rest)
    return bool(config.get("preprocessing.task_is_rest", False))


def _handle_missing_events(
    epochs: mne.Epochs,
    align: str,
    subject: str,
    task: str,
    logger: logging.Logger,
    task_is_rest: bool,
) -> Tuple[mne.Epochs, Optional[pd.DataFrame]]:
    if task_is_rest:
        logger.info(
            "Clean events.tsv not found for sub-%s, task-%s; synthesizing resting-state trial alignment.",
            subject,
            task,
        )
        rest_events = pd.DataFrame(
            {"trial_id": np.arange(1, len(epochs) + 1, dtype=int)}
        )
        return epochs, rest_events

    if align == "strict":
        raise ValueError(
            f"Clean events.tsv not found for sub-{subject}, task-{task}. "
            f"Required when align='strict'"
        )
    logger.warning("Clean events.tsv not found; metadata will not be set.")
    return epochs, None


def load_epochs_for_analysis(
    subject: str,
    task: str,
    align: str = "strict",
    preload: bool = False,
    deriv_root: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
    config: Optional[EEGConfig] = None,
    task_is_rest: Optional[bool] = None,
    constants: Optional[Any] = None,
    use_cache: bool = True,
    required_event_groups: Optional[Any] = None,
) -> Tuple[Optional[mne.Epochs], Optional[pd.DataFrame]]:
    """Load epochs and clean events.tsv (already aligned, no alignment needed)."""
    if logger is None:
        logger = logging.getLogger(__name__)

    if config is None:
        raise ValueError("config is required for load_epochs_for_analysis")

    _validate_align_mode(align)
    resolved_task_is_rest = _resolve_task_is_rest(config, task_is_rest)
    
    epochs_path = find_clean_epochs_path(
        subject, task, deriv_root=deriv_root, config=config, constants=constants
    )
    if epochs_path is None or not epochs_path.exists():
        logger.error(
            f"Could not find cleaned epochs file for sub-{subject}, task-{task}"
        )
        return None, None

    logger.info(f"Loading epochs: {epochs_path}")
    epochs = mne.read_epochs(epochs_path, preload=preload, verbose=False)

    clean_events_path = _find_clean_events_path(
        subject=subject,
        task=task,
        deriv_root=deriv_root,
        config=config,
        constants=constants,
    )
    
    if clean_events_path is None or not clean_events_path.exists():
        return _handle_missing_events(
            epochs,
            align,
            subject,
            task,
            logger,
            resolved_task_is_rest,
        )

    events_df = pd.read_csv(clean_events_path, sep="\t")
    require_trial_id_column(
        events_df,
        context=f"Clean events.tsv for sub-{subject}, task-{task}",
    )
    
    logger.info(f"Loaded clean events.tsv: {len(events_df)} rows")
    
    if len(events_df) != len(epochs):
        raise ValueError(
            f"Clean events.tsv length mismatch for sub-{subject}, task-{task}: "
            f"events={len(events_df)}, epochs={len(epochs)}"
        )
    
    _validate_event_columns(
        events_df,
        config,
        logger,
        required_groups=required_event_groups,
    )
    
    if use_cache:
        epochs._behavioral = events_df  # type: ignore[attr-defined]
    return epochs, events_df.reset_index(drop=True)


__all__ = [
    "load_epochs_for_analysis",
]
