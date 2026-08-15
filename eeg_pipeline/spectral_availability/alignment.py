from __future__ import annotations

import re
from numbers import Integral, Real

import numpy as np
import pandas as pd

from eeg_pipeline.spectral_availability.decomb import DecombManifest
from eeg_pipeline.spectral_availability.model import (
    EpochSpectralAvailability,
    RecordingExclusions,
    RecordingKey,
)


_BIDS_ENTITY_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _canonical_string_entity(value: object, entity: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{entity} must be an explicit string")
    prefix = f"{entity}-"
    canonical = value[len(prefix) :] if value.startswith(prefix) else value
    if _BIDS_ENTITY_PATTERN.fullmatch(canonical) is None:
        raise ValueError(f"{entity} must be a canonical BIDS entity value")
    return canonical


def _canonical_run(value: object) -> str:
    if isinstance(value, str):
        return _canonical_string_entity(value, "run")
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("run_id must be an explicit string or finite integer-valued number")
    if isinstance(value, Integral):
        return _canonical_string_entity(str(value), "run")
    if not isinstance(value, Real):
        raise TypeError("run_id must be an explicit string or finite integer-valued number")

    numeric = float(value)
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError("run_id numeric values must be finite and integer-valued")
    return _canonical_string_entity(str(int(numeric)), "run")


def _event_run(value: object, row_number: int) -> str:
    try:
        return _canonical_run(value)
    except (TypeError, ValueError) as error:
        message = f"event row {row_number} run_id: {error}"
        raise type(error)(message) from error


def _event_session(value: object, row_number: int) -> str:
    try:
        return _canonical_string_entity(value, "ses")
    except (TypeError, ValueError) as error:
        message = f"event row {row_number} session_id: {error}"
        raise type(error)(message) from error


def _is_missing_session(value: object) -> bool:
    if value is None or value is pd.NA or value == "":
        return True
    if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        return bool(np.isnan(float(value)))
    return False


def _unique_exclusions(
    manifest: DecombManifest,
) -> dict[RecordingKey, RecordingExclusions]:
    exclusions_by_key: dict[RecordingKey, RecordingExclusions] = {}
    for exclusion in manifest.exclusions:
        if not isinstance(exclusion, RecordingExclusions):
            raise TypeError("manifest exclusions must contain RecordingExclusions values")
        if exclusion.key in exclusions_by_key:
            raise ValueError(f"duplicate or ambiguous manifest recording key: {exclusion.key}")
        exclusions_by_key[exclusion.key] = exclusion
    return exclusions_by_key


def _column_position(events: pd.DataFrame, column: str) -> int:
    positions = [index for index, candidate in enumerate(events.columns) if candidate == column]
    if not positions:
        raise ValueError(f"events must contain {column}")
    if len(positions) > 1:
        raise ValueError(f"events must contain exactly one {column} column")
    return positions[0]


def align_decomb_to_epochs(
    manifest: DecombManifest,
    *,
    subject: str,
    task: str,
    events: pd.DataFrame,
) -> EpochSpectralAvailability:
    if not isinstance(manifest, DecombManifest):
        raise TypeError("manifest must be a DecombManifest")
    if not isinstance(events, pd.DataFrame):
        raise TypeError("events must be a pandas DataFrame")

    canonical_subject = _canonical_string_entity(subject, "sub")
    canonical_task = _canonical_string_entity(task, "task")
    run_position = _column_position(events, "run_id")
    session_positions = [
        index for index, candidate in enumerate(events.columns) if candidate == "session_id"
    ]
    if len(session_positions) > 1:
        raise ValueError("events must contain at most one session_id column")
    session_position = session_positions[0] if session_positions else None

    exclusions_by_key = _unique_exclusions(manifest)
    candidates = {
        key: exclusion
        for key, exclusion in exclusions_by_key.items()
        if key.subject == canonical_subject and key.task == canonical_task
    }
    session_modes = {key.session is not None for key in candidates}
    if len(session_modes) > 1:
        raise ValueError("selected manifest recordings have mixed or ambiguous session identity")
    uses_sessions = session_modes == {True}
    if uses_sessions and session_position is None:
        raise ValueError("events require session_id for session-specific manifest recordings")

    recording_keys = []
    exclusions_by_epoch = []
    for row_number, values in enumerate(events.itertuples(index=False, name=None)):
        run = _event_run(values[run_position], row_number)
        session = None
        if uses_sessions:
            session_value = values[session_position]
            if _is_missing_session(session_value):
                raise ValueError(
                    f"event row {row_number} session_id must identify a manifest session"
                )
            session = _event_session(session_value, row_number)
        elif session_position is not None:
            session_value = values[session_position]
            if not _is_missing_session(session_value):
                raise ValueError(
                    f"event row {row_number} session_id does not match no-session "
                    "manifest recordings"
                )

        key = RecordingKey(
            subject=canonical_subject,
            task=canonical_task,
            run=run,
            session=session,
        )
        exclusion = candidates.get(key)
        if exclusion is None:
            raise ValueError(f"event row {row_number} is unmatched for recording key {key}")
        recording_keys.append(key)
        exclusions_by_epoch.append(exclusion.intervals)

    return EpochSpectralAvailability(
        recording_keys=tuple(recording_keys),
        exclusions_by_epoch=tuple(exclusions_by_epoch),
    )


__all__ = ["align_decomb_to_epochs"]
