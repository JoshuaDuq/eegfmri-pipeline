from __future__ import annotations

from typing import Iterable, List, Tuple

import pandas as pd


TRIAL_ID_COLUMN = "trial_id"

FEATURE_ALIGNMENT_COLUMNS: Tuple[str, ...] = (TRIAL_ID_COLUMN,)


def feature_alignment_columns_in(frame: pd.DataFrame) -> List[str]:
    return [col for col in FEATURE_ALIGNMENT_COLUMNS if col in frame.columns]


def drop_feature_alignment_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [col for col in FEATURE_ALIGNMENT_COLUMNS if col in df.columns]
    if not columns_to_drop:
        return df
    return df.drop(columns=columns_to_drop, errors="ignore")


def filter_feature_payload_columns(columns: Iterable[str]) -> List[str]:
    blocked = set(FEATURE_ALIGNMENT_COLUMNS)
    return [str(col) for col in columns if str(col) not in blocked]


def attach_feature_alignment_columns(
    df: pd.DataFrame,
    events_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if events_df is None or df is None or df.empty:
        return df
    if len(df) != len(events_df):
        return df

    if TRIAL_ID_COLUMN not in events_df.columns:
        raise ValueError(
            "Aligned events are missing required canonical alignment column 'trial_id'. "
            "Re-run preprocessing to regenerate clean events with the current contract."
        )

    out = df.copy()
    event_series = events_df[TRIAL_ID_COLUMN].reset_index(drop=True)
    if TRIAL_ID_COLUMN in out.columns:
        current = out[TRIAL_ID_COLUMN].reset_index(drop=True)
        if not current.equals(event_series):
            raise ValueError(
                "Feature table column 'trial_id' conflicts with aligned events; "
                "cannot persist trial alignment safely."
            )
        return out

    out.insert(0, TRIAL_ID_COLUMN, event_series)
    return out


def require_trial_id_column(frame: pd.DataFrame, *, context: str) -> pd.Series:
    if TRIAL_ID_COLUMN not in frame.columns:
        raise ValueError(
            f"{context} is missing required canonical alignment column 'trial_id'."
        )
    return frame[TRIAL_ID_COLUMN]


__all__ = [
    "FEATURE_ALIGNMENT_COLUMNS",
    "TRIAL_ID_COLUMN",
    "attach_feature_alignment_columns",
    "drop_feature_alignment_columns",
    "feature_alignment_columns_in",
    "filter_feature_payload_columns",
    "require_trial_id_column",
]
