"""Small table and scalar parsing helpers for Study 2 stage I/O."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def require_columns(frame: pd.DataFrame, columns: tuple[str, ...], *, name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}.")


def require_npz_keys(payload: Any, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in payload.files]
    if missing:
        raise ValueError(f"Study 2 NPZ input is missing arrays: {missing}.")


def metric_mapping(frame: pd.DataFrame, *, value_column: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in frame.to_dict("records"):
        value = row[value_column]
        if pd.isna(value):
            continue
        values[str(row["metric"])] = float(value)
    return values


def format_mapping(values: Mapping[str, float]) -> str:
    return ";".join(f"{key}={values[key]:.12g}" for key in sorted(values))


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"Expected boolean value, got {value!r}.")


__all__ = [
    "format_mapping",
    "metric_mapping",
    "parse_bool",
    "require_columns",
    "require_npz_keys",
]
