"""Shared runtime config override helpers for CLI/script entry points."""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence


def apply_runtime_overrides(
    config: dict[str, Any],
    *,
    task: Optional[str] = None,
    source_root: Optional[str] = None,
    bids_root: Optional[str] = None,
    bids_rest_root: Optional[str] = None,
    bids_fmri_root: Optional[str] = None,
    deriv_root: Optional[str] = None,
    deriv_rest_root: Optional[str] = None,
    set_overrides: Optional[Sequence[str]] = None,
) -> None:
    """Apply optional runtime path/task overrides in-place.

    This keeps override behavior consistent between scripts and CLI commands.
    """
    paths = config.setdefault("paths", {})
    if source_root:
        paths["source_data"] = source_root
    if bids_root:
        paths["bids_root"] = bids_root
    if bids_rest_root:
        paths["bids_rest_root"] = bids_rest_root
    if bids_fmri_root:
        paths["bids_fmri_root"] = bids_fmri_root
    if deriv_root:
        paths["deriv_root"] = deriv_root
    if deriv_rest_root:
        paths["deriv_rest_root"] = deriv_rest_root

    if task:
        config.setdefault("project", {})["task"] = task

    apply_set_overrides(config, set_overrides)


def apply_set_overrides(
    config: dict[str, Any],
    set_overrides: Optional[Sequence[str]] = None,
) -> None:
    """Apply repeatable KEY=VALUE config overrides in-place."""
    for override in set_overrides or ():
        _apply_set_override(config, str(override))


def _apply_set_override(config: dict[str, Any], raw_override: str) -> None:
    override = raw_override.strip()
    if not override:
        return
    if "=" not in override:
        return

    key, raw_value = override.split("=", 1)
    path_parts = [part.strip() for part in key.split(".") if part.strip()]
    if not path_parts:
        return

    cursor: dict[str, Any] = config
    for part in path_parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value

    cursor[path_parts[-1]] = _coerce_set_value(raw_value.strip())


def _coerce_set_value(raw_value: str) -> Any:
    lowered = raw_value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "null":
        return None

    try:
        if raw_value and "." not in raw_value and "e" not in lowered:
            return int(raw_value)
    except ValueError:
        pass

    try:
        if "." in raw_value or "e" in lowered:
            return float(raw_value)
    except ValueError:
        pass

    if raw_value.startswith("[") or raw_value.startswith("{") or (
        raw_value.startswith('"') and raw_value.endswith('"')
    ):
        try:
            return json.loads(raw_value)
        except json.JSONDecodeError:
            return raw_value

    return raw_value
