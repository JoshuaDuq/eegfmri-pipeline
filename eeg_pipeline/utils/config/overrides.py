"""Shared runtime config override helpers for CLI/script entry points."""

from __future__ import annotations

from typing import Any, Optional


def apply_runtime_overrides(
    config: dict[str, Any],
    *,
    task: Optional[str] = None,
    source_root: Optional[str] = None,
    bids_root: Optional[str] = None,
    bids_fmri_root: Optional[str] = None,
    deriv_root: Optional[str] = None,
) -> None:
    """Apply optional runtime path/task overrides in-place.

    This keeps override behavior consistent between scripts and CLI commands.
    """
    paths = config.setdefault("paths", {})
    if source_root:
        paths["source_data"] = source_root
    if bids_root:
        paths["bids_root"] = bids_root
    if bids_fmri_root:
        paths["bids_fmri_root"] = bids_fmri_root
    if deriv_root:
        paths["deriv_root"] = deriv_root

    if task:
        config.setdefault("project", {})["task"] = task

