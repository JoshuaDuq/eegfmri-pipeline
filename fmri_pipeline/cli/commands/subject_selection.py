"""Shared subject selection helpers for fMRI CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from eeg_pipeline.utils.parsing import parse_group_arg


def subjects_from_bids_root(
    bids_root: Path, requested: Optional[list[str]]
) -> list[str]:
    """Resolve requested subjects against discovered ``sub-*`` directories."""
    if not bids_root.exists():
        raise FileNotFoundError(f"fMRI BIDS root does not exist: {bids_root}")

    discovered = sorted(
        path.name.replace("sub-", "")
        for path in bids_root.glob("sub-*")
        if path.is_dir()
    )
    if requested is None:
        return discovered

    missing = sorted(set(requested) - set(discovered))
    if missing:
        raise ValueError(
            f"Requested subject(s) not found in BIDS fMRI root: {missing}"
        )
    return requested


def resolve_subjects(args: Any, bids_root: Path, config: Any) -> list[str]:
    """Resolve fMRI CLI subject selection arguments."""
    requested: Optional[list[str]] = None

    if getattr(args, "group", None):
        requested = parse_group_arg(args.group)
    elif getattr(args, "all_subjects", False):
        requested = None
    elif getattr(args, "subjects", None):
        requested = list(dict.fromkeys(args.subjects))
    elif getattr(args, "subject", None):
        requested = list(dict.fromkeys(args.subject))
    else:
        cfg_subjects = getattr(config, "subjects", None) or []
        requested = list(cfg_subjects) if cfg_subjects else None

    return subjects_from_bids_root(bids_root, requested)

