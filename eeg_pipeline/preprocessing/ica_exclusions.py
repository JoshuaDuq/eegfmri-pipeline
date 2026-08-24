"""The single source of truth for which ICA components are excluded.

MNE-BIDS-Pipeline records the exclusion decision in ``*_proc-ica_components.tsv``, not in
the ``exclude`` attribute of the saved ``*_proc-ica_ica.fif``. The manual review step
edits that table, and ``_08a_apply_ica`` reads it to build the cleaned data.

Anything that shows "before versus after ICA" evidence therefore has to read the same
table. Reading ``ICA.exclude`` off the FIF instead is the dangerous failure: it does not
raise, it silently applies whatever exclusion set happened to be persisted at fit time —
in the worst case none at all, making every before/after panel a comparison of a signal
with itself, which looks like a reassuringly small correction rather than a bug.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import mne
import numpy as np
import pandas as pd

#: Value of the ``status`` column marking a component for removal.
EXCLUDED_STATUS = "bad"

_ICA_SUFFIX = "_proc-ica_ica.fif"
_COMPONENTS_SUFFIX = "_proc-ica_components.tsv"
_MANUAL_REVIEW_COLUMNS = (
    "manual_review_status",
    "manual_reviewed_by",
    "manual_reviewed_at",
)


def components_path_for_ica(ica_path: Path) -> Path:
    """Return the component table that decides the exclusions for an ICA solution."""
    if not ica_path.name.endswith(_ICA_SUFFIX):
        raise ValueError(f"Expected an ICA path ending in {_ICA_SUFFIX!r}, got {ica_path}.")
    return ica_path.with_name(ica_path.name.removesuffix(_ICA_SUFFIX) + _COMPONENTS_SUFFIX)


def read_component_statuses(path: Path, *, component_count: int) -> pd.DataFrame:
    """Read and validate the component status table for a decomposition."""
    if not path.is_file():
        raise FileNotFoundError(f"ICA component status table does not exist: {path}")
    statuses = pd.read_csv(path, sep="\t")
    required = {"component", "status", "status_description"}
    if not required.issubset(statuses.columns) or not np.array_equal(
        statuses["component"].to_numpy(), np.arange(component_count)
    ):
        raise ValueError(f"ICA component status table is invalid: {path}")
    unknown = sorted(set(statuses["status"].astype(str)) - {"good", EXCLUDED_STATUS})
    if unknown:
        raise ValueError(
            f"ICA component status table {path} has unrecognized status values: {unknown}. "
            f"Every row must be 'good' or {EXCLUDED_STATUS!r}."
        )
    statuses = statuses.copy()
    statuses["status_description"] = statuses["status_description"].fillna("")
    return statuses


def reviewed_exclusions(path: Path, *, component_count: int) -> list[int]:
    """Return the component indices the review marked for removal."""
    statuses = read_component_statuses(path, component_count=component_count)
    excluded = statuses.loc[statuses["status"] == EXCLUDED_STATUS, "component"]
    return [int(component) for component in excluded]


def initialize_manual_review_attestation(path: Path) -> pd.DataFrame:
    """Mark every component of a newly fitted decomposition as awaiting review."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ICA component status table does not exist: {path}")
    frame = pd.read_csv(path, sep="\t", keep_default_na=False)
    read_component_statuses(path, component_count=len(frame))
    frame["manual_review_status"] = "pending"
    frame["manual_reviewed_by"] = ""
    frame["manual_reviewed_at"] = ""
    frame.to_csv(path, sep="\t", index=False)
    return frame


def validate_manual_review_attestation(path: Path) -> pd.DataFrame:
    """Require an explicit human decision record for every ICA component.

    ``manual_review_complete`` is a workflow request, not evidence. The component table
    is the exclusion source of truth, so the attestation lives on each of its rows: who
    reviewed that component, when, and whether the pass is complete. Extra columns are
    retained by all component-table writers in this package.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"ICA component status table does not exist: {path}")
    frame = pd.read_csv(path, sep="\t", keep_default_na=False)
    missing = [column for column in _MANUAL_REVIEW_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Manual ICA review attestation is missing {missing} in {path}. Add "
            "manual_review_status='reviewed', manual_reviewed_by, and an ISO-8601 "
            "manual_reviewed_at value to every component row after inspection."
        )
    read_component_statuses(path, component_count=len(frame))

    review_status = frame["manual_review_status"].astype(str).str.strip().str.lower()
    incomplete = frame.loc[review_status != "reviewed", "component"].tolist()
    if incomplete:
        raise ValueError(f"ICA components are not reviewed in {path}: {incomplete}")

    reviewers = frame["manual_reviewed_by"].astype(str).str.strip()
    missing_reviewers = frame.loc[reviewers.isin({"", "n/a", "nan"}), "component"].tolist()
    if missing_reviewers:
        raise ValueError(
            f"Manual ICA review has no reviewer for components {missing_reviewers} in {path}."
        )

    invalid_dates = []
    for component, value in zip(frame["component"], frame["manual_reviewed_at"], strict=True):
        try:
            parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        except ValueError:
            invalid_dates.append(int(component))
            continue
        if parsed.tzinfo is None:
            invalid_dates.append(int(component))
    if invalid_dates:
        raise ValueError(
            "manual_reviewed_at must be an ISO-8601 timestamp with timezone for "
            f"components {invalid_dates} in {path}."
        )
    return frame


def promote_exclusions(
    path: Path,
    *,
    components: dict[int, str],
    component_count: int,
) -> list[int]:
    """Mark additional components bad in the table that decides the exclusions.

    ``components`` maps a component index to the description recorded for it. Rows that
    are already ``bad`` are left exactly as they are, so a description written by
    MNE-BIDS-Pipeline or edited by a reviewer is never overwritten and calling this twice
    changes nothing the second time.

    Returns the indices this call newly marked.
    """
    statuses = read_component_statuses(path, component_count=component_count)
    unknown = sorted(index for index in components if not 0 <= index < component_count)
    if unknown:
        raise ValueError(
            f"Cannot promote components {unknown} in {path}: "
            f"the decomposition has {component_count} components."
        )

    newly_excluded = []
    for component, description in sorted(components.items()):
        row = statuses["component"] == component
        if (statuses.loc[row, "status"] == EXCLUDED_STATUS).all():
            continue
        statuses.loc[row, "status"] = EXCLUDED_STATUS
        statuses.loc[row, "status_description"] = description
        newly_excluded.append(int(component))

    if newly_excluded:
        statuses.to_csv(path, sep="\t", index=False)
    return newly_excluded


def read_ica_with_reviewed_exclusions(ica_path: Path) -> mne.preprocessing.ICA:
    """Read an ICA solution with ``exclude`` set from its component status table.

    Use this everywhere a report or QC measurement needs the exclusions that actually
    produced the cleaned data, so the evidence and the derivative cannot disagree.
    """
    ica = mne.preprocessing.read_ica(ica_path, verbose="ERROR")
    ica.exclude = reviewed_exclusions(
        components_path_for_ica(ica_path),
        component_count=int(ica.n_components_),
    )
    return ica


__all__ = [
    "EXCLUDED_STATUS",
    "components_path_for_ica",
    "initialize_manual_review_attestation",
    "promote_exclusions",
    "read_component_statuses",
    "read_ica_with_reviewed_exclusions",
    "reviewed_exclusions",
    "validate_manual_review_attestation",
]
