"""Current ICA decisions and their component-level human attestation."""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.report.organize import remove_tagged_content
from eeg_pipeline.preprocessing.report.tables import Align, Column, grid_table

_REQUIRED_COLUMNS = (
    "component",
    "status",
    "status_description",
    "manual_review_status",
    "manual_reviewed_by",
    "manual_reviewed_at",
)


def manual_ica_review_html(components: pd.DataFrame) -> str:
    """Render the authoritative ICA decision and attestation for every component."""
    missing = [column for column in _REQUIRED_COLUMNS if column not in components.columns]
    if missing:
        raise ValueError(f"ICA manual review ledger is missing {missing}.")
    if not np.array_equal(
        components["component"].to_numpy(),
        np.arange(len(components)),
    ):
        raise ValueError("ICA manual review ledger must contain each component in order.")
    unknown_decisions = sorted(set(components["status"].astype(str)) - {"good", "bad"})
    if unknown_decisions:
        raise ValueError(f"ICA manual review ledger has invalid decisions: {unknown_decisions}.")
    review_status = components["manual_review_status"].astype(str).str.strip().str.lower()
    unknown_reviews = sorted(set(review_status) - {"pending", "reviewed"})
    if unknown_reviews:
        raise ValueError(
            f"ICA manual review ledger has invalid review statuses: {unknown_reviews}."
        )

    rows = []
    for row, attestation in zip(
        components.itertuples(index=False),
        review_status,
        strict=True,
    ):
        rows.append(
            [
                f"ICA{int(row.component):03d}",
                "excluded" if row.status == "bad" else "retained",
                str(row.status_description).strip() or None,
                attestation,
                str(row.manual_reviewed_by).strip() or None,
                str(row.manual_reviewed_at).strip() or None,
            ]
        )
    pending = int((review_status == "pending").sum())
    return (
        "<p>This is the <strong>authoritative current decision record</strong> read by "
        "ICA application. The component dossiers provide evidence for each decision; "
        "this table records the decision itself and whether a human attested to it. "
        f"{pending} of {len(components)} components remain pending.</p>"
        + grid_table(
            (
                Column("Component", align=Align.TEXT, code=True),
                Column("Decision", align=Align.TEXT),
                Column("Reason", align=Align.TEXT),
                Column("Manual status", align=Align.TEXT),
                Column("Reviewed by", align=Align.TEXT),
                Column("Reviewed at", align=Align.TEXT),
            ),
            rows,
        )
    )


def add_manual_ica_review(
    *,
    report: mne.Report,
    components: pd.DataFrame,
    section: str = "ICA decomposition quality",
) -> None:
    """Append the current component decisions and row-level review ledger."""
    remove_tagged_content(report, tag="manual-ica-review")
    report.remove(title="Why each component was excluded", remove_all=True)
    report.add_html(
        html=manual_ica_review_html(components),
        title="Manual component review ledger",
        section=section,
        tags=("ica", "manual-ica-review"),
        replace=True,
    )


__all__ = [
    "add_manual_ica_review",
    "manual_ica_review_html",
]
