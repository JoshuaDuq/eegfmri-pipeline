"""The HTML report carries the row-level evidence required by the ICA gate."""

from __future__ import annotations

import mne
import pandas as pd
import pytest

from eeg_pipeline.preprocessing.report.manual_ica_review import (
    add_manual_ica_review,
    manual_ica_review_html,
)


def _components() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "component": [0, 1],
            "status": ["bad", "good"],
            "status_description": ["blink", "retained after inspection"],
            "manual_review_status": ["reviewed", "pending"],
            "manual_reviewed_by": ["AB", ""],
            "manual_reviewed_at": ["2026-08-22T14:30:00-04:00", ""],
        }
    )


def test_manual_ledger_names_every_decision_and_attestation() -> None:
    document = manual_ica_review_html(_components())

    assert "ICA000" in document
    assert "excluded" in document
    assert "ICA001" in document
    assert "pending" in document
    assert "AB" in document
    assert "authoritative" in document.lower()


def test_missing_attestation_columns_fail_instead_of_looking_pending() -> None:
    with pytest.raises(ValueError, match="manual_review_status"):
        manual_ica_review_html(_components().drop(columns="manual_review_status"))


def test_rebuilding_replaces_the_manual_ledger() -> None:
    report = mne.Report(title="subject", verbose="ERROR")
    report.add_html(
        html="<p>duplicate automated decisions</p>",
        title="Why each component was excluded",
        section="ICA decomposition quality",
        tags=("ica", "ica-decomposition"),
    )

    add_manual_ica_review(report=report, components=_components())
    add_manual_ica_review(report=report, components=_components())

    assert len(report._content) == 1
    assert report._content[0].section == "ICA decomposition quality"
