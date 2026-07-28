"""The report's own account of how it was built.

The per-subject document ends with a build record because a reader meeting it six months
later needs to know which stage wrote which section and under which versions. A cohort
document has the same problem twice over: it was assembled from participants who were each
built at some earlier time, by some version, and the aggregate is no fresher than the
oldest of them.

So this section states three things the figures cannot. Which participants went in, and
which were found and left out. The settings the band regimes were decided under, so that a
panel with no median reads as a participant count rather than as a measurement that failed.
And the versions, both of this run and of the sidecar layout, because a document assembled
from sidecars written by two versions of the pipeline is comparing measurements that may
not have been made the same way.

The same content is written beside the document as JSON, so a downstream reader can have it
without parsing HTML.
"""

from __future__ import annotations

from typing import Any, Mapping

import mne

from eeg_pipeline.preprocessing.report.tables import Align, Column, Metric, grid_table, metric_table

PROVENANCE_SECTION = "How this report was built"
PROVENANCE_TITLE = "What went into this document"
PROVENANCE_TAG = "cohort-provenance"


def _versions_table(versions: Mapping[str, Any]) -> str:
    rows = [[str(name), str(value)] for name, value in sorted(versions.items())]
    columns = (
        Column("Package", align=Align.TEXT, code=True),
        Column("Version", align=Align.TEXT, code=True),
    )
    return grid_table(columns, rows)


def _not_aggregated_table(entries: Any) -> str:
    rows = [
        [str(entry.get("subject", "")), str(entry.get("reason", ""))]
        for entry in entries
        if isinstance(entry, Mapping)
    ]
    if not rows:
        return ""
    columns = (
        Column("Participant", align=Align.TEXT, code=True),
        Column("Why it was not aggregated", align=Align.TEXT),
    )
    return "<h4>Found and not aggregated</h4>" + grid_table(columns, rows)


def provenance_html(log: Mapping[str, Any]) -> str:
    """Render the cohort log as the document's account of itself."""
    gates = log.get("band_gates") or {}
    participants = list(log.get("participants") or ())
    audit = list(log.get("audit_tables") or ())

    rows: list[Metric | tuple[str, object]] = [
        ("Built (UTC)", str(log.get("written_at", "")).replace("T", " ")),
        ("Task", log.get("task") or "every task found"),
        ("Participants aggregated", log.get("n_participants")),
        ("Runs behind them", log.get("n_runs")),
        (
            "Median drawn from",
            f"{gates.get('min_subjects_for_median', '—')} participants",
        ),
        (
            "Outer band drawn from",
            f"{gates.get('min_subjects_for_outer_band', '—')} participants",
        ),
        ("Sidecar layout", log.get("sidecar_schema_version")),
        ("Log layout", log.get("schema_version")),
    ]

    document = (
        "<p>This document was assembled from per-subject QC sidecars, each written when its "
        "participant's report was built. Nothing here was measured by this run: the cohort "
        "command pools recorded measurements rather than recomputing them, which is what "
        "makes a cohort figure and the subject figure beneath it the same measurement.</p>"
        + metric_table(rows)
    )

    if participants:
        document += (
            "<h4>Participants aggregated</h4><p><code>"
            + "</code>, <code>".join(str(subject) for subject in participants)
            + "</code></p>"
        )
    document += _not_aggregated_table(log.get("not_aggregated") or ())

    document += (
        "<p>The band settings above decide what each panel is entitled to draw. A panel "
        "showing individual participants and no median has too few contributors for one, "
        "which is a fact about the cohort rather than a measurement that failed &mdash; and "
        "the settings are recorded here so that reading is available without guessing.</p>"
    )

    versions = log.get("versions") or {}
    if versions:
        document += "<h4>Versions this run used</h4>" + _versions_table(versions)
        document += (
            "<p>These are the versions that assembled the document. The measurements "
            "themselves were made when each participant's report was built, possibly under "
            "different versions; the homogeneity section reports whether they agreed.</p>"
        )

    if audit:
        document += (
            "<h4>Audit tables</h4><p>Every plotted value is written beside this document, "
            "so a figure can be checked without re-deriving it: <code>"
            + "</code>, <code>".join(str(name) for name in audit)
            + "</code>. The same record is written as <code>*_desc-cohort_log.json</code>.</p>"
        )
    return document


def add_provenance_section(*, report: mne.Report, log: Mapping[str, Any]) -> None:
    """Add the cohort provenance section."""
    report.add_html(
        html=provenance_html(log),
        title=PROVENANCE_TITLE,
        section=PROVENANCE_SECTION,
        tags=(PROVENANCE_TAG,),
        replace=True,
    )


__all__ = [
    "PROVENANCE_SECTION",
    "PROVENANCE_TAG",
    "PROVENANCE_TITLE",
    "add_provenance_section",
    "provenance_html",
]
