"""A durable record of which stages built a subject report, when, and what they measured.

A subject report is not written once. Each review stage reopens the same file, appends
its sections and saves again, so a report on disk routinely holds sections written by
different runs at different times — in the worst case under different settings or a
different pipeline version. The document gave no sign of this: the file's own timestamp
is the last save, which says nothing about the sections written half an hour earlier.

Two things follow from writing the record down, and this module exists for both.

The first is the panel: a reader can see which stages contributed to the document in
front of them, and whether they agree about the version of the code that produced them.
A stale section is not detectable from its content, only from its provenance.

The second is that the numbers a stage renders as prose become readable without parsing
HTML. A decision to drop a subject reads ``variance_removed`` from a JSON file rather
than from a paragraph, and the same file records which run of which stage measured it.

Nothing here judges a report. Differing versions across stages are stated as a fact about
the document, not as a fault: rebuilding one section against a newer pipeline is a normal
thing to do deliberately, and only the reader knows whether this one was deliberate.
"""

from __future__ import annotations

import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import mne

from eeg_pipeline.preprocessing.report.tables import Align, Column, grid_table

#: Version of the record's own layout, so a reader can tell whether it understands the
#: file. Bumped when a field changes meaning, not when a stage adds a measurement.
SCHEMA_VERSION = 1

#: Suffix of the sidecar, following the BIDS derivative convention the report itself uses.
_RECORD_SUFFIX = "_desc-reportbuild_log.json"

#: Suffix of the report the record describes.
_REPORT_SUFFIX = "_report.h5"

#: Packages whose version changes the numbers in a report, so a document built across a
#: version change is a document whose sections are not strictly comparable.
_TRACKED_PACKAGES = ("mne", "mne-bids-pipeline", "mne-bids", "mne-icalabel", "autoreject")

#: Tag identifying the panel, so a rebuild replaces it rather than adding a second one.
BUILD_RECORD_TAG = "report-build"

#: Section the panel joins. The record is provenance, and provenance already has a home.
BUILD_RECORD_SECTION = "Configuration"

BUILD_RECORD_TITLE = "How this report was built"


def record_path_for_report(report_path: Path | str) -> Path:
    """Return the record that belongs to a report.

    Derived from the report's own name rather than configured, so the two cannot be
    separated by a file move and a stage cannot be pointed at the wrong record.
    """
    path = Path(report_path)
    stem = path.name
    for suffix in (_REPORT_SUFFIX, ".h5", ".html"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return path.with_name(f"{stem}{_RECORD_SUFFIX}")


def _tracked_versions() -> dict[str, str]:
    """Return the versions of the packages that decide what a report contains."""
    versions: dict[str, str] = {"mne": mne.__version__}
    for package in _TRACKED_PACKAGES:
        try:
            versions[package.replace("-", "_")] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            # An optional dependency that is not installed did not shape this report, so
            # its absence is not worth a placeholder in every record.
            continue
    try:
        from eeg_pipeline import __version__ as pipeline_version
    except ImportError:  # pragma: no cover - the package is always importable here
        pipeline_version = "unknown"
    versions["eeg_pipeline"] = str(pipeline_version)
    return versions


def _empty_record() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "stages": []}


def read_build_record(report_path: Path | str) -> dict[str, Any]:
    """Read the record for a report, or an empty one when no stage has written yet.

    A missing record is the ordinary state before the first stage runs. A record that
    exists but cannot be parsed is not: overwriting it would destroy the provenance the
    file exists to hold, so it is raised rather than replaced.
    """
    path = record_path_for_report(report_path)
    if not path.is_file():
        return _empty_record()
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Report build record {path} is not valid JSON: {error}") from error
    if not isinstance(record, dict) or "stages" not in record:
        raise ValueError(f"Report build record {path} does not hold a stage list.")
    return record


def record_stage(
    report_path: Path | str,
    *,
    stage: str,
    measurements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record that ``stage`` wrote to this report, replacing any previous entry for it.

    Replacing rather than appending because the record describes the report as it stands.
    A stage run six times contributed the sections written by the sixth run, and a log
    listing all six would misdescribe the document by exactly the five that are gone.
    """
    record = read_build_record(report_path)
    entry = {
        "stage": str(stage),
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": _tracked_versions(),
        "measurements": dict(measurements or {}),
    }
    stages = [item for item in record["stages"] if item.get("stage") != str(stage)]
    stages.append(entry)
    record["schema_version"] = SCHEMA_VERSION
    record["stages"] = stages
    path = record_path_for_report(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record


def _measurement_text(measurements: Mapping[str, Any]) -> str | None:
    """Render one stage's recorded measurements as ``key=value`` pairs.

    Plain text rather than per-key ``<code>`` markup: the cell is rendered through the
    shared table builder, which escapes every cell, and a measurement list is the one
    place in this document where markup inside a cell bought nothing a separator does
    not. ``None`` for a stage that recorded nothing, so the builder draws its em dash.
    """
    if not measurements:
        return None
    return ", ".join(
        f"{key}={value:g}"
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else f"{key}={value}"
        for key, value in sorted(measurements.items())
    )


def build_record_html(record: Mapping[str, Any]) -> str:
    """Render the record as the report's own account of how it was assembled."""
    stages = list(record.get("stages", ()))
    if not stages:
        return (
            "<p>No stage recorded itself while building this report, so which parts of it "
            "were written when cannot be reconstructed.</p>"
        )
    ordered = sorted(stages, key=lambda entry: str(entry.get("written_at", "")))
    rows = [
        [
            entry.get("stage", ""),
            str(entry.get("written_at", "")).replace("T", " "),
            entry.get("versions", {}).get("eeg_pipeline", ""),
            entry.get("versions", {}).get("mne", ""),
            _measurement_text(entry.get("measurements", {})),
        ]
        for entry in ordered
    ]
    columns = (
        Column("Stage", align=Align.TEXT, code=True),
        Column("Written (UTC)", align=Align.TEXT),
        Column("Pipeline", align=Align.TEXT),
        Column("MNE", align=Align.TEXT),
        Column("Recorded measurements", align=Align.TEXT),
    )
    document = (
        "<p>Each stage below reopened this report, appended its own sections and saved it "
        "again. The sections in this document are therefore as old as the stage that "
        "wrote them, not as recent as the file's timestamp.</p>"
        + grid_table(columns, rows)
    )
    pipeline_versions = {
        str(entry.get("versions", {}).get("eeg_pipeline", "")) for entry in ordered
    }
    mne_versions = {str(entry.get("versions", {}).get("mne", "")) for entry in ordered}
    if len(pipeline_versions) > 1 or len(mne_versions) > 1:
        document += (
            "<p>These sections were written by <strong>more than one version</strong> of "
            f"the code: pipeline {', '.join(sorted(pipeline_versions))} and MNE "
            f"{', '.join(sorted(mne_versions))}. Rebuilding one section against a newer "
            "version is a normal thing to do on purpose, so this is stated rather than "
            "flagged &mdash; but panels written by different versions are not guaranteed "
            "to have measured their numbers the same way.</p>"
        )
    document += (
        "<p>Recorded measurements are also written beside this report as "
        "<code>*_desc-reportbuild_log.json</code>, so a downstream decision can read them "
        "without parsing this page. A stage with no measurements recorded contributed "
        "figures rather than numbers.</p>"
    )
    return document


def save_subject_report(
    report: mne.Report,
    report_path: Path | str,
    *,
    stage: str,
    measurements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record the stage, refresh the build panel, and write both report files.

    Every stage previously wrote the ``.h5``/``.html`` pair by hand. Doing it here instead
    puts the recording at the one point a stage cannot skip: a stage that appends sections
    without saving has changed nothing, and a stage that saves has recorded itself. It
    also removes the failure where a stage saved the archive and forgot the HTML, leaving
    the page a reader opens older than the file the next stage reopens.
    """
    from eeg_pipeline.preprocessing.report.at_a_glance import add_at_a_glance_section
    from eeg_pipeline.preprocessing.report.organize import order_sections

    path = Path(report_path)
    record = record_stage(path, stage=stage, measurements=measurements)
    report.add_html(
        html=build_record_html(record),
        title=BUILD_RECORD_TITLE,
        section=BUILD_RECORD_SECTION,
        tags=("summary", "provenance", BUILD_RECORD_TAG),
        replace=True,
    )
    # Rebuilt here rather than by any one stage. The landing panel summarises whatever has
    # been measured so far, and which stages those are depends on where the pipeline is;
    # written by a single stage it would freeze at that stage's view of the document.
    # Added after the build panel so it can be moved in front of it.
    add_at_a_glance_section(report=report, record=record)
    # Last, after every stage has added whatever it adds. Ordering here rather than in the
    # stages is what makes the document's shape a property of the pipeline instead of a
    # property of which stages were run and in what sequence.
    order_sections(report)
    report.save(path, overwrite=True, open_browser=False)
    report.save(path.with_suffix(".html"), overwrite=True, open_browser=False)
    return record


__all__ = [
    "BUILD_RECORD_SECTION",
    "BUILD_RECORD_TAG",
    "BUILD_RECORD_TITLE",
    "SCHEMA_VERSION",
    "build_record_html",
    "read_build_record",
    "record_path_for_report",
    "record_stage",
    "save_subject_report",
]
