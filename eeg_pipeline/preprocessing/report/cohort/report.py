"""Assembling the cohort document, and writing down what went into it.

Three outputs, and each exists for a different reader.

The HTML is for a person deciding whether a group analysis is viable. Its sections carry
the same names as the subject report so that moving between the two costs nothing.

The audit tables are for a person who does not believe a figure. Every plotted value is
written beside the document in a table, so a claim can be checked without re-deriving it
and without trusting that the figure and the number agree.

The log is for a person reading the document six months later. It records which
participants contributed, which were found and skipped and why, and the gate settings the
regimes were decided under -- so that a figure with no median can be understood as a
participant count rather than as a measurement that failed.

Sections are added in a fixed order that runs from what the cohort *is* to what survived
cleaning, because the later sections are only interpretable against the earlier ones.
"""

from __future__ import annotations

import importlib.metadata
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import mne
import pandas as pd

from eeg_pipeline.preprocessing.report.cohort.aggregate import DEFAULT_GATES, BandGates
from eeg_pipeline.preprocessing.report.cohort.at_a_glance import (
    add_at_a_glance_section,
    glance_frame,
    glance_rows,
)
from eeg_pipeline.preprocessing.report.cohort.collect import Cohort
from eeg_pipeline.preprocessing.report.cohort.composition import add_composition_section
from eeg_pipeline.preprocessing.report.cohort.continuity import (
    add_continuity_section,
    cohort_continuity,
    flagged_audit,
)
from eeg_pipeline.preprocessing.report.cohort.coverage import (
    add_coverage_section,
    cohort_coverage,
)
from eeg_pipeline.preprocessing.report.cohort.events import (
    add_events_section,
    cohort_events,
    events_audit,
)
from eeg_pipeline.preprocessing.report.cohort.homogeneity import add_homogeneity_section
from eeg_pipeline.preprocessing.report.cohort.ica import add_ica_section
from eeg_pipeline.preprocessing.report.cohort.multiplicity import (
    add_multiplicity_section,
    multiplicity_frame,
)
from eeg_pipeline.preprocessing.report.cohort.preservation import (
    add_preservation_section,
    alpha_frame,
    cleaning_versus_signal,
    reliability_frame,
)
from eeg_pipeline.preprocessing.report.cohort.provenance import add_provenance_section
from eeg_pipeline.preprocessing.report.cohort.rejection import (
    add_rejection_section,
    cohort_rejection,
    rejection_audit,
)
from eeg_pipeline.preprocessing.report.cohort.sidecar import SCHEMA_VERSION
from eeg_pipeline.preprocessing.report.cohort.spectra import (
    add_spectra_section,
    aperiodic_frame,
    spectra_audit,
)
from eeg_pipeline.preprocessing.report.style import apply_report_css

#: Layout version of the cohort log, so a reader can tell whether it understands the file.
LOG_SCHEMA_VERSION = 1

REPORT_SUFFIX = "_desc-cohort_report.html"
LOG_SUFFIX = "_desc-cohort_log.json"
AUDIT_PREFIX = "_desc-cohort"

#: Packages whose version changes the numbers in the document.
_TRACKED_PACKAGES = ("mne", "mne-bids-pipeline", "mne-bids", "mne-icalabel", "autoreject")


@dataclass(frozen=True)
class CohortReportPaths:
    """Everything one cohort run wrote."""

    html: Path
    log: Path
    audit: tuple[Path, ...] = field(default_factory=tuple)


def _entity(task: str | None) -> str:
    """BIDS-style prefix for the outputs, naming the task when there is exactly one."""
    return f"task-{task}" if task else "cohort"


def _tracked_versions() -> dict[str, str]:
    versions: dict[str, str] = {"mne": mne.__version__}
    for package in _TRACKED_PACKAGES:
        try:
            versions[package.replace("-", "_")] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    try:
        from eeg_pipeline import __version__ as pipeline_version
    except ImportError:  # pragma: no cover - the package is always importable here
        pipeline_version = "unknown"
    versions["eeg_pipeline"] = str(pipeline_version)
    return versions


def _audit_tables(cohort: Cohort, *, gates: BandGates) -> dict[str, pd.DataFrame]:
    """Every plotted value, keyed by the name its file takes.

    Built from the same functions the sections draw from, so a table and the figure beside
    it cannot disagree. A section that produced nothing contributes no file rather than an
    empty one, matching how the document itself behaves.
    """
    from eeg_pipeline.preprocessing.report.cohort.ica import decomposition_frame
    from eeg_pipeline.preprocessing.report.cohort.spectra import cohort_spectra

    tables: dict[str, pd.DataFrame] = {}
    builders: tuple[tuple[str, Callable[[], pd.DataFrame | None]], ...] = (
        ("glance", lambda: glance_frame(glance_rows(cohort, gates=gates))),
        ("spectra", lambda: _optional(cohort_spectra(cohort, gates=gates), spectra_audit)),
        ("aperiodic", lambda: aperiodic_frame(cohort)),
        ("decomposition", lambda: decomposition_frame(cohort)),
        ("alpha", lambda: alpha_frame(cohort)),
        ("reliability", lambda: reliability_frame(cohort)),
        ("cleaning", lambda: cleaning_versus_signal(cohort)),
        ("channels", lambda: _optional(cohort_coverage(cohort), lambda c: c.channels)),
        ("rejection", lambda: _optional(cohort_rejection(cohort), rejection_audit)),
        # The re-export worklist, as its own file: it is the one audit table a reader works
        # *through* rather than checks a figure against.
        ("flagged", lambda: _optional(cohort_continuity(cohort), flagged_audit)),
        ("events", lambda: _optional(cohort_events(cohort), events_audit)),
        ("multiplicity", lambda: multiplicity_frame(cohort, gates=gates)),
    )
    for name, build in builders:
        frame = build()
        if frame is not None and not frame.empty:
            tables[name] = frame
    return tables


def _optional(value: Any, render: Callable[[Any], pd.DataFrame]) -> pd.DataFrame | None:
    return None if value is None else render(value)


def _log(cohort: Cohort, *, task: str | None, gates: BandGates, audit: Sequence[Path]) -> dict:
    return {
        "schema_version": LOG_SCHEMA_VERSION,
        "sidecar_schema_version": SCHEMA_VERSION,
        "written_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "task": task,
        "tasks_found": list(cohort.tasks),
        "n_participants": cohort.n_participants,
        "participants": list(cohort.subjects),
        "n_runs": sum(participant.n_runs for participant in cohort.participants),
        "paradigms": [paradigm.value for paradigm in cohort.paradigms],
        "not_aggregated": [
            {"subject": entry.subject, "reason": entry.reason}
            for entry in cohort.not_aggregated
        ],
        # Recorded so a reader meeting a panel with no median can tell that the
        # participant count did not support one, rather than that a measurement failed.
        "band_gates": {
            "min_subjects_for_median": gates.min_subjects_for_median,
            "min_subjects_for_outer_band": gates.min_subjects_for_outer_band,
        },
        "versions": _tracked_versions(),
        "audit_tables": [path.name for path in audit],
    }


def build_cohort_report(
    cohort: Cohort,
    *,
    output_dir: Path | str,
    task: str | None = None,
    gates: BandGates = DEFAULT_GATES,
    title: str | None = None,
) -> CohortReportPaths:
    """Assemble the cohort document and write it with its audit tables and log.

    ``task`` names the outputs and is the caller's decision: :class:`Cohort` reports which
    tasks it turned out to cover but does not choose between them.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    entity = _entity(task)

    report = mne.Report(
        title=title or f"Cohort preprocessing QC · {task or 'all tasks'}",
        verbose="ERROR",
    )
    apply_report_css(report)

    # Ordered from what the cohort is, through how it was processed, to what survived:
    # the later sections are only interpretable against the earlier ones. Each section
    # returns nothing and adds nothing when no participant supports it, so a section is
    # absent rather than empty.
    add_composition_section(report=report, cohort=cohort, gates=gates)
    add_multiplicity_section(report=report, cohort=cohort, gates=gates)
    add_at_a_glance_section(report=report, cohort=cohort, gates=gates)
    add_homogeneity_section(report=report, cohort=cohort)
    add_coverage_section(report=report, cohort=cohort)
    add_rejection_section(report=report, cohort=cohort, gates=gates)
    add_ica_section(report=report, cohort=cohort, gates=gates)
    add_spectra_section(report=report, cohort=cohort, gates=gates)
    add_preservation_section(report=report, cohort=cohort, gates=gates)
    add_continuity_section(report=report, cohort=cohort, gates=gates)
    add_events_section(report=report, cohort=cohort)

    audit_paths: list[Path] = []
    for name, frame in _audit_tables(cohort, gates=gates).items():
        path = directory / f"{entity}{AUDIT_PREFIX}{name}_qc.tsv"
        frame.to_csv(path, sep="\t", index=False)
        audit_paths.append(path)

    # Written last, because it is the only section that describes the run as a whole and
    # therefore the only one that has to wait until the run is over: it names the audit
    # tables, which do not exist until the sections above have decided what there was to
    # write down.
    record = _log(cohort, task=task, gates=gates, audit=audit_paths)
    add_provenance_section(report=report, log=record)

    html = directory / f"{entity}{REPORT_SUFFIX}"
    report.save(html, overwrite=True, open_browser=False, verbose="ERROR")

    log = directory / f"{entity}{LOG_SUFFIX}"
    log.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    return CohortReportPaths(html=html, log=log, audit=tuple(audit_paths))


__all__ = [
    "AUDIT_PREFIX",
    "LOG_SCHEMA_VERSION",
    "LOG_SUFFIX",
    "REPORT_SUFFIX",
    "CohortReportPaths",
    "build_cohort_report",
]
