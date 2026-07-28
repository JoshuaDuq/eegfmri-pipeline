"""The landing panel of a subject report.

A subject report runs to tens of megabytes across roughly twenty-five sections, and the
first thing it shows is a configuration table. The numbers that decide whether a subject
is usable are spread across sections that are individually excellent and collectively
unscannable: how many components were removed lives eleven sections away from how much
variance that cost, which is another four from whether any signal survived it.

This panel is not a new measurement. Every number on it was measured by the stage that
owns the section it belongs to, recorded in the build record at that moment, and is
lifted from there unchanged. Assembling it from the record rather than recomputing is
what makes it impossible for the summary and the section to disagree: there is one
measurement and two places that render it.

Nothing here is graded. Reference values, the reasoning about what a number means, and
the caveats that come with it stay in the owning section, where there is room to state
them properly. A landing panel that said "variance removed: 93.9% ⚠" would be asserting
something this pipeline deliberately refuses to assert, because a high value is ordinary
inside a scanner and alarming outside one, and the panel does not know which it is
looking at.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import mne

from eeg_pipeline.preprocessing.report.tables import Align, Column, grid_table

#: Tag identifying the panel, so a rebuild replaces it rather than adding a second one.
AT_A_GLANCE_TAG = "at-a-glance"

#: Section the panel occupies. Its own, because it belongs to no pipeline stage.
AT_A_GLANCE_SECTION = "At a glance"

AT_A_GLANCE_TITLE = "What this report measured"


def _count(value: Any) -> str:
    return f"{int(value):,}"


def _percentage(value: Any) -> str:
    return f"{float(value):.1%}"


def _decimal(value: Any) -> str:
    return f"{float(value):.2f}"


@dataclass(frozen=True)
class Headline:
    """One measurement worth meeting before the document is scrolled."""

    #: Key the owning stage records it under in the build record.
    key: str
    label: str
    #: Renders the recorded value into the form a reader expects to see it in.
    format: Callable[[Any], str]
    #: Section holding the evidence behind the number, named so the reader can go there.
    section: str


#: The measurements the landing panel carries, in reading order.
#:
#: Deliberately a short, fixed list rather than every key any stage records. A stage
#: records what a downstream decision might read, which is a wider set than what a human
#: needs in the first screen — ``samples_per_squared_component`` belongs beside the
#: reference value that gives it meaning, not on a summary. Adding a row here is a claim
#: that a reviewer should see the number before deciding where to look, and most
#: measurements do not clear that bar.
#:
#: A key absent from the record is a section that did not run: an EEG-only dataset has no
#: scanner stage and a resting-state one has no trial retention. The row is dropped rather
#: than rendered empty, so the panel shrinks to what was actually measured.
HEADLINES: tuple[Headline, ...] = (
    Headline("n_runs", "Runs", _count, "Data quality over time"),
    Headline("n_channels", "EEG channels", _count, "Channel and region coverage"),
    Headline("n_bad_channels", "Bad channels", _count, "Channel and region coverage"),
    Headline("n_components", "ICA components fitted", _count, "ICA decomposition quality"),
    Headline("n_excluded", "Components excluded", _count, "ICA decomposition quality"),
    Headline("retained_dimensions", "Dimensions left", _count, "ICA decomposition quality"),
    Headline(
        "variance_removed", "Sensor variance removed", _percentage, "ICA decomposition quality"
    ),
    Headline(
        "worst_marker_agreement",
        "Lowest per-run beat-marker agreement",
        _percentage,
        "Scanner artifact correction (Analyzer)",
    ),
    # "Epoch rejection", not "Trial retention": the latter is the title of a figure inside
    # this section, and a reader following it as a section name finds nothing. The cohort
    # panel reuses this list to point at its own sections, so the wrong name is a dead
    # pointer in both documents.
    Headline("epochs_kept", "Epochs retained", _count, "Epoch rejection"),
    Headline("epochs_total", "Epochs before rejection", _count, "Epoch rejection"),
    Headline("split_half_r", "Split-half reliability", _decimal, "Signal preservation"),
    Headline(
        "alpha_prominence_db", "Posterior alpha prominence (dB)", _decimal, "Signal preservation"
    ),
)


def latest_measurements(record: Mapping[str, Any]) -> dict[str, Any]:
    """Collapse the per-stage measurements into the report's current state.

    Later stages win. The record describes the report as it stands, so when a rebuild
    remeasured something the earlier value describes a section that is no longer in the
    document.

    Public because the QC sidecar carries the same collapsed view: a cohort reads one file
    per participant rather than reassembling a stage list, and what it reads has to be the
    same numbers this panel shows.
    """
    stages = sorted(
        (entry for entry in record.get("stages", ()) if isinstance(entry, Mapping)),
        key=lambda entry: str(entry.get("written_at", "")),
    )
    measurements: dict[str, Any] = {}
    for entry in stages:
        recorded = entry.get("measurements") or {}
        if isinstance(recorded, Mapping):
            measurements.update(recorded)
    return measurements


def _is_reportable(value: Any) -> bool:
    """Whether a recorded value can be rendered as a measurement.

    A non-finite float reaches the record when a measurement was attempted and did not
    resolve — an empty band, a division by a zero count. Printed, "nan" sits in the table
    looking like a value that was measured, so it is dropped and the row disappears with
    it, which is the same thing the panel does for a stage that never ran.
    """
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, (int, str))


def at_a_glance_html(record: Mapping[str, Any]) -> str:
    """Render the headline measurements, or an empty string when there are none."""
    measurements = latest_measurements(record)
    rows = []
    for headline in HEADLINES:
        if headline.key not in measurements:
            continue
        value = measurements[headline.key]
        if not _is_reportable(value):
            continue
        try:
            rendered = headline.format(value)
        except (TypeError, ValueError):
            # A stage recorded something of the wrong shape under a known key. The panel
            # is a summary and has no business failing the report over it; the owning
            # section still renders whatever it holds.
            continue
        rows.append([headline.label, rendered, headline.section])
    if not rows:
        return ""
    columns = (
        Column("Measurement", align=Align.TEXT),
        Column("Value"),
        Column("Section", align=Align.TEXT),
    )
    return (
        "<p>The measurements below were each made by the section named beside them and "
        "are reproduced here unchanged, so this panel cannot disagree with the evidence "
        "it points at. What a value means &mdash; the reference figures, the caveats, and "
        "what would explain an unusual one &mdash; is stated in that section rather than "
        "here.</p>"
        + grid_table(columns, rows)
        + "<p>A measurement that no stage recorded has no row: a dataset recorded outside a "
        "scanner has no marker agreement, and a resting-state recording has no trial "
        "retention. The panel is therefore as short as the pipeline that produced it.</p>"
    )


def add_at_a_glance_section(*, report: mne.Report, record: Mapping[str, Any]) -> None:
    """Put the landing panel at the front of the report, when there is one to put.

    Called from every save, so it runs against whatever object the caller is saving. A
    real :class:`mne.Report` records the panel and is reordered; anything that accepted
    the panel without recording it has nothing to move, and failing the save over the
    placement of a summary would lose the sections the stage actually produced. The panel
    is the content; being first is a courtesy to the reader.
    """
    from eeg_pipeline.preprocessing.report.organize import move_tagged_content_first

    document = at_a_glance_html(record)
    if not document:
        return
    report.add_html(
        html=document,
        title=AT_A_GLANCE_TITLE,
        section=AT_A_GLANCE_SECTION,
        tags=("summary", AT_A_GLANCE_TAG),
        replace=True,
    )
    content = getattr(report, "_content", None)
    if not isinstance(content, list):
        return
    if not any(AT_A_GLANCE_TAG in getattr(element, "tags", ()) for element in content):
        return
    move_tagged_content_first(report, tag=AT_A_GLANCE_TAG)


__all__ = [
    "AT_A_GLANCE_SECTION",
    "AT_A_GLANCE_TAG",
    "AT_A_GLANCE_TITLE",
    "HEADLINES",
    "Headline",
    "add_at_a_glance_section",
    "at_a_glance_html",
    "latest_measurements",
]
