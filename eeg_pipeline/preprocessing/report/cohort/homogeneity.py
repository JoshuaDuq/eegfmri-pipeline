"""Whether the cohort was preprocessed the same way, which is what makes pooling legitimate.

Every other section pools measurements across participants. That is only meaningful if the
measurements were made the same way, and over a study that runs for months they routinely
are not: a package is upgraded, a threshold is retuned, a montage gains a channel. The
resulting figures still look like cohort figures.

So this section reports agreement rather than measurement. Where every participant matches,
it says so in a line and gets out of the way. Where they do not, it names who differs and
on what, which is the only form in which that information is actionable.

No figures. Agreement is a set of exact values, and a plot of exact values is a table with
extra steps. The drift questions this could ask instead -- does a metric move with
acquisition date, does it move with pipeline version -- would need a trend read off a few
dozen points, which this report does not do anywhere else and will not start doing here.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import mne

from eeg_pipeline.preprocessing.report.cohort.collect import Cohort
from eeg_pipeline.preprocessing.report.tables import Align, Column, grid_table

HOMOGENEITY_SECTION = "Preprocessing homogeneity"
HOMOGENEITY_TITLE = "Was the cohort processed the same way"
HOMOGENEITY_TAG = "cohort-homogeneity"

#: Settings whose value changes the numbers the cohort pools, so a disagreement about one
#: is a disagreement about what the figures mean. Others are display choices and are left
#: out: a site that changed a colour limit did not change a measurement.
COMPARED_SETTINGS = (
    "continuity_window_seconds",
    "spectra_fmax",
    "spectra_line_frequency",
    "notch_exclusion_half_width_hz",
    "aperiodic_fit_range_hz",
    "aperiodic_exclude_hz",
    "alpha_band_hz",
    "alpha_reference_band_hz",
    # Selects the sensors the rhythm is measured over, so it changes the pooled
    # prominence as much as the band does. Configurable since the panel was written and
    # absent from this list until the rest of the acquisition settings joined it.
    "posterior_channel_pattern",
    "response_window_s",
    "plausible_heart_rate_bpm",
    "marker_agreement_tolerance_s",
)


def _disagreements(
    values_by_subject: Mapping[str, Mapping[str, Any]],
    keys: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Keys whose value is not the same for every participant that recorded one."""
    found: dict[str, dict[str, Any]] = {}
    for key in keys:
        seen = {
            subject: values[key]
            for subject, values in values_by_subject.items()
            if key in values and values[key] is not None
        }
        if len(set(map(_hashable, seen.values()))) > 1:
            found[key] = seen
    return found


def _hashable(value: Any) -> Any:
    """A comparable form of a setting, whatever depth JSON gave it back at.

    Recursive because a setting may be a list of pairs -- the detector-prose patterns are
    -- and a one-level conversion leaves the inner lists unhashable, which fails the
    comparison with a TypeError instead of reporting a disagreement.
    """
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    return value


def version_disagreements(cohort: Cohort) -> dict[str, dict[str, Any]]:
    """Packages whose version differs between participants."""
    by_subject = {
        participant.subject: dict(participant.versions) for participant in cohort.participants
    }
    packages = sorted({name for versions in by_subject.values() for name in versions})
    return _disagreements(by_subject, packages)


def setting_disagreements(cohort: Cohort) -> dict[str, dict[str, Any]]:
    """Measurement settings whose value differs between participants."""
    by_subject = {
        participant.subject: dict(participant.settings) for participant in cohort.participants
    }
    return _disagreements(by_subject, COMPARED_SETTINGS)


def _disagreement_table(found: Mapping[str, Mapping[str, Any]], *, what: str) -> str:
    rows = []
    for key, values in sorted(found.items()):
        grouped: dict[str, list[str]] = {}
        for subject, value in sorted(values.items()):
            grouped.setdefault(str(value), []).append(subject)
        for value, subjects in sorted(grouped.items()):
            rows.append([key, value, ", ".join(subjects)])
    columns = (
        Column(what, align=Align.TEXT, code=True),
        Column("Value", align=Align.TEXT, code=True),
        Column("Participants", align=Align.TEXT),
    )
    return grid_table(columns, rows)


def homogeneity_html(cohort: Cohort) -> str:
    """Render the agreement section."""
    versions = version_disagreements(cohort)
    settings = setting_disagreements(cohort)
    dates = sorted(
        participant.acquisition_date
        for participant in cohort.participants
        if participant.acquisition_date
    )

    parts = [
        "<p>Every other section pools measurements across participants, which is only "
        "meaningful if they were measured the same way. Over a study that runs for months "
        "they routinely are not, and the resulting figures still look like cohort "
        "figures.</p>"
    ]
    if dates:
        span = dates[0] if dates[0] == dates[-1] else f"{dates[0]} to {dates[-1]}"
        parts.append(
            f"<p>Recorded {span}, over {len(dates)} participant(s) carrying an acquisition "
            "date. That date indexes what changes in the recording &mdash; cap ageing, "
            "electrode wear, a replaced amplifier &mdash; and is deliberately not the "
            "processing date, which indexes what changes in the pipeline.</p>"
        )

    if not versions:
        parts.append(
            "<p><strong>Every participant was processed by the same package versions.</strong>"
            "</p>"
        )
    else:
        parts.append(
            "<p><strong>Package versions differ across the cohort.</strong> Numbers "
            "produced under different versions are not strictly comparable, which does not "
            "make them wrong: reprocessing part of a study deliberately is an ordinary "
            "thing to do, and only the reader knows whether this was that.</p>"
            + _disagreement_table(versions, what="Package")
        )

    if not settings:
        parts.append(
            "<p><strong>Every participant was measured under the same settings.</strong></p>"
        )
    else:
        parts.append(
            "<p><strong>Measurement settings differ across the cohort.</strong> These are "
            "the settings that decide what the numbers mean rather than how they are drawn, "
            "so a disagreement here is a disagreement about the figures themselves.</p>"
            + _disagreement_table(settings, what="Setting")
        )
    return "".join(parts)


def add_homogeneity_section(*, report: mne.Report, cohort: Cohort) -> None:
    """Add the agreement section to a cohort report."""
    report.add_html(
        html=homogeneity_html(cohort),
        title=HOMOGENEITY_TITLE,
        section=HOMOGENEITY_SECTION,
        tags=("summary", HOMOGENEITY_TAG),
        replace=True,
    )


__all__ = [
    "COMPARED_SETTINGS",
    "HOMOGENEITY_SECTION",
    "HOMOGENEITY_TAG",
    "HOMOGENEITY_TITLE",
    "add_homogeneity_section",
    "homogeneity_html",
    "setting_disagreements",
    "version_disagreements",
]
