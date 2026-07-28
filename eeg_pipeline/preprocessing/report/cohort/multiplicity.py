"""How often each participant is extreme, out of how many chances they had.

This report draws several dozen panels over a few dozen participants. Somebody is in the
outer decile of something by chance alone -- that is what an outer decile is -- and a
document that highlights extremes manufactures suspicion at exactly the rate the arithmetic
guarantees.

The answer is not to hide extremes but to give them their denominator. A participant who is
extreme on one metric out of thirty-one has done nothing; a participant who is extreme on
six of the eight scanner metrics and on nothing else has a scanner problem. That second
pattern is a finding and the first is noise, and the only thing that distinguishes them is
convergence within a mechanistically related family.

So the metrics are grouped by mechanism, and each participant gets a count per family with
the family size beside it. No cutoff is applied to the count, no score is derived from it,
and no participant is called an outlier: the table is a count with a denominator, and what
follows from it is the reader's.

Nothing is rendered below ten participants, where an outer decile does not exist: with nine
participants the tenth percentile lies outside the observed values, so every "extreme" would
be an artefact of asking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import mne
import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.report.cohort.aggregate import (
    DECILES,
    DEFAULT_GATES,
    BandGates,
    quantile_is_supported,
)
from eeg_pipeline.preprocessing.report.cohort.collect import Cohort
from eeg_pipeline.preprocessing.report.cohort.sidecar import SubjectSidecar
from eeg_pipeline.preprocessing.report.tables import Align, Column, grid_table

MULTIPLICITY_SECTION = "Cohort composition"
MULTIPLICITY_TITLE = "How often each participant is extreme"
MULTIPLICITY_TAG = "cohort-multiplicity"


@dataclass(frozen=True)
class MetricSource:
    """One metric, and where in a participant's sidecar to read it from."""

    key: str
    family: str
    read: Callable[[SubjectSidecar], float]


def _measurement(name: str) -> Callable[[SubjectSidecar], float]:
    def read(participant: SubjectSidecar) -> float:
        return _numeric(dict(participant.measurements).get(name))

    return read


def _run_median(column: str) -> Callable[[SubjectSidecar], float]:
    """The participant's median across runs, for a run-level scalar.

    A median rather than a mean, matching how every other run-to-participant reduction of a
    non-rate quantity is done in this package: each run is an independent estimate of the
    same thing.
    """

    def read(participant: SubjectSidecar) -> float:
        table = participant.runs
        if table.empty or column not in table.columns:
            return float("nan")
        values = pd.to_numeric(table[column], errors="coerce")
        values = values[np.isfinite(values)]
        return float(values.median()) if not values.empty else float("nan")

    return read


def _bad_channel_count(participant: SubjectSidecar) -> float:
    table = participant.channels
    if table.empty:
        return _numeric(dict(participant.measurements).get("n_bad_channels"))
    counts = pd.to_numeric(table["n_runs_bad"], errors="coerce").fillna(0)
    return float((counts > 0).sum())


#: The families, and what belongs in each.
#:
#: Membership is by mechanism rather than by which section drew the number. Flagged time
#: and bad channels are both "the sensors were not making good contact", so they converge
#: for the same physical reason and belong together; heart rate and marker agreement are
#: both "the beat detector was working", and are separate from the gradient metrics even
#: though both only exist inside a scanner.
CHANNEL_FAMILY = "Channel-level"
ICA_FAMILY = "ICA-level"
SCANNER_FAMILY = "Scanner-level"
PHYSIOLOGY_FAMILY = "Physiology-level"

METRIC_SOURCES: tuple[MetricSource, ...] = (
    MetricSource("n_bad_channels", CHANNEL_FAMILY, _bad_channel_count),
    MetricSource("flagged_fraction", CHANNEL_FAMILY, _run_median("flagged_fraction")),
    MetricSource("continuity_max_db", CHANNEL_FAMILY, _run_median("continuity_max_db")),
    MetricSource("n_components", ICA_FAMILY, _measurement("n_components")),
    MetricSource("n_excluded", ICA_FAMILY, _measurement("n_excluded")),
    MetricSource("variance_removed", ICA_FAMILY, _measurement("variance_removed")),
    MetricSource("retained_dimensions", ICA_FAMILY, _measurement("retained_dimensions")),
    MetricSource(
        "volume_locked_corrected_uv",
        SCANNER_FAMILY,
        _run_median("volume_locked_corrected_uv"),
    ),
    MetricSource("volume_jitter_s", SCANNER_FAMILY, _run_median("volume_jitter_s")),
    # Deliberately not the repetition time. It is a property of the sequence somebody
    # chose, not of how well the recording went, so a participant scanned under a second
    # protocol would be counted as extreme on the scanner family for having been scanned
    # differently -- a design fact wearing the clothes of a quality finding. The gradient
    # section reports repetition-time consistency directly, which is where that belongs.
    MetricSource("median_bpm", PHYSIOLOGY_FAMILY, _run_median("median_bpm")),
    MetricSource(
        "marker_matched_fraction", PHYSIOLOGY_FAMILY, _run_median("marker_matched_fraction")
    ),
    MetricSource("beat_dropouts", PHYSIOLOGY_FAMILY, _run_median("beat_dropouts")),
)

FAMILY_ORDER = (CHANNEL_FAMILY, ICA_FAMILY, SCANNER_FAMILY, PHYSIOLOGY_FAMILY)


def _numeric(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _outer_members(measured: Mapping[str, float]) -> set[str]:
    """The participants in the outer decile of one metric, by rank rather than by value.

    Scored against the participants that recorded this metric, not against the cohort. A
    metric only the in-scanner participants have is a metric the others did not sit in the
    outer decile of; they had no chance to, which is why they also do not count towards the
    denominator either.

    By rank, because a threshold test collapses on ties and this report is full of bounded
    counts that tie constantly. Comparing against the tenth percentile of a cohort where
    eight of twelve participants have no bad channels puts that percentile *at* zero, and
    every one of the eight is then "in the outer decile" -- ten of twelve participants
    flagged on a metric that separated nobody. Taking the ``ceil(0.1 * n)`` most extreme
    participants at each end bounds the count by construction, so the table can report a
    convergence rather than manufacturing one.

    Ties at the boundary are resolved by excluding the tied group rather than by admitting
    an arbitrary member of it: if the tenth and eleventh participants hold the same value,
    neither is more extreme than the other and picking one by sort order would be a
    statement about the order they were read in.
    """
    if len(measured) < 2:
        return set()
    values = sorted(measured.values())
    take = max(1, int(np.ceil(DECILES[0] * len(values))))
    return _tail(measured, boundary=values[take - 1], take=take, lower=True) | _tail(
        measured, boundary=values[-take], take=take, lower=False
    )


def _tail(
    measured: Mapping[str, float], *, boundary: float, take: int, lower: bool
) -> set[str]:
    """One end's members: everyone past the boundary, plus the tied group if it fits.

    Admitting a tied group only when all of it fits inside the budget is what keeps the
    count honest at both extremes. Half a tied group is an ordering artefact, and the whole
    of an oversized one is the threshold test this replaces.
    """
    beyond = {
        name
        for name, value in measured.items()
        if (value < boundary if lower else value > boundary)
    }
    tied = {name for name, value in measured.items() if value == boundary}
    return beyond | tied if len(beyond) + len(tied) <= take else beyond


@dataclass(frozen=True)
class Multiplicity:
    """Per participant and family: extremes observed, out of chances available."""

    #: Columns: subject, family, n_extreme, n_metrics.
    counts: pd.DataFrame
    #: Metrics that had enough contributors to be scored at all.
    scored_metrics: tuple[str, ...]

    @property
    def n_participants(self) -> int:
        return int(self.counts["subject"].nunique()) if not self.counts.empty else 0

    @property
    def families(self) -> tuple[str, ...]:
        if self.counts.empty:
            return ()
        found = set(self.counts["family"])
        return tuple(family for family in FAMILY_ORDER if family in found)


def multiplicity(
    cohort: Cohort,
    *,
    gates: BandGates = DEFAULT_GATES,
    sources: Sequence[MetricSource] = METRIC_SOURCES,
) -> Multiplicity | None:
    """Count each participant's outer-decile placements per family.

    ``None`` below the outer-band gate, where a decile is not supported by the sample and
    every placement would be an artefact of asking rather than a property of the data.
    """
    if cohort.n_participants < gates.min_subjects_for_outer_band:
        return None

    tallies: dict[tuple[str, str], dict[str, int]] = {}
    scored: list[str] = []
    for source in sources:
        measured = {
            participant.subject: source.read(participant)
            for participant in cohort.participants
        }
        measured = {name: value for name, value in measured.items() if np.isfinite(value)}
        if len(measured) < gates.min_subjects_for_outer_band:
            continue
        if not quantile_is_supported(DECILES[0], len(measured)):
            continue
        # A metric on which every contributor agrees has no outer decile to be in, and
        # scoring it would make everybody extreme on a measurement that separated nobody.
        outer = _outer_members(measured)
        if not outer or len(outer) == len(measured):
            continue
        scored.append(source.key)
        for subject in measured:
            entry = tallies.setdefault(
                (subject, source.family), {"n_extreme": 0, "n_metrics": 0}
            )
            entry["n_metrics"] += 1
            entry["n_extreme"] += int(subject in outer)

    if not tallies:
        return None
    counts = pd.DataFrame(
        [
            {
                "subject": subject,
                "family": family,
                "n_extreme": entry["n_extreme"],
                "n_metrics": entry["n_metrics"],
            }
            for (subject, family), entry in tallies.items()
        ]
    )
    return Multiplicity(
        counts=counts.sort_values(["subject", "family"]).reset_index(drop=True),
        scored_metrics=tuple(scored),
    )


def multiplicity_frame(cohort: Cohort, *, gates: BandGates = DEFAULT_GATES) -> pd.DataFrame | None:
    """The counts, for the audit table."""
    result = multiplicity(cohort, gates=gates)
    return None if result is None else result.counts.copy()


def multiplicity_table(result: Multiplicity) -> str:
    """One row per participant, one column per family, as ``extreme / available``."""
    families = result.families
    wide = {
        subject: {
            str(row["family"]): (int(row["n_extreme"]), int(row["n_metrics"]))
            for _, row in group.iterrows()
        }
        for subject, group in result.counts.groupby("subject")
    }
    totals = {
        subject: sum(count for count, _ in entries.values()) for subject, entries in wide.items()
    }
    rows = []
    for subject in sorted(wide, key=lambda name: (-totals[name], name)):
        entries = wide[subject]
        row: list[object] = [str(subject)]
        for family in families:
            found = entries.get(family)
            row.append(None if found is None else f"{found[0]} / {found[1]}")
        rows.append(row)
    columns = (Column("Participant", align=Align.TEXT, code=True),) + tuple(
        Column(family) for family in families
    )
    return grid_table(columns, rows)


def add_multiplicity_section(
    *,
    report: mne.Report,
    cohort: Cohort,
    gates: BandGates = DEFAULT_GATES,
) -> Multiplicity | None:
    """Add the convergence table, or nothing below the participant count it needs."""
    result = multiplicity(cohort, gates=gates)
    if result is None:
        return None

    document = (
        "<p>This report draws several dozen panels. Somebody is in the outer decile of "
        "something by chance alone &mdash; that is what an outer decile is &mdash; so a "
        "count of extremes without its denominator manufactures suspicion at exactly the "
        "rate the arithmetic guarantees.</p>"
        "<p>Each cell reads <em>extremes observed / metrics available</em> for one family "
        f"of mechanically related measurements, over {result.n_participants} participants "
        f"and {len(result.scored_metrics)} scored metrics. Being extreme on one metric of "
        "many is expected. Being extreme on most of one family is a pattern, because the "
        "metrics in a family fail together for a single physical reason.</p>"
        + multiplicity_table(result)
        + "<p>No cutoff is applied to any count and no participant is called an outlier. "
        "Each metric is scored against the participants that recorded it, so a participant "
        "who never had a chance at a metric is not counted as having passed it. A metric "
        "every participant agreed on is not scored at all: it has no outer decile, and "
        "scoring it would make everybody extreme on a measurement that separated "
        "nobody.</p>"
    )
    report.add_html(
        html=document,
        title=MULTIPLICITY_TITLE,
        section=MULTIPLICITY_SECTION,
        tags=(MULTIPLICITY_TAG,),
        replace=True,
    )
    return result


__all__ = [
    "CHANNEL_FAMILY",
    "FAMILY_ORDER",
    "ICA_FAMILY",
    "METRIC_SOURCES",
    "MULTIPLICITY_SECTION",
    "MULTIPLICITY_TAG",
    "MULTIPLICITY_TITLE",
    "PHYSIOLOGY_FAMILY",
    "SCANNER_FAMILY",
    "MetricSource",
    "Multiplicity",
    "add_multiplicity_section",
    "multiplicity",
    "multiplicity_frame",
    "multiplicity_table",
]
