"""Finding the participants a cohort report is about, and saying who was left out.

:mod:`sidecar` is deliberately strict: a sidecar written under a layout this code does not
understand, or describing a participant other than the one in its path, is raised rather
than absorbed, because either would contribute a wrong row to a cohort figure that looks
exactly like a right one.

Strictness is the reader's job. Policy is this module's. One stale sidecar out of forty
should not cost the reader the other thirty-nine, so a participant that cannot be read is
recorded as not aggregated together with the reason, and the run continues. The reason then
appears in the composition table, where it is a fact the reader can act on, rather than a
traceback they have to interpret.

The rule that follows from this, and that the rest of the package depends on: a participant
is either aggregated or listed with a reason. There is no third path in which somebody
quietly disappears and a denominator silently shrinks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from eeg_pipeline.preprocessing.report.cohort.sidecar import (
    Paradigm,
    SubjectSidecar,
    has_sidecar,
    read_sidecar,
)

#: Name of the report a sidecar sits beside, and therefore what discovery looks for.
_REPORT_GLOB = "sub-*/eeg/sub-*_report.h5"


@dataclass(frozen=True)
class NotAggregated:
    """A participant the cohort found and could not use, with why.

    Carried into the document rather than logged. A reader deciding whether a cohort
    figure describes their study needs to know that six participants were discovered and
    two of them were skipped for a stale sidecar -- a fact that changes what the other
    thirty-eight mean, and that no denominator on its own conveys.
    """

    subject: str
    reason: str


@dataclass(frozen=True)
class Cohort:
    """The participants a cohort document is built from, and those it is not."""

    participants: tuple[SubjectSidecar, ...]
    not_aggregated: tuple[NotAggregated, ...] = ()

    def __post_init__(self) -> None:
        subjects = [participant.subject for participant in self.participants]
        duplicated = sorted({name for name in subjects if subjects.count(name) > 1})
        if duplicated:
            raise ValueError(
                f"A participant appears twice in the cohort and would be weighted twice: "
                f"{', '.join(duplicated)}."
            )

    @property
    def subjects(self) -> tuple[str, ...]:
        return tuple(participant.subject for participant in self.participants)

    @property
    def n_participants(self) -> int:
        return len(self.participants)

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(sorted({participant.task for participant in self.participants}))


    @property
    def paradigms(self) -> tuple[Paradigm, ...]:
        found = {participant.paradigm for participant in self.participants}
        return tuple(paradigm for paradigm in Paradigm if paradigm in found)


    def select(
        self,
        *,
        paradigm: Paradigm | None = None,
    ) -> "Cohort":
        """The sub-cohort a section applies to.

        Returns a cohort rather than a list so that a section's denominator comes from the
        same type as the document's, and cannot be computed a different way by accident.
        """
        chosen = tuple(
            participant
            for participant in self.participants
            if paradigm is None or participant.paradigm is paradigm
        )
        return Cohort(participants=chosen, not_aggregated=self.not_aggregated)



def discover_reports(
    deriv_eeg_root: Path | str,
    *,
    subjects: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Find the subject reports under a derivatives root.

    ``subjects`` restricts the search. A requested participant that does not exist is an
    error rather than an absence: a typo in a subject list would otherwise produce a
    smaller cohort that is indistinguishable from a correct one.
    """
    root = Path(deriv_eeg_root)
    if not root.is_dir():
        raise NotADirectoryError(f"No derivatives root at {root}.")
    found: dict[str, Path] = {}
    for path in sorted(root.glob(_REPORT_GLOB)):
        subject = _subject_of(path)
        if subject is not None:
            found[subject] = path
    if subjects is None:
        return found
    wanted = [str(subject) for subject in subjects]
    missing = sorted(set(wanted) - set(found))
    if missing:
        raise ValueError(
            f"No subject report was found for {', '.join(missing)} under {root}. "
            f"Discovered: {', '.join(sorted(found)) or 'none'}."
        )
    return {subject: found[subject] for subject in sorted(wanted)}


def collect_cohort(
    deriv_eeg_root: Path | str,
    *,
    task: str | None = None,
    subjects: Sequence[str] | None = None,
) -> Cohort:
    """Load every participant that can be aggregated, and account for those that cannot.

    ``task`` restricts the cohort to one task. Left unset, whatever is on disk is
    collected and :attr:`Cohort.tasks` reports what that turned out to be; naming the
    document is the caller's decision and not this function's.
    """
    reports = discover_reports(deriv_eeg_root, subjects=subjects)
    participants: list[SubjectSidecar] = []
    skipped: list[NotAggregated] = []
    for subject, report in reports.items():
        if not has_sidecar(report):
            skipped.append(
                NotAggregated(
                    subject=subject,
                    reason=(
                        "No QC sidecar beside the report. Re-run the subject report stage "
                        "to write one."
                    ),
                )
            )
            continue
        try:
            loaded = read_sidecar(report)
        except (ValueError, FileNotFoundError, KeyError) as error:
            skipped.append(NotAggregated(subject=subject, reason=str(error)))
            continue
        if task is not None and loaded.task != task:
            skipped.append(
                NotAggregated(
                    subject=subject,
                    reason=f"Recorded under task {loaded.task!r} rather than {task!r}.",
                )
            )
            continue
        participants.append(loaded)

    if not participants:
        raise ValueError(
            f"No participant under {deriv_eeg_root} could be aggregated"
            + (f" for task {task!r}" if task else "")
            + ". "
            + _skipped_summary(skipped)
        )
    return Cohort(participants=tuple(participants), not_aggregated=tuple(skipped))


def _skipped_summary(skipped: Iterable[NotAggregated]) -> str:
    entries = list(skipped)
    if not entries:
        return "No subject reports were discovered at all."
    return "Skipped: " + "; ".join(f"{entry.subject} ({entry.reason})" for entry in entries)


def _subject_of(report_path: Path) -> str | None:
    for part in report_path.name.split("_"):
        if part.startswith("sub-"):
            return part[len("sub-") :]
    return None


__all__ = [
    "Cohort",
    "NotAggregated",
    "collect_cohort",
    "discover_reports",
]
