"""The document's phases, and the section order they imply.

The order was a flat tuple whose grouping lived in comments, so the grouping could not be
rendered and could drift from the order beside it. Deriving the order from the phases
leaves one source for both.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Phase:
    title: str
    #: Sections belonging to this phase, in document order. Membership says where a
    #: section goes if it is present, never that it will be: the cardiac review, the
    #: exploratory band ICAs and the event evidence each depend on configuration or on
    #: what was recorded.
    sections: tuple[str, ...]


PHASES: tuple[Phase, ...] = (
    Phase(
        "What this report says",
        ("At a glance", "Configuration", "Filter response"),
    ),
    Phase(
        "What came in",
        (
            "Channel and region coverage",
            "Data quality over time",
            "Raw (original)",
            "Raw (filtered)",
        ),
    ),
    Phase(
        "What the decomposition did",
        (
            "ICA: epochs for fitting",
            "ICA decomposition quality",
            "ICA cardiac artifact review",
            "Cardiac rhythm",
            "ICA ocular artifact review",
            "ICA component review",
            "ICA: components",
            "ICA: removals",
            "Exploratory band-fitted ICAs",
        ),
    ),
    Phase(
        "Whether the cleaning worked",
        ("Sensor spectra before and after ICA",),
    ),
    Phase(
        "What survived",
        (
            "Events",
            "Epoch rejection",
            "Signal preservation",
            "Epochs (before cleaning)",
            "Epochs (clean)",
            "Raw (clean)",
        ),
    ),
)

#: The order sections appear in, as a reading order rather than a build order.
#:
#: Matched as prefixes, so ``ICA component review`` covers the per-band sections whose
#: names carry the band. A section matching nothing is appended in the order it was
#: added, which is what a new stage gets until it is named here.
#:
#: The document previously had no declared order at all. Each stage moved its own content
#: relative to whatever happened to exist when it ran, so the shape of the report depended
#: on which stages ran and in what sequence -- visible in the archive as sections
#: occupying two separate positions each, and as the events panel rendering behind the
#: trial counts it describes because it was placed last.
SECTION_ORDER: tuple[str, ...] = tuple(section for phase in PHASES for section in phase.sections)


def phases_present(section_titles: Iterable[str]) -> tuple[Phase, ...]:
    # Matched by prefix, as section ranking is: MNE appends a qualifier to some titles,
    # so "ICA component review: Broadband 1-100 Hz" has to reach "ICA component review".
    titles = tuple(section_titles)
    return tuple(
        phase
        for phase in PHASES
        if any(title.startswith(section) for section in phase.sections for title in titles)
    )


__all__ = ["PHASES", "SECTION_ORDER", "Phase", "phases_present"]
