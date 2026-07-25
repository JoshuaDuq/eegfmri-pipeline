"""Placement of appended review content inside an MNE report.

MNE appends new content to the end of a report, which for a review section means it
lands after the configuration and system-information footer, detached from the other
evidence about the same decision. Reordering relies on ``Report._content`` and
``Report.reorder``; keeping that coupling in one module means an MNE upgrade breaks one
place rather than every module that appends content.
"""

from __future__ import annotations

from typing import Callable

import mne


def _content_elements(report: mne.Report) -> list:
    content = getattr(report, "_content", None)
    if content is None:
        raise AttributeError(
            "This MNE version does not expose Report._content, which report section "
            "ordering depends on."
        )
    return list(content)


def move_tagged_content_before(
    report: mne.Report,
    *,
    tag: str,
    anchor: Callable[[object], bool],
) -> None:
    """Move every element carrying ``tag`` to sit just before the first anchor element.

    Missing tagged content is a programming error and raises. A missing anchor is not:
    which sections exist depends on which pipeline stages ran, so a resting-state or
    EEG-only report legitimately has no ICA or epochs section to sit in front of. In
    that case the content keeps its position rather than failing the whole report.
    """
    content = _content_elements(report)
    moving = [index for index, element in enumerate(content) if tag in element.tags]
    if not moving:
        raise ValueError(f"The report has no content tagged {tag!r} to place.")

    remaining = [index for index in range(len(content)) if index not in moving]
    anchors = [index for index in remaining if anchor(content[index])]
    if not anchors:
        return

    insertion = remaining.index(anchors[0])
    report.reorder(remaining[:insertion] + moving + remaining[insertion:])


def remove_tagged_content(report: mne.Report, *, tag: str) -> None:
    """Drop every element carrying ``tag`` before the section is rebuilt.

    ``add_figure(replace=True)`` matches on title, so renaming a panel leaves the
    previous one behind and an incrementally updated report accumulates stale
    duplicates. Clearing by tag keys the removal to something that does not change when
    a title is reworded.
    """
    for title in {element.name for element in _content_elements(report) if tag in element.tags}:
        report.remove(title=title, tags=(tag,), remove_all=True)


def move_tagged_content_first(report: mne.Report, *, tag: str) -> None:
    """Move every element carrying ``tag`` to the front of the report.

    Distinct from :func:`move_tagged_content_before`, which needs something to sit in
    front of. A minimal report can consist of the tagged content alone, and moving it
    ahead of nothing is a no-op rather than an error.
    """
    content = _content_elements(report)
    moving = [index for index, element in enumerate(content) if tag in element.tags]
    if not moving:
        raise ValueError(f"The report has no content tagged {tag!r} to place.")
    remaining = [index for index in range(len(content)) if index not in moving]
    report.reorder(moving + remaining)


def before_raw_sections(element: object) -> bool:
    """Match MNE's first raw section, so input-quality evidence sits ahead of it."""
    return str(element.section or "").startswith("Raw")


def before_epoch_sections(element: object) -> bool:
    """Match MNE's first epochs section, so rejection evidence sits beside it."""
    return str(element.section or "").startswith("Epochs")


def before_ica_component_review(element: object) -> bool:
    """Match the first element of MNE's own ICA component section or our review of it."""
    return "ica-component-review" in element.tags or element.section == "ICA: components"


__all__ = [
    "before_epoch_sections",
    "move_tagged_content_first",
    "before_raw_sections",
    "before_ica_component_review",
    "move_tagged_content_before",
    "remove_tagged_content",
]
