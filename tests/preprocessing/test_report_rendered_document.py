"""What a reader of the finished document sees, as opposed to what the archive records.

Every other report test reaches for ``Report._content``, which is the right handle for
most questions and the wrong one for this class of bug: MNE renders each element when it
is added and stores the markup, so ``element.name`` is metadata that a change can move
without moving the page. Two defects shipped through a passing suite that way in one
session -- a panel renamed in the ``.h5`` and unchanged in the ``.html``, and the
correction to it that then read the renamed metadata as proof there was nothing to do.

These tests assert on the rendered markup and on the saved HTML, so a change that agrees
with itself in the archive and disagrees with the page fails here.
"""

from __future__ import annotations

import re

import matplotlib
import mne

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from eeg_pipeline.preprocessing.report.organize import (  # noqa: E402
    SECTION_ORDER,
    name_cleaning_overlays_by_stage,
    order_sections,
)

#: MNE wraps every panel heading in this anchor, whatever the panel holds.
_HEADING = re.compile(r'class="text-decoration-none"[^>]*>\s*(.*?)\s*</a>', re.S)


def rendered_heading(element: object) -> str | None:
    """The heading text MNE baked into one element when it was added."""
    match = _HEADING.search(str(element.html))
    return None if match is None else match.group(1)


def assert_headings_match_names(report: mne.Report) -> None:
    """Every panel's rendered heading says what its metadata says it says.

    The invariant that catches the whole class. A change that edits ``name`` without the
    stored markup leaves the document reading one thing and the archive recording
    another, and nothing downstream -- the table of contents, a reader, a later pass
    keyed on the name -- can then agree about what the panel is called.
    """
    mismatched = [
        (element.name, rendered_heading(element))
        for element in report._content
        if rendered_heading(element) is not None
        and rendered_heading(element) != str(element.name)
    ]
    assert not mismatched, f"panel metadata and rendered heading disagree: {mismatched}"


def _panel(report: mne.Report, title: str, section: str | None) -> None:
    report.add_html(html="<p>evidence</p>", title=title, section=section, tags=("x",))


def test_renaming_a_panel_keeps_the_page_and_the_archive_agreeing() -> None:
    """The bug this module exists for, in its simplest form."""
    report = mne.Report(title="ica", verbose="ERROR")
    figure = plt.figure()
    for section in ("ICA: components", "ICA: removals"):
        report.add_figure(
            fig=figure, title="Original and cleaned signal", section=section, tags=("ica",)
        )
    plt.close(figure)

    name_cleaning_overlays_by_stage(report)

    assert_headings_match_names(report)


def test_a_rename_that_touched_only_the_metadata_is_caught() -> None:
    """Proof the invariant has teeth: the exact mistake, asserted to fail."""
    report = mne.Report(title="ica", verbose="ERROR")
    _panel(report, "Original and cleaned signal", "ICA: removals")
    report._content[0].name = "Something else entirely"

    try:
        assert_headings_match_names(report)
    except AssertionError:
        return
    raise AssertionError("the invariant did not notice a metadata-only rename")


def test_no_two_panels_in_one_section_share_a_heading() -> None:
    """A reader cannot tell two identically-titled panels apart, and a later pass keyed
    on the title cannot either."""
    report = mne.Report(title="ica", verbose="ERROR")
    figure = plt.figure()
    for section in ("ICA: components", "ICA: removals"):
        report.add_figure(
            fig=figure, title="Original and cleaned signal", section=section, tags=("ica",)
        )
    plt.close(figure)
    name_cleaning_overlays_by_stage(report)

    seen: dict[tuple[str, str], int] = {}
    for element in report._content:
        heading = rendered_heading(element)
        if heading is None:
            continue
        key = (str(element.section or ""), heading)
        seen[key] = seen.get(key, 0) + 1
    assert not [key for key, count in seen.items() if count > 1]


def test_the_saved_html_lists_its_sections_in_the_declared_order(tmp_path) -> None:
    """The order is asserted elsewhere on ``_content``; this asserts it on the page.

    MNE renders a section where its first element sits and merges the rest into it, so
    the contents a reader scrolls is a separate artefact from the list the sort produced.
    """
    report = mne.Report(title="subject", verbose="ERROR")
    written = ["Epochs (clean)", "Signal preservation", "Raw (original)", "At a glance"]
    for section in written:
        _panel(report, f"panel in {section}", section)

    order_sections(report)
    path = tmp_path / "sub-0001_report.html"
    report.save(path, overwrite=True, open_browser=False)

    document = path.read_text(encoding="utf-8")
    start = document.find("Table of contents")
    navigation = document[start : document.find("</nav>", start)]
    listed = re.findall(r'href="#[^"]+"[^>]*>([^<]+)</a>', navigation)
    assert listed, "the saved document has no table of contents to check"
    ranked = [name for name in SECTION_ORDER if name in written]
    assert [name for name in listed if name in written] == ranked


def test_the_saved_html_carries_the_renamed_headings(tmp_path) -> None:
    """End to end: rename, save, and read the page back."""
    report = mne.Report(title="ica", verbose="ERROR")
    figure = plt.figure()
    for section in ("ICA: components", "ICA: removals"):
        report.add_figure(
            fig=figure, title="Original and cleaned signal", section=section, tags=("ica",)
        )
    plt.close(figure)
    name_cleaning_overlays_by_stage(report)

    path = tmp_path / "sub-0001_report.html"
    report.save(path, overwrite=True, open_browser=False)
    document = path.read_text(encoding="utf-8")

    assert ">Original and cleaned signal</a>" not in document
    for stage in ("exclusions proposed at fitting", "exclusions as applied"):
        assert f">Original and cleaned signal ({stage})</a>" in document
