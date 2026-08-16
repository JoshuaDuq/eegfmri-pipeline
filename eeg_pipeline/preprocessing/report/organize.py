"""Placement of appended review content inside an MNE report.

MNE appends new content to the end of a report, which for a review section means it
lands after the configuration and system-information footer, detached from the other
evidence about the same decision. Reordering relies on ``Report._content`` and
``Report.reorder``; keeping that coupling in one module means an MNE upgrade breaks one
place rather than every module that appends content.
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Callable

import mne

from eeg_pipeline.preprocessing.report.style import apply_report_css


#: Title MNE gives the per-epoch metadata table it renders inside an epochs section.
_METADATA_TABLE_TITLE = "Metadata"

#: Panel MNE draws for a raw recording: a butterfly of every channel over a few seconds.
_RAW_TIME_SERIES_TITLE = "Time series"

#: Panel MNE draws for a spectrum, in every section that has one.
_SPECTRUM_TITLE = "PSD"

#: Section MNE's own ICA panels land in.
_ICA_COMPONENTS_SECTION = "ICA: components"

#: The ``ICA: components`` panels the per-run cardiac review supersedes.
#:
#: Deliberately not the whole section. "Original and cleaned signal" overlays the raw
#: traces either side of the exclusions, which nothing else in this report draws, so it
#: is kept while the two ECG-specific panels beside it go.
_REPLACED_ICA_ECG_TITLES = (
    "Scores for matching ECG patterns",
    "Original and cleaned ECG epochs",
)

#: The ``ICA: components`` panels the per-run ocular review supersedes.
#:
#: The same two quantities as the ECG pair above, for the other artifact, and dropped for
#: the same reason: the ocular review draws the EOG correlation for every component across
#: every run, and the blink-locked overlay either side of the exclusions, both per run and
#: both beside the blink counts that say whether to believe them. MNE draws its versions
#: for one concatenated recording with no indication of how many blinks were behind them.
#:
#: These were previously left in place while the ECG pair went, which was an oversight
#: rather than a judgement: it left a reviewer meeting the ocular evidence twice and the
#: cardiac evidence once.
_REPLACED_ICA_EOG_TITLES = (
    "Scores for matching EOG patterns",
    "Original and cleaned EOG epochs",
)

#: Panels MNE-BIDS-Pipeline writes that the authoritative component review supersedes.
#:
#: ``ICA component properties`` is one ``plot_properties`` figure per component, which the
#: component dossier renders itself with the ICLabel verdict on each slide; ``ICA component
#: topographies`` is the grid ``plot_component_overview`` replaced. Keeping both costs
#: duplicated pictures and gives a reviewer two places to look for one answer.
#:
#: Deliberately absent: MNE's ``Info`` block, its original-versus-cleaned overlays, and its
#: score panel. Nothing in this pipeline reproduces those.
_SUPERSEDED_MNE_ICA_PANELS = (
    "ICA component properties",
    "ICA component topographies",
)

#: Prefix of the MNE-ICALabel panels the authoritative review supersedes.
#:
#: Spans the per-class topography grids and ``ICALabel: report``, the numeric table of
#: per-class probabilities. Two things carry those numbers: the exclusion ledger states
#: each component's decision, its deciding detector and its variance cost, and every
#: dossier draws the full ICLabel class distribution as a stacked bar.
_SUPERSEDED_ICLABEL_GRID_PREFIX = "ICALabel: "

#: Tags marking the content that stands in for the panels above.
#:
#: ``ica-component-review`` is the per-component evidence and ``ica-decomposition`` the
#: whole-decomposition evidence. Either one present means the authoritative review is in
#: this document and MNE's version of it is a duplicate.
_AUTHORITATIVE_REVIEW_TAGS = frozenset({"ica-component-review", "ica-decomposition"})

#: Section MNE-BIDS-Pipeline puts its per-run bad-channel items in.
#:
#: Matched as a prefix, which also spans "Data quality over time". Nothing in that
#: section carries one of the titles below, and a future panel there that did would be
#: naming itself after the very evidence this removal replaces.
_DATA_QUALITY_SECTION = "Data quality"

#: Titles MNE-BIDS-Pipeline repeats once per run, each with a run entity appended.
_REPLACED_BAD_CHANNEL_TITLES = ("Bad channels", "Bad channel detection")

#: Tag MNE-BIDS-Pipeline puts on its events panel.
#:
#: Keyed on the tag rather than the "Events" title because a title is prose and gets
#: reworded, while this tag is what the rest of the document already filters on.
_EVENTS_TAG = "events"

#: Section the events panel is given, which MNE-BIDS-Pipeline leaves unset.
_EVENTS_SECTION = "Events"


def drop_per_epoch_metadata_tables(report: mne.Report) -> None:
    """Remove MNE's per-epoch metadata tables from the report.

    One row per epoch and one column per marker type, rendered once for the ICA fitting
    epochs and again for the task epochs. It is a data dump rather than evidence: it grows
    with the trial count, the same frame is written beside the report as ``_events.tsv``
    for anything that needs the numbers, and the questions a reviewer asks of it — how many
    trials survived, and in which condition — are what the trial-retention panel answers.

    Unlike the ICA panels this pipeline replaces, nothing here stands in for the table, so
    the judgement is that a browser is the wrong place to read it rather than that it is
    duplicated. The other panels of the epochs section are untouched.
    """
    if any(element.name == _METADATA_TABLE_TITLE for element in _content_elements(report)):
        report.remove(title=_METADATA_TABLE_TITLE, remove_all=True)


def drop_replaced_panels(
    report: mne.Report,
    *,
    section_prefix: str,
    titles: Collection[str] = (),
    title_prefixes: Collection[str] = (),
) -> None:
    """Drop panels from every section whose name starts with ``section_prefix``.

    ``Report.remove`` keys on title alone and searches the whole document. MNE reuses a
    handful of titles — "Time series", "PSD", "Info" — in every section it builds, so
    removing a raw butterfly by title would also take the spectrum out of the epochs
    section. Scoping the match to a section is what makes the removal say what it means.

    ``titles`` matches exactly. ``title_prefixes`` exists because MNE-BIDS-Pipeline
    appends a run entity to the titles it repeats per run, so the set of titles to remove
    is not known until the data are read.

    A missing panel is not an error. Which sections exist depends on which stages ran,
    and a stage that prunes what an earlier stage did not add has nothing to do.
    """
    exact = set(titles)
    prefixes = tuple(title_prefixes)

    def is_replaced(element: object) -> bool:
        if not str(element.section or "").startswith(section_prefix):
            return False
        name = str(element.name or "")
        if name in exact:
            return True
        return bool(prefixes) and name.startswith(prefixes)

    report._content = [element for element in _content_elements(report) if not is_replaced(element)]


def drop_replaced_raw_time_series(report: mne.Report) -> None:
    """Drop MNE's raw butterfly panels, which the time-resolved quality section replaces.

    The butterfly draws every channel unlabelled over a few seconds, with no amplitude
    scale, repeated for a handful of arbitrary segments of the recording. It is the panel
    a reviewer would reach for to answer "was this run clean throughout", and it cannot
    answer that: the segments are not chosen for being informative, and a 63-channel
    overlay hides the single misbehaving sensor that the question is about.

    ``Amplitude over time by channel`` answers it directly — every channel, every second
    of the run, relative to that channel's own median — so the butterfly is removed by
    the section that adds the replacement rather than document-wide. Running the
    continuity stage alone must not leave a report with neither panel.
    """
    drop_replaced_panels(
        report,
        section_prefix="Raw",
        titles=(_RAW_TIME_SERIES_TITLE,),
    )


def drop_replaced_filtered_spectrum(report: mne.Report) -> None:
    """Drop the filtered raw spectrum, which the per-run sensor spectra replace.

    MNE draws the spectrum to Nyquist. On a 500 Hz recording low-passed at 100 Hz that
    spends four fifths of the axis on filter roll-off falling into the noise floor, and
    compresses the band anyone is reading into the left fifth. The sensor-spectra section
    draws the same band per run over 1-100 Hz, with across-channel percentile bands, the
    aperiodic fit, and markers on the line-noise and gradient harmonics.

    Only the filtered one. The original raw spectrum is the report's single view of the
    data before filtering, and there the full bandwidth is the point: it is where the
    anti-alias corner and the gradient harmonics above the low-pass are visible.
    """
    drop_replaced_panels(
        report,
        section_prefix="Raw (filtered)",
        titles=(_SPECTRUM_TITLE,),
    )


def drop_replaced_ica_ecg_panels(report: mne.Report) -> None:
    """Drop the ``ICA: components`` ECG panels that the cardiac review supersedes.

    MNE renders the ECG match scores and the ECG epoch overlay for one concatenated
    recording, with no indication of whether the beats behind them were detected well.
    The cardiac review renders both per run, beside the detected R peaks and the
    beat-to-beat intervals that say whether to believe them, so keeping MNE's version
    means a reviewer meets the same quantity twice and has to work out which to trust.
    """
    drop_replaced_panels(
        report,
        section_prefix=_ICA_COMPONENTS_SECTION,
        titles=_REPLACED_ICA_ECG_TITLES,
    )


def drop_replaced_ica_eog_panels(report: mne.Report) -> None:
    """Drop the ``ICA: components`` EOG panels that the ocular review supersedes.

    The counterpart of :func:`drop_replaced_ica_ecg_panels`, and called from the ocular
    review for the same reason that one is called from the cardiac review: the removal
    belongs with the section that renders the replacement, so a run of one review alone
    cannot take a panel away and leave nothing behind it.
    """
    drop_replaced_panels(
        report,
        section_prefix=_ICA_COMPONENTS_SECTION,
        titles=_REPLACED_ICA_EOG_TITLES,
    )


#: ``Raw (clean)`` panels, each paired with the tag of the section that replaces it.
#:
#: ``drop_replaced_raw_time_series`` already spans this section by prefix, but it runs
#: from the continuity stage and MNE-BIDS-Pipeline writes ``Raw (clean)`` afterwards, when
#: it applies the ICA. The panels came back after their replacement had been added and
#: stayed, which is the same recurrence that kept MNE's ICA panels alive.
#:
#: The spectrum is handled here rather than by ``drop_replaced_filtered_spectrum``, which
#: deliberately keeps the *original* raw spectrum: the full bandwidth before filtering is
#: the report's one view of the anti-alias corner and the gradient harmonics above the
#: low-pass. After cleaning, nothing is left on that axis that the sensor-spectra section
#: does not draw over the band a reader is actually reading.
_REPLACED_CLEAN_RAW_PANELS = (
    (_RAW_TIME_SERIES_TITLE, "run-continuity"),
    (_SPECTRUM_TITLE, "sensor-spectra"),
)


def drop_replaced_clean_raw_panels(report: mne.Report) -> None:
    """Drop each ``Raw (clean)`` panel whose replacement is in this report.

    Per panel rather than per section, because the two replacements are added by
    different stages: a report carrying the continuity section but not the sensor spectra
    should lose the butterfly and keep the spectrum, not lose both or neither.
    """
    present = {tag for element in _content_elements(report) for tag in element.tags}
    replaced = tuple(title for title, tag in _REPLACED_CLEAN_RAW_PANELS if tag in present)
    if not replaced:
        return
    drop_replaced_panels(report, section_prefix="Raw (clean)", titles=replaced)


def drop_replaced_per_run_bad_channels(report: mne.Report) -> None:
    """Drop the one-item-per-run bad-channel panels the coverage table replaces.

    MNE-BIDS-Pipeline adds one accordion item per run, each holding a single list. A
    six-run session therefore spends six table-of-contents entries and six clicks to say
    "none" six times, and never puts two runs on one screen — which is the comparison
    that matters, because a channel bad in every run is a montage fact while one bad in a
    single run is something that happened during the session.
    """
    drop_replaced_panels(
        report,
        section_prefix=_DATA_QUALITY_SECTION,
        title_prefixes=_REPLACED_BAD_CHANNEL_TITLES,
    )


#: The order sections appear in, as a reading order rather than a build order.
#:
#: Matched as prefixes, so ``ICA component review`` covers the per-band sections whose
#: names carry the band. A section matching nothing is appended in the order it was
#: added, which is what a new stage gets until it is named here.
#:
#: The document previously had no declared order at all. Each stage moved its own content
#: relative to whatever happened to exist when it ran, so the shape of the report depended
#: on which stages ran and in what sequence -- visible in the archive as the Analyzer and
#: Configuration sections occupying two separate positions each, and as the events panel
#: rendering behind the trial counts it describes because it was placed last.
#:
#: The sequence: what the run was, what came in, what the decomposition did, what survived.
SECTION_ORDER = (
    # What this report says, and what produced it.
    "At a glance",
    "Configuration",
    "Filter response",
    # What came in, and what the upstream correction left in it.
    "Scanner artifact correction (Analyzer)",
    "Channel and region coverage",
    "Residual scanner gradient",
    "Data quality over time",
    "Raw (original)",
    "Raw (filtered)",
    # The decomposition: the whole of it, then each detector, then each component.
    "ICA: epochs for fitting",
    "ICA decomposition quality",
    "ICA cardiac artifact review",
    "ICA ocular artifact review",
    "ICA component review",
    "ICA: components",
    "ICA: removals",
    # Whether the cleaning worked.
    "Sensor spectra before and after ICA",
    # What was presented, what survived, and whether the survivors carry signal.
    "Events",
    "Epoch rejection",
    "Signal preservation",
    "Epochs (before cleaning)",
    "Epochs (clean)",
    "Raw (clean)",
)


def _section_rank(element: object) -> int:
    """Where an element's section sits in :data:`SECTION_ORDER`.

    Sectionless content sorts last. MNE leaves the configuration file and the system
    information without a section, and both are reference material a reader reaches for
    after the evidence rather than before it.
    """
    section = str(element.section or "")
    if not section:
        return len(SECTION_ORDER) + 1
    for rank, name in enumerate(SECTION_ORDER):
        if section.startswith(name):
            return rank
    return len(SECTION_ORDER)


def order_sections(report: mne.Report) -> None:
    """Put the document in :data:`SECTION_ORDER`, keeping each section's own order.

    A stable sort on section rank alone: within a section the panels keep the order the
    stage that wrote them chose, because that ordering is a judgement about one body of
    evidence and this function has no view on it.

    MNE renders a section wherever its first element sits and merges later elements into
    it, so sorting the content by section is what makes the rendered contents match this
    list -- and it also gathers the sections that were previously split across two
    positions in the archive.
    """
    content = _content_elements(report)
    order = sorted(range(len(content)), key=lambda index: (_section_rank(content[index]), index))
    if order != list(range(len(content))):
        report.reorder(order)


def drop_superseded_mne_ica_panels(report: mne.Report) -> None:
    """Drop MNE's ICA panels, but only from a report carrying the review that replaces them.

    The guard is what makes this callable from anywhere, including from
    :func:`open_subject_report`. Without it the function could only be called from the
    code adding the replacements, and that is what let the duplicates survive: MNE-BIDS-
    Pipeline rewrites these panels every time ``_08a_apply_ica`` runs, while the last
    stage that dropped them was reached only when ``ica.band_specific_report.comparisons``
    was configured. A dataset with no condition contrasts -- which is most of them, and is
    what the shipped ``eeg_only`` preset sets -- therefore kept both copies forever, and
    turning the expensive review *off* produced the more duplicated document of the two.

    Keying the guard to the replacement rather than to configuration also preserves the
    rule the rest of this module follows: a report can never end up with the panel removed
    and nothing in its place, whichever subset of stages ran.
    """
    content = _content_elements(report)
    if not any(_AUTHORITATIVE_REVIEW_TAGS & set(element.tags) for element in content):
        return
    titles = {
        element.name
        for element in content
        if element.name in _SUPERSEDED_MNE_ICA_PANELS
        or str(element.name or "").startswith(_SUPERSEDED_ICLABEL_GRID_PREFIX)
    }
    for title in titles:
        report.remove(title=title, remove_all=True)


def place_events_with_epochs(report: mne.Report) -> None:
    """Give the events panel a section and move it in front of the epochs it describes.

    MNE-BIDS-Pipeline adds this panel with no ``section``, and MNE only wraps sectioned
    content in a section element. The panel therefore rendered as a bare accordion item —
    the one item in the document with no heading around it — and, being appended last, it
    landed behind every review section this pipeline adds. In a report carrying the
    exploratory band ICAs that is tens of megabytes of sliders between the trial counts
    and the panel explaining where those trials came from.

    A report with no events panel is not an error. Resting-state recordings have no
    events, and a report built from the bad-channel stage alone has no epochs section to
    sit in front of either; in both cases the report is left as it is.
    """
    content = _content_elements(report)
    events = [element for element in content if _EVENTS_TAG in element.tags]
    if not events:
        return
    for element in events:
        element.section = _EVENTS_SECTION
    move_tagged_content_before(report, tag=_EVENTS_TAG, anchor=before_trial_evidence)


def open_subject_report(report_path: Path | str) -> mne.Report:
    """Open an existing subject report and apply the document-wide policies to it.

    Every review stage reopens the same report to append its section, so wrapping
    ``mne.open_report`` is the one place that cannot be forgotten when a stage is added.
    Doing it at each call site instead left whichever stage saved last deciding whether
    the report had a stylesheet.

    Only policies that hold regardless of which stage is running belong here. Dropping
    content this pipeline renders a *replacement* for does not: that has to happen in the
    function adding the replacement, or a run of one stage alone removes a panel and puts
    nothing in its place.
    """
    report = mne.open_report(report_path)
    apply_report_css(report)
    drop_per_epoch_metadata_tables(report)
    # Guarded on the replacement being present, so this is a document-wide policy rather
    # than a removal that depends on which stage is running. It has to be re-applied on
    # every reopen because MNE-BIDS-Pipeline rewrites the panels it drops each time
    # ``_08a_apply_ica`` runs, which is after the stages that add their replacements.
    drop_superseded_mne_ica_panels(report)
    # Same recurrence, same treatment: MNE-BIDS-Pipeline writes "Raw (clean)" when it
    # applies the ICA, after the continuity and sensor-spectra sections that replace its
    # panels have been added.
    drop_replaced_clean_raw_panels(report)
    place_events_with_epochs(report)
    return report


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


def before_trial_evidence(element: object) -> bool:
    """Match the first panel that counts trials, or the epochs section itself.

    Three sections anchor on :func:`before_epoch_sections` — the events panel, epoch
    rejection, and signal preservation — so whichever moves last ends up nearest the
    epochs and the other two sit ahead of it. The events panel is placed on every reopen
    and therefore always moved last, which put "how many trials survived" and "did signal
    survive" in front of the panel saying what the trials were.

    Anchoring the events panel here instead puts it ahead of both, which is the reading
    order: what was presented, then what survived, then whether the survivors carry
    signal.
    """
    tags = set(element.tags)
    return (
        bool(tags & {"epoch-rejection", "signal-preservation"})
        or str(element.section or "").startswith("Epochs")
    )


def before_ica_component_review(element: object) -> bool:
    """Match the first element of MNE's own ICA component section or our review of it."""
    return "ica-component-review" in element.tags or element.section == "ICA: components"


__all__ = [
    "before_epoch_sections",
    "before_trial_evidence",
    "move_tagged_content_first",
    "before_raw_sections",
    "before_ica_component_review",
    "drop_per_epoch_metadata_tables",
    "drop_replaced_clean_raw_panels",
    "drop_replaced_filtered_spectrum",
    "drop_replaced_ica_eog_panels",
    "drop_replaced_ica_ecg_panels",
    "drop_replaced_panels",
    "drop_replaced_per_run_bad_channels",
    "drop_replaced_raw_time_series",
    "drop_superseded_mne_ica_panels",
    "order_sections",
    "SECTION_ORDER",
    "move_tagged_content_before",
    "open_subject_report",
    "place_events_with_epochs",
    "remove_tagged_content",
]
