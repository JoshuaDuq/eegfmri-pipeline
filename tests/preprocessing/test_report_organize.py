"""Report-wide policies applied every time a subject report is reopened.

Placement and content-pruning decisions that belong to the document as a whole rather
than to any one section live in ``report.organize``. These tests pin the ones a reader
would notice: what gets dropped, and what must never be dropped with it.
"""

from __future__ import annotations

import matplotlib
import mne
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from eeg_pipeline.preprocessing.report.organize import (  # noqa: E402
    drop_per_epoch_metadata_tables,
    drop_replaced_filtered_spectrum,
    drop_replaced_ica_ecg_panels,
    drop_replaced_panels,
    drop_replaced_per_run_bad_channels,
    drop_replaced_raw_time_series,
    drop_superseded_mne_ica_panels,
    open_subject_report,
    place_events_with_epochs,
)


def _mne_ica_panels(report: mne.Report) -> None:
    """Add the ICA panels MNE-BIDS-Pipeline writes, superseded and not."""
    figure = plt.figure()
    for title in (
        "ICA component properties",
        "ICA component topographies",
        "ICALabel: eye blink components",
        "ICALabel: heart beat components",
        "ICALabel: report",
        "Original and cleaned signal",
        "Scores for matching ECG patterns",
    ):
        report.add_figure(fig=figure, title=title, section="ICA: components", tags=("ica",))
    plt.close(figure)


def _authoritative_review(report: mne.Report) -> None:
    """Add a stand-in for the review this pipeline renders itself."""
    figure = plt.figure()
    report.add_figure(
        fig=figure,
        title="All component topographies",
        section="ICA decomposition quality",
        tags=("ica", "ica-decomposition"),
    )
    plt.close(figure)


def test_mne_ica_panels_are_dropped_when_the_review_replaces_them() -> None:
    """The duplicated pictures go; the panels nothing here reproduces stay.

    Every ``ICALabel:`` panel goes, the numeric ``ICALabel: report`` table included: the
    exclusion ledger carries each component's decision and deciding detector, and every
    dossier draws its full class distribution. MNE's overlay figures and its score panel
    have no counterpart here and are left alone.
    """
    report = mne.Report(title="ica", verbose="ERROR")
    _authoritative_review(report)
    _mne_ica_panels(report)

    drop_superseded_mne_ica_panels(report)

    remaining = {element.name for element in report._content}
    assert remaining == {
        "All component topographies",
        "Original and cleaned signal",
        "Scores for matching ECG patterns",
    }


def test_mne_ica_panels_survive_a_report_with_no_replacement_for_them() -> None:
    """The invariant: never remove a panel and leave nothing in its place.

    A run that appends only the cardiac or ocular review adds nothing standing in for
    these, so on that report MNE's panels are the only component evidence there is.
    """
    report = mne.Report(title="ica", verbose="ERROR")
    _mne_ica_panels(report)

    drop_superseded_mne_ica_panels(report)

    assert "ICA component properties" in {element.name for element in report._content}


def test_dropping_superseded_panels_is_safe_when_they_were_never_added() -> None:
    """A resting-state or EEG-only run may never have produced them."""
    report = mne.Report(title="ica", verbose="ERROR")

    drop_superseded_mne_ica_panels(report)

    assert report._content == []


def test_reopening_drops_mne_ica_panels_written_after_the_review(tmp_path) -> None:
    """The regression this move exists for.

    MNE-BIDS-Pipeline rewrites these panels every time ``_08a_apply_ica`` runs, which is
    after the stage that added their replacement. The drop used to live in that stage and
    was reached again only when ``ica.band_specific_report.comparisons`` was configured,
    so a dataset with no condition contrasts — what the shipped ``eeg_only`` preset sets —
    kept both copies of every component picture forever.
    """
    path = tmp_path / "sub-0001_report.h5"
    report = mne.Report(title="ica", verbose="ERROR")
    _authoritative_review(report)
    report.save(path, overwrite=True, open_browser=False)

    # Whatever MNE-BIDS-Pipeline writes next lands in the same file.
    rewritten = mne.open_report(path)
    _mne_ica_panels(rewritten)
    rewritten.save(path, overwrite=True, open_browser=False)

    assert "ICA component properties" in {element.name for element in rewritten._content}

    reopened = open_subject_report(path)

    remaining = {element.name for element in reopened._content}
    assert "ICA component properties" not in remaining
    assert "ICA component topographies" not in remaining
    assert "All component topographies" in remaining


def _raw() -> mne.io.BaseRaw:
    info = mne.create_info(["C0", "C1"], 100.0, "eeg")
    data = np.random.default_rng(0).normal(0, 1e-5, (2, 1000))
    return mne.io.RawArray(data, info, verbose="ERROR")


def _epochs() -> mne.BaseEpochs:
    info = mne.create_info(["C0", "C1"], 100.0, "eeg")
    data = np.random.default_rng(0).normal(0, 1e-5, (4, 2, 100))
    return mne.EpochsArray(
        data,
        info,
        metadata=pd.DataFrame({"event_name": ["a"] * 4, "latency": [0.1, 0.2, 0.3, 0.4]}),
        verbose="ERROR",
    )


def test_the_per_epoch_metadata_dump_is_dropped() -> None:
    """One row per epoch times one column per marker, rendered twice in one report.

    It is a data dump rather than evidence — it scales with the trial count, it is already
    written beside the report as ``_events.tsv``, and the retention questions a reviewer
    actually asks are answered by the trial-retention panel.
    """
    report = mne.Report(title="epochs", verbose="ERROR")
    report.add_epochs(_epochs(), title="Epochs (before cleaning)", psd=False)
    assert any(element.name == "Metadata" for element in report._content)

    drop_per_epoch_metadata_tables(report)

    assert not any(element.name == "Metadata" for element in report._content)


def test_dropping_the_dump_keeps_the_rest_of_the_epochs_section() -> None:
    """Info, the drop log and the ERP image are evidence and must survive."""
    report = mne.Report(title="epochs", verbose="ERROR")
    report.add_epochs(_epochs(), title="Epochs (before cleaning)", psd=False)

    drop_per_epoch_metadata_tables(report)

    remaining = {element.name for element in report._content}
    assert "Info" in remaining
    assert "Drop log" in remaining


def test_dropping_the_dump_is_safe_when_there_is_none() -> None:
    report = mne.Report(title="epochs", verbose="ERROR")

    drop_per_epoch_metadata_tables(report)

    assert report._content == []


def test_dropping_a_replaced_panel_is_scoped_to_its_section() -> None:
    """MNE reuses panel titles across sections, so title alone is the wrong key.

    ``Report.remove`` matches on title across the whole document. "Time series" and
    "PSD" each appear in several sections, so removing the raw butterfly by title would
    take the spectra of unrelated sections with it.
    """
    report = mne.Report(title="raw", verbose="ERROR")
    report.add_raw(_raw(), title="Raw (original)", psd=True, butterfly=True)
    report.add_raw(_raw(), title="Something else", psd=True, butterfly=True)

    drop_replaced_panels(report, section_prefix="Raw (original)", titles=("Time series",))

    survivors = {(element.section, element.name) for element in report._content}
    assert ("Raw (original)", "Time series") not in survivors
    assert ("Raw (original)", "PSD") in survivors
    assert ("Something else", "Time series") in survivors


def test_the_raw_butterfly_panels_are_dropped_for_both_raw_sections() -> None:
    """The original and the filtered raw each carry one, and both are replaced."""
    report = mne.Report(title="raw", verbose="ERROR")
    report.add_raw(_raw(), title="Raw (original)", psd=True, butterfly=True)
    report.add_raw(_raw(), title="Raw (filtered)", psd=True, butterfly=True)

    drop_replaced_raw_time_series(report)

    assert not any(element.name == "Time series" for element in report._content)
    assert sum(element.name == "PSD" for element in report._content) == 2


def test_dropping_the_raw_butterfly_is_safe_when_there_is_none() -> None:
    """A report built without ``add_raw`` must not fail the stage that prunes it."""
    report = mne.Report(title="raw", verbose="ERROR")

    drop_replaced_raw_time_series(report)

    assert report._content == []


def test_the_mne_ecg_panels_are_dropped_but_the_overlay_survives() -> None:
    """The cardiac review replaces the score and epoch panels, not the raw overlay.

    ``ICA: components`` carries three MNE panels. Two of them — the ECG match scores and
    the ECG epoch overlay — are single-run versions of what the cardiac review renders
    per run with its own detection QC. The signal overlay is not, so it stays.
    """
    report = mne.Report(title="ica", verbose="ERROR")
    for title in (
        "Original and cleaned signal",
        "Scores for matching ECG patterns",
        "Original and cleaned ECG epochs",
    ):
        report.add_html(html="<p/>", title=title, section="ICA: components", tags=("ica",))

    drop_replaced_ica_ecg_panels(report)

    survivors = {element.name for element in report._content}
    assert survivors == {"Original and cleaned signal"}


def test_the_per_run_bad_channel_items_are_dropped_together() -> None:
    """One accordion item per run, each stating a single list, becomes one table.

    A six-run session spends six table-of-contents entries and six clicks to report what
    is usually "none" six times, and never puts two runs side by side — which is the
    comparison the reviewer is actually making.
    """
    report = mne.Report(title="dq", verbose="ERROR")
    for run in (1, 2, 3):
        report.add_html(
            html="<p>Bad channels marked in original data:</p>",
            title=f"Bad channels: run-{run}",
            section="Data quality",
            tags=("raw", "data-quality"),
        )
        report.add_html(
            html="<p/>",
            title=f"Bad channel detection: run-{run}",
            section="Data quality",
            tags=("raw", "data-quality"),
        )

    drop_replaced_per_run_bad_channels(report)

    assert report._content == []


def test_dropping_per_run_bad_channels_keeps_the_time_resolved_section() -> None:
    """ "Data quality over time" shares the prefix but is a different section."""
    report = mne.Report(title="dq", verbose="ERROR")
    report.add_html(
        html="<p/>",
        title="Bad channels: run-1",
        section="Data quality",
        tags=("raw", "data-quality"),
    )
    report.add_html(
        html="<p/>",
        title="When each run departed from its baseline",
        section="Data quality over time",
        tags=("raw", "run-continuity"),
    )

    drop_replaced_per_run_bad_channels(report)

    assert [element.name for element in report._content] == [
        "When each run departed from its baseline"
    ]


def test_reopening_a_report_applies_the_document_policies(tmp_path) -> None:
    """The policies have to hold however the report is reached, so they sit on the open."""
    path = tmp_path / "sub-01_report.h5"
    report = mne.Report(title="epochs", verbose="ERROR")
    report.add_epochs(_epochs(), title="Epochs (before cleaning)", psd=False)
    report.save(path, overwrite=True, open_browser=False)

    reopened = open_subject_report(path)

    assert not any(element.name == "Metadata" for element in reopened._content)
    assert "figure svg" in reopened.include


def test_reopening_a_report_places_the_events_panel(tmp_path) -> None:
    """Every review stage reopens the report and saves it again, so the placement has to
    sit on the open: whichever stage saved last otherwise decided where events ended up."""
    path = tmp_path / "sub-01_report.h5"
    _report_with_a_stranded_events_panel().save(path, overwrite=True, open_browser=False)

    reopened = open_subject_report(path)

    events = [element for element in reopened._content if element.name == "Events"]
    assert [element.section for element in events] == ["Events"]


def _report_with_a_stranded_events_panel() -> mne.Report:
    """A report shaped like MNE-BIDS-Pipeline leaves it: events added with no section."""
    report = mne.Report(title="events", verbose="ERROR")
    report.add_epochs(_epochs(), title="Epochs (before cleaning)", psd=False)
    figure = plt.figure()
    report.add_figure(fig=figure, title="Events", tags=("events",))
    plt.close(figure)
    report.add_html(html="<p>exploratory</p>", title="Band ICAs", section="Exploratory")
    return report


def test_the_events_panel_is_given_a_section_of_its_own() -> None:
    """MNE-BIDS-Pipeline adds it with no section, and MNE only wraps sectioned content.

    The panel therefore rendered as a bare accordion item with no section heading around
    it, the one item in the document that did not look like everything beside it.
    """
    report = _report_with_a_stranded_events_panel()
    assert all(element.section is None for element in report._content if element.name == "Events")

    place_events_with_epochs(report)

    sections = {element.section for element in report._content if element.name == "Events"}
    assert sections == {"Events"}


def test_the_events_panel_sits_beside_the_epochs_it_describes() -> None:
    """Appended last, it landed after 30 MB of exploratory ICA sliders — twenty sections
    away from the epochs section whose trial counts it explains."""
    report = _report_with_a_stranded_events_panel()

    place_events_with_epochs(report)

    order = [element.name for element in report._content]
    assert order.index("Events") < order.index("Band ICAs")


def test_a_report_without_events_is_left_alone() -> None:
    """A resting-state recording has no events panel, and that is not a failure."""
    report = mne.Report(title="rest", verbose="ERROR")
    report.add_epochs(_epochs(), title="Epochs (before cleaning)", psd=False)
    before = [element.name for element in report._content]

    place_events_with_epochs(report)

    assert [element.name for element in report._content] == before


def test_the_filtered_raw_spectrum_is_dropped_but_the_original_survives() -> None:
    """The filtered PSD duplicates the custom spectra; the original one does not.

    Both MNE panels run to Nyquist, so on a 500 Hz recording low-passed at 100 Hz the
    filtered one spends four fifths of its axis on roll-off below the noise floor, and
    what remains is the 1-100 Hz view the sensor-spectra section already draws per run
    with percentile bands and harmonic markers.

    The original raw spectrum is not a duplicate: nothing else in the report shows the
    data before filtering, where the full bandwidth is exactly what makes gradient
    harmonics and the anti-alias corner visible.
    """
    report = mne.Report(title="raw", verbose="ERROR")
    report.add_raw(_raw(), title="Raw (original)", psd=True, butterfly=False)
    report.add_raw(_raw(), title="Raw (filtered)", psd=True, butterfly=False)

    drop_replaced_filtered_spectrum(report)

    survivors = {(element.section, element.name) for element in report._content}
    assert ("Raw (filtered)", "PSD") not in survivors
    assert ("Raw (original)", "PSD") in survivors
    assert ("Raw (filtered)", "Info") in survivors


def test_the_ocular_review_supersedes_mnes_eog_panels_like_the_cardiac_one() -> None:
    """The same two quantities as the ECG pair, for the other artifact. These were left
    in place while the ECG pair went, which left a reviewer meeting the ocular evidence
    twice and the cardiac evidence once."""
    from eeg_pipeline.preprocessing.report.organize import drop_replaced_ica_eog_panels

    report = mne.Report(title="ica", verbose="ERROR")
    figure = plt.figure()
    for title in (
        "Scores for matching EOG patterns",
        "Original and cleaned EOG epochs",
        "Original and cleaned signal",
    ):
        report.add_figure(fig=figure, title=title, section="ICA: components", tags=("ica",))
    plt.close(figure)

    drop_replaced_ica_eog_panels(report)

    # "Original and cleaned signal" overlays the raw traces either side of the exclusions,
    # which nothing else in this report draws.
    assert {element.name for element in report._content} == {"Original and cleaned signal"}


def test_clean_raw_panels_go_only_where_their_replacement_exists() -> None:
    """Two replacements added by two stages, so the drop is per panel: a report carrying
    the continuity section but not the sensor spectra loses the butterfly and keeps the
    spectrum."""
    from eeg_pipeline.preprocessing.report.organize import drop_replaced_clean_raw_panels

    report = mne.Report(title="raw", verbose="ERROR")
    figure = plt.figure()
    report.add_figure(
        fig=figure,
        title="Amplitude over time by channel",
        section="Data quality over time",
        tags=("raw", "run-continuity"),
    )
    for title in ("Time series", "PSD"):
        report.add_figure(fig=figure, title=title, section="Raw (clean)", tags=("raw", "clean"))
    plt.close(figure)

    drop_replaced_clean_raw_panels(report)

    remaining = {element.name for element in report._content}
    assert "Time series" not in remaining
    assert "PSD" in remaining


def test_reopening_drops_clean_raw_panels_written_after_their_replacement(tmp_path) -> None:
    """MNE-BIDS-Pipeline writes 'Raw (clean)' when it applies the ICA, after the
    continuity stage that replaced its butterfly has run."""
    path = tmp_path / "sub-0001_report.h5"
    report = mne.Report(title="raw", verbose="ERROR")
    figure = plt.figure()
    report.add_figure(
        fig=figure,
        title="Amplitude over time by channel",
        section="Data quality over time",
        tags=("raw", "run-continuity"),
    )
    report.save(path, overwrite=True, open_browser=False)

    rewritten = mne.open_report(path)
    rewritten.add_figure(
        fig=figure, title="Time series", section="Raw (clean)", tags=("raw", "clean")
    )
    rewritten.save(path, overwrite=True, open_browser=False)
    plt.close(figure)

    reopened = open_subject_report(path)

    assert "Time series" not in {element.name for element in reopened._content}


def test_the_events_panel_sits_ahead_of_the_panels_that_count_its_trials() -> None:
    """Reading order: what was presented, then what survived, then whether the survivors
    carry signal. Three sections anchor near the epochs, and the events panel is placed on
    every reopen and so always moved last -- which put the trial counts in front of the
    panel saying what the trials were."""
    report = mne.Report(title="events", verbose="ERROR")
    figure = plt.figure()
    report.add_figure(
        fig=figure, title="Trial retention", section="Epoch rejection", tags=("epoch-rejection",)
    )
    report.add_figure(
        fig=figure,
        title="Evidence that signal survived",
        section="Signal preservation",
        tags=("signal-preservation",),
    )
    report.add_figure(fig=figure, title="Events", section=None, tags=("events",))
    plt.close(figure)

    place_events_with_epochs(report)

    order = [element.name for element in report._content]
    assert order.index("Events") < order.index("Trial retention")
    assert order.index("Trial retention") < order.index("Evidence that signal survived")


def _sectioned(report: mne.Report, *sections: str) -> None:
    figure = plt.figure()
    for name in sections:
        report.add_figure(fig=figure, title=f"panel in {name}", section=name, tags=("x",))
    plt.close(figure)


def test_the_document_follows_the_declared_order_whatever_order_stages_ran_in() -> None:
    """The shape of the report must be a property of the pipeline, not of the run path."""
    from eeg_pipeline.preprocessing.report.organize import order_sections

    report = mne.Report(title="subject", verbose="ERROR")
    # Deliberately backwards: epochs written before the raw they came from.
    _sectioned(
        report,
        "Epochs (clean)",
        "Signal preservation",
        "Raw (original)",
        "At a glance",
        "ICA decomposition quality",
    )

    order_sections(report)

    assert [str(element.section) for element in report._content] == [
        "At a glance",
        "Raw (original)",
        "ICA decomposition quality",
        "Signal preservation",
        "Epochs (clean)",
    ]


def test_a_per_band_review_section_is_placed_by_its_prefix() -> None:
    """The band is part of the section name, so the order list cannot spell it out."""
    from eeg_pipeline.preprocessing.report.organize import order_sections

    report = mne.Report(title="subject", verbose="ERROR")
    _sectioned(report, "Epoch rejection", "ICA component review: Broadband 1–100 Hz")

    order_sections(report)

    assert [str(element.section) for element in report._content] == [
        "ICA component review: Broadband 1–100 Hz",
        "Epoch rejection",
    ]


def test_panels_keep_their_own_order_inside_a_section() -> None:
    """Within one section the order is a judgement about one body of evidence, and this
    function has no view on it."""
    from eeg_pipeline.preprocessing.report.organize import order_sections

    report = mne.Report(title="subject", verbose="ERROR")
    figure = plt.figure()
    for title in ("first", "second", "third"):
        report.add_figure(fig=figure, title=title, section="Epoch rejection", tags=("x",))
    report.add_figure(fig=figure, title="glance", section="At a glance", tags=("x",))
    plt.close(figure)

    order_sections(report)

    assert [element.name for element in report._content] == ["glance", "first", "second", "third"]


def test_an_unknown_section_is_appended_rather_than_interleaved() -> None:
    """A stage this list does not name yet must not land in the middle of the evidence."""
    from eeg_pipeline.preprocessing.report.organize import order_sections

    report = mne.Report(title="subject", verbose="ERROR")
    _sectioned(report, "Something new", "At a glance", "Epochs (clean)")

    order_sections(report)

    sections = [str(element.section) for element in report._content]
    assert sections[0] == "At a glance"
    assert sections[-1] == "Something new"


def test_sectionless_reference_material_sorts_last() -> None:
    """MNE leaves the configuration file and system information unsectioned, and both are
    reference a reader reaches for after the evidence."""
    from eeg_pipeline.preprocessing.report.organize import order_sections

    report = mne.Report(title="subject", verbose="ERROR")
    report.add_html(html="<p>sys</p>", title="System information")
    _sectioned(report, "Epochs (clean)", "At a glance")

    order_sections(report)

    assert [element.name for element in report._content][-1] == "System information"


def test_a_raw_section_reduced_to_metadata_is_dropped() -> None:
    """Its butterfly and spectrum are replaced elsewhere; an Info table alone is a
    contents entry pointing at nothing a reader came for."""
    from eeg_pipeline.preprocessing.report.organize import drop_metadata_only_raw_sections

    report = mne.Report(title="raw", verbose="ERROR")
    figure = plt.figure()
    report.add_figure(
        fig=figure, title="Magnitude and step response", section="Filter response",
        tags=("filter-response",),
    )
    report.add_figure(
        fig=figure, title="Channels remaining per region", section="Channel and region coverage",
        tags=("channel-coverage",),
    )
    report.add_figure(fig=figure, title="Info", section="Raw (filtered)", tags=("raw",))
    report.add_figure(fig=figure, title="Info", section="Raw (original)", tags=("raw",))
    report.add_figure(fig=figure, title="PSD", section="Raw (original)", tags=("raw",))
    plt.close(figure)

    drop_metadata_only_raw_sections(report)

    sections = [str(element.section) for element in report._content]
    assert "Raw (filtered)" not in sections
    # Raw (original) keeps evidence, so its metadata keeps company with it.
    assert sections.count("Raw (original)") == 2


def test_a_raw_section_keeps_its_metadata_while_it_still_holds_evidence() -> None:
    from eeg_pipeline.preprocessing.report.organize import drop_metadata_only_raw_sections

    report = mne.Report(title="raw", verbose="ERROR")
    figure = plt.figure()
    for section, tag in (("Filter response", "filter-response"),
                         ("Channel and region coverage", "channel-coverage")):
        report.add_figure(fig=figure, title=f"panel {tag}", section=section, tags=(tag,))
    report.add_figure(fig=figure, title="Info", section="Raw (clean)", tags=("raw",))
    report.add_figure(fig=figure, title="Time series", section="Raw (clean)", tags=("raw",))
    plt.close(figure)

    drop_metadata_only_raw_sections(report)

    assert "Info" in {element.name for element in report._content}


def test_metadata_survives_when_nothing_replaces_it() -> None:
    """A report built from a stage subset must not lose the only record of the file."""
    from eeg_pipeline.preprocessing.report.organize import drop_metadata_only_raw_sections

    report = mne.Report(title="raw", verbose="ERROR")
    figure = plt.figure()
    report.add_figure(fig=figure, title="Info", section="Raw (filtered)", tags=("raw",))
    plt.close(figure)

    drop_metadata_only_raw_sections(report)

    assert [element.name for element in report._content] == ["Info"]


def test_the_two_cleaning_overlays_are_told_apart_by_stage() -> None:
    """MNE draws this overlay when it fits the ICA and again when it applies it, under one
    title both times. The figures differ -- different exclusion sets -- and nothing said
    which was which."""
    from eeg_pipeline.preprocessing.report.organize import name_cleaning_overlays_by_stage

    report = mne.Report(title="ica", verbose="ERROR")
    figure = plt.figure()
    for section in ("ICA: components", "ICA: removals"):
        report.add_figure(
            fig=figure, title="Original and cleaned signal", section=section, tags=("ica",)
        )
    plt.close(figure)

    name_cleaning_overlays_by_stage(report)

    names = [element.name for element in report._content]
    assert names == [
        "Original and cleaned signal (exclusions proposed at fitting)",
        "Original and cleaned signal (exclusions as applied)",
    ]
    assert len(set(names)) == 2
    # The heading a reader actually sees, not only the metadata the archive carries.
    # MNE renders each element when it is added and stores the markup, so an earlier
    # version of this that set ``name`` alone left the rendered page untouched -- and
    # this test passed anyway, because it only looked at ``name``.
    rendered = "".join(str(element.html) for element in report._content)
    assert ">Original and cleaned signal</a>" not in rendered
    for stage in ("exclusions proposed at fitting", "exclusions as applied"):
        assert f">Original and cleaned signal ({stage})</a>" in rendered


def test_the_overlay_rename_repairs_a_half_renamed_report() -> None:
    """The state an earlier, broken version of this left behind.

    That version set ``name`` and not the stored markup, so reports came out with renamed
    metadata and an unchanged page. Keyed on the name, the corrected version then read
    those as already done and skipped them -- a fix that could not repair its own
    predecessor's output.
    """
    from eeg_pipeline.preprocessing.report.organize import name_cleaning_overlays_by_stage

    report = mne.Report(title="ica", verbose="ERROR")
    figure = plt.figure()
    report.add_figure(
        fig=figure, title="Original and cleaned signal", section="ICA: removals", tags=("ica",)
    )
    plt.close(figure)
    # Exactly what the broken version produced: metadata renamed, markup untouched.
    report._content[0].name = "Original and cleaned signal (exclusions as applied)"

    name_cleaning_overlays_by_stage(report)

    rendered = str(report._content[0].html)
    assert ">Original and cleaned signal</a>" not in rendered
    assert ">Original and cleaned signal (exclusions as applied)</a>" in rendered


def test_the_overlay_rename_is_idempotent() -> None:
    """It runs on every reopen, so a second pass must not append the stage twice."""
    from eeg_pipeline.preprocessing.report.organize import name_cleaning_overlays_by_stage

    report = mne.Report(title="ica", verbose="ERROR")
    figure = plt.figure()
    report.add_figure(
        fig=figure, title="Original and cleaned signal", section="ICA: removals", tags=("ica",)
    )
    plt.close(figure)

    name_cleaning_overlays_by_stage(report)
    once = str(report._content[0].html)
    name_cleaning_overlays_by_stage(report)

    assert str(report._content[0].html) == once
    # Once as the heading and once as the image's alternative text, and no more: a second
    # pass appending the stage again would read "(exclusions as applied) (exclusions as
    # applied)".
    assert once.count(">Original and cleaned signal (exclusions as applied)</a>") == 1
    assert once.count("(exclusions as applied)") == 2


def test_the_task_epoch_drop_logs_go_where_the_rejection_section_replaces_them() -> None:
    """Two copies of one accounting, either side of the cleaning, and on a session that
    dropped nothing both are empty by construction. The rejection section answers what a
    reviewer asks of them from the same drop log, with positions and per-group rates."""
    from eeg_pipeline.preprocessing.report.organize import drop_replaced_epoch_drop_logs

    report = mne.Report(title="epochs", verbose="ERROR")
    figure = plt.figure()
    report.add_figure(
        fig=figure, title="Trial retention", section="Epoch rejection", tags=("epoch-rejection",)
    )
    for section in ("Epochs (before cleaning)", "Epochs (clean)", "ICA: epochs for fitting"):
        report.add_figure(fig=figure, title="Drop log", section=section, tags=("epochs",))
    plt.close(figure)

    drop_replaced_epoch_drop_logs(report)

    remaining = {(str(e.section), str(e.name)) for e in report._content}
    assert ("Epochs (before cleaning)", "Drop log") not in remaining
    assert ("Epochs (clean)", "Drop log") not in remaining
    # A different epoch set, cut for fitting rather than analysis, and nothing in this
    # report accounts for what was dropped from it.
    assert ("ICA: epochs for fitting", "Drop log") in remaining


def test_the_drop_logs_survive_without_the_section_that_replaces_them() -> None:
    from eeg_pipeline.preprocessing.report.organize import drop_replaced_epoch_drop_logs

    report = mne.Report(title="epochs", verbose="ERROR")
    figure = plt.figure()
    report.add_figure(
        fig=figure, title="Drop log", section="Epochs (clean)", tags=("epochs",)
    )
    plt.close(figure)

    drop_replaced_epoch_drop_logs(report)

    assert [e.name for e in report._content] == ["Drop log"]


def test_section_order_names_no_scanner_section():
    from eeg_pipeline.preprocessing.report.organize import SECTION_ORDER

    assert "Residual scanner gradient" not in SECTION_ORDER


def test_the_report_javascript_is_applied_once_however_often_it_is_reopened() -> None:
    from eeg_pipeline.preprocessing.report.style import apply_report_js

    report = mne.Report(title="subject", verbose="ERROR")
    apply_report_js(report)
    apply_report_js(report)
    apply_report_js(report)

    assert report.include.count("/* eeg-pipeline report js */") == 1


def test_reopening_a_report_applies_the_javascript(tmp_path) -> None:
    path = tmp_path / "sub-0001_report.h5"
    mne.Report(title="subject", verbose="ERROR").save(
        path, overwrite=True, open_browser=False
    )

    report = open_subject_report(path)

    assert "/* eeg-pipeline report js */" in report.include


def test_the_collapse_script_targets_only_top_level_sections() -> None:
    from eeg_pipeline.preprocessing.report.style import REPORT_JS

    # The selector is the whole guarantee: collapsing nested items too would hide a
    # section's contents after the reader expanded it.
    assert "closest('.accordion-item')" in REPORT_JS
    assert "accordion-collapse" in REPORT_JS
