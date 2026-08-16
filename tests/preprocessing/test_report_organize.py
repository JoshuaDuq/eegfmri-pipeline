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
