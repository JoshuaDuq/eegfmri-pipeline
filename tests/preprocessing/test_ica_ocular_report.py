from __future__ import annotations

import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest

from eeg_pipeline.preprocessing.ica_ocular_report import (
    OcularReviewSettings,
    _absolute_scores,
    _ocular_review_guide_html,
    _resolve_eog_channels,
    _surrogate_channels,
)


def _raw(channel_names: list[str], channel_types: list[str]) -> mne.io.BaseRaw:
    info = mne.create_info(channel_names, 100.0, channel_types)
    return mne.io.RawArray(np.zeros((len(channel_names), 100)), info, verbose="ERROR")


def test_frontopolar_surrogates_are_accepted_when_no_eog_electrode_exists() -> None:
    raw = _raw(["Fp1", "Fp2", "Cz"], ["eeg", "eeg", "eeg"])
    settings = OcularReviewSettings(enabled=True, eog_channels=["Fp1", "Fp2"])

    assert _resolve_eog_channels(raw, settings) == ["Fp1", "Fp2"]
    assert _surrogate_channels(raw, ["Fp1", "Fp2"]) == ("Fp1", "Fp2")


def test_dedicated_eog_channels_are_not_treated_as_surrogates() -> None:
    raw = _raw(["EOG1", "Cz"], ["eog", "eeg"])
    settings = OcularReviewSettings(enabled=True, eog_channels="eog")

    assert _resolve_eog_channels(raw, settings) == ["EOG1"]
    assert _surrogate_channels(raw, ["EOG1"]) == ()


def test_missing_configured_ocular_channel_fails_fast() -> None:
    raw = _raw(["Fp1", "Cz"], ["eeg", "eeg"])
    settings = OcularReviewSettings(enabled=True, eog_channels=["Fp1", "Fp2"])

    with pytest.raises(ValueError, match=r"\['Fp2'\]"):
        _resolve_eog_channels(raw, settings)


def test_automatic_eog_selection_names_the_surrogate_remedy() -> None:
    raw = _raw(["Fp1", "Cz"], ["eeg", "eeg"])
    settings = OcularReviewSettings(enabled=True, eog_channels="eog")

    with pytest.raises(ValueError, match="Fp1"):
        _resolve_eog_channels(raw, settings)


def test_guide_declares_circularity_only_when_surrogates_are_used() -> None:
    settings = OcularReviewSettings(enabled=True, eog_channels=["Fp1", "Fp2"])

    with_surrogates = _ocular_review_guide_html(settings, surrogates=("Fp1", "Fp2"))
    without_surrogates = _ocular_review_guide_html(settings, surrogates=())

    assert "partly circular" in with_surrogates
    assert "Fp1, Fp2" in with_surrogates
    assert "partly circular" not in without_surrogates


def test_scores_combine_eog_channels_by_largest_absolute_correlation() -> None:
    scores = np.array([[0.1, -0.9, 0.2], [-0.7, 0.3, 0.05]])

    combined = _absolute_scores(scores, component_count=3)

    np.testing.assert_allclose(combined, [0.7, 0.9, 0.2])


def test_scores_that_do_not_describe_every_component_fail_fast() -> None:
    with pytest.raises(ValueError, match="expected 4"):
        _absolute_scores(np.array([0.1, 0.2, 0.3]), component_count=4)


def test_non_finite_scores_fail_fast() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        _absolute_scores(np.array([0.1, np.nan]), component_count=2)


def _run_review(scores: np.ndarray, flagged: tuple[int, ...] = ()):
    """One run's EOG evidence, without the blink overlay figure the panels do not read."""
    from eeg_pipeline.preprocessing.ica_ocular_report import RunOcularReview

    return RunOcularReview(
        recording_id="sub-0001_task-x_run-1",
        blink_epoch_count=10,
        duration_s=300.0,
        absolute_scores=scores,
        flagged_components=flagged,
        overlay_figure=None,
    )


def test_the_component_score_axis_is_logarithmic() -> None:
    """One blink component at r ~ 0.9 must not flatten the other sixty onto the floor.

    A linear axis scaled to the blink component leaves everything else, including the
    components the detector flagged at 0.2, indistinguishable from zero.
    """
    import matplotlib

    matplotlib.use("Agg")
    from eeg_pipeline.preprocessing.ica_ocular_report import _plot_component_scores

    scores = np.array([0.94, 0.05, 0.30, 0.04, 0.06, 0.05])

    figure = _plot_component_scores([_run_review(scores, flagged=(0, 2))], excluded=[0, 3])

    assert figure.axes[0].get_yscale() == "log"


def test_the_score_panel_uses_markers_not_bars_on_the_log_axis() -> None:
    """A bar reads its value as a length from zero, and zero is not on a log axis."""
    import matplotlib

    matplotlib.use("Agg")
    from eeg_pipeline.preprocessing.ica_ocular_report import _plot_component_scores

    scores = np.array([0.94, 0.05, 0.30, 0.04, 0.06, 0.05])

    figure = _plot_component_scores([_run_review(scores)], excluded=[0])

    # The status strip below is the only axis allowed to carry bars.
    assert len(figure.axes[0].patches) == 0
    assert len(figure.axes[1].patches) > 0


def test_the_panel_shows_where_the_detector_drew_its_line() -> None:
    """Crosses said which components were flagged and never said why.

    ``find_bads_eog`` thresholds an adaptive z-score, so there is no fixed correlation to
    print -- but the decisions themselves bracket the cutoff: within a run it lies above
    every unflagged component and at or below every flagged one. On sub-0012 a component
    at r=0.3 was flagged while one at r=0.2 was not, and the panel gave a reader nothing
    to reconcile that with.

    Measured from the decisions rather than by reimplementing MNE's rule, so the band
    cannot drift away from the flags drawn beside it.
    """
    import matplotlib

    matplotlib.use("Agg")
    from eeg_pipeline.preprocessing.ica_ocular_report import _plot_component_scores

    scores = np.array([0.94, 0.05, 0.30, 0.04, 0.06, 0.05])

    figure = _plot_component_scores([_run_review(scores, flagged=(0, 2))], excluded=[0])

    spans = [
        collection
        for collection in figure.axes[0].collections
        if type(collection).__name__ == "PolyCollection"
    ]
    assert spans, "the decision boundary is not drawn"
    low, high = figure.axes[0]._eog_threshold_band
    # Above every score the run left unflagged, and no higher than the lowest it flagged.
    assert low == pytest.approx(0.06)
    assert high == pytest.approx(0.30)


def test_a_run_that_flagged_nothing_gets_no_invented_boundary() -> None:
    """With no flags there is no bracket: the cutoff sits above every score, unbounded.

    Drawing a band there would put a threshold on the figure that no decision supports.
    """
    import matplotlib

    matplotlib.use("Agg")
    from eeg_pipeline.preprocessing.ica_ocular_report import _plot_component_scores

    scores = np.array([0.94, 0.05, 0.30, 0.04, 0.06, 0.05])

    figure = _plot_component_scores([_run_review(scores)], excluded=[0])

    assert not hasattr(figure.axes[0], "_eog_threshold_band")


def test_exclusion_status_is_not_drawn_in_the_detector_flag_colour() -> None:
    """ "Excluded" and "flagged by find_bads_eog" are different claims about a component."""
    import matplotlib

    matplotlib.use("Agg")
    from eeg_pipeline.preprocessing.ica_ocular_report import _plot_component_scores
    from eeg_pipeline.preprocessing.report.style import EXCLUDED_COLOR, FLAG_COLOR

    scores = np.array([0.94, 0.05, 0.30, 0.04])

    figure = _plot_component_scores([_run_review(scores, flagged=(0,))], excluded=[0, 1])

    strip_colours = {
        matplotlib.colors.to_hex(patch.get_facecolor()) for patch in figure.axes[1].patches
    }
    assert matplotlib.colors.to_hex(EXCLUDED_COLOR) in strip_colours
    assert matplotlib.colors.to_hex(FLAG_COLOR) not in strip_colours


def test_the_component_axis_carries_integer_ticks() -> None:
    """Component index is categorical: there is no component 2.5.

    The default locator put ticks at 0.0, 2.5, 5.0 and so on, which named nothing on the
    axis and did not line up with the cells of the status strip beneath it.
    """
    import matplotlib

    matplotlib.use("Agg")
    from eeg_pipeline.preprocessing.ica_ocular_report import _plot_component_scores

    scores = np.array([0.94, 0.05, 0.30, 0.04, 0.06, 0.05])

    figure = _plot_component_scores([_run_review(scores)], excluded=[0])

    ticks = figure.axes[1].get_xticks()
    assert all(float(tick).is_integer() for tick in ticks)
    assert set(ticks) <= set(range(scores.size))


def test_a_large_decomposition_thins_the_ticks_rather_than_crowding_them() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from eeg_pipeline.preprocessing.ica_ocular_report import _plot_component_scores

    scores = np.linspace(0.01, 0.9, 64)

    figure = _plot_component_scores([_run_review(scores)], excluded=[0])

    ticks = figure.axes[1].get_xticks()
    assert all(float(tick).is_integer() for tick in ticks)
    assert len(ticks) <= 33


def test_the_status_strip_says_which_shade_means_excluded() -> None:
    """Two greys with no key: the strip's label gave a count but never said which was which."""
    import matplotlib

    matplotlib.use("Agg")
    from eeg_pipeline.preprocessing.ica_ocular_report import _plot_component_scores

    scores = np.array([0.94, 0.05, 0.30, 0.04])

    figure = _plot_component_scores([_run_review(scores)], excluded=[0, 1])

    legend = figure.axes[1].get_legend()
    assert legend is not None
    labels = {text.get_text().lower() for text in legend.get_texts()}
    assert "excluded" in labels and "retained" in labels


def test_event_locked_time_axes_agree_across_the_review_sections() -> None:
    """The blink panel used milliseconds while its cardiac sibling used seconds.

    The two figures answer the same question about two artifacts over windows of the same
    length, and a reader moving between them had to rescale by 1000 to compare them.
    """
    import inspect

    from eeg_pipeline.preprocessing import ica_cardiac_report, ica_ocular_report

    ocular = inspect.getsource(ica_ocular_report)
    cardiac = inspect.getsource(ica_cardiac_report)

    assert "Time from blink peak (s)" in ocular
    assert "Time from blink peak (ms)" not in ocular
    assert "Time from R peak (s)" in cardiac


def _detection_reviews():
    """Two runs whose blink counts differ by two orders of magnitude."""
    from eeg_pipeline.preprocessing.ica_ocular_report import RunOcularReview

    return [
        RunOcularReview(
            recording_id="sub-0001_task-x_run-1",
            blink_epoch_count=203,
            duration_s=498.0,
            absolute_scores=np.array([0.9, 0.1]),
            flagged_components=(0,),
            overlay_figure=None,
        ),
        RunOcularReview(
            recording_id="sub-0001_task-x_run-2",
            blink_epoch_count=4,
            duration_s=498.0,
            absolute_scores=np.array([0.8, 0.2]),
            flagged_components=(0,),
            overlay_figure=None,
        ),
    ]


def test_blink_detection_puts_every_run_on_one_table() -> None:
    """The count existed only inside each slide's title, so comparing runs meant stepping
    through a carousel and remembering numbers. A run with four blinks produces a
    correlation that looks like any other run's, and nothing on the page said so."""
    from eeg_pipeline.preprocessing.ica_ocular_report import ocular_detection_html

    document = ocular_detection_html(_detection_reviews())

    assert "run-1" in document and "run-2" in document
    assert "203" in document and "4" in document


def test_blink_detection_reports_a_rate_that_runs_of_different_length_can_be_read_against() -> None:
    """A count alone confounds blink rate with run duration."""
    from eeg_pipeline.preprocessing.ica_ocular_report import ocular_detection_html

    document = ocular_detection_html(_detection_reviews())

    assert "24.5" in document  # 203 blinks over 8.3 min
    assert "0.5" in document  # 4 blinks over 8.3 min


def _ocular_review_inputs(tmp_path):
    """Two short runs with frontopolar surrogates, an ICA, and its component table."""
    import pandas as pd

    montage = mne.channels.make_standard_montage("standard_1020")
    names = ["Fp1", "Fp2", "Cz", "Pz", "O1", "O2"]
    rng = np.random.default_rng(0)
    paths = []
    for run in (1, 2):
        info = mne.create_info(names, 100.0, "eeg")
        raw = mne.io.RawArray(rng.normal(0, 1e-5, (len(names), 3000)), info, verbose="ERROR")
        raw.set_montage(montage)
        path = tmp_path / f"sub-0001_task-x_run-{run}_proc-filt_raw.fif"
        raw.save(path, overwrite=True, verbose="ERROR")
        paths.append(path)

    first = mne.io.read_raw_fif(paths[0], preload=True, verbose="ERROR")
    ica = mne.preprocessing.ICA(n_components=3, random_state=0, max_iter=200)
    ica.fit(first, verbose="ERROR")
    ica_path = tmp_path / "sub-0001_proc-ica_ica.fif"
    ica.save(ica_path, overwrite=True, verbose="ERROR")
    pd.DataFrame(
        {
            "component": [0, 1, 2],
            "status": ["bad", "good", "good"],
            "status_description": ["Auto-detected eye blink (MNE-ICALabel)", "", ""],
        }
    ).to_csv(tmp_path / "sub-0001_proc-ica_components.tsv", sep="\t", index=False)
    return paths, ica_path


def test_the_ocular_section_carries_the_blink_counts_behind_its_correlations(tmp_path) -> None:
    """The correlations were rendered with nothing on the page saying how many blinks each
    was measured from, so a run the detector barely fired on read like any other."""
    from eeg_pipeline.preprocessing.ica_ocular_report import (
        OCULAR_DETECTION_TITLE,
        generate_ica_ocular_review,
    )

    paths, ica_path = _ocular_review_inputs(tmp_path)
    report_path = tmp_path / "sub-0001_report.h5"
    mne.Report(title="sub-0001", verbose="ERROR").save(
        report_path, overwrite=True, open_browser=False
    )

    generate_ica_ocular_review(
        filtered_raw_paths=paths,
        ica_path=ica_path,
        report_path=report_path,
        output_path=tmp_path / "sub-0001_desc-icaeog_components.tsv",
        settings=OcularReviewSettings(enabled=True, eog_channels=["Fp1", "Fp2"]),
    )

    report = mne.open_report(report_path)
    panel = next(element for element in report._content if element.name == OCULAR_DETECTION_TITLE)
    assert "Blinks per minute" in panel.html
    # 3000 samples at 100 Hz is half a minute, and both runs must appear with a duration.
    assert "0.5" in panel.html


def test_the_ocular_stage_records_itself_in_the_report_build_record(tmp_path) -> None:
    """A stage that appends sections without recording itself leaves the build panel
    implying it never ran."""
    from eeg_pipeline.preprocessing.ica_ocular_report import generate_ica_ocular_review
    from eeg_pipeline.preprocessing.report.build_record import read_build_record

    paths, ica_path = _ocular_review_inputs(tmp_path)
    report_path = tmp_path / "sub-0001_report.h5"
    mne.Report(title="sub-0001", verbose="ERROR").save(
        report_path, overwrite=True, open_browser=False
    )

    generate_ica_ocular_review(
        filtered_raw_paths=paths,
        ica_path=ica_path,
        report_path=report_path,
        output_path=tmp_path / "sub-0001_desc-icaeog_components.tsv",
        settings=OcularReviewSettings(enabled=True, eog_channels=["Fp1", "Fp2"]),
    )

    stages = [entry["stage"] for entry in read_build_record(report_path)["stages"]]
    assert stages == ["ica-ocular-review"]


def _ocular_inputs_with_lengths(tmp_path, *, sample_counts):
    """Runs of the given lengths, an ICA, and its component table.

    A run too short to form a single blink epoch stands in for one the blink detector
    cannot resolve. What is under test is the review's response to an empty epoch set, not
    the reason it came back empty.
    """
    import pandas as pd

    montage = mne.channels.make_standard_montage("standard_1020")
    names = ["Fp1", "Fp2", "Cz", "Pz", "O1", "O2"]
    rng = np.random.default_rng(0)
    paths = []
    for run, samples in enumerate(sample_counts, start=1):
        info = mne.create_info(names, 100.0, "eeg")
        raw = mne.io.RawArray(rng.normal(0, 1e-5, (len(names), samples)), info, verbose="ERROR")
        raw.set_montage(montage)
        path = tmp_path / f"sub-0001_task-x_run-{run}_proc-filt_raw.fif"
        raw.save(path, overwrite=True, verbose="ERROR")
        paths.append(path)

    first = mne.io.read_raw_fif(paths[0], preload=True, verbose="ERROR")
    ica = mne.preprocessing.ICA(n_components=3, random_state=0, max_iter=200)
    ica.fit(first, verbose="ERROR")
    ica_path = tmp_path / "sub-0001_proc-ica_ica.fif"
    ica.save(ica_path, overwrite=True, verbose="ERROR")
    pd.DataFrame(
        {
            "component": [0, 1, 2],
            "status": ["bad", "good", "good"],
            "status_description": ["Auto-detected eye blink (MNE-ICALabel)", "", ""],
        }
    ).to_csv(tmp_path / "sub-0001_proc-ica_components.tsv", sep="\t", index=False)

    report_path = tmp_path / "sub-0001_report.h5"
    mne.Report(title="sub-0001", verbose="ERROR").save(
        report_path, overwrite=True, open_browser=False
    )
    return paths, ica_path, report_path


def test_a_run_with_no_resolvable_blinks_does_not_abort_the_ocular_review(tmp_path) -> None:
    """The failure this pins: sub-0008 run-1 yielded no blink epoch and the whole run died.

    ``create_eog_epochs`` returned an empty ``Epochs`` and ``average()`` raised out of the
    review, out of the ICA stage and out of the pipeline -- after seven subjects had already
    been reviewed successfully, leaving the remaining eight unprocessed.
    """
    from eeg_pipeline.preprocessing.ica_ocular_report import generate_ica_ocular_review

    paths, ica_path, report_path = _ocular_inputs_with_lengths(
        tmp_path, sample_counts=(3000, 60, 3000)
    )

    output = generate_ica_ocular_review(
        filtered_raw_paths=paths,
        ica_path=ica_path,
        report_path=report_path,
        output_path=tmp_path / "sub-0001_desc-icaeog_components.tsv",
        settings=OcularReviewSettings(enabled=True, eog_channels=["Fp1", "Fp2"]),
    )

    assert output.is_file()


def test_the_unresolvable_run_is_named_in_the_ocular_report(tmp_path) -> None:
    """A run absent from the evidence has to be named, or the panel's denominator lies."""
    from eeg_pipeline.preprocessing.ica_ocular_report import generate_ica_ocular_review

    paths, ica_path, report_path = _ocular_inputs_with_lengths(
        tmp_path, sample_counts=(3000, 60, 3000)
    )

    generate_ica_ocular_review(
        filtered_raw_paths=paths,
        ica_path=ica_path,
        report_path=report_path,
        output_path=tmp_path / "sub-0001_desc-icaeog_components.tsv",
        settings=OcularReviewSettings(enabled=True, eog_channels=["Fp1", "Fp2"]),
    )

    report = mne.open_report(report_path)
    document = " ".join(element.html for element in report._content if element.html)
    assert "run-2" in document


def test_a_subject_with_no_resolvable_blinks_anywhere_still_reports_that(tmp_path) -> None:
    """With no run resolvable there is no overlay to draw, and the absence is the finding."""
    from eeg_pipeline.preprocessing.ica_ocular_report import generate_ica_ocular_review

    paths, ica_path, report_path = _ocular_inputs_with_lengths(tmp_path, sample_counts=(60, 60))

    output = generate_ica_ocular_review(
        filtered_raw_paths=paths,
        ica_path=ica_path,
        report_path=report_path,
        output_path=tmp_path / "sub-0001_desc-icaeog_components.tsv",
        settings=OcularReviewSettings(enabled=True, eog_channels=["Fp1", "Fp2"]),
    )

    assert output.is_file()
    report = mne.open_report(report_path)
    document = " ".join(element.html for element in report._content if element.html)
    assert "no resolvable" in document.lower() or "no run" in document.lower()


def _blink_raw(bads: list[str] | None = None) -> mne.io.BaseRaw:
    """Frontally-weighted blink-like deflections on a small montage."""
    sfreq, duration = 200.0, 60.0
    n = int(sfreq * duration)
    rng = np.random.default_rng(0)
    names = ["Fp1", "Fp2", "Fz", "Cz", "Pz", "Oz"]
    blink = np.zeros(n)
    for onset in np.arange(2.0, duration - 2.0, 3.0):
        start = int(onset * sfreq)
        width = int(0.3 * sfreq)
        blink[start : start + width] += np.hanning(width)
    topo = np.array([1.0, 0.95, 0.5, 0.15, 0.05, 0.02])
    data = rng.normal(scale=1.0, size=(len(names), n)) + np.outer(topo, blink) * 90.0
    info = mne.create_info(names, sfreq, ["eeg"] * len(names))
    raw = mne.io.RawArray(data * 1e-6, info, verbose="ERROR")
    raw.set_montage("standard_1020", verbose="ERROR")
    if bads:
        raw.info["bads"] = list(bads)
    return raw


def _fitted_ica(raw: mne.io.BaseRaw) -> mne.preprocessing.ICA:
    ica = mne.preprocessing.ICA(n_components=3, random_state=0, max_iter=800)
    ica.fit(raw, verbose="ERROR")
    return ica


def test_a_bad_surrogate_does_not_abort_the_overlay() -> None:
    """Regression: PyPREP marking a surrogate bad raised "Channel(s) Fp1 not found".

    The surrogate exclusion exists to remove circularity — blinks were detected on those
    channels, so their attenuation is guaranteed. A surrogate that is a bad channel was
    never in the decomposition, so it carries no circularity and there is nothing to drop.
    Asking to drop it anyway ended the whole cohort at the ocular review.
    """
    from eeg_pipeline.preprocessing.ica_ocular_report import _plot_run_overlay

    raw = _blink_raw(bads=["Fp1"])
    ica = _fitted_ica(raw)
    assert "Fp1" not in ica.ch_names, "fixture must exclude the bad surrogate"

    figure, blink_count = _plot_run_overlay(
        raw,
        ica=ica,
        eog_channels=["Fp1", "Fp2"],
        surrogates=("Fp1", "Fp2"),
        recording_id="sub-0012_task-x_run-1",
    )

    assert blink_count > 0
    assert figure is not None


def test_the_panel_says_which_surrogate_was_outside_the_decomposition() -> None:
    """A scope that changes between subjects with nothing to explain it is unreadable."""
    from eeg_pipeline.preprocessing.ica_ocular_report import _plot_run_overlay

    raw = _blink_raw(bads=["Fp1"])
    ica = _fitted_ica(raw)

    figure, _ = _plot_run_overlay(
        raw,
        ica=ica,
        eog_channels=["Fp1", "Fp2"],
        surrogates=("Fp1", "Fp2"),
        recording_id="sub-0012_task-x_run-1",
    )

    text = " ".join(t.get_text() for t in figure.findobj(match=plt.Text))
    assert "Fp2" in text
    assert "Fp1" in text and "outside the decomposition" in text


def test_good_surrogates_are_still_excluded_from_the_panel() -> None:
    from eeg_pipeline.preprocessing.ica_ocular_report import _plot_run_overlay

    raw = _blink_raw()
    ica = _fitted_ica(raw)

    figure, _ = _plot_run_overlay(
        raw,
        ica=ica,
        eog_channels=["Fp1", "Fp2"],
        surrogates=("Fp1", "Fp2"),
        recording_id="sub-0001_task-x_run-1",
    )

    text = " ".join(t.get_text() for t in figure.findobj(match=plt.Text))
    assert "excluding surrogates Fp1, Fp2" in text
    assert "outside the decomposition" not in text
