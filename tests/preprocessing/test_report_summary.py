from __future__ import annotations

import mne
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.text import Text

from eeg_pipeline.preprocessing.report.summary import (
    DecompositionSummary,
    decomposition_summary_html,
    plot_variance_overview,
    summarize_decomposition,
)
from tests.utils.figure_layout import colliding_text


def _summary(**overrides) -> DecompositionSummary:
    defaults = {
        "n_channels": 64,
        "n_components": 62,
        "data_rank": 62,
        "condition_number": 184.0,
        "samples_per_squared_component": 189.0,
        "individual_variance": np.array([0.45, 0.20, 0.05] + [0.005] * 59),
        "excluded": (0, 1),
        # Measured jointly, so it is supplied rather than derived from the array above.
        # Deliberately not 0.65: the whole point is that the joint value is its own
        # measurement and is not the sum of the individual shares.
        "variance_removed": 0.60,
        "variance_floor": 0.01,
        "variance_above_floor": 0.59,
        "variance_below_floor": 0.02,
    }
    defaults.update(overrides)
    return DecompositionSummary(**defaults)


def test_variance_removed_is_the_measured_joint_value_not_a_sum() -> None:
    """Individual explained-variance ratios are not additive and must never be summed.

    ICA components are not orthogonal, so the variance a set of them accounts for is not
    the sum of what each accounts for alone. MNE says so explicitly, and on real data the
    gap is large: on sub-0015 the sum reads 99.0% removed where the joint value is 91.8%,
    understating retained variance roughly eightfold.
    """
    summary = _summary()

    assert summary.variance_removed == pytest.approx(0.60)
    assert summary.variance_retained == pytest.approx(0.40)
    # The sum of the individual shares is 0.65 and must not appear anywhere.
    assert summary.variance_removed != pytest.approx(0.65)


def test_no_exclusions_removes_no_variance() -> None:
    assert _summary(excluded=(), variance_removed=0.0).variance_removed == 0.0


def _overlapping_ica(seed: int = 0):
    """Fit an ICA whose excluded components share most of their scalp projection.

    Overlap is the whole point. When components project to near-identical topographies
    they each "explain" much of the same sensor variance, so their individual shares
    double-count it and summing them overshoots badly — which is the failure mode on real
    EEG, where blink and cardiac components routinely overlap frontally.
    """
    rng = np.random.default_rng(seed)
    n_channels, n_sources = 12, 6
    sources = rng.standard_normal((n_sources, 8000)) ** 3  # non-Gaussian, so ICA can work
    shared = rng.standard_normal((n_channels, 1))
    mixing = rng.standard_normal((n_channels, n_sources))
    # The three components that will be excluded are nudged onto one shared direction.
    mixing[:, :3] = shared + 0.10 * rng.standard_normal((n_channels, 3))
    data = (mixing @ sources).reshape(n_channels, 8, 1000).transpose(1, 0, 2) * 1e-7

    info = mne.create_info([f"C{index}" for index in range(n_channels)], 200.0, "eeg")
    epochs = mne.EpochsArray(data, info, verbose="ERROR")
    ica = mne.preprocessing.ICA(n_components=n_sources, random_state=seed, max_iter=800)
    ica.fit(epochs, verbose="ERROR")
    ica.exclude = [0, 1, 2]
    return ica, epochs


def _variance_ratio_spy(monkeypatch) -> list[list[int] | None]:
    """Record the component sets ``get_explained_variance_ratio`` is asked about."""
    calls: list[list[int] | None] = []
    original = mne.preprocessing.ICA.get_explained_variance_ratio

    def spy(self, inst, *, components=None, ch_type=None):
        calls.append(None if components is None else list(np.atleast_1d(components)))
        return original(self, inst, components=components, ch_type=ch_type)

    monkeypatch.setattr(mne.preprocessing.ICA, "get_explained_variance_ratio", spy)
    return calls


def test_summarize_asks_for_the_whole_exclusion_set_in_one_call(monkeypatch) -> None:
    """The regression guard: the joint call must actually be made.

    Pinned on the call rather than on a numeric gap between summing and joint. A
    synthetic ICA fitted to data that genuinely follows the ICA model recovers
    near-orthogonal components, where the two calculations agree to within 0.1% — so a
    numeric assertion would pass happily against the summing bug and guard nothing. The
    gap only opens up on real, ill-conditioned decompositions like sub-0015's, which
    cannot be checked into a test. Asking whether the set was scored in one call is the
    property that actually distinguishes correct code from the bug.
    """
    ica, epochs = _overlapping_ica()
    calls = _variance_ratio_spy(monkeypatch)

    summarize_decomposition(ica=ica, epochs=epochs)

    assert list(ica.exclude) in calls, (
        "the exclusion set was never scored jointly; variance was summed per component"
    )


def test_variance_removed_matches_a_direct_before_after_measurement() -> None:
    """Ground truth neither calculation can argue with: the variance that actually went."""
    ica, epochs = _overlapping_ica()

    summary = summarize_decomposition(ica=ica, epochs=epochs)

    before = epochs.get_data(copy=True)
    after = ica.apply(epochs.copy(), verbose="ERROR").get_data(copy=False)
    direct = 1.0 - after.var(axis=-1).sum() / before.var(axis=-1).sum()
    assert summary.variance_removed == pytest.approx(direct, abs=0.02)


def test_the_subgroup_split_is_also_measured_jointly(monkeypatch) -> None:
    """Splitting the exclusion set and summing each half repeats the same error."""
    ica, epochs = _overlapping_ica(seed=1)
    calls = _variance_ratio_spy(monkeypatch)

    summary = summarize_decomposition(ica=ica, epochs=epochs, variance_floor=0.05)
    above_count, above, below_count, below = summary.exclusion_cost(variance_floor=0.05)

    assert above_count + below_count == len(ica.exclude)
    above_set = [i for i in ica.exclude if summary.individual_variance[i] >= 0.05]
    below_set = [i for i in ica.exclude if summary.individual_variance[i] < 0.05]
    for subset in (above_set, below_set):
        if subset:
            assert subset in calls, f"subgroup {subset} was summed rather than scored jointly"
    for value in (above, below):
        assert 0.0 <= value <= 1.0


def test_restating_the_split_at_another_floor_is_refused() -> None:
    """The subgroup variances are measurements, so they cannot be re-derived on request.

    Silently returning values measured at a different floor than the caller asked for is
    how a correct-looking number ends up describing the wrong set of components.
    """
    summary = _summary()

    with pytest.raises(ValueError, match="variance floor"):
        summary.exclusion_cost(variance_floor=0.25)


def test_the_summary_warns_that_individual_shares_are_not_additive() -> None:
    """A reader who sums the per-component markers must be told not to."""
    document = decomposition_summary_html(_summary())

    assert "not additive" in document.lower()


def test_there_is_no_cumulative_variance_curve() -> None:
    """A running total over non-orthogonal components in arbitrary order means nothing.

    The x axis is ICA's own component order, not a variance ranking, so the curve was
    neither a scree plot nor a valid cumulative total. It cannot be fixed by computing it
    jointly, because the quantity it plots is not defined.
    """
    summary = _summary()
    figure = plot_variance_overview(summary)

    labels = [axis.get_ylabel().lower() for axis in figure.axes]
    assert not any("cumulative" in label for label in labels)
    # Asserted against the data rather than against the axis count, which the exclusion
    # status strip legitimately raised to two. A running total would show up as a series
    # whose values are the cumulative sum, on whichever axis it was drawn.
    running_total = np.cumsum(summary.individual_variance * 100.0)
    for axis in figure.axes:
        for line in axis.get_lines():
            ydata = np.asarray(line.get_ydata(), dtype=float)
            if ydata.shape == running_total.shape:
                assert not np.allclose(ydata, running_total)


def test_fitting_beyond_the_data_rank_is_flagged() -> None:
    """More components than rank splits sources and manufactures numerical components."""
    assert _summary(n_components=64, data_rank=62).is_rank_deficient
    assert not _summary(n_components=62, data_rank=62).is_rank_deficient


def test_insufficient_samples_per_component_is_flagged() -> None:
    assert _summary(samples_per_squared_component=8.0).is_under_determined
    assert not _summary(samples_per_squared_component=25.0).is_under_determined


def test_dimensions_the_fit_never_saw_are_counted() -> None:
    """Components fitted below the data rank leave dimensions outside the decomposition.

    ``ICA.apply`` restores the PCA components between ``n_components_`` and the data rank
    unmodified, so artifact living in them cannot be removed by any exclusion. A variance
    criterion collapses to very few components exactly when artifact dominates variance —
    on sub-0015, blink and cardiac took 85% of it and ``n_components=0.99`` fitted 22 of
    62 — so the count is the measurement that makes that collapse visible.
    """
    assert _summary(n_components=22, data_rank=62).unfitted_dimensions == 40
    assert _summary(n_components=62, data_rank=62).unfitted_dimensions == 0


def test_a_fit_that_spans_the_rank_leaves_nothing_outside_it() -> None:
    """Over-fitting past the rank is a different fault and must not read as negative."""
    assert _summary(n_components=64, data_rank=62).unfitted_dimensions == 0


def test_summary_html_states_what_the_unfitted_dimensions_mean() -> None:
    document = decomposition_summary_html(_summary(n_components=22, data_rank=62))

    assert "40" in document
    assert "unmodified" in document


def test_summary_html_omits_the_unfitted_dimensions_when_there_are_none() -> None:
    """A full-rank fit has no pass-through caveat, so the report must not invent one."""
    document = decomposition_summary_html(_summary(n_components=62, data_rank=62))

    assert "unmodified" not in document


def test_summary_html_states_the_variance_actually_removed() -> None:
    document = decomposition_summary_html(_summary())

    assert "60.0%" in document
    assert "40.0%" in document
    assert "2 of 62" in document


def test_summary_html_reports_rank_and_components_without_grading_them() -> None:
    """Rank against component count is a measurement, not a verdict."""
    document = decomposition_summary_html(_summary(n_components=64, data_rank=62))

    assert "64" in document and "62" in document
    for verdict in ("numerical rather than physiological", "&#9888;", "under-determined"):
        assert verdict not in document


def test_summary_html_never_grades_the_fit() -> None:
    document = decomposition_summary_html(_summary())

    for verdict in ("numerical rather than physiological", "under-determined", "&#9888;"):
        assert verdict not in document


def test_variance_overview_marks_excluded_components() -> None:
    summary = _summary()

    figure = plot_variance_overview(summary)

    share_axis = figure.axes[0]
    assert "60.0%" in share_axis.get_title()
    # Excluded and retained are drawn as two scatter collections, one per group.
    plotted = sum(
        collection.get_offsets().shape[0]
        for collection in share_axis.collections
        if isinstance(collection, PathCollection)
    )
    assert plotted == summary.n_components


def test_variance_is_not_encoded_as_bar_length_on_a_log_axis() -> None:
    """Bar length measures from zero, which is at -inf on a log axis and so meaningless."""
    figure = plot_variance_overview(_summary())
    share_axis = figure.axes[0]

    assert share_axis.get_yscale() == "log"
    assert len(share_axis.patches) == 0


def test_variance_axis_does_not_claim_an_ordering_it_does_not_use() -> None:
    """The x axis is component index, not variance rank, and must not say otherwise.

    ICA returns components in its own order, so the variance markers are not monotonic.
    A figure promising a decreasing sort invites the reader to read it as a scree plot
    and conclude the decomposition is degenerate when it is not.

    The caveat sits in the figure's footnote rather than the axis label, beside the
    non-additivity warning: both say what this panel is not, and the shares frequently
    do fall monotonically, which is exactly when the misreading happens.
    """
    summary = _summary(individual_variance=np.array([0.1, 0.5, 0.05, 0.35]), n_components=4)

    figure = plot_variance_overview(summary)

    xlabel = figure.axes[-1].get_xlabel()
    assert "decreasing variance" not in xlabel
    # Found across every text artist rather than in ``figure.texts``: the footnote is a
    # ``supxlabel``, which the layout engine places but which that list does not hold.
    footnotes = " ".join(text.get_text().lower() for text in figure.findobj(Text))
    assert "component order carries no ranking" in footnotes


def test_the_variance_footnote_does_not_print_over_the_axis_label() -> None:
    """The footnote was drawn in figure coordinates, which no layout engine consults.

    ``figure.text`` at y=0.005 sits below everything ``constrained_layout`` measured,
    which on sub-0012 was the status strip's own "ICA component" label. The two printed
    through each other and neither could be read.
    """
    figure = plot_variance_overview(_summary())

    assert colliding_text(figure) == []


def test_summarize_decomposition_recovers_rank_and_variance() -> None:
    """An average-referenced montage loses exactly one rank, which must be detected."""
    rng = np.random.default_rng(0)
    info = mne.create_info([f"C{index}" for index in range(8)], 100.0, "eeg")
    data = rng.normal(0, 1e-5, (6, 8, 400))
    data -= data.mean(axis=1, keepdims=True)  # average reference drops one rank
    epochs = mne.EpochsArray(data, info, verbose="ERROR")

    ica = mne.preprocessing.ICA(n_components=7, random_state=0, max_iter=200)
    ica.fit(epochs, verbose="ERROR")
    ica.exclude = [0]

    summary = summarize_decomposition(ica=ica, epochs=epochs)

    assert summary.data_rank == 7
    assert summary.n_components == 7
    assert not summary.is_rank_deficient
    assert 0.0 < summary.variance_removed < 1.0
    assert summary.individual_variance.shape == (7,)


def test_component_overview_shows_every_component_on_one_sheet() -> None:
    """The per-component panels hide behind sliders; this sheet must show all of them."""
    from types import SimpleNamespace

    import matplotlib

    matplotlib.use("Agg")
    from eeg_pipeline.preprocessing.report.summary import plot_component_overview

    rng = np.random.default_rng(0)
    info = mne.create_info([f"C{index}" for index in range(8)], 100.0, "eeg")
    info.set_montage(
        mne.channels.make_dig_montage(
            ch_pos={
                name: pos
                for name, pos in zip(info["ch_names"], rng.normal(0, 0.05, (8, 3)), strict=True)
            },
            coord_frame="head",
        )
    )
    epochs = mne.EpochsArray(rng.normal(0, 1e-5, (5, 8, 200)), info, verbose="ERROR")
    ica = mne.preprocessing.ICA(n_components=5, random_state=0, max_iter=200)
    ica.fit(epochs, verbose="ERROR")
    ica.exclude = [1, 3]
    labels = [SimpleNamespace(label="brain", probability=0.9) for _ in range(5)]

    figure = plot_component_overview(ica=ica, labels=labels, columns=3)

    titles = [axis.get_title() for axis in figure.axes]
    assert len(titles) == 5
    assert titles[0].startswith("IC000")
    assert "×" in titles[1] and "×" in titles[3]
    assert "×" not in titles[0]
    assert "2 excluded" in figure._suptitle.get_text()


def test_component_overview_shades_the_excluded_panels() -> None:
    """``plot_topomap`` turns the axis off, so spines cannot carry the exclusion mark.

    The caption promises excluded components are outlined. Setting spine visibility after
    ``plot_topomap`` has called ``set_axis_off`` is silently discarded, which left the
    status resting on a 0.25-grey title against a black one. A patch drawn in axes
    coordinates still renders with the axis off, so that is what marks the decision.
    """
    from types import SimpleNamespace

    import matplotlib

    matplotlib.use("Agg")
    from eeg_pipeline.preprocessing.report.summary import plot_component_overview

    rng = np.random.default_rng(0)
    info = mne.create_info([f"C{index}" for index in range(8)], 100.0, "eeg")
    info.set_montage(
        mne.channels.make_dig_montage(
            ch_pos={
                name: pos
                for name, pos in zip(info["ch_names"], rng.normal(0, 0.05, (8, 3)), strict=True)
            },
            coord_frame="head",
        )
    )
    epochs = mne.EpochsArray(rng.normal(0, 1e-5, (5, 8, 200)), info, verbose="ERROR")
    ica = mne.preprocessing.ICA(n_components=5, random_state=0, max_iter=200)
    ica.fit(epochs, verbose="ERROR")
    ica.exclude = [1, 3]
    labels = [SimpleNamespace(label="brain", probability=0.9) for _ in range(5)]

    figure = plot_component_overview(ica=ica, labels=labels, columns=3)

    shaded = [
        index
        for index, axis in enumerate(figure.axes)
        if any(getattr(patch, "_is_exclusion_mark", False) for patch in axis.patches)
    ]
    assert shaded == [1, 3]
    # The caption must describe the mark that is actually drawn.
    assert "outlined" not in figure._suptitle.get_text()


def test_component_overview_rejects_mismatched_labels() -> None:
    from types import SimpleNamespace

    from eeg_pipeline.preprocessing.report.summary import plot_component_overview

    ica = SimpleNamespace(n_components_=4, exclude=[])

    with pytest.raises(ValueError, match="do not match"):
        plot_component_overview(ica=ica, labels=[SimpleNamespace(label="brain", probability=1.0)])


def test_raster_format_is_one_mne_supports() -> None:
    from mne.report.report import _ALLOWED_IMAGE_FORMATS

    from eeg_pipeline.preprocessing.report.style import (
        REPORT_IMAGE_FORMAT,
        REPORT_RASTER_IMAGE_FORMAT,
    )

    assert REPORT_RASTER_IMAGE_FORMAT in _ALLOWED_IMAGE_FORMATS
    assert REPORT_IMAGE_FORMAT in _ALLOWED_IMAGE_FORMATS


def test_figure_lists_never_request_svg() -> None:
    """MNE's slider template emits the invalid MIME type ``image/svg`` for figure lists.

    Browsers refuse ``data:image/svg;base64`` (the valid type is ``image/svg+xml``), so a
    slider built from SVG renders nothing at all. Lists must stay raster.
    """
    from eeg_pipeline.preprocessing.report.style import report_image_format

    for kwargs in ({}, {"has_dense_image": False}, {"has_dense_image": True}):
        assert report_image_format(is_figure_list=True, **kwargs) != "svg"
    # Single figures are inlined by MNE's image template, which does handle SVG.
    assert report_image_format() == "svg"
    assert report_image_format(has_dense_image=True) != "svg"


def test_no_report_module_passes_svg_to_a_figure_list() -> None:
    """Guard the call sites, not just the helper."""
    import re

    from tests import REPO_ROOT

    root = REPO_ROOT / "eeg_pipeline" / "preprocessing"
    offenders = []
    for path in sorted(root.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for match in re.finditer(r"add_figure\((.*?)\n    \)", source, re.S):
            block = match.group(1)
            passes_list = "fig=[" in block or "fig=figures" in block
            if passes_list and "REPORT_IMAGE_FORMAT" in block:
                offenders.append(f"{path.name}: {block[:60]}")
    assert offenders == []


def test_renaming_a_panel_does_not_leave_a_stale_duplicate() -> None:
    """add_figure(replace=True) matches on title, so renames must be cleared by tag."""
    from types import SimpleNamespace

    from eeg_pipeline.preprocessing.report.organize import remove_tagged_content

    removed = []
    report = SimpleNamespace(
        _content=[
            SimpleNamespace(name="Component properties", tags=("ica-decomposition",)),
            SimpleNamespace(name="Sensor variance", tags=("ica-decomposition",)),
            SimpleNamespace(name="Unrelated", tags=("ica", "ecg")),
        ],
        remove=lambda title, tags, remove_all: removed.append(title),
    )

    remove_tagged_content(report, tag="ica-decomposition")

    assert sorted(removed) == ["Component properties", "Sensor variance"]


def test_run_colours_are_shared_across_report_modules() -> None:
    """A run must keep one colour in every panel, or panels cannot be cross-read."""
    from eeg_pipeline.preprocessing import ica_cardiac_report
    from eeg_pipeline.preprocessing.report import summary as summary_module
    from eeg_pipeline.preprocessing.report.style import OKABE_ITO, RUN_COLORS

    assert ica_cardiac_report.RUN_COLORS is RUN_COLORS
    assert summary_module.RUN_COLORS is RUN_COLORS
    assert set(RUN_COLORS) <= set(OKABE_ITO.values())


def test_quantile_band_needs_enough_runs_to_mean_anything() -> None:
    """With a handful of runs the 16-84% band is just the min-max envelope."""
    from eeg_pipeline.preprocessing.ica_cardiac_report import (
        MINIMUM_RUNS_FOR_QUANTILE_BAND,
    )

    assert MINIMUM_RUNS_FOR_QUANTILE_BAND >= 5


def test_expected_rank_accounts_for_the_average_reference() -> None:
    """A rank one below the channel count is what an average reference produces."""
    summary = _summary(n_channels=60, uses_average_reference=True, data_rank=59)

    assert summary.expected_rank == 59
    assert summary.rank_shortfall == 0
    assert "average reference" in summary.rank_accounting


def test_expected_rank_without_a_reference_is_the_channel_count() -> None:
    summary = _summary(n_channels=60, uses_average_reference=False, data_rank=60)

    assert summary.expected_rank == 60
    assert summary.rank_shortfall == 0


def test_a_rank_below_the_expectation_is_reported_as_a_shortfall() -> None:
    summary = _summary(n_channels=60, uses_average_reference=True, data_rank=55)

    assert summary.rank_shortfall == 4


def test_rank_accounting_names_the_excluded_bad_channels() -> None:
    summary = _summary(n_channels=58, n_bad_channels=2, uses_average_reference=True)

    assert "2 excluded as bad" in summary.rank_accounting


def test_summary_html_states_the_expected_rank_beside_the_measured_one() -> None:
    document = decomposition_summary_html(
        _summary(n_channels=60, uses_average_reference=True, data_rank=59)
    )

    assert "Expected rank" in document
    assert "average reference" in document


def test_summary_html_flags_only_a_genuine_shortfall() -> None:
    benign = decomposition_summary_html(
        _summary(n_channels=60, uses_average_reference=True, data_rank=59)
    )
    deficient = decomposition_summary_html(
        _summary(n_channels=60, uses_average_reference=True, data_rank=55)
    )

    assert "below expected" not in benign
    assert "4 below expected" in deficient


def test_summarize_decomposition_excludes_bad_channels() -> None:
    """Epochs.pick("eeg") keeps bads; a channel the ICA never saw must not be counted."""
    import mne

    rng = np.random.default_rng(0)
    info = mne.create_info([f"C{index}" for index in range(10)], 250.0, "eeg")
    epochs = mne.EpochsArray(rng.normal(0, 1e-5, (25, 10, 250)), info, verbose="ERROR")
    epochs.info["bads"] = ["C0", "C1"]
    ica = mne.preprocessing.ICA(n_components=4, random_state=0, max_iter=200)
    ica.fit(epochs, verbose="ERROR")

    summary = summarize_decomposition(ica=ica, epochs=epochs)

    assert summary.n_channels == 8
    assert summary.n_bad_channels == 2


def test_removal_topography_separates_focal_from_uniform_removal() -> None:
    """A large variance figure means opposite things in these two cases."""
    from eeg_pipeline.preprocessing.report.summary import RemovalTopography

    import mne

    info = mne.create_info([f"C{index}" for index in range(10)], 250.0, "eeg")
    focal = RemovalTopography(
        channel_names=tuple(info["ch_names"]),
        change_db=np.array([-24.0, -19.0] + [-0.4] * 8),
        info=info,
    )
    uniform = RemovalTopography(
        channel_names=tuple(info["ch_names"]),
        change_db=np.full(10, -6.0),
        info=info,
    )

    assert focal.spatial_spread_db > 10.0
    assert uniform.spatial_spread_db < 1.0
    assert focal.worst_channel == "C0"


def _removal_topography():
    """A removal that is focal, so the two panels have something to disagree about."""
    import mne
    import numpy as np

    from eeg_pipeline.preprocessing.report.summary import RemovalTopography

    names = ["Fp1", "Fp2", "Cz", "Pz", "O1", "O2"]
    info = mne.create_info(names, 250.0, "eeg")
    info.set_montage("standard_1020", verbose="ERROR")
    change_db = np.array([-10.0, -9.0, -3.0, -2.5, -4.0, -3.5])
    return RemovalTopography(channel_names=tuple(names), change_db=change_db, info=info)


def test_the_removal_figure_names_its_units_exactly_once() -> None:
    """The colourbar and the ranked panel measure the same thing in the same units.

    Constrained layout puts them side by side, so labelling both printed "Amplitude
    change (dB)" twice, overlapping, in the gap between the panels.
    """
    import matplotlib

    matplotlib.use("Agg")
    from eeg_pipeline.preprocessing.report.summary import plot_removal_topography

    figure = plot_removal_topography(_removal_topography())

    labels = [axis.get_ylabel() for axis in figure.axes] + [
        axis.get_xlabel() for axis in figure.axes
    ]
    assert labels.count("Amplitude change (dB)") == 1


def _ledger(**overrides) -> str:
    """Render the ledger for a decomposition shaped like the one that motivated it.

    Component 2 is the case the report could not explain: ICLabel called it brain with
    high confidence and it was excluded anyway, by MNE's own ECG correlation.
    """
    from eeg_pipeline.preprocessing.report.summary import exclusion_ledger_html

    summary = _summary(
        n_components=4,
        individual_variance=np.array([0.45, 0.20, 0.004, 0.10]),
        excluded=(0, 2),
    )
    defaults = {
        "status_descriptions": (
            "Auto-detected eye blink (MNE-ICALabel)",
            "",
            "Auto-detected ECG artifact (MNE)",
            "",
        )
    }
    defaults.update(overrides)
    return exclusion_ledger_html(summary, **defaults)


def test_the_ledger_names_the_detector_behind_each_exclusion() -> None:
    """The ICLabel table showed "brain 0.929" beside "Excluded: yes" and stopped there.

    The reason was already written to ``*_proc-ica_components.tsv`` and never reached the
    report, so the one panel that stated the decision made it look like a contradiction.
    """
    document = _ledger()

    assert "Auto-detected ECG artifact (MNE)" in document
    assert "Auto-detected eye blink (MNE-ICALabel)" in document


def test_the_ledger_is_a_census_rather_than_a_list_of_removals() -> None:
    """A reviewer checking for an over-eager detector needs the components it left alone
    on the same page; a list of exclusions alone cannot show what was spared."""
    document = _ledger()

    for component in range(4):
        assert f"ICA{component:03d}" in document


def test_the_ledger_carries_the_variance_each_exclusion_cost() -> None:
    """Nine of this subject's sixteen exclusions were below 1% of variance each. Reason
    without size cannot separate removing an artifact from spending a dimension."""
    document = _ledger()

    assert "0.4%" in document


def test_the_recorded_measurements_are_the_ones_a_drop_decision_reads() -> None:
    """The decomposition panel states these in prose. A decision to drop a subject should
    not have to parse that prose back out of the HTML."""
    from eeg_pipeline.preprocessing.report.summary import decomposition_measurements

    measurements = decomposition_measurements(_summary(n_components=22, excluded=(0, 1)))

    assert measurements["n_components"] == 22
    assert measurements["n_excluded"] == 2
    assert measurements["variance_removed"] == pytest.approx(0.60)
    assert measurements["data_rank"] == 62


def test_recorded_measurements_are_json_safe() -> None:
    """The record is written as JSON, and numpy scalars are not serialisable."""
    import json

    from eeg_pipeline.preprocessing.report.summary import decomposition_measurements

    json.dumps(decomposition_measurements(_summary()))
