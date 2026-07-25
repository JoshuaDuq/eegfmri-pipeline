from __future__ import annotations

import mne
import numpy as np
import pytest
from matplotlib.collections import PathCollection

from eeg_pipeline.preprocessing.report.summary import (
    DecompositionSummary,
    decomposition_summary_html,
    plot_variance_overview,
    summarize_decomposition,
)


def _summary(**overrides) -> DecompositionSummary:
    defaults = {
        "n_channels": 64,
        "n_components": 62,
        "data_rank": 62,
        "condition_number": 184.0,
        "samples_per_squared_component": 189.0,
        "explained_variance": np.array([0.45, 0.20, 0.05] + [0.005] * 59),
        "excluded": (0, 1),
    }
    defaults.update(overrides)
    return DecompositionSummary(**defaults)


def test_variance_removed_sums_only_the_excluded_components() -> None:
    summary = _summary()

    assert summary.variance_removed == pytest.approx(0.65)
    assert summary.variance_retained == pytest.approx(0.35)


def test_no_exclusions_removes_no_variance() -> None:
    assert _summary(excluded=()).variance_removed == 0.0


def test_fitting_beyond_the_data_rank_is_flagged() -> None:
    """More components than rank splits sources and manufactures numerical components."""
    assert _summary(n_components=64, data_rank=62).is_rank_deficient
    assert not _summary(n_components=62, data_rank=62).is_rank_deficient


def test_insufficient_samples_per_component_is_flagged() -> None:
    assert _summary(samples_per_squared_component=8.0).is_under_determined
    assert not _summary(samples_per_squared_component=25.0).is_under_determined


def test_summary_html_states_the_variance_actually_removed() -> None:
    document = decomposition_summary_html(_summary())

    assert "65.0%" in document
    assert "35.0%" in document
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
    assert "65.0%" in share_axis.get_title()
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
    assert summary.explained_variance.shape == (7,)


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
