from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.preprocessing.band_ica_report import (
    BAND_ICA_DEFINITIONS,
    EXPLORATORY_BAND_SECTION,
    BandIcaReportSettings,
    generate_band_ica_report,
)


def plt_figure():
    """A throwaway figure for tests that only care about report bookkeeping."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    axis.plot([0.0, 1.0])
    plt.close(figure)
    return figure


def test_band_definitions_match_requested_report_sections() -> None:
    assert [(band.slug, band.fmin, band.fmax) for band in BAND_ICA_DEFINITIONS] == [
        ("deltatheta", 1.0, 8.0),
        ("alpha", 8.0, 13.0),
        ("beta", 13.0, 30.0),
        ("gamma", 30.0, 100.0),
        ("broadband1to30", 1.0, 30.0),
    ]


def test_band_report_settings_fail_fast_on_invalid_values() -> None:
    with pytest.raises(ValueError, match="fit_decim"):
        BandIcaReportSettings.from_mapping({"fit_decim": 0})
    with pytest.raises(ValueError, match="frequency_step_hz"):
        BandIcaReportSettings.from_mapping({"tfr": {"frequency_step_hz": 0}})
    with pytest.raises(ValueError, match="time_step_s"):
        BandIcaReportSettings.from_mapping({"tfr": {"time_step_s": 0}})


def test_band_report_settings_parse_metadata_comparison() -> None:
    settings = BandIcaReportSettings.from_mapping(
        {
            "comparisons": [
                {
                    "name": "high_vs_low",
                    "column": "stimulus_temp",
                    "group_a": {"label": "High", "values": [48.3, 49.3]},
                    "group_b": {"label": "Low", "values": [44.3, 45.3]},
                }
            ]
        }
    )

    comparison = settings.comparisons[0]
    assert comparison.name == "high_vs_low"
    assert comparison.column == "stimulus_temp"
    assert comparison.group_a.values == (48.3, 49.3)


def test_comparison_masks_select_configured_clean_event_values() -> None:
    from eeg_pipeline.preprocessing.band_ica_report import _comparison_masks

    settings = BandIcaReportSettings.from_mapping(
        {
            "comparisons": [
                {
                    "name": "pain",
                    "column": "pain_binary_coded",
                    "group_a": {"label": "Painful", "values": [1]},
                    "group_b": {"label": "Non-painful", "values": [0]},
                }
            ]
        }
    )
    metadata = pd.DataFrame({"pain_binary_coded": [0, 1, 1, 0]})

    group_a, group_b = _comparison_masks(metadata, settings.comparisons[0])

    np.testing.assert_array_equal(group_a, [False, True, True, False])
    np.testing.assert_array_equal(group_b, [True, False, False, True])


@pytest.mark.parametrize(
    ("band_index", "window_seconds", "smoothing_hz"),
    [(0, 3.0, 1.0), (1, 2.0, 1.5), (2, 2.0, 2.5), (3, 1.0, 5.0)],
)
def test_fieldtrip_tfr_uses_requested_band_parameters(
    band_index: int,
    window_seconds: float,
    smoothing_hz: float,
) -> None:
    from eeg_pipeline.preprocessing.band_ica_report import _fieldtrip_tfr

    from eeg_pipeline.preprocessing.band_ica_report import _tfr_parameter_groups

    band = BAND_ICA_DEFINITIONS[band_index]
    # The display grid is tied to the smoothing, so the expected frequencies come from
    # the same source of truth rather than from a hardcoded 1 Hz arange.
    ((_, frequencies),) = _tfr_parameter_groups(band, BandIcaReportSettings())
    power = np.ones((2, len(frequencies), 221))
    with patch(
        "eeg_pipeline.preprocessing.band_ica_report.mne.time_frequency.tfr_array_multitaper",
        return_value=power,
    ) as multitaper:
        _fieldtrip_tfr(
            data=np.ones((4, 2, 2201)),
            sfreq=100.0,
            times=np.linspace(-7.0, 15.0, 2201),
            band=band,
            settings=BandIcaReportSettings(),
        )

    kwargs = multitaper.call_args.kwargs
    np.testing.assert_allclose(kwargs["n_cycles"], frequencies * window_seconds)
    assert kwargs["time_bandwidth"] == 2.0 * window_seconds * smoothing_hz
    assert kwargs["decim"] == 10


def test_generate_band_report_writes_each_band_as_exploratory_outputs(tmp_path) -> None:
    epochs = SimpleNamespace(info={"sfreq": 250.0})
    report = Mock()
    # Content list the tag-keyed rebuild iterates; a bare Mock is not iterable.
    report._content = []
    report_path = tmp_path / "sub-0001_report.h5"
    report_path.write_text("report", encoding="utf-8")
    (tmp_path / "sub-0001_proc-ica_ica.fif").write_text("ica", encoding="utf-8")
    output_dir = tmp_path / "band-ica"

    fitted_icas = []

    def fit_band_ica(**_kwargs):
        fitted = SimpleNamespace(
            n_components_=2,
            exclude=[],
            save=Mock(),
        )
        fitted_icas.append(fitted)
        return fitted

    component_labels = [
        SimpleNamespace(label="brain", probability=0.9),
        SimpleNamespace(label="muscle artifact", probability=0.8),
    ]
    figures = [plt_figure(), plt_figure()]

    with (
        patch(
            "eeg_pipeline.preprocessing.band_ica_report.mne.read_epochs",
            return_value=epochs,
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report.open_subject_report",
            return_value=report,
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report.read_ica_with_reviewed_exclusions",
            return_value=SimpleNamespace(n_components_=2, exclude=[]),
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._fit_band_ica",
            side_effect=fit_band_ica,
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._band_epochs",
            return_value=epochs,
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._label_components",
            return_value=component_labels,
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._build_component_figures",
            return_value=figures,
        ),
        patch("eeg_pipeline.preprocessing.band_ica_report._add_standard_component_review"),
    ):
        generated = generate_band_ica_report(
            epochs_path=tmp_path / "sub-0001_proc-icafit_epo.fif",
            report_path=report_path,
            output_dir=output_dir,
            output_prefix="sub-0001",
            random_state=42,
            settings=BandIcaReportSettings(),
        )

    # Two files per band (ICA + component table), plus the one exploratory report holding
    # every band's figures.
    assert len(generated) == 2 * len(BAND_ICA_DEFINITIONS) + 1
    assert len(fitted_icas) == len(BAND_ICA_DEFINITIONS)
    assert all(ica.exclude == [] for ica in fitted_icas)
    # The figures no longer go into the subject report; see the separate-file tests below.
    assert report.add_figure.call_count == 0
    assert report.save.call_count == 2
    assert report.save.call_args_list[0].args == (report_path,)
    assert report.save.call_args_list[1].args == (report_path.with_suffix(".html"),)


def _run_band_report(tmp_path, *, settings, report):
    """Drive ``generate_band_ica_report`` with every expensive step stubbed out."""
    epochs = SimpleNamespace(info={"sfreq": 250.0})
    report_path = tmp_path / "sub-0001_report.h5"
    report_path.write_text("report", encoding="utf-8")
    (tmp_path / "sub-0001_proc-ica_ica.fif").write_text("ica", encoding="utf-8")

    with (
        patch("eeg_pipeline.preprocessing.band_ica_report.mne.read_epochs", return_value=epochs),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report.open_subject_report",
            return_value=report,
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report.read_ica_with_reviewed_exclusions",
            return_value=SimpleNamespace(n_components_=2, exclude=[]),
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._fit_band_ica",
            return_value=SimpleNamespace(n_components_=2, exclude=[], save=Mock()),
        ),
        patch("eeg_pipeline.preprocessing.band_ica_report._band_epochs", return_value=epochs),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._label_components",
            return_value=[
                SimpleNamespace(label="brain", probability=0.9),
                SimpleNamespace(label="muscle artifact", probability=0.8),
            ],
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._build_component_figures",
            return_value=[plt_figure(), plt_figure()],
        ),
        patch("eeg_pipeline.preprocessing.band_ica_report._add_standard_component_review"),
    ):
        return generate_band_ica_report(
            epochs_path=tmp_path / "sub-0001_proc-icafit_epo.fif",
            report_path=report_path,
            output_dir=tmp_path / "band-ica",
            output_prefix="sub-0001",
            random_state=42,
            settings=settings,
        )


def test_the_exploratory_bands_are_written_to_their_own_report(tmp_path) -> None:
    """Thirty megabytes of content the section itself says controls nothing.

    Measured on sub-0015: the five exploratory sliders are 30 MB of a 101 MB report, and
    the guide above them states that ICLabel was not validated for narrow-band
    decompositions, that component numbers correspond to nothing, and that artifact
    removal is decided elsewhere. Keeping it costs every reader of the subject report the
    download; moving it beside them costs the reader who wants it one click.
    """
    report = Mock()
    report._content = []

    generated = _run_band_report(tmp_path, settings=BandIcaReportSettings(), report=report)

    exploratory = tmp_path / "band-ica" / "sub-0001_desc-exploratorybandica_report.html"
    assert exploratory in generated
    assert report.add_figure.call_count == 0


def test_the_subject_report_links_to_the_exploratory_file_it_no_longer_holds(tmp_path) -> None:
    """Content moved without a pointer is content deleted, as far as a reader is concerned."""
    report = Mock()
    report._content = []

    _run_band_report(tmp_path, settings=BandIcaReportSettings(), report=report)

    linking = [
        call
        for call in report.add_html.call_args_list
        if call.kwargs.get("section") == EXPLORATORY_BAND_SECTION
    ]
    assert linking, "the subject report must keep a pointer to the exploratory file"
    document = "".join(str(call.kwargs["html"]) for call in linking)
    assert "sub-0001_desc-exploratorybandica_report.html" in document


def test_keeping_the_exploratory_bands_inline_is_still_available(tmp_path) -> None:
    """A site that wants one self-contained file must not have that taken away."""
    report = Mock()
    report._content = []

    generated = _run_band_report(
        tmp_path,
        settings=BandIcaReportSettings(exploratory_separate_file=False),
        report=report,
    )

    assert report.add_figure.call_count == len(BAND_ICA_DEFINITIONS)
    sections = [call.kwargs["section"] for call in report.add_figure.call_args_list]
    assert sections == [EXPLORATORY_BAND_SECTION] * len(BAND_ICA_DEFINITIONS)
    assert not any(str(path).endswith("exploratorybandica_report.html") for path in generated)


def test_generate_band_report_adds_authoritative_standard_ica_review(tmp_path) -> None:
    from eeg_pipeline.preprocessing.band_ica_report import ComponentLabel

    epochs = SimpleNamespace(info={"sfreq": 250.0})
    standard_ica = SimpleNamespace(n_components_=1, exclude=[])
    band_ica = SimpleNamespace(n_components_=1, exclude=[], save=Mock())
    labels = [ComponentLabel("brain", 0.9)]
    report = Mock()
    report._content = []
    report_path = tmp_path / "sub-0001_report.h5"
    report_path.write_text("report", encoding="utf-8")
    standard_ica_path = tmp_path / "sub-0001_proc-ica_ica.fif"
    standard_ica_path.write_text("ica", encoding="utf-8")

    with (
        patch("eeg_pipeline.preprocessing.band_ica_report.mne.read_epochs", return_value=epochs),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report.read_ica_with_reviewed_exclusions",
            return_value=standard_ica,
        ) as read_ica,
        patch(
            "eeg_pipeline.preprocessing.band_ica_report.open_subject_report",
            return_value=report,
        ),
        patch("eeg_pipeline.preprocessing.band_ica_report._fit_band_ica", return_value=band_ica),
        patch("eeg_pipeline.preprocessing.band_ica_report._band_epochs", return_value=epochs),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._label_components",
            return_value=labels,
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._build_component_figures",
            return_value=[plt_figure()],
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._add_standard_component_review"
        ) as add_standard_review,
    ):
        generate_band_ica_report(
            epochs_path=tmp_path / "sub-0001_proc-icafit_epo.fif",
            report_path=report_path,
            output_dir=tmp_path / "band-specific-ica",
            output_prefix="sub-0001",
            random_state=42,
            settings=BandIcaReportSettings(),
        )

    read_ica.assert_called_once_with(standard_ica_path)
    assert add_standard_review.call_args.kwargs["ica"] is standard_ica
    assert add_standard_review.call_args.kwargs["epochs"] is epochs
    assert add_standard_review.call_args.kwargs["labels"] == labels
    assert (
        add_standard_review.call_args.kwargs["analysis_status"] == "Pending provisional task epochs"
    )


def test_review_guide_makes_empty_comparison_configuration_explicit() -> None:
    from eeg_pipeline.preprocessing.band_ica_report import _review_guide_html

    guide = _review_guide_html(
        BandIcaReportSettings(),
        "Pending provisional task epochs",
    )

    assert "No condition comparisons configured" in guide
    assert "do not correspond numerically" in guide


def test_component_diagnostics_use_only_requested_frequency_range() -> None:
    from eeg_pipeline.preprocessing.band_ica_report import _component_spectrum

    frequencies = np.array([0.5, 1.0, 4.0, 8.0, 12.0])
    spectrum = np.arange(10, dtype=float).reshape(2, 5)

    selected_frequencies, selected_spectrum = _component_spectrum(
        frequencies=frequencies,
        spectrum=spectrum,
        fmin=1.0,
        fmax=8.0,
    )

    np.testing.assert_array_equal(selected_frequencies, [1.0, 4.0, 8.0])
    np.testing.assert_array_equal(selected_spectrum, spectrum[:, 1:4])


def test_source_diagnostics_include_misc_typed_ica_sources() -> None:
    from eeg_pipeline.preprocessing.band_ica_report import _source_diagnostics

    spectrum = SimpleNamespace(
        ch_names=["ICA000", "ICA001"],
        freqs=np.array([8.0, 10.0, 13.0]),
        get_data=Mock(return_value=np.ones((2, 2, 3))),
    )
    sources = SimpleNamespace(
        info={"sfreq": 100.0},
        # Spans the default baseline, which must clear the DPSS half-window before 0.
        times=np.linspace(-6.0, 1.0, 20),
        compute_psd=Mock(return_value=spectrum),
        get_data=Mock(return_value=np.ones((2, 2, 20))),
    )
    ica = SimpleNamespace(get_sources=Mock(return_value=sources))

    with patch(
        "eeg_pipeline.preprocessing.band_ica_report.mne.time_frequency.tfr_array_multitaper",
        return_value=np.ones((2, 6, 10)),
    ) as multitaper:
        _source_diagnostics(
            ica=ica,
            epochs=SimpleNamespace(),
            band=BAND_ICA_DEFINITIONS[1],
            settings=BandIcaReportSettings(time_step_s=0.02),
        )

    assert sources.compute_psd.call_args.kwargs["picks"] == "all"
    spectrum.get_data.assert_called_once_with(picks=spectrum.ch_names)
    assert multitaper.call_args.kwargs["time_bandwidth"] == 6.0
    np.testing.assert_allclose(
        multitaper.call_args.kwargs["n_cycles"],
        np.arange(8.0, 14.0) * 2.0,
    )


def test_component_figures_pair_topomap_spectrum_tfr_and_icalabel() -> None:
    from eeg_pipeline.preprocessing.band_ica_report import (
        ComponentLabel,
        _build_component_figures,
    )

    ica = SimpleNamespace(n_components_=2, plot_components=Mock())
    diagnostics = (
        np.array([8.0, 10.0, 13.0]),
        np.ones((2, 3)),
        np.array([8.0, 10.0, 13.0]),
        np.array([-1.0, 0.0, 1.0]),
        np.ones((2, 3, 3)),
    )
    labels = [
        ComponentLabel("brain", 0.91),
        ComponentLabel("muscle artifact", 0.82),
    ]

    with patch(
        "eeg_pipeline.preprocessing.band_ica_report._source_diagnostics",
        return_value=diagnostics,
    ):
        figures = _build_component_figures(
            ica=ica,
            epochs=SimpleNamespace(),
            band=BAND_ICA_DEFINITIONS[1],
            labels=labels,
            settings=BandIcaReportSettings(),
        )

    assert len(figures) == 2
    assert ica.plot_components.call_count == 2
    assert "exploratory ICLabel: brain (0.910)" in figures[0]._suptitle.get_text()
    assert [axis.get_title() for axis in figures[0].axes[:3]] == [
        "ICA000 topomap",
        "Source spectrum",
        "Source time-frequency power",
    ]


def test_condition_groups_share_scale_but_difference_has_own_scale() -> None:
    from eeg_pipeline.preprocessing.band_ica_report import _comparison_color_limits

    group_a = np.full((1, 100), 2.0)
    group_b = np.full((1, 100), -3.0)

    condition_limit, difference_limit = _comparison_color_limits(group_a, group_b)

    assert condition_limit == pytest.approx(3.0)
    assert difference_limit == pytest.approx(5.0)


def test_every_power_panel_on_one_slide_shares_one_scale() -> None:
    """Grand average and the conditions are the same quantity, so they need one scale.

    A dossier draws baseline-relative power in the grand-average panel and again in each
    condition panel. Scaling them separately means a condition that looks stronger than
    the grand average it contributes to may only have a narrower colourbar, which is the
    one comparison the layout invites a reviewer to make.
    """
    from eeg_pipeline.preprocessing.band_ica_report import dossier_color_limits

    grand_average = np.full((1, 100), 2.0)
    group_a = np.full((1, 100), 8.0)
    group_b = np.full((1, 100), -8.0)

    limits = dossier_color_limits(grand_average, [(group_a, group_b)])

    assert limits.power == pytest.approx(8.0)
    assert limits.differences == (pytest.approx(16.0),)


def test_the_shared_power_scale_survives_a_slide_with_no_comparison() -> None:
    """A dossier can carry a grand average alone, and still needs a limit for it."""
    from eeg_pipeline.preprocessing.band_ica_report import dossier_color_limits

    limits = dossier_color_limits(np.full((1, 100), 4.0), [])

    assert limits.power == pytest.approx(4.0)
    assert limits.differences == ()


def test_a_slide_with_no_time_frequency_data_has_no_limits() -> None:
    from eeg_pipeline.preprocessing.band_ica_report import dossier_color_limits

    limits = dossier_color_limits(None, [])

    assert limits.power is None
    assert limits.differences == ()


def test_color_limits_are_not_dominated_by_a_single_extreme_component() -> None:
    """A single outlying component must not flatten every other component's TFR."""
    from eeg_pipeline.preprocessing.band_ica_report import _comparison_color_limits

    group_a = np.concatenate([np.full(999, 2.0), np.full(1, 500.0)]).reshape(1, -1)
    group_b = np.zeros((1, 1000))

    condition_limit, _ = _comparison_color_limits(group_a, group_b)

    assert condition_limit == pytest.approx(2.0)


def test_standard_component_dossier_keeps_all_evidence_on_one_slide() -> None:
    from eeg_pipeline.preprocessing.band_ica_report import (
        BandReviewData,
        ComponentLabel,
        ConditionTfrResult,
        SourceDiagnostics,
        _build_standard_component_dossiers,
    )

    settings = BandIcaReportSettings.from_mapping(
        {
            "comparisons": [
                {
                    "name": "pain",
                    "column": "pain_binary_coded",
                    "group_a": {"label": "Painful", "values": [1]},
                    "group_b": {"label": "Non-painful", "values": [0]},
                }
            ]
        }
    )
    comparison = settings.comparisons[0]
    diagnostics = SourceDiagnostics(
        frequencies=np.array([8.0, 10.0, 13.0]),
        power_db=np.ones((2, 3)),
        tfr_frequencies=np.array([8.0, 10.0, 13.0]),
        tfr_times=np.array([-1.0, 0.0, 1.0]),
        tfr=np.ones((2, 3, 3)),
    )
    result = ConditionTfrResult(
        comparison=comparison,
        group_a_tfr=np.full((2, 3, 3), 2.0),
        group_b_tfr=np.full((2, 3, 3), -4.0),
        group_a_count=12,
        group_b_count=10,
    )
    review = BandReviewData(
        band=BAND_ICA_DEFINITIONS[1],
        diagnostics=diagnostics,
        comparisons=(result,),
    )
    ica = SimpleNamespace(n_components_=2, exclude=[1], plot_components=Mock())
    labels = [
        ComponentLabel("brain", 0.91),
        ComponentLabel("eye blink", 0.94),
    ]

    figures = _build_standard_component_dossiers(
        ica=ica,
        review=review,
        labels=labels,
        settings=settings,
        analysis_status="Provisional — all task epochs",
    )

    assert len(figures) == 2
    assert ica.plot_components.call_count == 2
    first_titles = [axis.get_title() for axis in figures[0].axes]
    assert first_titles[:3] == [
        "ICA000 topomap",
        "Band-limited source spectrum",
        "Grand average",
    ]
    assert any("Painful" in title and "n=12" in title for title in first_titles)
    assert any("Non-painful" in title and "n=10" in title for title in first_titles)
    assert any("Painful − Non-painful" in title for title in first_titles)
    assert any("pain_binary_coded ∈ [1]" in title for title in first_titles)
    assert any("pain_binary_coded ∈ [0]" in title for title in first_titles)
    assert any("pain" in title for title in first_titles)
    assert "RETAINED" in figures[0]._suptitle.get_text()
    assert "MARKED BAD" in figures[1]._suptitle.get_text()
    assert "Provisional — all task epochs" in figures[0]._suptitle.get_text()

    # The grand average and every condition share one power scale by construction, so
    # the slide states it once. Drawn per row it was one identical colourbar per
    # comparison plus the grand average's, on every dossier of every band.
    labels_drawn = [axis.get_ylabel() for axis in figures[0].axes]
    power_bars = [text for text in labels_drawn if text.startswith("Baseline-relative power")]
    difference_bars = [text for text in labels_drawn if text.startswith("Group difference")]
    assert len(power_bars) == 1
    assert len(difference_bars) == len(review.comparisons)


def test_component_review_content_is_grouped_before_mne_ica_components() -> None:
    import mne

    from eeg_pipeline.preprocessing.band_ica_report import _organize_component_review

    report = mne.Report(title="Review order")
    report.add_html("fit", title="Fit information", section="ICA: epochs for fitting")
    report.add_html("components", title="Topographies", section="ICA: components")
    report.add_html(
        "review",
        title="Component dossiers",
        section="ICA review: Alpha",
        tags=("ica", "ica-component-review"),
    )
    report.add_html(
        "appendix",
        title="Exploratory components",
        section="Band-specific ICA: Alpha",
        tags=("ica", "band-specific-ica"),
    )

    _organize_component_review(report)

    sections = [element.section for element in report._content]
    assert sections == [
        "ICA: epochs for fitting",
        "ICA review: Alpha",
        "ICA: components",
        "Band-specific ICA: Alpha",
    ]


def test_condition_tfr_results_use_configured_trials_from_standard_sources() -> None:
    from eeg_pipeline.preprocessing.band_ica_report import _condition_tfr_results

    settings = BandIcaReportSettings.from_mapping(
        {
            "comparisons": [
                {
                    "name": "pain",
                    "column": "pain_binary_coded",
                    "group_a": {"label": "Painful", "values": [1]},
                    "group_b": {"label": "Non-painful", "values": [0]},
                }
            ]
        }
    )
    source_data = np.arange(4 * 2 * 20, dtype=float).reshape(4, 2, 20)
    metadata = pd.DataFrame({"pain_binary_coded": [0, 1, 1, 0]})
    tfr = np.ones((2, 3, 4))

    with patch(
        "eeg_pipeline.preprocessing.band_ica_report._fieldtrip_tfr",
        side_effect=[
            (np.array([8.0, 10.0, 12.0]), np.arange(4), tfr),
            (np.array([8.0, 10.0, 12.0]), np.arange(4), -tfr),
        ],
    ) as fieldtrip_tfr:
        results = _condition_tfr_results(
            source_data=source_data,
            metadata=metadata,
            sfreq=100.0,
            times=np.linspace(-1.0, 1.0, 20),
            band=BAND_ICA_DEFINITIONS[1],
            settings=settings,
        )

    assert len(results) == 1
    assert results[0].group_a_count == 2
    assert results[0].group_b_count == 2
    np.testing.assert_array_equal(
        fieldtrip_tfr.call_args_list[0].kwargs["data"], source_data[[1, 2]]
    )
    np.testing.assert_array_equal(
        fieldtrip_tfr.call_args_list[1].kwargs["data"], source_data[[0, 3]]
    )


def test_add_standard_review_creates_one_authoritative_carousel_per_band() -> None:
    from eeg_pipeline.preprocessing.band_ica_report import (
        BandReviewData,
        ComponentLabel,
        SourceDiagnostics,
        _add_standard_component_review,
    )

    report = Mock()
    report._content = []
    ica = SimpleNamespace(n_components_=2, exclude=[1])
    labels = [ComponentLabel("brain", 0.9), ComponentLabel("eye blink", 0.8)]
    diagnostics = SourceDiagnostics(
        frequencies=np.array([1.0]),
        power_db=np.ones((2, 1)),
        tfr_frequencies=np.array([1.0]),
        tfr_times=np.array([0.0]),
        tfr=np.ones((2, 1, 1)),
    )
    reviews = [BandReviewData(band=band, diagnostics=diagnostics) for band in BAND_ICA_DEFINITIONS]
    figures = [[Mock(), Mock()] for _ in BAND_ICA_DEFINITIONS]

    with (
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._build_band_review_data",
            side_effect=reviews,
        ) as build_review,
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._build_standard_component_dossiers",
            side_effect=figures,
        ),
        patch("eeg_pipeline.preprocessing.band_ica_report._organize_component_review"),
        patch("eeg_pipeline.preprocessing.band_ica_report._remove_legacy_condition_tfr_entries"),
        patch("eeg_pipeline.preprocessing.band_ica_report.remove_tagged_content") as clear,
        patch("eeg_pipeline.preprocessing.band_ica_report._add_decomposition_summary"),
        patch("eeg_pipeline.preprocessing.band_ica_report._add_component_properties"),
    ):
        _add_standard_component_review(
            report=report,
            ica=ica,
            epochs=SimpleNamespace(),
            metadata=None,
            labels=labels,
            settings=BandIcaReportSettings(),
            analysis_status="Pending provisional task epochs",
        )

    clear.assert_called_once_with(report, tag="ica-component-review")
    assert build_review.call_count == len(BAND_ICA_DEFINITIONS)
    assert all(call.kwargs["ica"] is ica for call in build_review.call_args_list)
    assert report.add_figure.call_count == len(BAND_ICA_DEFINITIONS)
    assert [call.kwargs["section"] for call in report.add_figure.call_args_list] == [
        f"ICA component review: {band.title}" for band in BAND_ICA_DEFINITIONS
    ]
    for call in report.add_figure.call_args_list:
        assert call.kwargs["title"].startswith("Component dossiers")
        assert "use the slider" in call.kwargs["title"]
        assert call.kwargs["replace"] is True
        assert call.kwargs["caption"] == [
            "ICA000 · brain (0.900) · RETAINED",
            "ICA001 · eye blink (0.800) · MARKED BAD",
        ]


def test_legacy_separate_condition_sections_are_removed() -> None:
    import mne

    from eeg_pipeline.preprocessing.band_ica_report import (
        _remove_legacy_condition_tfr_entries,
    )

    report = mne.Report(title="Legacy cleanup")
    report.add_html(
        "legacy",
        title="pain comparison",
        section="Band-specific ICA comparison: Alpha",
        tags=("ica", "band-specific-ica", "condition-tfr"),
    )
    report.add_html(
        "legacy config",
        title="Condition comparison configuration",
        section="TFR comparisons",
        tags=("ica", "band-specific-ica", "condition-tfr-configuration"),
    )
    report.add_html(
        "keep",
        title="Component dossiers",
        section="ICA component review: Alpha",
        tags=("ica", "ica-component-review", "condition-tfr"),
    )

    _remove_legacy_condition_tfr_entries(report)

    titles = [element.name for element in report._content]
    assert titles == ["Component dossiers"]


def test_append_condition_tfr_report_updates_standard_ica_dossiers_once(tmp_path) -> None:
    from eeg_pipeline.preprocessing.band_ica_report import (
        ComponentLabel,
        append_condition_tfr_report,
    )

    ica_fit_epochs_path = tmp_path / "sub-0001_proc-icafit_epo.fif"
    pre_ica_epochs_path = tmp_path / "sub-0001_task-pain_epo.fif"
    clean_epochs_path = tmp_path / "sub-0001_task-pain_proc-clean_epo.fif"
    clean_events_path = tmp_path / "sub-0001_task-pain_proc-clean_events.tsv"
    standard_ica_path = tmp_path / "sub-0001_proc-ica_ica.fif"
    report_path = tmp_path / "sub-0001_report.h5"
    for path in (
        ica_fit_epochs_path,
        pre_ica_epochs_path,
        clean_epochs_path,
        clean_events_path,
        standard_ica_path,
        report_path,
    ):
        path.write_text("test", encoding="utf-8")

    ica_fit_epochs = MagicMock()
    pre_ica_epochs = MagicMock()
    clean_epochs = MagicMock()
    retained_epochs = MagicMock()
    clean_epochs.__len__.return_value = 2
    clean_epochs.selection = np.array([0, 2])
    pre_ica_epochs.__len__.return_value = 3
    pre_ica_epochs.selection = np.array([0, 1, 2])
    retained_epochs.__len__.return_value = 2
    pre_ica_epochs.__getitem__.return_value = retained_epochs
    standard_ica = SimpleNamespace(n_components_=1, exclude=[])
    labels = [ComponentLabel("brain", 0.9)]
    metadata = pd.DataFrame({"epoch_index": [0, 1], "pain_binary_coded": [0, 1]})
    report = Mock()
    report._content = []
    settings = BandIcaReportSettings.from_mapping(
        {
            "comparisons": [
                {
                    "name": "pain",
                    "column": "pain_binary_coded",
                    "group_a": {"label": "Painful", "values": [1]},
                    "group_b": {"label": "Non-painful", "values": [0]},
                }
            ]
        }
    )

    with (
        patch(
            "eeg_pipeline.preprocessing.band_ica_report.mne.read_epochs",
            side_effect=[ica_fit_epochs, pre_ica_epochs, clean_epochs],
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report.read_ica_with_reviewed_exclusions",
            return_value=standard_ica,
        ) as read_ica,
        patch(
            "eeg_pipeline.preprocessing.band_ica_report.pd.read_csv",
            return_value=metadata,
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._label_components",
            return_value=labels,
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report.open_subject_report",
            return_value=report,
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._add_standard_component_review"
        ) as add_standard_review,
    ):
        append_condition_tfr_report(
            ica_fit_epochs_path=ica_fit_epochs_path,
            standard_ica_path=standard_ica_path,
            pre_ica_epochs_path=pre_ica_epochs_path,
            clean_epochs_path=clean_epochs_path,
            clean_events_path=clean_events_path,
            report_path=report_path,
            settings=settings,
            analysis_status="Finalized — retained epochs",
        )

    read_ica.assert_called_once_with(standard_ica_path)
    assert add_standard_review.call_count == 1
    assert add_standard_review.call_args.kwargs["ica"] is standard_ica
    assert add_standard_review.call_args.kwargs["epochs"] is retained_epochs
    assert add_standard_review.call_args.kwargs["metadata"] is metadata
    assert add_standard_review.call_args.kwargs["labels"] == labels
    assert report.save.call_count == 2
    assert report.save.call_args_list[0].args == (report_path,)
    assert report.save.call_args_list[1].args == (report_path.with_suffix(".html"),)


def test_retained_epochs_are_mapped_by_original_mne_selection_values() -> None:
    import mne

    from eeg_pipeline.preprocessing.band_ica_report import _select_retained_epochs

    info = mne.create_info(["Cz"], sfreq=100.0, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((1, 500)), info, verbose="ERROR")
    events = np.array(
        [
            [50, 0, 1],
            [150, 0, 2],
            [250, 0, 1],
            [350, 0, 2],
        ]
    )
    pre_ica_epochs = mne.Epochs(
        raw,
        events,
        event_id={"target": 1},
        tmin=0.0,
        tmax=0.1,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )
    clean_epochs = pre_ica_epochs.copy().drop([0], verbose="ERROR")

    retained_epochs = _select_retained_epochs(pre_ica_epochs, clean_epochs)

    assert pre_ica_epochs.selection.tolist() == [0, 2]
    assert clean_epochs.selection.tolist() == [2]
    assert retained_epochs.selection.tolist() == [2]


def test_generate_band_report_persists_real_mne_html_sections(tmp_path) -> None:
    import matplotlib.pyplot as plt
    import mne

    report_path = tmp_path / "sub-0001_report.h5"
    mne.Report(title="Band ICA test").save(
        report_path,
        overwrite=True,
        open_browser=False,
    )
    epochs = SimpleNamespace(info={"sfreq": 250.0})
    (tmp_path / "sub-0001_proc-ica_ica.fif").write_text("ica", encoding="utf-8")

    class _Ica:
        n_components_ = 1
        exclude = []

        def save(self, path, overwrite):
            assert overwrite
            path.write_text("ica", encoding="utf-8")

    def component_figure(**_kwargs):
        figure, axis = plt.subplots()
        axis.plot([0.0, 1.0])
        plt.close(figure)
        return [figure]

    def add_standard_review(**kwargs):
        kwargs["report"].add_html(
            "authoritative",
            title="How to review ICA component dossiers",
            section="ICA component review guide",
            tags=("ica", "ica-component-review"),
        )

    with (
        patch(
            "eeg_pipeline.preprocessing.band_ica_report.mne.read_epochs",
            return_value=epochs,
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._band_epochs",
            return_value=epochs,
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._fit_band_ica",
            return_value=_Ica(),
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report.read_ica_with_reviewed_exclusions",
            return_value=_Ica(),
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._label_components",
            return_value=[SimpleNamespace(label="brain", probability=0.9)],
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._build_component_figures",
            side_effect=component_figure,
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._add_standard_component_review",
            side_effect=add_standard_review,
        ),
    ):
        generate_band_ica_report(
            epochs_path=tmp_path / "sub-0001_proc-icafit_epo.fif",
            report_path=report_path,
            output_dir=tmp_path / "band-specific-ica",
            output_prefix="sub-0001",
            random_state=42,
            settings=BandIcaReportSettings(),
        )

    html = report_path.with_suffix(".html").read_text(encoding="utf-8")
    assert "ICA component review guide" in html
    # The subject report keeps the section and the pointer; the figures themselves live in
    # the linked file, so the heavy sliders are not in this document.
    assert html.count(EXPLORATORY_BAND_SECTION) >= 1
    assert "sub-0001_desc-exploratorybandica_report.html" in html
    for band in BAND_ICA_DEFINITIONS:
        assert f"Band-specific ICA: {band.title}" not in html

    # Every band still has its own titled block, one section along, in the linked file.
    exploratory = (
        tmp_path / "band-specific-ica" / "sub-0001_desc-exploratorybandica_report.html"
    ).read_text(encoding="utf-8")
    for band in BAND_ICA_DEFINITIONS:
        assert band.title in exploratory


def test_mne_panels_our_own_sections_supersede_are_dropped() -> None:
    """MNE-BIDS-pipeline renders 22 ``plot_properties`` figures and a topography grid that
    our decomposition section now renders itself, so the report carried both — 2.3 MB of
    the same pictures twice, and two places to look for one answer.

    Every ``ICALabel:`` panel goes, the numeric ``ICALabel: report`` table included: the
    exclusion ledger carries each component's decision and deciding detector, and every
    dossier draws its full class distribution. MNE's overlay figures and its score panel
    have no counterpart here and are left alone.
    """
    import mne

    from eeg_pipeline.preprocessing.band_ica_report import drop_superseded_mne_ica_panels

    report = mne.Report(title="ica", verbose="ERROR")
    figure = plt_figure()
    for title in (
        "ICA component properties",
        "ICA component topographies",
        "ICALabel: eye blink components",
        "ICALabel: heart beat components",
        "Original and cleaned signal",
        "Scores for matching ECG patterns",
        "ICALabel: report",
    ):
        report.add_figure(fig=figure, title=title, section="ICA: components", tags=("ica",))

    drop_superseded_mne_ica_panels(report)

    remaining = {element.name for element in report._content}
    assert remaining == {
        "Original and cleaned signal",
        "Scores for matching ECG patterns",
    }


def test_dropping_superseded_panels_is_safe_when_they_were_never_added() -> None:
    """A resting-state or EEG-only run may never have produced them."""
    import mne

    from eeg_pipeline.preprocessing.band_ica_report import drop_superseded_mne_ica_panels

    report = mne.Report(title="ica", verbose="ERROR")

    drop_superseded_mne_ica_panels(report)

    assert report._content == []


def test_exploratory_bands_share_one_section_so_the_toc_has_one_entry() -> None:
    """Ten TOC entries differing only by an ``ICA component review:``/``Band-specific``
    prefix could not be told apart without opening one, and the exploratory half is the
    one that does not control artifact removal."""
    from eeg_pipeline.preprocessing.band_ica_report import (
        EXPLORATORY_BAND_SECTION,
        _exploratory_component_captions,
    )

    assert EXPLORATORY_BAND_SECTION != "Band-specific ICA"
    labels = [
        SimpleNamespace(label="brain", probability=0.91),
        SimpleNamespace(label="unlabeled", probability=0.0),
    ]

    captions = _exploratory_component_captions(labels)

    assert captions[0] == "ICA000 · brain (0.91)"
    # An unlabelled component carries no classification, so the caption states none.
    assert captions[1] == "ICA001"
    # Nothing is excluded by an exploratory fit, so no caption may imply a decision.
    assert not any("MARKED BAD" in caption or "RETAINED" in caption for caption in captions)


def test_exploratory_slides_are_captioned_by_component_not_by_position(tmp_path) -> None:
    """``add_figure`` without ``caption`` labels the slides "Figure 1..62", so the slider
    is unnavigable and its caption disagrees with the component named on the figure."""
    import matplotlib.pyplot as plt
    import mne

    from eeg_pipeline.preprocessing.band_ica_report import ComponentLabel

    report_path = tmp_path / "sub-0001_report.h5"
    mne.Report(title="Band ICA test").save(report_path, overwrite=True, open_browser=False)
    epochs = SimpleNamespace(info={"sfreq": 250.0})
    (tmp_path / "sub-0001_proc-ica_ica.fif").write_text("ica", encoding="utf-8")

    class _Ica:
        n_components_ = 2
        exclude = []

        def save(self, path, overwrite):
            path.write_text("ica", encoding="utf-8")

    def component_figures(**_kwargs):
        built = []
        for _ in range(2):
            figure, axis = plt.subplots()
            axis.plot([0.0, 1.0])
            plt.close(figure)
            built.append(figure)
        return built

    captured: list[dict] = []
    with (
        patch("eeg_pipeline.preprocessing.band_ica_report.mne.read_epochs", return_value=epochs),
        patch("eeg_pipeline.preprocessing.band_ica_report._band_epochs", return_value=epochs),
        patch("eeg_pipeline.preprocessing.band_ica_report._fit_band_ica", return_value=_Ica()),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report.read_ica_with_reviewed_exclusions",
            return_value=_Ica(),
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._label_components",
            return_value=[
                ComponentLabel("brain", 0.9),
                ComponentLabel("unlabeled", 0.0),
            ],
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._build_component_figures",
            side_effect=component_figures,
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._add_standard_component_review",
        ),
        patch.object(
            mne.Report,
            "add_figure",
            autospec=True,
            side_effect=lambda self, **kwargs: captured.append(kwargs),
        ),
    ):
        generate_band_ica_report(
            epochs_path=tmp_path / "sub-0001_proc-icafit_epo.fif",
            report_path=report_path,
            output_dir=tmp_path / "band-specific-ica",
            output_prefix="sub-0001",
            random_state=42,
            # ICLabel is off for band fits by default, which is the configuration that
            # produced "Figure 1..62": there was no label to fall back to as a caption.
            settings=BandIcaReportSettings.from_mapping({"run_iclabel": True}),
        )

    exploratory = [call for call in captured if call.get("section") == EXPLORATORY_BAND_SECTION]
    assert len(exploratory) == len(BAND_ICA_DEFINITIONS)
    for call in exploratory:
        assert call["caption"] == ["ICA000 · brain (0.90)", "ICA001"]


def test_resting_state_settings_disable_event_locked_time_frequency() -> None:
    """Fixed-length rest epochs have no baseline, so the TFR must be switchable off."""
    settings = BandIcaReportSettings.from_mapping({"tfr": {"enabled": False}})

    assert settings.tfr_enabled is False


def test_condition_comparisons_are_rejected_without_event_locked_time_frequency() -> None:
    with pytest.raises(ValueError, match="Resting-state recordings have no events"):
        BandIcaReportSettings.from_mapping(
            {
                "tfr": {"enabled": False},
                "comparisons": [
                    {
                        "name": "high_vs_low",
                        "column": "stimulus_temp",
                        "group_a": {"label": "High", "values": [48.3]},
                        "group_b": {"label": "Low", "values": [44.3]},
                    }
                ],
            }
        )


def test_disabled_time_frequency_skips_the_multitaper_computation() -> None:
    """Rest dossiers must not compute a baseline-relative TFR at all."""
    import mne

    from eeg_pipeline.preprocessing.band_ica_report import _source_diagnostics_from_sources

    info = mne.create_info(["C1", "C2"], 200.0, "eeg")
    sources = mne.EpochsArray(
        np.random.default_rng(0).normal(0, 1e-5, (4, 2, 2000)),
        info,
        tmin=0.0,
        verbose="ERROR",
    )

    with patch("eeg_pipeline.preprocessing.band_ica_report._fieldtrip_tfr") as tfr:
        diagnostics = _source_diagnostics_from_sources(
            sources=sources,
            band=BAND_ICA_DEFINITIONS[1],
            settings=BandIcaReportSettings.from_mapping({"tfr": {"enabled": False}}),
        )

    tfr.assert_not_called()
    assert diagnostics.has_tfr is False
    assert diagnostics.tfr is None


def test_disabled_time_frequency_renders_topography_and_spectrum_only() -> None:
    from eeg_pipeline.preprocessing.band_ica_report import (
        ComponentLabel,
        _build_component_figures,
    )

    ica = SimpleNamespace(n_components_=2, plot_components=Mock())
    diagnostics = (np.array([8.0, 10.0, 13.0]), np.ones((2, 3)), None, None, None)

    with patch(
        "eeg_pipeline.preprocessing.band_ica_report._source_diagnostics",
        return_value=diagnostics,
    ):
        figures = _build_component_figures(
            ica=ica,
            epochs=SimpleNamespace(),
            band=BAND_ICA_DEFINITIONS[1],
            labels=[ComponentLabel("brain", 0.91), ComponentLabel("other", 0.5)],
            settings=BandIcaReportSettings.from_mapping({"tfr": {"enabled": False}}),
        )

    assert [axis.get_title() for axis in figures[0].axes] == [
        "ICA000 topomap",
        "Source spectrum",
    ]


def test_iclabel_keeps_the_full_class_distribution_not_only_the_winner() -> None:
    from eeg_pipeline.preprocessing.band_ica_report import _ICLABEL_CLASSES, _label_components

    probabilities = np.array([[0.45, 0.42, 0.05, 0.03, 0.02, 0.02, 0.01]])
    ica = SimpleNamespace(n_components_=1)

    with patch(
        "mne_icalabel.iclabel.iclabel_label_components",
        return_value=probabilities,
        create=True,
    ):
        labels = _label_components(epochs=SimpleNamespace(), ica=ica)

    assert labels[0].label == "brain"
    assert labels[0].probability == pytest.approx(0.45)
    assert labels[0].has_distribution
    assert len(labels[0].probabilities) == len(_ICLABEL_CLASSES)
    assert labels[0].probabilities[1] == pytest.approx(0.42)


def test_component_table_records_every_iclabel_class_probability() -> None:
    import csv as csv_module
    import tempfile
    from pathlib import Path

    from eeg_pipeline.preprocessing.band_ica_report import (
        ComponentLabel,
        _ICLABEL_CLASSES,
        _write_component_table,
    )

    path = Path(tempfile.mkdtemp()) / "components.tsv"
    _write_component_table(
        path=path,
        labels=[ComponentLabel("brain", 0.45, (0.45, 0.42, 0.05, 0.03, 0.02, 0.02, 0.01))],
    )

    with path.open(encoding="utf-8") as file:
        rows = list(csv_module.DictReader(file, delimiter="\t"))

    assert rows[0]["iclabel"] == "brain"
    assert float(rows[0]["probability_muscle_artifact"]) == pytest.approx(0.42)
    for name in _ICLABEL_CLASSES:
        assert f"probability_{name.replace(' ', '_')}" in rows[0]


def test_unlabelled_components_render_without_an_iclabel_distribution() -> None:
    """Band-specific ICAs may skip ICLabel; the dossier must still render."""
    import matplotlib.pyplot as plt

    from eeg_pipeline.preprocessing.band_ica_report import ComponentLabel, _add_iclabel_panel

    figure, axis = plt.subplots()
    _add_iclabel_panel(axis, ComponentLabel("unlabeled", 0.0))
    assert axis.child_axes == []

    _add_iclabel_panel(
        axis, ComponentLabel("brain", 0.9, (0.9, 0.04, 0.02, 0.02, 0.01, 0.005, 0.005))
    )
    assert len(axis.child_axes) == 1
    assert len(axis.child_axes[0].patches) == 7
    plt.close(figure)


def test_dossier_format_follows_figure_content() -> None:
    """Dense time-frequency meshes stay raster; spectrum-only dossiers go vector."""
    from eeg_pipeline.preprocessing.report.style import report_image_format

    assert report_image_format(has_dense_image=True) != "svg"
    assert report_image_format(has_dense_image=False) == "svg"


def test_baseline_must_clear_the_dpss_half_window() -> None:
    """A baseline ending near the event draws post-stimulus data into the baseline."""
    with pytest.raises(ValueError, match="baseline_tmax_s must be at most"):
        BandIcaReportSettings.from_mapping({"tfr": {"baseline_tmax_s": -0.01}})

    settings = BandIcaReportSettings.from_mapping({"tfr": {"baseline_tmax_s": -1.5}})
    assert settings.baseline_tmax_s == -1.5


def test_baseline_margin_is_not_required_without_the_time_frequency_transform() -> None:
    settings = BandIcaReportSettings.from_mapping(
        {"tfr": {"enabled": False, "baseline_tmax_s": -0.01}}
    )

    assert settings.tfr_enabled is False


def test_display_frequencies_never_exceed_the_smoothing_resolution() -> None:
    """Sampling far finer than the smoothing buys no resolution, only compute."""
    from eeg_pipeline.preprocessing.band_ica_report import (
        DISPLAY_SAMPLES_PER_SMOOTHING_HALF_WIDTH,
        _tfr_parameter_groups,
    )

    settings = BandIcaReportSettings()
    for band in BAND_ICA_DEFINITIONS:
        for parameters, frequencies in _tfr_parameter_groups(band, settings):
            assert frequencies.size > 1
            step = float(frequencies[1] - frequencies[0])
            expected = max(
                settings.frequency_step_hz,
                parameters.smoothing_hz / DISPLAY_SAMPLES_PER_SMOOTHING_HALF_WIDTH,
            )
            assert step <= expected + 1e-6


def test_every_band_is_covered_exactly_once_end_to_end() -> None:
    """Gaps let the next parameter set smooth a lone bin by its much wider kernel."""
    from eeg_pipeline.preprocessing.band_ica_report import _tfr_parameter_groups

    for band in BAND_ICA_DEFINITIONS:
        groups = _tfr_parameter_groups(band, BandIcaReportSettings())
        combined = np.concatenate([frequencies for _, frequencies in groups])
        assert np.all(np.diff(combined) > 0)
        assert combined[0] == pytest.approx(band.fmin)
        assert combined[-1] == pytest.approx(band.fmax)


def test_a_band_ending_on_a_parameter_boundary_keeps_the_narrower_smoothing() -> None:
    """Broadband ends at 30 Hz; that bin must not be smoothed by the gamma kernel."""
    from eeg_pipeline.preprocessing.band_ica_report import _tfr_parameter_groups

    band = next(b for b in BAND_ICA_DEFINITIONS if b.slug == "broadband1to30")
    groups = _tfr_parameter_groups(band, BandIcaReportSettings())

    smoothings = [parameters.smoothing_hz for parameters, _ in groups]
    assert 5.0 not in smoothings
    assert groups[-1][1][-1] == pytest.approx(30.0)


def _exploratory_figures(tfr: np.ndarray):
    """Build the band-fitted component figures over a supplied TFR array."""
    from eeg_pipeline.preprocessing.band_ica_report import (
        ComponentLabel,
        _build_component_figures,
    )

    diagnostics = (
        np.array([8.0, 10.0, 13.0]),
        np.ones((tfr.shape[0], 3)),
        np.array([8.0, 10.0, 13.0]),
        np.array([-1.0, 0.0, 1.0]),
        tfr,
    )
    with patch(
        "eeg_pipeline.preprocessing.band_ica_report._source_diagnostics",
        return_value=diagnostics,
    ):
        return _build_component_figures(
            ica=SimpleNamespace(n_components_=tfr.shape[0], plot_components=Mock()),
            epochs=SimpleNamespace(),
            band=BAND_ICA_DEFINITIONS[1],
            labels=[ComponentLabel("brain", 0.9) for _ in range(tfr.shape[0])],
            settings=BandIcaReportSettings(),
        )


def test_each_tfr_panel_states_its_own_peak_against_the_shared_scale() -> None:
    """A shared colour limit renders a modest component almost entirely neutral.

    The scale is shared on purpose, so components can be compared. Without each panel
    declaring its own peak, a component whose modulation is 1 dB on a scale reaching
    6 dB is a blank rectangle that reads as a broken figure rather than a small effect.
    """
    tfr = np.zeros((2, 3, 3))
    tfr[0] = 6.0
    tfr[1] = 1.0

    figures = _exploratory_figures(tfr)

    annotations = [
        text.get_text() for text in figures[1].axes[2].texts if "peak" in text.get_text()
    ]
    assert annotations == ["peak |1.0| dB"]


def test_exploratory_band_figures_say_so_on_the_figure() -> None:
    """These figures and the authoritative dossiers are otherwise indistinguishable.

    Both carry a topography, a spectrum and a TFR in the same arrangement, and their
    component numbers refer to different decompositions. Only the accordion heading
    told them apart, which is one collapsed section away from a reviewer acting on a
    band-fitted index as though it were an authoritative one.
    """
    figures = _exploratory_figures(np.ones((1, 3, 3)))

    stamps = [text.get_text() for text in figures[0].texts]
    assert any(text.startswith("EXPLORATORY") for text in stamps)
    assert any("do not match" in text for text in stamps)


def test_the_dossier_does_not_repeat_the_transform_configuration_per_slide() -> None:
    """Sixty slides carrying the same taper and baseline spent a title line on nothing.

    The configuration is stated once in the review context above the carousel.
    """
    figures = _exploratory_figures(np.ones((1, 3, 3)))

    title = figures[0]._suptitle.get_text()
    assert "DPSS" not in title
    assert "ICA000" in title


def _decomposition_review_report():
    """A report carrying decomposition evidence from a first, fully-equipped pass."""
    import mne

    report = mne.Report(title="ica", verbose="ERROR")
    for title in ("Decomposition summary", "Component variance by run"):
        report.add_html(
            html="<p/>",
            title=title,
            section="ICA decomposition quality",
            tags=("ica", "ica-decomposition"),
        )
    report.add_html(
        html="<p/>",
        title="Cardiac marker fallback",
        section="ICA decomposition quality",
        tags=("ica", "ica-decomposition", "ica-fallback-warning"),
    )
    report.add_html(
        html="<p/>",
        title="How to review ICA component dossiers",
        section="ICA component review guide",
        tags=("ica", "ica-component-review", "ica-review-guide"),
    )
    report.add_html(html="<p/>", title="Info", section="ICA: components", tags=("ica",))
    return report


def test_a_condition_rebuild_keeps_evidence_it_cannot_recreate() -> None:
    """The second pass has no run files, so it must not rebuild the run-level panels.

    ``append_condition_tfr_report`` reopens the report to replace the dossiers with
    condition-aware ones. It has no ``filtered_raw_paths`` and no fallback state, so a
    rebuild of the decomposition section silently dropped "Component variance by run"
    and the Analyzer-marker fallback warning — both present after the first pass and
    absent from the delivered HTML.
    """
    from eeg_pipeline.preprocessing.band_ica_report import _should_rebuild_decomposition

    report = _decomposition_review_report()

    assert not _should_rebuild_decomposition(report, filtered_raw_paths=None)


def test_a_first_pass_builds_the_decomposition_evidence() -> None:
    """Skipping the rebuild must not leave a fresh report with no decomposition at all."""
    import mne

    from eeg_pipeline.preprocessing.band_ica_report import _should_rebuild_decomposition

    empty = mne.Report(title="ica", verbose="ERROR")

    assert _should_rebuild_decomposition(empty, filtered_raw_paths=None)


def test_a_pass_carrying_run_files_rebuilds_even_when_evidence_exists() -> None:
    """Re-running the full stage must refresh the panels, not skip them as already there."""
    from pathlib import Path

    from eeg_pipeline.preprocessing.band_ica_report import _should_rebuild_decomposition

    report = _decomposition_review_report()

    assert _should_rebuild_decomposition(report, filtered_raw_paths=[Path("run-1.fif")])


def test_the_component_review_rebuild_does_not_clear_decomposition_evidence() -> None:
    """The two bodies of content are cleared by different tags, so one cannot take the other."""
    from eeg_pipeline.preprocessing.report.organize import remove_tagged_content

    report = _decomposition_review_report()

    remove_tagged_content(report, tag="ica-component-review")

    survivors = {element.name for element in report._content}
    assert "Component variance by run" in survivors
    assert "Cardiac marker fallback" in survivors
    assert "How to review ICA component dossiers" not in survivors


def test_decomposition_evidence_is_ordered_with_the_review() -> None:
    """Losing the review tag must not leave the panels stranded after MNE's ICA section."""
    from eeg_pipeline.preprocessing.band_ica_report import _organize_component_review

    report = _decomposition_review_report()

    _organize_component_review(report)

    names = [element.name for element in report._content]
    assert names.index("Decomposition summary") < names.index("Info")
    assert names.index("Component variance by run") < names.index("Info")


def _montaged_epochs_and_ica():
    """A small fitted decomposition with sensor positions, enough for the topographies."""
    import matplotlib
    import mne

    matplotlib.use("Agg")
    montage = mne.channels.make_standard_montage("standard_1020")
    names = montage.ch_names[:10]
    info = mne.create_info(names, 250.0, "eeg")
    rng = np.random.default_rng(0)
    epochs = mne.EpochsArray(rng.normal(0, 1e-5, (25, len(names), 250)), info, verbose="ERROR")
    epochs.set_montage(montage)
    ica = mne.preprocessing.ICA(n_components=4, random_state=0, max_iter=200)
    ica.fit(epochs, verbose="ERROR")
    ica.exclude = [0, 2]
    return epochs, ica


def test_the_decomposition_section_names_the_detector_behind_each_exclusion() -> None:
    """ "brain 0.93 / excluded: yes" read as a contradiction because the reason the
    pipeline had already recorded never reached the report."""
    import mne

    from eeg_pipeline.preprocessing.band_ica_report import (
        ComponentLabel,
        _add_decomposition_summary,
    )

    epochs, ica = _montaged_epochs_and_ica()
    report = mne.Report(title="ica", verbose="ERROR")

    _add_decomposition_summary(
        report=report,
        ica=ica,
        epochs=epochs,
        labels=[ComponentLabel(label="brain", probability=0.9) for _ in range(4)],
        filtered_raw_paths=None,
        status_descriptions=(
            "Auto-detected eye blink (MNE-ICALabel)",
            "",
            "Auto-detected ECG artifact (MNE)",
            "",
        ),
    )

    document = " ".join(element.html for element in report._content)
    assert "Auto-detected ECG artifact (MNE)" in document


def test_the_standard_review_forwards_the_exclusion_reasons_it_was_given() -> None:
    """The reasons are read once, beside the ICA, and the decomposition panel is the only
    thing that renders them; a review that drops them on the way loses them silently."""
    from eeg_pipeline.preprocessing.band_ica_report import (
        BandReviewData,
        ComponentLabel,
        SourceDiagnostics,
        _add_standard_component_review,
    )

    report = Mock()
    report._content = []
    diagnostics = SourceDiagnostics(
        frequencies=np.array([1.0]),
        power_db=np.ones((2, 1)),
        tfr_frequencies=np.array([1.0]),
        tfr_times=np.array([0.0]),
        tfr=np.ones((2, 1, 1)),
    )
    reasons = ("", "Auto-detected ECG artifact (MNE)")

    with (
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._build_band_review_data",
            side_effect=[
                BandReviewData(band=band, diagnostics=diagnostics) for band in BAND_ICA_DEFINITIONS
            ],
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._build_standard_component_dossiers",
            side_effect=[[Mock(), Mock()] for _ in BAND_ICA_DEFINITIONS],
        ),
        patch("eeg_pipeline.preprocessing.band_ica_report._organize_component_review"),
        patch("eeg_pipeline.preprocessing.band_ica_report._remove_legacy_condition_tfr_entries"),
        patch("eeg_pipeline.preprocessing.band_ica_report.remove_tagged_content"),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._add_decomposition_summary"
        ) as decomposition,
        patch("eeg_pipeline.preprocessing.band_ica_report._add_component_properties"),
    ):
        _add_standard_component_review(
            report=report,
            ica=SimpleNamespace(n_components_=2, exclude=[1]),
            epochs=SimpleNamespace(),
            metadata=None,
            labels=[ComponentLabel("brain", 0.9), ComponentLabel("brain", 0.93)],
            settings=BandIcaReportSettings(),
            analysis_status="Pending provisional task epochs",
            status_descriptions=reasons,
        )

    assert decomposition.call_args.kwargs["status_descriptions"] == reasons


def test_generate_band_report_reads_the_exclusion_reasons_from_the_component_table(
    tmp_path,
) -> None:
    """The reasons live in the same table that decides the exclusions, so reading them
    from anywhere else risks a report that disagrees with the cleaned data."""
    epochs = SimpleNamespace(info={"sfreq": 250.0})
    report = Mock()
    report._content = []
    report_path = tmp_path / "sub-0001_report.h5"
    report_path.write_text("report", encoding="utf-8")
    (tmp_path / "sub-0001_proc-ica_ica.fif").write_text("ica", encoding="utf-8")
    pd.DataFrame(
        {
            "component": [0, 1],
            "status": ["good", "bad"],
            "status_description": ["", "Auto-detected ECG artifact (MNE)"],
        }
    ).to_csv(tmp_path / "sub-0001_proc-ica_components.tsv", sep="\t", index=False)

    with (
        patch("eeg_pipeline.preprocessing.band_ica_report.mne.read_epochs", return_value=epochs),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report.open_subject_report",
            return_value=report,
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report.read_ica_with_reviewed_exclusions",
            return_value=SimpleNamespace(n_components_=2, exclude=[1]),
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._fit_band_ica",
            return_value=SimpleNamespace(n_components_=2, exclude=[], save=Mock()),
        ),
        patch("eeg_pipeline.preprocessing.band_ica_report._band_epochs", return_value=epochs),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._label_components",
            return_value=[
                SimpleNamespace(label="brain", probability=0.9),
                SimpleNamespace(label="brain", probability=0.93),
            ],
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._build_component_figures",
            return_value=[plt_figure(), plt_figure()],
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._add_standard_component_review"
        ) as review,
    ):
        generate_band_ica_report(
            epochs_path=tmp_path / "sub-0001_proc-icafit_epo.fif",
            report_path=report_path,
            output_dir=tmp_path / "band-ica",
            output_prefix="sub-0001",
            random_state=42,
            settings=BandIcaReportSettings(),
        )

    assert list(review.call_args.kwargs["status_descriptions"]) == [
        "",
        "Auto-detected ECG artifact (MNE)",
    ]
