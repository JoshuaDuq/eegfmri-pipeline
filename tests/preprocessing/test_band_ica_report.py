from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from eeg_pipeline.preprocessing.band_ica_report import (
    BAND_ICA_DEFINITIONS,
    BandIcaReportSettings,
    generate_band_ica_report,
)


def test_band_definitions_match_requested_report_sections() -> None:
    assert [(band.slug, band.fmin, band.fmax) for band in BAND_ICA_DEFINITIONS] == [
        ("deltatheta", 1.0, 8.0),
        ("alpha", 8.0, 13.0),
        ("beta", 13.0, 30.0),
        ("gamma", 30.0, 100.0),
        ("broadband1to30", 1.0, 30.0),
        ("broadband30to100", 30.0, 100.0),
    ]


def test_band_report_settings_fail_fast_on_invalid_values() -> None:
    with pytest.raises(ValueError, match="fit_decim"):
        BandIcaReportSettings.from_mapping({"fit_decim": 0})
    with pytest.raises(ValueError, match="tfr_frequency_count"):
        BandIcaReportSettings.from_mapping({"tfr_frequency_count": 1})


def test_generate_band_report_writes_each_band_as_exploratory_outputs(tmp_path) -> None:
    epochs = SimpleNamespace(info={"sfreq": 250.0})
    report = Mock()
    report_path = tmp_path / "sub-0001_report.h5"
    report_path.write_text("report", encoding="utf-8")
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
    figures = [Mock(), Mock()]

    with (
        patch(
            "eeg_pipeline.preprocessing.band_ica_report.mne.read_epochs",
            return_value=epochs,
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report.mne.open_report",
            return_value=report,
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
            "eeg_pipeline.preprocessing.band_ica_report._label_band_components",
            return_value=component_labels,
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._build_component_figures",
            return_value=figures,
        ),
    ):
        generated = generate_band_ica_report(
            epochs_path=tmp_path / "sub-0001_proc-icafit_epo.fif",
            report_path=report_path,
            output_dir=output_dir,
            output_prefix="sub-0001",
            random_state=42,
            settings=BandIcaReportSettings(),
        )

    assert len(generated) == 2 * len(BAND_ICA_DEFINITIONS)
    assert len(fitted_icas) == len(BAND_ICA_DEFINITIONS)
    assert all(ica.exclude == [] for ica in fitted_icas)
    assert report.add_figure.call_count == len(BAND_ICA_DEFINITIONS)
    sections = [call.kwargs["section"] for call in report.add_figure.call_args_list]
    assert sections == [f"Band-specific ICA: {band.title}" for band in BAND_ICA_DEFINITIONS]
    assert report.save.call_count == 2 * len(BAND_ICA_DEFINITIONS)
    for index in range(0, report.save.call_count, 2):
        assert report.save.call_args_list[index].args == (report_path,)
        assert report.save.call_args_list[index + 1].args == (report_path.with_suffix(".html"),)


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
        freqs=np.array([8.0, 10.0, 13.0]),
        get_data=Mock(return_value=np.ones((2, 2, 3))),
    )
    sources = SimpleNamespace(
        info={"sfreq": 100.0},
        times=np.linspace(-1.0, 1.0, 20),
        compute_psd=Mock(return_value=spectrum),
        get_data=Mock(return_value=np.ones((2, 2, 20))),
    )
    ica = SimpleNamespace(get_sources=Mock(return_value=sources))

    with patch(
        "eeg_pipeline.preprocessing.band_ica_report.mne.time_frequency.tfr_array_morlet",
        return_value=np.ones((2, 4, 10)),
    ):
        _source_diagnostics(
            ica=ica,
            epochs=SimpleNamespace(),
            band=BAND_ICA_DEFINITIONS[1],
            settings=BandIcaReportSettings(tfr_frequency_count=4, tfr_decim=2),
        )

    assert sources.compute_psd.call_args.kwargs["picks"] == "all"


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
            "eeg_pipeline.preprocessing.band_ica_report._label_band_components",
            return_value=[SimpleNamespace(label="brain", probability=0.9)],
        ),
        patch(
            "eeg_pipeline.preprocessing.band_ica_report._build_component_figures",
            side_effect=component_figure,
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
    for band in BAND_ICA_DEFINITIONS:
        assert f"Band-specific ICA: {band.title}" in html
