from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from studies.pain_study.study1.config.loader import load_study1_config
from studies.pain_study.study1.figures.power_construct_validity import (
    PowerConstructValiditySummary,
)
from studies.pain_study.study1.figures.power_construct_models import (
    build_rating_association,
    build_temperature_association,
)
from studies.tests.pipelines.test_study1_power_construct_validity import (
    BANDS,
    TEMPERATURES,
    association_trials,
)


def test_build_power_construct_figure_aligns_temperature_rows_with_forest_plot() -> None:
    from studies.pain_study.study1.figures.power_construct_validity_plot import (
        build_power_construct_validity_figure,
    )

    figure = build_power_construct_validity_figure(power_summary(), load_study1_config())

    try:
        assert np.allclose(figure.get_size_inches(), (183.0 / 25.4, 145.0 / 25.4))
        assert len(figure.axes) == 6
        temperature_axes = figure.axes[:5]
        rating_axis = figure.axes[5]
        assert [axis.get_label() for axis in temperature_axes] == [
            "temperature:alpha",
            "temperature:beta",
            "temperature:gamma_low_clean",
            "temperature:gamma_mid_clean",
            "temperature:gamma_high_clean",
        ]
        assert rating_axis.get_label() == "adjusted-intensity"
        assert all(
            axis.get_position().x0 == pytest.approx(temperature_axes[0].get_position().x0)
            and axis.get_position().width == pytest.approx(temperature_axes[0].get_position().width)
            for axis in temperature_axes
        )
        assert [axis.get_xlabel() for axis in temperature_axes] == [
            "",
            "",
            "",
            "",
            "Temperature (°C)",
        ]
        assert all(
            text.get_rotation() == 0.0
            for axis in temperature_axes
            for text in axis.get_xticklabels()
        )
        assert [text.get_text() for text in temperature_axes[0].texts] == [
            "Alpha",
            "8–12.9 Hz",
        ]
        assert len(figure.legends) == 1
        assert figure.legends[0]._ncols == 2
    finally:
        plt.close(figure)


def test_power_construct_figure_states_sample_and_interval_estimands() -> None:
    from studies.pain_study.study1.figures.power_construct_validity_plot import (
        build_power_construct_validity_figure,
    )

    figure = build_power_construct_validity_figure(power_summary(), load_study1_config())

    try:
        figure_text = [text.get_text() for text in figure.texts]
        assert "EEG band-power construct validity" in figure_text
        assert (
            "Preliminary descriptive analysis · n=4 participants · retained trials/participant: "
            "median 36, range 36–36 · Fp1/Fp2 included"
        ) in figure_text
        assert "a  Temperature response" in figure_text
        assert (
            "Simultaneous 95% participant-bootstrap band across 30 band-temperature cells"
        ) in figure_text
        assert "b  Adjusted intensity association" in figure_text
        assert "Pointwise 95% participant-bootstrap CI across participants" in figure_text
        assert [text.get_text() for text in figure.legends[0].get_texts()] == [
            "Participant estimate",
            "Equal-weight cohort mean",
        ]
        assert figure.axes[5].get_xlabel() == ("Adjusted within-participant correlation, partial r")
        assert all(
            text.get_position()[0] > 1.0 and not text.get_clip_on() for text in figure.axes[5].texts
        )
    finally:
        plt.close(figure)


def test_power_construct_figure_uses_shared_scales_and_zero_references() -> None:
    from studies.pain_study.study1.figures.power_construct_validity_plot import (
        build_power_construct_validity_figure,
    )

    figure = build_power_construct_validity_figure(power_summary(), load_study1_config())

    try:
        x_limits = [axis.get_xlim() for axis in figure.axes[:5]]
        y_limits = [axis.get_ylim() for axis in figure.axes[:5]]
        assert all(np.allclose(limits, x_limits[0]) for limits in x_limits)
        assert all(np.allclose(limits, y_limits[0]) for limits in y_limits)
        assert figure.axes[5].get_xlim() == (-1.0, 1.0)
        for axis in figure.axes[:5]:
            assert any(
                np.allclose(line.get_ydata(), (0.0, 0.0))
                for line in axis.lines
                if len(line.get_ydata()) == 2
            )
        assert any(
            np.allclose(line.get_xdata(), (0.0, 0.0))
            for line in figure.axes[5].lines
            if len(line.get_xdata()) == 2
        )
    finally:
        plt.close(figure)


def test_power_construct_figure_rejects_mislabeled_temperature_intervals() -> None:
    from studies.pain_study.study1.figures.power_construct_validity_plot import (
        build_power_construct_validity_figure,
    )

    summary = power_summary()
    summary.temperature_summary["interval_type"] = "pointwise_percentile_bootstrap"

    with pytest.raises(ValueError, match="simultaneous max-studentized"):
        build_power_construct_validity_figure(summary, load_study1_config())


def test_power_construct_writer_creates_svg_and_seven_audit_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import studies.pain_study.study1.figures.plot_power_construct_validity as module

    summary = power_summary()
    monkeypatch.setattr(
        module,
        "load_power_construct_validity_summary",
        lambda **kwargs: summary,
    )
    outputs = [
        module.write_power_construct_validity(
            task="thermalactive",
            config=load_study1_config(),
            output_path=tmp_path / directory / "power_construct_validity.svg",
        )
        for directory in ("first", "second")
    ]

    first, second = outputs
    assert first.svg.read_bytes() == second.svg.read_bytes()
    root = ElementTree.parse(first.svg).getroot()
    assert float(root.attrib["width"].removesuffix("pt")) * 25.4 / 72.0 == pytest.approx(
        183.0, abs=0.01
    )
    assert float(root.attrib["height"].removesuffix("pt")) * 25.4 / 72.0 == pytest.approx(
        145.0, abs=0.01
    )
    assert first.svg.read_text(encoding="utf-8").count("<text") > 0
    assert {path.name for path in first.svg.parent.iterdir()} == {
        "power_construct_validity.svg",
        "power_construct_validity_trials.tsv",
        "power_construct_validity_trials.parquet",
        "power_temperature_by_subject.tsv",
        "power_temperature_by_subject.parquet",
        "power_temperature_summary.tsv",
        "power_temperature_summary.parquet",
        "power_rating_by_subject.tsv",
        "power_rating_by_subject.parquet",
        "power_rating_summary.tsv",
        "power_rating_summary.parquet",
        "power_fp1_fp2_sensitivity_by_subject.tsv",
        "power_fp1_fp2_sensitivity_by_subject.parquet",
        "power_fp1_fp2_sensitivity_summary.tsv",
        "power_fp1_fp2_sensitivity_summary.parquet",
    }
    assert pd.read_csv(first.trials_tsv, sep="\t").columns.tolist() == (
        summary.trials.columns.tolist()
    )
    assert pd.read_parquet(first.rating_summary_parquet).columns.tolist() == (
        summary.rating_summary.columns.tolist()
    )


def power_summary() -> PowerConstructValiditySummary:
    config = load_study1_config()
    trials = association_trials(n_subjects=4)
    temperature_by_subject, temperature_summary = build_temperature_association(
        trials,
        bands=BANDS,
        temperatures=TEMPERATURES,
        config=config,
    )
    rating_by_subject, rating_summary = build_rating_association(
        trials,
        bands=BANDS,
        config=config,
    )
    sensitivity_by_subject = pd.DataFrame(
        {
            "subject_id": ["sub-0000"],
            "band": ["alpha"],
            "partial_r_difference": [0.01],
        }
    )
    sensitivity_summary = pd.DataFrame({"band": ["alpha"], "mean_partial_r_difference": [0.01]})
    return PowerConstructValiditySummary(
        trials=trials,
        temperature_by_subject=temperature_by_subject,
        temperature_summary=temperature_summary,
        rating_by_subject=rating_by_subject,
        rating_summary=rating_summary,
        sensitivity_by_subject=sensitivity_by_subject,
        sensitivity_summary=sensitivity_summary,
        bands=BANDS,
        temperatures=TEMPERATURES,
        primary_include_fp1_fp2=True,
        n_subjects=4,
        article_ready=False,
    )
