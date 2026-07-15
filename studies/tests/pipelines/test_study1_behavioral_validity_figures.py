from __future__ import annotations

import importlib
from pathlib import Path
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.utils.config.loader import ConfigDict
from studies.pain_study.study1.config.loader import load_study1_config
from studies.pain_study.study1.figures.behavioral_validity import (
    BehavioralValiditySummary,
)
from studies.pain_study.study1.figures.validity_data import ValidityTrialData


def test_coefficient_figure_draws_scientific_layers(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.coefficient_plot import (
        build_behavioral_validity_figure,
    )

    figure = build_behavioral_validity_figure(
        _summary("NPS"),
        color_config_key="nps",
        config=_config(tmp_path),
    )
    axis = figure.axes[0]

    assert axis.get_xlabel() == "Standardized partial coefficient (β)"
    assert [tick.get_text() for tick in axis.get_yticklabels()] == [
        "Within-scale intensity",
        "Painful report",
    ]
    assert axis.get_title() == ""
    assert axis.get_xlim()[0] == pytest.approx(-axis.get_xlim()[1])
    assert any(np.allclose(line.get_xdata(), [0.0, 0.0]) for line in axis.lines)
    assert len(axis.collections) >= 4
    assert [text.get_text() for text in axis.texts].count("n = 3") == 2
    figure.canvas.draw()
    assert axis.get_legend() is None
    assert len(figure.legends) == 1
    assert figure.legends[0].get_window_extent().y0 >= axis.get_window_extent().y1
    plt.close(figure)


@pytest.mark.parametrize(
    ("writer_name", "filename", "color", "target"),
    [
        (
            "write_nps_behavioral_validity",
            "nps_behavioral_validity.svg",
            "#0072b2",
            "NPS",
        ),
        (
            "write_siips1_behavioral_validity",
            "siips1_behavioral_validity.svg",
            "#d55e00",
            "SIIPS1",
        ),
    ],
)
def test_behavioral_validity_writer_creates_only_assigned_svg(
    tmp_path: Path,
    writer_name: str,
    filename: str,
    color: str,
    target: str,
) -> None:
    from studies.pain_study.study1 import figures

    writer = getattr(figures, writer_name)
    output_path = writer(summary=_summary(target), config=_config(tmp_path))

    assert output_path.name == filename
    assert output_path.parent.parts[-2:] == ("supplementary", "validity")
    assert sorted(output_path.parent.iterdir()) == [output_path]
    svg = output_path.read_text(encoding="utf-8").lower()
    assert color in svg
    assert "standardized partial coefficient" in svg
    assert "within-scale intensity" in svg

    root = ElementTree.parse(output_path).getroot()
    width_pt = float(root.attrib["width"].removesuffix("pt"))
    height_pt = float(root.attrib["height"].removesuffix("pt"))
    assert width_pt * 25.4 / 72.0 == pytest.approx(89.0, abs=0.01)
    assert height_pt * 25.4 / 72.0 == pytest.approx(70.0, abs=0.01)


@pytest.mark.parametrize(
    ("writer_name", "wrong_target", "expected_target"),
    [
        ("write_nps_behavioral_validity", "SIIPS1", "NPS"),
        ("write_siips1_behavioral_validity", "NPS", "SIIPS1"),
    ],
)
def test_target_writer_rejects_mismatched_summary(
    tmp_path: Path,
    writer_name: str,
    wrong_target: str,
    expected_target: str,
) -> None:
    from studies.pain_study.study1 import figures

    writer = getattr(figures, writer_name)

    with pytest.raises(ValueError, match=f"requires a {expected_target} summary"):
        writer(summary=_summary(wrong_target), config=_config(tmp_path))


@pytest.mark.parametrize(
    ("module_name", "filename"),
    [
        ("plot_nps_behavioral_validity", "nps_behavioral_validity.svg"),
        ("plot_siips1_behavioral_validity", "siips1_behavioral_validity.svg"),
    ],
)
def test_plot_module_main_writes_exactly_one_svg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    module_name: str,
    filename: str,
) -> None:
    module = importlib.import_module(f"studies.pain_study.study1.figures.{module_name}")
    config = _config(tmp_path)
    monkeypatch.setattr(module, "load_config", lambda path: config)
    monkeypatch.setattr(module, "apply_study1_config_defaults", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "load_validity_trial_data", lambda **kwargs: _trial_data())

    output_path = module.main(["--config", "pipeline.yaml", "--task", "thermalactive"])

    assert output_path.name == filename
    assert sorted(output_path.parent.iterdir()) == [output_path]
    assert capsys.readouterr().out.strip() == str(output_path)


def test_behavioral_validity_svg_is_byte_reproducible(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.coefficient_plot import (
        build_behavioral_validity_figure,
    )
    from studies.pain_study.study1.figures.validity_style import save_validity_svg

    config = _config(tmp_path)
    output_paths = [tmp_path / "first.svg", tmp_path / "second.svg"]
    for output_path in output_paths:
        figure = build_behavioral_validity_figure(
            _summary("NPS"),
            color_config_key="nps",
            config=config,
        )
        save_validity_svg(figure, output_path, config)

    assert output_paths[0].read_bytes() == output_paths[1].read_bytes()


def _config(tmp_path: Path) -> ConfigDict:
    config = load_study1_config()
    deriv_root = str(tmp_path / "derivatives")
    config["paths"] = {"deriv_root": deriv_root}
    config["deriv_root"] = deriv_root
    config["study1"]["cohort"]["min_subjects"] = 3
    validity = config["study1"]["figures"]["validity"]
    validity["temperatures"] = [44.3, 45.3, 46.3]
    validity["bootstrap"].update(
        iterations=30,
        confidence_level=0.95,
        seed=42,
        max_invalid_fraction=0.20,
    )
    return ConfigDict(config)


def _summary(target: str) -> BehavioralValiditySummary:
    participant_models = pd.DataFrame(
        {
            "subject_id": ["sub-01", "sub-02", "sub-03"],
            "target": [target] * 3,
            "estimable": [True] * 3,
            "non_estimability_reason": ["none"] * 3,
            "painful_report_beta": [0.15, 0.25, 0.35],
            "within_scale_intensity_beta": [0.30, 0.40, 0.50],
            "n_trials": [24] * 3,
            "residual_degrees_of_freedom": [19] * 3,
        }
    )
    cohort_estimates = pd.DataFrame(
        {
            "target": [target, target],
            "term": ["painful_report", "within_scale_intensity"],
            "mean": [0.25, 0.40],
            "ci_low": [0.15, 0.30],
            "ci_high": [0.35, 0.50],
            "n_subjects": [3, 3],
        }
    )
    return BehavioralValiditySummary(
        target=target,
        participant_models=participant_models,
        cohort_estimates=cohort_estimates,
    )


def _trial_data() -> ValidityTrialData:
    rows: list[dict[str, float | str]] = []
    pain_pattern = np.array([0.0, 0.0, 1.0, 1.0])
    intensity_pattern = np.array([20.0, 60.0, 60.0, 20.0])
    nuisance_pattern = np.array([1.0, -1.0, 1.0, -1.0])
    for subject_index, subject_id in enumerate(("sub-01", "sub-02", "sub-03")):
        for temperature_index, temperature in enumerate((44.3, 45.3, 46.3)):
            for pain_report, intensity, nuisance in zip(
                pain_pattern,
                intensity_pattern,
                nuisance_pattern,
                strict=True,
            ):
                nps = (
                    0.5 * pain_report
                    + 0.01 * intensity
                    + 0.2 * nuisance
                    + 0.1 * temperature_index
                    + 0.05 * subject_index
                )
                rows.append(
                    {
                        "subject_id": subject_id,
                        "stimulus_temp": temperature,
                        "pain_binary_coded": pain_report,
                        "within_scale_intensity": intensity,
                        "NPS": nps,
                        "SIIPS1": (
                            0.4 * pain_report
                            + 0.02 * intensity
                            + 0.3 * nps
                            + 0.15 * nuisance
                            - 0.1 * temperature_index
                        ),
                    }
                )
    enriched = pd.DataFrame(rows)
    return ValidityTrialData(
        targets=enriched[["subject_id", "stimulus_temp", "NPS", "SIIPS1"]].copy(),
        clean_events=pd.DataFrame(),
        enriched_targets=enriched,
    )
