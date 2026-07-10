from __future__ import annotations

import importlib
from pathlib import Path
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib import font_manager

from eeg_pipeline.utils.config.loader import ConfigDict
from studies.pain_study.study1.config.loader import load_study1_config
from studies.pain_study.study1.figures.validity_data import (
    DoseResponseSummary,
    ValidityTrialData,
)


def test_require_configured_font_rejects_missing_font(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studies.pain_study.study1.figures.validity_style import require_configured_font

    def raise_missing(*args, **kwargs):
        raise ValueError("font missing")

    monkeypatch.setattr(font_manager, "findfont", raise_missing)

    with pytest.raises(ValueError, match="Required figure font 'Arial' is unavailable"):
        require_configured_font(_config(tmp_path))


def test_build_dose_response_figure_draws_scientific_layers(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.dose_response import (
        DoseResponseSpecification,
        build_dose_response_figure,
    )

    figure = build_dose_response_figure(
        summary=_summary(),
        specification=DoseResponseSpecification(
            ylabel="NPS expression (a.u.)",
            color_config_key="nps",
        ),
        config=_config(tmp_path),
    )
    axis = figure.axes[0]

    assert axis.get_xlabel() == "Temperature (°C)"
    assert axis.get_ylabel() == "NPS expression (a.u.)"
    assert axis.get_title() == ""
    assert axis.get_xticks().tolist() == [44.3, 49.3]
    assert not axis.spines["top"].get_visible()
    assert not axis.spines["right"].get_visible()
    assert not any(
        line.get_visible()
        for line in [*axis.get_xgridlines(), *axis.get_ygridlines()]
    )
    assert len(axis.lines) >= 4
    assert len(axis.collections) >= 1
    plt.close(figure)


def test_save_validity_svg_writes_one_editable_publication_svg(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.validity_style import save_validity_svg

    figure, axis = plt.subplots()
    axis.set_xlabel("Temperature (°C)")
    axis.plot([44.3, 49.3], [1.0, 2.0])
    output_path = tmp_path / "figure.svg"

    save_validity_svg(figure, output_path, _config(tmp_path))

    root = ElementTree.parse(output_path).getroot()
    width_pt = float(root.attrib["width"].removesuffix("pt"))
    height_pt = float(root.attrib["height"].removesuffix("pt"))
    assert width_pt * 25.4 / 72.0 == pytest.approx(89.0, abs=0.01)
    assert height_pt * 25.4 / 72.0 == pytest.approx(70.0, abs=0.01)
    assert output_path.read_text(encoding="utf-8").count("<text") > 0
    assert sorted(tmp_path.iterdir()) == [output_path]


def test_save_validity_svg_is_byte_reproducible(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.validity_style import save_validity_svg

    output_paths = [tmp_path / "first.svg", tmp_path / "second.svg"]
    for output_path in output_paths:
        figure, axis = plt.subplots()
        axis.set_xlabel("Temperature (°C)")
        axis.plot([44.3, 49.3], [1.0, 2.0])
        save_validity_svg(figure, output_path, _config(tmp_path))

    assert output_paths[0].read_bytes() == output_paths[1].read_bytes()


@pytest.mark.parametrize(
    ("writer_name", "filename", "label", "color"),
    [
        (
            "write_behavioral_dose_response",
            "behavioral_dose_response.svg",
            "Displayed rating",
            "#222222",
        ),
        (
            "write_nps_dose_response",
            "nps_dose_response.svg",
            "NPS expression (a.u.)",
            "#0072B2",
        ),
        (
            "write_siips1_dose_response",
            "siips1_dose_response.svg",
            "SIIPS1 expression (a.u.)",
            "#D55E00",
        ),
    ],
)
def test_validity_writer_creates_only_its_assigned_svg(
    tmp_path: Path,
    writer_name: str,
    filename: str,
    label: str,
    color: str,
) -> None:
    from studies.pain_study.study1 import figures

    writer = getattr(figures, writer_name)
    config = _config(tmp_path)
    output_path = writer(trial_data=_trial_data(), config=config)

    assert output_path.name == filename
    assert output_path.parent.parts[-2:] == ("supplementary", "validity")
    assert sorted(output_path.parent.iterdir()) == [output_path]
    svg = output_path.read_text(encoding="utf-8").lower()
    assert label.lower() in svg
    assert color.lower() in svg


def test_behavioral_plot_marks_protocol_pain_threshold(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures import write_behavioral_dose_response

    output_path = write_behavioral_dose_response(
        trial_data=_trial_data(),
        config=_config(tmp_path),
    )

    text = " ".join(ElementTree.parse(output_path).getroot().itertext())
    assert "Pain threshold" in text
    assert "100" in text


@pytest.mark.parametrize(
    ("module_name", "filename"),
    [
        ("plot_behavioral_dose_response", "behavioral_dose_response.svg"),
        ("plot_nps_dose_response", "nps_dose_response.svg"),
        ("plot_siips1_dose_response", "siips1_dose_response.svg"),
    ],
)
def test_plot_module_main_writes_its_svg(
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
    assert output_path.exists()
    assert capsys.readouterr().out.strip() == str(output_path)


def _config(tmp_path: Path) -> ConfigDict:
    config = load_study1_config()
    deriv_root = str(tmp_path / "derivatives")
    config["paths"] = {"deriv_root": deriv_root}
    config["deriv_root"] = deriv_root
    validity = config["study1"]["figures"]["validity"]
    validity["temperatures"] = [44.3, 49.3]
    validity["bootstrap"].update(
        iterations=30,
        confidence_level=0.95,
        seed=42,
        max_invalid_fraction=0.20,
    )
    return ConfigDict(config)


def _summary() -> DoseResponseSummary:
    temperatures = (44.3, 49.3)
    matrix = pd.DataFrame(
        [[1.0, 4.0], [2.0, 6.0], [1.5, 5.0]],
        index=["sub-01", "sub-02", "sub-03"],
        columns=temperatures,
    )
    participant_means = (
        matrix.rename_axis("subject_id")
        .reset_index()
        .melt(id_vars="subject_id", var_name="stimulus_temp", value_name="value")
    )
    cohort = pd.DataFrame(
        {
            "stimulus_temp": temperatures,
            "mean": [1.5, 5.0],
            "ci_low": [1.0, 4.0],
            "ci_high": [2.0, 6.0],
            "n_subjects": [3, 3],
        }
    )
    return DoseResponseSummary(
        outcome="NPS",
        temperatures=temperatures,
        participant_means=participant_means,
        participant_matrix=matrix,
        cohort_estimates=cohort,
    )


def _trial_data() -> ValidityTrialData:
    rows: list[dict[str, object]] = []
    for subject_index, subject_id in enumerate(("sub-01", "sub-02", "sub-03")):
        for repetition in range(2):
            for temperature in (44.3, 49.3):
                rows.append(
                    {
                        "subject_id": subject_id,
                        "task": "thermalactive",
                        "run": repetition + 1,
                        "trial_index": repetition * 2 + int(temperature > 45.0) + 1,
                        "within_run_trial": int(temperature > 45.0) + 1,
                        "stimulus_temp": temperature,
                        "NPS": subject_index + temperature / 10.0,
                        "SIIPS1": subject_index * 100.0 + temperature * 20.0,
                        "vas_final_coded_rating": (
                            20.0 + subject_index if temperature < 45.0 else 160.0 + subject_index
                        ),
                    }
                )
    enriched = pd.DataFrame(rows)
    targets = enriched.drop(columns="vas_final_coded_rating")
    return ValidityTrialData(
        targets=targets,
        clean_events=pd.DataFrame(),
        enriched_targets=enriched,
    )
