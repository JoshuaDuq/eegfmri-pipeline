from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.utils.config.loader import ConfigDict
from studies.pain_study.study1.config.loader import load_study1_config


def test_load_validity_trial_data_uses_only_retained_target_trials(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.validity_data import load_validity_trial_data

    config = _config(tmp_path)
    _write_targets(config)
    _write_events(config, include_extra_event=True)

    data = load_validity_trial_data(task="thermalactive", config=config)

    assert len(data.targets) == 12
    assert len(data.enriched_targets) == 12
    assert set(data.enriched_targets["trial_index"]) == {1, 2, 3, 4}
    assert "run" in data.clean_events
    assert "run_id" not in data.clean_events
    assert data.clean_events["trial_number"].max() == 3
    assert data.enriched_targets["within_scale_intensity"].tolist() == [
        20.0,
        60.0,
        25.0,
        65.0,
    ] * 3


def test_load_validity_trial_data_rejects_duplicate_target_keys(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.validity_data import load_validity_trial_data

    config = _config(tmp_path)
    _write_targets(config, duplicate_first_row=True)
    _write_events(config)

    with pytest.raises(ValueError, match="duplicate target trial keys"):
        load_validity_trial_data(task="thermalactive", config=config)


def test_load_validity_trial_data_rejects_temperature_disagreement(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.validity_data import load_validity_trial_data

    config = _config(tmp_path)
    _write_targets(config)
    _write_events(config, first_temperature=46.3)

    with pytest.raises(ValueError, match="stimulus temperature disagrees"):
        load_validity_trial_data(task="thermalactive", config=config)


@pytest.mark.parametrize("rating", [-0.1, 200.1])
def test_load_validity_trial_data_rejects_out_of_range_retained_rating(
    tmp_path: Path,
    rating: float,
) -> None:
    from studies.pain_study.study1.figures.validity_data import load_validity_trial_data

    config = _config(tmp_path)
    _write_targets(config)
    _write_events(config, first_rating=rating)

    with pytest.raises(ValueError, match=r"within \[0, 200\]"):
        load_validity_trial_data(task="thermalactive", config=config)


def test_add_within_scale_intensity_uses_protocol_scales() -> None:
    from studies.pain_study.study1.figures.validity_data import (
        add_within_scale_intensity,
    )

    trials = pd.DataFrame(
        {
            "pain_binary_coded": [0, 0, 1, 1],
            "vas_final_coded_rating": [0.0, 99.0, 100.0, 200.0],
        }
    )

    scored = add_within_scale_intensity(trials)

    assert scored["within_scale_intensity"].tolist() == [0.0, 99.0, 0.0, 100.0]
    assert "within_scale_intensity" not in trials.columns


@pytest.mark.parametrize(
    ("pain_report", "rating"),
    [
        (0, 100.0),
        (1, 99.0),
        (2, 150.0),
        (0.5, 50.0),
    ],
)
def test_add_within_scale_intensity_rejects_inconsistent_protocol_codes(
    pain_report: float,
    rating: float,
) -> None:
    from studies.pain_study.study1.figures.validity_data import (
        add_within_scale_intensity,
    )

    trials = pd.DataFrame(
        {
            "pain_binary_coded": [pain_report],
            "vas_final_coded_rating": [rating],
        }
    )

    with pytest.raises(ValueError, match="pain/rating protocol coding"):
        add_within_scale_intensity(trials)


def test_build_dose_response_summary_weights_participants_equally(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.validity_data import build_dose_response_summary

    config = _config(tmp_path)
    trials = pd.DataFrame(
        {
            "subject_id": ["sub-01"] * 4 + ["sub-02"] * 2,
            "stimulus_temp": [44.3, 44.3, 44.3, 49.3, 44.3, 49.3],
            "NPS": [0.0, 0.0, 0.0, 4.0, 2.0, 8.0],
        }
    )

    summary = build_dose_response_summary(trials, outcome="NPS", config=config)

    low_temperature = summary.participant_means.loc[
        summary.participant_means["stimulus_temp"] == 44.3,
        "value",
    ]
    assert low_temperature.tolist() == [0.0, 2.0]
    cohort = summary.cohort_estimates.set_index("stimulus_temp")
    assert cohort.loc[44.3, "mean"] == pytest.approx(1.0)
    assert cohort.loc[49.3, "mean"] == pytest.approx(6.0)


def test_build_dose_response_summary_preserves_missing_cell_gap(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.validity_data import build_dose_response_summary

    config = _config(tmp_path)
    trials = _summary_trials().query(
        "not (subject_id == 'sub-03' and stimulus_temp == 49.3)"
    )

    summary = build_dose_response_summary(trials, outcome="NPS", config=config)

    assert np.isnan(summary.participant_matrix.loc["sub-03", 49.3])
    assert summary.cohort_estimates.set_index("stimulus_temp").loc[49.3, "n_subjects"] == 2


def test_build_dose_response_summary_is_deterministic(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.validity_data import build_dose_response_summary

    config = _config(tmp_path)

    first = build_dose_response_summary(_summary_trials(), outcome="NPS", config=config)
    second = build_dose_response_summary(_summary_trials(), outcome="NPS", config=config)

    pd.testing.assert_frame_equal(first.cohort_estimates, second.cohort_estimates)


def test_build_dose_response_summary_enforces_invalid_draw_budget(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.validity_data import build_dose_response_summary

    config = _config(tmp_path)
    bootstrap = config["study1"]["figures"]["validity"]["bootstrap"]
    bootstrap.update(iterations=1, seed=4, max_invalid_fraction=0.0)
    trials = pd.DataFrame(
        {
            "subject_id": ["sub-01", "sub-02", "sub-03", "sub-04"],
            "stimulus_temp": [44.3, 44.3, 49.3, 49.3],
            "NPS": [1.0, 2.0, 3.0, 4.0],
        }
    )

    with pytest.raises(ValueError, match="valid=0, attempted=1, invalid=1"):
        build_dose_response_summary(trials, outcome="NPS", config=config)


def _config(tmp_path: Path) -> ConfigDict:
    config = load_study1_config()
    deriv_root = str(tmp_path / "derivatives")
    config["paths"] = {"deriv_root": deriv_root}
    config["deriv_root"] = deriv_root
    validity = config["study1"]["figures"]["validity"]
    validity["temperatures"] = [44.3, 49.3]
    validity["bootstrap"].update(
        iterations=40,
        confidence_level=0.95,
        seed=42,
        max_invalid_fraction=0.50,
    )
    return ConfigDict(config)


def _write_targets(
    config: ConfigDict,
    *,
    duplicate_first_row: bool = False,
) -> None:
    rows: list[dict[str, object]] = []
    for subject_index, subject_id in enumerate(("sub-01", "sub-02", "sub-03")):
        for trial_index, (run, trial_number, temperature) in enumerate(
            ((1, 1, 44.3), (1, 2, 49.3), (2, 1, 44.3), (2, 2, 49.3)),
            start=1,
        ):
            rows.append(
                {
                    "subject_id": subject_id,
                    "task": "thermalactive",
                    "run": run,
                    "trial_index": trial_index,
                    "within_run_trial": trial_number,
                    "onset": float(trial_index * 10),
                    "duration": 0.001,
                    "NPS": float(subject_index + temperature / 10.0),
                    "SIIPS1": float(subject_index * 100 + temperature * 20.0),
                    "stimulus_temp": temperature,
                    "selected_surface": 1,
                }
            )
    if duplicate_first_row:
        rows.append(dict(rows[0]))
    target_dir = (
        Path(config.get("paths.deriv_root"))
        / "group"
        / "multimodal"
        / "study1"
        / "targets"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(target_dir / "primary_targets.parquet", index=False)


def _write_events(
    config: ConfigDict,
    *,
    include_extra_event: bool = False,
    first_temperature: float = 44.3,
    first_rating: float = 20.0,
) -> None:
    for subject_id in ("sub-01", "sub-02", "sub-03"):
        rows = [
            _event_row(1, 1, first_temperature, first_rating),
            _event_row(1, 2, 49.3, 160.0),
            _event_row(2, 1, 44.3, 25.0),
            _event_row(2, 2, 49.3, 165.0),
        ]
        if include_extra_event:
            rows.append(_event_row(2, 3, 49.3, 170.0))
        event_dir = (
            Path(config.get("paths.deriv_root"))
            / "preprocessed"
            / "eeg"
            / subject_id
            / "eeg"
        )
        event_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            event_dir / f"{subject_id}_task-thermalactive_proc-clean_events.tsv",
            sep="\t",
            index=False,
        )


def _event_row(
    run: int,
    trial_number: int,
    temperature: float,
    rating: float,
) -> dict[str, object]:
    return {
        "run_id": run,
        "trial_number": trial_number,
        "stimulus_temp": temperature,
        "selected_surface": 1,
        "pain_binary_coded": int(rating >= 100.0),
        "vas_final_coded_rating": rating,
        "residual_ecg_coupling": 0.04,
        "fp1_fp2_high_frequency_power": 0.05,
    }


def _summary_trials() -> pd.DataFrame:
    rows = []
    for subject_index, subject_id in enumerate(("sub-01", "sub-02", "sub-03")):
        for temperature in (44.3, 49.3):
            rows.append(
                {
                    "subject_id": subject_id,
                    "stimulus_temp": temperature,
                    "NPS": float(subject_index + temperature / 10.0),
                }
            )
    return pd.DataFrame(rows)
