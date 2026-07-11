from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.utils.config.loader import ConfigDict


def test_target_specifications_preserve_distinct_adjustment_models() -> None:
    from studies.pain_study.study1.figures.behavioral_validity import (
        NPS_SPECIFICATION,
        SIIPS1_SPECIFICATION,
    )

    assert NPS_SPECIFICATION.target == "NPS"
    assert NPS_SPECIFICATION.adjustment_columns == ()
    assert SIIPS1_SPECIFICATION.target == "SIIPS1"
    assert SIIPS1_SPECIFICATION.adjustment_columns == ("NPS",)


@pytest.mark.parametrize("specification_name", ["NPS_SPECIFICATION", "SIIPS1_SPECIFICATION"])
def test_summary_matches_independent_participant_ols(specification_name: str) -> None:
    from studies.pain_study.study1.figures import behavioral_validity

    specification = getattr(behavioral_validity, specification_name)
    trials = _synthetic_trials()

    summary = behavioral_validity.build_behavioral_validity_summary(
        trials,
        specification=specification,
        config=_config(),
    )

    estimable = summary.participant_models.query("estimable")
    assert estimable["subject_id"].tolist() == ["sub-01", "sub-02", "sub-03"]
    for _, participant in estimable.iterrows():
        rows = trials.loc[trials["subject_id"] == participant["subject_id"]]
        expected = _independent_coefficients(rows, specification)
        assert participant["painful_report_beta"] == pytest.approx(expected[0])
        assert participant["within_scale_intensity_beta"] == pytest.approx(expected[1])

    assert summary.cohort_estimates["term"].tolist() == [
        "painful_report",
        "within_scale_intensity",
    ]
    assert summary.cohort_estimates["n_subjects"].tolist() == [3, 3]


def test_summary_is_deterministic_and_weights_participants_equally() -> None:
    from studies.pain_study.study1.figures.behavioral_validity import (
        NPS_SPECIFICATION,
        build_behavioral_validity_summary,
    )

    trials = _synthetic_trials()
    first = build_behavioral_validity_summary(
        trials,
        specification=NPS_SPECIFICATION,
        config=_config(),
    )
    second = build_behavioral_validity_summary(
        trials,
        specification=NPS_SPECIFICATION,
        config=_config(),
    )

    pd.testing.assert_frame_equal(first.cohort_estimates, second.cohort_estimates)
    participants = first.participant_models.query("estimable")
    cohort = first.cohort_estimates.set_index("term")
    assert cohort.loc["painful_report", "mean"] == pytest.approx(
        participants["painful_report_beta"].mean()
    )
    assert cohort.loc["within_scale_intensity", "mean"] == pytest.approx(
        participants["within_scale_intensity_beta"].mean()
    )


def test_constant_pain_report_is_recorded_as_non_estimable() -> None:
    from studies.pain_study.study1.figures.behavioral_validity import (
        NPS_SPECIFICATION,
        build_behavioral_validity_summary,
    )

    trials = _synthetic_trials()
    subject_mask = trials["subject_id"] == "sub-03"
    trials.loc[subject_mask, "pain_binary_coded"] = 1.0

    summary = build_behavioral_validity_summary(
        trials,
        specification=NPS_SPECIFICATION,
        config=_config(min_subjects=2),
    )

    excluded = summary.participant_models.query("not estimable").iloc[0]
    assert excluded["subject_id"] == "sub-03"
    assert excluded["non_estimability_reason"] == "constant_pain_binary_coded"
    assert np.isnan(excluded["painful_report_beta"])


def test_missing_temperature_is_recorded_as_non_estimable() -> None:
    from studies.pain_study.study1.figures.behavioral_validity import (
        NPS_SPECIFICATION,
        build_behavioral_validity_summary,
    )

    trials = _synthetic_trials().query(
        "not (subject_id == 'sub-03' and stimulus_temp == 46.3)"
    )

    summary = build_behavioral_validity_summary(
        trials,
        specification=NPS_SPECIFICATION,
        config=_config(min_subjects=2),
    )

    excluded = summary.participant_models.query("not estimable").iloc[0]
    assert excluded["non_estimability_reason"] == "missing_temperature_levels"


def test_rank_deficiency_is_recorded_as_non_estimable() -> None:
    from studies.pain_study.study1.figures.behavioral_validity import (
        NPS_SPECIFICATION,
        build_behavioral_validity_summary,
    )

    trials = _synthetic_trials()
    subject_mask = trials["subject_id"] == "sub-03"
    trials.loc[subject_mask, "within_scale_intensity"] = (
        10.0 + 20.0 * trials.loc[subject_mask, "pain_binary_coded"]
    )

    summary = build_behavioral_validity_summary(
        trials,
        specification=NPS_SPECIFICATION,
        config=_config(min_subjects=2),
    )

    excluded = summary.participant_models.query("not estimable").iloc[0]
    assert excluded["non_estimability_reason"] == "rank_deficient_design"


def test_summary_rejects_too_few_estimable_participants() -> None:
    from studies.pain_study.study1.figures.behavioral_validity import (
        NPS_SPECIFICATION,
        build_behavioral_validity_summary,
    )

    trials = _synthetic_trials().query("subject_id != 'sub-03'").copy()

    with pytest.raises(ValueError, match="requires at least 3 estimable participants"):
        build_behavioral_validity_summary(
            trials,
            specification=NPS_SPECIFICATION,
            config=_config(min_subjects=3),
        )


def _config(*, min_subjects: int = 3) -> ConfigDict:
    return ConfigDict(
        {
            "study1": {
                "cohort": {"min_subjects": min_subjects},
                "figures": {
                    "validity": {
                        "temperatures": [44.3, 45.3, 46.3],
                        "bootstrap": {
                            "iterations": 100,
                            "confidence_level": 0.95,
                            "seed": 42,
                            "max_invalid_fraction": 0.20,
                        },
                    }
                },
            }
        }
    )


def _synthetic_trials() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    pain_pattern = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    intensity_pattern = np.array([10.0, 30.0, 50.0, 70.0, 70.0, 50.0, 30.0, 10.0])
    nuisance_pattern = np.array([1.0, -1.0, -0.5, 0.5, -0.75, 0.25, 0.75, -0.25])
    temperatures = (44.3, 45.3, 46.3)
    for subject_index, subject_id in enumerate(("sub-01", "sub-02", "sub-03")):
        subject_rows: list[dict[str, float | str]] = []
        for temperature_index, temperature in enumerate(temperatures):
            for pain_report, intensity, nuisance in zip(
                pain_pattern,
                intensity_pattern,
                nuisance_pattern,
                strict=True,
            ):
                subject_rows.append(
                    {
                        "subject_id": subject_id,
                        "stimulus_temp": temperature,
                        "pain_binary_coded": pain_report,
                        "within_scale_intensity": intensity,
                        "_temperature_index": float(temperature_index),
                        "_nuisance": nuisance + 0.1 * subject_index,
                    }
                )
        subject = pd.DataFrame(subject_rows)
        pain_z = _standardize(subject["pain_binary_coded"])
        intensity_z = _standardize(subject["within_scale_intensity"])
        nuisance_z = _standardize(subject["_nuisance"])
        subject["NPS"] = (
            0.55 * pain_z
            + 0.25 * intensity_z
            + 0.40 * nuisance_z
            + 0.20 * subject["_temperature_index"]
        )
        nps_z = _standardize(subject["NPS"])
        subject["SIIPS1"] = (
            0.30 * pain_z
            + 0.45 * intensity_z
            + 0.35 * nps_z
            + 0.50 * nuisance_z
            - 0.10 * subject["_temperature_index"]
        )
        rows.extend(
            subject.drop(columns=["_temperature_index", "_nuisance"]).to_dict("records")
        )
    return pd.DataFrame(rows)


def _independent_coefficients(rows: pd.DataFrame, specification) -> tuple[float, float]:
    ordered = rows.sort_values("stimulus_temp", kind="stable")
    temperature = pd.Categorical(
        ordered["stimulus_temp"],
        categories=(44.3, 45.3, 46.3),
        ordered=True,
    )
    temperature_dummies = pd.get_dummies(temperature, drop_first=True, dtype=float)
    columns = [np.ones(len(ordered)), *temperature_dummies.to_numpy(dtype=float).T]
    for adjustment in specification.adjustment_columns:
        columns.append(_standardize(ordered[adjustment]).to_numpy(dtype=float))
    columns.extend(
        [
            _standardize(ordered["pain_binary_coded"]).to_numpy(dtype=float),
            _standardize(ordered["within_scale_intensity"]).to_numpy(dtype=float),
        ]
    )
    design = np.column_stack(columns)
    target = _standardize(ordered[specification.target]).to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(coefficients[-2]), float(coefficients[-1])


def _standardize(values: pd.Series) -> pd.Series:
    return (values - values.mean()) / values.std(ddof=1)
