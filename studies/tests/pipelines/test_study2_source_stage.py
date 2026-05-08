from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _source_stage_frame(
    *,
    n_blocks: int = 3,
    trials_per_block: int = 11,
    collinear_opposite_band: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows: list[dict[str, float | int | str]] = []
    for block in range(1, n_blocks + 1):
        for trial_index in range(1, trials_per_block + 1):
            rows.append(
                {
                    "subject_id": "sub-0001",
                    "block": block,
                    "trial_index": trial_index,
                    "onset": float(rng.normal()),
                    "framewise_displacement": float(rng.normal(scale=0.1)),
                    "std_dvars": float(rng.normal(scale=0.1)),
                    "fp1_fp2_high_frequency_power": float(rng.normal(scale=0.1)),
                    "residual_ecg_coupling": float(rng.normal(scale=0.1)),
                    "stimulus_temp": float(rng.choice([44.3, 46.3, 48.3])),
                    "selected_surface": float(rng.choice([1, 2, 3, 4, 5])),
                    "eta_alpha_z": float(rng.normal()),
                    "eta_beta_z": float(rng.normal()),
                }
            )
    frame = pd.DataFrame(rows)
    if collinear_opposite_band:
        centered = frame["onset"] - frame["onset"].mean()
        onset_z = centered / centered.std(ddof=0)
        frame["eta_beta_z"] = 0.90 * onset_z + 0.10 * rng.normal(size=len(frame))
    return frame


def test_evaluate_source_stage_subject_accepts_valid_readme_design() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_stage import evaluate_source_stage_subject

    qc = evaluate_source_stage_subject(
        _source_stage_frame(),
        band="alpha",
        config=load_study2_config(),
    )

    assert qc.eligible is True
    assert qc.subject_id == "sub-0001"
    assert qc.band == "alpha"
    assert qc.retained_trials == 33
    assert qc.valid_blocks == 3
    assert qc.residual_degrees_of_freedom >= 15
    assert qc.condition_number <= 100
    assert qc.opposite_band_vif <= 5
    assert qc.reason == ""


def test_evaluate_source_stage_subject_rejects_too_few_valid_blocks() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_stage import evaluate_source_stage_subject

    qc = evaluate_source_stage_subject(
        _source_stage_frame(n_blocks=2),
        band="alpha",
        config=load_study2_config(),
    )

    assert qc.eligible is False
    assert qc.valid_blocks == 2
    assert "valid_blocks=2" in qc.reason


def test_evaluate_source_stage_subject_rejects_high_opposite_band_vif() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_stage import evaluate_source_stage_subject

    qc = evaluate_source_stage_subject(
        _source_stage_frame(collinear_opposite_band=True),
        band="alpha",
        config=load_study2_config(),
    )

    assert qc.eligible is False
    assert qc.opposite_band_vif > 5
    assert "opposite-band VIF" in qc.reason


def test_evaluate_source_stage_subject_requires_one_subject() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_stage import evaluate_source_stage_subject

    frame = pd.concat(
        [
            _source_stage_frame(),
            _source_stage_frame().assign(subject_id="sub-0002"),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="exactly one subject"):
        evaluate_source_stage_subject(
            frame,
            band="alpha",
            config=load_study2_config(),
        )


def test_evaluate_source_stage_cohort_accepts_enough_source_valid_subjects() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_stage import evaluate_source_stage_cohort

    config = load_study2_config(
        "studies/pain_study/study2/config/study2_smoketest.yaml"
    )
    frame = pd.concat(
        [
            _source_stage_frame().assign(subject_id="sub-0001"),
            _source_stage_frame().assign(subject_id="sub-0002"),
        ],
        ignore_index=True,
    )

    qc, status = evaluate_source_stage_cohort(frame, band="alpha", config=config)

    assert qc["eligible"].tolist() == [True, True]
    assert status.confirmatory_eligible is True
    assert status.n_source_valid_subjects == 2
    assert status.reason == ""


def test_evaluate_source_stage_cohort_downgrades_for_too_few_valid_subjects() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_stage import evaluate_source_stage_cohort

    config = load_study2_config(
        "studies/pain_study/study2/config/study2_smoketest.yaml"
    )
    frame = pd.concat(
        [
            _source_stage_frame().assign(subject_id="sub-0001"),
            _source_stage_frame(n_blocks=1, trials_per_block=7).assign(subject_id="sub-0002"),
        ],
        ignore_index=True,
    )

    qc, status = evaluate_source_stage_cohort(frame, band="alpha", config=config)

    assert qc["eligible"].tolist() == [True, False]
    assert status.confirmatory_eligible is False
    assert status.n_source_valid_subjects == 1
    assert "fewer than 2 source-valid subjects" in status.reason


def test_evaluate_source_stage_cohort_downgrades_for_collinearity_failure_fraction() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_stage import evaluate_source_stage_cohort

    config = load_study2_config(
        "studies/pain_study/study2/config/study2_smoketest.yaml"
    )
    frame = pd.concat(
        [
            _source_stage_frame().assign(subject_id="sub-0001"),
            _source_stage_frame().assign(subject_id="sub-0002"),
            _source_stage_frame(collinear_opposite_band=True).assign(subject_id="sub-0003"),
        ],
        ignore_index=True,
    )

    qc, status = evaluate_source_stage_cohort(frame, band="alpha", config=config)

    assert qc["eligible"].tolist() == [True, True, False]
    assert status.confirmatory_eligible is False
    assert status.n_source_valid_subjects == 2
    assert status.collinearity_failure_fraction == pytest.approx(1 / 3)
    assert "collinearity failure fraction" in status.reason
