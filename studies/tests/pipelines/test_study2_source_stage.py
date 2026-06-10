from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest


STIMULUS_TEMPS = (44.3, 45.3, 46.3, 47.3, 48.3, 49.3)
SELECTED_SURFACES = (1, 2, 3, 4, 5)


def _source_stage_frame(
    *,
    n_blocks: int = 6,
    trials_per_block: int = 11,
    collinear_combined_score: bool = False,
    collinear_adjacent_band: bool = False,
    collinear_target_band: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows: list[dict[str, float | int | str]] = []
    for block in range(1, n_blocks + 1):
        for trial_index in range(1, trials_per_block + 1):
            trial_offset = (block - 1) * trials_per_block + trial_index - 1
            rows.append(
                {
                    "subject_id": "sub-0001",
                    "block": block,
                    "trial_index": trial_index,
                    "trial_index_within_block": trial_index,
                    "onset": float(rng.normal()),
                    "hrf_weighted_framewise_displacement": float(rng.normal(scale=0.1)),
                    "hrf_weighted_std_dvars": float(rng.normal(scale=0.1)),
                    "hrf_weighted_fp1_fp2_high_frequency_power": float(
                        rng.normal(scale=0.1)
                    ),
                    "residual_ecg_coupling": float(rng.normal(scale=0.1)),
                    "stimulus_temp": STIMULUS_TEMPS[trial_offset % len(STIMULUS_TEMPS)],
                    "selected_surface": SELECTED_SURFACES[
                        trial_offset % len(SELECTED_SURFACES)
                    ],
                    "eta_combined_z": float(rng.normal()),
                    "eta_alpha_z": float(rng.normal()),
                    "eta_beta_z": float(rng.normal()),
                    "eta_gamma_z": float(rng.normal()),
                }
            )
    frame = pd.DataFrame(rows)
    centered = frame["onset"] - frame["onset"].mean()
    onset_z = centered / centered.std(ddof=0)
    if collinear_combined_score:
        frame["eta_combined_z"] = onset_z
    if collinear_adjacent_band:
        frame["eta_beta_z"] = 0.90 * onset_z + 0.10 * rng.normal(size=len(frame))
    if collinear_target_band:
        frame["eta_alpha_z"] = onset_z
    return frame


def test_evaluate_source_stage_subject_accepts_valid_combined_score_design() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_stage import evaluate_source_stage_subject

    qc = evaluate_source_stage_subject(
        _source_stage_frame(),
        config=load_study2_config(),
    )

    assert qc.eligible is True
    assert qc.subject_id == "sub-0001"
    assert qc.band == "combined"
    assert qc.retained_trials == 66
    assert qc.valid_blocks == 6
    assert qc.residual_degrees_of_freedom >= 15
    assert qc.condition_number <= 100
    assert math.isnan(qc.max_adjacent_band_vif)
    assert qc.reason == ""


def test_evaluate_source_stage_subject_omits_unobserved_fixed_levels() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_stage import evaluate_source_stage_subject

    frame = _source_stage_frame()
    frame = frame[frame["selected_surface"] != 5].reset_index(drop=True)

    qc = evaluate_source_stage_subject(frame, config=load_study2_config())

    assert qc.eligible is True
    assert qc.reason == ""


def test_evaluate_source_stage_subject_rejects_raw_level2_artifact_columns() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_stage import evaluate_source_stage_subject

    config = load_study2_config()
    config["study2"]["source_stage"]["continuous_columns"] = [
        "onset",
        "trial_index_within_block",
        "framewise_displacement",
        "std_dvars",
        "fp1_fp2_high_frequency_power",
        "residual_ecg_coupling",
    ]

    with pytest.raises(ValueError, match="HRF-weighted"):
        evaluate_source_stage_subject(_source_stage_frame(), config=config)


def test_evaluate_source_stage_subject_rejects_too_few_valid_blocks() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_stage import evaluate_source_stage_subject

    qc = evaluate_source_stage_subject(
        _source_stage_frame(n_blocks=2),
        config=load_study2_config(),
    )

    assert qc.eligible is False
    assert qc.valid_blocks == 2
    assert "valid_blocks=2" in qc.reason


def test_evaluate_source_stage_subject_requires_combined_score_column() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_stage import evaluate_source_stage_subject

    frame = _source_stage_frame().drop(columns=["eta_combined_z"])

    qc = evaluate_source_stage_subject(frame, config=load_study2_config())

    assert qc.eligible is False
    assert "eta_combined_z" in qc.reason


def test_evaluate_source_stage_subject_rejects_collinear_combined_score() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_stage import evaluate_source_stage_subject

    qc = evaluate_source_stage_subject(
        _source_stage_frame(collinear_combined_score=True),
        config=load_study2_config(),
    )

    assert qc.eligible is False
    assert "contribution design" in qc.reason


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
        evaluate_source_stage_subject(frame, config=load_study2_config())


def test_evaluate_band_unique_source_stage_subject_accepts_valid_design() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_stage import (
        evaluate_band_unique_source_stage_subject,
    )

    qc = evaluate_band_unique_source_stage_subject(
        _source_stage_frame(),
        band="alpha",
        config=load_study2_config(),
    )

    assert qc.eligible is True
    assert qc.band == "alpha"
    assert qc.max_adjacent_band_vif <= 5
    assert qc.reason == ""


def test_evaluate_band_unique_source_stage_subject_rejects_high_adjacent_band_vif() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_stage import (
        evaluate_band_unique_source_stage_subject,
    )

    qc = evaluate_band_unique_source_stage_subject(
        _source_stage_frame(collinear_adjacent_band=True),
        band="alpha",
        config=load_study2_config(),
    )

    assert qc.eligible is False
    assert qc.max_adjacent_band_vif > 5
    assert "adjacent-band VIF" in qc.reason


def test_evaluate_band_unique_source_stage_subject_rejects_collinear_target_band() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_stage import (
        evaluate_band_unique_source_stage_subject,
    )

    qc = evaluate_band_unique_source_stage_subject(
        _source_stage_frame(collinear_target_band=True),
        band="alpha",
        config=load_study2_config(),
    )

    assert qc.eligible is False
    assert "contribution design" in qc.reason


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

    qc, status = evaluate_source_stage_cohort(frame, config=config)

    assert qc["eligible"].tolist() == [True, True]
    assert status.band == "combined"
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

    qc, status = evaluate_source_stage_cohort(frame, config=config)

    assert qc["eligible"].tolist() == [True, False]
    assert status.confirmatory_eligible is False
    assert status.feasibility_eligible is False
    assert status.n_source_valid_subjects == 1
    assert "fewer than 2" in status.reason


def test_evaluate_source_stage_cohort_marks_feasibility_limited_tier() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_stage import evaluate_source_stage_cohort

    config = load_study2_config(
        "studies/pain_study/study2/config/study2_smoketest.yaml"
    )
    config["study2"]["source_stage"]["min_source_valid_subjects"] = 3
    config["study2"]["source_stage"]["min_feasibility_subjects"] = 2
    frame = pd.concat(
        [
            _source_stage_frame().assign(subject_id="sub-0001"),
            _source_stage_frame().assign(subject_id="sub-0002"),
        ],
        ignore_index=True,
    )

    qc, status = evaluate_source_stage_cohort(frame, config=config)

    assert qc["eligible"].tolist() == [True, True]
    assert status.confirmatory_eligible is False
    assert status.feasibility_eligible is True
    assert status.n_source_valid_subjects == 2
    assert "fewer than 3 confirmatory source-valid subjects" in status.reason


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
            _source_stage_frame(collinear_combined_score=True).assign(subject_id="sub-0003"),
        ],
        ignore_index=True,
    )

    qc, status = evaluate_source_stage_cohort(frame, config=config)

    assert qc["eligible"].tolist() == [True, True, False]
    assert status.confirmatory_eligible is False
    assert status.n_source_valid_subjects == 2
    assert status.collinearity_failure_fraction == pytest.approx(1 / 3)
    assert "collinearity failure fraction" in status.reason


def test_evaluate_source_stage_cohort_collinearity_fraction_uses_otherwise_valid_subjects() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_stage import evaluate_source_stage_cohort

    config = load_study2_config(
        "studies/pain_study/study2/config/study2_smoketest.yaml"
    )
    frame = pd.concat(
        [
            _source_stage_frame().assign(subject_id="sub-0001"),
            _source_stage_frame().assign(subject_id="sub-0002"),
            _source_stage_frame(collinear_combined_score=True).assign(subject_id="sub-0003"),
            _source_stage_frame(n_blocks=1, trials_per_block=7).assign(subject_id="sub-0004"),
            _source_stage_frame(n_blocks=1, trials_per_block=7).assign(subject_id="sub-0005"),
        ],
        ignore_index=True,
    )

    qc, status = evaluate_source_stage_cohort(frame, config=config)

    assert qc["eligible"].tolist() == [True, True, False, False, False]
    assert status.confirmatory_eligible is False
    assert status.n_source_valid_subjects == 2
    assert status.collinearity_failure_fraction == pytest.approx(1 / 3)
    assert "collinearity failure fraction" in status.reason


def test_evaluate_band_unique_source_stage_cohort_counts_adjacent_band_failures() -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.source_stage import (
        evaluate_band_unique_source_stage_cohort,
    )

    config = load_study2_config(
        "studies/pain_study/study2/config/study2_smoketest.yaml"
    )
    frame = pd.concat(
        [
            _source_stage_frame().assign(subject_id="sub-0001"),
            _source_stage_frame().assign(subject_id="sub-0002"),
            _source_stage_frame(collinear_adjacent_band=True).assign(subject_id="sub-0003"),
        ],
        ignore_index=True,
    )

    qc, status = evaluate_band_unique_source_stage_cohort(frame, band="alpha", config=config)

    assert qc["eligible"].tolist() == [True, True, False]
    assert status.band == "alpha"
    assert status.confirmatory_eligible is False
    assert status.collinearity_failure_fraction == pytest.approx(1 / 3)
    assert "collinearity failure fraction" in status.reason
