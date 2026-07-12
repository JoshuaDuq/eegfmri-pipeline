from __future__ import annotations

import pandas as pd

from studies.pain_study.study2.config import load_study2_config


def _behavior_frame(*, constant_second_subject_rating: bool = False) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for subject_index, subject_id in enumerate(("sub-0001", "sub-0002")):
        for block in range(1, 4):
            for trial in range(1, 5):
                nuisance = float(block)
                signal = float(trial + subject_index)
                rating = 5.0 if constant_second_subject_rating and subject_index == 1 else signal
                rows.append(
                    {
                        "subject_id": subject_id,
                        "run": block,
                        "expression": signal + 0.1 * nuisance,
                        "rating": rating + 0.2 * nuisance,
                        "nuisance": nuisance,
                    }
                )
    return pd.DataFrame(rows)


def test_behavioral_convergence_computes_subject_slopes_and_permutation_p_value() -> None:
    from studies.pain_study.study2.behavioral_convergence import (
        compute_behavioral_convergence,
    )

    result = compute_behavioral_convergence(
        _behavior_frame(),
        expression_column="expression",
        rating_column="rating",
        design_columns=("nuisance",),
        config=load_study2_config("studies/pain_study/study2/config/study2_smoketest.yaml"),
        n_permutations=20,
        random_state=7,
    )

    assert result.n_subjects == 2
    assert result.mean_beta > 0.95
    assert 0.0 < result.p_value <= 1.0
    assert result.subject_results["behavioral_convergence_criteria_met"].tolist() == [
        True,
        True,
    ]


def test_behavioral_convergence_excludes_zero_variance_rating_subjects() -> None:
    from studies.pain_study.study2.behavioral_convergence import (
        compute_behavioral_convergence,
    )

    result = compute_behavioral_convergence(
        _behavior_frame(constant_second_subject_rating=True),
        expression_column="expression",
        rating_column="rating",
        design_columns=("nuisance",),
        config=load_study2_config("studies/pain_study/study2/config/study2_smoketest.yaml"),
        n_permutations=5,
        random_state=7,
    )

    assert result.n_subjects == 1
    assert result.subject_results["behavioral_convergence_criteria_met"].tolist() == [
        True,
        False,
    ]
    assert result.subject_results.loc[1, "unmet_criteria"] == "zero_residual_variance_rating"


def test_behavioral_convergence_counts_only_shiftable_runs() -> None:
    from studies.pain_study.study2.behavioral_convergence import _unmet_subject_criteria

    frame = pd.DataFrame(
        {
            "run": [1, 2, 3, 3],
            "expression": [1.0, 2.0, 3.0, 4.0],
            "rating": [1.0, 2.0, 4.0, 3.0],
            "nuisance": [0.0, 1.0, 0.0, 1.0],
        }
    )
    unmet = _unmet_subject_criteria(
        frame,
        run_column="run",
        expression_column="expression",
        rating_column="rating",
        design_columns=("nuisance",),
        min_trials=4,
        min_runs=2,
    )

    assert unmet == ("min_valid_runs",)
