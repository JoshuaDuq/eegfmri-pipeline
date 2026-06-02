from __future__ import annotations

import numpy as np
import pytest


def test_target_retrained_source_permutations_resample_invalid_draws() -> None:
    from studies.pain_study.study2.target_permutations import (
        InvalidPermutationDraw,
        run_target_retrained_source_permutations,
    )

    calls = {"permuter": 0}

    def permute_target(rng: np.random.Generator, attempt_index: int) -> np.ndarray:
        calls["permuter"] += 1
        if attempt_index == 1:
            raise InvalidPermutationDraw("zero-variance held-out score")
        return np.asarray([attempt_index, attempt_index + 1], dtype=float)

    def fit_predict_score(permuted_target: np.ndarray) -> np.ndarray:
        return permuted_target + 10.0

    def compute_source_maps(score: np.ndarray) -> np.ndarray:
        return np.asarray([score[0], score[1], np.mean(score)], dtype=float)

    result = run_target_retrained_source_permutations(
        n_valid_draws=2,
        max_invalid_fraction=0.50,
        random_state=3,
        permute_target=permute_target,
        fit_predict_score=fit_predict_score,
        compute_source_maps=compute_source_maps,
    )

    assert result.n_valid_draws == 2
    assert result.n_attempted_draws == 3
    assert result.n_invalid_draws == 1
    assert result.null_source_maps.shape == (2, 3)
    assert calls["permuter"] == 3


def test_target_retrained_source_permutations_raise_when_budget_exhausted() -> None:
    from studies.pain_study.study2.target_permutations import (
        InvalidPermutationDraw,
        run_target_retrained_source_permutations,
    )

    def invalid_permuter(
        rng: np.random.Generator,
        attempt_index: int,
    ) -> np.ndarray:
        raise InvalidPermutationDraw(f"invalid draw {attempt_index}")

    with pytest.raises(RuntimeError, match="requested valid draw count"):
        run_target_retrained_source_permutations(
            n_valid_draws=2,
            max_invalid_fraction=0.50,
            random_state=3,
            permute_target=invalid_permuter,
            fit_predict_score=lambda target: target,
            compute_source_maps=lambda score: score,
        )


def test_target_retrained_source_permutations_allow_flat_source_maps() -> None:
    from studies.pain_study.study2.target_permutations import (
        run_target_retrained_source_permutations,
    )

    result = run_target_retrained_source_permutations(
        n_valid_draws=1,
        max_invalid_fraction=0.0,
        random_state=3,
        permute_target=lambda rng, attempt_index: np.asarray([1.0, 2.0], dtype=float),
        fit_predict_score=lambda target: target + 1.0,
        compute_source_maps=lambda score: np.zeros(3, dtype=float),
    )

    assert result.null_source_maps.tolist() == [[0.0, 0.0, 0.0]]
