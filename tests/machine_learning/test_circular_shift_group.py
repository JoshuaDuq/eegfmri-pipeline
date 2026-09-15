"""Circular-shift permutations have to form a group to justify their p-values."""

from __future__ import annotations

import numpy as np
import pytest

from eeg_pipeline.analysis.machine_learning.circular_shift import (
    circular_shift_group,
    is_permutation_valid_run,
)


@pytest.mark.parametrize("n_retained", [8, 9, 10, 11, 12])
def test_the_shift_set_is_closed_under_composition(n_retained: int) -> None:
    """Composing two admissible shifts must land on an admissible shift.

    An upper-tail permutation p-value is justified by the transformations forming a
    group under composition. A set chosen for maximal displacement is not one: with 11
    trials it holds {5, 6}, and applying shift 5 twice gives the excluded shift 10.
    """
    shifts = set(circular_shift_group(n_retained))

    assert 0 in shifts
    for first in shifts:
        for second in shifts:
            assert (first + second) % n_retained in shifts


def test_the_shift_set_is_the_full_cycle() -> None:
    assert circular_shift_group(11) == tuple(range(11))


@pytest.mark.parametrize("n_retained", [0, 1, 7])
def test_short_runs_carry_no_shift(n_retained: int) -> None:
    assert not is_permutation_valid_run(np.arange(1, n_retained + 1, dtype=int))


@pytest.mark.parametrize("n_retained", [8, 11])
def test_runs_with_enough_retained_trials_are_permutation_valid(n_retained: int) -> None:
    assert is_permutation_valid_run(np.arange(1, n_retained + 1, dtype=int))


def test_sampled_shifts_cover_the_whole_cycle() -> None:
    from eeg_pipeline.analysis.machine_learning.orchestration import (
        _permutation_indices_by_scheme,
    )

    groups = np.array(["sub-0001"] * 11, dtype=object)
    runs = np.ones(11, dtype=float)
    trial_indices = np.arange(1, 12, dtype=float)
    rng = np.random.default_rng(0)

    observed = set()
    for _ in range(400):
        source = _permutation_indices_by_scheme(
            groups,
            runs=runs,
            trial_indices=trial_indices,
            rng=rng,
            scheme="circular_shift_within_run",
        )
        observed.add(int(source[0]))

    assert observed == set(range(11))
