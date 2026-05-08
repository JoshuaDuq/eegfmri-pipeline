"""Circular-shift permutation helpers shared by Study 1 and Study 2."""

from __future__ import annotations

import numpy as np


def admissible_circular_shifts(
    trial_indices: np.ndarray,
    *,
    min_retained_trials: int = 8,
    min_original_distance: int = 5,
    original_block_length: int = 11,
    min_admissible_shifts: int = 4,
) -> tuple[int, ...]:
    retained = np.asarray(trial_indices, dtype=int)
    if retained.size < min_retained_trials:
        return tuple()

    shifts: list[int] = []
    for shift in range(1, retained.size):
        source_trials = np.roll(retained, shift)
        forward_distance = (retained - source_trials) % int(original_block_length)
        if np.all(forward_distance >= int(min_original_distance)):
            shifts.append(int(shift))

    if len(shifts) < min_admissible_shifts:
        return tuple()
    return tuple(shifts)


__all__ = ["admissible_circular_shifts"]
