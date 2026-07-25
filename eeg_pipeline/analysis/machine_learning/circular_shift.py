"""Circular-shift permutation helpers shared by Study 1 and Study 2."""

from __future__ import annotations

import numpy as np


def admissible_circular_shifts(
    trial_indices: np.ndarray,
    *,
    min_retained_trials: int = 8,
    original_block_length: int = 11,
) -> tuple[int, ...]:
    """Return shifts with the largest minimum displacement in original trial space."""
    retained = np.asarray(trial_indices, dtype=int)
    if retained.size < min_retained_trials:
        return tuple()

    shift_distances: list[tuple[int, int]] = []
    for shift in range(1, retained.size):
        source_trials = np.roll(retained, shift)
        forward_distance = (retained - source_trials) % int(original_block_length)
        circular_distance = np.minimum(
            forward_distance,
            int(original_block_length) - forward_distance,
        )
        shift_distances.append((int(shift), int(np.min(circular_distance))))

    maximum_distance = max(distance for _shift, distance in shift_distances)
    return tuple(shift for shift, distance in shift_distances if distance == maximum_distance)


__all__ = ["admissible_circular_shifts"]
