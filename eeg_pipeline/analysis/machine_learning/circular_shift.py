"""Circular-shift permutation helpers shared by Study 1 and Study 2."""

from __future__ import annotations

import numpy as np

#: A run needs at least this many retained trials before a within-run circular shift
#: says anything. Below it the cycle is too short for the shifted series to be
#: meaningfully different from the observed one.
MIN_RETAINED_TRIALS = 8


def circular_shift_group(n_retained: int) -> tuple[int, ...]:
    """Return every within-run circular shift, the identity included.

    The upper-tail permutation p-value is justified by the transformations forming a
    group under composition, with the observed statistic as the identity element. The
    full cycle is that group; any subset of it generally is not. Selecting, say, the
    shifts of largest minimum displacement leaves ``{5, 6}`` for an intact 11-trial run,
    and composing shift 5 with itself gives the excluded shift 10 -- so the ordinary
    upper-tail calculation loses its randomization argument, and its calibration with it.
    """
    count = int(n_retained)
    if count <= 0:
        return tuple()
    return tuple(range(count))


def is_permutation_valid_run(
    trial_indices: np.ndarray,
    *,
    min_retained_trials: int = MIN_RETAINED_TRIALS,
) -> bool:
    """Whether a run retained enough trials to carry a circular shift at all."""
    retained = np.asarray(trial_indices, dtype=int)
    return bool(retained.size >= int(min_retained_trials) and retained.size >= 2)


__all__ = [
    "MIN_RETAINED_TRIALS",
    "circular_shift_group",
    "is_permutation_valid_run",
]
