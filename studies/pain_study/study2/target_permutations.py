"""Target-retrained source-permutation orchestration for Study 2."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


class InvalidPermutationDraw(RuntimeError):
    """Expected invalid draw, such as zero-variance held-out prediction."""


@dataclass(frozen=True)
class TargetRetrainedSourcePermutationResult:
    null_source_maps: np.ndarray
    n_valid_draws: int
    n_requested_draws: int
    n_attempted_draws: int
    n_invalid_draws: int
    max_invalid_fraction: float


def run_target_retrained_source_permutations(
    *,
    n_valid_draws: int,
    max_invalid_fraction: float,
    random_state: int,
    permute_target: Callable[[np.random.Generator, int], np.ndarray],
    fit_predict_score: Callable[[np.ndarray], np.ndarray],
    compute_source_maps: Callable[[np.ndarray], np.ndarray],
) -> TargetRetrainedSourcePermutationResult:
    """Generate null source maps by permuting targets and refitting the score model."""
    requested = _validate_requested_draws(n_valid_draws)
    invalid_fraction = _validate_invalid_fraction(max_invalid_fraction)
    max_attempts = int(np.ceil(requested / (1.0 - invalid_fraction)))
    rng = np.random.default_rng(random_state)

    valid_maps: list[np.ndarray] = []
    map_shape: tuple[int, ...] | None = None
    attempts = 0
    while len(valid_maps) < requested and attempts < max_attempts:
        attempts += 1
        try:
            permuted_target = _finite_array(
                permute_target(rng, attempts),
                name="permuted target",
                require_variance=True,
            )
            score = _finite_array(
                fit_predict_score(permuted_target),
                name="permuted score",
                require_variance=True,
            )
            source_map = _finite_array(
                compute_source_maps(score),
                name="permuted source map",
                require_variance=False,
            )
        except InvalidPermutationDraw:
            continue

        if map_shape is None:
            map_shape = source_map.shape
        elif source_map.shape != map_shape:
            raise ValueError("Study 2 target-retrained source maps must share one shape.")
        valid_maps.append(source_map)

    if len(valid_maps) < requested:
        raise RuntimeError(
            "Study 2 target-retrained source permutations did not reach the requested "
            f"valid draw count: valid={len(valid_maps)}, requested={requested}, "
            f"attempted={attempts}, max_attempts={max_attempts}."
        )

    return TargetRetrainedSourcePermutationResult(
        null_source_maps=np.stack(valid_maps, axis=0),
        n_valid_draws=len(valid_maps),
        n_requested_draws=requested,
        n_attempted_draws=attempts,
        n_invalid_draws=attempts - len(valid_maps),
        max_invalid_fraction=invalid_fraction,
    )


def _validate_requested_draws(value: int) -> int:
    if isinstance(value, bool):
        raise TypeError("Study 2 target-retrained n_valid_draws must be an integer.")
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError("Study 2 target-retrained n_valid_draws must be an integer.")
    count = int(numeric_value)
    if count <= 0:
        raise ValueError("Study 2 target-retrained n_valid_draws must be positive.")
    return count


def _validate_invalid_fraction(value: float) -> float:
    if isinstance(value, bool):
        raise TypeError("Study 2 target-retrained max_invalid_fraction must be numeric.")
    fraction = float(value)
    if not np.isfinite(fraction) or fraction < 0.0 or fraction >= 1.0:
        raise ValueError("Study 2 target-retrained max_invalid_fraction must be in [0, 1).")
    return fraction


def _finite_array(
    values: np.ndarray,
    *,
    name: str,
    require_variance: bool,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise InvalidPermutationDraw(f"Study 2 target-retrained {name} is empty.")
    if not np.all(np.isfinite(array)):
        raise InvalidPermutationDraw(f"Study 2 target-retrained {name} contains non-finite values.")
    if require_variance and float(np.std(array, ddof=0)) <= 1.0e-12:
        raise InvalidPermutationDraw(f"Study 2 target-retrained {name} has zero variance.")
    return array


__all__ = [
    "InvalidPermutationDraw",
    "TargetRetrainedSourcePermutationResult",
    "run_target_retrained_source_permutations",
]
