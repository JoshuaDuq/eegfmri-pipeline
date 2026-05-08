from __future__ import annotations

import numpy as np
import pytest


def test_compute_source_power_association_map_returns_partial_correlations() -> None:
    from studies.pain_study.study2.association import compute_source_power_association_map

    nuisance = np.linspace(-1.0, 1.0, 40)
    score_signal = np.sin(np.linspace(0.0, 4.0 * np.pi, 40))
    score = score_signal + 0.5 * nuisance
    source_power = np.column_stack(
        [
            2.0 * score_signal + 0.25 * nuisance,
            -1.5 * score_signal + 0.25 * nuisance,
            np.ones(40, dtype=float),
        ]
    )

    result = compute_source_power_association_map(
        source_power=source_power,
        score=score,
        design=nuisance.reshape(-1, 1),
    )

    assert result.partial_r.shape == (3,)
    assert result.fisher_z.shape == (3,)
    assert result.valid_vertices.tolist() == [True, True, False]
    assert result.partial_r[0] > 0.99
    assert result.partial_r[1] < -0.99
    assert np.isnan(result.partial_r[2])
    assert np.isfinite(result.fisher_z[:2]).all()
    assert np.isnan(result.fisher_z[2])


def test_compute_source_power_association_map_rejects_zero_variance_score() -> None:
    from studies.pain_study.study2.association import compute_source_power_association_map

    with pytest.raises(ValueError, match="score has zero variance"):
        compute_source_power_association_map(
            source_power=np.ones((4, 2), dtype=float),
            score=np.ones(4, dtype=float),
            design=np.arange(4, dtype=float).reshape(-1, 1),
        )


def test_compute_source_power_association_map_validates_shapes() -> None:
    from studies.pain_study.study2.association import compute_source_power_association_map

    with pytest.raises(ValueError, match="same number of trials"):
        compute_source_power_association_map(
            source_power=np.ones((4, 2), dtype=float),
            score=np.ones(3, dtype=float),
            design=np.ones((4, 1), dtype=float),
        )

