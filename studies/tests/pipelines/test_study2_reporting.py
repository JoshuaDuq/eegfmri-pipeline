from __future__ import annotations

import numpy as np
import pytest


def test_bootstrap_mean_interval_is_deterministic_and_contains_observed_mean() -> None:
    from studies.pain_study.study2.reporting import bootstrap_mean_interval

    result = bootstrap_mean_interval(
        np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float),
        n_resamples=200,
        random_state=11,
    )

    assert result.mean == pytest.approx(2.5)
    assert result.ci_low < result.mean < result.ci_high
    assert result.n_resamples == 200


def test_bootstrap_mean_interval_rejects_nonpositive_resample_count() -> None:
    from studies.pain_study.study2.reporting import bootstrap_mean_interval

    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_mean_interval(np.asarray([1.0, 2.0], dtype=float), n_resamples=0)
