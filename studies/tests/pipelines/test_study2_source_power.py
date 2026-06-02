from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import hilbert as scipy_hilbert


def test_sloreta_hilbert_logratio_power_extracts_trial_vertex_matrix() -> None:
    from studies.pain_study.study2.source_power import (
        compute_sloreta_hilbert_logratio_power,
    )

    times = np.asarray([-1.0, -0.5, 0.0, 0.5], dtype=float)
    stcs = [
        SimpleNamespace(
            data=np.asarray(
                [
                    [1.0, 1.0, 2.0, 2.0],
                    [2.0, 2.0, 1.0, 1.0],
                ],
                dtype=float,
            )
        ),
        SimpleNamespace(
            data=np.asarray(
                [
                    [1.0, 1.0, 4.0, 4.0],
                    [4.0, 4.0, 1.0, 1.0],
                ],
                dtype=float,
            )
        ),
    ]

    result = compute_sloreta_hilbert_logratio_power(
        stcs=stcs,
        times=times,
        baseline_window_s=(-1.0, -0.5),
        active_window_s=(0.0, 0.5),
        epsilon=1.0e-12,
    )

    assert result.power_logratio.shape == (2, 2)
    assert result.n_trials == 2
    assert result.n_vertices == 2
    assert result.power_logratio[0, 0] > 0.0
    assert result.power_logratio[0, 1] < 0.0
    assert result.power_logratio[1, 0] > result.power_logratio[0, 0]


def test_sloreta_hilbert_logratio_power_processes_vertex_chunks(monkeypatch) -> None:
    from studies.pain_study.study2 import source_power

    times = np.linspace(-1.0, 1.0, 8)
    stcs = [
        SimpleNamespace(data=np.tile(np.linspace(1.0, 2.0, times.size), (5, 1))),
        SimpleNamespace(data=np.tile(np.linspace(2.0, 1.0, times.size), (5, 1))),
    ]
    hilbert_shapes: list[tuple[int, ...]] = []

    def recording_hilbert(data, *, axis):
        hilbert_shapes.append(tuple(data.shape))
        return scipy_hilbert(data, axis=axis)

    monkeypatch.setattr(source_power, "hilbert", recording_hilbert)

    result = source_power.compute_sloreta_hilbert_logratio_power(
        stcs=stcs,
        times=times,
        baseline_window_s=(-1.0, -0.5),
        active_window_s=(0.5, 1.0),
        chunk_size_vertices=2,
    )

    assert result.power_logratio.shape == (2, 5)
    assert len(hilbert_shapes) == 3
    assert all(shape[0] == 2 for shape in hilbert_shapes)
    assert all(shape[1] <= 2 for shape in hilbert_shapes)
    assert all(shape[2] == times.size for shape in hilbert_shapes)


def test_sloreta_hilbert_logratio_power_rejects_missing_window_samples() -> None:
    from studies.pain_study.study2.source_power import (
        compute_sloreta_hilbert_logratio_power,
    )

    with pytest.raises(ValueError, match="baseline"):
        compute_sloreta_hilbert_logratio_power(
            stcs=[SimpleNamespace(data=np.ones((2, 4), dtype=float))],
            times=np.asarray([0.0, 0.1, 0.2, 0.3], dtype=float),
            baseline_window_s=(-1.0, -0.5),
            active_window_s=(0.0, 0.2),
        )


def test_compute_sloreta_source_estimates_requires_noise_covariance() -> None:
    from studies.pain_study.study2.source_power import compute_sloreta_source_estimates

    with pytest.raises(ValueError, match="noise_cov"):
        compute_sloreta_source_estimates(
            epochs=object(),
            forward=object(),
            noise_cov=None,
            snr=3.0,
            loose=0.2,
            depth=0.8,
        )
