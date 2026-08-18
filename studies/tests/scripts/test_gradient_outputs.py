# The gradient workflow writes tables and figures, not report sections.

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from studies.pain_study.scripts.gradient import tables  # noqa: E402


def test_comb_table_carries_one_row_per_run(tmp_path):
    # Build two CombResidual fixtures with the same helper the moved tests use.
    from studies.tests.analysis.test_gradient_comb import _comb_residual

    frame = tables.comb_frame([_comb_residual("sub-01_run-1"), _comb_residual("sub-01_run-2")])
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 2
    assert {"run", "repetition_time_s", "median_before_excess_db"} <= set(frame.columns)


def test_a_non_positive_excess_is_unresolved_and_never_zero():
    # The odd-even floor declined to resolve an amplitude here. Zero would claim a
    # measurement that was not made.
    from studies.pain_study.analysis.gradient.locked import VolumeLockedAverage
    import numpy as np

    locked = VolumeLockedAverage(
        recording_id="sub-01_run-1",
        times_s=np.linspace(0.0, 0.9, 8),
        before_rms_uv=np.ones(8),
        after_rms_uv=np.ones(8),
        n_volumes=100,
        before_locked_rms_uv=1.0,
        after_locked_rms_uv=1.0,
        before_noise_floor_uv=2.0,
        after_noise_floor_uv=0.5,
        before_excess_power_uv2=-3.0,
        after_excess_power_uv2=0.75,
        before_half_correlation=-0.9,
        after_half_correlation=0.8,
    )

    frame = tables.locked_frame([locked])
    amplitudes = dict(zip(frame["stage"], frame["resolved_amplitude_uv"]))
    assert amplitudes["Before ICA"] == tables.UNRESOLVED
    assert amplitudes["After ICA"] != 0
    assert float(amplitudes["After ICA"]) > 0
