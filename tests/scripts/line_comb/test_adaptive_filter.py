"""Adaptive filtering must reconstruct a continuous run without hidden fallbacks."""

from __future__ import annotations

import numpy as np
import pytest

from studies.pain_study.analysis.line_comb import removal as lr
from studies.pain_study.scripts.line_comb import remove as rlc


def _estimate(fundamental_hz: float) -> lr.CombEstimate:
    return lr.CombEstimate(
        fundamental_hz=fundamental_hz,
        harmonics_used=tuple(range(24, 80)),
        harmonic_positions_hz=tuple(fundamental_hz * harmonic for harmonic in range(24, 80)),
        residual_rms_hz=0.01,
        max_abs_residual_hz=0.02,
        fundamental_jackknife_se_hz=1e-4,
        isolated_hz=(),
        isolated_prominence_db=(),
    )


def test_adaptive_windows_cover_the_run_and_anchor_the_tail():
    bounds = rlc.adaptive_window_bounds(
        n_times=1_025,
        window_samples=400,
        hop_samples=200,
    )

    assert bounds[0] == (0, 400)
    assert bounds[-1] == (625, 1_025)
    assert all(
        right_start < left_stop for (_, left_stop), (right_start, _) in zip(bounds, bounds[1:])
    )


def test_adaptive_windows_reject_a_run_shorter_than_one_window():
    with pytest.raises(ValueError, match="fewer than"):
        rlc.adaptive_window_bounds(n_times=399, window_samples=400, hop_samples=200)


def test_spectrum_fit_grids_include_the_longer_tail_window():
    grids = rlc.spectrum_fit_frequency_grids(
        sampling_frequency_hz=1_000.0,
        filter_length="20s",
        window_samples=54_000,
    )

    resolutions = {round(float(freqs[1]), 8) for freqs in grids}
    assert resolutions == {round(1.0 / 20.0, 8), round(1.0 / 24.0, 8)}
    assert all(freqs[-1] == pytest.approx(500.0) for freqs in grids)


def test_squared_sine_weights_form_an_exact_partition_of_unity():
    bounds = rlc.adaptive_window_bounds(
        n_times=1_025,
        window_samples=400,
        hop_samples=200,
    )

    weights = rlc.squared_sine_weights(bounds, n_times=1_025)
    total = np.zeros(1_025)
    for (start, stop), weight in zip(bounds, weights):
        assert weight.shape == (stop - start,)
        assert np.all(weight > 0.0)
        total[start:stop] += weight

    assert np.allclose(total, 1.0, atol=1e-12)


def test_overlap_add_exactly_reconstructs_identity_segments():
    signal = np.vstack(
        [
            np.linspace(-1.0, 1.0, 1_025),
            np.sin(np.arange(1_025) / 19.0),
        ]
    )
    bounds = rlc.adaptive_window_bounds(
        n_times=signal.shape[-1],
        window_samples=400,
        hop_samples=200,
    )
    segments = tuple(signal[:, start:stop].copy() for start, stop in bounds)

    reconstructed = rlc.overlap_add_segments(segments, bounds, signal.shape[-1])

    assert np.allclose(reconstructed, signal, atol=1e-12)


def test_spectrum_fit_parallelism_uses_shared_memory_threads():
    class FakeRaw:
        backend_name = None

        def notch_filter(self, **kwargs):
            from joblib.parallel import get_active_backend

            self.backend_name = type(get_active_backend()[0]).__name__
            assert kwargs["n_jobs"] == 4
            return self

    raw = FakeRaw()
    rlc.clean_raw(
        raw,
        (30.0,),
        filter_length="30s",
        filter_jobs=4,
        mt_bandwidth=0.6,
        notch_widths=np.asarray((0.1,)),
    )

    assert raw.backend_name == "ThreadingBackend"


def test_adaptive_cleaning_filters_each_window_plan_and_preserves_non_eeg(monkeypatch):
    import mne

    info = mne.create_info(["Cz", "STI"], sfreq=10.0, ch_types=["eeg", "stim"])
    data = np.vstack([np.zeros(150), np.arange(150, dtype=float)])
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    bounds = ((0, 100), (50, 150))
    model = lr.build_adaptive_comb_model(
        _estimate(1.2),
        (_estimate(1.2), _estimate(1.21)),
    )
    plan = rlc.build_removal_plan(
        model,
        bounds=bounds,
        narrow_targets_hz=((), ()),
        settings=rlc.RemovalSettings(),
    )

    offsets = iter((1.0, 3.0))

    def fake_clean(segment, targets, **settings):
        assert targets
        segment._data += next(offsets)
        return segment

    monkeypatch.setattr(rlc, "clean_raw", fake_clean)

    cleaned = rlc.clean_continuous_raw(raw.copy(), plan, rlc.RemovalSettings())

    expected_eeg = rlc.overlap_add_segments(
        (np.ones((1, 100)), np.full((1, 100), 3.0)),
        bounds,
        n_times=150,
    )[0]
    assert np.allclose(cleaned.get_data(picks=["Cz"])[0], expected_eeg)
    assert np.array_equal(cleaned.get_data(picks=["STI"])[0], data[1])
