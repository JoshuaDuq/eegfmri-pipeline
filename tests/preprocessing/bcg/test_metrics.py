import numpy as np

from eeg_pipeline.preprocessing.bcg.metrics import (
    ReductionResult,
    epoch_stack,
    held_out_reduction,
    naive_peak_to_peak,
    rlocked_reduction,
)

SFREQ = 1000.0
WINDOW = (-0.3, 0.7)


def _beats(n=200, rr=0.9, jitter=0.02, seed=0, start=5.0):
    rng = np.random.default_rng(seed)
    intervals = rr + rng.normal(0, jitter, n)
    return start + np.cumsum(intervals)


def _noise(n_channels, duration_s, seed=0, sd=10.0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, sd, (n_channels, int(duration_s * SFREQ)))


def _inject(data, beats, sfreq, amplitude_uv):
    out = data.copy()
    t = np.arange(int(0.4 * sfreq)) / sfreq
    shape = amplitude_uv * np.sin(2 * np.pi * 5.0 * t) * np.exp(-t / 0.1)
    for beat in beats:
        start = int(round(beat * sfreq))
        stop = start + shape.size
        if stop <= out.shape[1]:
            out[:, start:stop] += shape
    return out


def test_epoch_stack_shape_and_mean_removal():
    data = _noise(4, 60)
    onsets = np.round(_beats(50) * SFREQ).astype(int)

    stack = epoch_stack(data, onsets, SFREQ, WINDOW)

    assert stack.shape[0] == 4
    assert stack.shape[2] == int(round((WINDOW[1] - WINDOW[0]) * SFREQ))
    assert np.allclose(stack.mean(axis=2), 0.0, atol=1e-9)


def test_held_out_reduction_is_near_zero_without_artifact():
    data = _noise(8, 240, seed=1)
    onsets = np.round(_beats(250, seed=2) * SFREQ).astype(int)

    reduction, _ = held_out_reduction(data, onsets, SFREQ, WINDOW)

    assert np.nanmax(reduction) < 0.02


def test_held_out_reduction_recovers_injected_artifact():
    # 60 uV injected puts the measured reduction at ~0.44, the scale the cohort's own
    # unmarked beats show (37.3%). At 30 uV the same estimator reads 0.16 -- correct, but
    # too close to the clean-data ceiling of 0.02 to be a discriminating assertion.
    beats = _beats(250, seed=3)
    clean = _noise(8, 240, seed=4)
    contaminated = _inject(clean, beats, SFREQ, amplitude_uv=60.0)
    onsets = np.round(beats * SFREQ).astype(int)

    reduction, template_pp = held_out_reduction(contaminated, onsets, SFREQ, WINDOW)

    assert np.nanmax(reduction) > 0.20
    assert np.nanmax(template_pp) > 15.0


def test_rlocked_reduction_reports_nothing_above_null_on_clean_data():
    data = _noise(8, 240, seed=5)
    beats = _beats(250, seed=6)

    result = rlocked_reduction(data, beats, SFREQ, window=WINDOW, n_surrogate=15, seed=0)

    assert isinstance(result, ReductionResult)
    assert result.channels_above_null == 0


def test_rlocked_reduction_flags_every_channel_when_artifact_present():
    beats = _beats(250, seed=7)
    data = _inject(_noise(8, 240, seed=8), beats, SFREQ, amplitude_uv=60.0)

    result = rlocked_reduction(data, beats, SFREQ, window=WINDOW, n_surrogate=15, seed=0)

    assert result.channels_above_null == 8
    assert result.max_value > 0.20


def test_naive_peak_to_peak_fails_where_held_out_statistic_does_not():
    """The naive statistic must stay in the codebase only as a documented failure.

    On artifact-free data it returns several microvolts, which is what made an earlier
    reported BCG amplitude meaningless.
    """
    data = _noise(32, 240, seed=9)
    onsets = np.round(_beats(250, seed=10) * SFREQ).astype(int)

    naive = naive_peak_to_peak(data, onsets, SFREQ, WINDOW, measure=(0.0, 0.6))
    reduction, _ = held_out_reduction(data, onsets, SFREQ, WINDOW)

    assert np.nanmax(naive) > 1.0
    assert np.nanmax(reduction) < 0.02
