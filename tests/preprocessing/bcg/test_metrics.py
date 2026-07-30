import numpy as np

from eeg_pipeline.preprocessing.bcg.metrics import epoch_stack, held_out_reduction

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
