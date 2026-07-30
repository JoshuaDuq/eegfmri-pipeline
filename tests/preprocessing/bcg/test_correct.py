import numpy as np
import pytest

from eeg_pipeline.preprocessing.bcg.correct import (
    _correct_obs,
    _epoch_bounds,
    correct_beats,
    substitute_stretches,
)

SFREQ = 1000.0


def _beats(n=120, rr=0.9, start=5.0):
    return start + np.arange(n) * rr


def _inject(data, beats, sfreq, amplitude_uv):
    out = data.copy()
    t = np.arange(int(0.4 * sfreq)) / sfreq
    shape = amplitude_uv * np.sin(2 * np.pi * 5.0 * t) * np.exp(-t / 0.1)
    for beat in beats:
        start = int(round(beat * sfreq))
        if start + shape.size <= out.shape[1]:
            out[:, start : start + shape.size] += shape
    return out


@pytest.mark.parametrize("method", ["obs", "aas"])
def test_correction_reduces_injected_artifact(method):
    rng = np.random.default_rng(0)
    beats = _beats()
    clean = rng.normal(0, 10.0, (4, int(130 * SFREQ)))
    dirty = _inject(clean, beats, SFREQ, amplitude_uv=60.0)

    corrected = correct_beats(dirty, beats, SFREQ, method=method)

    before = np.var(dirty - clean)
    after = np.var(corrected - clean)
    assert after < before


@pytest.mark.parametrize("method", ["obs", "aas"])
def test_correction_is_confined_to_the_supplied_beats(method):
    rng = np.random.default_rng(1)
    data = rng.normal(0, 10.0, (3, int(120 * SFREQ)))
    beats = np.array([20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0])

    corrected = correct_beats(data, beats, SFREQ, method=method, window=(-0.3, 0.7))

    untouched = slice(int(60 * SFREQ), int(100 * SFREQ))
    assert np.array_equal(corrected[:, untouched], data[:, untouched])


def test_mne_pca_obs_is_not_confined_on_its_own():
    """Documented limitation of `mne.preprocessing.apply_pca_obs`, measured not assumed.

    MNE modifies the whole recording rather than the neighbourhoods of `qrs_times`, in two
    ways: it demeans every channel over its full length, and it reaches 200 ms past the
    leading epoch edge. `correct_beats` therefore splices rather than trusting MNE, which
    is what `test_correction_is_confined_to_the_supplied_beats` then holds it to.
    """
    rng = np.random.default_rng(1)
    data = rng.normal(0, 10.0, (3, int(120 * SFREQ)))
    beats = np.array([20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0])

    raw_mne = _correct_obs(data, beats, SFREQ, 4, None)

    starts, length = _epoch_bounds(beats, SFREQ, (-0.3, 0.7), data.shape[1])
    inside = np.zeros(data.shape[1], dtype=bool)
    inside[(starts[:, None] + np.arange(length)[None, :]).ravel()] = True

    difference = raw_mne - data
    assert np.all(difference[:, ~inside] != 0.0)  # every out-of-epoch sample moves

    # Almost all of that movement is one whole-channel demean.
    demean = -data.mean(axis=1)
    beyond_demean = np.abs(difference[:, ~inside] - demean[:, None]).max(axis=0) > 1e-9
    assert beyond_demean.sum() == 201

    # The rest is a 200 ms reach past the leading epoch edge, worth microvolts.
    outside_times = np.flatnonzero(~inside)[beyond_demean] / SFREQ
    assert outside_times.min() == pytest.approx(beats.min() - 0.501, abs=1e-3)
    assert outside_times.max() == pytest.approx(beats.min() - 0.301, abs=1e-3)
    assert np.abs(difference[:, ~inside]).max() > 1.0


def test_substitute_stretches_replaces_only_named_windows():
    base = np.zeros((2, 10000))
    replacement = np.ones((2, 10000))

    out = substitute_stretches(base, replacement, [(2.0, 4.0)], SFREQ)

    assert np.all(out[:, 2000:4000] == 1.0)
    assert np.all(out[:, :2000] == 0.0)
    assert np.all(out[:, 4000:] == 0.0)


@pytest.mark.parametrize("n_beats", [0, 1, 2, 4])
def test_obs_states_its_epoch_requirement_instead_of_failing_inside_mne(n_beats):
    """Too few epochs for the basis must be a stated precondition, not an MNE crash.

    Measured on MNE 1.12.1: one beat raises `cannot convert float NaN to integer`, two
    raise `SVD did not converge`, and three "succeed" on a basis of three epochs. A real
    run hit this -- sub0009 run 5 recovers exactly one beat -- and it surfaced as an
    unreadable error row in the cohort report.
    """
    rng = np.random.default_rng(3)
    data = rng.normal(0, 10.0, (4, int(60 * SFREQ)))
    beats = 20.0 + np.arange(n_beats) * 1.0

    with pytest.raises(ValueError, match="needs at least 5 beats"):
        correct_beats(data, beats, SFREQ, method="obs", n_components=4)


def test_aas_has_no_such_floor():
    """AAS averages neighbours, so it degrades rather than failing on a thin beat set."""
    rng = np.random.default_rng(4)
    data = rng.normal(0, 10.0, (4, int(60 * SFREQ)))

    corrected = correct_beats(data, np.array([20.0]), SFREQ, method="aas")

    assert np.all(np.isfinite(corrected))


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown method"):
        correct_beats(np.zeros((2, 5000)), np.array([1.0, 2.0]), SFREQ, method="nope")
