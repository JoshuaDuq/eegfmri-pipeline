"""Measurements for event-locked artifact, and the nulls that make them interpretable.

Every statistic here is held out or controlled. Raw event-locked amplitude is not a
measurement of artifact: averaging 500 epochs of ordinary EEG produces several microvolts
of peak-to-peak by itself, and on this cohort a naive peak-to-peak read 5.71 uV against
its own null of 6.99 uV.

No function returns a verdict. Thresholds belong to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch


def epoch_stack(
    data_uv: np.ndarray,
    onsets_samples: np.ndarray,
    sfreq: float,
    window: tuple[float, float],
) -> np.ndarray:
    """Mean-removed epochs shaped (n_channels, n_epochs, n_times).

    Epochs running off either end of the recording are dropped rather than padded.
    """
    pre = int(round(window[0] * sfreq))
    length = int(round((window[1] - window[0]) * sfreq))
    starts = np.asarray(onsets_samples, dtype=int) + pre
    keep = (starts >= 0) & (starts + length <= data_uv.shape[1])
    starts = starts[keep]
    if starts.size == 0:
        return np.empty((data_uv.shape[0], 0, length))
    index = starts[:, None] + np.arange(length)[None, :]
    epochs = data_uv[:, index]
    return epochs - epochs.mean(axis=2, keepdims=True)


def held_out_reduction(
    data_uv: np.ndarray,
    onsets_samples: np.ndarray,
    sfreq: float,
    window: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Variance removed from held-out epochs by an event-locked template.

    The template is built from even-indexed epochs and scored on odd-indexed ones, so
    averaging noise cannot inflate the result: with no event-locked structure the
    expectation is approximately zero.

    Returns (per-channel reduction fraction, per-channel template peak-to-peak in uV).
    """
    epochs = epoch_stack(data_uv, onsets_samples, sfreq, window)
    n_channels = data_uv.shape[0]
    if epochs.shape[1] < 4:
        nan = np.full(n_channels, np.nan)
        return nan, nan

    template = epochs[:, 0::2, :].mean(axis=1, keepdims=True)
    test = epochs[:, 1::2, :]
    before = test.var(axis=(1, 2))
    after = (test - template).var(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        reduction = 1.0 - (after / before)
    flat = template[:, 0, :]
    return reduction, flat.max(axis=1) - flat.min(axis=1)


@dataclass(frozen=True)
class ReductionResult:
    per_channel: np.ndarray
    template_pp_uv: np.ndarray
    null_max: float
    max_value: float
    max_channel: int
    channels_above_null: int
    n_epochs: int
    n_surrogate: int


def circular_shift_null(
    data_uv: np.ndarray,
    onset_seconds: np.ndarray,
    sfreq: float,
    window: tuple[float, float],
    n_surrogate: int,
    seed: int,
) -> np.ndarray:
    """Maximum held-out reduction under random circular shifts of the whole event train.

    Shifting preserves the inter-event structure exactly and changes only the train's
    alignment to the data, which is what isolates phase-locking from mere periodicity.
    """
    rng = np.random.default_rng(seed)
    duration = data_uv.shape[1] / sfreq
    out = np.empty(n_surrogate)
    for index in range(n_surrogate):
        shifted = np.sort((np.asarray(onset_seconds) + rng.uniform(2.0, duration - 2.0)) % duration)
        reduction, _ = held_out_reduction(
            data_uv, np.round(shifted * sfreq).astype(int), sfreq, window
        )
        out[index] = np.nanmax(reduction)
    return out


def rlocked_reduction(
    data_uv: np.ndarray,
    onset_seconds: np.ndarray,
    sfreq: float,
    *,
    window: tuple[float, float] = (-0.3, 0.7),
    n_surrogate: int = 20,
    seed: int = 0,
) -> ReductionResult:
    """Held-out event-locked reduction with its circular-shift null."""
    onsets = np.asarray(onset_seconds, dtype=float)
    samples = np.round(onsets * sfreq).astype(int)
    reduction, template_pp = held_out_reduction(data_uv, samples, sfreq, window)
    null = circular_shift_null(data_uv, onsets, sfreq, window, n_surrogate, seed)
    null_max = float(np.max(null)) if null.size else float("nan")
    finite = np.nan_to_num(reduction, nan=-np.inf)
    return ReductionResult(
        per_channel=reduction,
        template_pp_uv=template_pp,
        null_max=null_max,
        max_value=float(np.nanmax(reduction)),
        max_channel=int(np.argmax(finite)),
        channels_above_null=int(np.sum(finite > null_max)),
        n_epochs=int(epoch_stack(data_uv, samples, sfreq, window).shape[1]),
        n_surrogate=int(n_surrogate),
    )


def naive_peak_to_peak(
    data_uv: np.ndarray,
    onsets_samples: np.ndarray,
    sfreq: float,
    window: tuple[float, float],
    measure: tuple[float, float],
) -> np.ndarray:
    """Peak-to-peak of the event-locked average. Retained only as a documented failure.

    This is not a measurement of artifact. It is kept so the regression test can assert it
    still misbehaves, and so nobody reintroduces it believing it is safe.
    """
    epochs = epoch_stack(data_uv, onsets_samples, sfreq, window)
    if epochs.shape[1] == 0:
        return np.full(data_uv.shape[0], np.nan)
    evoked = epochs.mean(axis=1)
    lo = int(round((measure[0] - window[0]) * sfreq))
    hi = int(round((measure[1] - window[0]) * sfreq))
    segment = evoked[:, lo:hi]
    return segment.max(axis=1) - segment.min(axis=1)


def band_power(
    data_uv: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    picks: np.ndarray | None = None,
) -> float:
    """Mean power in a band, averaged over the selected channels."""
    selected = data_uv if picks is None else data_uv[picks]
    nperseg = min(int(4 * sfreq), selected.shape[1])
    freqs, power = welch(selected, fs=sfreq, nperseg=nperseg, axis=1)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return float(np.mean(power[:, mask]))


def band_retention(
    before_uv: np.ndarray,
    after_uv: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    picks: np.ndarray | None = None,
) -> float:
    """Fraction of band power surviving a correction.

    Not a preservation measure on its own -- the artifact contributes power inside the
    band, so a value below 1 is expected even for a perfect correction. Pair it with the
    same quantity computed under a sham correction.
    """
    baseline = band_power(before_uv, sfreq, band, picks)
    if baseline == 0.0:
        return float("nan")
    return band_power(after_uv, sfreq, band, picks) / baseline


def lock_ratio(
    signal_uv: np.ndarray,
    onset_seconds: np.ndarray,
    sfreq: float,
    window: tuple[float, float],
) -> float:
    """Event-locked average peak-to-peak over mean single-trial SD, for one channel.

    Used on the ECG channel to judge whether a marker set sits on the QRS complex rather
    than on a T-wave or on noise.
    """
    samples = np.round(np.asarray(onset_seconds) * sfreq).astype(int)
    stack = epoch_stack(signal_uv[None, :], samples, sfreq, window)
    if stack.shape[1] == 0:
        return float("nan")
    evoked = stack.mean(axis=1)[0]
    single = float(np.sqrt(stack[0].var(axis=1)).mean())
    if single == 0.0:
        return float("nan")
    return float((evoked.max() - evoked.min()) / single)


@dataclass(frozen=True)
class EvokedPreservation:
    correlation: np.ndarray
    amplitude_ratio: np.ndarray


def evoked_preservation(
    before_uv: np.ndarray,
    after_uv: np.ndarray,
    onset_seconds: np.ndarray,
    sfreq: float,
    window: tuple[float, float],
) -> EvokedPreservation:
    """Per-channel survival of the stimulus-locked average.

    Stimulus events are not cardiac-locked, so the ballistocardiogram averages out of this
    waveform. What remains is brain response, which a correction must not distort.
    """
    samples = np.round(np.asarray(onset_seconds) * sfreq).astype(int)
    left = epoch_stack(before_uv, samples, sfreq, window).mean(axis=1)
    right = epoch_stack(after_uv, samples, sfreq, window).mean(axis=1)

    correlation = np.empty(left.shape[0])
    ratio = np.empty(left.shape[0])
    for channel in range(left.shape[0]):
        a, b = left[channel], right[channel]
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        correlation[channel] = float(np.dot(a, b) / denominator) if denominator else np.nan
        span = a.max() - a.min()
        ratio[channel] = float((b.max() - b.min()) / span) if span else np.nan
    return EvokedPreservation(correlation=correlation, amplitude_ratio=ratio)
