"""Measurements for event-locked artifact, and the nulls that make them interpretable.

Every statistic here is held out or controlled. Raw event-locked amplitude is not a
measurement of artifact: averaging 500 epochs of ordinary EEG produces several microvolts
of peak-to-peak by itself, and on this cohort a naive peak-to-peak read 5.71 uV against
its own null of 6.99 uV.

No function returns a verdict. Thresholds belong to the caller.
"""

from __future__ import annotations

import numpy as np


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
