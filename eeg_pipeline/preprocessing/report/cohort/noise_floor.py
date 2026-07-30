"""The volume-locked average and the noise floor it has to clear.

Averaging N epochs suppresses everything not locked to the marker by sqrt(N), so the
residual amplitude a subject report prints carries a floor of sigma / sqrt(N). A
participant scanned for four hundred volumes reports a smaller residual than one scanned
for a hundred even when the correction worked equally well on both, and a cohort figure
built on the raw number would rank participants by session length.

The floor is estimated from an odd-even split. Writing s for a stationary locked waveform
and n for independent, equal-variance noise:

    A = mean over all epochs           -> s          + noise, power sigma^2 / N
    D = mean(odd) - mean(even)         -> s cancels  + noise, power 4 sigma^2 / N

The locked waveform cancels in D under those assumptions. When ``N`` is odd, the split
uses ``N_p = N - 1`` epochs and its power is rescaled from ``sigma^2 / N_p`` to the
``sigma^2 / N`` floor of the full average. In expectation,

    mean(s^2) = mean(A^2) - (N_p / N) mean(D^2) / 4

is an unbiased expression for locked power. A single estimate can be negative, which
means the locked signal was unresolved at this averaging floor; it must not be clipped
into evidence of an exact zero.

Alternating rather than splitting at the midpoint, for the same reason the evoked
reliability does: slow drift in impedance or arousal then falls equally on both halves
instead of being counted as noise.

Pure arithmetic on arrays, so the estimator is testable against data with a known answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class LockedAverage:
    """One stage's volume-locked average, with its own noise floor measured.

    ``average`` is the waveform the subject panel draws and is large. Callers reduce it
    and drop it; only the scalars are worth keeping.
    """

    #: ``(n_channels, n_times)`` volume-locked average, in the input's units.
    #:
    #: Excluded from comparison. A dataclass that compares a numpy array raises on ``==``
    #: rather than answering it, which turns an ordinary equality check anywhere
    #: downstream into an exception. The scalars beside it identify the measurement
    #: completely, since they are derived from this array.
    average: np.ndarray = field(compare=False)
    #: Across-channel, across-latency RMS of that average, in microvolts. What the subject
    #: report prints, and what a cohort must not pool.
    locked_rms_uv: float
    #: The floor at this epoch count, in microvolts, measured from the odd-even split.
    noise_floor_uv: float
    #: Signed locked power after subtracting the estimated floor, in microvolts squared.
    #: Negative means unresolved, not a negative physical power.
    excess_power_uv2: float
    n_epochs: int
    #: Epochs entering the odd-even split. One fewer than ``n_epochs`` when that is odd, so
    #: the two halves stay equal and the algebra above stays exact.
    n_paired_epochs: int

    @property
    def is_resolved(self) -> bool:
        """Whether locked power exceeded the estimated averaging floor."""
        return self.excess_power_uv2 > 0.0

    @property
    def resolved_amplitude_uv(self) -> float | None:
        """Floor-adjusted amplitude, or ``None`` where it did not resolve."""
        if not self.is_resolved:
            return None
        return float(np.sqrt(self.excess_power_uv2))

    @property
    def detectability(self) -> float:
        """How far the locked average sits above its own floor.

        Worth reporting per participant and never pooled: for a fixed artifact it grows as
        sqrt(N), so a cohort distribution of it would describe session lengths. It answers
        "was this detectable", not "how large is it".
        """
        if self.noise_floor_uv <= 0.0:
            return float("inf")
        return self.locked_rms_uv / self.noise_floor_uv


def measure_locked_average(epoch_data: np.ndarray) -> LockedAverage:
    """Average epochs time-locked to a marker and measure the floor of that average.

    ``epoch_data`` is ``(n_epochs, n_channels, n_times)`` in volts, with each epoch's own
    mean already removed. Returns microvolts.

    The average is returned rather than recomputed by the caller, because the caller needs
    it for the waveform it draws and a second pass over a few hundred megabytes to
    re-derive something already computed is exactly the kind of cost this module exists to
    avoid.
    """
    data = np.asarray(epoch_data, dtype=float)
    if data.ndim != 3:
        raise ValueError("The locked average needs an epochs-by-channels-by-times array.")
    n_epochs = int(data.shape[0])
    if n_epochs < 2:
        raise ValueError(
            f"Measuring the noise floor needs at least two epochs; {n_epochs} were given."
        )

    average = data.mean(axis=0)
    locked_power = float(np.mean(average**2))

    paired = 2 * (n_epochs // 2)
    difference = data[0:paired:2].mean(axis=0) - data[1:paired:2].mean(axis=0)
    paired_floor_power = float(np.mean(difference**2)) / 4.0
    floor_power = paired_floor_power * paired / n_epochs

    return LockedAverage(
        average=average,
        locked_rms_uv=float(np.sqrt(locked_power) * 1e6),
        noise_floor_uv=float(np.sqrt(floor_power) * 1e6),
        excess_power_uv2=float((locked_power - floor_power) * 1e12),
        n_epochs=n_epochs,
        n_paired_epochs=paired,
    )


__all__ = ["LockedAverage", "measure_locked_average"]
