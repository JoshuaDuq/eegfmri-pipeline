"""Aperiodic (1/f) fit of a sensor power spectrum.

Every other spectral panel in the report shows what cleaning *removed*. The aperiodic
component is the compact readout of the opposite question. Broadband artifact raises
the spectrum roughly uniformly and flattens the slope; removing too many components
takes the broadband background down with it. Reporting the exponent and offset either
side of ICA turns "84% of variance removed" into two numbers a reviewer can weigh.

This is a robust log-log line fit, not spectral parameterisation: no oscillatory peaks
are modelled and no peak parameters are reported. It is deliberately the simpler
estimator, because the quantity of interest here is the background level and tilt, and
a peak model would add fitting failures without changing either.

Nothing here is graded. The fit is a measurement, and what a shift in exponent means
for a given recording is the reviewer's call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

#: Default band for the fit. The lower bound stays above the high-pass transition, and
#: the upper bound stays below the line-noise fundamental so that the notch and its
#: skirts cannot tilt the slope.
DEFAULT_FIT_RANGE_HZ = (2.0, 45.0)

#: Residual quantile above which a bin is treated as an oscillatory peak and dropped
#: before the line is refitted. Peaks are positive deviations from the aperiodic
#: background, so they concentrate in the upper tail of the residuals.
PEAK_RESIDUAL_QUANTILE = 0.75

#: Bins needed before a slope is meaningful at all.
MINIMUM_FIT_BINS = 8


@dataclass(frozen=True)
class AperiodicFit:
    """Robust log-log line through the background of one spectrum."""

    #: Power in dB at 1 Hz, extrapolated from the fitted line.
    offset_db: float
    #: Negated slope in dB per decade divided by 10, i.e. the exponent of ``1/f**k``.
    exponent: float
    r_squared: float
    fit_range_hz: tuple[float, float]
    #: Bins that survived peak removal and defined the final fit.
    n_bins_used: int
    n_bins_available: int
    #: RMS departure of the background bins from the fitted line, in decibels.
    #:
    #: The scale of the spectrum's own roughness, measured after oscillatory peaks were
    #: trimmed, so it describes the background rather than what sits on it. A peak worth
    #: reporting has to clear it: the largest bin in a band is always above the line by
    #: something, and without a scale to compare that something against, a recording with
    #: no rhythm still yields a confident-looking peak frequency.
    residual_db: float = 0.0

    @property
    def slope_db_per_decade(self) -> float:
        """Slope in the units the figure is drawn in."""
        return -10.0 * self.exponent


def _excluded_mask(
    frequencies: np.ndarray,
    windows: Sequence[tuple[float, float]],
) -> np.ndarray:
    mask = np.zeros(frequencies.shape, dtype=bool)
    for low, high in windows:
        if high <= low:
            raise ValueError(f"Exclusion window {(low, high)!r} is empty or reversed.")
        mask |= (frequencies >= low) & (frequencies <= high)
    return mask


def fit_aperiodic(
    frequencies: np.ndarray,
    power_db: np.ndarray,
    *,
    fit_range_hz: tuple[float, float] = DEFAULT_FIT_RANGE_HZ,
    excluded_windows: Sequence[tuple[float, float]] = (),
) -> AperiodicFit | None:
    """Fit the aperiodic background of a spectrum already expressed in decibels.

    The fit runs twice. Bins whose first-pass residual sits in either tail are dropped and
    the line refitted on the remainder, without having to name in advance what a given
    recording contains. The upper tail is oscillatory peaks and narrowband artifact that
    survived ``excluded_windows``; the lower tail is filter stopbands, which are an
    absence of signal rather than a measurement of the background and pull the slope just
    as hard.

    Returns ``None`` when too few bins survive to define a slope, so a short or heavily
    masked run simply has no fit rather than a fabricated one.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    power_db = np.asarray(power_db, dtype=float)
    if frequencies.shape != power_db.shape:
        raise ValueError("Aperiodic fit requires matching frequency and power arrays.")
    if frequencies.ndim != 1:
        raise ValueError("Aperiodic fit requires a one-dimensional spectrum.")

    low, high = fit_range_hz
    if high <= low:
        raise ValueError(f"Aperiodic fit range {fit_range_hz!r} is empty or reversed.")

    in_band = (frequencies >= low) & (frequencies <= high)
    # A zero or negative frequency has no logarithm, and a non-finite power value would
    # propagate silently through the least-squares solve.
    usable = in_band & (frequencies > 0) & np.isfinite(power_db)
    usable &= ~_excluded_mask(frequencies, excluded_windows)
    available = int(usable.sum())
    if available < MINIMUM_FIT_BINS:
        return None

    log_frequency = np.log10(frequencies[usable])
    values = power_db[usable]

    slope, intercept = np.polyfit(log_frequency, values, deg=1)
    residual = values - (slope * log_frequency + intercept)
    threshold = float(np.quantile(residual, PEAK_RESIDUAL_QUANTILE))
    # Trimmed from both tails, not only the upper one.
    #
    # The upper trim removes oscillatory peaks, which is what this fit was written for.
    # Keeping everything below it meant ``residual <= threshold`` retained -- and
    # preferentially so -- bins that sit far *under* the background, and those are not
    # background either: a notch stopband is an absence of signal, not a measurement of
    # it. On data cleaned of line noise upstream, sub-0000 run-5 carries four-bin holes
    # 25 to 31 dB deep at 28 and 38 Hz, inside the 2-45 Hz fit range, and they dragged the
    # slope from -12.0 to -15.8 dB per decade.
    #
    # Symmetric in the quantile already defined rather than a depth in decibels, so the
    # rule stays a statement about the residual distribution and introduces no threshold
    # to tune per recording.
    floor = float(np.quantile(residual, 1.0 - PEAK_RESIDUAL_QUANTILE))
    background = (residual <= threshold) & (residual >= floor)
    if int(background.sum()) < MINIMUM_FIT_BINS:
        # Refitting on too few bins is less trustworthy than keeping the first pass.
        background = np.ones_like(residual, dtype=bool)

    slope, intercept = np.polyfit(log_frequency[background], values[background], deg=1)
    fitted = slope * log_frequency[background] + intercept
    observed = values[background]
    total_variance = float(np.sum((observed - observed.mean()) ** 2))
    residual_variance = float(np.sum((observed - fitted) ** 2))
    r_squared = 1.0 - residual_variance / total_variance if total_variance > 0 else 0.0

    # The scatter is measured over the peak-trimmed set, not the two-sided one that the
    # slope is fitted on. The two want different things from the same residuals: a slope
    # must not be pulled by a hole, while the scatter is the noise a peak has to clear and
    # is understated by any set the low tail has been cut out of. Measured on the
    # symmetric set it fell far enough that a recording with no rhythm at all was credited
    # with a resolvable peak twenty-six times in a hundred instead of six.
    scatter_set = residual <= threshold
    if int(scatter_set.sum()) < MINIMUM_FIT_BINS:
        scatter_set = np.ones_like(residual, dtype=bool)
    scatter_residual = values[scatter_set] - (slope * log_frequency[scatter_set] + intercept)

    return AperiodicFit(
        offset_db=float(intercept),
        exponent=float(-slope / 10.0),
        r_squared=float(r_squared),
        fit_range_hz=(float(low), float(high)),
        n_bins_used=int(background.sum()),
        n_bins_available=available,
        residual_db=float(np.sqrt(np.mean(scatter_residual**2))),
    )


def aperiodic_line_db(fit: AperiodicFit, frequencies: np.ndarray) -> np.ndarray:
    """Evaluate a fitted aperiodic line, for drawing it over the spectrum it came from."""
    frequencies = np.asarray(frequencies, dtype=float)
    if np.any(frequencies <= 0):
        raise ValueError("The aperiodic line is undefined at or below zero frequency.")
    return fit.offset_db + fit.slope_db_per_decade * np.log10(frequencies)


__all__ = [
    "DEFAULT_FIT_RANGE_HZ",
    "MINIMUM_FIT_BINS",
    "PEAK_RESIDUAL_QUANTILE",
    "AperiodicFit",
    "aperiodic_line_db",
    "fit_aperiodic",
]
