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

    The fit runs twice. Oscillatory peaks are positive deviations from the background,
    so bins whose first-pass residual sits in the upper quartile are dropped and the
    line refitted on the remainder. That removes the alpha bump, and any residual
    narrowband artifact that survived ``excluded_windows``, without having to name in
    advance which peaks a given recording contains.

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
    background = residual <= threshold
    if int(background.sum()) < MINIMUM_FIT_BINS:
        # Refitting on too few bins is less trustworthy than keeping the first pass.
        background = np.ones_like(residual, dtype=bool)

    slope, intercept = np.polyfit(log_frequency[background], values[background], deg=1)
    fitted = slope * log_frequency[background] + intercept
    observed = values[background]
    total_variance = float(np.sum((observed - observed.mean()) ** 2))
    residual_variance = float(np.sum((observed - fitted) ** 2))
    r_squared = 1.0 - residual_variance / total_variance if total_variance > 0 else 0.0

    return AperiodicFit(
        offset_db=float(intercept),
        exponent=float(-slope / 10.0),
        r_squared=float(r_squared),
        fit_range_hz=(float(low), float(high)),
        n_bins_used=int(background.sum()),
        n_bins_available=available,
        # Free: the residual sum of squares is already formed for the coefficient of
        # determination above, and this is only its per-bin root.
        residual_db=float(np.sqrt(residual_variance / max(int(background.sum()), 1))),
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
