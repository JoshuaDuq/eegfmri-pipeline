"""Per-band accounting of what is still in the delivered epochs.

Three costs are reported separately because they need different fixes. Independent lines
are what ``line-comb apply`` can remove. Comb sidebands are residual volume-to-volume
gradient variability in the shoulders of each k/TR harmonic, which a fixed-frequency notch
centred on the nulled comb bin cannot reach. Holes are power *missing* below background
from removal already applied. Pooling them makes every band look equally contaminated and
hides which fix is the one that helps: measured this way, 1-45 Hz carries no removable
lines at all, so the whole of its contamination is cardiac and no line work touches it.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from studies.pain_study.analysis.line_comb import diagnosis as hd

WIDE_HALF_WIDTH_HZ = 100.0 / 21.6
NARROW_HALF_WIDTH_HZ = 1.0
SCALE_SWITCH_HZ = 13.0
SIDEBAND_HALF_WIDTH_HZ = 0.20
HOLE_THRESHOLD_DB = 1.0


def adaptive_background_db(
    spectrum_db: np.ndarray,
    freqs: np.ndarray,
    *,
    wide_hz: float = WIDE_HALF_WIDTH_HZ,
    narrow_hz: float = NARROW_HALF_WIDTH_HZ,
    switch_hz: float = SCALE_SWITCH_HZ,
) -> np.ndarray:
    """Running-median background that also reaches delta.

    The production 4.63 Hz half-width returns NaN within 4.63 Hz of DC, so it cannot see
    below about 5 Hz at all -- the estimator trap that once read the 1/f slope as a 4 dB
    line below 3 Hz. Below ``switch_hz`` the half-width drops to 1.0 Hz, still fifteen
    times a line's width, and a running median over a monotone background returns the
    centre value exactly, so the 1/f slope is not mistaken for structure.
    """
    values = np.asarray(spectrum_db, dtype=float)
    frequency = np.asarray(freqs, dtype=float)
    if values.shape != frequency.shape:
        raise ValueError("spectrum_db and freqs must have the same shape.")
    if values.ndim != 1:
        raise ValueError("spectrum_db must be one-dimensional.")

    bin_width = float(frequency[1] - frequency[0])
    wide = hd.local_background_db(values, half_width_bins=int(round(wide_hz / bin_width)))
    narrow = hd.local_background_db(values, half_width_bins=int(round(narrow_hz / bin_width)))
    return np.where(frequency < switch_hz, narrow, wide)


def _mask_near(freqs: np.ndarray, centres: Iterable[float], half_width_hz: float) -> np.ndarray:
    mask = np.zeros(freqs.size, dtype=bool)
    for centre in centres:
        mask |= np.abs(freqs - centre) <= half_width_hz
    return mask


def band_costs(
    freqs: Sequence[float],
    psd: Sequence[float],
    *,
    low_hz: float,
    high_hz: float,
    independent_hz: Iterable[float],
    comb_hz: Iterable[float],
    sideband_hz: float = SIDEBAND_HALF_WIDTH_HZ,
) -> dict[str, float]:
    """Independent-line excess, comb-sideband excess and hole loss, each as a percentage.

    Excess is clipped at zero so a bin the removal has dug below its surroundings cannot
    count as negative artifact; that deficit is the business of ``hole_pct`` instead.
    """
    frequency = np.asarray(freqs, dtype=float)
    spectrum = np.asarray(psd, dtype=float)
    if frequency.shape != spectrum.shape:
        raise ValueError("freqs and psd must have the same shape.")
    if low_hz >= high_hz:
        raise ValueError("low_hz must be below high_hz.")

    bin_width = float(frequency[1] - frequency[0])
    spectrum_db = hd.to_db(spectrum)
    background_db = adaptive_background_db(spectrum_db, frequency)
    background = 10.0 ** (background_db / 10.0)

    inside = (frequency >= low_hz) & (frequency <= high_hz) & np.isfinite(background_db)
    if not np.any(inside):
        raise ValueError("The band holds no bin with a usable background.")

    comb_centres = list(comb_hz)
    at_independent = _mask_near(frequency, independent_hz, sideband_hz)
    at_sideband = _mask_near(frequency, comb_centres, sideband_hz)
    # The comb centre is nulled by the volume-average subtraction; charging it as a line
    # would report the correction itself as contamination.
    for centre in comb_centres:
        at_sideband &= ~(np.abs(frequency - centre) <= 1.5 * bin_width)
    at_sideband &= ~at_independent

    total = float(np.sum(spectrum[inside]))
    costs: dict[str, float] = {}
    for name, mask in (("independent_pct", at_independent), ("sideband_pct", at_sideband)):
        selected = inside & mask
        excess = float(np.sum(np.clip(spectrum[selected] - background[selected], 0.0, None)))
        costs[name] = 100.0 * excess / total if total > 0 else 0.0

    deep = inside & ((background_db - spectrum_db) > HOLE_THRESHOLD_DB)
    expected = float(np.sum(background[inside]))
    missing = float(np.sum(background[deep] - spectrum[deep]))
    costs["hole_pct"] = 100.0 * missing / expected if expected > 0 else 0.0
    return costs
