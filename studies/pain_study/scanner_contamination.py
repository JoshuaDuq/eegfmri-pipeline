"""Scanner-contamination frequency definitions for pain-study EEG analyses."""

from __future__ import annotations

SCANNER_HARMONIC_EXCLUSION_RANGES_HZ = (
    (38.0, 43.0),
    (56.0, 67.0),
    (77.0, 85.0),
)

SCANNER_CLEAN_GAMMA_RANGES_HZ = {
    "gamma_low_clean": (30.1, 38.0),
    "gamma_mid_clean": (43.0, 56.0),
    "gamma_high_clean": (67.0, 77.0),
}

SCANNER_CLEAN_GAMMA_BANDS = tuple(SCANNER_CLEAN_GAMMA_RANGES_HZ)

SCANNER_CLEAN_BETA_RANGES_HZ = {
    "beta_low_clean": (13.0, 17.9),
    "beta_high_clean": (23.1, 30.0),
}

SCANNER_CLEAN_BETA_BANDS = tuple(SCANNER_CLEAN_BETA_RANGES_HZ)
