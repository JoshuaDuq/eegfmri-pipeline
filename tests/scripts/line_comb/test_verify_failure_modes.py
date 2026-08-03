"""An analysis that could not run must not be recorded as a clean result.

``verify_cohort`` caught every ``RuntimeError`` and wrote ``n_lines: 0``. The detector
raises that one type for three different things -- no usable gradient-free window, no
usable background estimate, and no line surviving FDR control. Only the last means clean;
the first two mean the measurement failed. Reported identically, a numerical or data
failure reads as successful cleaning, which is the most dangerous direction for it to fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from studies.pain_study.scripts.line_comb import diagnose as ds


def test_no_lines_found_is_its_own_exception():
    assert issubclass(ds.NoLinesDetected, RuntimeError)


def test_an_unusable_background_is_not_a_clean_result():
    """Every bin non-finite: the analysis cannot run, and must not report zero lines."""
    grid = ds.Grid(
        freqs=np.linspace(20.0, 100.0, 64),
        subject_psd=np.full((3, 64), np.nan),
        subject_prominence=np.full((3, 64), np.nan),
        half_width_bins=8,
    )
    with pytest.raises(RuntimeError) as raised:
        ds.detect_cohort_lines(grid)
    assert not isinstance(raised.value, ds.NoLinesDetected), (
        "an unusable background was raised as 'no lines detected', so verify would "
        "record it as a clean cohort"
    )


def test_a_genuinely_clean_cohort_raises_the_reportable_one():
    """Flat prominence everywhere: nothing survives FDR, which is a measurement."""
    rng = np.random.default_rng(0)
    freqs = np.linspace(20.0, 100.0, 4096)
    flat = rng.normal(0.0, 0.01, (4, freqs.size))
    grid = ds.Grid(
        freqs=freqs,
        subject_psd=np.abs(flat) + 1.0,
        subject_prominence=flat,
        half_width_bins=64,
    )
    with pytest.raises(ds.NoLinesDetected):
        ds.detect_cohort_lines(grid)
