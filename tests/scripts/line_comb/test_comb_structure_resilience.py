"""Recovering no comb structure is a measurement, not a fault.

``comb_structure`` already treats "no repeated spacing to find" that way, and says so.
The least-squares fit underneath it was left unguarded, so a family that finds a spacing
but whose members collapse onto duplicate harmonic indices raised straight through the
caller and killed the stage.

``verify`` is where it lands, and it lands after ``apply`` has written the derivative:
the surviving narrow lines are what a *successful* removal left behind, so the better the
cleaning, the likelier the fit has nothing to sit on. Aborting there reports no result at
all for data already on disk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from studies.pain_study.analysis.line_comb import diagnosis as hd
from studies.pain_study.scripts.line_comb import diagnose as ds


def _lines(frequencies) -> pd.DataFrame:
    return pd.DataFrame(
        {"refined_hz": list(frequencies), "is_narrow": [True] * len(frequencies)}
    )


COLLAPSING = (10.0, 10.01, 11.2, 12.4)
"""Two lines closer to each other than half the spacing the family as a whole shows,
so rounding puts both on harmonic index 0."""


def test_a_family_that_collapses_onto_one_index_does_not_raise():
    result = ds.comb_structure(_lines(COLLAPSING))

    assert isinstance(result, pd.DataFrame)


def test_a_real_comb_survives_a_pair_too_close_to_index():
    """Discarding the family here was wrong, and sub-0008 showed it: 41 members at
    1.20004 Hz were thrown away because two of them sat 19 mHz apart, so `verify`
    reported zero comb lines on *uncleaned* data. Only the free intercept is lost."""
    # Two lines within the tolerance of the SAME harmonic, which is how sub-0008's family
    # collapsed: harmonic 48 sits at 57.6 and both of these are inside 0.06 Hz of it, so
    # both join the family and then round to one index.
    comb = [f for f in 1.2 * np.arange(22, 63) if abs(f - 57.6) > 1e-9]
    result = ds.comb_structure(_lines(comb + [57.55, 57.65]))

    family = result.loc[result.family == "narrow_comb"]
    assert len(family) == 1
    assert float(family.fundamental_hz.iloc[0]) == pytest.approx(1.2, abs=1e-3)


def test_the_unindexable_intercept_is_reported_as_undefined():
    """It is the one quantity the failed fit was carrying; nothing else depends on it."""
    # Two lines within the tolerance of the SAME harmonic, which is how sub-0008's family
    # collapsed: harmonic 48 sits at 57.6 and both of these are inside 0.06 Hz of it, so
    # both join the family and then round to one index.
    comb = [f for f in 1.2 * np.arange(22, 63) if abs(f - 57.6) > 1e-9]
    result = ds.comb_structure(_lines(comb + [57.55, 57.65]))

    family = result.loc[result.family == "narrow_comb"]
    assert np.isnan(float(family.free_intercept_hz.iloc[0]))


def test_a_clean_comb_is_still_recovered():
    """The guard must not swallow the structure wherever there really is one."""
    result = ds.comb_structure(_lines(1.2 * np.arange(24, 32)))

    comb = result.loc[result.family == "narrow_comb"]
    assert len(comb) == 1
    assert float(comb.fundamental_hz.iloc[0]) == pytest.approx(1.2, abs=1e-6)


def test_the_estimator_itself_still_refuses_an_unfittable_family():
    """The tolerance belongs to the caller. A fit that cannot index its own input is
    still an error, or a later caller would read a fabricated comb."""
    with pytest.raises(ValueError, match="distinct comb indices"):
        hd.fit_arithmetic_comb(COLLAPSING)
