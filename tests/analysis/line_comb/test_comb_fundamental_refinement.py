"""Refining a repeated spacing into the fundamental that generates the comb.

Two failures compounded on sub-0008's baseline, and between them ``comb_structure``
reported no comb at all on the *uncleaned* arm -- so ``verify``'s ``n_comb_lines == 0``
half of ``verification_passed`` could not fail.

1. The starting gap is a pair spacing, good to a few mHz. Membership is tested against an
   absolute tolerance, but a spacing error grows as ``k * error``: 6 mHz at harmonic 78 is
   0.47 Hz, eight times the 0.06 Hz tolerance. So a nearly-right spacing explains almost
   nothing, and the subdivision rule then has a base of 2 lines to beat.
2. Subdivision was judged on raw membership. A denser grid catches more lines by chance --
   at 0.06 Hz tolerance a 0.2 Hz grid covers 60% of the axis against 10% for a 1.2 Hz one
   -- so subdividing always looked better. Measured on that recording: 1.199 explained 42
   of 62 narrow lines at 30 mHz, and 0.200 explained 54, of which ~37 are chance.
"""

from __future__ import annotations

import numpy as np
import pytest

from studies.pain_study.analysis.line_comb import diagnosis as hd

TOLERANCE_HZ = 0.06


def _comb(fundamental: float, harmonics) -> np.ndarray:
    return fundamental * np.asarray(list(harmonics), dtype=float)


def _members(frequencies, spacing) -> int:
    return int(np.sum(np.abs(frequencies - np.rint(frequencies / spacing) * spacing) <= TOLERANCE_HZ))


class TestASlightlyWrongStartingGap:
    """The real case: the dominant pair gap is 6 mHz off and reaches harmonic 78."""

    frequencies = _comb(1.2, range(22, 79))

    def test_the_true_fundamental_is_recovered(self):
        spacing, _ = hd.refine_comb_fundamental(
            self.frequencies, 1.206, tolerance_hz=TOLERANCE_HZ
        )

        assert spacing == pytest.approx(1.2, abs=2e-3)

    def test_it_does_not_collapse_to_a_subharmonic(self):
        spacing, _ = hd.refine_comb_fundamental(
            self.frequencies, 1.206, tolerance_hz=TOLERANCE_HZ
        )

        assert spacing > 0.9

    def test_the_recovered_spacing_explains_the_whole_comb(self):
        spacing, members = hd.refine_comb_fundamental(
            self.frequencies, 1.206, tolerance_hz=TOLERANCE_HZ
        )

        assert int(members.sum()) == self.frequencies.size
        assert _members(self.frequencies, spacing) == self.frequencies.size


class TestSubdivisionThatIsReal:
    """The behaviour the divisor sweep exists for must survive."""

    def test_a_genuine_halving_is_still_found(self):
        """Every harmonic of 0.6 is present, so the 1.2 gap is twice the true period."""
        frequencies = _comb(0.6, range(20, 120))

        spacing, members = hd.refine_comb_fundamental(
            frequencies, 1.2, tolerance_hz=TOLERANCE_HZ
        )

        assert spacing == pytest.approx(0.6, abs=2e-3)
        assert int(members.sum()) == frequencies.size

    def test_an_exact_spacing_is_left_alone(self):
        frequencies = _comb(1.2, range(22, 79))

        spacing, _ = hd.refine_comb_fundamental(
            frequencies, 1.2, tolerance_hz=TOLERANCE_HZ
        )

        assert spacing == pytest.approx(1.2, abs=1e-4)


class TestChanceMembership:
    def test_a_dense_subharmonic_is_not_preferred_over_the_real_comb(self):
        """Scattered lines a fine grid would catch by chance must not buy a subdivision."""
        rng = np.random.default_rng(3)
        frequencies = np.sort(
            np.concatenate([_comb(1.2, range(22, 79)), rng.uniform(20.0, 95.0, size=25)])
        )

        spacing, _ = hd.refine_comb_fundamental(
            frequencies, 1.206, tolerance_hz=TOLERANCE_HZ
        )

        assert spacing == pytest.approx(1.2, abs=5e-3)


class TestStructureThatChanceWouldSupply:
    """Searching over spacing can manufacture membership; the excess has to pay for it.

    A grid of spacing ``s`` admits any line within the tolerance of a multiple, so a fine
    enough grid admits everything. Once the spacing is searched rather than taken as given,
    a handful of lines can always be made to look periodic -- and on sub-0008's cleaned
    baseline they were: three surviving 57 Hz cluster peaks were fitted at 0.167 Hz, a
    spacing with no relation to the 1.2 Hz comb, and `verify` failed the recording for it.
    """

    CLUSTER = np.array([57.26166, 57.40741, 57.6])

    def test_three_cluster_peaks_are_not_a_comb(self):
        _, members = hd.refine_comb_fundamental(
            self.CLUSTER, 0.16917, tolerance_hz=TOLERANCE_HZ
        )

        assert int(members.sum()) < 3

    def test_a_real_comb_is_still_a_comb(self):
        """The same rule must not cost the case the refinement exists for."""
        frequencies = _comb(1.2, range(22, 79))

        _, members = hd.refine_comb_fundamental(
            frequencies, 1.206, tolerance_hz=TOLERANCE_HZ
        )

        assert int(members.sum()) == frequencies.size


class TestContract:
    def test_a_non_positive_spacing_is_still_refused(self):
        with pytest.raises(ValueError, match="finite positive"):
            hd.refine_comb_fundamental([1.0, 2.0, 3.0], 0.0, tolerance_hz=TOLERANCE_HZ)

    def test_a_zero_divisor_ceiling_is_still_refused(self):
        with pytest.raises(ValueError, match="max_divisor"):
            hd.refine_comb_fundamental(
                [1.0, 2.0, 3.0], 1.0, tolerance_hz=TOLERANCE_HZ, max_divisor=0
            )
