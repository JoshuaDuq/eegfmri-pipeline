"""A band handed to the wide notch must not also be chased by the sinusoid removal.

``config.yaml`` declares ``notch_bands`` for contamination that is a *cluster* rather than
a set of resolvable lines -- 56.8-57.7 Hz at this site, where 2 mHz resolution shows 75-118
distinct non-stationary peaks. Sinusoid subtraction cannot clear a cluster: it takes the
summit it aimed at and the neighbour beside it becomes the new summit.

Left targeting that band, the removal deadlocks the workflow. The residual criterion sees
the surviving peak and refuses ``apply``; the notch stage that would have removed the band
outright reads what ``apply`` wrote, so it can never run. Measured on sub-0008: every
recording failed at p=0.0244, the floor with 40 controls, each with its worst residual
between 57.15 and 57.35 Hz, while sub-0000 -- a tenth of the 57 Hz burden -- passed at
p=1.0 with its worst residual nowhere near the band.

So the two stages have to divide the spectrum between them, exactly as ``exclude_mains``
already divides it with the downstream FIR notch.
"""

from __future__ import annotations

import numpy as np
import pytest

from studies.pain_study.analysis.line_comb import removal as lr
from studies.pain_study.scripts.line_comb import remove as rlc

NOTCH_BAND = (56.8, 57.7)


class _Config:
    """The two keys ``RemovalSettings.from_config`` reads, without a YAML on disk."""

    def __init__(self, **data):
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)


def _estimate(*, isolated_hz=()) -> lr.CombEstimate:
    return lr.CombEstimate(
        fundamental_hz=1.2,
        harmonics_used=tuple(range(24, 80)),
        harmonic_positions_hz=tuple(1.2 * harmonic for harmonic in range(24, 80)),
        residual_rms_hz=0.05,
        max_abs_residual_hz=0.10,
        fundamental_jackknife_se_hz=8e-5,
        isolated_hz=tuple(isolated_hz),
        isolated_prominence_db=tuple(15.0 for _ in isolated_hz),
    )


def _in_band(frequencies) -> list[float]:
    return [f for f in frequencies if NOTCH_BAND[0] <= f <= NOTCH_BAND[1]]


class TestSettings:
    def test_the_declared_notch_bands_reach_the_removal_settings(self):
        """`notch_bands` is a top-level key; the removal never read it, which is the bug."""
        settings = rlc.RemovalSettings.from_config(
            _Config(notch_bands=[list(NOTCH_BAND)], dataset={"task": "rest"})
        )

        assert settings.excluded_bands_hz == (NOTCH_BAND,)

    def test_protected_bands_cover_both_the_notch_bands_and_mains(self):
        """Each band belongs to a different downstream stage; both are off limits here."""
        settings = rlc.RemovalSettings(
            excluded_bands_hz=(NOTCH_BAND,),
            mains_notch_hz=(59.5, 60.5),
            exclude_mains=True,
        )

        assert settings.protected_bands_hz == (NOTCH_BAND, (59.5, 60.5))

    def test_mains_stays_out_of_the_protected_bands_when_this_pass_owns_it(self):
        settings = rlc.RemovalSettings(excluded_bands_hz=(NOTCH_BAND,), exclude_mains=False)

        assert settings.protected_bands_hz == (NOTCH_BAND,)

    def test_a_descending_notch_band_is_refused(self):
        with pytest.raises(ValueError, match="increasing"):
            rlc.RemovalSettings(excluded_bands_hz=((57.7, 56.8),))

    def test_a_notch_band_that_is_not_a_pair_of_edges_is_named_in_the_error(self):
        """`notch_bands: [57.0]` is the plausible typo; unpacking it raises unhelpfully."""
        with pytest.raises(ValueError, match="notch_bands"):
            rlc.RemovalSettings.from_config(_Config(notch_bands=[57.0]))

    def test_the_fingerprint_changes_when_a_band_is_handed_over(self):
        """A benchmark taken while 57 Hz was a target cannot certify one where it is not."""
        targeted = rlc.settings_fingerprint(rlc.RemovalSettings())
        handed_over = rlc.settings_fingerprint(
            rlc.RemovalSettings(excluded_bands_hz=(NOTCH_BAND,))
        )

        assert targeted != handed_over


class TestPlanTargets:
    @staticmethod
    def _plan(settings, *, isolated_hz=(), narrow_hz=()):
        """One plan over the two overlapping windows the adaptive model requires."""
        model = lr.build_adaptive_comb_model(
            _estimate(isolated_hz=isolated_hz),
            (_estimate(isolated_hz=isolated_hz), _estimate(isolated_hz=isolated_hz)),
        )
        return rlc.build_removal_plan(
            model,
            bounds=((0, 100), (50, 150)),
            narrow_targets_hz=(tuple(narrow_hz), tuple(narrow_hz)),
            settings=settings,
        )

    def test_a_line_the_estimate_carries_inside_a_notch_band_is_not_a_target(self):
        settings = rlc.RemovalSettings(excluded_bands_hz=(NOTCH_BAND,))

        plan = self._plan(settings, isolated_hz=(57.25,))

        assert _in_band(plan.windows[0].targets_hz) == []

    def test_a_comb_adjacent_narrow_target_inside_a_notch_band_is_not_a_target(self):
        """Narrow targets arrive beside the comb model rather than through it."""
        settings = rlc.RemovalSettings(excluded_bands_hz=(NOTCH_BAND,))

        plan = self._plan(settings, narrow_hz=(57.3,))

        assert _in_band(plan.windows[0].targets_hz) == []
        assert _in_band(plan.windows[0].narrow_targets_hz) == []

    def test_the_same_line_is_a_target_when_no_band_is_handed_over(self):
        """Proves these tests see the exclusion rather than some other rule dropping 57 Hz."""
        plan = self._plan(rlc.RemovalSettings(), isolated_hz=(57.25,))

        assert _in_band(plan.windows[0].targets_hz) != []

    def test_a_line_outside_the_notch_band_is_still_removed(self):
        """The exclusion has to be the band, not the neighbourhood of the band."""
        settings = rlc.RemovalSettings(excluded_bands_hz=(NOTCH_BAND,))

        plan = self._plan(settings, isolated_hz=(58.4,))

        assert any(abs(target - 58.4) < 0.01 for target in plan.windows[0].targets_hz)


class TestDetection:
    """The detector must not spend evidence on a band another stage owns."""

    def _spectrum(self, peaks, df=0.002):
        freqs = np.arange(1.0, 100.0, df)
        spectrum = np.zeros_like(freqs)
        sigma = 0.109 / 2.355
        for centre, height in peaks:
            spectrum[:] = np.maximum(
                spectrum, height * np.exp(-0.5 * ((freqs - centre) / sigma) ** 2)
            )
        return freqs, spectrum, spectrum.copy()

    def test_a_strong_line_inside_an_excluded_band_is_not_detected(self):
        freqs, spectrum, prominence = self._spectrum([(57.25, 28.0)])

        found = lr.detect_isolated_lines(
            freqs,
            spectrum,
            prominence,
            fundamental_hz=1.2,
            harmonic_range=(22, 83),
            min_prominence_db=6.0,
            excluded_bands_hz=(NOTCH_BAND,),
        )

        assert found == ()

    def test_the_same_line_is_detected_when_no_band_is_excluded(self):
        """Proves the exclusion is what suppressed it, not the detector's own rules."""
        freqs, spectrum, prominence = self._spectrum([(57.25, 28.0)])

        found = lr.detect_isolated_lines(
            freqs,
            spectrum,
            prominence,
            fundamental_hz=1.2,
            harmonic_range=(22, 83),
            min_prominence_db=6.0,
        )

        assert len(found) == 1
        assert found[0] == pytest.approx(57.25, abs=0.01)


class TestShippedConfiguration:
    def test_the_shipped_config_hands_its_notch_bands_to_the_notch_stage(self):
        """The two files are coupled and nothing else checks the coupling."""
        from studies.pain_study.scripts.line_comb import notch as rn
        from studies.pain_study.scripts.workflow_config import load_workflow_config

        config = load_workflow_config("line_comb")
        settings = rlc.RemovalSettings.from_config(config)
        bands = rn.NotchSettings.from_config(config).bands

        for band in bands:
            assert (band.low_hz, band.high_hz) in settings.protected_bands_hz, (
                f"{band} is notched wholesale by `line-comb notch` but is still a removal "
                "target; the residual it leaves refuses the apply that notch depends on"
            )
