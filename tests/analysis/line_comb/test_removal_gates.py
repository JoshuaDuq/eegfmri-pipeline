"""The preservation gates have to be able to fail.

Two of them could not. ``lines_suppressed`` was named for a maximum and evaluated a
median, so half a run's lines could stand above the threshold and still pass. The transient
gates compared the removal's effect on data+probe against its effect on the probe alone --
quantities that are identical for a linear operator, which spectrum_fit is, so they read
1.0 by construction on any data with any settings.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from studies.pain_study.analysis.line_comb import removal as lr


def _metrics(**overrides):
    base = {
        "median_residual_prominence_db": -5.0,
        "max_residual_prominence_db": -3.0,
        "null_max_95_db": -6.0,
        "residual_excess_db": 3.0,
        "focal_residual_excess_db": -2.0,
        "study_residual_excess_db": -2.0,
        "study_focal_residual_excess_db": -2.0,
        "study_significant_focal_residual_count": 0,
        "max_boundary_discontinuity_ratio": 1.0,
        "median_suppression_db": 20.0,
        "max_probe_deviation_db": 0.0,
        "max_nonline_change_db": 0.0,
        "study_max_probe_deviation_db": 0.0,
        "study_max_nonline_change_db": 0.0,
        "burst_energy_ratio": 1.0,
        "burst_correlation": 1.0,
        "intrinsic_energy_ratio": 0.95,
        "removed_band_fraction": 0.12,
        "base_removed_band_fraction": 0.12,
        "measured_band_attenuated_1db": 0.12,
        "base_band_fraction_bin_size": 1.0 / 1810.0,
        "band_fraction_bin_size": 1.0 / 1810.0,
        "measured_band_bin_size": 1.0 / 1810.0,
    }
    base.update(overrides)
    return base


def test_a_line_its_own_controls_never_reach_is_a_discovery():
    """The 90-run manifest had a worst residual of +13.90 dB while every gate passed.

    That gate compared the excess against a decibel cushion. The decision is now the exact
    probability that a matched control search reaches the observation, so a line standing
    where the controls never go is a discovery whatever its size in decibels.
    """
    controls = np.linspace(-8.0, -3.0, 40)

    p_value = lr.null_exceedance_p_value(13.9, controls)

    assert p_value == pytest.approx(1 / 41)
    assert not lr.residual_randomization_verdict([p_value])["passed"]


def test_a_residual_inside_its_control_spread_is_not_a_discovery():
    controls = np.linspace(-8.0, 4.0, 40)

    p_value = lr.null_exceedance_p_value(-1.0, controls)

    assert p_value > 0.05
    assert lr.residual_randomization_verdict([p_value])["passed"]


def test_one_recording_is_decided_by_its_own_exact_test():
    """Benjamini-Hochberg over a single recording reduces to p <= alpha.

    A lone continuous acquisition has no cohort to borrow strength from, and must still be
    decidable.
    """
    assert not lr.residual_randomization_verdict([0.02])["passed"]
    assert lr.residual_randomization_verdict([0.20])["passed"]


def test_the_cohort_tolerates_the_null_rate_it_creates():
    """About one recording in twenty exceeds by construction; that is not a failure."""
    null_like = np.linspace(0.05, 0.95, 90)

    assert lr.residual_randomization_verdict(null_like)["passed"]


def test_a_clean_run_still_passes():
    assert lr.PreservationGate().passed(_metrics(residual_excess_db=-2.0))


def test_the_transient_gate_reads_the_measurement_that_can_fail():
    """intrinsic_energy_ratio is the one that varies; it was reported but never gated.

    On the delivered benchmark it ranged 0.899 to 0.960 while burst_energy_ratio was
    exactly 1.0 on every row.
    """
    gate = lr.PreservationGate()
    assert not gate.evaluate(_metrics(intrinsic_energy_ratio=0.40))[
        "transient_preserved"
    ], "a transient reduced to 40% of its injected energy passed the transient gate"
    assert gate.evaluate(_metrics(intrinsic_energy_ratio=0.95))["transient_preserved"]


def test_a_residual_displaced_by_a_bin_is_not_missed():
    """Suppression was read at the nearest bin only, so a line that moved slipped through.

    Reproduced: a target at 50 Hz whose centre falls to -10 dB while a residual at
    50.05 Hz still stands at +15 dB was reported as a maximum residual of -10 dB. That is
    the exact shape of the failure this work has been chasing -- removing a line exposes or
    displaces its neighbour -- so measuring only the centre cannot see it.
    """
    freqs = np.arange(45.0, 55.0, 0.01)
    before = np.full_like(freqs, -20.0)
    after = np.full_like(freqs, -20.0)
    before[np.argmin(np.abs(freqs - 50.0))] = 25.0
    after[np.argmin(np.abs(freqs - 50.0))] = -10.0
    after[np.argmin(np.abs(freqs - 50.05))] = 15.0

    result = lr.line_suppression(freqs, before, after, [50.0], widths=[0.2])
    assert (
        result["max_residual_prominence_db"] >= 15.0 - 1e-6
    ), f"the displaced residual was invisible: {result['max_residual_prominence_db']}"


def test_suppression_still_reads_the_centre_when_nothing_moved():
    freqs = np.arange(45.0, 55.0, 0.01)
    before = np.full_like(freqs, -20.0)
    after = np.full_like(freqs, -20.0)
    before[np.argmin(np.abs(freqs - 50.0))] = 25.0
    after[np.argmin(np.abs(freqs - 50.0))] = -12.0

    result = lr.line_suppression(freqs, before, after, [50.0], widths=[0.2])
    assert result["max_residual_prominence_db"] == pytest.approx(-12.0)


def test_a_residual_outside_the_claimed_window_is_still_seen():
    """The window the removal claimed is not where a missed line will be.

    The previous fix searched only the notch's own width. But the failure being chased is a
    target that missed -- the line is then just outside what the notch claimed, which is
    exactly where searching the claimed width cannot look. The search has to cover the
    frequency uncertainty of the estimate, not the footprint of the correction.
    """
    freqs = np.arange(45.0, 55.0, 0.01)
    before = np.full_like(freqs, -20.0)
    after = np.full_like(freqs, -20.0)
    before[np.argmin(np.abs(freqs - 50.0))] = 25.0
    after[np.argmin(np.abs(freqs - 50.0))] = -10.0
    after[np.argmin(np.abs(freqs - 50.12))] = 18.0  # outside a 0.2 Hz notch, missed line

    result = lr.line_suppression(freqs, before, after, [50.0], widths=[0.2])
    assert (
        result["max_residual_prominence_db"] >= 18.0 - 1e-6
    ), f"a missed line 0.12 Hz away was invisible: {result['max_residual_prominence_db']}"


def test_the_search_does_not_reach_a_neighbouring_comb_line():
    """It must not charge one target with the line belonging to the next harmonic."""
    freqs = np.arange(45.0, 55.0, 0.01)
    before = np.full_like(freqs, -20.0)
    after = np.full_like(freqs, -20.0)
    before[np.argmin(np.abs(freqs - 50.0))] = 25.0
    after[np.argmin(np.abs(freqs - 50.0))] = -10.0
    after[np.argmin(np.abs(freqs - 51.2))] = 22.0  # the next harmonic, not this target's

    result = lr.line_suppression(freqs, before, after, [50.0], widths=[0.2])
    assert result["max_residual_prominence_db"] == pytest.approx(-10.0)


def test_the_residual_is_judged_against_a_blind_control():
    """Searching a window round every target has a noise floor; the control measures it.

    A +/-0.15 Hz window at this resolution is about sixteen bins, and there are sixty-odd
    targets, so the largest of a thousand background bins is several dB on noise alone. A
    fixed threshold cannot tell that from a surviving line. Equivalent windows placed away
    from any target measure the same floor, and the residual has to beat it.
    """
    rng = np.random.default_rng(0)
    freqs = np.arange(20.0, 100.0, 0.01)
    before = rng.normal(0.0, 1.0, freqs.size)
    after = rng.normal(0.0, 1.0, freqs.size)
    targets = [k * 1.2 for k in range(20, 80)]

    result = lr.line_suppression(freqs, before, after, targets, widths=[0.2] * len(targets))
    assert "null_max_95_db" in result
    assert result["residual_excess_db"] == pytest.approx(
        result["max_residual_prominence_db"] - result["null_max_95_db"]
    )
    assert result["residual_excess_db"] < 3.0, (
        "pure noise registered as a surviving line: "
        f"{result['residual_excess_db']:.2f} dB over the control"
    )


def test_a_real_survivor_still_beats_the_control():
    rng = np.random.default_rng(0)
    freqs = np.arange(20.0, 100.0, 0.01)
    before = rng.normal(0.0, 1.0, freqs.size)
    after = rng.normal(0.0, 1.0, freqs.size)
    after[np.argmin(np.abs(freqs - 60.0))] = 25.0
    targets = [k * 1.2 for k in range(20, 80)]

    result = lr.line_suppression(freqs, before, after, targets, widths=[0.2] * len(targets))
    assert result["residual_excess_db"] > 15.0


def test_large_focal_maximum_below_its_matched_control_passes() -> None:
    gate = lr.PreservationGate()

    assert gate.passed(
        _metrics(
            residual_excess_db=-1.0,
            focal_residual_excess_db=-1.0,
        )
    )


def test_target_prominence_retains_channel_and_block_axes():
    freqs = np.arange(20.0, 30.0, 0.01)
    spectra = np.zeros((2, 3, freqs.size))
    spectra[1, 2, np.argmin(np.abs(freqs - 25.0))] = 14.0

    prominence = lr.spatiotemporal_target_prominence(
        freqs,
        spectra,
        spectra,
        targets=[25.0],
        widths=[0.2],
        background_half_width_hz=2.0,
    )

    assert prominence.shape == (2, 3, 1)
    assert prominence[1, 2, 0] == pytest.approx(14.0)


def test_target_prominence_uses_the_preclean_background_floor():
    """A cleaner cannot manufacture prominence by lowering its own reference floor."""
    freqs = np.arange(20.0, 30.0, 0.01)
    before = np.zeros((1, freqs.size))
    after = before.copy()
    target = np.argmin(np.abs(freqs - 25.0))
    local = np.abs(freqs - 25.0) <= 2.0
    after[:, local] = -20.0
    after[:, target] = 10.0

    prominence = lr.spatiotemporal_target_prominence(
        freqs,
        before,
        after,
        targets=[25.0],
        widths=[0.2],
        background_half_width_hz=2.0,
    )

    assert prominence[0, 0] == pytest.approx(10.0)


def test_spatiotemporal_search_controls_channel_window_multiplicity():
    rng = np.random.default_rng(18)
    freqs = np.arange(20.0, 100.0, 0.02)
    spectra = rng.normal(size=(8, 4, freqs.size))
    targets = tuple(tuple(k * 1.2 for k in range(20, 80)) for _ in range(4))
    widths = tuple((0.2,) * len(window_targets) for window_targets in targets)

    result = lr.adaptive_spatiotemporal_suppression(
        freqs,
        spectra,
        spectra,
        targets,
        widths,
        background_half_width_hz=4.0,
    )

    assert result["focal_residual_excess_db"] < 3.0


def test_spatiotemporal_search_detects_one_focal_window_survivor():
    rng = np.random.default_rng(18)
    freqs = np.arange(20.0, 100.0, 0.02)
    spectra = rng.normal(size=(8, 4, freqs.size))
    targets = tuple(tuple(k * 1.2 for k in range(20, 80)) for _ in range(4))
    widths = tuple((0.2,) * len(window_targets) for window_targets in targets)
    spectra[6, 2, np.argmin(np.abs(freqs - targets[2][30]))] = 30.0

    result = lr.adaptive_spatiotemporal_suppression(
        freqs,
        spectra,
        spectra,
        targets,
        widths,
        background_half_width_hz=4.0,
    )

    assert result["focal_residual_excess_db"] > 15.0


def test_a_boundary_jump_in_the_correction_fails_the_cohort_criterion():
    """A destroyed seam is caught by the exact cohort null, not a per-run cutoff."""
    evidence = [
        lr.BoundaryDiscontinuityEvidence(5.0, (1.0,) * 40),
        *[lr.BoundaryDiscontinuityEvidence(0.5, (1.0,) * 40) for _ in range(89)],
    ]

    assert not lr.seam_randomization_verdict(evidence)["passed"]
    assert lr.PreservationGate().passed(
        _metrics(residual_excess_db=-2.0, max_boundary_discontinuity_ratio=5.0)
    )


def test_boundary_discontinuity_is_measured_on_the_filter_correction():
    original = np.zeros((2, 1_000))
    cleaned = original.copy()
    cleaned[:, 500:] = 1.0

    ratio = lr.boundary_discontinuity_ratio(original, cleaned, boundaries=[500])

    assert ratio > 10.0


def test_matched_null_uses_repeated_complete_target_sized_searches():
    freqs = np.arange(20.0, 100.0, 0.01)
    after = np.zeros_like(freqs)
    targets = [k * 1.2 for k in range(20, 80)]
    reaches = np.full(len(targets), 0.15)

    maxima = lr._matched_null_maxima(freqs, after, np.asarray(targets), reaches)

    assert len(maxima) >= 20
    assert np.all(maxima == 0.0)


def test_matched_null_placement_scales_to_an_adaptive_spectrum_grid():
    freqs = np.linspace(0.0, 500.0, 27_001)
    targets = np.linspace(26.4, 99.6, 70)
    reaches = np.linspace(0.15, 0.25, targets.size)

    started = time.perf_counter()
    placements = lr._matched_null_centres(
        freqs,
        np.ones(freqs.shape, dtype=bool),
        targets,
        reaches,
        edge_margin_hz=0.5,
    )
    elapsed = time.perf_counter() - started

    assert len(placements) == 40
    assert all(placement.shape == targets.shape for placement in placements)
    assert elapsed < 10.0


def test_adaptive_suppression_controls_the_search_across_all_windows():
    rng = np.random.default_rng(8)
    freqs = np.arange(20.0, 100.0, 0.01)
    before = rng.normal(size=(4, freqs.size))
    after = rng.normal(size=(4, freqs.size))
    targets = tuple(tuple(k * (1.2 + index * 1e-4) for k in range(20, 80)) for index in range(4))
    widths = tuple((0.2,) * len(window_targets) for window_targets in targets)

    result = lr.adaptive_line_suppression(freqs, before, after, targets, widths)

    assert result["residual_excess_db"] < 3.0


def test_adaptive_suppression_detects_a_survivor_in_one_window():
    rng = np.random.default_rng(8)
    freqs = np.arange(20.0, 100.0, 0.01)
    before = rng.normal(size=(4, freqs.size))
    after = rng.normal(size=(4, freqs.size))
    targets = tuple(tuple(k * (1.2 + index * 1e-4) for k in range(20, 80)) for index in range(4))
    widths = tuple((0.2,) * len(window_targets) for window_targets in targets)
    after[2, np.argmin(np.abs(freqs - targets[2][30]))] = 25.0

    result = lr.adaptive_line_suppression(freqs, before, after, targets, widths)

    assert result["residual_excess_db"] > 15.0


def test_adaptive_suppression_does_not_call_a_broad_rhythm_a_residual_line():
    """The removal must not be forced to erase broad neural spectral structure."""
    freqs = np.arange(20.0, 100.0, 0.01)
    before = np.zeros((2, freqs.size))
    after = np.zeros_like(before)
    centre = 57.0
    broad = 8.0 * np.exp(-0.5 * ((freqs - centre) / 0.15) ** 2)
    before[0] += broad
    after[0] += broad
    targets = ((57.0,), (57.0,))
    widths = ((0.12,), (0.12,))

    result = lr.adaptive_line_suppression(freqs, before, after, targets, widths)

    assert result["residual_excess_db"] <= 1.0


def test_the_per_run_gate_does_not_apply_a_multiplicity_invalid_seam_cutoff():
    verdict = lr.PreservationGate().evaluate(_metrics(max_boundary_discontinuity_ratio=1.5))

    assert "adaptive_boundaries_continuous" not in verdict


class TestSeamRandomizationUsesTheMeasuredControls:
    def test_a_single_gross_seam_fails_against_the_exact_cohort_maximum_null(self):
        evidence = [
            lr.BoundaryDiscontinuityEvidence(10.0, (1.0,) * 40),
            *[lr.BoundaryDiscontinuityEvidence(1.0, (1.0,) * 40) for _ in range(5)],
        ]

        verdict = lr.seam_randomization_verdict(evidence)

        assert not verdict["passed"]
        assert verdict["max_p_value"] == pytest.approx(1.0 / 41.0)

    def test_systematic_small_seams_fail_the_exact_exceedance_count_null(self):
        evidence = [lr.BoundaryDiscontinuityEvidence(1.1, (1.0,) * 40) for _ in range(12)]

        verdict = lr.seam_randomization_verdict(evidence)

        assert not verdict["passed"]
        assert verdict["count_p_value"] == pytest.approx(1.0 / 41.0)

    def test_an_ordinary_observed_shift_passes_without_a_fitted_dispersion_parameter(self):
        controls = tuple(float(value) for value in range(1, 41))
        evidence = [lr.BoundaryDiscontinuityEvidence(20.0, controls) for _ in range(12)]

        verdict = lr.seam_randomization_verdict(evidence)

        assert verdict["passed"]
        assert "dispersion" not in verdict
