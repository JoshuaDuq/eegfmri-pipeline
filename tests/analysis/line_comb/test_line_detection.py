"""Detecting the isolated lines per participant instead of listing them cohort-wide.

A cohort-wide list is wrong for somebody by construction: measured on the uncleaned root,
these lines scatter 0.19 Hz to 0.595 Hz between participants, while one seed window reaches
0.30 Hz. The 94 Hz line was caught in 11 of 15 participants and left standing at +20 to
+28 dB in the rest, and in sub-0008 the delivered data was worse after cleaning than
before, because its neighbours went and it did not.

What the detector must not do is remove signal. Three things protect that, and each has a
test here: it stays clear of the comb, whose harmonics are the comb pass's business; it
stays clear of the benchmark probes, which exist to prove signal survives; and it takes
only narrow peaks, because a scanner line is a sinusoid at the width of the spectral
resolution while alpha and beta rhythms are whole hertz wide.
"""

from __future__ import annotations

import numpy as np
import pytest

from studies.pain_study.analysis.line_comb import removal as lr


def _spectrum(peaks=(), *, rhythms=(), f0=1.2, harmonics=(), df=0.002, high=100.0):
    """A flat-background spectrum in dB with narrow lines and/or broad rhythms added.

    ``peaks`` and ``harmonics`` are (frequency, prominence_db); ``rhythms`` are
    (frequency, prominence_db, width_hz).
    """
    freqs = np.arange(1.0, high, df)
    spectrum = np.zeros_like(freqs)

    def add(centre, height, width):
        spectrum[:] = np.maximum(
            spectrum, height * np.exp(-0.5 * ((freqs - centre) / width) ** 2)
        )

    line_width = 0.109 / 2.355  # half-power width -> gaussian sigma
    for centre, height in peaks:
        add(centre, height, line_width)
    for harmonic, height in harmonics:
        add(harmonic * f0, height, line_width)
    for centre, height, width in rhythms:
        add(centre, height, width / 2.355)
    return freqs, spectrum, spectrum.copy()


def _detect(freqs, spectrum, prominence, **kwargs):
    options = dict(
        fundamental_hz=1.2,
        harmonic_range=(22, 83),
        min_prominence_db=6.0,
        low_hz=20.0,
        high_hz=100.0,
    )
    options.update(kwargs)
    return lr.detect_isolated_lines(freqs, spectrum, prominence, **options)


def test_a_clean_narrow_line_is_found():
    freqs, spec, prom = _spectrum(peaks=[(94.3453, 28.0)])
    found = _detect(freqs, spec, prom)
    assert len(found) == 1
    assert found[0] == pytest.approx(94.3453, abs=0.01)


def test_the_same_line_is_found_wherever_the_participant_puts_it():
    """The whole point: no seed, so every position in the cohort's span is equally findable.

    These are the measured positions of the 94 Hz line, from sub-0000's 93.9345 Hz to
    sub-0008's 94.3453 Hz. A single seed window reaches 0.30 Hz and this span is 0.41 Hz,
    which is why the listed version missed four participants.
    """
    for position in (93.9345, 94.0163, 94.0910, 94.2141, 94.3453):
        freqs, spec, prom = _spectrum(peaks=[(position, 24.0)])
        found = _detect(freqs, spec, prom)
        assert found and found[0] == pytest.approx(position, abs=0.01), position


def test_the_one_comb_adjacent_member_of_that_span_is_declined():
    """sub-0001's 93.7503 Hz peak is not the same line, and must not be taken as one.

    Measured against the fitted fundamental it sits 0.152 Hz from comb harmonic 78, while
    the other twelve participants carry the line 0.336 to 0.747 Hz clear of any harmonic.
    Inside that distance a peak cannot be told from a harmonic's sideband -- and sidebands
    are the residual a fixed-frequency notch cannot follow, so removing it would claim a
    fix it cannot deliver. Declining it is the conservative reading and the honest one.
    """
    freqs, spec, prom = _spectrum(peaks=[(93.7503, 26.4)], f0=1.19998)
    assert _detect(freqs, spec, prom, fundamental_hz=1.19998) == ()


def test_a_comb_harmonic_is_not_taken_as_an_isolated_line():
    """The comb pass removes those; taking them here would target them twice."""
    freqs, spec, prom = _spectrum(harmonics=[(60, 25.0), (65, 20.0)])
    assert _detect(freqs, spec, prom) == ()


def test_a_line_just_off_the_comb_is_still_rejected_within_the_clearance():
    """Inside the clearance the detector cannot tell a sideband from a separate line."""
    freqs, spec, prom = _spectrum(peaks=[(60 * 1.2 + 0.1, 25.0)])
    assert _detect(freqs, spec, prom, comb_clearance_hz=0.2) == ()


def test_a_benchmark_probe_tone_is_never_taken():
    """The probes prove signal survives; removing them would fake that proof."""
    probe = lr.Probe()
    for tone in probe.sinusoid_hz + (probe.burst_hz,):
        if not 20.0 < tone < 100.0:
            continue
        freqs, spec, prom = _spectrum(peaks=[(tone, 30.0)])
        assert _detect(freqs, spec, prom) == (), tone


def test_a_neural_rhythm_is_not_taken_as_a_line():
    """A 2 Hz-wide beta rhythm is signal, however tall its peak.

    This is the failure that would matter most: a detector that ranks on height alone
    removes the participant's own rhythm and reports it as cleaning.
    """
    freqs, spec, prom = _spectrum(rhythms=[(22.0, 20.0, 2.0)])
    assert _detect(freqs, spec, prom) == ()


def test_a_narrow_line_sitting_on_a_rhythm_is_still_found():
    """Rejecting rhythms must not blind the detector to a line inside one."""
    freqs, spec, prom = _spectrum(peaks=[(23.75, 18.0)], rhythms=[(22.0, 12.0, 2.0)])
    found = _detect(freqs, spec, prom)
    assert found and found[0] == pytest.approx(23.75, abs=0.02)


def test_a_peak_below_the_threshold_is_left_alone():
    freqs, spec, prom = _spectrum(peaks=[(47.04, 3.0)])
    assert _detect(freqs, spec, prom, min_prominence_db=6.0) == ()


def test_one_line_is_reported_once():
    """Adjacent bins of the same peak must not each become a target."""
    freqs, spec, prom = _spectrum(peaks=[(81.11, 15.0)])
    assert len(_detect(freqs, spec, prom)) == 1


def test_detection_is_ordered_and_repeatable():
    freqs, spec, prom = _spectrum(peaks=[(94.3, 20.0), (23.75, 25.0), (47.04, 22.0)])
    first = _detect(freqs, spec, prom)
    assert first == _detect(freqs, spec, prom)
    assert list(first) == sorted(first)


def test_the_budget_keeps_the_strongest_lines():
    """A cap bounds how much spectrum removal can claim; it must spend it on the worst."""
    peaks = [(30.5, 9.0), (47.04, 25.0), (81.11, 12.0), (94.3, 20.0)]
    freqs, spec, prom = _spectrum(peaks=peaks)
    found = _detect(freqs, spec, prom, max_lines=2)
    assert len(found) == 2
    assert set(np.round(found, 1)) == {47.0, 94.3}


def test_nothing_is_found_in_a_spectrum_with_no_lines():
    freqs, spec, prom = _spectrum()
    assert _detect(freqs, spec, prom) == ()


def test_a_comb_position_outside_the_removal_range_is_still_not_a_line():
    """The comb exists at harmonics the removal range does not cover.

    sub-0011 carries a peak at 20.401 Hz, 0.001 Hz from harmonic 17. Harmonic 17 is below
    the removal range, so an earlier version of this detector offered it as an isolated
    line. That is the wrong remedy: if a comb harmonic there should be removed, the comb's
    range is what should say so -- the range is reasoned about deliberately, and harmonic
    11 at 13.23 Hz is left in place on purpose because it lands where real rhythms live.
    Smuggling one in as "isolated" bypasses that reasoning.
    """
    freqs, spec, prom = _spectrum(peaks=[(20.401, 14.1)], f0=1.19998)
    found = _detect(
        freqs, spec, prom, fundamental_hz=1.19998, harmonic_range=(22, 83), low_hz=18.0
    )
    assert found == (), f"harmonic 17 at 20.400 Hz was offered as an isolated line: {found}"


def test_the_prominence_floor_sits_above_the_noise_population():
    """The threshold is calibrated, not chosen, and the calibration is the reason it holds.

    Measured across the fifteen uncleaned run-1 recordings: of 889 peaks clearing 6 dB,
    three quarters sat at or below 8.3 dB with a median of 7.4. Dropping the floor from
    10 dB to 6 dB took the count from 7.2 to 59.3 peaks per participant and added no
    frequency that recurs across the cohort -- it bought noise only.
    """
    assert lr.LINE_PROMINENCE_FLOOR_DB >= 10.0, (
        "below 10 dB the detector picks up a noise population that does not recur across "
        "participants, and removal would then take spectrum at frequencies nothing occupies"
    )


def test_the_cap_leaves_room_for_what_the_cohort_actually_carries():
    """A cap that binds turns detection into 'the strongest N peaks', which is not the same
    measurement. At the calibrated floor this cohort yields 7.2 lines per participant and
    at most 12, so the cap bounds pathology without shaping the ordinary result."""
    assert lr.MAX_ISOLATED_LINES > 12
