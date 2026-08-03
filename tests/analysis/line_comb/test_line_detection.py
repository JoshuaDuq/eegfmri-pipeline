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


def test_a_comb_harmonic_is_not_taken_as_an_isolated_line():
    """The comb pass removes those; taking them here would target them twice."""
    freqs, spec, prom = _spectrum(harmonics=[(60, 25.0), (65, 20.0)])
    assert _detect(freqs, spec, prom) == ()


def test_a_peak_just_off_the_comb_is_judged_by_strength_not_distance():
    """Proximity alone does not decide it; the carrier beside it does.

    This test used to assert that anything inside the clearance was rejected, with no
    harmonic in the spectrum at all -- so it proved only that the distance rule fired, not
    that the peak was a sideband. Under a carrier of equal height the peak is the comb's
    and goes; 17 dB above it, it is its own line and stays.
    """
    near = 60 * 1.2 + 0.12
    beneath = _spectrum(harmonics=[(60, 25.0)], peaks=[(near, 19.0)])
    assert not any(abs(f - near) < 0.05 for f in _detect(*beneath))

    above = _spectrum(harmonics=[(60, 9.0)], peaks=[(near, 26.0)])
    assert any(abs(f - near) < 0.05 for f in _detect(*above))


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


def test_a_line_stronger_than_the_harmonic_beside_it_is_taken():
    """sub-0001 carries a line 0.139 Hz from harmonic 78 that is 17 dB above it.

    A blanket clearance treated "near the comb" as "is comb" and declined it, and the apply
    then made that participant worse than before cleaning: its neighbours went, it did not,
    and it rose from 7.1% to 10.6% of the participant's 62-95 Hz power. But a sideband
    cannot exceed its own carrier, so a peak this far above the local harmonic is not one.
    """
    freqs, spec, prom = _spectrum(
        harmonics=[(78, 9.3)], peaks=[(93.7503, 26.4)], f0=1.20015
    )
    found = _detect(freqs, spec, prom, fundamental_hz=1.20015)
    assert any(abs(f - 93.7503) < 0.02 for f in found), (
        f"a line 17 dB above the neighbouring harmonic was declined as its sideband: {found}"
    )


def test_a_sideband_weaker_than_its_carrier_is_still_declined():
    """The +/-0.11 Hz population is what the clearance is for, and it must stay rejected."""
    freqs, spec, prom = _spectrum(harmonics=[(60, 25.0)], peaks=[(60 * 1.2 + 0.11, 19.0)])
    found = _detect(freqs, spec, prom)
    assert not any(abs(f - (60 * 1.2 + 0.11)) < 0.05 for f in found), found


def test_a_peak_sitting_on_a_harmonic_is_still_declined():
    """sub-0011's 20.401 Hz is 0.001 Hz from harmonic 17: it has no excess over itself."""
    freqs, spec, prom = _spectrum(peaks=[(20.401, 14.1)], f0=1.19998)
    found = _detect(
        freqs, spec, prom, fundamental_hz=1.19998, harmonic_range=(22, 83), low_hz=18.0
    )
    assert found == (), f"a comb harmonic was taken as an isolated line: {found}"


def test_a_peak_within_one_line_width_of_a_harmonic_is_always_declined():
    """Inside a line width the strength comparison measures the candidate against itself.

    sub-0001 has a peak 0.064 Hz from harmonic 39. That is closer than the 0.109 Hz a line
    occupies, so the "harmonic strength" sampled at the comb position is really that peak's
    own skirt, and it outranks itself by construction. Admitting it would hand the comb
    pass's own harmonic back as an isolated line and target it twice.
    """
    f0 = 1.20015
    near = 39 * f0 + 0.064
    freqs, spec, prom = _spectrum(peaks=[(near, 24.0)], f0=f0)
    found = _detect(freqs, spec, prom, fundamental_hz=f0)
    assert not any(abs(f - near) < 0.05 for f in found), (
        f"a peak 0.064 Hz from harmonic 39 was taken as its own line: {found}"
    )


def test_production_detection_is_not_blind_at_the_probe_frequencies():
    """Probes are injected after targets are chosen, so excluding them blinds production.

    benchmark_run computes targets from the raw recording and only then adds the probe
    waveform, so the probe tones are not in the spectrum detection sees and there is
    nothing there to protect. The exclusion bought nothing for the benchmark and cost five
    permanent 0.7 Hz blind spots in delivered data: five 30 dB lines placed on 35.55, 40,
    44.05, 65.35 and 78.45 Hz were all rejected under the production default.

    check_probe_clearance remains the guard that matters, and it is the right one -- if a
    real line ever does sit on a probe tone it raises, which says move the probe rather
    than stop looking.
    """
    probe = lr.Probe()
    tones = probe.sinusoid_hz + (probe.burst_hz,)
    freqs = np.arange(1.0, 100.0, 0.002)
    spectrum = np.zeros_like(freqs)
    sigma = 0.109 / 2.355
    for tone in tones:
        spectrum[:] = np.maximum(spectrum, 30.0 * np.exp(-0.5 * ((freqs - tone) / sigma) ** 2))

    found = _detect(freqs, spectrum, spectrum)
    for tone in tones:
        assert any(abs(f - tone) < 0.05 for f in found), (
            f"a 30 dB line at {tone} Hz was invisible to production detection: {found}"
        )
