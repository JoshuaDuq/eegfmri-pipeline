"""The line-comb workflow's output must be the root the rest of the pipeline reads.

These two configs are coupled and nothing else checks the coupling. The core config names
the dataset every downstream stage consumes; the line-comb workflow is what produces it.
If the workflow inherits ``bids_root`` from the core config it reads its own output and
writes ``eeg_linecleaned_linecleaned``; if the core config names the uncleaned root instead,
the pipeline silently preprocesses uncleaned data, which is what used to happen whenever
``--bids-root`` was left off the command line.
"""

from __future__ import annotations

import pytest

from eeg_pipeline.utils.config.loader import load_config
from studies.pain_study.analysis.line_comb import removal as lr
from studies.pain_study.scripts.workflow_config import load_workflow_config


def _configs():
    core = load_config(apply_thread_limits=False)
    return core, load_workflow_config("line_comb", core_config=core)


def test_the_pipeline_reads_what_the_line_comb_workflow_writes() -> None:
    core, workflow = _configs()

    consumed = str(core.get("paths.bids_root"))
    produced = str(workflow.path("output_root").resolve())

    assert consumed.rstrip("/").endswith("_linecleaned"), (
        "the core bids_root must name the cleaned copy: leaving it on the raw root is what "
        "made every delivered run depend on --bids-root being typed"
    )
    assert produced == consumed, (
        f"line-comb writes {produced} but the pipeline reads {consumed}; "
        "the cleaning would never reach an analysis"
    )


def test_the_workflow_reads_the_uncleaned_root_not_its_own_output() -> None:
    core, workflow = _configs()

    read = str(workflow.path("bids_root").resolve())
    written = str(workflow.path("output_root").resolve())

    assert read != written, "the workflow would be cleaning its own output"
    assert not read.rstrip("/").endswith("_linecleaned"), (
        "pin paths.bids_root in the line-comb config; inheriting it from the core config "
        "now yields the cleaned root"
    )


#: Measured on the delivered epochs, 14 participants, sub-0008 excluded. Each entry is the
#: frequency, how many participants carry it, and what it is.
#: 61.0353 Hz is absent on purpose: it falls 0.128 Hz from comb harmonic 51, inside
#: isolated_search_hz, so estimate_comb rejects it as a seed that would find the comb. It
#: sits in the unanalysed 58-62 gap and the comb already covers that position.
AUDITED_RESIDUALS = {
    23.7776: "narrow, off both combs, 7/14, beta",
    29.6854: "narrow, off both combs, 9/14, 0.41 Hz below the gamma_low edge",
    81.1111: "gradient harmonic 73 at TR = 0.9 s, 3/14, inside 62-95 Hz",
}


def test_the_audited_residual_lines_are_all_targeted():
    _, workflow = _configs()
    isolated = workflow.get("line_comb_removal.isolated_hz")
    search = float(workflow.get("line_comb_removal.isolated_search_hz"))

    for frequency, why in AUDITED_RESIDUALS.items():
        nearest = min(isolated, key=lambda seed: abs(seed - frequency))
        assert abs(nearest - frequency) <= search, (
            f"{frequency} Hz ({why}) has no seed within the {search} Hz search window; "
            f"nearest is {nearest}"
        )


#: Where the 94 Hz line actually sits, per participant, measured on the uncleaned root
#: (outputs/line_comb_mains/seed_positions.csv). It spans 0.595 Hz, which no single seed
#: can cover: the widest isolated_search_hz the guard permits is 0.224 Hz, set by how close
#: the 23.75 seed sits to comb harmonic 20.
NINETY_FOUR_HZ_SPAN = (93.7503, 94.3453)


def test_the_94_hz_line_has_enough_seeds_to_span_the_cohort():
    """One seed caught 11 of 15 and left the rest at +20.5 to +28.0 dB.

    In sub-0008 the missed line was *worse* in the delivered data than before cleaning,
    +28.0 dB against +31.2 dB, because the neighbours around it were removed and it was
    not. Two seeds are needed because the line drifts further between participants than a
    single window reaches.
    """
    _, workflow = _configs()
    isolated = workflow.get("line_comb_removal.isolated_hz")
    search = float(workflow.get("line_comb_removal.isolated_search_hz"))

    lo, hi = NINETY_FOUR_HZ_SPAN
    covered = [
        seed for seed in isolated if lo - search <= seed <= hi + search
    ]
    assert len(covered) >= 2, (
        f"the 94 Hz line spans {hi - lo:.3f} Hz across the cohort but only {len(covered)} "
        f"seed(s) sit near it; one window reaches {2 * search:.2f} Hz"
    )

    for position, subject in ((lo, "sub-0001"), (hi, "sub-0008")):
        nearest = min(covered, key=lambda seed: abs(seed - position))
        assert abs(nearest - position) <= search, (
            f"{subject} carries the line at {position} Hz; nearest seed {nearest} is "
            f"{abs(nearest - position):.3f} Hz away, outside the {search} Hz window"
        )


def test_no_isolated_seed_collides_with_a_benchmark_probe():
    _, workflow = _configs()
    isolated = workflow.get("line_comb_removal.isolated_hz")
    probe = lr.Probe()

    for frequency in isolated:
        for sinusoid in probe.sinusoid_hz + (probe.burst_hz,):
            assert abs(frequency - sinusoid) > 0.3, (
                f"seed {frequency} Hz sits on probe tone {sinusoid} Hz; the benchmark would "
                "remove the probe by design and report it as signal loss"
            )


def test_the_gradient_harmonics_are_derived_from_a_nine_tenths_second_tr():
    _, workflow = _configs()
    isolated = workflow.get("line_comb_removal.isolated_hz")

    for harmonic in (73, 74):
        expected = harmonic / 0.9
        nearest = min(isolated, key=lambda seed: abs(seed - expected))
        assert nearest == pytest.approx(expected, abs=0.01), (
            f"gradient harmonic {harmonic} should be seeded at {expected:.4f} Hz "
            f"(TR is exactly 0.9 s by the Volume markers), found {nearest}"
        )


def test_exactly_one_stage_removes_mains():
    core, workflow = _configs()

    fir_notch = core.get("preprocessing.notch_freq")
    exclude_mains = bool(workflow.get("line_comb_removal.exclude_mains"))

    # Mains reaches the removal as a comb harmonic, not as an isolated line: the comb is
    # mains-synchronous, so its fundamental is mains/50 and 60 Hz is harmonic 50. All
    # exclude_mains controls is whether removal_frequencies drops it again on the way out.
    low, high = workflow.get("line_comb_removal.removal_harmonic_range")
    mains_harmonic = round(60.0 / float(workflow.get("line_comb_removal.nominal_fundamental_hz")))
    assert low <= mains_harmonic <= high, (
        f"60 Hz is comb harmonic {mains_harmonic}, outside removal_harmonic_range "
        f"[{low}, {high}]; turning exclude_mains off would then remove nothing at mains"
    )
    removal_takes_mains = not exclude_mains

    assert bool(fir_notch) != removal_takes_mains, (
        f"preprocessing.notch_freq={fir_notch!r} and the line-comb pass "
        f"{'takes' if removal_takes_mains else 'does not take'} mains. Exactly one must: "
        "both means a second bite of the spectrum, neither means 60 Hz survives."
    )


def test_the_wide_fir_notch_is_the_one_in_use():
    """Mains is a cluster, not a line, so only a wide notch clears it.

    The spec proposed trading the 0.97 Hz FIR notch for a 0.133 Hz spectrum_fit pass to
    recover 0.84 Hz. Measured on the cleaned BIDS, that trade left 60 Hz standing at
    +8.04 dB median over background, up to +15.38 dB, in 13 of 15 participants, against
    -42 dB in the generation the FIR notch produced.

    The reason is that 60 Hz here is not one line: 59-61 Hz carries 32-38 distinct peaks
    above 3 dB at 0.002 Hz resolution (sub-0008/0012/0001). Every narrow method removes
    the tallest and surfaces the next. Measured against the uncorrected cleaned data on
    sub-0008, as a change in absolute power over 59.5-60.5 Hz:

        FIR notch, 1.0 Hz wide   -56.66 dB     45-58 Hz: -0.00   62-95 Hz: -0.00
        spectrum_fit @ 60.034     -2.77 dB     45-58 Hz: -0.00   62-95 Hz: +0.00
        ZapLine, rank 4           -1.65 dB     45-58 Hz: -0.06   62-95 Hz: -0.21

    So the 0.84 Hz the trade would recover is mains structure rather than usable
    spectrum, it sits inside the unanalysed 58-62 gap, and the notch's measured cost in
    the bands that are analysed is 0.00 dB.
    """
    core, workflow = _configs()

    assert float(core.get("preprocessing.notch_freq")) == 60.0, (
        "60 Hz is a 2 Hz-wide cluster of 32-38 lines; spectrum_fit removed 2.77 dB of it "
        "and ZapLine 1.65 dB, against 56.66 dB for the FIR notch"
    )
    assert bool(workflow.get("line_comb_removal.exclude_mains")), (
        "with the FIR notch taking mains, the comb must not also remove at grid harmonic "
        "50: it cannot hit the cluster and only digs a 0.41 Hz hole at 59.999 Hz"
    )
