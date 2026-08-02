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


def test_the_narrow_mains_notch_is_the_one_in_use():
    core, workflow = _configs()

    assert core.get("preprocessing.notch_freq") in (None, False), (
        "the FIR notch occupies 0.97 Hz against 0.133 Hz for spectrum_fit at freq/450; "
        "leaving it on forfeits 0.84 Hz and keeps 58-62 Hz unusable"
    )
    ratio = float(workflow.get("line_comb_removal.notch_width_ratio"))
    assert 60.0 / ratio < 0.2, (
        f"mains notch would be {60.0 / ratio:.3f} Hz wide; the point of the move is a "
        "notch narrower than 0.2 Hz"
    )
