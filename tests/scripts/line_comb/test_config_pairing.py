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
AUDITED_RESIDUALS = {
    23.7776: "narrow, off both combs, 7/14, beta",
    29.6854: "narrow, off both combs, 9/14, 0.41 Hz below the gamma_low edge",
    61.0353: "mains +1.02 Hz sideband, 11/14",
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
