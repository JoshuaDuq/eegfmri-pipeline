"""The line-comb workflow's output must be the root the rest of the pipeline reads.

These two configs are coupled and nothing else checks the coupling. The core config names
the dataset every downstream stage consumes; the line-comb workflow is what produces it.
If the workflow inherits ``bids_root`` from the core config it reads its own output and
writes ``eeg_linecleaned_linecleaned``; if the core config names the uncleaned root instead,
the pipeline silently preprocesses uncleaned data, which is what used to happen whenever
``--bids-root`` was left off the command line.
"""

from __future__ import annotations

from eeg_pipeline.utils.config.loader import load_config
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
