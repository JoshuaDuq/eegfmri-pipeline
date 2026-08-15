"""The per-recording plan line has to say what its two numbers are.

It read ``7 supported artifact source(s) at 27.9815, 28.0000, 47.1667, ...`` -- a count of
seven beside ten frequencies. Both are right: a source is a cluster of nominals within the
spectral resolution, and 27.9815/28.0000 are one source seen twice. Printed together
without saying so, it reads as an arithmetic error in output that ends up quoted in a
methods section.
"""

from __future__ import annotations

from studies.pain_study.scripts.line_comb import remove as rlc


def _plan(*, whole_hz, source_count):
    return rlc.RunIsolatedLinePlan(
        whole_hz=tuple(whole_hz),
        window_hz=((),),
        narrow_window_hz=((),),
        source_count=source_count,
    )


def test_both_counts_are_named_when_nominals_outnumber_sources():
    plan = _plan(whole_hz=(27.9815, 28.0, 57.2593), source_count=2)

    summary = rlc.isolated_line_summary("sub-0008_task-baseline_eeg", plan)

    assert "2 artifact source(s)" in summary
    assert "3 nominal(s)" in summary


def test_the_frequencies_are_still_listed():
    plan = _plan(whole_hz=(27.9815, 28.0), source_count=1)

    summary = rlc.isolated_line_summary("sub-0000_task-thermalactive_run-1_eeg", plan)

    assert "27.9815" in summary
    assert "28.0000" in summary
    assert summary.startswith("  sub-0000_task-thermalactive_run-1_eeg:")


def test_a_recording_with_no_isolated_lines_says_so():
    plan = _plan(whole_hz=(), source_count=0)

    summary = rlc.isolated_line_summary("sub-0001_task-rest_eeg", plan)

    assert "none" in summary
