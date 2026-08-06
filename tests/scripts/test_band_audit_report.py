"""The audit must score the removal against the lines the removal actually targets.

This report produces the before/after tables that judge whether the line work helped. If
its line list drifts from the removal's, the tables measure something else: excess is
charged at frequencies nothing removed, and the frequencies that were removed go unscored.
"""

from __future__ import annotations

import pytest

from studies.pain_study.analysis.line_comb import removal as lr
from studies.pain_study.scripts import band_audit_report as bar


STALE_HARDCODED = (23.7776, 29.6854, 46.5839, 57.1925, 59.0168, 61.0353, 61.4039, 99.5982)


def test_the_independent_lines_are_not_a_hardcoded_copy():
    """The copy this replaced carried five frequencies nothing removes and missed 94 Hz.

    Among them 61.0353 Hz, dropped from the removal for sitting 0.128 Hz from comb
    harmonic 51. Since this report is what judges whether the line work helped, a drifted
    list charges excess where nothing was removed and leaves what was removed unscored.
    """
    lines = bar._audited_lines("sub-0000")
    assert tuple(lines) != STALE_HARDCODED
    assert not any(abs(f - 61.0353) <= 0.05 for f in lines), (
        "61.0353 Hz is not a removal target; masking it discards untouched spectrum"
    )


def test_the_independent_lines_track_what_the_removal_recorded():
    """The required manifest is the source.

    Reading the config was the first fix, and it is no longer sufficient: lines are
    detected per session now, so only the manifest knows the latter, and it changes with
    each apply -- which is the point, since the audit scores the derivatives that apply
    produced.
    """
    assert tuple(bar._audited_lines("sub-0000")) == lr.removed_isolated_lines(
        bar.MANIFEST,
        subject="sub-0000",
    )


def test_the_audit_uses_the_active_workflow_report_directory():
    assert bar.MANIFEST.parent.name == "line_comb_removal"


def test_the_gradient_comb_stays_derived_from_tr_not_listed():
    """TR is a scanner constant, so deriving the comb from it is right, not a hardcoding.

    The distinction this file cares about is whether a constant varies between
    participants. The isolated lines do -- 0.19 to 0.595 Hz of scatter -- so they must be
    read from the config. TR does not: it is 0.9 s by the Volume markers for everyone, and
    k/TR is the honest way to name the gradient comb.
    """
    assert bar.TR == pytest.approx(0.9)
    assert bar.COMB_HZ[0] == pytest.approx(1 / 0.9)
    spacing = [b - a for a, b in zip(bar.COMB_HZ, bar.COMB_HZ[1:])]
    assert all(s == pytest.approx(1 / 0.9) for s in spacing), (
        "the gradient comb must stay an arithmetic series in k/TR"
    )


def test_the_audit_does_not_inherit_a_cardiac_participant_exclusion():
    """sub-0008 is excluded for a cardiac reason, which a spectral audit does not share.

    Its BCG detection is unreliable because it sits at 60.0 bpm, at the edge of the
    detector's rate window. That says nothing about narrowband line contamination -- and
    sub-0008 carries the second-worst case of it, 10.1% of its 62-95 Hz power in the 94 Hz
    line. Defaulting it out hid the audit's own worst example from the audit.

    The flag stays, so a caller who needs an exclusion can pass one. What is wrong is
    carrying one participant's cardiac problem as a default of a spectral measurement.
    """
    import argparse

    parser = argparse.ArgumentParser()
    bar._add_arguments(parser)
    args = parser.parse_args(["--deriv-root", "/tmp/x", "--out", "/tmp/y.csv"])

    assert list(args.exclude) == [], (
        "no participant should be excluded by default; the cardiac exclusion belongs to "
        "the analyses that depend on cardiac correction, not to a spectral audit"
    )
