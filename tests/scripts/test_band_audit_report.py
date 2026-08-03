"""The audit must score the removal against the lines the removal actually targets.

This report produces the before/after tables that judge whether the line work helped. If
its line list drifts from the removal's, the tables measure something else: excess is
charged at frequencies nothing removed, and the frequencies that were removed go unscored.
"""

from __future__ import annotations

import pytest

from studies.pain_study.scripts import band_audit_report as bar
from studies.pain_study.scripts.workflow_config import load_workflow_config


def _configured_lines() -> list[float]:
    workflow = load_workflow_config("line_comb")
    return [float(f) for f in workflow.get("line_comb_removal.isolated_hz")]


def test_the_independent_lines_come_from_the_removal_config():
    """A hardcoded copy drifts, and this one had.

    It carried 61.0353 Hz, dropped from the removal for sitting 0.128 Hz from comb
    harmonic 51, and four other frequencies the removal does not target -- while carrying
    nothing near 94 Hz, where the strongest residual in the cohort sits.
    """
    assert sorted(bar.INDEPENDENT_HZ) == sorted(_configured_lines()), (
        "INDEPENDENT_HZ must be read from line_comb_removal.isolated_hz; a copy drifts "
        "and the audit then scores lines nobody removed"
    )


def test_the_audit_covers_the_lines_that_wander_between_participants():
    """The 94 Hz line spans 93.750-94.345 Hz across the cohort and must be scored."""
    lines = list(bar.INDEPENDENT_HZ)
    for position in (93.7503, 94.3453):
        nearest = min(lines, key=lambda f: abs(f - position))
        assert abs(nearest - position) <= 0.30, (
            f"no audited line within 0.30 Hz of {position} Hz; nearest is {nearest}"
        )


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
