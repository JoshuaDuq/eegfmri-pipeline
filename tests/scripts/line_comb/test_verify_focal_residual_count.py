"""``verify`` must count focal residuals by the rule ``apply`` decides on.

It counted them against ``PreservationGate().max_focal_residual_excess_db``, a 1.0 dB
cushion the gate rewrite deleted -- so the stage raised ``AttributeError`` on every
dataset. Nothing caught it because ``verify`` runs only after ``apply`` has written a
derivative, and ``apply`` had been refusing.

The replacement is not another decibel constant. The README is explicit that the cushion
sat at 1.0 dB while 0.0 dB was already the calibrated boundary, and that the decision now
belongs to the matched-control randomization test the benchmark and ``apply`` both use.
Reusing it here is what keeps ``verify`` from disagreeing with the gate that authorised
the very data it is reading.
"""

from __future__ import annotations

import pytest

from studies.pain_study.scripts.line_comb import remove as rlc


def _rows(*p_values):
    return [
        {
            "focal_residual_null_p": p,
            "focal_residual_excess_db": 0.0,
            "max_channel_block_residual_prominence_db": 1.0,
        }
        for p in p_values
    ]


def test_a_clean_cohort_reports_no_focal_residual():
    assert rlc.focal_residual_discoveries(_rows(1.0, 1.0, 0.8)) == 0


def test_a_recording_over_its_own_controls_is_counted():
    """0.0244 is 1/41, the floor a recording can reach against 40 matched controls."""
    assert rlc.focal_residual_discoveries(_rows(0.0244)) == 1


def test_the_count_matches_the_verdict_apply_refuses_on():
    """Not merely similar -- the same function, so the two stages cannot drift apart."""
    from studies.pain_study.analysis.line_comb import removal as lr

    p_values = (0.0244, 0.0244, 0.5, 1.0)

    verdict = lr.residual_randomization_verdict([row["focal_residual_null_p"] for row in _rows(*p_values)])

    assert rlc.focal_residual_discoveries(_rows(*p_values)) == int(verdict["n_discoveries"])


def test_a_single_recording_is_decided_by_its_own_exact_test():
    """With one row Benjamini-Hochberg reduces to p <= 0.05, which is what lets a lone
    continuous acquisition be verified at all."""
    assert rlc.focal_residual_discoveries(_rows(0.0244)) == 1
    assert rlc.focal_residual_discoveries(_rows(0.20)) == 0


def test_a_missing_p_value_is_refused_rather_than_counted_as_clean():
    """The direction that matters: an absent measurement must not read as a pass."""
    with pytest.raises(KeyError):
        rlc.focal_residual_discoveries([{"focal_residual_excess_db": 0.0}])
