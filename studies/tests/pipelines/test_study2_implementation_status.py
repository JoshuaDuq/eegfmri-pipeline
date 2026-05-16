from __future__ import annotations

import pytest


def test_assert_confirmatory_pipeline_ready_surfaces_unimplemented_protocol_components() -> None:
    from studies.pain_study.study2.implementation_status import (
        assert_confirmatory_pipeline_ready,
    )

    with pytest.raises(NotImplementedError) as excinfo:
        assert_confirmatory_pipeline_ready()

    message = str(excinfo.value)
    assert "source-stage precision and calibration simulation" in message
    assert "true-target and prediction-error diagnostic maps" in message
    assert "artifact and robustness interpretation gates" in message
    assert "target-retrained source permutations" in message
    assert "group-level cluster inference" in message
    assert "directional-consistency gating" in message
    assert "BrainSMASH spatial comparison" in message
    assert "cross-validated criterion overlap" in message
