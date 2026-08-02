"""The chain must be able to say which generation became the delivered data."""

from __future__ import annotations

import pytest

data_chain = pytest.importorskip("studies.pain_study.analysis.data_chain")


def _vmrk(path, n_markers, *, sfreq_us=1000):
    """A minimal BrainVision .vmrk carrying n R markers."""
    lines = ["Brain Vision Data Exchange Marker File, Version 1.0", "", "[Marker Infos]"]
    for i in range(n_markers):
        lines.append(f"Mk{i + 1}=Pulse Artifact,R,{1 + i * sfreq_us},1,0")
    path.write_text("\n".join(lines), encoding="latin-1")
    return path


def test_a_stage_is_fingerprinted_by_its_marker_counts(tmp_path):
    stage = tmp_path / "step1"
    stage.mkdir()
    _vmrk(stage / "ThermalPainEEGFMRI_run1_sub0001_x.vmrk", 400)
    _vmrk(stage / "ThermalPainEEGFMRI_run2_sub0001_x.vmrk", 410)
    _vmrk(stage / "BaselineEEG_sub0001_x.vmrk", 500)

    counts = data_chain.stage_marker_counts(stage)

    assert counts == {("sub0001", "1"): 400, ("sub0001", "2"): 410, ("sub0001", "baseline"): 500}


def test_macos_resource_forks_are_not_mistaken_for_recordings(tmp_path):
    """Writing to this drive leaves ._ AppleDouble files beside every real one."""
    stage = tmp_path / "step1"
    stage.mkdir()
    _vmrk(stage / "ThermalPainEEGFMRI_run1_sub0001_x.vmrk", 400)
    (stage / "._ThermalPainEEGFMRI_run1_sub0001_x.vmrk").write_bytes(b"\x00\x05\x16\x07")

    assert len(data_chain.stage_marker_counts(stage)) == 1


def test_the_generation_that_fed_the_delivered_data_is_identified(tmp_path):
    """Three step-2 directories existed and the layout doc named the wrong one.

    Agreement with the delivered runs is what tells them apart: on this cohort the
    superseded pass matched 15 of 90 and the real one 89 of 90.
    """
    delivered = {("sub0001", "1"): 381, ("sub0001", "2"): 542}
    candidates = {
        "step2_v1": {("sub0001", "1"): 369, ("sub0001", "2"): 500},
        "step2_v2": {("sub0001", "1"): 381, ("sub0001", "2"): 542},
    }

    scores = {name: data_chain.agreement(counts, delivered) for name, counts in candidates.items()}

    assert scores["step2_v2"].matched == 2
    assert scores["step2_v1"].matched == 0
    assert data_chain.best_match(candidates, delivered) == "step2_v2"


def test_a_chain_whose_stages_disagree_with_the_delivered_data_fails(tmp_path):
    delivered = {("sub0001", "1"): 381}
    problems = data_chain.verify_chain(
        {"step3_bcg_corrected": {("sub0001", "1"): 999}},
        delivered,
        expected_source="step3_bcg_corrected",
    )

    assert problems, "a stage that does not match the delivered data must be reported"
    assert "step3_bcg_corrected" in problems[0]


def test_a_consistent_chain_reports_nothing(tmp_path):
    delivered = {("sub0001", "1"): 381, ("sub0001", "2"): 542}
    assert (
        data_chain.verify_chain(
            {"step3_bcg_corrected": dict(delivered)},
            delivered,
            expected_source="step3_bcg_corrected",
        )
        == []
    )


def test_a_stage_missing_recordings_the_delivered_data_has_is_reported():
    delivered = {("sub0001", "1"): 381, ("sub0001", "2"): 542}
    problems = data_chain.verify_chain(
        {"step3_bcg_corrected": {("sub0001", "1"): 381}},
        delivered,
        expected_source="step3_bcg_corrected",
    )

    assert any("missing" in p for p in problems)
