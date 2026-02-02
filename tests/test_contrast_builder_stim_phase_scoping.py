from __future__ import annotations

import pandas as pd


def test_contrast_builder_defaults_to_plateau_only_when_stim_phase_present() -> None:
    from fmri_pipeline.analysis.contrast_builder import _apply_stimulation_phase_scoping

    events = pd.DataFrame(
        [
            {"onset": 0.0, "duration": 3.0, "trial_type": "stimulation", "stim_phase": "ramp_up"},
            {"onset": 3.0, "duration": 7.5, "trial_type": "stimulation", "stim_phase": "plateau"},
            {"onset": 10.5, "duration": 2.0, "trial_type": "stimulation", "stim_phase": "ramp_down"},
            {"onset": 13.0, "duration": 1.0, "trial_type": "pain_question", "stim_phase": None},
        ]
    )

    scoped = _apply_stimulation_phase_scoping(events, allowed_stim_phases=None)
    stim = scoped[scoped["trial_type"] == "stimulation"]
    assert stim["stim_phase"].tolist() == ["plateau"]
    other = scoped[scoped["trial_type"] != "stimulation"]
    assert len(other) == 1


def test_contrast_builder_allows_disabling_stim_phase_scoping() -> None:
    from fmri_pipeline.analysis.contrast_builder import _apply_stimulation_phase_scoping

    events = pd.DataFrame(
        [
            {"onset": 0.0, "duration": 3.0, "trial_type": "stimulation", "stim_phase": "ramp_up"},
            {"onset": 3.0, "duration": 7.5, "trial_type": "stimulation", "stim_phase": "plateau"},
        ]
    )

    scoped = _apply_stimulation_phase_scoping(events, allowed_stim_phases=["all"])
    stim = scoped[scoped["trial_type"] == "stimulation"]
    assert set(stim["stim_phase"].tolist()) == {"ramp_up", "plateau"}
