from __future__ import annotations

from pathlib import Path

import pandas as pd


def _write_behavior_csv_with_two_trials(path: Path, *, run_id: int, t0: float) -> None:
    # Paradigm timing (scheduled):
    # - ramp up:   3.0s
    # - plateau:   7.5s
    # - ramp down: 2.0s
    # - total stim: 12.5s
    df = pd.DataFrame(
        [
            {
                "run_id": int(run_id),
                "trial_number": 1,
                "stimulus_temp": 46.0,
                "selected_surface": 1,
                "pain_binary_coded": 1,
                "vas_final_coded_rating": 150.0,
                "iti_start_time": t0 + 0.0,
                "iti_end_time": t0 + 10.0,
                "stim_start_time": t0 + 10.0,
                "stim_end_time": t0 + 22.5,
                "pain_q_start_time": t0 + 25.0,
                "pain_q_end_time": t0 + 26.0,
                "vas_start_time": t0 + 26.0,
                "vas_end_time": t0 + 28.0,
            },
            {
                "run_id": int(run_id),
                "trial_number": 2,
                "stimulus_temp": 48.0,
                "selected_surface": 2,
                "pain_binary_coded": 0,
                "vas_final_coded_rating": 35.0,
                "iti_start_time": t0 + 30.0,
                "iti_end_time": t0 + 40.0,
                "stim_start_time": t0 + 40.0,
                "stim_end_time": t0 + 52.5,
                "pain_q_start_time": t0 + 54.0,
                "pain_q_end_time": t0 + 55.0,
                "vas_start_time": t0 + 55.0,
                "vas_end_time": t0 + 57.0,
            },
        ]
    )
    df.to_csv(path, index=False)


def test_raw_to_bids_event_granularity_phases_creates_stim_phase_and_plateau(tmp_path: Path) -> None:
    from fmri_pipeline.analysis.raw_to_bids import _write_events_tsv_for_run

    beh = tmp_path / "TrialSummary.csv"
    _write_behavior_csv_with_two_trials(beh, run_id=1, t0=0.0)

    out_tsv = tmp_path / "events.tsv"
    _write_events_tsv_for_run(
        behavior_csv=beh,
        out_tsv=out_tsv,
        run_id=1,
        onset_reference="as_is",
        onset_offset_s=0.0,
        event_granularity="phases",
        bold_nifti=None,
        bold_json=None,
    )

    df = pd.read_csv(out_tsv, sep="\t")
    assert "stim_phase" in df.columns

    stim = df[df["trial_type"] == "stimulation"].copy()
    assert len(stim) == 6  # 2 trials x 3 phases

    # Trial 1: stim_start=10.0 => plateau at 13.0 for 7.5s.
    t1 = stim[stim["trial_number"] == 1]
    assert set(t1["stim_phase"].tolist()) == {"ramp_up", "plateau", "ramp_down"}
    plateau = t1[t1["stim_phase"] == "plateau"].iloc[0]
    assert float(plateau["onset"]) == 13.0
    assert float(plateau["duration"]) == 7.5

    # Trial 2: stim_start=40.0 => plateau at 43.0 for 7.5s.
    t2 = stim[stim["trial_number"] == 2]
    plateau = t2[t2["stim_phase"] == "plateau"].iloc[0]
    assert float(plateau["onset"]) == 43.0
    assert float(plateau["duration"]) == 7.5


def test_raw_to_bids_event_granularity_trial_does_not_create_stim_phase_column(tmp_path: Path) -> None:
    from fmri_pipeline.analysis.raw_to_bids import _write_events_tsv_for_run

    beh = tmp_path / "TrialSummary.csv"
    _write_behavior_csv_with_two_trials(beh, run_id=1, t0=0.0)

    out_tsv = tmp_path / "events.tsv"
    _write_events_tsv_for_run(
        behavior_csv=beh,
        out_tsv=out_tsv,
        run_id=1,
        onset_reference="as_is",
        onset_offset_s=0.0,
        event_granularity="trial",
        bold_nifti=None,
        bold_json=None,
    )

    df = pd.read_csv(out_tsv, sep="\t")
    assert "stim_phase" not in df.columns
