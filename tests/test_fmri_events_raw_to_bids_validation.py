from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _write_minimal_behavior_csv(path: Path, *, run_id: int, t0: float) -> None:
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
                "iti_end_time": t0 + 1.0,
                "stim_start_time": t0 + 1.0,
                "stim_end_time": t0 + 3.0,
                "pain_q_start_time": t0 + 3.2,
                "pain_q_end_time": t0 + 4.0,
                "vas_start_time": t0 + 4.0,
                "vas_end_time": t0 + 6.0,
            }
        ]
    )
    df.to_csv(path, index=False)


def test_raw_to_bids_events_validation_raises_when_out_of_bounds(tmp_path: Path) -> None:
    import nibabel as nib  # type: ignore

    from fmri_pipeline.analysis.raw_to_bids import _write_events_tsv_for_run

    # Create a short BOLD NIfTI: 10 volumes, TR=1s => duration ~10s
    bold = tmp_path / "sub-01_task-test_run-01_bold.nii.gz"
    data = np.zeros((5, 5, 5, 10), dtype=np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(img, str(bold))

    js = tmp_path / "sub-01_task-test_run-01_bold.json"
    js.write_text(json.dumps({"RepetitionTime": 1.0}), encoding="utf-8")

    beh = tmp_path / "TrialSummary.csv"
    _write_minimal_behavior_csv(beh, run_id=1, t0=100.0)  # clearly not aligned to run start

    out_tsv = tmp_path / "events.tsv"
    with pytest.raises(ValueError, match="out of bounds"):
        _write_events_tsv_for_run(
            behavior_csv=beh,
            out_tsv=out_tsv,
            run_id=1,
            onset_reference="as_is",
            onset_offset_s=0.0,
            event_granularity="trial",
            bold_nifti=bold,
            bold_json=js,
        )


def test_raw_to_bids_events_validation_writes_when_in_bounds(tmp_path: Path) -> None:
    import nibabel as nib  # type: ignore

    from fmri_pipeline.analysis.raw_to_bids import _write_events_tsv_for_run

    bold = tmp_path / "sub-01_task-test_run-01_bold.nii.gz"
    data = np.zeros((5, 5, 5, 10), dtype=np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(img, str(bold))

    js = tmp_path / "sub-01_task-test_run-01_bold.json"
    js.write_text(json.dumps({"RepetitionTime": 1.0}), encoding="utf-8")

    beh = tmp_path / "TrialSummary.csv"
    _write_minimal_behavior_csv(beh, run_id=1, t0=0.0)

    out_tsv = tmp_path / "events.tsv"
    _write_events_tsv_for_run(
        behavior_csv=beh,
        out_tsv=out_tsv,
        run_id=1,
        onset_reference="as_is",
        onset_offset_s=0.0,
        event_granularity="trial",
        bold_nifti=bold,
        bold_json=js,
    )
    assert out_tsv.exists()
    text = out_tsv.read_text(encoding="utf-8")
    assert "trial_type" in text

