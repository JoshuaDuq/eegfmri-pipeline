from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from studies.pain_study.study2.config import load_study2_config


def test_build_source_stage_frame_writes_standardized_band_scores(monkeypatch) -> None:
    from studies.pain_study.scripts import study2_prepare_source_stage_input as builder

    groups = np.asarray(["sub-0001"] * 3 + ["sub-0002"] * 3, dtype=object)
    context = SimpleNamespace(
        groups=groups,
        meta=pd.DataFrame(
            {
                "subject_id": groups,
                "within_run_trial": [1, 2, 3, 1, 2, 3],
            }
        ),
    )
    raw_scores = pd.DataFrame(
        {
            "subject_id": groups,
            "trial_id": np.arange(6, dtype=int),
            "eta_combined": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "eta_alpha": [2.0, 4.0, 6.0, 3.0, 6.0, 9.0],
            "eta_beta": [3.0, 6.0, 9.0, 2.0, 4.0, 6.0],
            "eta_gamma": [4.0, 8.0, 12.0, 5.0, 10.0, 15.0],
        }
    )
    monkeypatch.setattr(
        builder,
        "compute_held_out_contribution_scores",
        lambda *_args, **_kwargs: raw_scores,
        raising=False,
    )

    frame, qc = builder.build_source_stage_frame(context, load_study2_config())

    assert frame["trial_index_within_run"].tolist() == [1, 2, 3, 1, 2, 3]
    for column in ("eta_combined_z", "eta_alpha_z", "eta_beta_z", "eta_gamma_z"):
        assert column in frame.columns
        for subject in ("sub-0001", "sub-0002"):
            values = frame.loc[frame["subject_id"].eq(subject), column]
            assert np.mean(values) == pytest.approx(0.0)
            assert np.std(values, ddof=0) == pytest.approx(1.0)
    assert qc["contribution_criteria_met"].all()
