from __future__ import annotations

import pandas as pd

from studies.tests.test_support import DotConfig


def _config() -> DotConfig:
    return DotConfig(
        {
            "study1": {
                "temporal_negative_controls": {
                    "feature_transform": "raw_log_power",
                    "feature_baseline_window": None,
                    "windows": {
                        "prestimulus_wide": [-5.0, -0.01],
                        "immediate_prestimulus": [-0.2, -0.01],
                    },
                    "wrong_lag_windows": {"ramp_up": [0.0, 3.0]},
                    "plateau_windows": {
                        "early_plateau": [3.0, 5.5],
                        "mid_plateau": [5.5, 8.0],
                        "late_plateau": [8.0, 10.5],
                    },
                },
            },
            "time_frequency_analysis": {"active_window": [3.0, 10.5]},
        }
    )


def _targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": ["sub-0001", "sub-0001"],
            "task": ["thermalactive", "thermalactive"],
            "run": [1, 1],
            "trial_index": [1, 2],
            "within_run_trial": [1, 2],
            "onset": [0.0, 20.0],
            "duration": [0.001, 0.001],
        }
    )


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial_id": [101, 102, 103],
            "run": [1, 1, 1],
            "trial_number": [1, 2, 3],
            "onset": [0.0, 20.0, 40.0],
            "duration": [0.001, 0.001, 0.001],
            "stim_start_time": [30.0, 50.0, 70.0],
            "stim_end_time": [42.5, 62.5, 82.5],
        }
    )


def _fmri_events() -> dict[str, pd.DataFrame]:
    return {
        "sub-0001": pd.DataFrame(
            {
                "onset": [0.0, 3.0, 10.5, 20.0, 23.0, 30.5],
                "duration": [3.0, 7.5, 2.0, 3.0, 7.5, 2.0],
                "trial_type": ["stimulation"] * 6,
                "stim_phase": [
                    "ramp_up",
                    "plateau",
                    "ramp_down",
                    "ramp_up",
                    "plateau",
                    "ramp_down",
                ],
                "run": [1, 1, 1, 1, 1, 1],
                "trial_number": [1, 1, 1, 2, 2, 2],
            }
        )
    }


def _lss_trials() -> dict[str, pd.DataFrame]:
    return {
        "sub-0001": pd.DataFrame(
            {
                "run": ["run-01", "run-01"],
                "trial_index": [1, 2],
                "events_trial_number": [1, 2],
                "onset": [3.0, 23.0],
                "duration": [7.5, 7.5],
                "events_stim_phase": ["plateau", "plateau"],
            }
        )
    }


def _temporal_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial_id": [101, 102, 103],
            "power_prestimulus_wide_alpha_ch_Cz_log10raw": [1.0, 2.0, 3.0],
            "power_immediate_prestimulus_alpha_ch_Cz_log10raw": [1.1, 2.1, 3.1],
            "power_ramp_up_alpha_ch_Cz_log10raw": [1.2, 2.2, 3.2],
            "power_early_plateau_alpha_ch_Cz_log10raw": [1.3, 2.3, 3.3],
            "power_mid_plateau_alpha_ch_Cz_log10raw": [1.4, 2.4, 3.4],
            "power_late_plateau_alpha_ch_Cz_log10raw": [1.5, 2.5, 3.5],
        }
    )


def test_build_timing_audit_maps_targets_to_clean_events_and_features() -> None:
    from studies.pain_study.scripts.study1_timing_audit import build_timing_audit

    summary, trials = build_timing_audit(
        targets=_targets(),
        events_by_subject={"sub-0001": _events()},
        fmri_events_by_subject=_fmri_events(),
        lss_trials_by_subject=_lss_trials(),
        temporal_features_by_subject={"sub-0001": _temporal_features()},
        config=_config(),
    )

    assert "alignment_classification" not in summary.columns
    assert summary.loc[0, "n_target_trials"] == 2
    assert summary.loc[0, "n_clean_events"] == 3
    assert summary.loc[0, "n_temporal_feature_rows"] == 3
    assert summary.loc[0, "n_missing_fmri_plateau_events"] == 0
    assert summary.loc[0, "n_invalid_fmri_plateau_events"] == 0
    assert summary.loc[0, "n_missing_lss_plateau_trials"] == 0
    assert summary.loc[0, "n_invalid_lss_plateau_trials"] == 0
    assert summary.loc[0, "max_abs_fmri_plateau_start_delta_s"] == 0.0
    assert summary.loc[0, "max_abs_lss_plateau_start_delta_s"] == 0.0
    assert trials["event_trial_id"].tolist() == [101, 102]
    assert trials["fmri_plateau_onset"].tolist() == [3.0, 23.0]
    assert trials["fmri_plateau_duration"].tolist() == [7.5, 7.5]
    assert trials["lss_plateau_onset"].tolist() == [3.0, 23.0]
    assert trials["lss_plateau_duration"].tolist() == [7.5, 7.5]
    assert trials["temporal_feature_row_present"].tolist() == [True, True]


def test_build_timing_audit_fails_when_target_event_has_no_temporal_feature_row() -> None:
    from studies.pain_study.scripts.study1_timing_audit import build_timing_audit

    bad_features = _temporal_features().copy()
    bad_features = bad_features.loc[bad_features["trial_id"] != 102].copy()

    summary, trials = build_timing_audit(
        targets=_targets(),
        events_by_subject={"sub-0001": _events()},
        fmri_events_by_subject=_fmri_events(),
        lss_trials_by_subject=_lss_trials(),
        temporal_features_by_subject={"sub-0001": bad_features},
        config=_config(),
    )

    assert "alignment_classification" not in summary.columns
    assert summary.loc[0, "n_missing_temporal_feature_rows"] == 1
    assert trials["temporal_feature_row_present"].tolist() == [True, False]
