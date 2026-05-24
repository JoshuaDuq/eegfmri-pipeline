import pandas as pd

from eeg_pipeline.utils.config.loader import ConfigDict, get_config_value
from studies.pain_study.study1.cohort import primary_targets_parquet_path
from studies.pain_study.study1.config import load_study1_config
from studies.pain_study.study1.feature_benchmark import (
    PRIMARY_BAND_PRESETS,
    _feature_benchmark_config,
)

LEVEL2_CONTINUOUS_COLUMNS = [
    "block",
    "onset",
    "within_block_trial",
    "hrf_weighted_framewise_displacement",
    "hrf_weighted_std_dvars",
    "hrf_weighted_fp1_fp2_high_frequency_power",
    "residual_ecg_coupling",
]
LEVEL2_CATEGORICAL_COLUMNS = ["stimulus_temp", "selected_surface"]
LEVEL2_COLUMNS = LEVEL2_CONTINUOUS_COLUMNS + [
    "stimulus_temp_level_46_0",
    "selected_surface_level_2_0",
]


def test_study1_default_residualization_matches_level2_estimand() -> None:
    config = load_study1_config()

    nuisance = config["study1"]["targets"]["nuisance_regression"]

    assert nuisance["enabled"] is True
    assert nuisance["continuous_columns"] == LEVEL2_CONTINUOUS_COLUMNS
    assert nuisance["categorical_columns"] == LEVEL2_CATEGORICAL_COLUMNS


def test_study1_default_subject_minimum_supports_inner_group_kfold() -> None:
    config = load_study1_config()

    min_subjects = config["study1"]["cohort"]["min_subjects"]
    inner_splits = config["study1"]["feature_benchmark"]["inner_splits"]

    assert min_subjects >= inner_splits + 1


def test_study1_default_circular_shift_structure_rules_match_readme() -> None:
    config = load_study1_config()
    circular_shift = config["study1"]["feature_benchmark"]["circular_shift"]

    assert circular_shift["min_valid_blocks_per_subject"] == 3
    assert circular_shift["min_retained_trials_per_subject"] == 25
    assert config["study1"]["feature_benchmark"]["max_invalid_permutation_fraction"] == 0.20


def test_study1_default_clean_events_qc_matches_artifact_proxy_estimand() -> None:
    config = load_study1_config()
    qc = config["preprocessing"]["clean_events_qc"]["peripheral_low_gamma"]

    assert qc["enabled"] is True
    assert qc["output_column"] == "fp1_fp2_high_frequency_power"
    assert qc["channels"] == ["Fp1", "Fp2"]
    assert qc["band"] == [70.0, 95.0]


def test_study1_power_presets_include_low_and_high_frequency_exploration() -> None:
    assert PRIMARY_BAND_PRESETS["delta"] == ["delta"]
    assert PRIMARY_BAND_PRESETS["theta"] == ["theta"]
    assert PRIMARY_BAND_PRESETS["gamma"] == ["gamma"]
    assert PRIMARY_BAND_PRESETS["all_bands"] == [
        "delta",
        "theta",
        "alpha",
        "beta",
        "gamma",
    ]


def test_feature_benchmark_primary_config_is_non_transductive(tmp_path) -> None:
    config = ConfigDict(load_study1_config())
    config["paths"] = {"deriv_root": str(tmp_path / "derivatives")}
    target_table_path = primary_targets_parquet_path(config)
    target_table_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "subject_id": ["sub-0001", "sub-0001"],
            "NPS": [1.0, 2.0],
            "SIIPS1": [2.0, 3.0],
            "block": [1, 1],
            "onset": [10.0, 20.0],
            "trial_index": [1, 2],
            "within_block_trial": [1, 2],
            "hrf_weighted_framewise_displacement": [0.1, 0.2],
            "hrf_weighted_std_dvars": [0.5, 0.6],
            "hrf_weighted_fp1_fp2_high_frequency_power": [1.1, 1.2],
            "residual_ecg_coupling": [0.01, 0.02],
            "stimulus_temp": [44.0, 46.0],
            "selected_surface": [1.0, 2.0],
            "stimulus_temp_level_46_0": [0.0, 1.0],
            "selected_surface_level_2_0": [0.0, 1.0],
        }
    ).to_parquet(target_table_path)

    feature_config = _feature_benchmark_config(config, target_name="NPS")

    assert (
        get_config_value(
            feature_config,
            "machine_learning.preprocessing.subject_standardize_features",
            None,
        )
        is False
    )
    assert (
        get_config_value(feature_config, "machine_learning.target_residualization.columns", None)
        == LEVEL2_COLUMNS
    )
    assert (
        get_config_value(feature_config, "machine_learning.cv.permutation_scheme", None)
        == "circular_shift_within_run"
    )
    assert (
        get_config_value(
            feature_config,
            "machine_learning.cv.circular_shift.min_valid_blocks_per_subject",
            None,
        )
        == 3
    )
    assert (
        get_config_value(
            feature_config,
            "machine_learning.cv.circular_shift.min_retained_trials_per_subject",
            None,
        )
        == 25
    )
    assert (
        get_config_value(
            feature_config,
            "machine_learning.cv.max_invalid_permutation_fraction",
            None,
        )
        == 0.20
    )
    assert get_config_value(feature_config, "machine_learning.data.excluded_channels", None) == [
        "Fp1",
        "Fp2",
    ]
