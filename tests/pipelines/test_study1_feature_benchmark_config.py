import pandas as pd

from eeg_pipeline.utils.config.loader import ConfigDict, get_config_value
from studies.pain_study.study1.cohort import primary_targets_parquet_path
from studies.pain_study.study1.config import load_study1_config
from studies.pain_study.study1.feature_benchmark import _feature_benchmark_config

LEVEL2_COLUMNS = ["block", "onset", "stimulus_temp_level_46_0"]


def test_study1_default_residualization_matches_level2_estimand() -> None:
    config = load_study1_config()

    nuisance = config["study1"]["targets"]["nuisance_regression"]

    assert nuisance["enabled"] is True
    assert nuisance["continuous_columns"] == ["block", "onset"]
    assert nuisance["categorical_columns"] == ["stimulus_temp"]


def test_study1_default_subject_minimum_supports_inner_group_kfold() -> None:
    config = load_study1_config()

    min_subjects = config["study1"]["cohort"]["min_subjects"]
    inner_splits = config["study1"]["feature_benchmark"]["inner_splits"]

    assert min_subjects >= inner_splits + 1


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
            "stimulus_temp": [44.0, 46.0],
            "stimulus_temp_level_46_0": [0.0, 1.0],
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
