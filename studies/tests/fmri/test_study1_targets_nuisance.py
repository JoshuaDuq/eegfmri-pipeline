"""Tests for studies.pain_study.study1.targets nuisance regression helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from studies.tests.test_support import DotConfig


def _nuisance_config(
    *,
    enabled: bool = True,
    continuous: list[str] | None = None,
    categorical: list[str] | None = None,
) -> DotConfig:
    return DotConfig(
        {
            "study1": {
                "targets": {
                    "names": ["NPS", "SIIPS1"],
                    "nuisance_regression": {
                        "enabled": enabled,
                        "continuous_columns": continuous or [],
                        "categorical_columns": categorical or [],
                    },
                }
            }
        }
    )


###################################################################
# nuisance_regression_enabled
###################################################################


def test_nuisance_regression_enabled_returns_false_when_not_configured() -> None:
    from studies.pain_study.study1.targets import nuisance_regression_enabled

    cfg = DotConfig({"study1": {"targets": {}}})

    assert nuisance_regression_enabled(cfg) is False


def test_nuisance_regression_enabled_returns_true_when_set() -> None:
    from studies.pain_study.study1.targets import nuisance_regression_enabled

    cfg = _nuisance_config(enabled=True)

    assert nuisance_regression_enabled(cfg) is True


###################################################################
# nuisance_source_columns
###################################################################


def test_nuisance_source_columns_combines_continuous_and_categorical() -> None:
    from studies.pain_study.study1.targets import nuisance_source_columns

    cfg = _nuisance_config(
        continuous=["block", "onset"],
        categorical=["stimulus_temp"],
    )

    assert nuisance_source_columns(cfg) == ("block", "onset", "stimulus_temp")


def test_nuisance_source_columns_rejects_duplicates() -> None:
    from studies.pain_study.study1.targets import nuisance_source_columns

    cfg = _nuisance_config(
        continuous=["block", "onset"],
        categorical=["block"],
    )

    with pytest.raises(ValueError, match="unique"):
        nuisance_source_columns(cfg)


def test_nuisance_source_columns_requires_at_least_one_when_enabled() -> None:
    from studies.pain_study.study1.targets import nuisance_source_columns

    cfg = _nuisance_config(continuous=[], categorical=[])

    with pytest.raises(ValueError, match="at least one"):
        nuisance_source_columns(cfg)


def test_nuisance_source_columns_rejects_raw_artifact_columns_for_level2() -> None:
    from studies.pain_study.study1.targets import nuisance_source_columns

    cfg = _nuisance_config(
        continuous=[
            "block",
            "framewise_displacement",
            "std_dvars",
            "fp1_fp2_high_frequency_power",
        ],
        categorical=[],
    )

    with pytest.raises(ValueError, match="HRF-weighted"):
        nuisance_source_columns(cfg)


###################################################################
# _confound_timeseries
###################################################################


def test_confound_timeseries_handles_read_only_column_arrays(monkeypatch) -> None:
    from studies.pain_study.study1.targets import _confound_timeseries

    confounds = pd.DataFrame({"framewise_displacement": [None, 0.1, 0.2]})
    original_to_numpy = pd.Series.to_numpy

    def read_only_to_numpy(series, *args, **kwargs):
        values = original_to_numpy(series, *args, **kwargs)
        values.setflags(write=False)
        return values

    monkeypatch.setattr(pd.Series, "to_numpy", read_only_to_numpy)

    values = _confound_timeseries(
        confounds,
        "framewise_displacement",
        n_scans=3,
    )

    assert values.tolist() == [0.0, 0.1, 0.2]


###################################################################
# _ordered_categorical_levels
###################################################################


def test_ordered_categorical_levels_sorts_numerically() -> None:
    from studies.pain_study.study1.targets import _ordered_categorical_levels

    values = pd.Series([46.0, 44.0, 47.0, 44.0])
    levels = _ordered_categorical_levels(values)

    assert levels == (44.0, 46.0, 47.0)


def test_ordered_categorical_levels_sorts_strings_alphabetically() -> None:
    from studies.pain_study.study1.targets import _ordered_categorical_levels

    values = pd.Series(["cherry", "apple", "banana"])
    levels = _ordered_categorical_levels(values)

    assert levels == ("apple", "banana", "cherry")


###################################################################
# _level_token
###################################################################


def test_level_token_formats_numeric_value() -> None:
    from studies.pain_study.study1.targets import _level_token

    assert _level_token(46.0) == "46_0"
    assert _level_token(47) == "47_0"


def test_level_token_formats_string_value() -> None:
    from studies.pain_study.study1.targets import _level_token

    assert _level_token("high") == "high"
    assert _level_token("left_arm") == "left_arm"


def test_level_token_rejects_empty_result() -> None:
    from studies.pain_study.study1.targets import _level_token

    with pytest.raises(ValueError, match="Cannot create"):
        _level_token("   ")


###################################################################
# _categorical_level_column
###################################################################


def test_categorical_level_column_format() -> None:
    from studies.pain_study.study1.targets import _categorical_level_column

    assert _categorical_level_column("stimulus_temp", 46.0) == "stimulus_temp_level_46_0"
    assert _categorical_level_column("surface", "left") == "surface_level_left"


###################################################################
# resolve_residualization_columns
###################################################################


def test_resolve_residualization_columns_returns_empty_when_disabled() -> None:
    from studies.pain_study.study1.targets import resolve_residualization_columns

    cfg = _nuisance_config(enabled=False)
    frame = pd.DataFrame({"block": [1]})

    assert resolve_residualization_columns(frame=frame, config=cfg) == ()


def test_resolve_residualization_columns_includes_continuous_and_categorical_dummies() -> None:
    from studies.pain_study.study1.targets import resolve_residualization_columns

    cfg = _nuisance_config(
        continuous=["block"],
        categorical=["stimulus_temp"],
    )
    frame = pd.DataFrame(
        {
            "block": [1, 2, 3],
            "stimulus_temp": [44.0, 46.0, 47.0],
            "stimulus_temp_level_46_0": [0.0, 1.0, 0.0],
            "stimulus_temp_level_47_0": [0.0, 0.0, 1.0],
        }
    )

    columns = resolve_residualization_columns(frame=frame, config=cfg)

    assert columns == ("block", "stimulus_temp_level_46_0", "stimulus_temp_level_47_0")


def test_resolve_residualization_columns_rejects_missing_continuous(tmp_path) -> None:
    from studies.pain_study.study1.targets import resolve_residualization_columns

    cfg = _nuisance_config(continuous=["missing_col"])
    frame = pd.DataFrame({"block": [1]})

    with pytest.raises(ValueError, match="missing continuous"):
        resolve_residualization_columns(frame=frame, config=cfg)


###################################################################
# _optional_string_tuple
###################################################################


def test_optional_string_tuple_returns_none_for_none() -> None:
    from studies.pain_study.study1.targets import _optional_string_tuple

    assert _optional_string_tuple(None, field_name="test") is None


def test_optional_string_tuple_normalizes_strings() -> None:
    from studies.pain_study.study1.targets import _optional_string_tuple

    result = _optional_string_tuple(
        ["  stimulation  ", "rating"],
        field_name="test",
    )

    assert result == ("stimulation", "rating")


def test_optional_string_tuple_returns_none_for_all_empty() -> None:
    from studies.pain_study.study1.targets import _optional_string_tuple

    assert _optional_string_tuple(["", "  "], field_name="test") is None


def test_optional_string_tuple_rejects_non_list() -> None:
    from studies.pain_study.study1.targets import _optional_string_tuple

    with pytest.raises(ValueError, match="list"):
        _optional_string_tuple("stimulation", field_name="test")


###################################################################
# iter_primary_subjects
###################################################################


def test_iter_primary_subjects_deduplicates_and_sorts() -> None:
    from studies.pain_study.study1.targets import iter_primary_subjects

    table = pd.DataFrame(
        {
            "subject_id": ["sub-0003", "sub-0001", "sub-0003", "sub-0001", "sub-0002"],
        }
    )

    subjects = list(iter_primary_subjects(table))

    assert subjects == ["sub-0001", "sub-0002", "sub-0003"]


###################################################################
# _primary_signatures validation
###################################################################


def test_primary_signatures_rejects_wrong_order() -> None:
    from studies.pain_study.study1.targets import _primary_signatures

    cfg = DotConfig({"study1": {"targets": {"names": ["SIIPS1", "NPS"]}}})

    with pytest.raises(ValueError, match="exactly"):
        _primary_signatures(cfg)


def test_primary_signatures_rejects_non_list() -> None:
    from studies.pain_study.study1.targets import _primary_signatures

    cfg = DotConfig({"study1": {"targets": {"names": "NPS"}}})

    with pytest.raises(ValueError, match="list"):
        _primary_signatures(cfg)
