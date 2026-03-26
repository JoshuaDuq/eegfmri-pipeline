from __future__ import annotations

import numpy as np
import pandas as pd

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.plotting.features.asymmetry import (
    _collect_column_comparison_data,
    _collect_window_comparison_data,
    _get_asymmetry_columns,
    _get_asymmetry_identifiers,
)


def test_naming_schema_parses_logdiff_activation_stat() -> None:
    parsed = NamingSchema.parse("asymmetry_analysis_alpha_chpair_F3-F4_logdiff_activation")

    assert parsed["valid"] is True
    assert parsed["identifier"] == "F3-F4"
    assert parsed["stat"] == "logdiff_activation"


def test_get_asymmetry_columns_requires_both_channels_inside_roi() -> None:
    features_df = pd.DataFrame(
        {
            "asymmetry_analysis_alpha_chpair_C3-C4_index": [0.1, 0.2],
            "asymmetry_analysis_alpha_chpair_F3-F4_index": [0.3, 0.4],
        }
    )
    rois = {
        "Sensorimotor_Left": [r"^C3$"],
        "Frontal": [r"^(F3|F4)$"],
    }

    left_columns = _get_asymmetry_columns(
        features_df=features_df,
        segment="analysis",
        band="alpha",
        metric="index",
        roi_name="Sensorimotor_Left",
        rois=rois,
    )
    frontal_columns = _get_asymmetry_columns(
        features_df=features_df,
        segment="analysis",
        band="alpha",
        metric="index",
        roi_name="Frontal",
        rois=rois,
    )

    assert left_columns == []
    assert frontal_columns == ["asymmetry_analysis_alpha_chpair_F3-F4_index"]


def test_get_asymmetry_identifiers_returns_pair_specific_identifiers() -> None:
    features_df = pd.DataFrame(
        {
            "asymmetry_baseline_alpha_chpair_F3-F4_index": [0.1, 0.2],
            "asymmetry_active_alpha_chpair_F3-F4_index": [0.3, 0.4],
            "asymmetry_baseline_alpha_chpair_F7-F8_index": [0.5, 0.6],
            "asymmetry_active_alpha_chpair_F7-F8_index": [0.7, 0.8],
        }
    )
    rois = {"Frontal": [r"^(F3|F4|F7|F8)$"]}

    identifiers = _get_asymmetry_identifiers(
        features_df=features_df,
        metric="index",
        roi_name="Frontal",
        rois=rois,
    )

    assert identifiers == ["F3-F4", "F7-F8"]


def test_collect_window_comparison_data_keeps_pair_specific_values() -> None:
    features_df = pd.DataFrame(
        {
            "asymmetry_baseline_alpha_chpair_F3-F4_index": [0.1, 0.2, 0.3],
            "asymmetry_active_alpha_chpair_F3-F4_index": [0.4, 0.5, 0.6],
            "asymmetry_baseline_alpha_chpair_F7-F8_index": [10.0, 20.0, 30.0],
            "asymmetry_active_alpha_chpair_F7-F8_index": [40.0, 50.0, 60.0],
        }
    )

    data_by_band = _collect_window_comparison_data(
        features_df=features_df,
        segment1="baseline",
        segment2="active",
        bands=["alpha"],
        metric="index",
        roi_name="all",
        rois={},
        pair_id="F3-F4",
    )

    baseline_values, active_values = data_by_band["alpha"]
    np.testing.assert_allclose(baseline_values, np.array([0.1, 0.2, 0.3]))
    np.testing.assert_allclose(active_values, np.array([0.4, 0.5, 0.6]))


def test_collect_column_comparison_data_keeps_pair_specific_values() -> None:
    features_df = pd.DataFrame(
        {
            "asymmetry_analysis_alpha_chpair_F3-F4_index": [0.1, 0.2, 0.3, 0.4],
            "asymmetry_analysis_alpha_chpair_F7-F8_index": [10.0, 20.0, 30.0, 40.0],
        }
    )
    mask1 = pd.Series([True, True, False, False])
    mask2 = pd.Series([False, False, True, True])

    cell_data = _collect_column_comparison_data(
        features_df=features_df,
        segment_name="analysis",
        bands=["alpha"],
        metric="index",
        roi_name="all",
        rois={},
        mask1=mask1,
        mask2=mask2,
        pair_id="F3-F4",
    )

    np.testing.assert_allclose(cell_data[0]["v1"], np.array([0.1, 0.2]))
    np.testing.assert_allclose(cell_data[0]["v2"], np.array([0.3, 0.4]))
