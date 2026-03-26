from __future__ import annotations

import numpy as np
import pandas as pd

from eeg_pipeline.plotting.features.ratios import (
    _collect_ratio_cell_data,
    _get_ratio_columns_for_segment_pair_roi,
)


def test_get_ratio_columns_for_segment_pair_roi_selects_only_power_ratio() -> None:
    features_df = pd.DataFrame(
        {
            "ratios_active_theta_beta_global_power_ratio": [2.0, 3.0],
            "ratios_active_theta_beta_global_log_ratio": [0.69, 1.10],
            "ratios_active_alpha_beta_global_power_ratio": [1.2, 1.4],
        }
    )

    columns = _get_ratio_columns_for_segment_pair_roi(
        features_df=features_df,
        segment="active",
        pair="theta_beta",
        roi_name="all",
    )

    assert columns == ["ratios_active_theta_beta_global_power_ratio"]


def test_collect_ratio_cell_data_uses_power_ratio_scale_only() -> None:
    features_df = pd.DataFrame(
        {
            "ratios_active_theta_beta_global_power_ratio": [2.0, 4.0, 6.0, 8.0],
            "ratios_active_theta_beta_global_log_ratio": [0.69, 1.39, 1.79, 2.08],
        }
    )
    mask1 = pd.Series([True, True, False, False])
    mask2 = pd.Series([False, False, True, True])

    cell_data = _collect_ratio_cell_data(
        features_df=features_df,
        pairs=["theta_beta"],
        segment="active",
        roi_name="all",
        mask1=mask1,
        mask2=mask2,
    )

    np.testing.assert_allclose(cell_data[0]["v1"], np.array([2.0, 4.0]))
    np.testing.assert_allclose(cell_data[0]["v2"], np.array([6.0, 8.0]))
