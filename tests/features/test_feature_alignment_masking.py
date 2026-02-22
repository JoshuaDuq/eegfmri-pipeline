from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_pipeline.utils.data.features import align_feature_dataframes
from tests.pipelines_test_utils import DotConfig


class TestFeatureAlignmentMasking(unittest.TestCase):
    def test_drop_mask_accounts_for_extra_blocks(self):
        pow_df = pd.DataFrame(
            {"power_active_alpha_global_logratio_mean": [0.1, 0.2, 0.3]}
        )
        baseline_df = pd.DataFrame(
            {"power_baseline_alpha_global_mean": [1.0, 1.0, 1.0]}
        )
        conn_df = pd.DataFrame(
            {"conn_active_alpha_global_wpli_mean": [0.2, 0.3, 0.4]}
        )
        aper_df = pd.DataFrame(
            {"aperiodic_active_broadband_global_slope": [1.0, 1.1, 1.2]}
        )
        y = pd.Series([10.0, 20.0, 30.0], name="target")
        aligned_events = pd.DataFrame({"trial": [1, 2, 3]})
        extra_blocks = {
            "pac_trials": pd.DataFrame(
                {"pac_active_theta_gamma_global_val": [0.5, np.nan, 0.7]}
            )
        }

        (
            pow_aligned,
            baseline_aligned,
            conn_aligned,
            aper_aligned,
            y_aligned,
            retention_stats,
        ) = align_feature_dataframes(
            pow_df=pow_df,
            baseline_df=baseline_df,
            conn_df=conn_df,
            aper_df=aper_df,
            y=y,
            aligned_events=aligned_events,
            features_dir=Path(tempfile.mkdtemp()),
            logger=logging.getLogger("test-feature-alignment-masking"),
            config=DotConfig({}),
            extra_blocks=extra_blocks,
            requested_categories=["power", "connectivity", "aperiodic", "pac"],
        )

        self.assertEqual(len(pow_aligned), 2)
        self.assertEqual(len(baseline_aligned), 2)
        self.assertEqual(len(conn_aligned), 2)
        self.assertEqual(len(aper_aligned), 2)
        self.assertEqual(len(y_aligned), 2)
        self.assertIsNotNone(retention_stats.get("mask"))
        np.testing.assert_array_equal(
            retention_stats["mask"], np.array([True, False, True], dtype=bool)
        )

        pac_aligned = retention_stats["extra_blocks"]["pac_trials"]
        self.assertEqual(len(pac_aligned), 2)
        self.assertTrue(
            np.isfinite(
                pac_aligned["pac_active_theta_gamma_global_val"].to_numpy(dtype=float)
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
