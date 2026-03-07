from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from eeg_pipeline.infra.tsv import read_table
from eeg_pipeline.utils.data.feature_discovery import STANDARD_FEATURE_FILES
from eeg_pipeline.utils.data.feature_io import save_all_features
from tests.pipelines_test_utils import DotConfig


class TestFeatureIoPacOutputs(unittest.TestCase):
    def test_feature_discovery_prefers_pac_trials_file(self):
        self.assertEqual(
            STANDARD_FEATURE_FILES.get("pac"),
            "features_pac_trials.parquet",
        )

    def test_save_all_features_writes_distinct_pac_and_pac_trials(self):
        features_dir = Path(tempfile.mkdtemp())

        pow_df = pd.DataFrame(
            {"power_active_alpha_global_logratio_mean": [0.1, 0.2]}
        )
        pac_df = pd.DataFrame({"pac_summary_metric": [1.0, 2.0]})
        pac_trials_df = pd.DataFrame(
            {"pac_active_theta_gamma_global_mvl": [10.0, 20.0]}
        )

        save_all_features(
            pow_df=pow_df,
            pow_cols=list(pow_df.columns),
            baseline_df=pd.DataFrame(),
            baseline_cols=[],
            conn_df=None,
            conn_cols=[],
            aper_df=None,
            aper_cols=[],
            pac_df=pac_df,
            pac_trials_df=pac_trials_df,
            features_dir=features_dir,
            config=DotConfig({}),
        )

        pac_path = features_dir / "pac" / "features_pac.parquet"
        pac_trials_path = features_dir / "pac" / "features_pac_trials.parquet"

        self.assertTrue(pac_path.exists())
        self.assertTrue(pac_trials_path.exists())

        pac_saved = read_table(pac_path)
        pac_trials_saved = read_table(pac_trials_path)

        self.assertIn("pac_summary_metric", pac_saved.columns)
        self.assertNotIn("pac_active_theta_gamma_global_mvl", pac_saved.columns)
        self.assertIn("pac_active_theta_gamma_global_mvl", pac_trials_saved.columns)

    def test_save_all_features_routes_source_localization_by_method(self):
        features_dir = Path(tempfile.mkdtemp())

        pow_df = pd.DataFrame(
            {"power_active_alpha_global_logratio_mean": [0.1, 0.2]}
        )
        source_df = pd.DataFrame(
            {"src_full_eloreta_alpha_global_power": [0.3, 0.4]}
        )
        source_df.attrs["method"] = "eloreta"

        save_all_features(
            pow_df=pow_df,
            pow_cols=list(pow_df.columns),
            baseline_df=pd.DataFrame(),
            baseline_cols=[],
            conn_df=None,
            conn_cols=[],
            aper_df=None,
            aper_cols=[],
            source_df=source_df,
            source_cols=list(source_df.columns),
            features_dir=features_dir,
            config=DotConfig(
                {
                    "feature_engineering": {
                        "sourcelocalization": {
                            "method": "lcmv",
                        }
                    }
                }
            ),
        )

        eloreta_path = (
            features_dir
            / "sourcelocalization"
            / "eloreta"
            / "features_sourcelocalization.parquet"
        )
        lcmv_path = (
            features_dir
            / "sourcelocalization"
            / "lcmv"
            / "features_sourcelocalization.parquet"
        )

        self.assertTrue(eloreta_path.exists())
        self.assertFalse(lcmv_path.exists())

    def test_save_all_features_persists_trial_alignment_columns_for_trialwise_tables(self):
        features_dir = Path(tempfile.mkdtemp())

        pow_df = pd.DataFrame(
            {"power_active_alpha_global_logratio_mean": [0.1, 0.2]}
        )
        aligned_events = pd.DataFrame(
            {
                "trial_id": [10, 11],
            }
        )

        save_all_features(
            pow_df=pow_df,
            pow_cols=list(pow_df.columns),
            baseline_df=pd.DataFrame(),
            baseline_cols=[],
            conn_df=None,
            conn_cols=[],
            aper_df=None,
            aper_cols=[],
            features_dir=features_dir,
            config=DotConfig({}),
            aligned_events=aligned_events,
        )

        power_path = features_dir / "power" / "features_power.parquet"
        saved = read_table(power_path)

        self.assertIn("trial_id", saved.columns)
        self.assertIn("power_active_alpha_global_logratio_mean", saved.columns)


if __name__ == "__main__":
    unittest.main()
