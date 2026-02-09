from __future__ import annotations

import unittest

from eeg_pipeline.domain.features.naming import infer_feature_provenance
from tests.pipelines_test_utils import DotConfig


class TestFeatureProvenance(unittest.TestCase):
    def test_microstates_subject_fitted_templates_mark_non_iid(self):
        col = "microstates_active_broadband_global_coverage_state1"
        out = infer_feature_provenance(
            feature_columns=[col],
            config=DotConfig({}),
            df_attrs={"microstate_template_source": "subject_fitted"},
        )
        props = out["columns"][col]
        self.assertEqual(props["analysis_unit"], "subject")
        self.assertFalse(props["broadcasted"])
        self.assertTrue(props["cross_trial_dependence"])
        self.assertFalse(props["trialwise_valid"])

    def test_microstates_fixed_templates_mark_trialwise_valid(self):
        col = "microstates_active_broadband_global_coverage_a"
        out = infer_feature_provenance(
            feature_columns=[col],
            config=DotConfig({}),
            df_attrs={"microstate_template_source": "fixed"},
        )
        props = out["columns"][col]
        self.assertEqual(props["analysis_unit"], "trial")
        self.assertFalse(props["broadcasted"])
        self.assertFalse(props["cross_trial_dependence"])
        self.assertTrue(props["trialwise_valid"])

    def test_microstates_unknown_template_source_marked_non_iid(self):
        col = "microstates_active_broadband_global_coverage_state1"
        out = infer_feature_provenance(
            feature_columns=[col],
            config=DotConfig({}),
            df_attrs={},
        )
        props = out["columns"][col]
        self.assertEqual(props["analysis_unit"], "unknown")
        self.assertFalse(props["broadcasted"])
        self.assertTrue(props["cross_trial_dependence"])
        self.assertFalse(props["trialwise_valid"])

    def test_connectivity_provenance_prefers_df_attrs_phase_estimator(self):
        col = "conn_active_alpha_global_wpli_mean"
        out = infer_feature_provenance(
            feature_columns=[col],
            config=DotConfig(
                {
                    "feature_engineering": {
                        "connectivity": {
                            "phase_estimator": "within_epoch",
                            "granularity": "trial",
                        }
                    }
                }
            ),
            df_attrs={
                "feature_granularity": "trial",
                "phase_estimator": "across_epochs",
            },
        )
        props = out["columns"][col]
        self.assertTrue(props["broadcasted"])
        self.assertTrue(props["cross_trial_dependence"])
        self.assertFalse(props["trialwise_valid"])
        self.assertEqual(out["methods"]["connectivity_phase_estimator"], "across_epochs")


if __name__ == "__main__":
    unittest.main()
