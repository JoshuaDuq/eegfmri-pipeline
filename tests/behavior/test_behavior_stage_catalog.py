from __future__ import annotations

import unittest
from types import SimpleNamespace

from eeg_pipeline.analysis.behavior.stage_catalog import (
    COMPUTATION_TO_PIPELINE_ATTR,
    STAGE_SPEC_DEFINITIONS,
    apply_computation_flags_impl,
    config_to_stage_names_impl,
)
from eeg_pipeline.analysis.behavior.stage_registry import StageRegistry
from eeg_pipeline.pipelines.constants import BEHAVIOR_COMPUTATIONS


class TestBehaviorStageCatalog(unittest.TestCase):
    def test_config_to_stage_names_preserves_default_enables(self):
        stages = config_to_stage_names_impl(SimpleNamespace())

        self.assertEqual(stages[0], "load")
        self.assertIn("trial_table", stages)
        self.assertIn("correlate_design", stages)
        self.assertIn("correlate_fdr", stages)
        self.assertIn("hierarchical_fdr_summary", stages)
        self.assertEqual(stages[-1], "export")
        self.assertNotIn("report", stages)

    def test_config_to_stage_names_respects_explicit_flags(self):
        cfg = SimpleNamespace(
            run_trial_table=False,
            run_correlations=False,
            run_validation=False,
            run_report=True,
            run_cluster_tests=True,
        )
        stages = config_to_stage_names_impl(cfg)

        self.assertNotIn("trial_table", stages)
        self.assertNotIn("correlate_design", stages)
        self.assertNotIn("hierarchical_fdr_summary", stages)
        self.assertIn("report", stages)
        self.assertIn("cluster", stages)

    def test_apply_computation_flags_updates_pipeline_config(self):
        cfg = SimpleNamespace(
            run_trial_table=False,
            run_lag_features=False,
            run_pain_residual=False,
            run_temperature_models=False,
            run_regression=False,
            run_models=False,
            run_stability=False,
            run_icc=False,
            run_consistency=False,
            run_influence=False,
            run_report=False,
            run_correlations=False,
            run_multilevel_correlations=False,
            run_condition_comparison=False,
            run_temporal_correlations=False,
            run_cluster_tests=False,
            run_mediation=False,
            run_moderation=False,
            run_mixed_effects=False,
            compute_pain_sensitivity=False,
        )
        flags = {name: False for name in COMPUTATION_TO_PIPELINE_ATTR}
        flags["report"] = True
        flags["stability"] = True
        flags["icc"] = False
        flags["multilevel_correlations"] = True

        apply_computation_flags_impl(cfg, flags)

        self.assertTrue(cfg.run_report)
        self.assertTrue(cfg.run_stability)
        self.assertTrue(cfg.run_icc)  # stability implies ICC
        self.assertTrue(cfg.run_multilevel_correlations)

    def test_apply_computation_flags_rejects_unknown_keys(self):
        cfg = SimpleNamespace()
        with self.assertRaises(KeyError):
            apply_computation_flags_impl(cfg, {"not_a_real_computation": True})

    def test_computation_mapping_covers_cli_supported_computations(self):
        missing = sorted(set(BEHAVIOR_COMPUTATIONS) - set(COMPUTATION_TO_PIPELINE_ATTR))
        self.assertEqual(missing, [])

    def test_stage_registry_uses_canonical_config_keys(self):
        self.assertEqual(
            StageRegistry.get("trial_table").config_key,
            "behavior_analysis.trial_table.enabled",
        )
        self.assertEqual(
            StageRegistry.get("regression").config_key,
            "behavior_analysis.regression.enabled",
        )
        self.assertEqual(
            StageRegistry.get("pain_sensitivity").config_key,
            "behavior_analysis.pain_sensitivity.enabled",
        )

    def test_stage_registry_matches_catalog_definitions(self):
        expected_names = [spec.name for spec in STAGE_SPEC_DEFINITIONS]
        actual_names = list(StageRegistry.all_stages().keys())
        self.assertEqual(actual_names, expected_names)
