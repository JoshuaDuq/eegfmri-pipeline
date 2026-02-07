import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import ConfigDict


class _DummyCtx:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("behavior-validity-tests")
        self.also_save_csv = False
        self.group_ids = None
        self.data_qc = {}
        self.selected_feature_files = []
        self.feature_categories = []
        self.subject = "0000"
        self.task = "pain"


class TestBehaviorAnalysisValidityFixes(unittest.TestCase):
    def test_stage_runners_pass_results_to_consistency_and_influence(self):
        from eeg_pipeline.analysis.behavior.orchestration import _get_stage_runners

        stage_runners = _get_stage_runners()
        with patch(
            "eeg_pipeline.analysis.behavior.orchestration.stage_consistency"
        ) as mock_consistency, patch(
            "eeg_pipeline.analysis.behavior.orchestration.stage_influence"
        ) as mock_influence:
            outputs = {
                "correlate_fdr": pd.DataFrame({"feature": ["f1"], "p_primary": [0.1]}),
                "regression": pd.DataFrame({"feature": ["f1"], "beta": [0.2]}),
                "models": pd.DataFrame({"feature": ["f1"], "coef": [0.3]}),
            }
            stage_runners["consistency"](SimpleNamespace(), SimpleNamespace(), outputs)
            stage_runners["influence"](SimpleNamespace(), SimpleNamespace(), outputs)

            self.assertIsNotNone(mock_consistency.call_args[0][2])
            self.assertIsNotNone(mock_influence.call_args[0][2])

    def test_correlate_primary_run_unit_does_not_fallback_to_trial_level(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_correlate_primary_selection

        ctx = _DummyCtx(
            ConfigDict(
                {
                "behavior_analysis": {
                    "correlations": {
                        "primary_unit": "run_mean",
                        "p_primary_mode": "perm_if_available",
                    }
                }
                }
            )
        )
        config = SimpleNamespace(control_temperature=False)
        design = SimpleNamespace(cov_df=None, temperature_series=None)
        records = [
            {
                "feature": "power_f1",
                "target": "rating",
                "p_raw": 0.01,
                "r_raw": 0.5,
                "p_run_mean": np.nan,
                "r_run_mean": np.nan,
                "robust_method": None,
            }
        ]

        out = stage_correlate_primary_selection(ctx, config, design, records)
        self.assertTrue(np.isnan(out[0]["p_primary"]))
        self.assertEqual(out[0]["p_kind_primary"], "p_run_mean")

    def test_window_comparison_run_unit_uses_run_level_even_if_run_adjust_disabled(self):
        from eeg_pipeline.analysis.behavior.orchestration import _run_window_comparison

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ConfigDict({
                "behavior_analysis": {
                    "run_adjustment": {
                        "enabled": False,
                        "column": "run_id",
                    },
                    "condition": {
                        "window_comparison": {
                            "primary_unit": "run_mean",
                        }
                    },
                    "statistics": {"allow_iid_trials": False},
                }
            })
            ctx = _DummyCtx(config)

            df_trials = pd.DataFrame(
                {
                    "run_id": [1] * 10 + [2] * 10,
                    "power_baseline_alpha_ch_Fp1_mean": [1.0] * 10 + [2.0] * 10,
                    "power_active_alpha_ch_Fp1_mean": [2.0] * 10 + [2.1] * 10,
                }
            )

            feature_cols = [
                "power_baseline_alpha_ch_Fp1_mean",
                "power_active_alpha_ch_Fp1_mean",
            ]

            def _identity_fdr(_ctx, _cfg, df, **_kwargs):
                df = df.copy()
                df["p_fdr"] = df["p_primary"]
                return df

            with patch(
                "eeg_pipeline.analysis.behavior.orchestration._compute_unified_fdr",
                side_effect=_identity_fdr,
            ), patch(
                "eeg_pipeline.analysis.behavior.orchestration._cache.get_feature_type",
                return_value="power",
            ):
                out = _run_window_comparison(
                    ctx,
                    df_trials,
                    feature_cols,
                    windows=["baseline", "active"],
                    min_samples=0,
                    fdr_alpha=0.05,
                    suffix="",
                )

            self.assertFalse(out.empty)
            self.assertTrue(pd.notna(out.iloc[0]["p_value_run"]))
            self.assertAlmostEqual(float(out.iloc[0]["p_primary"]), float(out.iloc[0]["p_value_run"]))

    def test_condition_column_uses_groups_aligned_to_current_df(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_condition_column

        config = ConfigDict({
            "behavior_analysis": {
                "condition": {
                    "compare_column": "pain",
                    "compare_values": [1, 0],
                    "primary_unit": "run_mean",
                    "permutation": {"enabled": True},
                    "overwrite": True,
                },
                "run_adjustment": {
                    "column": "run_id",
                },
                "statistics": {"allow_iid_trials": False},
            }
        })
        ctx = _DummyCtx(config)

        n_runs = 12
        run_ids = np.repeat(np.arange(1, n_runs + 1), 2)
        pain_vals = np.repeat([1, 0] * (n_runs // 2), 2)
        df_trials = pd.DataFrame(
            {
                "run_id": run_ids,
                "pain": pain_vals,
                "power_active_alpha_ch_Fp1_mean": np.linspace(0.1, 2.4, n_runs * 2),
            }
        )
        ctx.group_ids = pd.Series(np.repeat([100, 200], n_runs), index=df_trials.index)

        captured = {}

        def _fake_condition_effects(*args, **kwargs):
            captured["groups"] = kwargs.get("groups", None)
            return pd.DataFrame(
                {
                    "feature": ["power_active_alpha_ch_Fp1_mean"],
                    "p_value": [0.5],
                    "q_value": [0.5],
                    "hedges_g": [0.1],
                }
            )

        def _identity_fdr(_ctx, _cfg, df, **_kwargs):
            df = df.copy()
            df["p_fdr"] = pd.to_numeric(df["p_primary"], errors="coerce")
            return df

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "eeg_pipeline.analysis.behavior.api.split_by_condition",
            return_value=(
                np.array([True] * (n_runs // 2) + [False] * (n_runs // 2)),
                np.array([False] * (n_runs // 2) + [True] * (n_runs // 2)),
                n_runs // 2,
                n_runs // 2,
            ),
        ), patch(
            "eeg_pipeline.analysis.behavior.api.compute_condition_effects",
            side_effect=_fake_condition_effects,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._compute_unified_fdr",
            side_effect=_identity_fdr,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_stats_subfolder",
            return_value=Path(tmpdir),
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._write_parquet_with_optional_csv",
            return_value=None,
        ):
            out = stage_condition_column(
                ctx,
                SimpleNamespace(fdr_alpha=0.05, n_jobs=1),
                df_trials=df_trials,
                feature_cols=["power_active_alpha_ch_Fp1_mean"],
            )

        self.assertFalse(out.empty)
        groups = captured.get("groups")
        self.assertIsNotNone(groups)
        self.assertEqual(len(groups), n_runs)

    def test_condition_multigroup_enforces_non_iid_trial_guard(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_condition_multigroup

        ctx = _DummyCtx(
            ConfigDict(
                {
                "behavior_analysis": {
                    "condition": {
                        "compare_column": "pain",
                        "compare_values": [0, 1, 2],
                        "primary_unit": "trial",
                    },
                    "statistics": {"allow_iid_trials": False},
                }
                }
            )
        )

        df_trials = pd.DataFrame(
            {
                "pain": np.tile([0, 1, 2], 10),
                "power_active_alpha_ch_Fp1_mean": np.linspace(0.1, 3.0, 30),
            }
        )

        with self.assertRaises(ValueError):
            stage_condition_multigroup(
                ctx,
                SimpleNamespace(fdr_alpha=0.05),
                df_trials=df_trials,
                feature_cols=["power_active_alpha_ch_Fp1_mean"],
            )

    def test_stage_moderation_uses_configured_permutations_for_primary_p(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_moderation

        ctx = _DummyCtx(
            ConfigDict(
                {
                "behavior_analysis": {
                    "moderation": {
                        "n_permutations": 25,
                        "p_primary_mode": "perm_if_available",
                    }
                }
                }
            )
        )

        df_trials = pd.DataFrame(
            {
                "temperature": np.linspace(40.0, 48.0, 30),
                "rating": np.linspace(2.0, 8.0, 30),
                "power_active_alpha_ch_Fp1_mean": np.linspace(0.1, 1.5, 30),
            }
        )

        class _FakeModerationResult:
            def __init__(self):
                self.n = 30
                self.b1 = 0.1
                self.b2 = 0.2
                self.b3 = 0.3
                self.se_b3 = 0.05
                self.p_b3 = 0.20
                self.p_b3_perm = 0.01
                self.slope_low_w = 0.1
                self.slope_mean_w = 0.2
                self.slope_high_w = 0.3
                self.p_slope_low = 0.1
                self.p_slope_mean = 0.1
                self.p_slope_high = 0.1
                self.r_squared = 0.2
                self.r_squared_change = 0.05
                self.f_interaction = 2.0
                self.p_f_interaction = 0.1
                self.jn_low = np.nan
                self.jn_high = np.nan
                self.jn_type = "none"

            def is_significant_moderation(self, alpha=0.05):
                return False

        captured = {}

        def _fake_run_moderation_analysis(**kwargs):
            captured["n_perm"] = kwargs.get("n_perm")
            return _FakeModerationResult()

        def _identity_fdr(_ctx, _cfg, df, **_kwargs):
            df = df.copy()
            df["p_fdr"] = pd.to_numeric(df["p_primary"], errors="coerce")
            return df

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_active_alpha_ch_Fp1_mean"],
        ), patch(
            "eeg_pipeline.utils.analysis.stats.moderation.run_moderation_analysis",
            side_effect=_fake_run_moderation_analysis,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._compute_unified_fdr",
            side_effect=_identity_fdr,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._cache.get_feature_type",
            return_value="power",
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_stats_subfolder",
            return_value=Path(tmpdir),
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._write_parquet_with_optional_csv",
            return_value=None,
        ):
            out = stage_moderation(
                ctx,
                SimpleNamespace(fdr_alpha=0.05, moderation_max_features=None),
            )

        self.assertEqual(captured.get("n_perm"), 25)
        self.assertAlmostEqual(float(out.iloc[0]["p_primary"]), 0.01, places=7)

    def test_stage_mediation_adds_primary_p_and_fdr(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_mediation

        ctx = _DummyCtx(
            ConfigDict(
                {
                "behavior_analysis": {
                    "mediation": {
                        "n_bootstrap": 100,
                        "n_permutations": 50,
                        "min_effect_size": 0.0,
                    }
                }
                }
            )
        )

        df_trials = pd.DataFrame(
            {
                "temperature": np.linspace(40.0, 48.0, 30),
                "rating": np.linspace(2.0, 8.0, 30),
                "power_active_alpha_ch_Fp1_mean": np.linspace(0.1, 1.5, 30),
            }
        )

        fake_result = pd.DataFrame(
            {
                "mediator": ["power_active_alpha_ch_Fp1_mean", "power_active_alpha_ch_Fp2_mean"],
                "sobel_p": [0.01, 0.04],
                "p_ab_perm": [0.02, np.nan],
                "indirect_effect": [0.1, 0.08],
            }
        )

        def _identity_fdr(_ctx, _cfg, df, **_kwargs):
            df = df.copy()
            p = pd.to_numeric(df["p_primary"], errors="coerce")
            df["p_fdr"] = p
            return df

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_active_alpha_ch_Fp1_mean", "power_active_alpha_ch_Fp2_mean"],
        ), patch(
            "eeg_pipeline.analysis.behavior.api.run_mediation_analysis",
            return_value=fake_result,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._compute_unified_fdr",
            side_effect=_identity_fdr,
        ):
            out = stage_mediation(ctx, SimpleNamespace(fdr_alpha=0.05))

        self.assertIn("p_primary", out.columns)
        self.assertIn("p_fdr", out.columns)
        self.assertAlmostEqual(float(out.loc[out["mediator"] == "power_active_alpha_ch_Fp1_mean", "p_primary"].iloc[0]), 0.02, places=7)

    def test_mediation_single_permutation_respects_groups(self):
        from eeg_pipeline.utils.analysis.stats import mediation

        captured = {}

        def _fake_compute_paths(X, M, Y):
            captured["M"] = M.copy()
            return SimpleNamespace(ab=0.1)

        with patch(
            "eeg_pipeline.utils.analysis.stats.mediation.compute_mediation_paths",
            side_effect=_fake_compute_paths,
        ):
            mediation._single_permutation_mediation(
                42,
                independent_var=np.array([1.0, 2.0, 3.0, 4.0]),
                mediator=np.array([10.0, 11.0, 20.0, 21.0]),
                dependent_var=np.array([5.0, 6.0, 7.0, 8.0]),
                groups=np.array([0, 0, 1, 1]),
                scheme="shuffle",
            )

        shuffled = captured["M"]
        self.assertCountEqual(shuffled[:2].tolist(), [10.0, 11.0])
        self.assertCountEqual(shuffled[2:].tolist(), [20.0, 21.0])

    def test_moderation_single_permutation_respects_groups(self):
        from eeg_pipeline.utils.analysis.stats import moderation

        captured = {}

        def _fake_compute_moderation(X, W, Y_shuffled, center_predictors):
            captured["Y"] = Y_shuffled.copy()
            return SimpleNamespace(b3=0.2)

        with patch(
            "eeg_pipeline.utils.analysis.stats.moderation.compute_moderation_effect",
            side_effect=_fake_compute_moderation,
        ):
            moderation._single_permutation_moderation(
                7,
                X=np.array([1.0, 2.0, 3.0, 4.0]),
                W=np.array([0.1, 0.2, 0.3, 0.4]),
                Y=np.array([10.0, 11.0, 20.0, 21.0]),
                center_predictors=True,
                groups=np.array([0, 0, 1, 1]),
                scheme="shuffle",
            )

        y_perm = captured["Y"]
        self.assertCountEqual(y_perm[:2].tolist(), [10.0, 11.0])
        self.assertCountEqual(y_perm[2:].tolist(), [20.0, 21.0])

    def test_stage_regression_run_unit_requires_run_column(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_regression

        ctx = _DummyCtx(
            ConfigDict(
                {
                    "behavior_analysis": {
                        "regression": {"primary_unit": "run_mean"},
                        "run_adjustment": {"column": "run_id"},
                    }
                }
            )
        )
        df_trials = pd.DataFrame(
            {
                "rating": np.linspace(1.0, 5.0, 20),
                "temperature": np.linspace(40.0, 48.0, 20),
                "power_active_alpha_ch_Fp1_mean": np.linspace(0.1, 2.0, 20),
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_active_alpha_ch_Fp1_mean"],
        ):
            with self.assertRaises(ValueError):
                stage_regression(ctx, SimpleNamespace(method_label=""))

    def test_stage_regression_trial_unit_requires_non_iid_inference(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_regression

        ctx = _DummyCtx(
            ConfigDict(
                {
                    "behavior_analysis": {
                        "regression": {
                            "primary_unit": "trial",
                            "n_permutations": 0,
                        },
                        "statistics": {"allow_iid_trials": False},
                    }
                }
            )
        )
        df_trials = pd.DataFrame(
            {
                "run_id": np.repeat(np.arange(1, 11), 2),
                "rating": np.linspace(1.0, 5.0, 20),
                "temperature": np.linspace(40.0, 48.0, 20),
                "power_active_alpha_ch_Fp1_mean": np.linspace(0.1, 2.0, 20),
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_active_alpha_ch_Fp1_mean"],
        ):
            with self.assertRaises(ValueError):
                stage_regression(ctx, SimpleNamespace(method_label=""))

    def test_stage_regression_groups_align_after_run_aggregation(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_regression

        ctx = _DummyCtx(
            ConfigDict(
                {
                    "behavior_analysis": {
                        "regression": {
                            "primary_unit": "run_mean",
                            "n_permutations": 10,
                        },
                        "run_adjustment": {"column": "run_id"},
                    }
                }
            )
        )
        n_runs = 10
        df_trials = pd.DataFrame(
            {
                "run_id": np.repeat(np.arange(1, n_runs + 1), 2),
                "rating": np.linspace(1.0, 5.0, n_runs * 2),
                "temperature": np.linspace(40.0, 48.0, n_runs * 2),
                "power_active_alpha_ch_Fp1_mean": np.linspace(0.1, 2.0, n_runs * 2),
            }
        )
        # Deliberately mismatched with aggregated shape to catch silent drop.
        ctx.group_ids = np.repeat([100, 200], n_runs)

        captured = {}

        def _fake_run_regressions(*args, **kwargs):
            captured["groups_for_permutation"] = kwargs.get("groups_for_permutation")
            return pd.DataFrame(), {}

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_active_alpha_ch_Fp1_mean"],
        ), patch(
            "eeg_pipeline.utils.analysis.stats.trialwise_regression.run_trialwise_feature_regressions",
            side_effect=_fake_run_regressions,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_stats_subfolder",
            return_value=Path(tmpdir),
        ):
            stage_regression(ctx, SimpleNamespace(method_label=""))

        groups = captured.get("groups_for_permutation")
        self.assertIsNotNone(groups)
        self.assertEqual(len(groups), n_runs)

    def test_stage_pain_sensitivity_trial_unit_requires_non_iid_inference(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_pain_sensitivity

        ctx = _DummyCtx(
            ConfigDict(
                {
                    "behavior_analysis": {
                        "pain_sensitivity": {
                            "primary_unit": "trial",
                            "n_permutations": 0,
                        },
                        "statistics": {"allow_iid_trials": False},
                    }
                }
            )
        )
        df_trials = pd.DataFrame(
            {
                "run_id": np.repeat(np.arange(1, 11), 2),
                "rating": np.linspace(1.0, 5.0, 20),
                "temperature": np.linspace(40.0, 48.0, 20),
                "power_active_alpha_ch_Fp1_mean": np.linspace(0.1, 2.0, 20),
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_active_alpha_ch_Fp1_mean"],
        ):
            with self.assertRaises(ValueError):
                stage_pain_sensitivity(ctx, SimpleNamespace(method="spearman", min_samples=5))

    def test_stage_models_trial_unit_requires_iid_override_or_run_level(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_models

        ctx = _DummyCtx(
            ConfigDict(
                {
                    "behavior_analysis": {
                        "models": {
                            "primary_unit": "trial",
                        },
                        "statistics": {"allow_iid_trials": False},
                    }
                }
            )
        )
        df_trials = pd.DataFrame(
            {
                "run_id": np.repeat(np.arange(1, 11), 2),
                "rating": np.linspace(1.0, 5.0, 20),
                "temperature": np.linspace(40.0, 48.0, 20),
                "power_active_alpha_ch_Fp1_mean": np.linspace(0.1, 2.0, 20),
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_active_alpha_ch_Fp1_mean"],
        ):
            with self.assertRaises(ValueError):
                stage_models(ctx, SimpleNamespace(method_label=""))

    def test_pain_sensitivity_permutation_uses_grouped_indices(self):
        from eeg_pipeline.utils.analysis.stats.correlation import run_pain_sensitivity_correlations

        features_df = pd.DataFrame(
            {
                "f1": [1.0, 3.0, 4.0, 6.0, 9.0, 11.0, 14.0, 18.0],
            }
        )
        ratings = pd.Series([1.0, 4.0, 3.0, 6.0, 8.0, 9.0, 11.0, 13.0])
        temperatures = pd.Series(np.linspace(40.0, 48.0, 8))
        groups = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        captured = {"groups": None, "calls": 0}

        def _capture_permute(n, rng, groups_arg=None, **kwargs):
            captured["groups"] = np.array(groups_arg) if groups_arg is not None else None
            captured["calls"] += 1
            return np.arange(n)

        with patch(
            "eeg_pipeline.utils.analysis.stats.permutation.permute_within_groups",
            side_effect=_capture_permute,
        ):
            out = run_pain_sensitivity_correlations(
                features_df=features_df,
                ratings=ratings,
                temperatures=temperatures,
                method="spearman",
                min_samples=3,
                n_perm=5,
                groups=groups,
                permutation_scheme="shuffle",
                p_primary_mode="perm_if_available",
            )

        self.assertFalse(out.empty)
        self.assertGreater(captured["calls"], 0)
        self.assertIsNotNone(captured["groups"])
        self.assertEqual(captured["groups"].tolist(), groups.tolist())
        self.assertIn("p_perm", out.columns)


if __name__ == "__main__":
    unittest.main()
