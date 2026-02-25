import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from scipy import stats

from tests.pipelines_test_utils import DotConfig


class TestBehaviorValidityFixes(unittest.TestCase):
    def _ctx(self, config: DotConfig) -> SimpleNamespace:
        return SimpleNamespace(
            subject="0001",
            task="task",
            config=config,
            logger=Mock(),
            deriv_root=Path(tempfile.mkdtemp()),
            stats_dir=Path(tempfile.mkdtemp()),
            overwrite=True,
            also_save_csv=False,
            selected_feature_files=None,
            feature_categories=None,
            computation_features=None,
            selected_bands=None,
            group_ids=None,
            data_qc={},
            rng=np.random.default_rng(0),
        )

    def test_stage_registry_includes_validation_and_qc(self):
        from eeg_pipeline.analysis.behavior.orchestration import StageRegistry

        self.assertIsNotNone(StageRegistry.get("feature_qc"))
        self.assertIsNotNone(StageRegistry.get("hierarchical_fdr_summary"))

        primary_spec = StageRegistry.get("correlate_primary_selection")
        self.assertIsNotNone(primary_spec)
        self.assertIn(StageRegistry.RESOURCE_EFFECT_SIZES, primary_spec.requires)
        self.assertNotIn(StageRegistry.RESOURCE_PVALUES, primary_spec.requires)

    def test_lag_and_residual_stages_update_trial_table_cache(self):
        import pandas as pd

        from eeg_pipeline.analysis.behavior import orchestration as orch

        ctx = self._ctx(
            DotConfig(
                {
                    "behavior_analysis": {
                        "run_adjustment": {"column": "run_id"},
                        "predictor_residual": {"enabled": True, "crossfit": {"enabled": False}},
                    }
                }
            )
        )

        base_df = pd.DataFrame(
            {
                "epoch": [0, 1, 2, 3],
                "run_id": [1, 1, 2, 2],
                "temperature": [46.0, 46.0, 48.0, 48.0],
                "rating": [10.0, 20.0, 30.0, 40.0],
                "power_alpha": [0.1, 0.2, 0.3, 0.4],
            }
        )

        runtime = orch.create_behavior_runtime()
        setattr(ctx, "_behavior_runtime", runtime)
        runtime.cache._trial_table_df = base_df

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._write_parquet_with_optional_csv",
            return_value=None,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._write_metadata_file",
            return_value=None,
        ):
            orch.stage_lag_features(ctx, SimpleNamespace())
            df_after_lags = orch._load_trial_table_df(ctx)
            self.assertIn("trial_index_within_group", df_after_lags.columns)

            orch.stage_predictor_residual(ctx, SimpleNamespace())
            df_after_resid = orch._load_trial_table_df(ctx)
            self.assertIn("predictor_residual", df_after_resid.columns)

    def test_correlate_design_does_not_auto_resolve_optional_enrichments(self):
        from eeg_pipeline.analysis.behavior.orchestration import run_selected_stages

        ctx = self._ctx(DotConfig({}))
        plan = run_selected_stages(
            ctx=ctx,
            config=SimpleNamespace(),
            selected_stages=["correlate_design"],
            dry_run=True,
        )
        # Lag and residual enrichment stages are optional and only included
        # when explicitly requested by config/selection.
        self.assertNotIn("lag_features", plan["resolved"])
        self.assertNotIn("predictor_residual", plan["resolved"])
    def test_correlate_pvalues_not_reintroduced_when_disabled(self):
        from eeg_pipeline.analysis.behavior.orchestration import run_selected_stages

        ctx = self._ctx(
            DotConfig(
                {
                    "behavior_analysis": {
                        "correlations": {"permutation": {"enabled": False}},
                    }
                }
            )
        )
        plan = run_selected_stages(
            ctx=ctx,
            config=SimpleNamespace(),
            selected_stages=[
                "correlate_design",
                "correlate_effect_sizes",
                "correlate_pvalues",
                "correlate_primary_selection",
            ],
            dry_run=True,
        )
        self.assertNotIn("correlate_pvalues", plan["resolved"])
        self.assertIn("correlate_primary_selection", plan["resolved"])

    def test_resolve_condition_compare_column_uses_pain_mapping(self):
        from eeg_pipeline.analysis.behavior.orchestration import _resolve_condition_compare_column

        cfg = DotConfig(
            {
                "behavior_analysis": {"condition": {"compare_column": ""}},
                "event_columns": {"binary_outcome": ["binary_outcome"]},
            }
        )
        df_trials = pd.DataFrame({"binary_outcome": [0, 1]})
        self.assertEqual(_resolve_condition_compare_column(df_trials, cfg), "binary_outcome")

    def test_condition_run_mean_aggregation_and_min_samples_forwarded(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_condition_column

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "condition": {
                        "compare_column": "",
                        "primary_unit": "run_mean",
                        "compare_values": [],
                        "permutation": {"enabled": True},
                    },
                    "run_adjustment": {"column": "run_id"},
                },
                "event_columns": {"binary_outcome": ["binary_outcome"]},
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "run_id": [1, 1, 2, 2],
                "binary_outcome": [1, 1, 0, 0],
                "power_alpha": [0.2, 0.4, 0.8, 1.0],
            }
        )
        captured = {}

        def _fake_effects(features_df, pain_mask, nonpain_mask, min_samples, **kwargs):
            captured["n_rows"] = len(features_df)
            captured["min_samples"] = min_samples
            captured["paired"] = bool(kwargs.get("paired", False))
            captured["pair_ids"] = kwargs.get("pair_ids", None)
            return pd.DataFrame(
                {
                    "feature": ["power_alpha"],
                    "hedges_g": [0.5],
                    "p_value": [0.2],
                    "p_primary": [0.2],
                }
            )

        with patch(
            "eeg_pipeline.analysis.behavior.api.split_by_condition",
            return_value=(np.array([True, False]), np.array([False, True]), 1, 1),
        ), patch(
            "eeg_pipeline.analysis.behavior.api.compute_condition_effects",
            side_effect=_fake_effects,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._compute_unified_fdr",
            side_effect=lambda _ctx, _cfg, df, **_kw: df,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._write_parquet_with_optional_csv",
            return_value=None,
        ):
            out = stage_condition_column(
                ctx,
                SimpleNamespace(fdr_alpha=0.05, n_jobs=1, min_samples=7),
                df_trials=df_trials,
                feature_cols=["power_alpha"],
            )

        self.assertEqual(captured["n_rows"], 2)
        self.assertEqual(captured["min_samples"], 7)
        self.assertTrue(captured["paired"])
        self.assertIsNotNone(captured["pair_ids"])
        self.assertEqual(len(captured["pair_ids"]), 2)
        self.assertEqual(str(out["condition_column"].iloc[0]), "binary_outcome")

    def test_condition_run_level_keeps_run_condition_cells(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_condition_column

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "condition": {
                        "compare_column": "",
                        "primary_unit": "run_mean",
                        "compare_values": [],
                        "permutation": {"enabled": True},
                    },
                    "run_adjustment": {"column": "run_id"},
                },
                "event_columns": {"binary_outcome": ["binary_outcome"]},
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "run_id": [1, 1, 2, 2],
                "binary_outcome": [1, 0, 1, 0],
                "power_alpha": [0.2, 0.4, 0.8, 1.0],
            }
        )
        captured = {}

        def _fake_effects(features_df, pain_mask, nonpain_mask, min_samples, **kwargs):
            captured["n_rows"] = len(features_df)
            captured["paired"] = bool(kwargs.get("paired", False))
            return pd.DataFrame(
                {
                    "feature": ["power_alpha"],
                    "hedges_g": [0.5],
                    "p_value": [0.2],
                    "p_primary": [0.2],
                }
            )

        with patch(
            "eeg_pipeline.analysis.behavior.api.split_by_condition",
            return_value=(np.array([True, False, True, False]), np.array([False, True, False, True]), 2, 2),
        ), patch(
            "eeg_pipeline.analysis.behavior.api.compute_condition_effects",
            side_effect=_fake_effects,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._compute_unified_fdr",
            side_effect=lambda _ctx, _cfg, df, **_kw: df,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._write_parquet_with_optional_csv",
            return_value=None,
        ):
            stage_condition_column(
                ctx,
                SimpleNamespace(fdr_alpha=0.05, n_jobs=1, min_samples=2),
                df_trials=df_trials,
                feature_cols=["power_alpha"],
            )

        self.assertEqual(captured["n_rows"], 4)
        self.assertTrue(captured["paired"])

    def test_condition_column_non_iid_forces_strict_permutation_primary_mode(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_condition_column

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "condition": {
                        "compare_column": "binary_outcome",
                        "primary_unit": "trial",
                        "compare_values": [1, 0],
                        "permutation": {"enabled": True},
                        "p_primary_mode": "asymptotic",
                    },
                    "statistics": {"allow_iid_trials": False},
                    "run_adjustment": {"column": "run_id"},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "run_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                "binary_outcome": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                "power_alpha": [0.2, 0.4, 0.8, 1.0, 0.3, 0.5, 0.7, 1.1, 0.6, 0.9],
            }
        )
        captured = {}

        def _fake_effects(features_df, pain_mask, nonpain_mask, min_samples, **kwargs):
            captured["p_primary_mode"] = kwargs.get("p_primary_mode")
            return pd.DataFrame(
                {
                    "feature": ["power_alpha"],
                    "hedges_g": [0.5],
                    "p_value": [0.2],
                    "p_primary": [0.2],
                }
            )

        with patch(
            "eeg_pipeline.analysis.behavior.api.split_by_condition",
            return_value=(
                np.array([True, False, True, False, True, False, True, False, True, False]),
                np.array([False, True, False, True, False, True, False, True, False, True]),
                5,
                5,
            ),
        ), patch(
            "eeg_pipeline.analysis.behavior.api.compute_condition_effects",
            side_effect=_fake_effects,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._compute_unified_fdr",
            side_effect=lambda _ctx, _cfg, df, **_kw: df,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._write_parquet_with_optional_csv",
            return_value=None,
        ):
            out = stage_condition_column(
                ctx,
                SimpleNamespace(fdr_alpha=0.05, n_jobs=1, min_samples=2),
                df_trials=df_trials,
                feature_cols=["power_alpha"],
            )

        self.assertFalse(out.empty)
        self.assertEqual(str(captured.get("p_primary_mode")), "perm")

    def test_group_trial_table_discovery_finds_canonical_all_trials(self):
        from eeg_pipeline.analysis.behavior.orchestration import _find_trial_table_path

        root = Path(tempfile.mkdtemp())
        trials_path = root / "trial_table" / "all" / "trials_all.tsv"
        trials_path.parent.mkdir(parents=True, exist_ok=True)
        trials_path.write_text("rating\tpower_alpha\n1\t0.1\n")

        found = _find_trial_table_path(root, feature_files=None)
        self.assertEqual(found, trials_path)

    def test_group_trial_table_discovery_prefers_parquet_when_both_formats_exist(self):
        from eeg_pipeline.analysis.behavior.orchestration import _find_trial_table_path

        root = Path(tempfile.mkdtemp())
        parquet_path = root / "trial_table" / "power" / "trials_power.parquet"
        tsv_path = root / "trial_table" / "power" / "trials_power.tsv"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path.write_bytes(b"PAR1")
        tsv_path.write_text("rating\tpower\n1\t0.1\n")

        found = _find_trial_table_path(root, feature_files=None)
        self.assertEqual(found, parquet_path)

    def test_group_trial_table_discovery_uses_canonical_all_for_multiple_features(self):
        from eeg_pipeline.analysis.behavior.orchestration import _find_trial_table_path

        root = Path(tempfile.mkdtemp())
        canonical_path = root / "trial_table" / "all" / "trials_all.parquet"
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        canonical_path.write_bytes(b"PAR1")

        found = _find_trial_table_path(root, feature_files=["power", "connectivity"])
        self.assertEqual(found, canonical_path)

    def test_group_correlations_read_tsv_via_read_table(self):
        from eeg_pipeline.analysis.behavior.orchestration import run_group_level_correlations

        fake_trial_path = Path("/tmp/trials.tsv")
        cfg = DotConfig({})
        logger = Mock()

        def _fake_read_table(_path):
            return pd.DataFrame(
                {
                    "rating": np.linspace(10, 70, 12),
                    "power_alpha": np.linspace(0.1, 1.2, 12),
                    "run_id": np.repeat([1, 2, 3], 4),
                }
            )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._find_trial_table_path",
            return_value=fake_trial_path,
        ), patch(
            "eeg_pipeline.infra.paths.deriv_stats_path",
            side_effect=lambda _root, _sub: Path("/tmp"),
        ), patch(
            "eeg_pipeline.infra.tsv.read_table",
            side_effect=_fake_read_table,
        ) as read_table_mock, patch(
            "eeg_pipeline.analysis.behavior.orchestration.compute_correlation",
            return_value=(0.2, 0.5),
        ), patch(
            "eeg_pipeline.utils.analysis.stats.fdr.hierarchical_fdr",
            side_effect=lambda df, **_kwargs: df,
        ):
            out = run_group_level_correlations(
                subjects=["0001", "0002"],
                deriv_root=Path("/tmp"),
                config=cfg,
                logger=logger,
                use_block_permutation=False,
                n_perm=2,
            )

        self.assertFalse(out.empty)
        self.assertGreaterEqual(read_table_mock.call_count, 2)
        self.assertTrue(all(call.args[0].suffix == ".tsv" for call in read_table_mock.call_args_list))

    def test_group_correlations_use_within_subject_centering(self):
        from eeg_pipeline.analysis.behavior.orchestration import run_group_level_correlations

        rng = np.random.default_rng(0)
        n = 24
        run_ids = np.repeat([1, 2, 3], 8)
        df_a = pd.DataFrame(
            {
                "rating": 80 + rng.normal(scale=0.8, size=n),
                "power_alpha": 5 + rng.normal(scale=0.8, size=n),
                "run_id": run_ids,
            }
        )
        df_b = pd.DataFrame(
            {
                "rating": 20 + rng.normal(scale=0.8, size=n),
                "power_alpha": -5 + rng.normal(scale=0.8, size=n),
                "run_id": run_ids,
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._find_trial_table_path",
            return_value=Path("/tmp/trials.tsv"),
        ), patch(
            "eeg_pipeline.infra.paths.deriv_stats_path",
            side_effect=lambda _root, sub: Path(f"/tmp/{sub}"),
        ), patch(
            "eeg_pipeline.infra.tsv.read_table",
            side_effect=[df_a, df_b],
        ):
            out = run_group_level_correlations(
                subjects=["0001", "0002"],
                deriv_root=Path("/tmp"),
                config=DotConfig({}),
                logger=Mock(),
                use_block_permutation=False,
                n_perm=40,
            )

        self.assertFalse(out.empty)
        self.assertLess(abs(float(out.iloc[0]["r"])), 0.35)

    def test_group_mixed_effects_formula_includes_temperature_and_run_covariates(self):
        from eeg_pipeline.analysis.behavior.orchestration import run_group_level_mixed_effects

        n = 20
        base_df = pd.DataFrame(
            {
                "rating": np.linspace(10, 70, n),
                "power_alpha": np.linspace(0.1, 1.2, n),
                "temperature": np.linspace(42, 47, n),
                "trial_index": np.arange(n),
                "run_id": np.repeat([1, 2], n // 2),
            }
        )

        formulas = []

        class _FakeFit:
            fe_params = {"feature_value": 0.1}
            bse = {"feature_value": 0.05}
            tvalues = {"feature_value": 2.0}
            pvalues = {"feature_value": 0.04}
            aic = 1.0
            bic = 1.5
            converged = True

        class _FakeModel:
            def fit(self, reml=True):
                return _FakeFit()

        def _fake_mixedlm(formula, data, groups, re_formula=None):
            formulas.append(formula)
            return _FakeModel()

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._find_trial_table_path",
            return_value=Path("/tmp/trials.tsv"),
        ), patch(
            "eeg_pipeline.infra.paths.deriv_stats_path",
            side_effect=lambda _root, sub: Path(f"/tmp/{sub}"),
        ), patch(
            "eeg_pipeline.infra.tsv.read_table",
            side_effect=[base_df.copy(), base_df.copy()],
        ), patch(
            "statsmodels.formula.api.mixedlm",
            side_effect=_fake_mixedlm,
        ), patch(
            "eeg_pipeline.utils.analysis.stats.fdr.hierarchical_fdr",
            side_effect=lambda df, **_kw: df.assign(q_within_family=pd.to_numeric(df["fixed_p"], errors="coerce")),
        ):
            result = run_group_level_mixed_effects(
                subjects=["0001", "0002"],
                deriv_root=Path("/tmp"),
                config=DotConfig({}),
                logger=Mock(),
                max_features=1,
                fdr_alpha=0.05,
            )

        self.assertIsNotNone(result)
        self.assertFalse(result.df.empty)
        self.assertTrue(formulas)
        self.assertIn("temperature", formulas[0])
        self.assertIn("trial_index", formulas[0])
        self.assertIn("C(run_id)", formulas[0])

    def test_mediation_requires_non_iid_inference_when_iid_disallowed(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_mediation

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "mediation": {"n_permutations": 0},
                    "statistics": {"allow_iid_trials": False},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "temperature": np.linspace(40, 46, 12),
                "rating": np.linspace(10, 70, 12),
                "power_alpha": np.linspace(0.1, 1.2, 12),
                "run_id": np.repeat([1, 2, 3], 4),
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_alpha"],
        ):
            with self.assertRaises(ValueError):
                stage_mediation(ctx, SimpleNamespace(fdr_alpha=0.05))

    def test_moderation_requires_non_iid_inference_when_iid_disallowed(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_moderation

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "moderation": {"n_permutations": 0},
                    "statistics": {"allow_iid_trials": False},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "temperature": np.linspace(40, 46, 12),
                "rating": np.linspace(10, 70, 12),
                "power_alpha": np.linspace(0.1, 1.2, 12),
                "run_id": np.repeat([1, 2, 3], 4),
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_alpha"],
        ):
            with self.assertRaises(ValueError):
                stage_moderation(ctx, SimpleNamespace(fdr_alpha=0.05, moderation_max_features=None, method_label="", moderation_min_samples=5))

    def test_moderation_simple_slope_uses_covariance_term(self):
        from eeg_pipeline.utils.analysis.stats.moderation import compute_moderation_effect

        rng = np.random.default_rng(0)
        n = 300
        x = rng.normal(size=n)
        w = 0.6 * x + rng.normal(scale=0.8, size=n)
        y = 0.5 + 0.8 * x - 0.4 * w + 0.9 * (x * w) + rng.normal(scale=1.0, size=n)

        result = compute_moderation_effect(x, w, y, center_predictors=True)
        self.assertTrue(np.isfinite(result.cov_b1_b3))

        w_std = float(np.std(w))
        slope_high = result.b1 + result.b3 * w_std
        var_slope_high = result.var_b1 + (w_std**2) * result.var_b3 + 2 * w_std * result.cov_b1_b3
        self.assertGreater(var_slope_high, 0.0)

        dof = max(result.n - 4, 1)
        t_value = slope_high / np.sqrt(var_slope_high)
        expected_p = float(2 * (1 - stats.t.cdf(np.abs(t_value), dof)))
        self.assertAlmostEqual(float(result.p_slope_high), expected_p, places=7)

    def test_condition_effects_respect_min_samples(self):
        from eeg_pipeline.utils.analysis.stats.effect_size import compute_batch_condition_effects
        from eeg_pipeline.utils.parallel import _compute_single_condition_effect

        features = pd.DataFrame({"power_alpha": [0.1, 0.2, 0.3, 0.4]})
        pain_mask = np.array([True, True, False, False])
        nonpain_mask = ~pain_mask

        single = _compute_single_condition_effect(
            "power_alpha",
            features,
            pain_mask,
            nonpain_mask,
            min_samples=3,
            n_perm=0,
        )
        self.assertIsNone(single)

        batch = compute_batch_condition_effects(
            feature_columns=["power_alpha"],
            features_df=features,
            pain_mask=pain_mask,
            nonpain_mask=nonpain_mask,
            min_samples=3,
            n_perm=0,
        )
        self.assertEqual(batch, [])

    def test_condition_run_level_paired_effect_drops_unmatched_pairs(self):
        from eeg_pipeline.utils.parallel import _compute_single_condition_effect

        features = pd.DataFrame({"power_alpha": [10.0, 1.0, 9.0, 2.0, 8.0]})
        pain_mask = np.array([True, False, True, False, True], dtype=bool)
        nonpain_mask = ~pain_mask
        pair_ids = np.array([1, 1, 2, 2, 3], dtype=int)  # run 3 missing non-pain

        out = _compute_single_condition_effect(
            "power_alpha",
            features,
            pain_mask,
            nonpain_mask,
            min_samples=2,
            paired=True,
            pair_ids=pair_ids,
            n_perm=0,
        )

        self.assertIsNotNone(out)
        self.assertTrue(bool(out["paired_test"]))
        self.assertEqual(int(out["n_pairs"]), 2)
        self.assertEqual(int(out["n_condition_a"]), 2)
        self.assertEqual(int(out["n_condition_b"]), 2)

    def test_condition_primary_mode_perm_marks_missing_when_permutation_unavailable(self):
        from eeg_pipeline.utils.analysis.stats.effect_size import compute_condition_effects

        features = pd.DataFrame({"power_alpha": [0.1, 0.2, 0.8, 1.0]})
        pain_mask = np.array([True, True, False, False], dtype=bool)
        nonpain_mask = ~pain_mask
        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "condition": {
                        "p_primary_mode": "perm",
                        "permutation": {"enabled": False},
                    }
                }
            }
        )

        out = compute_condition_effects(
            features_df=features,
            pain_mask=pain_mask,
            nonpain_mask=nonpain_mask,
            min_samples=2,
            fdr_alpha=0.05,
            n_jobs=1,
            config=cfg,
            groups=None,
        )

        self.assertFalse(out.empty)
        self.assertTrue(np.isnan(float(out.iloc[0]["p_primary"])))
        self.assertEqual(str(out.iloc[0]["p_primary_source"]), "perm_missing_required")

    def test_permutation_groups_with_singletons_raise_by_default(self):
        from eeg_pipeline.utils.analysis.stats.permutation import permute_within_groups

        rng = np.random.default_rng(123)
        with self.assertRaises(ValueError):
            permute_within_groups(
                4,
                rng,
                groups=np.array([10, 11, 12, 13]),
                scheme="shuffle",
            )

    def test_permutation_groups_with_singletons_can_fall_back_when_explicit(self):
        from eeg_pipeline.utils.analysis.stats.permutation import permute_within_groups

        rng = np.random.default_rng(123)
        idx = permute_within_groups(
            4,
            rng,
            groups=np.array([10, 11, 12, 13]),
            scheme="shuffle",
            strict=False,
        )
        self.assertEqual(sorted(idx.tolist()), [0, 1, 2, 3])

    def test_predictor_sensitivity_perm_mode_marks_missing_when_permutation_unavailable(self):
        from eeg_pipeline.utils.analysis.stats.correlation import run_predictor_sensitivity_correlations

        out = run_predictor_sensitivity_correlations(
            features_df=pd.DataFrame({"power_alpha": [0.1, 0.2, 0.3, 0.4, 0.6, 0.7]}),
            outcomes=pd.Series([10, 20, 30, 40, 50, 60], dtype=float),
            predictors=pd.Series([44.1, 44.3, 44.6, 45.0, 45.3, 45.6], dtype=float),
            method="spearman",
            min_samples=5,
            n_perm=0,
            p_primary_mode="perm",
        )

        self.assertFalse(out.empty)
        self.assertTrue(np.isnan(float(out.iloc[0]["p_primary"])))
        self.assertEqual(str(out.iloc[0]["p_kind_primary"]), "perm_missing_required")
        self.assertEqual(str(out.iloc[0]["p_primary_source"]), "perm_missing_required")

    def test_behavior_pipeline_config_parses_run_validation_flag(self):
        from eeg_pipeline.pipelines.behavior import BehaviorPipelineConfig

        cfg = DotConfig({"behavior_analysis": {"validation": {"enabled": False}}})
        parsed = BehaviorPipelineConfig.from_config(cfg)
        self.assertFalse(parsed.run_validation)

    def test_behavior_pipeline_config_uses_canonical_correlation_method_key(self):
        from eeg_pipeline.pipelines.behavior import BehaviorPipelineConfig

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "correlation_method": "pearson",
                    "statistics": {"correlation_method": "spearman"},
                }
            }
        )
        parsed = BehaviorPipelineConfig.from_config(cfg)
        self.assertEqual(parsed.method, "spearman")

    def test_behavior_pipeline_config_rejects_unknown_robust_method(self):
        from eeg_pipeline.pipelines.behavior import BehaviorPipelineConfig

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "statistics": {"correlation_method": "spearman"},
                    "robust_correlation": "invalid_name",
                }
            }
        )
        with self.assertRaises(ValueError):
            BehaviorPipelineConfig.from_config(cfg)

    def test_correlate_design_accepts_canonical_targets_key(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_correlate_design

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "correlations": {
                        "targets": ["rating", "temperature"],
                        "permutation": {"enabled": True, "n_permutations": 20},
                    },
                    "statistics": {"allow_iid_trials": False},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "rating": np.linspace(10, 50, 8),
                "temperature": np.linspace(43, 46, 8),
                "predictor_residual": np.linspace(-1, 1, 8),
                "run_id": np.repeat([1, 2], 4),
                "power_alpha": np.linspace(0.1, 0.8, 8),
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_alpha"],
        ):
            design = stage_correlate_design(
                ctx,
                SimpleNamespace(control_temperature=True, control_trial_order=True),
            )
        self.assertIsNotNone(design)
        self.assertEqual(design.targets, ["rating", "temperature"])

    def test_correlate_design_with_no_explicit_targets_returns_none(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_correlate_design

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "correlations": {
                        "permutation": {"enabled": True, "n_permutations": 20},
                    },
                    "statistics": {"allow_iid_trials": False},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "rating": np.linspace(10, 50, 8),
                "temperature": np.linspace(43, 46, 8),
                "run_id": np.repeat([1, 2], 4),
                "power_alpha": np.linspace(0.1, 0.8, 8),
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_alpha"],
        ):
            design = stage_correlate_design(
                ctx,
                SimpleNamespace(control_temperature=True, control_trial_order=True),
            )
        self.assertIsNone(design)

    def test_correlate_design_requires_positive_permutation_count_in_non_iid_trial_mode(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_correlate_design

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "correlations": {
                        "permutation": {"enabled": True, "n_permutations": 0},
                    },
                    "statistics": {"allow_iid_trials": False},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "rating": np.linspace(10, 50, 8),
                "temperature": np.linspace(43, 46, 8),
                "run_id": np.repeat([1, 2], 4),
                "power_alpha": np.linspace(0.1, 0.8, 8),
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_alpha"],
        ):
            with self.assertRaises(ValueError):
                stage_correlate_design(
                    ctx,
                    SimpleNamespace(control_temperature=True, control_trial_order=True),
                )

    def test_correlate_design_uses_only_explicit_target_column(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_correlate_design

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "correlations": {
                        "target_column": "vas_custom",
                        "permutation": {"enabled": True, "n_permutations": 20},
                    },
                    "statistics": {"allow_iid_trials": False},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "rating": np.linspace(10, 50, 8),
                "temperature": np.linspace(43, 46, 8),
                "predictor_residual": np.linspace(-1, 1, 8),
                "vas_custom": np.linspace(0.2, 0.9, 8),
                "run_id": np.repeat([1, 2], 4),
                "power_alpha": np.linspace(0.1, 0.8, 8),
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_alpha"],
        ):
            design = stage_correlate_design(
                ctx,
                SimpleNamespace(control_temperature=True, control_trial_order=True),
            )
        self.assertIsNotNone(design)
        self.assertEqual(design.targets, ["vas_custom"])

    def test_correlate_design_prefers_crossfit_predictor_residual_when_available(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_correlate_design

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "correlations": {
                        "targets": ["rating", "temperature", "predictor_residual"],
                        "prefer_predictor_residual": True,
                        "use_crossfit_predictor_residual": True,
                        "permutation": {"enabled": True, "n_permutations": 20},
                    },
                    "statistics": {"allow_iid_trials": False},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "rating": np.linspace(10, 50, 8),
                "temperature": np.linspace(43, 46, 8),
                "predictor_residual": np.linspace(-1, 1, 8),
                "predictor_residual_cv": np.linspace(-1.2, 0.8, 8),
                "run_id": np.repeat([1, 2], 4),
                "power_alpha": np.linspace(0.1, 0.8, 8),
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_alpha"],
        ):
            design = stage_correlate_design(
                ctx,
                SimpleNamespace(control_temperature=True, control_trial_order=True),
            )
        self.assertIsNotNone(design)
        self.assertEqual(design.targets, ["predictor_residual_cv", "rating", "temperature"])

    def test_correlate_design_prefers_predictor_residual_first_when_enabled(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_correlate_design

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "correlations": {
                        "targets": ["rating", "temperature", "predictor_residual"],
                        "prefer_predictor_residual": True,
                        "use_crossfit_predictor_residual": False,
                        "permutation": {"enabled": True, "n_permutations": 20},
                    },
                    "statistics": {"allow_iid_trials": False},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "rating": np.linspace(10, 50, 8),
                "temperature": np.linspace(43, 46, 8),
                "predictor_residual": np.linspace(-1, 1, 8),
                "run_id": np.repeat([1, 2], 4),
                "power_alpha": np.linspace(0.1, 0.8, 8),
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_alpha"],
        ):
            design = stage_correlate_design(
                ctx,
                SimpleNamespace(control_temperature=True, control_trial_order=True),
            )
        self.assertIsNotNone(design)
        self.assertEqual(design.targets, ["predictor_residual", "rating", "temperature"])

    def test_trial_table_stage_reuses_cached_output_when_input_hash_matches(self):
        from eeg_pipeline.analysis.behavior.orchestration import (
            create_behavior_runtime,
            stage_trial_table,
        )
        from eeg_pipeline.utils.data.trial_table import compute_trial_table_schema_hash

        cfg = DotConfig({"behavior_analysis": {"trial_table": {"format": "tsv"}}})
        ctx = self._ctx(cfg)
        out_dir = ctx.stats_dir / "trial_table" / "all"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "trials_all.tsv"
        df_cached = pd.DataFrame({"rating": [10], "power_alpha": [0.1]})
        df_cached.to_csv(out_path, sep="\t", index=False)
        schema_hash = compute_trial_table_schema_hash(df_cached)
        (out_dir / "trials_all.metadata.json").write_text(
            '{"n_trials": 1, "n_columns": 2, "contract": {"version": "1.0", "schema_hash": "'
            + schema_hash
            + '", "input_hash": "abc"}}',
            encoding="utf-8",
        )

        runtime = create_behavior_runtime()
        setattr(ctx, "_behavior_runtime", runtime)
        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._compute_trial_table_input_hash",
            return_value="abc",
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration.compute_trial_table",
            side_effect=AssertionError("compute_trial_table should not run when cache key matches"),
        ):
            resolved = stage_trial_table(ctx, SimpleNamespace())
        self.assertEqual(resolved, out_path)

    def test_trial_table_contract_validation_rejects_schema_mismatch(self):
        from eeg_pipeline.analysis.behavior.orchestration import create_behavior_runtime

        cfg = DotConfig({"behavior_analysis": {"trial_table": {"format": "tsv"}}})
        ctx = self._ctx(cfg)
        out_dir = ctx.stats_dir / "trial_table" / "all"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trials_all.tsv").write_text("rating\tpower_alpha\n10\t0.1\n", encoding="utf-8")
        (out_dir / "trials_all.metadata.json").write_text(
            '{"n_trials": 1, "n_columns": 2, "contract": {"version": "1.0", "schema_hash": "definitely_wrong"}}',
            encoding="utf-8",
        )

        runtime = create_behavior_runtime()
        setattr(ctx, "_behavior_runtime", runtime)
        with self.assertRaises(ValueError):
            runtime.cache.get_trial_table(ctx)

    def test_write_trial_table_uses_canonical_all_name_for_multi_feature_selection(self):
        from eeg_pipeline.analysis.behavior.orchestration import TrialTableResult, write_trial_table

        cfg = DotConfig({"behavior_analysis": {"trial_table": {"format": "tsv"}}})
        ctx = self._ctx(cfg)
        ctx.selected_feature_files = ["power", "connectivity"]
        result = TrialTableResult(
            df=pd.DataFrame({"rating": [1.0], "power_alpha": [0.1], "connectivity_imcoh": [0.2]}),
            metadata={"n_trials": 1, "n_columns": 3, "contract": {"version": "1.0"}},
        )

        out_path = write_trial_table(ctx, result)
        self.assertEqual(out_path, ctx.stats_dir / "trial_table" / "all" / "trials_all.tsv")
        self.assertTrue(out_path.exists())

    def test_predictor_sensitivity_preserves_permutation_primary_pvalue(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_predictor_sensitivity

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "predictor_sensitivity": {
                        "n_permutations": 20,
                        "p_primary_mode": "perm_if_available",
                    },
                    "statistics": {"allow_iid_trials": False},
                    "run_adjustment": {"column": "run_id"},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "temperature": np.linspace(40, 46, 12),
                "rating": np.linspace(10, 70, 12),
                "run_id": np.repeat([1, 2, 3], 4),
                "power_alpha": np.linspace(0.1, 1.2, 12),
            }
        )
        psi_df = pd.DataFrame(
            {
                "feature": ["power_alpha"],
                "feature_type": ["power"],
                "band": ["alpha"],
                "p_psi": [0.8],
                "p_perm": [0.01],
                "p_primary": [0.01],
                "p_primary_source": ["psi_perm"],
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_alpha"],
        ), patch(
            "eeg_pipeline.analysis.behavior.api.run_predictor_sensitivity_correlations",
            return_value=psi_df.copy(),
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._compute_unified_fdr",
            side_effect=lambda _ctx, _cfg, df, **_kw: df,
        ):
            out = stage_predictor_sensitivity(ctx, SimpleNamespace(method="spearman", min_samples=5, fdr_alpha=0.05))

        self.assertFalse(out.empty)
        self.assertAlmostEqual(float(out.iloc[0]["p_primary"]), 0.01, places=12)
        self.assertEqual(str(out.iloc[0]["p_primary_source"]), "psi_perm")

    def test_predictor_sensitivity_non_iid_forces_strict_permutation_primary_mode(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_predictor_sensitivity

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "predictor_sensitivity": {
                        "n_permutations": 20,
                        "p_primary_mode": "perm_if_available",
                    },
                    "statistics": {"allow_iid_trials": False},
                    "run_adjustment": {"column": "run_id"},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "temperature": np.linspace(40, 46, 12),
                "rating": np.linspace(10, 70, 12),
                "run_id": np.repeat([1, 2, 3], 4),
                "power_alpha": np.linspace(0.1, 1.2, 12),
            }
        )
        captured = {}

        def _fake_run_predictor_sensitivity_correlations(**kwargs):
            captured["p_primary_mode"] = kwargs.get("p_primary_mode")
            return pd.DataFrame(
                {
                    "feature": ["power_alpha"],
                    "feature_type": ["power"],
                    "band": ["alpha"],
                    "p_psi": [0.8],
                    "p_perm": [0.01],
                    "p_primary": [0.01],
                    "p_primary_source": ["psi_perm"],
                }
            )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_alpha"],
        ), patch(
            "eeg_pipeline.analysis.behavior.api.run_predictor_sensitivity_correlations",
            side_effect=_fake_run_predictor_sensitivity_correlations,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._compute_unified_fdr",
            side_effect=lambda _ctx, _cfg, df, **_kw: df,
        ):
            out = stage_predictor_sensitivity(ctx, SimpleNamespace(method="spearman", min_samples=5, fdr_alpha=0.05))

        self.assertFalse(out.empty)
        self.assertEqual(str(captured.get("p_primary_mode")), "perm")

    def test_predictor_sensitivity_non_iid_overrides_asymptotic_primary_mode(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_predictor_sensitivity

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "predictor_sensitivity": {
                        "n_permutations": 20,
                        "p_primary_mode": "asymptotic",
                    },
                    "statistics": {"allow_iid_trials": False},
                    "run_adjustment": {"column": "run_id"},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "temperature": np.linspace(40, 46, 12),
                "rating": np.linspace(10, 70, 12),
                "run_id": np.repeat([1, 2, 3], 4),
                "power_alpha": np.linspace(0.1, 1.2, 12),
            }
        )
        captured = {}

        def _fake_run_predictor_sensitivity_correlations(**kwargs):
            captured["p_primary_mode"] = kwargs.get("p_primary_mode")
            return pd.DataFrame(
                {
                    "feature": ["power_alpha"],
                    "feature_type": ["power"],
                    "band": ["alpha"],
                    "p_psi": [0.8],
                    "p_perm": [0.01],
                    "p_primary": [0.01],
                    "p_primary_source": ["psi_perm"],
                }
            )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_alpha"],
        ), patch(
            "eeg_pipeline.analysis.behavior.api.run_predictor_sensitivity_correlations",
            side_effect=_fake_run_predictor_sensitivity_correlations,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._compute_unified_fdr",
            side_effect=lambda _ctx, _cfg, df, **_kw: df,
        ):
            out = stage_predictor_sensitivity(ctx, SimpleNamespace(method="spearman", min_samples=5, fdr_alpha=0.05))

        self.assertFalse(out.empty)
        self.assertEqual(str(captured.get("p_primary_mode")), "perm")

    def test_predictor_sensitivity_rejects_unknown_robust_method(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_predictor_sensitivity

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "robust_correlation": "not_a_method",
                    "predictor_sensitivity": {
                        "n_permutations": 20,
                        "p_primary_mode": "perm_if_available",
                    },
                    "statistics": {"allow_iid_trials": False},
                    "run_adjustment": {"column": "run_id"},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "temperature": np.linspace(40, 46, 12),
                "rating": np.linspace(10, 70, 12),
                "run_id": np.repeat([1, 2, 3], 4),
                "power_alpha": np.linspace(0.1, 1.2, 12),
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_alpha"],
        ):
            with self.assertRaises(ValueError):
                stage_predictor_sensitivity(ctx, SimpleNamespace(method="spearman", min_samples=5, fdr_alpha=0.05))

    def test_stage_condition_multigroup_run_mean_uses_paired_ids(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_condition_multigroup

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "condition": {
                        "compare_column": "condition",
                        "compare_values": [0, 1, 2],
                        "primary_unit": "run_mean",
                        "overwrite": True,
                    },
                    "run_adjustment": {"column": "run_id"},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "run_id": [1, 1, 1, 2, 2, 2],
                "condition": [0, 1, 2, 0, 1, 2],
                "power_alpha": [0.2, 0.3, 0.4, 0.5, 0.7, 0.8],
            }
        )
        captured = {}

        def _fake_multigroup(features_df, group_masks, group_labels, **kwargs):
            captured["paired"] = bool(kwargs.get("paired", False))
            captured["pair_ids"] = kwargs.get("pair_ids")
            return pd.DataFrame(
                {
                    "feature": ["power_alpha"],
                    "group1": ["0"],
                    "group2": ["1"],
                    "n1": [2],
                    "n2": [2],
                    "mean1": [0.35],
                    "mean2": [0.50],
                    "cohens_d": [0.9],
                    "hedges_g": [0.8],
                    "p_value": [0.2],
                    "q_value": [0.2],
                    "significant_fdr": [False],
                }
            )

        with patch(
            "eeg_pipeline.utils.analysis.stats.effect_size.compute_multigroup_condition_effects",
            side_effect=_fake_multigroup,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._write_parquet_with_optional_csv",
            return_value=None,
        ):
            out = stage_condition_multigroup(
                ctx,
                SimpleNamespace(fdr_alpha=0.05),
                df_trials=df_trials,
                feature_cols=["power_alpha"],
            )

        self.assertFalse(out.empty)
        self.assertTrue(captured["paired"])
        self.assertEqual(len(captured["pair_ids"]), 6)

    def test_temporal_cluster_mode_requires_cluster_pvalues(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_temporal_stats

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "temporal": {"correction_method": "cluster"},
                    "statistics": {"fdr_alpha": 0.05, "allow_iid_trials": True},
                }
            }
        )
        ctx = self._ctx(cfg)
        ctx.use_spearman = True
        ctx.selected_feature_files = ["power"]

        with patch(
            "eeg_pipeline.analysis.behavior.api.compute_temporal_from_context",
            return_value={
                "records": [
                    {
                        "feature": "power",
                        "channel": "Cz",
                        "p_raw": 0.01,
                        "r": 0.3,
                        "n": 12,
                    }
                ]
            },
        ):
            with self.assertRaisesRegex(ValueError, "p_cluster"):
                stage_temporal_stats(ctx)

    def test_temporal_stats_respects_temporal_feature_toggles(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_temporal_stats

        aligned_events = pd.DataFrame(
            {
                "run_id": [1, 1, 2, 2],
                "sample_idx": [0, 1, 2, 3],
            }
        )

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "temporal": {
                        "features": {
                            "power": False,
                            "itpc": True,
                            "erds": False,
                        },
                        "correction_method": "fdr",
                    },
                    "statistics": {"fdr_alpha": 0.05, "allow_iid_trials": True},
                }
            }
        )
        ctx = self._ctx(cfg)
        ctx.use_spearman = True
        ctx.selected_feature_files = None
        ctx.feature_categories = None
        ctx.aligned_events = aligned_events

        with patch(
            "eeg_pipeline.analysis.behavior.api.compute_temporal_from_context",
            return_value={
                "records": [
                    {"feature": "power", "channel": "Cz", "p_raw": 0.01, "r": 0.2, "n": 12}
                ]
            },
        ) as power_mock, patch(
            "eeg_pipeline.utils.analysis.stats.temporal.compute_itpc_temporal_from_context",
            return_value={
                "records": [
                    {"feature": "itpc", "channel": "Cz", "p_raw": 0.02, "r": 0.3, "n": 12}
                ]
            },
        ) as itpc_mock, patch(
            "eeg_pipeline.analysis.behavior.orchestration._write_stats_table",
            return_value=None,
        ):
            out = stage_temporal_stats(ctx)

        self.assertIsNotNone(out)
        self.assertIsNone(out["power"])
        self.assertIsNotNone(out["itpc"])
        power_mock.assert_not_called()
        itpc_mock.assert_called_once()

    def test_correlate_primary_selection_does_not_fallback_to_raw_when_controlled_stat_missing(self):
        from eeg_pipeline.analysis.behavior.orchestration import CorrelateDesign, stage_correlate_primary_selection

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "statistics": {"allow_iid_trials": False},
                    "correlations": {
                        "p_primary_mode": "perm_if_available",
                    }
                }
            }
        )
        ctx = self._ctx(cfg)
        design = CorrelateDesign(
            df_trials=pd.DataFrame({"run_id": [1, 1]}),
            feature_cols=["power_alpha"],
            targets=["rating"],
            cov_df=pd.DataFrame({"trial_index": [0.0, 1.0]}),
            temperature_series=pd.Series([45.0, 46.0]),
            predictor_column="temperature",
            run_col="run_id",
            run_adjust_in_correlations=False,
            groups_for_perm=None,
        )
        records = [
            {
                "feature": "power_alpha",
                "target": "rating",
                "r_raw": 0.5,
                "p_raw": 0.01,
                "r_partial_cov_predictor": np.nan,
                "p_partial_cov_predictor": np.nan,
                "p_perm_partial_cov_predictor": np.nan,
                "robust_method": None,
            }
        ]

        out = stage_correlate_primary_selection(
            ctx,
            SimpleNamespace(control_temperature=True, control_trial_order=True),
            design,
            records,
        )

        self.assertEqual(len(out), 1)
        self.assertTrue(np.isnan(float(out[0]["p_primary"])))
        self.assertEqual(str(out[0]["p_kind_primary"]), "perm_missing_required")
        self.assertEqual(str(out[0]["p_primary_source"]), "perm_missing_required")

    def test_correlate_primary_selection_uses_robust_permutation_when_available(self):
        from eeg_pipeline.analysis.behavior.orchestration import CorrelateDesign, stage_correlate_primary_selection

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "correlations": {"p_primary_mode": "perm_if_available"},
                }
            }
        )
        ctx = self._ctx(cfg)
        design = CorrelateDesign(
            df_trials=pd.DataFrame({"run_id": [1, 1, 2, 2]}),
            feature_cols=["power_alpha"],
            targets=["rating"],
            cov_df=None,
            temperature_series=None,
            predictor_column="temperature",
            run_col="run_id",
            run_adjust_in_correlations=False,
            groups_for_perm=None,
        )
        records = [
            {
                "feature": "power_alpha",
                "target": "rating",
                "r_raw": 0.4,
                "p_raw": 0.03,
                "p_perm_raw": 0.01,
                "robust_method": "winsorized",
            }
        ]

        out = stage_correlate_primary_selection(
            ctx,
            SimpleNamespace(control_temperature=True, control_trial_order=True),
            design,
            records,
        )
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out[0]["p_primary"]), 0.01, places=12)
        self.assertEqual(str(out[0]["p_kind_primary"]), "p_perm_raw")
        self.assertEqual(str(out[0]["p_primary_source"]), "raw_robust_perm")

    def test_correlate_primary_selection_run_mean_precedes_robust_branch(self):
        from eeg_pipeline.analysis.behavior.orchestration import CorrelateDesign, stage_correlate_primary_selection

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "correlations": {
                        "primary_unit": "run_mean",
                        "p_primary_mode": "perm_if_available",
                    }
                }
            }
        )
        ctx = self._ctx(cfg)
        design = CorrelateDesign(
            df_trials=pd.DataFrame({"run_id": [1, 1, 2, 2]}),
            feature_cols=["power_alpha"],
            targets=["rating"],
            cov_df=None,
            temperature_series=None,
            predictor_column="temperature",
            run_col="run_id",
            run_adjust_in_correlations=False,
            groups_for_perm=None,
        )
        records = [
            {
                "feature": "power_alpha",
                "target": "rating",
                "r_raw": 0.1,
                "p_raw": 0.20,
                "r_run_mean": 0.8,
                "p_run_mean": 0.01,
                "robust_method": "winsorized",
            }
        ]

        out = stage_correlate_primary_selection(
            ctx,
            SimpleNamespace(control_temperature=True, control_trial_order=True),
            design,
            records,
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(str(out[0]["p_kind_primary"]), "p_run_mean")
        self.assertAlmostEqual(float(out[0]["p_primary"]), 0.01, places=12)
        self.assertEqual(str(out[0]["p_primary_source"]), "run_mean")

    def test_correlate_primary_selection_non_iid_overrides_asymptotic_to_permutation(self):
        from eeg_pipeline.analysis.behavior.orchestration import CorrelateDesign, stage_correlate_primary_selection

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "statistics": {"allow_iid_trials": False},
                    "correlations": {"p_primary_mode": "asymptotic"},
                }
            }
        )
        ctx = self._ctx(cfg)
        design = CorrelateDesign(
            df_trials=pd.DataFrame({"run_id": [1, 1, 2, 2]}),
            feature_cols=["power_alpha"],
            targets=["rating"],
            cov_df=None,
            temperature_series=None,
            predictor_column="temperature",
            run_col="run_id",
            run_adjust_in_correlations=False,
            groups_for_perm=None,
        )
        records = [
            {
                "feature": "power_alpha",
                "target": "rating",
                "r_raw": 0.3,
                "p_raw": 0.20,
                "p_perm_raw": 0.01,
                "robust_method": None,
            }
        ]

        out = stage_correlate_primary_selection(
            ctx,
            SimpleNamespace(control_temperature=False, control_trial_order=False),
            design,
            records,
        )

        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out[0]["p_primary"]), 0.01, places=12)
        self.assertEqual(str(out[0]["p_kind_primary"]), "p_perm_raw")

    def test_correlate_effect_sizes_rejects_robust_mode_with_covariate_controls(self):
        from eeg_pipeline.analysis.behavior.orchestration import CorrelateDesign, stage_correlate_effect_sizes

        ctx = self._ctx(DotConfig({}))
        design = CorrelateDesign(
            df_trials=pd.DataFrame(
                {
                    "run_id": [1, 1, 2, 2],
                    "power_alpha": [0.2, 0.3, 0.4, 0.5],
                    "rating": [10.0, 20.0, 30.0, 40.0],
                }
            ),
            feature_cols=["power_alpha"],
            targets=["rating"],
            cov_df=pd.DataFrame({"trial_index": [0.0, 1.0, 0.0, 1.0]}),
            temperature_series=pd.Series([44.0, 44.5, 45.0, 45.5]),
            predictor_column="temperature",
            run_col="run_id",
            run_adjust_in_correlations=False,
            groups_for_perm=None,
        )

        with self.assertRaises(ValueError):
            stage_correlate_effect_sizes(
                ctx,
                SimpleNamespace(
                    method="spearman",
                    robust_method="winsorized",
                    method_label="spearman_winsorized",
                    min_samples=2,
                ),
                design,
            )

    def test_correlate_effect_size_run_mean_respects_min_runs_threshold(self):
        from eeg_pipeline.analysis.behavior.orchestration import _compute_single_effect_size

        df_trials = pd.DataFrame(
            {
                "run_id": np.repeat([1, 2, 3, 4], 2),
                "power_alpha": np.linspace(0.1, 1.6, 8),
                "rating": np.linspace(10.0, 80.0, 8),
            }
        )

        rec = _compute_single_effect_size(
            feat="power_alpha",
            target="rating",
            df_trials=df_trials,
            cov_df=None,
            temperature_series=None,
            predictor_column="temperature",
            run_col="run_id",
            run_adjust_in_correlations=False,
            method="spearman",
            robust_method=None,
            method_label="spearman",
            min_samples=2,
            run_min_samples=5,
            want_raw=True,
            want_partial_cov=False,
            want_partial_predictor=False,
            want_partial_cov_predictor=False,
            want_run_mean=True,
            config=DotConfig({}),
        )

        self.assertEqual(int(rec.get("n_runs", 0)), 4)
        self.assertTrue(np.isnan(float(rec.get("p_run_mean", np.nan))))

    def test_group_correlations_use_subject_balanced_estimator_under_trial_imbalance(self):
        from eeg_pipeline.analysis.behavior.orchestration import run_group_level_correlations

        n_a = 120
        n_b = 12
        df_a = pd.DataFrame(
            {
                "rating": np.arange(n_a, dtype=float),
                "power_alpha": np.arange(n_a, dtype=float),
                "run_id": np.repeat(np.arange(1, 7), n_a // 6),
            }
        )
        df_b = pd.DataFrame(
            {
                "rating": np.arange(n_b, dtype=float),
                "power_alpha": -np.arange(n_b, dtype=float),
                "run_id": np.repeat(np.arange(1, 4), n_b // 3),
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._find_trial_table_path",
            return_value=Path("/tmp/trials.tsv"),
        ), patch(
            "eeg_pipeline.infra.paths.deriv_stats_path",
            side_effect=lambda _root, sub: Path(f"/tmp/{sub}"),
        ), patch(
            "eeg_pipeline.infra.tsv.read_table",
            side_effect=[df_a, df_b],
        ), patch(
            "eeg_pipeline.utils.analysis.stats.fdr.hierarchical_fdr",
            side_effect=lambda df, **_kwargs: df,
        ):
            out = run_group_level_correlations(
                subjects=["0001", "0002"],
                deriv_root=Path("/tmp"),
                config=DotConfig({}),
                logger=Mock(),
                use_block_permutation=False,
                n_perm=0,
            )

        self.assertFalse(out.empty)
        self.assertLess(abs(float(out.iloc[0]["r"])), 0.3)

    def test_group_correlations_can_partial_temperature(self):
        from eeg_pipeline.analysis.behavior.orchestration import run_group_level_correlations

        rng = np.random.default_rng(7)
        n = 30
        run_ids = np.repeat([1, 2, 3], 10)
        temp_a = np.linspace(41, 47, n)
        temp_b = np.linspace(42, 48, n)
        noise_scale = 0.3

        df_a = pd.DataFrame(
            {
                "rating": temp_a + rng.normal(scale=noise_scale, size=n),
                "temperature": temp_a,
                "power_alpha": temp_a + rng.normal(scale=noise_scale, size=n),
                "run_id": run_ids,
            }
        )
        df_b = pd.DataFrame(
            {
                "rating": temp_b + rng.normal(scale=noise_scale, size=n),
                "temperature": temp_b,
                "power_alpha": temp_b + rng.normal(scale=noise_scale, size=n),
                "run_id": run_ids,
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._find_trial_table_path",
            return_value=Path("/tmp/trials.tsv"),
        ), patch(
            "eeg_pipeline.infra.paths.deriv_stats_path",
            side_effect=lambda _root, sub: Path(f"/tmp/{sub}"),
        ), patch(
            "eeg_pipeline.infra.tsv.read_table",
            side_effect=[df_a, df_b],
        ), patch(
            "eeg_pipeline.utils.analysis.stats.fdr.hierarchical_fdr",
            side_effect=lambda df, **_kwargs: df,
        ):
            out = run_group_level_correlations(
                subjects=["0001", "0002"],
                deriv_root=Path("/tmp"),
                config=DotConfig({}),
                logger=Mock(),
                use_block_permutation=False,
                n_perm=0,
                target_col="rating",
                control_temperature=True,
                control_trial_order=False,
                control_run_effects=False,
            )

        self.assertFalse(out.empty)
        self.assertLess(abs(float(out.iloc[0]["r"])), 0.25)
        self.assertIn("partial", str(out.iloc[0]["estimator"]))

    def test_group_correlations_reports_effective_permutation_count(self):
        from eeg_pipeline.analysis.behavior.orchestration import run_group_level_correlations

        n = 24
        run_ids = np.repeat([1, 2, 3], 8)
        df_a = pd.DataFrame(
            {
                "rating": np.linspace(10, 70, n),
                "power_alpha": np.linspace(0.1, 1.2, n),
                "run_id": run_ids,
            }
        )
        df_b = pd.DataFrame(
            {
                "rating": np.linspace(12, 72, n),
                "power_alpha": np.linspace(0.2, 1.3, n),
                "run_id": run_ids,
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._find_trial_table_path",
            return_value=Path("/tmp/trials.tsv"),
        ), patch(
            "eeg_pipeline.infra.paths.deriv_stats_path",
            side_effect=lambda _root, sub: Path(f"/tmp/{sub}"),
        ), patch(
            "eeg_pipeline.infra.tsv.read_table",
            side_effect=[df_a, df_b],
        ), patch(
            "eeg_pipeline.utils.analysis.stats.fdr.hierarchical_fdr",
            side_effect=lambda df, **_kwargs: df,
        ):
            out = run_group_level_correlations(
                subjects=["0001", "0002"],
                deriv_root=Path("/tmp"),
                config=DotConfig({}),
                logger=Mock(),
                use_block_permutation=False,
                n_perm=15,
            )

        self.assertFalse(out.empty)
        self.assertIn("n_perm_requested", out.columns)
        self.assertIn("n_perm_effective", out.columns)
        self.assertEqual(int(out.iloc[0]["n_perm_requested"]), 15)
        self.assertEqual(int(out.iloc[0]["n_perm"]), int(out.iloc[0]["n_perm_effective"]))
        self.assertGreaterEqual(int(out.iloc[0]["n_perm_effective"]), 0)
        self.assertLessEqual(int(out.iloc[0]["n_perm_effective"]), 15)

    def test_group_correlations_use_project_random_state(self):
        from eeg_pipeline.analysis.behavior.orchestration import run_group_level_correlations

        n = 12
        df = pd.DataFrame(
            {
                "rating": np.linspace(10, 70, n),
                "power_alpha": np.linspace(0.1, 1.2, n),
                "run_id": np.repeat([1, 2, 3], 4),
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._find_trial_table_path",
            return_value=Path("/tmp/trials.tsv"),
        ), patch(
            "eeg_pipeline.infra.paths.deriv_stats_path",
            side_effect=lambda _root, _sub: Path("/tmp"),
        ), patch(
            "eeg_pipeline.infra.tsv.read_table",
            side_effect=[df, df],
        ), patch(
            "eeg_pipeline.utils.analysis.stats.fdr.hierarchical_fdr",
            side_effect=lambda df, **_kwargs: df,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration.np.random.default_rng",
            wraps=np.random.default_rng,
        ) as rng_factory:
            run_group_level_correlations(
                subjects=["0001", "0002"],
                deriv_root=Path("/tmp"),
                config=DotConfig({"project": {"random_state": 13}}),
                logger=Mock(),
                use_block_permutation=False,
                n_perm=2,
            )

        self.assertTrue(rng_factory.called)
        self.assertEqual(int(rng_factory.call_args.args[0]), 13)

    def test_group_correlations_respect_configured_correlation_method(self):
        from eeg_pipeline.analysis.behavior.orchestration import run_group_level_correlations

        n = 12
        df = pd.DataFrame(
            {
                "rating": np.linspace(10, 70, n),
                "power_alpha": np.linspace(0.1, 1.2, n),
                "run_id": np.repeat([1, 2, 3], 4),
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._find_trial_table_path",
            return_value=Path("/tmp/trials.tsv"),
        ), patch(
            "eeg_pipeline.infra.paths.deriv_stats_path",
            side_effect=lambda _root, _sub: Path("/tmp"),
        ), patch(
            "eeg_pipeline.infra.tsv.read_table",
            side_effect=[df.copy(), df.copy()],
        ), patch(
            "eeg_pipeline.utils.analysis.stats.fdr.hierarchical_fdr",
            side_effect=lambda df_in, **_kwargs: df_in,
        ):
            out = run_group_level_correlations(
                subjects=["0001", "0002"],
                deriv_root=Path("/tmp"),
                config=DotConfig({"behavior_analysis": {"statistics": {"correlation_method": "pearson"}}}),
                logger=Mock(),
                use_block_permutation=False,
                n_perm=0,
            )

        self.assertFalse(out.empty)
        self.assertIn("pearson", str(out.iloc[0]["estimator"]))

    def test_unified_fdr_applies_family_gate_before_within_family_significance(self):
        from eeg_pipeline.analysis.behavior.orchestration import _compute_unified_fdr

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "statistics": {"hierarchical_fdr": True},
                }
            }
        )
        ctx = self._ctx(cfg)

        records = []
        # Five strongly significant families
        for family_idx in range(5):
            records.append(
                {
                    "feature": f"f_strong_{family_idx}",
                    "feature_type": f"strong_{family_idx}",
                    "band": "alpha",
                    "target": "rating",
                    "analysis_kind": "correlation",
                    "p_primary": 1e-4,
                }
            )
        # One moderate family: within-family BH significant at 0.05, but should fail family gate
        records.extend(
            [
                {
                    "feature": "f_moderate_1",
                    "feature_type": "moderate",
                    "band": "alpha",
                    "target": "rating",
                    "analysis_kind": "correlation",
                    "p_primary": 0.020,
                },
                {
                    "feature": "f_moderate_2",
                    "feature_type": "moderate",
                    "band": "alpha",
                    "target": "rating",
                    "analysis_kind": "correlation",
                    "p_primary": 0.021,
                },
            ]
        )
        # Many weak families to make the moderate family fail family-level BH
        for family_idx in range(7, 26):
            records.append(
                {
                    "feature": f"f_weak_{family_idx}",
                    "feature_type": f"weak_{family_idx}",
                    "band": "alpha",
                    "target": "rating",
                    "analysis_kind": "correlation",
                    "p_primary": 0.9,
                }
            )

        df = pd.DataFrame(records)
        out = _compute_unified_fdr(
            ctx,
            SimpleNamespace(fdr_alpha=0.05),
            df,
            p_col="p_primary",
            family_cols=["feature_type", "band", "target", "analysis_kind"],
            analysis_type="correlations",
        )

        moderate = out[out["feature_type"] == "moderate"].reset_index(drop=True)
        self.assertFalse(moderate.empty)
        self.assertTrue((pd.to_numeric(moderate["q_within_family"], errors="coerce") < 0.05).all())
        self.assertFalse(bool(moderate["family_reject_gate"].iloc[0]))
        self.assertTrue((pd.to_numeric(moderate["p_fdr"], errors="coerce") >= 0.05).all())

    def test_regression_permutation_uses_feature_valid_subset_length(self):
        from eeg_pipeline.utils.analysis.stats.trialwise_regression import _compute_permutation_pvalues

        calls = []

        def _perm(n, rng, groups):
            calls.append(int(n))
            return np.arange(n, dtype=int)

        X = np.array([[1.0, 0.2], [1.0, -0.2]], dtype=float)
        y_f = np.array([0.1, -0.1], dtype=float)
        beta = np.array([0.0, 0.5], dtype=float)
        y_hat_z = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
        resid_z = np.array([0.2, -0.1, 0.3, -0.2], dtype=float)
        valid_feat = np.array([True, False, True, False], dtype=bool)
        groups_v = np.array([1, 1, 2, 2], dtype=int)

        with patch(
            "eeg_pipeline.utils.analysis.stats.trialwise_regression.permute_within_groups",
            side_effect=_perm,
        ):
            _compute_permutation_pvalues(
                X=X,
                y_f=y_f,
                beta=beta,
                y_hat_z=y_hat_z,
                resid_z=resid_z,
                valid_feat=valid_feat,
                groups_v=groups_v,
                names=["intercept", "feature"],
                idx_feature=1,
                beta_feature=0.5,
                beta_int=np.nan,
                n_permutations=3,
                rng_seed=1,
            )

        self.assertTrue(calls)
        self.assertTrue(all(n == int(valid_feat.sum()) for n in calls))

    def test_regression_strict_non_iid_marks_missing_when_permutation_unavailable(self):
        from eeg_pipeline.utils.analysis.stats.trialwise_regression import run_trialwise_feature_regressions

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "regression": {
                        "outcome": "rating",
                        "include_temperature": False,
                        "include_trial_order": False,
                        "include_prev_terms": False,
                        "include_run_block": False,
                        "include_interaction": False,
                        "standardize": True,
                        "min_samples": 5,
                        "n_permutations": 10,
                    },
                    "statistics": {"base_seed": 42},
                }
            }
        )
        trial_df = pd.DataFrame(
            {
                "rating": [1, 2, 3, 4, 5, 6],
                "power_alpha": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
            }
        )

        with patch(
            "eeg_pipeline.utils.analysis.stats.trialwise_regression._compute_permutation_pvalues",
            return_value=(np.nan, np.nan),
        ):
            out, _meta = run_trialwise_feature_regressions(
                trial_df,
                feature_cols=["power_alpha"],
                config=cfg,
                groups_for_permutation=np.array([1, 1, 2, 2, 3, 3], dtype=int),
                strict_permutation_primary=True,
            )

        self.assertFalse(out.empty)
        self.assertTrue(np.isnan(float(out.iloc[0]["p_primary"])))
        self.assertEqual(str(out.iloc[0]["p_primary_source"]), "perm_missing_required")

    def test_models_trial_level_requires_explicit_iid_force_override(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_models

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "models": {"enabled": True, "primary_unit": "trial"},
                    "statistics": {"allow_iid_trials": True},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "rating": np.linspace(10, 70, 20),
                "power_alpha": np.linspace(0.1, 1.2, 20),
                "run_id": np.repeat([1, 2, 3, 4], 5),
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_alpha"],
        ):
            with self.assertRaises(ValueError) as exc:
                stage_models(ctx, SimpleNamespace())

        self.assertIn("force_trial_iid_asymptotic", str(exc.exception))

    def test_trial_order_accepts_within_run_monotonic_sequence(self):
        from eeg_pipeline.context.behavior import BehaviorContext

        ctx = BehaviorContext(
            subject="0001",
            task="task",
            config=DotConfig({"behavior_analysis": {"run_adjustment": {"column": "run_id"}}}),
            logger=Mock(),
            deriv_root=Path(tempfile.mkdtemp()),
            stats_dir=Path(tempfile.mkdtemp()),
            control_trial_order=True,
        )
        ctx.aligned_events = pd.DataFrame(
            {
                "run_id": [1, 1, 2, 2],
                "trial_in_run": [1, 2, 1, 2],
            }
        )
        ctx.covariates_df = None
        ctx.data_qc = {}

        ctx._setup_trial_order_covariate()

        self.assertTrue(ctx.control_trial_order)
        self.assertIsNotNone(ctx.covariates_df)
        self.assertIn("trial_index", list(ctx.covariates_df.columns))

    def test_feature_table_alignment_reindexes_by_shared_epoch_key(self):
        from eeg_pipeline.context.behavior import BehaviorContext

        ctx = BehaviorContext(
            subject="0001",
            task="task",
            config=DotConfig({}),
            logger=Mock(),
            deriv_root=Path(tempfile.mkdtemp()),
            stats_dir=Path(tempfile.mkdtemp()),
        )
        ctx.aligned_events = pd.DataFrame({"epoch": [2, 1], "rating": [10.0, 20.0]})

        feature_df = pd.DataFrame(
            {
                "epoch": [1, 2],
                "power_alpha": [0.1, 0.9],
            }
        )
        report = {}
        out = ctx._align_single_feature_table(
            name="power",
            df=feature_df,
            base_index=ctx.aligned_events.index,
            alignment_report=report,
        )

        self.assertIsNotNone(out)
        self.assertEqual([float(v) for v in out["power_alpha"].tolist()], [0.9, 0.1])
        self.assertEqual(report["power"]["status"], "reindexed_by_event_keys")

    def test_feature_table_alignment_can_disallow_positional_only_matching(self):
        from eeg_pipeline.context.behavior import BehaviorContext

        ctx = BehaviorContext(
            subject="0001",
            task="task",
            config=DotConfig({"behavior_analysis": {"trial_table": {"disallow_positional_alignment": True}}}),
            logger=Mock(),
            deriv_root=Path(tempfile.mkdtemp()),
            stats_dir=Path(tempfile.mkdtemp()),
        )
        ctx.aligned_events = pd.DataFrame({"rating": [10.0, 20.0]})

        feature_df = pd.DataFrame({"power_alpha": [0.1, 0.9]})
        report = {}
        out = ctx._align_single_feature_table(
            name="power",
            df=feature_df,
            base_index=ctx.aligned_events.index,
            alignment_report=report,
        )

        self.assertIsNone(out)
        self.assertEqual(report["power"]["reason"], "positional_alignment_disallowed")

    def test_condition_window_forwards_configured_min_samples(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_condition_window

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "condition": {
                        "compare_windows": ["baseline", "active"],
                        "window_comparison": {"primary_unit": "trial", "min_samples": 9},
                    },
                    "statistics": {"allow_iid_trials": True},
                },
                "event_columns": {"binary_outcome": ["binary_outcome"]},
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "power_baseline_alpha_ch_Fp1_mean": [1.0, 2.0, 3.0, 2.0, 1.0, 4.0, 5.0, 3.0, 2.0, 1.0],
                "power_active_alpha_ch_Fp1_mean": [1.5, 2.5, 3.5, 2.5, 1.5, 4.5, 5.5, 3.5, 2.5, 1.5],
            }
        )
        captured = {}

        def _fake_window(_ctx, _df, _features, _windows, min_samples, _fdr_alpha, _suffix):
            captured["min_samples"] = int(min_samples)
            return pd.DataFrame()

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._run_window_comparison",
            side_effect=_fake_window,
        ):
            stage_condition_window(
                ctx,
                SimpleNamespace(fdr_alpha=0.05, min_samples=3),
                df_trials=df_trials,
                feature_cols=[
                    "power_baseline_alpha_ch_Fp1_mean",
                    "power_active_alpha_ch_Fp1_mean",
                ],
                compare_windows=["baseline", "active"],
            )

        self.assertEqual(int(captured["min_samples"]), 9)

    def test_window_run_level_reports_run_level_descriptives(self):
        from eeg_pipeline.analysis.behavior.orchestration import _run_window_comparison

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "condition": {
                        "window_comparison": {"primary_unit": "run_mean"},
                    },
                    "statistics": {"allow_iid_trials": False},
                    "run_adjustment": {"column": "run_id"},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "run_id": [1, 1, 1, 1, 2],
                "power_baseline_alpha_ch_Fp1_mean": [1.0, 1.0, 1.0, 1.0, 5.0],
                "power_active_alpha_ch_Fp1_mean": [2.0, 2.0, 2.0, 2.0, 6.0],
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._compute_unified_fdr",
            side_effect=lambda _ctx, _cfg, df, **_kw: df,
        ):
            out = _run_window_comparison(
                ctx,
                df_trials,
                [
                    "power_baseline_alpha_ch_Fp1_mean",
                    "power_active_alpha_ch_Fp1_mean",
                ],
                ["baseline", "active"],
                min_samples=2,
                fdr_alpha=0.05,
                suffix="",
            )

        self.assertFalse(out.empty)
        self.assertEqual(int(out.iloc[0]["n_pairs"]), 2)
        self.assertAlmostEqual(float(out.iloc[0]["mean_window1"]), 3.0, places=12)
        self.assertAlmostEqual(float(out.iloc[0]["mean_window2"]), 4.0, places=12)

    def test_paired_condition_permutation_does_not_use_unpaired_label_shuffle(self):
        from eeg_pipeline.utils.parallel import _compute_single_condition_effect

        features = pd.DataFrame({"power_alpha": [10.0, 1.0, 9.0, 2.0]})
        pain_mask = np.array([True, False, True, False], dtype=bool)
        nonpain_mask = ~pain_mask
        pair_ids = np.array([1, 1, 2, 2], dtype=int)

        with patch(
            "eeg_pipeline.utils.analysis.stats.permutation.perm_pval_mean_difference",
            side_effect=AssertionError("unpaired permutation should not run for paired tests"),
        ):
            out = _compute_single_condition_effect(
                "power_alpha",
                features,
                pain_mask,
                nonpain_mask,
                min_samples=2,
                paired=True,
                pair_ids=pair_ids,
                n_perm=10,
                base_seed=1,
            )

        self.assertIsNotNone(out)
        self.assertTrue(bool(out["paired_test"]))
        self.assertTrue(np.isfinite(float(out["p_perm"])))

    def test_mediation_strict_non_iid_does_not_fallback_to_sobel(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_mediation

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "mediation": {
                        "n_permutations": 20,
                        "p_primary_mode": "perm_if_available",
                    },
                    "statistics": {"allow_iid_trials": False},
                    "run_adjustment": {"column": "run_id"},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "temperature": np.linspace(40, 46, 12),
                "rating": np.linspace(10, 70, 12),
                "power_alpha": np.linspace(0.1, 1.2, 12),
                "run_id": np.repeat([1, 2, 3], 4),
            }
        )
        mediation_df = pd.DataFrame(
            {
                "mediator": ["power_alpha"],
                "sobel_p": [0.01],
                "p_ab_perm": [np.nan],
                "p_value": [0.01],
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_alpha"],
        ), patch(
            "eeg_pipeline.analysis.behavior.api.run_mediation_analysis",
            return_value=mediation_df.copy(),
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._compute_unified_fdr",
            side_effect=lambda _ctx, _cfg, df, **_kw: df,
        ):
            out = stage_mediation(ctx, SimpleNamespace(fdr_alpha=0.05))

        self.assertFalse(out.empty)
        self.assertTrue(np.isnan(float(out.iloc[0]["p_primary"])))

    def test_mediation_non_iid_overrides_asymptotic_primary_mode(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_mediation

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "mediation": {
                        "n_permutations": 20,
                        "p_primary_mode": "asymptotic",
                    },
                    "statistics": {"allow_iid_trials": False},
                    "run_adjustment": {"column": "run_id"},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "temperature": np.linspace(40, 46, 12),
                "rating": np.linspace(10, 70, 12),
                "power_alpha": np.linspace(0.1, 1.2, 12),
                "run_id": np.repeat([1, 2, 3], 4),
            }
        )
        mediation_df = pd.DataFrame(
            {
                "mediator": ["power_alpha"],
                "sobel_p": [0.01],
                "p_ab_perm": [np.nan],
                "p_value": [0.01],
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_alpha"],
        ), patch(
            "eeg_pipeline.analysis.behavior.api.run_mediation_analysis",
            return_value=mediation_df.copy(),
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._compute_unified_fdr",
            side_effect=lambda _ctx, _cfg, df, **_kw: df,
        ):
            out = stage_mediation(ctx, SimpleNamespace(fdr_alpha=0.05))

        self.assertFalse(out.empty)
        self.assertTrue(np.isnan(float(out.iloc[0]["p_primary"])))
        self.assertEqual(str(out.iloc[0]["p_primary_source"]), "perm_missing_required")

    def test_group_correlations_block_permutation_failure_sets_nan_instead_of_fallback(self):
        from eeg_pipeline.analysis.behavior.orchestration import run_group_level_correlations

        df_a = pd.DataFrame(
            {
                "rating": np.linspace(10, 70, 24),
                "power_alpha": np.linspace(0.1, 1.2, 24),
                "run_id": np.repeat([1, 2, 3], 8),
            }
        )
        df_b = pd.DataFrame(
            {
                "rating": np.linspace(12, 72, 24),
                "power_alpha": np.linspace(0.2, 1.3, 24),
                "run_id": np.repeat([1, 2, 3], 8),
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._find_trial_table_path",
            return_value=Path("/tmp/trials.tsv"),
        ), patch(
            "eeg_pipeline.infra.paths.deriv_stats_path",
            side_effect=lambda _root, sub: Path(f"/tmp/{sub}"),
        ), patch(
            "eeg_pipeline.infra.tsv.read_table",
            side_effect=[df_a, df_b],
        ), patch(
            "eeg_pipeline.utils.analysis.stats.permutation.permute_within_groups",
            side_effect=ValueError("forced failure"),
        ), patch(
            "eeg_pipeline.utils.analysis.stats.fdr.hierarchical_fdr",
            side_effect=lambda df, **_kwargs: df,
        ):
            out = run_group_level_correlations(
                subjects=["0001", "0002"],
                deriv_root=Path("/tmp"),
                config=DotConfig({}),
                logger=Mock(),
                use_block_permutation=True,
                n_perm=20,
            )

        self.assertFalse(out.empty)
        self.assertTrue(np.isnan(float(out.iloc[0]["p_perm"])))
        self.assertTrue(np.isnan(float(out.iloc[0]["p_primary"])))
        self.assertEqual(str(out.iloc[0]["p_primary_kind"]), "perm_missing_required")
        self.assertIn("failed", str(out.iloc[0]["permutation_method"]))

    def test_moderation_strict_non_iid_does_not_fallback_to_asymptotic(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_moderation

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "moderation": {
                        "n_permutations": 20,
                        "p_primary_mode": "perm_if_available",
                    },
                    "statistics": {"allow_iid_trials": False},
                    "run_adjustment": {"column": "run_id"},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "temperature": np.linspace(40, 46, 12),
                "rating": np.linspace(10, 70, 12),
                "power_alpha": np.linspace(0.1, 1.2, 12),
                "run_id": np.repeat([1, 2, 3], 4),
            }
        )

        fake_result = SimpleNamespace(
            n=12,
            b1=0.1,
            b2=0.1,
            b3=0.2,
            se_b3=0.05,
            p_b3=0.01,
            p_b3_perm=np.nan,
            n_permutations=20,
            slope_low_w=0.0,
            slope_mean_w=0.0,
            slope_high_w=0.0,
            p_slope_low=0.5,
            p_slope_mean=0.5,
            p_slope_high=0.5,
            r_squared=0.2,
            r_squared_change=0.1,
            f_interaction=4.0,
            p_f_interaction=0.04,
            jn_low=np.nan,
            jn_high=np.nan,
            jn_type="none",
            is_significant_moderation=lambda alpha: True,
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_alpha"],
        ), patch(
            "eeg_pipeline.utils.analysis.stats.moderation.run_moderation_analysis",
            return_value=fake_result,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._compute_unified_fdr",
            side_effect=lambda _ctx, _cfg, df, **_kw: df,
        ):
            out = stage_moderation(
                ctx,
                SimpleNamespace(
                    fdr_alpha=0.05,
                    moderation_max_features=None,
                    method_label="",
                    moderation_min_samples=5,
                ),
            )

        self.assertFalse(out.empty)
        self.assertTrue(np.isnan(float(out.iloc[0]["p_primary"])))

    def test_moderation_non_iid_overrides_asymptotic_primary_mode(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_moderation

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "moderation": {
                        "n_permutations": 20,
                        "p_primary_mode": "asymptotic",
                    },
                    "statistics": {"allow_iid_trials": False},
                    "run_adjustment": {"column": "run_id"},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "temperature": np.linspace(40, 46, 12),
                "rating": np.linspace(10, 70, 12),
                "power_alpha": np.linspace(0.1, 1.2, 12),
                "run_id": np.repeat([1, 2, 3], 4),
            }
        )

        fake_result = SimpleNamespace(
            n=12,
            b1=0.1,
            b2=0.1,
            b3=0.2,
            se_b3=0.05,
            p_b3=0.01,
            p_b3_perm=np.nan,
            n_permutations=20,
            slope_low_w=0.0,
            slope_mean_w=0.0,
            slope_high_w=0.0,
            p_slope_low=0.5,
            p_slope_mean=0.5,
            p_slope_high=0.5,
            r_squared=0.2,
            r_squared_change=0.1,
            f_interaction=4.0,
            p_f_interaction=0.04,
            jn_low=np.nan,
            jn_high=np.nan,
            jn_type="none",
            is_significant_moderation=lambda alpha: True,
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_alpha"],
        ), patch(
            "eeg_pipeline.utils.analysis.stats.moderation.run_moderation_analysis",
            return_value=fake_result,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._compute_unified_fdr",
            side_effect=lambda _ctx, _cfg, df, **_kw: df,
        ):
            out = stage_moderation(
                ctx,
                SimpleNamespace(
                    fdr_alpha=0.05,
                    moderation_max_features=None,
                    method_label="",
                    moderation_min_samples=5,
                ),
            )

        self.assertFalse(out.empty)
        self.assertTrue(np.isnan(float(out.iloc[0]["p_primary"])))
        self.assertEqual(str(out.iloc[0]["p_primary_source"]), "perm_missing_required")

    def test_temporal_stats_requires_cluster_correction_when_iid_disallowed(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_temporal_stats

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "statistics": {"allow_iid_trials": False},
                    "temporal": {
                        "correction_method": "fdr",
                        "features": {"power": True, "itpc": False, "erds": False},
                    },
                }
            }
        )
        ctx = self._ctx(cfg)
        ctx.use_spearman = True
        ctx.aligned_events = pd.DataFrame({"run_id": [1, 1, 2, 2]})
        power_result = {
            "records": [
                {
                    "condition": "pain",
                    "feature": "power",
                    "band": "alpha",
                    "time_start": 0.0,
                    "time_end": 0.1,
                    "channel": "Cz",
                    "r": 0.2,
                    "p_raw": 0.03,
                    "p_cluster": 0.03,
                }
            ]
        }

        with patch(
            "eeg_pipeline.analysis.behavior.api.compute_temporal_from_context",
            return_value=power_result,
        ):
            with self.assertRaises(ValueError):
                stage_temporal_stats(ctx)

    def test_temporal_stats_cluster_mode_does_not_silently_fallback_without_cluster_pvalues(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_temporal_stats

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "statistics": {"allow_iid_trials": False},
                    "run_adjustment": {"column": "run_id"},
                    "temporal": {
                        "correction_method": "cluster",
                        "features": {"power": True, "itpc": False, "erds": False},
                    },
                }
            }
        )
        ctx = self._ctx(cfg)
        ctx.use_spearman = True
        ctx.aligned_events = pd.DataFrame({"run_id": [1, 1, 2, 2]})

        power_result = {
            "records": [
                {
                    "condition": "pain",
                    "feature": "power",
                    "band": "alpha",
                    "time_start": 0.0,
                    "time_end": 0.1,
                    "channel": "Cz",
                    "r": 0.2,
                    "p_raw": 0.03,
                }
            ]
        }

        with patch(
            "eeg_pipeline.analysis.behavior.api.compute_temporal_from_context",
            return_value=power_result,
        ):
            with self.assertRaises(ValueError):
                stage_temporal_stats(ctx)

    def test_correlate_design_defaults_to_crossfit_predictor_residual_when_available(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_correlate_design

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "statistics": {"allow_iid_trials": False},
                    "correlations": {"permutation": {"enabled": True, "n_permutations": 20}},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "rating": np.linspace(10, 50, 8),
                "temperature": np.linspace(43, 46, 8),
                "predictor_residual": np.linspace(-1, 1, 8),
                "predictor_residual_cv": np.linspace(-0.8, 0.8, 8),
                "run_id": np.repeat([1, 2], 4),
                "power_alpha": np.linspace(0.1, 0.8, 8),
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_alpha"],
        ):
            design = stage_correlate_design(
                ctx,
                SimpleNamespace(control_temperature=True, control_trial_order=True),
            )

        self.assertIsNotNone(design)
        self.assertGreaterEqual(len(design.targets), 1)
        self.assertEqual(design.targets[0], "predictor_residual_cv")

    def test_condition_multigroup_uses_unified_fdr_pipeline(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_condition_multigroup

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "condition": {
                        "primary_unit": "run_mean",
                        "compare_column": "pain_level",
                        "compare_values": [0, 1, 2],
                    },
                    "run_adjustment": {"column": "run_id"},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "run_id": [1, 1, 1, 2, 2, 2],
                "pain_level": [0, 1, 2, 0, 1, 2],
                "power_alpha": [0.2, 0.4, 0.6, 0.3, 0.5, 0.7],
            }
        )

        base_df = pd.DataFrame(
            {
                "feature": ["power_alpha"],
                "feature_type": ["power"],
                "band": ["alpha"],
                "p_value": [0.02],
                "hedges_g": [0.6],
            }
        )
        captured = {}

        def _capture_fdr(_ctx, _cfg, df, **kwargs):
            captured["analysis_type"] = kwargs.get("analysis_type")
            return df.assign(p_fdr=[0.02], q_global=[0.02])

        with patch(
            "eeg_pipeline.utils.analysis.stats.effect_size.compute_multigroup_condition_effects",
            return_value=base_df,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._compute_unified_fdr",
            side_effect=_capture_fdr,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._write_stats_table",
            return_value=Path("/tmp/fake.tsv"),
        ):
            out = stage_condition_multigroup(
                ctx,
                SimpleNamespace(fdr_alpha=0.05),
                df_trials=df_trials,
                feature_cols=["power_alpha"],
            )

        self.assertFalse(out.empty)
        self.assertEqual(captured.get("analysis_type"), "condition_multigroup")
        self.assertIn("p_fdr", out.columns)


if __name__ == "__main__":
    unittest.main()
