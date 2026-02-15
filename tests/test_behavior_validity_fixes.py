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
            task="thermalactive",
            config=config,
            logger=Mock(),
            deriv_root=Path(tempfile.mkdtemp()),
            stats_dir=Path(tempfile.mkdtemp()),
            overwrite=True,
            also_save_csv=False,
            selected_feature_files=None,
            feature_categories=None,
            selected_bands=None,
            group_ids=None,
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
                "event_columns": {"pain_binary": ["pain_binary"]},
            }
        )
        df_trials = pd.DataFrame({"pain_binary": [0, 1]})
        self.assertEqual(_resolve_condition_compare_column(df_trials, cfg), "pain_binary")

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
                "event_columns": {"pain_binary": ["pain_binary"]},
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "run_id": [1, 1, 2, 2],
                "pain_binary": [1, 1, 0, 0],
                "power_alpha": [0.2, 0.4, 0.8, 1.0],
            }
        )
        captured = {}

        def _fake_effects(features_df, pain_mask, nonpain_mask, min_samples, **kwargs):
            captured["n_rows"] = len(features_df)
            captured["min_samples"] = min_samples
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
        self.assertEqual(str(out["condition_column"].iloc[0]), "pain_binary")

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
                "event_columns": {"pain_binary": ["pain_binary"]},
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "run_id": [1, 1, 2, 2],
                "pain_binary": [1, 0, 1, 0],
                "power_alpha": [0.2, 0.4, 0.8, 1.0],
            }
        )
        captured = {}

        def _fake_effects(features_df, pain_mask, nonpain_mask, min_samples, **kwargs):
            captured["n_rows"] = len(features_df)
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

    def test_group_trial_table_discovery_finds_unsuffixed_trials(self):
        from eeg_pipeline.analysis.behavior.orchestration import _find_trial_table_path

        root = Path(tempfile.mkdtemp())
        trials_path = root / "trial_table" / "all" / "trials.tsv"
        trials_path.parent.mkdir(parents=True, exist_ok=True)
        trials_path.write_text("rating\tpower_alpha\n1\t0.1\n")

        found = _find_trial_table_path(root, feature_files=None)
        self.assertEqual(found, trials_path)

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

    def test_permutation_groups_with_singletons_fall_back_without_error(self):
        from eeg_pipeline.utils.analysis.stats.permutation import permute_within_groups

        rng = np.random.default_rng(123)
        idx = permute_within_groups(
            4,
            rng,
            groups=np.array([10, 11, 12, 13]),
            scheme="shuffle",
        )
        self.assertEqual(sorted(idx.tolist()), [0, 1, 2, 3])

    def test_behavior_pipeline_config_parses_run_validation_flag(self):
        from eeg_pipeline.pipelines.behavior import BehaviorPipelineConfig

        cfg = DotConfig({"behavior_analysis": {"validation": {"enabled": False}}})
        parsed = BehaviorPipelineConfig.from_config(cfg)
        self.assertFalse(parsed.run_validation)

    def test_behavior_pipeline_config_rejects_conflicting_correlation_method_keys(self):
        from eeg_pipeline.pipelines.behavior import BehaviorPipelineConfig

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "correlation_method": "pearson",
                    "statistics": {"correlation_method": "spearman"},
                }
            }
        )
        with self.assertRaises(ValueError):
            BehaviorPipelineConfig.from_config(cfg)

    def test_correlate_design_accepts_legacy_targets_key(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_correlate_design

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "targets": ["rating", "temperature"],
                    "correlations": {"permutation": {"enabled": True}},
                    "statistics": {"allow_iid_trials": False},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "rating": np.linspace(10, 50, 8),
                "temperature": np.linspace(43, 46, 8),
                "pain_residual": np.linspace(-1, 1, 8),
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

    def test_correlate_design_rejects_conflicting_target_keys(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_correlate_design

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "targets": ["rating"],
                    "correlations": {
                        "targets": ["temperature"],
                        "permutation": {"enabled": True},
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

    def test_trial_table_stage_reuses_cached_output_when_input_hash_matches(self):
        from eeg_pipeline.analysis.behavior.orchestration import _cache, stage_trial_table
        from eeg_pipeline.utils.data.trial_table import compute_trial_table_schema_hash

        cfg = DotConfig({"behavior_analysis": {"trial_table": {"format": "tsv"}}})
        ctx = self._ctx(cfg)
        out_dir = ctx.stats_dir / "trial_table" / "all"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "trials.tsv"
        df_cached = pd.DataFrame({"rating": [10], "power_alpha": [0.1]})
        df_cached.to_csv(out_path, sep="\t", index=False)
        schema_hash = compute_trial_table_schema_hash(df_cached)
        (out_dir / "trials.metadata.json").write_text(
            '{"n_trials": 1, "n_columns": 2, "contract": {"version": "1.0", "schema_hash": "'
            + schema_hash
            + '", "input_hash": "abc"}}',
            encoding="utf-8",
        )

        _cache.clear()
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
        from eeg_pipeline.analysis.behavior.orchestration import _cache

        cfg = DotConfig({"behavior_analysis": {"trial_table": {"format": "tsv"}}})
        ctx = self._ctx(cfg)
        out_dir = ctx.stats_dir / "trial_table" / "all"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trials.tsv").write_text("rating\tpower_alpha\n10\t0.1\n", encoding="utf-8")
        (out_dir / "trials.metadata.json").write_text(
            '{"n_trials": 1, "n_columns": 2, "contract": {"version": "1.0", "schema_hash": "definitely_wrong"}}',
            encoding="utf-8",
        )

        _cache.clear()
        with self.assertRaises(ValueError):
            _cache.get_trial_table(ctx)

    def test_pain_sensitivity_preserves_permutation_primary_pvalue(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_pain_sensitivity

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "pain_sensitivity": {
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
            "eeg_pipeline.analysis.behavior.api.run_pain_sensitivity_correlations",
            return_value=psi_df.copy(),
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._compute_unified_fdr",
            side_effect=lambda _ctx, _cfg, df, **_kw: df,
        ):
            out = stage_pain_sensitivity(ctx, SimpleNamespace(method="spearman", min_samples=5, fdr_alpha=0.05))

        self.assertFalse(out.empty)
        self.assertAlmostEqual(float(out.iloc[0]["p_primary"]), 0.01, places=12)
        self.assertEqual(str(out.iloc[0]["p_primary_source"]), "psi_perm")

    def test_correlate_primary_selection_does_not_fallback_to_raw_when_controlled_stat_missing(self):
        from eeg_pipeline.analysis.behavior.orchestration import CorrelateDesign, stage_correlate_primary_selection

        cfg = DotConfig(
            {
                "behavior_analysis": {
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
                "r_partial_cov_temp": np.nan,
                "p_partial_cov_temp": np.nan,
                "p_perm_partial_cov_temp": np.nan,
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
        self.assertEqual(str(out[0]["p_primary_source"]), "partial_cov_temp_missing")

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


if __name__ == "__main__":
    unittest.main()
