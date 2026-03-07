import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from scipy import stats

from tests.pipelines_test_utils import DotConfig


class TestBehaviorValidityFixes(unittest.TestCase):
    def _ctx(self, config: DotConfig) -> SimpleNamespace:
        event_columns = config.setdefault("event_columns", {})
        event_columns.setdefault("predictor", ["predictor", "temperature"])
        event_columns.setdefault("outcome", ["outcome", "rating"])
        event_columns.setdefault("binary_outcome", ["binary_outcome"])
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

        self.assertIsNotNone(StageRegistry.get("hierarchical_fdr_summary"))

        primary_spec = StageRegistry.get("correlate_primary_selection")
        self.assertIsNotNone(primary_spec)
        self.assertIn(StageRegistry.RESOURCE_EFFECT_SIZES, primary_spec.requires)
        self.assertNotIn(StageRegistry.RESOURCE_PVALUES, primary_spec.requires)

    def test_predictor_residual_stage_updates_trial_table_cache(self):
        import pandas as pd

        from eeg_pipeline.analysis.behavior import orchestration as orch

        ctx = self._ctx(
            DotConfig(
                {
                    "behavior_analysis": {
                        "run_adjustment": {"column": "run_id"},
                        "predictor_residual": {
                            "enabled": True,
                            "method": "poly",
                            "min_samples": 3,
                            "crossfit": {"enabled": False},
                        },
                    }
                }
            )
        )

        base_df = pd.DataFrame(
            {
                "epoch": list(range(8)),
                "run_id": [1, 1, 1, 1, 2, 2, 2, 2],
                "temperature": [44.0, 44.5, 45.0, 45.5, 46.0, 46.5, 47.0, 47.5],
                "rating": [10.0, 14.0, 18.0, 23.0, 30.0, 38.0, 47.0, 57.0],
                "predictor": [44.0, 44.5, 45.0, 45.5, 46.0, 46.5, 47.0, 47.5],
                "outcome": [10.0, 14.0, 18.0, 23.0, 30.0, 38.0, 47.0, 57.0],
                "power_alpha": [0.1, 0.15, 0.2, 0.24, 0.3, 0.36, 0.43, 0.5],
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

    def test_split_by_condition_requires_explicit_compare_values_for_multilevel_column(self):
        from eeg_pipeline.utils.analysis.stats.effect_size import split_by_condition

        cfg = DotConfig(
            {
                "behavior_analysis": {"condition": {"compare_column": "pain_level"}},
                "event_columns": {"binary_outcome": ["pain_level"]},
            }
        )
        df_trials = pd.DataFrame({"pain_level": [0, 1, 2, 1]})

        with self.assertRaises(ValueError):
            split_by_condition(df_trials, cfg, Mock())

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

    def test_condition_column_reports_binary_values_in_split_order(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_condition_column

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "condition": {
                        "compare_column": "",
                        "primary_unit": "trial",
                        "compare_values": [],
                    },
                    "statistics": {"allow_iid_trials": True},
                    "run_adjustment": {"column": "run_id"},
                },
                "event_columns": {"binary_outcome": ["binary_outcome"]},
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "run_id": [1, 1, 2, 2],
                "binary_outcome": [0, 1, 0, 1],
                "power_alpha": [0.2, 0.5, 0.3, 0.7],
            }
        )

        with patch(
            "eeg_pipeline.analysis.behavior.api.compute_condition_effects",
            return_value=pd.DataFrame(
                {
                    "feature": ["power_alpha"],
                    "hedges_g": [0.6],
                    "p_value": [0.2],
                    "p_primary": [0.2],
                }
            ),
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
        self.assertEqual(str(out.iloc[0]["condition_value1"]), "1")
        self.assertEqual(str(out.iloc[0]["condition_value2"]), "0")

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
                target_col="rating",
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
                target_col="rating",
            )

        self.assertFalse(out.empty)
        self.assertLess(abs(float(out.iloc[0]["r"])), 0.35)

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
            cond_a_mask=pain_mask,
            cond_b_mask=nonpain_mask,
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
            cond_a_mask=pain_mask,
            cond_b_mask=nonpain_mask,
            min_samples=2,
            fdr_alpha=0.05,
            n_jobs=1,
            config=cfg,
            groups=None,
        )

        self.assertFalse(out.empty)
        self.assertTrue(np.isnan(float(out.iloc[0]["p_primary"])))
        self.assertEqual(str(out.iloc[0]["p_primary_source"]), "perm_missing_required")

    def test_condition_effects_mark_reportable_effects_from_threshold(self):
        from eeg_pipeline.utils.analysis.stats.effect_size import compute_condition_effects

        features = pd.DataFrame(
            {
                "power_big": [1.0, 2.0, 10.0, 11.0],
                "power_small": [1.0, 2.0, 1.1, 2.1],
            }
        )
        cond_a_mask = np.array([True, True, False, False], dtype=bool)
        cond_b_mask = ~cond_a_mask
        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "condition": {
                        "effect_size_threshold": 0.5,
                        "permutation": {"enabled": False},
                    }
                }
            }
        )

        out = compute_condition_effects(
            features_df=features,
            cond_a_mask=cond_a_mask,
            cond_b_mask=cond_b_mask,
            min_samples=2,
            fdr_alpha=0.05,
            n_jobs=1,
            config=cfg,
            groups=None,
        )

        self.assertFalse(out.empty)
        self.assertIn("reportable_effect", out.columns)
        self.assertIn("effect_size_threshold", out.columns)
        by_feature = out.set_index("feature")
        self.assertTrue(bool(by_feature.loc["power_big", "reportable_effect"]))
        self.assertFalse(bool(by_feature.loc["power_small", "reportable_effect"]))
        self.assertAlmostEqual(float(by_feature.loc["power_big", "effect_size_threshold"]), 0.5, places=12)

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

    def test_combined_partial_permutation_reindexes_groups_to_complete_case_subset(self):
        from eeg_pipeline.utils.analysis.stats.permutation import (
            _compute_combined_covariates_predictor_pvalue,
        )

        index = pd.Index(["a", "b", "c", "d"])
        x = pd.Series([0.1, 0.2, 0.3, 0.4], index=index)
        y = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
        covariates = pd.DataFrame({"trial_index": [1.0, 2.0, np.nan, 4.0]}, index=index)
        predictor = pd.Series([44.0, 45.0, 46.0, 47.0], index=index)
        groups = np.array([10, 10, 20, 20], dtype=int)
        captured = {}

        def _fake_perm(x_sub, y_sub, z_sub, method, n_perm, rng, *, groups, config, scheme):
            captured["x_index"] = list(x_sub.index)
            captured["groups"] = groups
            return 0.25

        with patch(
            "eeg_pipeline.utils.analysis.stats.permutation._build_predictor_covariates",
            return_value=pd.DataFrame({"predictor": predictor}, index=index),
        ), patch(
            "eeg_pipeline.utils.analysis.stats.permutation.perm_pval_partial_freedman_lane",
            side_effect=_fake_perm,
        ):
            out = _compute_combined_covariates_predictor_pvalue(
                x_aligned=x,
                y_aligned=y,
                covariates_df=covariates,
                predictor_series=predictor,
                method="pearson",
                n_perm=10,
                n_eff=3,
                rng=np.random.default_rng(0),
                min_samples=3,
                config=DotConfig({}),
                groups=groups,
            )

        self.assertAlmostEqual(float(out), 0.25, places=12)
        self.assertEqual(captured["x_index"], ["a", "b", "d"])
        self.assertEqual(list(captured["groups"].index), ["a", "b", "d"])
        self.assertEqual(captured["groups"].tolist(), [10, 10, 20])

    def test_batch_condition_permutation_uses_feature_specific_finite_rows(self):
        from eeg_pipeline.utils.analysis.stats.effect_size import (
            _compute_batch_permutation_pvalues,
        )

        data_matrix = np.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
                [5.0, np.nan],
            ],
            dtype=float,
        )
        cond_a_mask = np.array([True, True, False, False, False], dtype=bool)
        cond_b_mask = ~cond_a_mask
        calls = []

        def _perm(n, rng, groups=None, scheme="shuffle", strict=True):
            calls.append(int(n))
            return np.arange(n, dtype=int)

        with patch(
            "eeg_pipeline.utils.analysis.stats.permutation.permute_within_groups",
            side_effect=_perm,
        ):
            out = _compute_batch_permutation_pvalues(
                data_matrix=data_matrix,
                cond_a_mask=cond_a_mask,
                cond_b_mask=cond_b_mask,
                n_perm=1,
                base_seed=1,
                groups=None,
                scheme="shuffle",
                logger=None,
            )

        self.assertEqual(out.shape, (2,))
        self.assertEqual(sorted(calls), [4, 5])

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

    def test_trialwise_regression_resolves_alias_columns_and_predictor_interaction(self):
        from eeg_pipeline.utils.analysis.stats.trialwise_regression import (
            run_trialwise_feature_regressions,
        )

        rng = np.random.default_rng(0)
        n_trials = 30
        temperature = np.tile(np.linspace(-1.0, 1.0, 10), 3)
        power = np.repeat(np.array([-1.0, 0.0, 1.0]), 10) + rng.normal(scale=0.05, size=n_trials)
        rating = (
            0.8 * temperature
            + 0.6 * power
            + 1.5 * temperature * power
            + rng.normal(scale=0.02, size=n_trials)
        )
        df_trials = pd.DataFrame(
            {
                "rating": rating,
                "temperature": temperature,
                "power_alpha": power,
                "run_id": np.repeat([1, 2, 3], 10),
            }
        )
        cfg = DotConfig(
            {
                "event_columns": {
                    "outcome": ["rating"],
                    "predictor": ["temperature"],
                },
                "behavior_analysis": {
                    "regression": {
                        "outcome": "outcome",
                        "include_predictor": True,
                        "predictor_control": "linear",
                        "include_trial_order": False,
                        "include_run_block": False,
                        "include_interaction": True,
                        "min_samples": 10,
                        "n_permutations": 0,
                    },
                    "n_jobs": 1,
                },
            }
        )

        out, meta = run_trialwise_feature_regressions(
            df_trials,
            feature_cols=["power_alpha"],
            config=cfg,
        )

        self.assertEqual(meta["outcome"], "rating")
        self.assertEqual(meta["predictor_column"], "temperature")
        self.assertFalse(out.empty)
        self.assertEqual(str(out.iloc[0]["target"]), "rating")
        self.assertTrue(np.isfinite(float(out.iloc[0]["beta_interaction"])))

    def test_stage_regression_run_mean_rejects_run_block_covariate(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_regression

        cfg = DotConfig(
            {
                "event_columns": {
                    "outcome": ["rating"],
                    "predictor": ["temperature"],
                },
                "behavior_analysis": {
                    "regression": {
                        "primary_unit": "run_mean",
                        "include_run_block": True,
                    },
                    "run_adjustment": {"column": "run_id"},
                },
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "run_id": [1, 1, 2, 2],
                "rating": [10.0, 12.0, 20.0, 22.0],
                "temperature": [44.0, 44.5, 45.0, 45.5],
                "power_alpha": [0.1, 0.2, 0.3, 0.4],
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
                stage_regression(
                    ctx,
                    SimpleNamespace(method_label="", min_samples=2),
                )

    def test_stage_regression_run_mean_uses_ungrouped_permutation(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_regression

        cfg = DotConfig(
            {
                "event_columns": {
                    "outcome": ["rating"],
                    "predictor": ["temperature"],
                },
                "behavior_analysis": {
                    "regression": {
                        "primary_unit": "run_mean",
                        "include_run_block": False,
                        "n_permutations": 12,
                    },
                    "run_adjustment": {"column": "run_id"},
                },
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "run_id": [1, 1, 2, 2, 3, 3],
                "rating": [10.0, 12.0, 20.0, 22.0, 30.0, 32.0],
                "temperature": [44.0, 44.5, 45.0, 45.5, 46.0, 46.5],
                "power_alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            }
        )
        captured = {}

        def _fake_regression(
            trial_df,
            *,
            feature_cols,
            config,
            groups_for_permutation,
            strict_permutation_primary,
        ):
            captured["n_rows"] = len(trial_df)
            captured["feature_cols"] = list(feature_cols)
            captured["groups"] = groups_for_permutation
            captured["strict_permutation_primary"] = strict_permutation_primary
            return (
                pd.DataFrame(
                    {
                        "feature": ["power_alpha"],
                        "target": ["rating"],
                        "p_primary": [0.1],
                    }
                ),
                {"status": "ok", "predictor_control": "linear"},
            )

        with patch(
            "eeg_pipeline.analysis.behavior.orchestration._load_trial_table_df",
            return_value=df_trials,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._get_feature_columns",
            return_value=["power_alpha"],
        ), patch(
            "eeg_pipeline.utils.analysis.stats.trialwise_regression.run_trialwise_feature_regressions",
            side_effect=_fake_regression,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._write_stats_table",
            return_value=Path("/tmp/regression.parquet"),
        ):
            out = stage_regression(
                ctx,
                SimpleNamespace(method_label="", min_samples=2),
            )

        self.assertEqual(captured["n_rows"], 6)
        self.assertEqual(captured["feature_cols"], ["power_alpha"])
        self.assertIsNone(captured["groups"])
        self.assertFalse(bool(captured["strict_permutation_primary"]))
        self.assertFalse(out.empty)

    def test_run_level_regression_aggregation_uses_complete_case_trials(self):
        from eeg_pipeline.utils.analysis.stats.trialwise_regression import (
            _aggregate_feature_to_run_level,
        )

        trial_df = pd.DataFrame(
            {
                "run_id": [1, 1, 2, 2],
                "rating": [10.0, 20.0, 30.0, 40.0],
                "temperature": [44.0, 45.0, 46.0, 47.0],
                "power_alpha": [0.2, np.nan, 0.4, 0.6],
                "trial_index": [1.0, 2.0, 1.0, 2.0],
            }
        )

        aggregated = _aggregate_feature_to_run_level(
            trial_df,
            run_col="run_id",
            outcome_column="rating",
            predictor_column="temperature",
            feature_column="power_alpha",
            base_covariate_columns=["trial_index"],
        )

        self.assertEqual(len(aggregated), 2)
        self.assertAlmostEqual(float(aggregated.iloc[0]["rating"]), 10.0, places=12)
        self.assertAlmostEqual(float(aggregated.iloc[0]["temperature"]), 44.0, places=12)
        self.assertAlmostEqual(float(aggregated.iloc[0]["trial_index"]), 1.0, places=12)

    def test_group_correlations_default_target_resolves_outcome_alias(self):
        from eeg_pipeline.analysis.behavior.orchestration import run_group_level_correlations

        df_a = pd.DataFrame(
            {
                "rating": np.linspace(10, 70, 12),
                "power_alpha": np.linspace(0.1, 1.2, 12),
                "run_id": np.repeat([1, 2, 3], 4),
            }
        )
        df_b = pd.DataFrame(
            {
                "rating": np.linspace(15, 75, 12),
                "power_alpha": np.linspace(0.2, 1.3, 12),
                "run_id": np.repeat([1, 2, 3], 4),
            }
        )
        cfg = DotConfig({"event_columns": {"outcome": ["rating"]}})

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
            "eeg_pipeline.analysis.behavior.group_level.compute_correlation",
            return_value=(0.2, 0.5),
        ), patch(
            "eeg_pipeline.utils.analysis.stats.fdr.hierarchical_fdr",
            side_effect=lambda df, **_kwargs: df,
        ):
            out = run_group_level_correlations(
                subjects=["0001", "0002"],
                deriv_root=Path("/tmp"),
                config=cfg,
                logger=Mock(),
                use_block_permutation=False,
                n_perm=0,
            )

        self.assertFalse(out.empty)
        self.assertEqual(str(out.iloc[0]["target"]), "rating")

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
                SimpleNamespace(control_predictor=True, control_trial_order=True),
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
                SimpleNamespace(control_predictor=True, control_trial_order=True),
            )
        self.assertIsNone(design)

    def test_correlate_design_does_not_force_run_adjustment_when_disabled(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_correlate_design

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "run_adjustment": {
                        "enabled": False,
                        "include_in_correlations": True,
                        "column": "run_id",
                    },
                    "correlations": {"targets": ["rating"]},
                    "statistics": {"allow_iid_trials": True},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "rating": np.linspace(10, 50, 8),
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
                SimpleNamespace(control_predictor=False, control_trial_order=False),
            )

        self.assertIsNotNone(design)
        self.assertFalse(bool(design.run_adjust_in_correlations))

    def test_correlate_design_requires_run_column_when_run_adjustment_enabled(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_correlate_design

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "run_adjustment": {
                        "enabled": True,
                        "include_in_correlations": True,
                        "column": "run_id",
                    },
                    "correlations": {"targets": ["rating"]},
                    "statistics": {"allow_iid_trials": True},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "rating": np.linspace(10, 50, 8),
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
                    SimpleNamespace(control_predictor=False, control_trial_order=False),
                )

    def test_correlate_design_rejects_run_adjustment_for_run_mean_primary_unit(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_correlate_design

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "run_adjustment": {
                        "enabled": True,
                        "include_in_correlations": True,
                        "column": "run_id",
                    },
                    "correlations": {
                        "targets": ["rating"],
                        "primary_unit": "run_mean",
                    },
                    "statistics": {"allow_iid_trials": True},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "run_id": np.repeat([1, 2], 4),
                "rating": np.linspace(10, 50, 8),
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
                    SimpleNamespace(control_predictor=False, control_trial_order=False),
                )

    def test_build_run_mean_design_uses_complete_case_trials(self):
        from eeg_pipeline.analysis.behavior.stages.correlate import _build_run_mean_design

        df_trials = pd.DataFrame({"run_id": [1, 1, 2, 2]})
        x = pd.Series([0.2, np.nan, 0.4, 0.6], index=df_trials.index)
        y = pd.Series([10.0, 20.0, 30.0, 40.0], index=df_trials.index)
        cov_df = pd.DataFrame({"trial_index": [1.0, 2.0, 1.0, 2.0]}, index=df_trials.index)
        predictor = pd.Series([44.0, 45.0, 46.0, 47.0], index=df_trials.index)

        x_run, y_run, cov_run, predictor_run = _build_run_mean_design(
            df_trials=df_trials,
            x=x,
            y=y,
            cov_df=cov_df,
            predictor_series=predictor,
            predictor_column="temperature",
            target="rating",
            run_col="run_id",
            run_adjust_in_correlations=False,
        )

        self.assertEqual(len(x_run), 2)
        self.assertAlmostEqual(float(x_run.iloc[0]), 0.2, places=12)
        self.assertAlmostEqual(float(y_run.iloc[0]), 10.0, places=12)
        self.assertAlmostEqual(float(cov_run.iloc[0]["trial_index"]), 1.0, places=12)
        self.assertAlmostEqual(float(predictor_run.iloc[0]), 44.0, places=12)

    def test_correlate_design_rejects_excessive_run_dummy_expansion(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_correlate_design

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "run_adjustment": {
                        "enabled": True,
                        "include_in_correlations": True,
                        "column": "run_id",
                        "max_dummies": 2,
                    },
                    "correlations": {"targets": ["rating"]},
                    "statistics": {"allow_iid_trials": True},
                }
            }
        )
        ctx = self._ctx(cfg)
        df_trials = pd.DataFrame(
            {
                "run_id": np.arange(1, 9),
                "rating": np.linspace(10, 50, 8),
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
                    SimpleNamespace(control_predictor=False, control_trial_order=False),
                )

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
                    SimpleNamespace(control_predictor=True, control_trial_order=True),
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
                SimpleNamespace(control_predictor=True, control_trial_order=True),
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
                SimpleNamespace(control_predictor=True, control_trial_order=True),
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
                SimpleNamespace(control_predictor=True, control_trial_order=True),
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

    def test_trial_table_input_hash_raises_when_event_hashing_fails(self):
        from eeg_pipeline.analysis.behavior.trial_table_helpers import compute_trial_table_input_hash

        ctx = self._ctx(DotConfig({}))
        ctx.aligned_events = pd.DataFrame({"epoch": [1, 2], "rating": [10.0, 20.0]})
        ctx.iter_feature_tables = lambda: []

        with patch("pandas.util.hash_pandas_object", side_effect=TypeError("hash failed")):
            with self.assertRaises(RuntimeError):
                compute_trial_table_input_hash(ctx)

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
            predictor_series=pd.Series([45.0, 46.0]),
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
            SimpleNamespace(control_predictor=True, control_trial_order=True),
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
            predictor_series=None,
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
            SimpleNamespace(control_predictor=True, control_trial_order=True),
            design,
            records,
        )
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out[0]["p_primary"]), 0.01, places=12)
        self.assertEqual(str(out[0]["p_kind_primary"]), "p_perm_raw")
        self.assertEqual(str(out[0]["p_primary_source"]), "raw_robust_perm")

    def test_correlate_primary_selection_run_mean_uses_controlled_estimand(self):
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
            cov_df=pd.DataFrame({"trial_index": [0.0, 1.0, 0.0, 1.0]}),
            predictor_series=pd.Series([44.0, 44.5, 45.0, 45.5]),
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
                "r_run_mean_partial_cov_predictor": 0.6,
                "p_run_mean_partial_cov_predictor": 0.03,
                "robust_method": None,
            }
        ]

        out = stage_correlate_primary_selection(
            ctx,
            SimpleNamespace(control_predictor=True, control_trial_order=True),
            design,
            records,
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(str(out[0]["p_kind_primary"]), "p_run_mean_partial_cov_predictor")
        self.assertAlmostEqual(float(out[0]["p_primary"]), 0.03, places=12)
        self.assertEqual(str(out[0]["p_primary_source"]), "run_mean_partial_cov_predictor")

    def test_correlate_primary_selection_honors_requested_raw_type(self):
        from eeg_pipeline.analysis.behavior.orchestration import CorrelateDesign, stage_correlate_primary_selection

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "correlations": {
                        "types": ["raw"],
                        "p_primary_mode": "perm_if_available",
                    },
                    "statistics": {"allow_iid_trials": False},
                }
            }
        )
        ctx = self._ctx(cfg)
        design = CorrelateDesign(
            df_trials=pd.DataFrame(
                {
                    "power_alpha": [0.2, 0.4, 0.6, 0.8],
                    "rating": [10.0, 20.0, 30.0, 40.0],
                    "temperature": [44.0, 44.5, 45.0, 45.5],
                }
            ),
            feature_cols=["power_alpha"],
            targets=["rating"],
            cov_df=pd.DataFrame({"trial_index": [1.0, 2.0, 3.0, 4.0]}),
            predictor_series=pd.Series([44.0, 44.5, 45.0, 45.5]),
            predictor_column="temperature",
            run_col="run_id",
            run_adjust_in_correlations=False,
            groups_for_perm=pd.Series([1, 1, 2, 2]),
        )
        out = stage_correlate_primary_selection(
            ctx,
            SimpleNamespace(control_predictor=True),
            design,
            [
                {
                    "feature": "power_alpha",
                    "target": "rating",
                    "r_raw": 0.5,
                    "p_raw": 0.04,
                    "p_perm_raw": 0.03,
                    "robust_method": None,
                }
            ],
        )

        self.assertEqual(str(out[0]["p_kind_primary"]), "p_perm_raw")
        self.assertAlmostEqual(float(out[0]["p_primary"]), 0.03, places=12)
        self.assertEqual(str(out[0]["p_primary_source"]), "raw_perm")

    def test_correlate_primary_selection_does_not_downgrade_combined_control_request(self):
        from eeg_pipeline.analysis.behavior.orchestration import CorrelateDesign, stage_correlate_primary_selection

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "correlations": {
                        "types": ["partial_cov_predictor", "raw"],
                        "p_primary_mode": "perm_if_available",
                    },
                    "statistics": {"allow_iid_trials": True},
                }
            }
        )
        ctx = self._ctx(cfg)
        design = CorrelateDesign(
            df_trials=pd.DataFrame({"run_id": [1, 1, 2, 2]}),
            feature_cols=["power_alpha"],
            targets=["rating"],
            cov_df=None,
            predictor_series=pd.Series([44.0, 44.5, 45.0, 45.5]),
            predictor_column="temperature",
            run_col="run_id",
            run_adjust_in_correlations=False,
            groups_for_perm=None,
        )

        out = stage_correlate_primary_selection(
            ctx,
            SimpleNamespace(control_predictor=True),
            design,
            [
                {
                    "feature": "power_alpha",
                    "target": "rating",
                    "r_raw": 0.5,
                    "p_raw": 0.04,
                    "r_partial_predictor": 0.6,
                    "p_partial_predictor": 0.02,
                    "robust_method": None,
                }
            ],
        )

        self.assertEqual(str(out[0]["p_kind_primary"]), "p_partial_cov_predictor")
        self.assertTrue(np.isnan(float(out[0]["p_primary"])))
        self.assertTrue(np.isnan(float(out[0]["r_primary"])))
        self.assertEqual(str(out[0]["p_primary_source"]), "partial_cov_predictor_missing")

    def test_correlate_effect_sizes_allows_robust_raw_with_unrequested_controls(self):
        from eeg_pipeline.analysis.behavior.orchestration import (
            CorrelateDesign,
            stage_correlate_effect_sizes,
        )

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "correlations": {"types": ["raw"]},
                    "n_jobs": 1,
                }
            }
        )
        ctx = self._ctx(cfg)
        design = CorrelateDesign(
            df_trials=pd.DataFrame(
                {
                    "power_alpha": [0.2, 0.4, 0.6, 0.8, 1.0],
                    "rating": [1.0, 2.0, 3.0, 4.0, 5.0],
                    "temperature": [44.0, 44.5, 45.0, 45.5, 46.0],
                }
            ),
            feature_cols=["power_alpha"],
            targets=["rating"],
            cov_df=pd.DataFrame({"trial_index": [1.0, 2.0, 3.0, 4.0, 5.0]}),
            predictor_series=pd.Series([44.0, 44.5, 45.0, 45.5, 46.0]),
            predictor_column="temperature",
            run_col="run_id",
            run_adjust_in_correlations=False,
            groups_for_perm=None,
        )

        out = stage_correlate_effect_sizes(
            ctx,
            SimpleNamespace(
                method="spearman",
                robust_method="winsorized",
                method_label="spearman_winsorized",
                min_samples=3,
            ),
            design,
        )

        self.assertEqual(len(out), 1)
        self.assertTrue(np.isfinite(float(out[0]["r_raw"])))

    def test_predictor_residual_spline_failure_surfaces_instead_of_falling_back(self):
        from eeg_pipeline.utils.analysis.stats.predictor_residual import fit_predictor_outcome_curve

        predictor = pd.Series(np.linspace(43.0, 46.0, 8))
        outcome = pd.Series(np.linspace(10.0, 50.0, 8))
        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "predictor_type": "continuous",
                    "predictor_residual": {
                        "method": "spline",
                        "min_samples": 3,
                    },
                }
            }
        )

        with patch(
            "eeg_pipeline.utils.analysis.stats.predictor_residual._fit_spline_model",
            return_value=None,
        ):
            with self.assertRaises(ValueError):
                fit_predictor_outcome_curve(predictor, outcome, config=cfg)

    def test_predictor_residual_crossfit_uses_same_spline_estimator_path(self):
        from eeg_pipeline.utils.analysis.stats.predictor_residual import (
            crossfit_predictor_outcome_curve,
        )

        predictor = pd.Series(np.linspace(43.0, 46.5, 8))
        outcome = pd.Series(np.linspace(10.0, 45.0, 8))
        groups = pd.Series([1, 1, 1, 1, 2, 2, 2, 2])
        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "predictor_type": "continuous",
                    "predictor_residual": {
                        "min_samples": 3,
                        "crossfit": {
                            "method": "spline",
                            "n_splits": 2,
                        },
                    },
                }
            }
        )
        fit_calls = []

        def _fake_fit(pred_values, out_values, _config):
            fit_calls.append((len(pred_values), len(out_values)))
            return object(), {"model": "spline", "status": "ok", "df": 3}

        def _fake_predict(_model, pred_values):
            return pd.Series(
                np.full(len(pred_values), float(np.mean(pred_values))),
                index=pred_values.index,
                dtype=float,
            )

        with patch(
            "eeg_pipeline.utils.analysis.stats.predictor_residual._fit_spline_model",
            side_effect=_fake_fit,
        ), patch(
            "eeg_pipeline.utils.analysis.stats.predictor_residual._predict_spline_model",
            side_effect=_fake_predict,
        ):
            prediction, residual, metadata = crossfit_predictor_outcome_curve(
                predictor,
                outcome,
                groups,
                config=cfg,
                method="spline",
            )

        self.assertEqual(fit_calls, [(4, 4), (4, 4)])
        self.assertEqual(str(metadata["status"]), "ok")
        self.assertTrue(np.isfinite(prediction.to_numpy(dtype=float)).all())
        self.assertTrue(np.isfinite(residual.to_numpy(dtype=float)).all())

    def test_predictor_residual_crossfit_inherits_main_method_when_not_overridden(self):
        from eeg_pipeline.utils.analysis.stats.predictor_residual import (
            crossfit_predictor_outcome_curve,
        )

        predictor = pd.Series(np.linspace(43.0, 46.5, 8))
        outcome = pd.Series(np.linspace(10.0, 45.0, 8))
        groups = pd.Series([1, 1, 1, 1, 2, 2, 2, 2])
        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "predictor_type": "continuous",
                    "predictor_residual": {
                        "method": "poly",
                        "min_samples": 3,
                        "crossfit": {
                            "n_splits": 2,
                        },
                    },
                }
            }
        )

        with patch(
            "eeg_pipeline.utils.analysis.stats.predictor_residual._fit_polynomial_model",
            return_value=(np.poly1d([1.0, 0.0]), {"model": "poly", "status": "ok"}),
        ) as poly_fit, patch(
            "eeg_pipeline.utils.analysis.stats.predictor_residual._predict_polynomial_model",
            side_effect=lambda _model, pred_values: pd.Series(
                np.zeros(len(pred_values)),
                index=pred_values.index,
                dtype=float,
            ),
        ):
            prediction, residual, metadata = crossfit_predictor_outcome_curve(
                predictor,
                outcome,
                groups,
                config=cfg,
            )

        self.assertEqual(poly_fit.call_count, 2)
        self.assertEqual(str(metadata["method"]), "poly")
        self.assertEqual(str(metadata["status"]), "ok")
        self.assertTrue(np.isfinite(prediction.to_numpy(dtype=float)).all())
        self.assertTrue(np.isfinite(residual.to_numpy(dtype=float)).all())

    def test_partial_correlation_returns_nan_for_rank_deficient_design(self):
        from eeg_pipeline.utils.analysis.stats.partial import partial_corr_xy_given_Z

        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])
        z = pd.DataFrame(
            {
                "z1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "z2": [2.0, 4.0, 6.0, 8.0, 10.0],
            }
        )

        r_value, p_value, n_obs = partial_corr_xy_given_Z(x, y, z, "pearson")
        self.assertTrue(np.isnan(float(r_value)))
        self.assertTrue(np.isnan(float(p_value)))
        self.assertEqual(int(n_obs), 0)

    def test_check_collinearity_ignores_constant_intercept_without_runtime_warning(self):
        from eeg_pipeline.utils.analysis.stats.partial import check_collinearity

        design_matrix = np.column_stack(
            [
                np.ones(6, dtype=float),
                np.arange(6, dtype=float),
                np.arange(6, dtype=float) * 2.0,
            ]
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            has_collinearity, max_correlation = check_collinearity(design_matrix)

        self.assertTrue(has_collinearity)
        self.assertAlmostEqual(max_correlation, 1.0)
        self.assertEqual(
            [
                warning
                for warning in caught
                if "invalid value encountered in divide" in str(warning.message)
            ],
            [],
        )

    def test_partial_correlation_does_not_emit_runtime_warning_for_intercept_column(self):
        from eeg_pipeline.utils.analysis.stats.partial import partial_corr_xy_given_Z

        x = pd.Series([0.2, 0.4, 0.8, 1.1, 1.5, 1.9], dtype=float)
        y = pd.Series([2.1, 2.4, 2.9, 3.1, 3.7, 4.2], dtype=float)
        z = pd.DataFrame({"run_order": [1, 2, 3, 4, 5, 6]}, dtype=float)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            r_value, p_value, n_obs = partial_corr_xy_given_Z(x, y, z, "pearson")

        self.assertTrue(np.isfinite(float(r_value)))
        self.assertTrue(np.isfinite(float(p_value)))
        self.assertEqual(int(n_obs), 6)
        self.assertEqual(
            [
                warning
                for warning in caught
                if "invalid value encountered in divide" in str(warning.message)
            ],
            [],
        )

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
            predictor_series=None,
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
            SimpleNamespace(control_predictor=False, control_trial_order=False),
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
            predictor_series=pd.Series([44.0, 44.5, 45.0, 45.5]),
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
            predictor_series=None,
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

    def test_correlate_effect_size_computes_run_mean_controlled_statistics(self):
        from eeg_pipeline.analysis.behavior.orchestration import _compute_single_effect_size

        df_trials = pd.DataFrame(
            {
                "run_id": np.repeat([1, 2, 3, 4, 5, 6], 2),
                "power_alpha": np.linspace(0.2, 2.4, 12),
                "rating": np.linspace(5.0, 60.0, 12),
            }
        )
        cov_df = pd.DataFrame({"trial_drift": np.repeat(np.arange(6, dtype=float), 2)})
        predictor = pd.Series(np.repeat([43.0, 44.5, 46.0, 45.0, 47.5, 49.0], 2))

        rec = _compute_single_effect_size(
            feat="power_alpha",
            target="rating",
            df_trials=df_trials,
            cov_df=cov_df,
            predictor_series=predictor,
            predictor_column="temperature",
            run_col="run_id",
            run_adjust_in_correlations=False,
            method="spearman",
            robust_method=None,
            method_label="spearman",
            min_samples=2,
            run_min_samples=3,
            want_raw=True,
            want_partial_cov=True,
            want_partial_predictor=True,
            want_partial_cov_predictor=True,
            want_run_mean=True,
            config=DotConfig({"behavior_analysis": {"statistics": {"predictor_control": "linear"}}}),
        )

        self.assertEqual(int(rec.get("n_runs", 0)), 6)
        self.assertIn("p_run_mean_partial_cov_predictor", rec)
        self.assertTrue(np.isfinite(float(rec.get("r_run_mean_partial_cov_predictor", np.nan))))

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
                target_col="rating",
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
                config=DotConfig({"behavior_analysis": {"predictor_column": "temperature"}}),
                logger=Mock(),
                use_block_permutation=False,
                n_perm=0,
                target_col="rating",
                control_predictor=True,
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
                target_col="rating",
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
            "eeg_pipeline.analysis.behavior.group_level.np.random.default_rng",
            wraps=np.random.default_rng,
        ) as rng_factory:
            run_group_level_correlations(
                subjects=["0001", "0002"],
                deriv_root=Path("/tmp"),
                config=DotConfig({"project": {"random_state": 13}}),
                logger=Mock(),
                use_block_permutation=False,
                n_perm=2,
                target_col="rating",
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
                target_col="rating",
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

        def _perm(n, rng, groups, scheme="shuffle", strict=True):
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
                scheme="shuffle",
            )

        self.assertTrue(calls)
        self.assertTrue(all(n == int(valid_feat.sum()) for n in calls))

    def test_regression_permutation_uses_configured_scheme(self):
        from eeg_pipeline.utils.analysis.stats.trialwise_regression import _compute_permutation_pvalues

        captured = {}

        def _perm(n, rng, groups, scheme="shuffle", strict=True):
            captured["scheme"] = scheme
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
                n_permutations=1,
                rng_seed=1,
                scheme="circular_shift",
            )

        self.assertEqual(str(captured.get("scheme")), "circular_shift")

    def test_regression_strict_non_iid_marks_missing_when_permutation_unavailable(self):
        from eeg_pipeline.utils.analysis.stats.trialwise_regression import run_trialwise_feature_regressions

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "regression": {
                        "outcome": "rating",
                        "include_predictor": False,
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

    def test_feature_table_alignment_reindexes_by_trial_id(self):
        from eeg_pipeline.context.behavior import BehaviorContext

        ctx = BehaviorContext(
            subject="0001",
            task="task",
            config=DotConfig({}),
            logger=Mock(),
            deriv_root=Path(tempfile.mkdtemp()),
            stats_dir=Path(tempfile.mkdtemp()),
        )
        ctx.aligned_events = pd.DataFrame({"trial_id": [2, 1], "rating": [10.0, 20.0]})

        feature_df = pd.DataFrame(
            {
                "trial_id": [1, 2],
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
        self.assertNotIn("trial_id", out.columns)
        self.assertEqual(report["power"]["status"], "reindexed_by_event_keys")
        self.assertEqual(report["power"]["keys"], ["trial_id"])

    def test_feature_table_alignment_rejects_missing_trial_id_by_default(self):
        from eeg_pipeline.context.behavior import BehaviorContext

        ctx = BehaviorContext(
            subject="0001",
            task="task",
            config=DotConfig({}),
            logger=Mock(),
            deriv_root=Path(tempfile.mkdtemp()),
            stats_dir=Path(tempfile.mkdtemp()),
        )
        ctx.aligned_events = pd.DataFrame({"trial_id": [1, 2], "rating": [10.0, 20.0]})

        feature_df = pd.DataFrame({"power_alpha": [0.1, 0.9]})
        report = {}
        out = ctx._align_single_feature_table(
            name="power",
            df=feature_df,
            base_index=ctx.aligned_events.index,
            alignment_report=report,
        )

        self.assertIsNone(out)
        self.assertEqual(report["power"]["reason"], "missing_trial_id")

    def test_feature_table_alignment_rejects_missing_trial_id_under_explicit_strict_config(self):
        from eeg_pipeline.context.behavior import BehaviorContext

        ctx = BehaviorContext(
            subject="0001",
            task="task",
            config=DotConfig({"behavior_analysis": {"trial_table": {"disallow_positional_alignment": True}}}),
            logger=Mock(),
            deriv_root=Path(tempfile.mkdtemp()),
            stats_dir=Path(tempfile.mkdtemp()),
        )
        ctx.aligned_events = pd.DataFrame({"trial_id": [1, 2], "rating": [10.0, 20.0]})

        feature_df = pd.DataFrame({"power_alpha": [0.1, 0.9]})
        report = {}
        out = ctx._align_single_feature_table(
            name="power",
            df=feature_df,
            base_index=ctx.aligned_events.index,
            alignment_report=report,
        )

        self.assertIsNone(out)
        self.assertEqual(report["power"]["reason"], "missing_trial_id")

    def test_feature_table_alignment_reindexes_by_trial_id_and_drops_key_columns(self):
        from eeg_pipeline.context.behavior import BehaviorContext

        ctx = BehaviorContext(
            subject="0001",
            task="task",
            config=DotConfig({}),
            logger=Mock(),
            deriv_root=Path(tempfile.mkdtemp()),
            stats_dir=Path(tempfile.mkdtemp()),
        )
        ctx.aligned_events = pd.DataFrame(
            {
                "trial_id": [100, 101, 102],
                "rating": [1.0, 2.0, 3.0],
            }
        )

        feature_df = pd.DataFrame(
            {
                "trial_id": [102, 100, 101],
                "power_alpha": [0.3, 0.1, 0.2],
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
        self.assertListEqual(out["power_alpha"].tolist(), [0.1, 0.2, 0.3])
        self.assertNotIn("trial_id", out.columns)
        self.assertEqual(report["power"]["status"], "reindexed_by_event_keys")
        self.assertEqual(report["power"]["keys"], ["trial_id"])

    def test_feature_table_alignment_requires_trial_id(self):
        from eeg_pipeline.context.behavior import BehaviorContext

        ctx = BehaviorContext(
            subject="0001",
            task="task",
            config=DotConfig({}),
            logger=Mock(),
            deriv_root=Path(tempfile.mkdtemp()),
            stats_dir=Path(tempfile.mkdtemp()),
        )
        ctx.aligned_events = pd.DataFrame({"trial_id": [1, 2], "rating": [10.0, 20.0]})

        feature_df = pd.DataFrame({"power_alpha": [0.1, 0.9]})
        report = {}
        out = ctx._align_single_feature_table(
            name="power",
            df=feature_df,
            base_index=ctx.aligned_events.index,
            alignment_report=report,
        )

        self.assertIsNone(out)
        self.assertEqual(report["power"]["reason"], "missing_trial_id")

    def test_icc_stage_updates_results_and_qc_metadata(self):
        from eeg_pipeline.analysis.behavior import orchestration as orch

        ctx = self._ctx(
            DotConfig(
                {
                    "behavior_analysis": {
                        "run_adjustment": {"column": "run_id"},
                        "icc": {"enabled": True},
                    }
                }
            )
        )
        runtime = orch.create_behavior_runtime()
        setattr(ctx, "_behavior_runtime", runtime)
        runtime.cache._trial_table_df = pd.DataFrame(
            {
                "run_id": [1, 1, 2, 2],
                "trial_index_within_group": [0, 1, 0, 1],
                "power_alpha": [0.1, 0.3, 0.2, 0.4],
            }
        )

        with patch(
            "eeg_pipeline.utils.analysis.stats.reliability.compute_icc",
            return_value=(0.75, 0.60, 0.85),
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._write_stats_table",
            side_effect=lambda _ctx, df, path: path,
        ), patch(
            "eeg_pipeline.analysis.behavior.orchestration._write_metadata_file",
            return_value=None,
        ):
            out = orch.stage_icc(ctx, SimpleNamespace(method_label="spearman"))

        self.assertEqual(list(out["feature"]), ["power_alpha"])
        self.assertAlmostEqual(float(out.iloc[0]["icc"]), 0.75, places=12)
        self.assertEqual(ctx.data_qc["icc_reliability"]["status"], "ok")

    def test_icc_stage_rejects_unstable_trial_alignment_across_runs(self):
        from eeg_pipeline.analysis.behavior import orchestration as orch

        ctx = self._ctx(
            DotConfig(
                {
                    "behavior_analysis": {
                        "run_adjustment": {"column": "run_id"},
                        "condition": {"compare_column": ""},
                        "icc": {"enabled": True},
                    },
                    "event_columns": {
                        "predictor": ["temperature"],
                        "outcome": ["rating"],
                        "binary_outcome": ["binary_outcome"],
                    },
                }
            )
        )
        runtime = orch.create_behavior_runtime()
        setattr(ctx, "_behavior_runtime", runtime)
        runtime.cache._trial_table_df = pd.DataFrame(
            {
                "run_id": [1, 1, 2, 2],
                "trial_index_within_group": [0, 1, 0, 1],
                "temperature": [44.0, 45.0, 45.0, 44.0],
                "power_alpha": [0.1, 0.3, 0.2, 0.4],
            }
        )

        with self.assertRaises(ValueError):
            orch.stage_icc(ctx, SimpleNamespace(method_label="spearman"))

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

    def test_group_correlations_use_configured_permutation_scheme(self):
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
        captured_schemes = []

        def _perm(n, rng, groups=None, scheme="shuffle", strict=True):
            captured_schemes.append(str(scheme))
            return np.arange(n, dtype=int)

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
            side_effect=_perm,
        ), patch(
            "eeg_pipeline.utils.analysis.stats.fdr.hierarchical_fdr",
            side_effect=lambda df, **_kwargs: df,
        ):
            out = run_group_level_correlations(
                subjects=["0001", "0002"],
                deriv_root=Path("/tmp"),
                config=DotConfig(
                    {
                        "behavior_analysis": {
                            "permutation": {"scheme": "circular_shift"},
                        }
                    }
                ),
                logger=Mock(),
                use_block_permutation=True,
                n_perm=2,
                target_col="rating",
            )

        self.assertFalse(out.empty)
        self.assertTrue(captured_schemes)
        self.assertEqual(set(captured_schemes), {"circular_shift"})
        self.assertEqual(str(out.iloc[0]["permutation_scheme"]), "circular_shift")

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
                target_col="rating",
            )

        self.assertFalse(out.empty)
        self.assertTrue(np.isnan(float(out.iloc[0]["p_perm"])))
        self.assertTrue(np.isnan(float(out.iloc[0]["p_primary"])))
        self.assertEqual(str(out.iloc[0]["p_primary_kind"]), "perm_missing_required")
        self.assertIn("failed", str(out.iloc[0]["permutation_method"]))

    def test_condition_stage_requires_positive_trialwise_permutation_count_in_non_iid_mode(self):
        from eeg_pipeline.analysis.behavior.orchestration import stage_condition_column

        cfg = DotConfig(
            {
                "behavior_analysis": {
                    "condition": {
                        "compare_column": "binary_outcome",
                        "primary_unit": "trial",
                        "compare_values": [1, 0],
                        "permutation": {"enabled": True, "n_permutations": 0},
                    },
                    "statistics": {"allow_iid_trials": False},
                    "run_adjustment": {"column": "run_id"},
                }
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

        with self.assertRaisesRegex(ValueError, "positive permutation count"):
            stage_condition_column(
                ctx,
                SimpleNamespace(fdr_alpha=0.05, n_jobs=1, min_samples=2),
                df_trials=df_trials,
                feature_cols=["power_alpha"],
            )

    def test_subject_report_reads_parquet_tables_and_filters_to_reportable_effects(self):
        from eeg_pipeline.analysis.behavior.stages.report import stage_report_impl

        ctx = self._ctx(DotConfig({"behavior_analysis": {"report": {"top_n": 5}}}))
        condition_dir = ctx.stats_dir / "condition_effects"
        condition_dir.mkdir(parents=True, exist_ok=True)
        condition_path = condition_dir / "condition_effects_column.parquet"
        condition_path.write_bytes(b"PAR1")

        def _get_stats_dir(_ctx, kind):
            out_dir = _ctx.stats_dir / kind
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir

        condition_df = pd.DataFrame(
            {
                "feature": ["power_big", "power_small"],
                "hedges_g": [1.2, 0.1],
                "reportable_effect": [True, False],
                "p_primary": [0.01, 0.02],
            }
        )

        with patch("eeg_pipeline.infra.tsv.read_table", return_value=condition_df):
            report_path = stage_report_impl(
                ctx,
                SimpleNamespace(
                    method="spearman",
                    method_label="spearman",
                    control_predictor=True,
                    control_trial_order=True,
                    fdr_alpha=0.05,
                ),
                feature_suffix_from_context_fn=lambda _ctx: "",
                get_config_int_fn=lambda _cfg, _key, default: default,
                load_trial_table_df_fn=lambda _ctx: pd.DataFrame({"power_alpha": [0.1], "rating": [10.0]}),
                get_stats_subfolder_fn=_get_stats_dir,
                feature_prefixes=("power_",),
            )

        text = report_path.read_text(encoding="utf-8")
        self.assertIn("condition_effects_column.parquet", text)
        self.assertIn("power_big", text)
        self.assertNotIn("power_small", text)

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
                SimpleNamespace(control_predictor=True, control_trial_order=True),
            )

        self.assertIsNotNone(design)
        self.assertGreaterEqual(len(design.targets), 1)
        self.assertEqual(design.targets[0], "predictor_residual_cv")
        self.assertIn("rating", design.targets)
        self.assertNotIn("temperature", design.targets)

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

    def test_group_level_uses_configured_run_column_for_block_permutation(self):
        from eeg_pipeline.analysis.behavior.orchestration import run_group_level_correlations

        df_a = pd.DataFrame(
            {
                "rating": np.linspace(10, 70, 12),
                "power_alpha": np.linspace(0.1, 1.2, 12),
                "acq_run": np.repeat([10, 20, 30], 4),
            }
        )
        df_b = pd.DataFrame(
            {
                "rating": np.linspace(12, 72, 12),
                "power_alpha": np.linspace(0.2, 1.3, 12),
                "acq_run": np.repeat([10, 20, 30], 4),
            }
        )
        captured_groups = []

        def _capture_perm(n, rng, groups, **kwargs):
            captured_groups.append(np.asarray(groups).copy())
            return np.arange(int(n), dtype=int)

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
            side_effect=_capture_perm,
        ), patch(
            "eeg_pipeline.utils.analysis.stats.fdr.hierarchical_fdr",
            side_effect=lambda df, **_kwargs: df,
        ):
            out = run_group_level_correlations(
                subjects=["0001", "0002"],
                deriv_root=Path("/tmp"),
                config=DotConfig(
                    {"behavior_analysis": {"run_adjustment": {"column": "acq_run"}}}
                ),
                logger=Mock(),
                use_block_permutation=True,
                n_perm=4,
                target_col="rating",
            )

        self.assertFalse(out.empty)
        self.assertTrue(captured_groups)
        self.assertTrue(
            all(set(np.unique(group_values)) == {10, 20, 30} for group_values in captured_groups)
        )

    def test_group_level_partial_permutation_state_matches_partial_correlation(self):
        from eeg_pipeline.analysis.behavior.group_level import (
            _build_subject_partial_permutation_state,
            _compute_permuted_subject_partial_r,
        )
        from eeg_pipeline.utils.analysis.stats.partial import compute_partial_corr

        cov = pd.DataFrame({"trial_index": [1, 2, 3, 4, 5, 6]}, dtype=float)
        x = pd.Series([0.5, 1.2, 1.9, 2.4, 3.8, 3.1], dtype=float)
        y = pd.Series([1.0, 1.8, 2.9, 2.2, 3.3, 4.1], dtype=float)

        state = _build_subject_partial_permutation_state(x, y, cov, method="spearman")

        self.assertIsNotNone(state)
        r_ref, _p_ref, _n_ref = compute_partial_corr(x, y, cov, method="spearman")
        self.assertAlmostEqual(float(state["r_observed"]), float(r_ref), places=12)

        identity_perm = np.arange(len(state["y_residuals"]), dtype=int)
        r_identity = _compute_permuted_subject_partial_r(state, identity_perm)
        self.assertAlmostEqual(float(r_identity), float(r_ref), places=12)

    def test_temporal_metric_helper_applies_covariates_and_cluster_outputs(self):
        from eeg_pipeline.utils.analysis.stats.temporal import _compute_metric_records_with_cluster

        cov_vals = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=float)
        partial_calls = []

        def _fake_partial(x, y, Z, method):
            partial_calls.append((len(x), list(Z.columns), method))
            return 0.5, 0.04, len(x)

        def _fake_cluster(**kwargs):
            n_windows = kwargs["correlations"].shape[0]
            return (
                np.ones((n_windows, 1), dtype=int),
                np.full((n_windows, 1), 0.03, dtype=float),
                np.ones((n_windows, 1), dtype=bool),
                [],
                [],
                2.0,
            )

        with patch(
            "eeg_pipeline.utils.analysis.stats.temporal.compute_partial_corr",
            side_effect=_fake_partial,
        ), patch(
            "eeg_pipeline.utils.analysis.stats.temporal.compute_cluster_correction_2d",
            side_effect=_fake_cluster,
        ):
            records = _compute_metric_records_with_cluster(
                band_metrics=[
                    (
                        "alpha",
                        [
                            np.array([[0.1], [0.2], [0.3], [0.4]], dtype=float),
                            np.array([[0.2], [0.3], [0.4], [0.5]], dtype=float),
                        ],
                    )
                ],
                y_condition=np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
                ch_names=["Cz"],
                win_s=np.array([0.0, 0.1], dtype=float),
                win_e=np.array([0.1, 0.2], dtype=float),
                condition_name="all",
                feature_name="itpc",
                use_spearman=True,
                logger=Mock(),
                config=DotConfig(
                    {
                        "behavior_analysis": {
                            "cluster": {"n_permutations": 10},
                            "statistics": {
                                "min_samples_per_covariate": 1,
                                "partial_corr_base_samples": 3,
                            },
                        }
                    }
                ),
                cov_vals=cov_vals,
                cov_cols=["trial_index"],
                groups=np.array([1, 1, 2, 2], dtype=int),
                req_samples=4,
            )

        self.assertEqual(len(partial_calls), 2)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(np.isclose(float(record["p_cluster"]), 0.03) for record in records))
        self.assertTrue(all(bool(record["cluster_significant"]) for record in records))


if __name__ == "__main__":
    unittest.main()
