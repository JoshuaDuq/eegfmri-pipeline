import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from tests.pipelines_test_utils import DotConfig, DummyProgress, NoopBatchProgress, NoopProgress

_DummyProgress = DummyProgress
_NoopBatchProgress = NoopBatchProgress
_NoopProgress = NoopProgress


def _make_package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _make_module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _make_pipeline_base_class() -> type:
    class _PipelineBase:
        def __init__(self, name, config=None):
            self.name = name
            self.config = config
            self.logger = Mock()
            self.deriv_root = Path(tempfile.mkdtemp())

        def _setup(self):
            return None

        def _create_run_metadata_context(self, *, subjects, task, kwargs):
            return {
                "run_id": "test-run",
                "started_at": 0,
                "task": task,
                "subjects": list(subjects),
                "specifications": {k: v for k, v in kwargs.items() if k != "progress"},
            }

        def _write_run_metadata(self, run_context, *, status, error=None, outputs=None, summary=None):
            metadata_dir = Path(self.deriv_root) / "logs" / "run_metadata" / self.name
            metadata_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "status": status,
                "task": run_context.get("task"),
                "subjects": run_context.get("subjects", []),
                "specifications": run_context.get("specifications", {}),
                "outputs": outputs or {},
                "summary": summary or {},
            }
            if error:
                payload["error"] = error
            out_path = metadata_dir / "run_test-run.json"
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return out_path

    return _PipelineBase


class _StubBehaviorContext:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _make_pipeline_base_class() -> type:
    class _PipelineBase:
        def __init__(self, name, config=None):
            self.name = name
            self.config = config
            self.logger = Mock()
            self.deriv_root = Path(tempfile.mkdtemp())

        def _create_run_metadata_context(self, *, subjects, task, kwargs):
            return {
                "run_id": "test-run",
                "started_at": 0,
                "task": task,
                "subjects": list(subjects),
                "specifications": {k: v for k, v in kwargs.items() if k != "progress"},
            }

        def _write_run_metadata(self, run_context, *, status, error=None, outputs=None, summary=None):
            metadata_dir = Path(self.deriv_root) / "logs" / "run_metadata" / self.name
            metadata_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "status": status,
                "task": run_context.get("task"),
                "subjects": run_context.get("subjects", []),
                "specifications": run_context.get("specifications", {}),
                "outputs": outputs or {},
                "summary": summary or {},
            }
            if error:
                payload["error"] = error
            out_path = metadata_dir / "run_test-run.json"
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return out_path

    return _PipelineBase


def _behavior_import_stubs() -> dict[str, types.ModuleType]:
    def _get_config_value(config, key, default=None):
        return config.get(key, default) if hasattr(config, "get") else default

    def _require_config_value(config, key):
        value = _get_config_value(config, key, None)
        if value is None:
            defaults = {
                "behavior_analysis.statistics.default_n_bootstrap": 0,
                "behavior_analysis.statistics.n_permutations": 0,
                "behavior_analysis.min_samples.default": 10,
                "behavior_analysis.predictor_control_enabled": True,
                "behavior_analysis.control_trial_order": True,
                "behavior_analysis.correlations.compute_change_scores": True,
                "behavior_analysis.statistics.compute_reliability": False,
                "behavior_analysis.correlations.compute_bayes_factors": False,
                "behavior_analysis.correlations.loso_stability": True,
                "behavior_analysis.correlations.types": ["partial_cov_predictor"],
                "behavior_analysis.trial_table.enabled": True,
                "behavior_analysis.predictor_residual.enabled": True,
                "behavior_analysis.regression.enabled": False,
                "behavior_analysis.icc.enabled": True,
                "behavior_analysis.validation.enabled": True,
                "behavior_analysis.correlations.enabled": True,
                "behavior_analysis.group_level.multilevel_correlations.enabled": False,
                "behavior_analysis.condition.enabled": True,
                "behavior_analysis.temporal.enabled": True,
                "behavior_analysis.cluster.enabled": False,
                "behavior_analysis.statistics.fdr_alpha": 0.05,
                "behavior_analysis.n_jobs": -1,
                "behavior_analysis.condition.effect_size_threshold": 0.5,
                "behavior_analysis.temporal.time_resolution_ms": 50,
                "behavior_analysis.temporal.smooth_window_ms": 100,
                "behavior_analysis.cluster.forming_threshold": 0.05,
                "behavior_analysis.cluster.min_cluster_size": 2,
                "behavior_analysis.cluster.tail": 0,
            }
            value = defaults.get(key)
        if value is None:
            raise KeyError(key)
        return value

    return {
        "eeg_pipeline.analysis": _make_package("eeg_pipeline.analysis"),
        "eeg_pipeline.analysis.behavior": _make_package("eeg_pipeline.analysis.behavior"),
        "eeg_pipeline.analysis.behavior.config_resolver": _make_module(
            "eeg_pipeline.analysis.behavior.config_resolver",
            resolve_correlation_method=lambda config: config.get(
                "behavior_analysis.statistics.correlation_method", "spearman"
            )
            if hasattr(config, "get")
            else "spearman",
        ),
        "eeg_pipeline.analysis.behavior.stage_catalog": _make_module(
            "eeg_pipeline.analysis.behavior.stage_catalog",
            COMPUTATION_TO_PIPELINE_ATTR={"icc": "run_icc", "regression": "run_regression"},
            apply_computation_flags_impl=lambda *args, **kwargs: None,
        ),
        "eeg_pipeline.analysis.behavior.orchestration": _make_module(
            "eeg_pipeline.analysis.behavior.orchestration",
            create_behavior_runtime=lambda *args, **kwargs: None,
            run_behavior_stages=lambda *args, **kwargs: None,
            write_analysis_metadata=lambda *args, **kwargs: None,
            write_outputs_manifest=lambda *args, **kwargs: None,
            get_behavior_output_dir=lambda *args, **kwargs: Path(tempfile.mkdtemp()),
            run_group_level_analysis=lambda *args, **kwargs: None,
        ),
        "eeg_pipeline.context": _make_package("eeg_pipeline.context"),
        "eeg_pipeline.context.behavior": _make_module(
            "eeg_pipeline.context.behavior",
            BehaviorContext=_StubBehaviorContext,
        ),
        "eeg_pipeline.pipelines.base": _make_module(
            "eeg_pipeline.pipelines.base",
            PipelineBase=_make_pipeline_base_class(),
        ),
        "eeg_pipeline.pipelines.progress": _make_module(
            "eeg_pipeline.pipelines.progress",
            ensure_progress_reporter=lambda progress=None: progress or _NoopProgress(),
        ),
        "eeg_pipeline.infra": _make_package("eeg_pipeline.infra"),
        "eeg_pipeline.infra.paths": _make_module(
            "eeg_pipeline.infra.paths",
            deriv_stats_path=lambda *args, **kwargs: Path(tempfile.mkdtemp()),
            ensure_dir=lambda path: Path(path).mkdir(parents=True, exist_ok=True),
        ),
        "eeg_pipeline.utils": _make_package("eeg_pipeline.utils"),
        "eeg_pipeline.utils.analysis": _make_package("eeg_pipeline.utils.analysis"),
        "eeg_pipeline.utils.analysis.stats": _make_package("eeg_pipeline.utils.analysis.stats"),
        "eeg_pipeline.utils.analysis.stats.base": _make_module(
            "eeg_pipeline.utils.analysis.stats.base",
            get_subject_seed=lambda *args, **kwargs: 0,
        ),
        "eeg_pipeline.utils.analysis.stats.correlation": _make_module(
            "eeg_pipeline.utils.analysis.stats.correlation",
            format_correlation_method_label=lambda method, robust_method=None: method,
            normalize_robust_correlation_method=lambda value, default=None, strict=False: value.strip() if isinstance(value, str) else value,
        ),
        "eeg_pipeline.utils.config": _make_package("eeg_pipeline.utils.config"),
        "eeg_pipeline.utils.config.behavior_loader": _make_module(
            "eeg_pipeline.utils.config.behavior_loader",
            ensure_behavior_config=lambda config: config,
        ),
        "eeg_pipeline.utils.config.loader": _make_module(
            "eeg_pipeline.utils.config.loader",
            get_config_value=_get_config_value,
            require_config_value=_require_config_value,
        ),
        "eeg_pipeline.analysis.behavior.trial_table_helpers": _make_module(
            "eeg_pipeline.analysis.behavior.trial_table_helpers",
            find_trial_table_path=lambda *args, **kwargs: Path("/tmp/trials.parquet"),
        ),
    }


class _BehaviorImportMixin:
    def setUp(self):
        sys.modules.pop("eeg_pipeline.pipelines.behavior", None)
        patcher = patch.dict(sys.modules, _behavior_import_stubs())
        patcher.start()
        import eeg_pipeline

        setattr(eeg_pipeline, "infra", sys.modules["eeg_pipeline.infra"])
        self.addCleanup(patcher.stop)


def _behavior_process_config() -> DotConfig:
    return DotConfig(
        {
            "project": {"task": "task"},
            "behavior_analysis": {
                "statistics": {"base_seed": 42},
                "output": {"also_save_csv": False, "overwrite": True},
            },
        }
    )


class TestBehaviorDeep(_BehaviorImportMixin, unittest.TestCase):
        def test_behavior_process_subject_success_path(self):
            from eeg_pipeline.pipelines.behavior import BehaviorPipeline

            p = object.__new__(BehaviorPipeline)
            p.pipeline_config = SimpleNamespace(
                method="spearman",
                bootstrap=0,
                n_permutations=0,
                control_predictor=True,
                control_trial_order=True,
                compute_change_scores=True,
                compute_reliability=False,
                run_correlations=True,
                run_condition_comparison=True,
                run_temporal_correlations=True,
                run_cluster_tests=True,
            )
            p.feature_categories = None
            p.feature_files = None
            p.computation_features = {}
            p.deriv_root = Path(tempfile.mkdtemp())
            p.config = _behavior_process_config()
            p.logger = Mock()

            fake_paths = types.SimpleNamespace(
                deriv_stats_path=lambda deriv_root, subject: Path(tempfile.mkdtemp()),
                ensure_dir=lambda path: None,
            )
            fake_logger = Mock()
            fake_logging = types.SimpleNamespace(get_subject_logger=lambda name, subject: fake_logger)
            progress = SimpleNamespace(
                subject_start=lambda *a, **k: None,
                subject_done=lambda *a, **k: None,
                error=lambda *a, **k: None,
            )
            fake_cli = types.SimpleNamespace(ProgressReporter=lambda enabled=False: progress)
            outdir = Path(tempfile.mkdtemp())
            fake_orch = types.SimpleNamespace(_cache={})
            with patch.dict(
                sys.modules,
                {
                    "eeg_pipeline.infra.paths": fake_paths,
                    "eeg_pipeline.infra.logging": fake_logging,
                    "eeg_pipeline.cli.common": fake_cli,
                    "eeg_pipeline.analysis.behavior.orchestration": fake_orch,
                },
            ), patch(
                "eeg_pipeline.pipelines.behavior.BehaviorContext",
                side_effect=lambda **kwargs: SimpleNamespace(**kwargs, data_qc={}),
            ), patch(
                "eeg_pipeline.pipelines.behavior.run_behavior_stages", side_effect=lambda **kwargs: None
            ), patch(
                "eeg_pipeline.pipelines.behavior.write_outputs_manifest",
                side_effect=lambda *a, **k: outdir / "manifest.json",
            ), patch(
                "eeg_pipeline.pipelines.behavior.get_behavior_output_dir",
                side_effect=lambda *a, **k: outdir,
            ), patch(
                "eeg_pipeline.pipelines.behavior._write_analysis_metadata_impl",
            ):
                out = p.process_subject("0001")
            self.assertEqual(out.subject, "0001")
            self.assertTrue((outdir / "summary.json").exists())

        def test_behavior_results_summary_rich(self):
            from eeg_pipeline.pipelines.behavior import BehaviorPipelineResults

            res = BehaviorPipelineResults(
                subject="0001",
                correlations=pd.DataFrame({"p_raw": [0.01, 0.2], "p_primary": [0.04, 0.6], "q_global": [0.03, 0.2]}),
                condition_effects=pd.DataFrame({"p_value": [0.02], "q_global": [0.03]}),
                regression=pd.DataFrame({"p_primary": [0.01], "q_global": [0.04], "hedges_g": [0.9]}),
                icc=pd.DataFrame({"feature": ["power_alpha"], "icc": [0.72]}),
                tf={"n_tests": 2, "n_sig_raw": 1, "n_sig_fdr": 1},
                temporal={"n_tests": 3, "n_sig_raw": 2, "n_sig_fdr": 1},
                cluster={"alpha": {"cluster_records": [{"q_global": 0.04}, {"p_value": 0.2}]}} ,
            )
            summary = res.to_summary()
            self.assertGreater(summary["n_features"], 0)
            self.assertGreater(summary["n_sig_raw"], 0)
            self.assertGreater(summary["n_sig_fdr"], 0)
            self.assertEqual(summary["n_clusters"], 2)
            self.assertEqual(summary["n_icc_features"], 1)

        def test_behavior_results_summary_counts_nested_temporal_outputs(self):
            from eeg_pipeline.pipelines.behavior import BehaviorPipelineResults

            res = BehaviorPipelineResults(
                subject="0001",
                temporal={
                    "power": {"n_tests": 7, "n_sig_raw": 3, "n_sig_fdr": 1},
                    "itpc": None,
                    "erds": {"n_tests": 5, "n_sig_raw": 2, "n_sig_fdr": 0},
                },
            )

            summary = res.to_summary()

            self.assertEqual(summary["n_temporal_tests"], 12)
            self.assertEqual(summary["n_features"], 12)
            self.assertEqual(summary["n_sig_raw"], 5)
            self.assertEqual(summary["n_sig_fdr"], 1)

        def test_behavior_process_subject_failure_path(self):
            from eeg_pipeline.pipelines.behavior import BehaviorPipeline

            p = object.__new__(BehaviorPipeline)
            p.pipeline_config = SimpleNamespace(
                method="spearman",
                bootstrap=0,
                n_permutations=0,
                control_predictor=True,
                control_trial_order=True,
                compute_change_scores=True,
                compute_reliability=False,
                run_correlations=True,
                run_condition_comparison=True,
                run_temporal_correlations=True,
                run_cluster_tests=True,
            )
            p.feature_categories = None
            p.feature_files = None
            p.computation_features = {}
            p.deriv_root = Path(tempfile.mkdtemp())
            p.config = _behavior_process_config()
            p.logger = Mock()

            fake_paths = types.SimpleNamespace(
                deriv_stats_path=lambda deriv_root, subject: Path(tempfile.mkdtemp()),
                ensure_dir=lambda path: None,
            )
            fake_logging = types.SimpleNamespace(get_subject_logger=lambda name, subject: Mock())
            fake_cli = types.SimpleNamespace(ProgressReporter=lambda enabled=False: SimpleNamespace(
                subject_start=lambda *a, **k: None,
                subject_done=lambda *a, **k: None,
                error=lambda *a, **k: None,
            ))
            fake_orch = types.SimpleNamespace(_cache={}, run_behavior_stages=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
            progress = SimpleNamespace(
                subject_start=lambda *a, **k: None,
                subject_done=lambda *a, **k: None,
                error=lambda *a, **k: None,
            )

            with patch.dict(
                sys.modules,
                {
                    "eeg_pipeline.infra.paths": fake_paths,
                    "eeg_pipeline.infra.logging": fake_logging,
                    "eeg_pipeline.cli.common": fake_cli,
                    "eeg_pipeline.analysis.behavior.orchestration": fake_orch,
                },
            ), patch(
                "eeg_pipeline.pipelines.behavior.run_behavior_stages",
                side_effect=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
            ):
                with self.assertRaises(RuntimeError):
                    p.process_subject("0001", progress=progress)

class TestBehaviorCompletion(_BehaviorImportMixin, unittest.TestCase):
        def test_behavior_init_and_group_level_logging_branches(self):
            from eeg_pipeline.pipelines.behavior import BehaviorPipeline, BehaviorPipelineConfig
            import pandas as pd

            cfg = DotConfig({})
            pcfg = BehaviorPipelineConfig()
            with patch("eeg_pipeline.pipelines.behavior.PipelineBase.__init__", lambda self, name, config=None: (setattr(self, "config", config or cfg), setattr(self, "logger", Mock()), setattr(self, "deriv_root", Path(tempfile.mkdtemp())))):
                b = BehaviorPipeline(config=cfg, pipeline_config=pcfg, computations=["icc"])
            self.assertTrue(b.pipeline_config.run_icc)

            fake_result = SimpleNamespace(
                    multilevel_correlations=pd.DataFrame({"q_within_family": [0.01, 0.2]}),
                )
            with patch(
                "eeg_pipeline.analysis.behavior.orchestration.run_group_level_analysis",
                return_value=fake_result,
            ), patch(
                "eeg_pipeline.analysis.behavior.trial_table_helpers.find_trial_table_path",
                return_value=Path("/tmp/trials.parquet"),
            ):
                out = b.run_group_level(["0001", "0002"], run_multilevel_correlations=True)
            self.assertIs(out, fake_result)

        def test_behavior_group_level_skips_by_default_when_not_selected(self):
            from eeg_pipeline.pipelines.behavior import BehaviorPipeline, BehaviorPipelineConfig

            cfg = DotConfig({})
            pcfg = BehaviorPipelineConfig(run_multilevel_correlations=False)
            with patch(
                "eeg_pipeline.pipelines.behavior.PipelineBase.__init__",
                lambda self, name, config=None: (
                    setattr(self, "config", config or cfg),
                    setattr(self, "logger", Mock()),
                    setattr(self, "deriv_root", Path(tempfile.mkdtemp())),
                ),
            ):
                b = BehaviorPipeline(config=cfg, pipeline_config=pcfg)

            with patch(
                "eeg_pipeline.analysis.behavior.orchestration.run_group_level_analysis",
            ) as run_group_level_analysis_mock:
                out = b.run_group_level(["0001", "0002"])

            self.assertIsNone(out)
            run_group_level_analysis_mock.assert_not_called()

        def test_behavior_group_level_forwards_feature_file_selection(self):
            from eeg_pipeline.pipelines.behavior import BehaviorPipeline, BehaviorPipelineConfig

            cfg = DotConfig({})
            pcfg = BehaviorPipelineConfig()
            with patch(
                "eeg_pipeline.pipelines.behavior.PipelineBase.__init__",
                lambda self, name, config=None: (
                    setattr(self, "config", config or cfg),
                    setattr(self, "logger", Mock()),
                    setattr(self, "deriv_root", Path(tempfile.mkdtemp())),
                ),
            ):
                b = BehaviorPipeline(
                    config=cfg,
                    pipeline_config=pcfg,
                    feature_files=["power"],
                )

            fake_result = SimpleNamespace(multilevel_correlations=None)
            with patch(
                "eeg_pipeline.infra.paths.deriv_stats_path",
                side_effect=lambda _root, sub: Path(f"/tmp/{sub}"),
            ), patch(
                "eeg_pipeline.analysis.behavior.trial_table_helpers.find_trial_table_path",
                return_value=Path("/tmp/trials_power.parquet"),
            ), patch(
                "eeg_pipeline.analysis.behavior.orchestration.run_group_level_analysis",
                return_value=fake_result,
            ) as run_group_level_analysis_mock:
                out = b.run_group_level(["0001", "0002"], run_multilevel_correlations=True)

            self.assertIs(out, fake_result)
            self.assertEqual(
                run_group_level_analysis_mock.call_args.kwargs["feature_files"],
                ["power"],
            )

        def test_behavior_group_level_requires_existing_trial_tables(self):
            from eeg_pipeline.pipelines.behavior import BehaviorPipeline, BehaviorPipelineConfig

            cfg = DotConfig({})
            pcfg = BehaviorPipelineConfig(run_multilevel_correlations=True)
            with patch(
                "eeg_pipeline.pipelines.behavior.PipelineBase.__init__",
                lambda self, name, config=None: (
                    setattr(self, "config", config or cfg),
                    setattr(self, "logger", Mock()),
                    setattr(self, "deriv_root", Path(tempfile.mkdtemp())),
                ),
            ):
                b = BehaviorPipeline(
                    config=cfg,
                    pipeline_config=pcfg,
                    feature_files=["power"],
                )

            with patch(
                "eeg_pipeline.infra.paths.deriv_stats_path",
                side_effect=lambda _root, sub: Path(f"/tmp/{sub}"),
            ), patch(
                "eeg_pipeline.analysis.behavior.trial_table_helpers.find_trial_table_path",
                return_value=None,
            ), patch(
                "eeg_pipeline.analysis.behavior.orchestration.run_group_level_analysis",
            ) as run_group_level_analysis_mock:
                with self.assertRaisesRegex(
                    ValueError,
                    "Group-level multilevel_correlations requires saved trial tables",
                ):
                    b.run_group_level(["0001", "0002"])

            run_group_level_analysis_mock.assert_not_called()

class TestBehaviorGapfill(_BehaviorImportMixin, unittest.TestCase):
        def test_behavior_small_helpers_cover_counts_and_defaults(self):
            with patch.dict(sys.modules, _behavior_import_stubs()):
                from eeg_pipeline.pipelines.behavior import (
                    _get_optional_int,
                    _summarize_nested_result_counts,
                )

                cfg = DotConfig({"behavior_analysis": {"n_jobs": "4"}})
                self.assertEqual(_get_optional_int(cfg, "behavior_analysis.n_jobs", None), 4)
                self.assertIsNone(_get_optional_int(cfg, "behavior_analysis.missing", None))

                self.assertEqual(_summarize_nested_result_counts(None), (0, 0, 0))
                self.assertEqual(
                    _summarize_nested_result_counts({"n_tests": 5, "n_sig_raw": 2, "n_sig_fdr": 1}),
                    (5, 2, 1),
                )
                self.assertEqual(
                    _summarize_nested_result_counts(
                        {
                            "alpha": {"n_tests": 3, "n_sig_raw": 1, "n_sig_fdr": 1},
                            "beta": {"n_tests": 2, "n_sig_raw": 0, "n_sig_fdr": 0},
                        }
                    ),
                    (5, 1, 1),
                    )

        def test_behavior_computation_flags_expand_bundles_and_warn(self):
            with patch.dict(sys.modules, _behavior_import_stubs()):
                from eeg_pipeline.pipelines.behavior import (
                    BEHAVIOR_COMPUTATION_BUNDLES,
                    _resolve_behavior_computation_flags,
                )

                original_bundles = dict(BEHAVIOR_COMPUTATION_BUNDLES)
                try:
                    BEHAVIOR_COMPUTATION_BUNDLES.clear()
                    BEHAVIOR_COMPUTATION_BUNDLES["bundle"] = ["icc", "regression"]

                    logger = Mock()
                    flags = _resolve_behavior_computation_flags(["bundle", "unknown"], logger=logger)

                    self.assertTrue(flags["icc"])
                    self.assertTrue(flags["regression"])
                    self.assertFalse(flags.get("run_cluster_tests", False))
                    self.assertTrue(logger.warning.called)
                finally:
                    BEHAVIOR_COMPUTATION_BUNDLES.clear()
                    BEHAVIOR_COMPUTATION_BUNDLES.update(original_bundles)

        def test_behavior_helpers_and_init_logging(self):
            from eeg_pipeline.pipelines.behavior import (
                _resolve_behavior_computation_flags,
                BehaviorPipelineConfig,
                _extract_p_value_column,
                _count_significant,
                BehaviorPipelineResults,
                BehaviorPipeline,
            )

            self.assertIsNone(_resolve_behavior_computation_flags(None))

            cfg = DotConfig({"behavior_analysis": {"statistics": {"correlation_method": "pearson"}, "robust_correlation": " winsorized "}})
            pcfg = BehaviorPipelineConfig.from_config(cfg)
            self.assertEqual(pcfg.method, "pearson")
            self.assertEqual(pcfg.robust_method, "winsorized")

            df = pd.DataFrame({"q_value": [0.01]})
            self.assertIsNotNone(_extract_p_value_column(df, ["p"], ["q_value"]))
            self.assertIsNone(_extract_p_value_column(df, ["p"], ["q"]))
            self.assertEqual(_count_significant(None), 0)

            s = BehaviorPipelineResults(subject="0001", trial_table_path="/a")
            out = s.to_summary()
            self.assertEqual(out["trial_table_path"], "/a")

            with patch(
                "eeg_pipeline.pipelines.behavior.PipelineBase.__init__",
                lambda self, name, config=None: (
                    setattr(self, "name", name),
                    setattr(self, "config", config or DotConfig({})),
                    setattr(self, "logger", Mock()),
                    setattr(self, "deriv_root", Path(tempfile.mkdtemp())),
                ),
            ):
                p = BehaviorPipeline(
                    config=DotConfig({}),
                    pipeline_config=BehaviorPipelineConfig(),
                    computations=["icc"],
                    feature_categories=["power"],
                    computation_features={"regression": ["power_alpha"]},
                )
            self.assertEqual(p.name, "behavior_analysis")

        def test_behavior_process_subject_cluster_logs(self):
            from eeg_pipeline.pipelines.behavior import BehaviorPipeline, BehaviorPipelineConfig

            p = object.__new__(BehaviorPipeline)
            p.pipeline_config = BehaviorPipelineConfig(
                run_correlations=True,
                run_condition_comparison=True,
                run_temporal_correlations=True,
                run_cluster_tests=True,
            )
            p.feature_categories = None
            p.feature_files = None
            p.computation_features = {}
            p.deriv_root = Path(tempfile.mkdtemp())
            p.config = _behavior_process_config()
            p.logger = Mock()

            stats_dir = Path(tempfile.mkdtemp())
            summary_dir = Path(tempfile.mkdtemp())
            fake_paths = types.SimpleNamespace(deriv_stats_path=lambda deriv_root, subject: stats_dir, ensure_dir=lambda path: None)
            subject_log = Mock()
            fake_logging = types.SimpleNamespace(get_subject_logger=lambda name, subject: subject_log)
            fake_cli = types.SimpleNamespace(
                ProgressReporter=lambda enabled=False: SimpleNamespace(
                    subject_start=lambda *a, **k: None,
                    subject_done=lambda *a, **k: None,
                    error=lambda *a, **k: None,
                )
            )

            def _run_stages(ctx, pipeline_config, results, progress):
                results.cluster = {"alpha": {"cluster_records": [{"q_global": 0.01}, {"q_global": 0.2}]}}

            fake_orch = types.SimpleNamespace(_cache={})

            with patch.dict(
                sys.modules,
                {
                    "eeg_pipeline.infra.paths": fake_paths,
                    "eeg_pipeline.infra.logging": fake_logging,
                    "eeg_pipeline.cli.common": fake_cli,
                    "eeg_pipeline.analysis.behavior.orchestration": fake_orch,
                },
            ), patch("eeg_pipeline.pipelines.behavior.run_behavior_stages", side_effect=_run_stages), patch(
                "eeg_pipeline.pipelines.behavior.write_outputs_manifest", return_value=stats_dir / "manifest.json"
            ), patch(
                "eeg_pipeline.pipelines.behavior.get_behavior_output_dir", return_value=summary_dir
            ), patch(
                "eeg_pipeline.pipelines.behavior._write_analysis_metadata_impl"
            ):
                p.process_subject("0001")

        def test_behavior_process_subject_cluster_output_log_lines(self):
            from eeg_pipeline.pipelines.behavior import BehaviorPipeline, BehaviorPipelineConfig, BehaviorPipelineResults

            p = object.__new__(BehaviorPipeline)
            p.pipeline_config = BehaviorPipelineConfig()
            p.feature_categories = None
            p.feature_files = None
            p.computation_features = {}
            p.deriv_root = Path(tempfile.mkdtemp())
            p.config = _behavior_process_config()
            p.logger = Mock()

            stats_dir = Path(tempfile.mkdtemp())
            summary_dir = Path(tempfile.mkdtemp())
            fake_paths = types.SimpleNamespace(deriv_stats_path=lambda deriv_root, subject: stats_dir, ensure_dir=lambda path: None)
            subject_log = Mock()
            fake_logging = types.SimpleNamespace(get_subject_logger=lambda name, subject: subject_log)
            fake_cli = types.SimpleNamespace(
                ProgressReporter=lambda enabled=False: SimpleNamespace(
                    subject_start=lambda *a, **k: None,
                    subject_done=lambda *a, **k: None,
                    error=lambda *a, **k: None,
                )
            )
            fake_orch = types.SimpleNamespace(_cache={})

            with patch.dict(
                sys.modules,
                {
                    "eeg_pipeline.infra.paths": fake_paths,
                    "eeg_pipeline.infra.logging": fake_logging,
                    "eeg_pipeline.cli.common": fake_cli,
                    "eeg_pipeline.analysis.behavior.orchestration": fake_orch,
                },
            ), patch("eeg_pipeline.pipelines.behavior.run_behavior_stages", side_effect=lambda **kwargs: None), patch(
                "eeg_pipeline.pipelines.behavior.write_outputs_manifest", return_value=stats_dir / "manifest.json"
            ), patch(
                "eeg_pipeline.pipelines.behavior.get_behavior_output_dir", return_value=summary_dir
            ), patch(
                "eeg_pipeline.pipelines.behavior._write_analysis_metadata_impl"
            ), patch.object(
                BehaviorPipelineResults,
                "to_summary",
                return_value={"n_features": 1, "n_sig_raw": 1, "n_sig_controlled": 1, "n_sig_fdr": 1, "n_clusters": 2, "n_sig_clusters": 1},
            ):
                p.process_subject("0001")
            self.assertTrue(
                any("Clusters identified" in str(call.args[0]) for call in subject_log.info.call_args_list)
            )
