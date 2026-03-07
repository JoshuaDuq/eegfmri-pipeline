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


class TestBehaviorDeep(unittest.TestCase):
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
            p.config = DotConfig({"project": {"task": "task"}})
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
            p.config = DotConfig({"project": {"task": "task"}})
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

            with patch.dict(
                sys.modules,
                {
                    "eeg_pipeline.infra.paths": fake_paths,
                    "eeg_pipeline.infra.logging": fake_logging,
                    "eeg_pipeline.cli.common": fake_cli,
                    "eeg_pipeline.analysis.behavior.orchestration": fake_orch,
                },
            ):
                with self.assertRaises(RuntimeError):
                    p.process_subject("0001")

class TestBehaviorCompletion(unittest.TestCase):
        def test_behavior_init_and_group_level_logging_branches(self):
            from eeg_pipeline.pipelines.behavior import BehaviorPipeline, BehaviorPipelineConfig
            import pandas as pd

            cfg = DotConfig({})
            pcfg = BehaviorPipelineConfig()
            with patch("eeg_pipeline.pipelines.behavior.PipelineBase.__init__", lambda self, name, config=None: (setattr(self, "config", config or cfg), setattr(self, "logger", Mock()), setattr(self, "deriv_root", Path(tempfile.mkdtemp())))):
                b = BehaviorPipeline(config=cfg, pipeline_config=pcfg, computations=["report"])
            self.assertTrue(b.pipeline_config.run_report)

            fake_result = SimpleNamespace(
                    multilevel_correlations=pd.DataFrame({"q_within_family": [0.01, 0.2]}),
                )
            with patch("eeg_pipeline.analysis.behavior.orchestration.run_group_level_analysis", return_value=fake_result):
                out = b.run_group_level(["0001", "0002"], run_multilevel_correlations=True)
            self.assertIs(out, fake_result)

class TestBehaviorGapfill(unittest.TestCase):
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

            s = BehaviorPipelineResults(subject="0001", trial_table_path="/a", report_path="/b")
            out = s.to_summary()
            self.assertEqual(out["trial_table_path"], "/a")
            self.assertEqual(out["report_path"], "/b")

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
                    computations=["report"],
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
            p.config = DotConfig({"project": {"task": "task"}})
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
            p.config = DotConfig({"project": {"task": "task"}})
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
