import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from tests.pipelines_test_utils import DotConfig, DummyProgress, NoopBatchProgress, NoopProgress

_DummyProgress = DummyProgress
_NoopBatchProgress = NoopBatchProgress
_NoopProgress = NoopProgress


class TestFeatureHelpers(unittest.TestCase):
    def test_unpack_feature_results_and_merge_single(self):
        from eeg_pipeline.pipelines.features import _unpack_feature_results, _merge_dataframes

        f = SimpleNamespace(
            pow_df=None, pow_cols=[], baseline_df=None, baseline_cols=[],
            conn_df=None, conn_cols=[], dconn_df=None, dconn_cols=[],
            source_df=None, source_cols=[],
            source_contrast_df=None, source_contrast_cols=[],
            aper_df=None, aper_cols=[],
            erp_df=None, erp_cols=[], phase_df=None, phase_cols=[],
            itpc_trial_df=None, itpc_trial_cols=[], pac_df=None,
            pac_trials_df=None, pac_time_df=None, comp_df=None, comp_cols=[],
            bursts_df=None, bursts_cols=[], spectral_df=None, spectral_cols=[],
            erds_df=None, erds_cols=[], ratios_df=None, ratios_cols=[],
            asymmetry_df=None, asymmetry_cols=[], quality_df=None, quality_cols=[],
            aper_qc=None,
        )
        out = _unpack_feature_results(f)
        self.assertIn("pow_df", out)
        df = pd.DataFrame({"x": [1]})
        self.assertTrue(_merge_dataframes([df]).equals(df))

    def test_feature_pipeline_early_exit_branches(self):
        from eeg_pipeline.pipelines.features import FeaturePipeline

        tmp = Path(tempfile.mkdtemp())
        p = object.__new__(FeaturePipeline)
        p.config = DotConfig({"project": {"task": "task"}, "bids_root": str(tmp / "bids")})
        p.logger = Mock()
        p.deriv_root = tmp / "deriv"
        p.deriv_root.mkdir(parents=True, exist_ok=True)
        progress = SimpleNamespace(subject_start=lambda *a, **k: None, step=lambda *a, **k: None, subject_done=lambda *a, **k: None, error=lambda *a, **k: None)

        with patch("eeg_pipeline.pipelines.features.resolve_feature_categories", return_value=["power"]), patch(
            "eeg_pipeline.pipelines.features.deriv_features_path", return_value=tmp / "f"
        ), patch("eeg_pipeline.pipelines.features.ensure_dir"), patch(
            "eeg_pipeline.pipelines.features.setup_matplotlib"
        ), patch(
            "eeg_pipeline.pipelines.features.load_epochs_for_analysis", return_value=(None, None)
        ):
            p.process_subject("0001", task="task", progress=progress)

        with patch("eeg_pipeline.pipelines.features.resolve_feature_categories", return_value=["power"]), patch(
            "eeg_pipeline.pipelines.features.deriv_features_path", return_value=tmp / "f"
        ), patch("eeg_pipeline.pipelines.features.ensure_dir"), patch(
            "eeg_pipeline.pipelines.features.setup_matplotlib"
        ), patch(
            "eeg_pipeline.pipelines.features.load_epochs_for_analysis", return_value=(SimpleNamespace(times=np.array([0.0]), info={"sfreq": 100.0}), None)
        ):
            p.process_subject("0001", task="task", progress=progress)

    def test_feature_pipeline_allows_missing_target(self):
        from eeg_pipeline.pipelines.features import FeaturePipeline

        tmp = Path(tempfile.mkdtemp())
        p = object.__new__(FeaturePipeline)
        p.config = DotConfig(
            {
                "project": {"task": "task"},
                "bids_root": str(tmp / "bids"),
                "event_columns": {"outcome": ["outcome", "rating"]},
            }
        )
        p.logger = Mock()
        p.deriv_root = tmp / "deriv"
        p.deriv_root.mkdir(parents=True, exist_ok=True)

        progress = SimpleNamespace(
            subject_start=lambda *a, **k: None,
            step=lambda *a, **k: None,
            subject_done=lambda *a, **k: None,
            error=lambda *a, **k: None,
        )
        epochs = SimpleNamespace(
            times=np.array([0.0, 0.1]),
            info={"sfreq": 100.0},
            ch_names=["Cz", "Pz"],
        )
        aligned = pd.DataFrame({"trial_type": ["a", "b"]})

        fake_features = SimpleNamespace(
            aper_qc=None,
            ratios_df=pd.DataFrame(),
            ratios_cols=[],
            asymmetry_df=pd.DataFrame(),
            asymmetry_cols=[],
            quality_df=pd.DataFrame(),
            quality_cols=[],
        )
        unpacked = {
            "pow_df": pd.DataFrame({"p": [1.0, 2.0]}), "pow_cols": ["p"],
            "baseline_df": pd.DataFrame({"b": [1.0, 1.0]}), "baseline_cols": ["b"],
            "conn_df": pd.DataFrame(), "conn_cols": [],
            "aper_df": pd.DataFrame(), "aper_cols": [],
            "dconn_df": pd.DataFrame(), "dconn_cols": [],
            "source_df": pd.DataFrame(), "source_cols": [],
            "source_contrast_df": pd.DataFrame(), "source_contrast_cols": [],
            "erp_df": pd.DataFrame(), "erp_cols": [],
            "itpc_df": pd.DataFrame(), "itpc_cols": [],
            "itpc_trial_df": pd.DataFrame(), "itpc_trial_cols": [],
            "pac_df": pd.DataFrame(), "pac_trials_df": pd.DataFrame(), "pac_time_df": pd.DataFrame(),
            "comp_df": pd.DataFrame(), "comp_cols": [],
            "bursts_df": pd.DataFrame(), "bursts_cols": [],
            "spectral_df": pd.DataFrame(), "spectral_cols": [],
            "erds_df": pd.DataFrame(), "erds_cols": [],
            "microstates_df": pd.DataFrame(), "microstates_cols": [],
        }

        with patch("eeg_pipeline.pipelines.features.resolve_feature_categories", return_value=["power"]), patch(
            "eeg_pipeline.pipelines.features.deriv_features_path", return_value=tmp / "f"
        ), patch("eeg_pipeline.pipelines.features.ensure_dir"), patch(
            "eeg_pipeline.pipelines.features.setup_matplotlib"
        ), patch(
            "eeg_pipeline.pipelines.features.load_epochs_for_analysis", return_value=(epochs, aligned)
        ), patch(
            "eeg_pipeline.pipelines.features._load_events_df", return_value=None
        ), patch(
            "eeg_pipeline.pipelines.features._load_fixed_templates", return_value=(None, None)
        ), patch(
            "eeg_pipeline.pipelines.features._precompute_tfr_if_needed", return_value=None
        ), patch(
            "eeg_pipeline.pipelines.features._precompute_complex_tfr_if_needed", return_value=None
        ), patch(
            "eeg_pipeline.pipelines.features._precompute_intermediates_if_needed", return_value=None
        ), patch(
            "eeg_pipeline.pipelines.features.extract_all_features", return_value=fake_features
        ), patch(
            "eeg_pipeline.pipelines.features._unpack_feature_results", return_value=unpacked
        ), patch(
            "eeg_pipeline.pipelines.features.align_feature_dataframes",
            return_value=(
                pd.DataFrame({"p": [1.0, 2.0]}),
                pd.DataFrame({"b": [1.0, 1.0]}),
                pd.DataFrame(),
                pd.DataFrame(),
                pd.Series(dtype=float),
                {"n_original": 2, "n_retained": 2, "extra_blocks": {}},
            ),
        ) as mock_align, patch(
            "eeg_pipeline.pipelines.features._build_feature_qc", return_value={}
        ), patch(
            "eeg_pipeline.pipelines.features.save_all_features", return_value=pd.DataFrame({"x": [1, 2]})
        ) as mock_save, patch(
            "eeg_pipeline.pipelines.features._save_extraction_config"
        ), patch(
            "eeg_pipeline.pipelines.features._save_canonical_trial_table_artifact"
        ):
            p.process_subject("0001", task="task", progress=progress)

        self.assertTrue(mock_align.called)
        self.assertIsNone(mock_align.call_args[0][4])  # y argument
        self.assertTrue(mock_save.called)

    def test_feature_pipeline_multi_range_merge_and_wrappers(self):
        from eeg_pipeline.pipelines import features as fmod
        from eeg_pipeline.pipelines.features import FeaturePipeline, process_subject as wrap_ps, extract_features_for_subjects

        tmp = Path(tempfile.mkdtemp())
        p = object.__new__(FeaturePipeline)
        p.config = DotConfig({"project": {"task": "task"}, "bids_root": str(tmp / "bids"), "event_columns": {"rating": ["rating"]}})
        p.logger = Mock()
        p.deriv_root = tmp / "deriv"
        p.deriv_root.mkdir(parents=True, exist_ok=True)
        progress = SimpleNamespace(subject_start=lambda *a, **k: None, step=lambda *a, **k: None, subject_done=lambda *a, **k: None, error=lambda *a, **k: None)
        epochs = SimpleNamespace(times=np.array([0.0, 0.1]), info={"sfreq": 100.0})
        aligned = pd.DataFrame({"rating": [1.0, 2.0], "trial_type": ["a", "b"]})
        precomputed = SimpleNamespace(data=np.zeros((2, 2)), metadata=None, condition_labels=None)

        fake_features = SimpleNamespace(
            aper_qc=None,
            ratios_df=pd.DataFrame({"r": [1]}), ratios_cols=["r"],
            asymmetry_df=pd.DataFrame({"a": [1]}), asymmetry_cols=["a"],
            quality_df=pd.DataFrame({"q": [1]}), quality_cols=["q"],
        )
        unpacked = {
            "pow_df": pd.DataFrame({"p": [1]}), "pow_cols": ["p"],
            "baseline_df": pd.DataFrame({"b": [1]}), "baseline_cols": ["b"],
            "conn_df": pd.DataFrame({"c": [1]}), "conn_cols": ["c"],
            "aper_df": pd.DataFrame({"ap": [1]}), "aper_cols": ["ap"],
            "dconn_df": pd.DataFrame(), "dconn_cols": [],
            "source_df": pd.DataFrame(), "source_cols": [],
            "source_contrast_df": pd.DataFrame(), "source_contrast_cols": [],
            "erp_df": pd.DataFrame(), "erp_cols": [],
            "itpc_df": pd.DataFrame(), "itpc_cols": [],
            "itpc_trial_df": pd.DataFrame(), "itpc_trial_cols": [],
            "pac_df": pd.DataFrame(), "pac_trials_df": pd.DataFrame(), "pac_time_df": pd.DataFrame(),
            "comp_df": pd.DataFrame(), "comp_cols": [],
            "bursts_df": pd.DataFrame(), "bursts_cols": [],
            "spectral_df": pd.DataFrame(), "spectral_cols": [],
            "erds_df": pd.DataFrame(), "erds_cols": [],
        }

        with patch("eeg_pipeline.pipelines.features.resolve_feature_categories", return_value=["power"]), patch(
            "eeg_pipeline.pipelines.features.deriv_features_path", return_value=tmp / "f"
        ), patch("eeg_pipeline.pipelines.features.ensure_dir"), patch(
            "eeg_pipeline.pipelines.features.setup_matplotlib"
        ), patch(
            "eeg_pipeline.pipelines.features.load_epochs_for_analysis", return_value=(epochs, aligned)
        ), patch(
            "eeg_pipeline.pipelines.features._load_events_df", return_value=None
        ), patch(
            "eeg_pipeline.pipelines.features._load_fixed_templates", return_value=(None, None)
        ), patch(
            "eeg_pipeline.pipelines.features._precompute_tfr_if_needed", return_value=None
        ), patch(
            "eeg_pipeline.pipelines.features._precompute_complex_tfr_if_needed", return_value=None
        ), patch(
            "eeg_pipeline.pipelines.features._precompute_intermediates_if_needed", return_value=precomputed
        ), patch(
            "eeg_pipeline.pipelines.features.extract_all_features", return_value=fake_features
        ), patch(
            "eeg_pipeline.pipelines.features._unpack_feature_results", return_value=unpacked
        ), patch(
            "eeg_pipeline.pipelines.features.align_feature_dataframes",
            return_value=(
                pd.DataFrame({"p": [1]}), pd.DataFrame({"b": [1]}), pd.DataFrame({"c": [1]}), pd.DataFrame({"ap": [1]}), pd.Series([1.0]), {"extra_blocks": {}}
            ),
        ), patch(
            "eeg_pipeline.pipelines.features._build_feature_qc", return_value={}
        ), patch(
            "eeg_pipeline.pipelines.features.save_all_features", return_value=pd.DataFrame({"x": [1]})
        ), patch(
            "eeg_pipeline.pipelines.features._save_extraction_config"
        ) as save_cfg, patch(
            "eeg_pipeline.pipelines.features._save_merged_features"
        ) as save_merged:
            p.process_subject(
                "0001",
                task="task",
                progress=progress,
                feature_categories=["power"],
                time_ranges=[{"name": "late", "tmin": 1.0, "tmax": 0.0}, {"name": "full", "tmin": 0.0, "tmax": 1.0}],
            )
        self.assertTrue(save_merged.called)
        self.assertGreaterEqual(save_cfg.call_count, 2)

        with patch.object(fmod, "FeaturePipeline") as mock_cls:
            inst = mock_cls.return_value
            inst.run_batch.return_value = [{"ok": True}]
            wrap_ps("0001", task="t")
            out = extract_features_for_subjects(["0001"], task="t")
        self.assertEqual(out, [{"ok": True}])

    def test_pipeline_constants_and_exports(self):
        import eeg_pipeline.pipelines as p
        from eeg_pipeline.pipelines import constants

        self.assertIn("FeaturePipeline", p.__all__)
        self.assertIn("alpha", constants.FREQUENCY_BANDS)
        self.assertIn("power", constants.FEATURE_CATEGORIES)
        self.assertNotIn("report", constants.BEHAVIOR_COMPUTATIONS)

    def test_feature_pipeline_process_subject_happy_path(self):
        from eeg_pipeline.pipelines.features import FeaturePipeline

        tmp = Path(tempfile.mkdtemp())
        pipeline = object.__new__(FeaturePipeline)
        pipeline.config = DotConfig(
            {
                "project": {"task": "task"},
                "event_columns": {"rating": ["rating"]},
                "bids_root": str(tmp / "bids"),
            }
        )
        pipeline.logger = Mock()
        pipeline.deriv_root = tmp / "deriv"
        pipeline.deriv_root.mkdir(parents=True, exist_ok=True)

        aligned_events = pd.DataFrame({"rating": [1.0, 2.0], "condition": ["a", "b"]})
        original_events = pd.DataFrame({"rating": [1.0, 2.0]})
        features_dir = tmp / "features_out"

        fake_features = SimpleNamespace(
            aper_qc={"ok": True},
            ratios_df=pd.DataFrame({"r": [1]}),
            ratios_cols=["r"],
            asymmetry_df=pd.DataFrame({"a": [1]}),
            asymmetry_cols=["a"],
            quality_df=pd.DataFrame({"q": [1]}),
            quality_cols=["q"],
        )
        unpacked = {
            "pow_df": pd.DataFrame({"p": [1]}),
            "pow_cols": ["p"],
            "baseline_df": pd.DataFrame({"b": [1]}),
            "baseline_cols": ["b"],
            "conn_df": pd.DataFrame({"c": [1]}),
            "conn_cols": ["c"],
            "dconn_df": pd.DataFrame({"dc": [1]}),
            "dconn_cols": ["dc"],
            "source_df": pd.DataFrame({"s": [1]}),
            "source_cols": ["s"],
            "source_contrast_df": pd.DataFrame(),
            "source_contrast_cols": [],
            "aper_df": pd.DataFrame({"ap": [1]}),
            "aper_cols": ["ap"],
            "erp_df": pd.DataFrame({"e": [1]}),
            "erp_cols": ["e"],
            "itpc_df": pd.DataFrame({"i": [1]}),
            "itpc_cols": ["i"],
            "itpc_trial_df": pd.DataFrame({"it": [1]}),
            "itpc_trial_cols": ["it"],
            "pac_df": pd.DataFrame({"pa": [1]}),
            "pac_trials_df": pd.DataFrame({"pat": [1]}),
            "pac_time_df": pd.DataFrame({"pt": [1]}),
            "comp_df": pd.DataFrame({"co": [1]}),
            "comp_cols": ["co"],
            "bursts_df": pd.DataFrame({"bu": [1]}),
            "bursts_cols": ["bu"],
            "spectral_df": pd.DataFrame({"sp": [1]}),
            "spectral_cols": ["sp"],
            "erds_df": pd.DataFrame({"er": [1]}),
            "erds_cols": ["er"],
            "ratios_df": fake_features.ratios_df,
            "ratios_cols": fake_features.ratios_cols,
            "asymmetry_df": fake_features.asymmetry_df,
            "asymmetry_cols": fake_features.asymmetry_cols,
            "quality_df": fake_features.quality_df,
            "quality_cols": fake_features.quality_cols,
            "aper_qc": fake_features.aper_qc,
        }

        progress = SimpleNamespace(
            subject_start=lambda *a, **k: None,
            step=lambda *a, **k: None,
            subject_done=lambda *a, **k: None,
            error=lambda *a, **k: None,
        )
        epochs = SimpleNamespace(times=np.array([0.0, 0.1]), info={"sfreq": 100.0})

        train_mask = np.array([True, False], dtype=bool)

        with patch("eeg_pipeline.pipelines.features.resolve_feature_categories", return_value=["power"]), patch(
            "eeg_pipeline.pipelines.features.deriv_features_path", return_value=features_dir
        ), patch("eeg_pipeline.pipelines.features.ensure_dir"), patch(
            "eeg_pipeline.pipelines.features.setup_matplotlib"
        ), patch(
            "eeg_pipeline.pipelines.features.load_epochs_for_analysis", return_value=(epochs, aligned_events)
        ), patch(
            "eeg_pipeline.pipelines.features._load_events_df", return_value=original_events
        ), patch(
            "eeg_pipeline.pipelines.features.save_dropped_trials_log"
        ), patch(
            "eeg_pipeline.pipelines.features._load_fixed_templates", return_value=(None, None)
        ), patch(
            "eeg_pipeline.pipelines.features._precompute_tfr_if_needed", return_value=None
        ), patch(
            "eeg_pipeline.pipelines.features._precompute_complex_tfr_if_needed", return_value=None
        ), patch(
            "eeg_pipeline.pipelines.features._precompute_intermediates_if_needed", return_value=None
        ), patch(
            "eeg_pipeline.pipelines.features.extract_all_features", return_value=fake_features
        ) as mock_extract, patch(
            "eeg_pipeline.pipelines.features._unpack_feature_results", return_value=unpacked
        ), patch(
            "eeg_pipeline.pipelines.features.align_feature_dataframes",
            return_value=(
                pd.DataFrame({"p": [1]}),
                pd.DataFrame({"b": [1]}),
                pd.DataFrame({"c": [1]}),
                pd.DataFrame({"ap": [1]}),
                pd.Series([1.0]),
                {"extra_blocks": {}},
            ),
        ), patch(
            "eeg_pipeline.pipelines.features._build_feature_qc", return_value={"qc": True}
        ), patch(
            "eeg_pipeline.pipelines.features.save_all_features", return_value=pd.DataFrame({"x": [1]})
        ), patch(
            "eeg_pipeline.pipelines.features._save_extraction_config"
        ):
            pipeline.process_subject(
                "0001",
                task="task",
                progress=progress,
                feature_categories=["power"],
                train_mask=train_mask,
                analysis_mode="trial_ml_safe",
            )
        ctx_used = mock_extract.call_args.args[0]
        np.testing.assert_array_equal(ctx_used.train_mask, train_mask)
        self.assertEqual(ctx_used.analysis_mode, "trial_ml_safe")

    def test_feature_pipeline_auto_writes_canonical_trial_table(self):
        fake_mne_bids = types.ModuleType("mne_bids")
        fake_mne_bids.BIDSPath = object
        fake_pyarrow = types.ModuleType("pyarrow")
        fake_pyarrow.Table = types.SimpleNamespace(from_pandas=lambda *args, **kwargs: None)
        fake_pyarrow_csv = types.ModuleType("pyarrow.csv")
        fake_pyarrow_csv.WriteOptions = lambda **kwargs: None
        fake_pyarrow_csv.write_csv = lambda *args, **kwargs: None
        fake_mne = types.ModuleType("mne")
        fake_seaborn = types.ModuleType("seaborn")
        with patch.dict(
            sys.modules,
            {
                "yaml": types.ModuleType("yaml"),
                "mne_bids": fake_mne_bids,
                "mne": fake_mne,
                "seaborn": fake_seaborn,
                "pyarrow": fake_pyarrow,
                "pyarrow.csv": fake_pyarrow_csv,
            },
        ):
            try:
                from eeg_pipeline.pipelines.features import FeaturePipeline
            except ModuleNotFoundError as exc:
                self.skipTest(f"FeaturePipeline import unavailable in test environment: {exc}")

        tmp = Path(tempfile.mkdtemp())
        pipeline = object.__new__(FeaturePipeline)
        pipeline.config = DotConfig(
            {
                "project": {"task": "task"},
                "event_columns": {"rating": ["rating"]},
                "bids_root": str(tmp / "bids"),
            }
        )
        pipeline.logger = Mock()
        pipeline.deriv_root = tmp / "deriv"
        pipeline.deriv_root.mkdir(parents=True, exist_ok=True)

        aligned_events = pd.DataFrame({"rating": [1.0, 2.0], "condition": ["a", "b"]})
        features_dir = tmp / "features_out"

        fake_features = SimpleNamespace(
            aper_qc={"ok": True},
            ratios_df=pd.DataFrame({"r": [1]}),
            ratios_cols=["r"],
            asymmetry_df=pd.DataFrame({"a": [1]}),
            asymmetry_cols=["a"],
            quality_df=pd.DataFrame({"q": [1]}),
            quality_cols=["q"],
        )
        unpacked = {
            "pow_df": pd.DataFrame({"p": [1]}),
            "pow_cols": ["p"],
            "baseline_df": pd.DataFrame({"b": [1]}),
            "baseline_cols": ["b"],
            "conn_df": pd.DataFrame({"c": [1]}),
            "conn_cols": ["c"],
            "dconn_df": pd.DataFrame({"dc": [1]}),
            "dconn_cols": ["dc"],
            "source_df": pd.DataFrame({"s": [1]}),
            "source_cols": ["s"],
            "source_contrast_df": pd.DataFrame(),
            "source_contrast_cols": [],
            "aper_df": pd.DataFrame({"ap": [1]}),
            "aper_cols": ["ap"],
            "erp_df": pd.DataFrame({"e": [1]}),
            "erp_cols": ["e"],
            "itpc_df": pd.DataFrame({"i": [1]}),
            "itpc_cols": ["i"],
            "itpc_trial_df": pd.DataFrame({"it": [1]}),
            "itpc_trial_cols": ["it"],
            "pac_df": pd.DataFrame({"pa": [1]}),
            "pac_trials_df": pd.DataFrame({"pat": [1]}),
            "pac_time_df": pd.DataFrame({"pt": [1]}),
            "comp_df": pd.DataFrame({"co": [1]}),
            "comp_cols": ["co"],
            "bursts_df": pd.DataFrame({"bu": [1]}),
            "bursts_cols": ["bu"],
            "spectral_df": pd.DataFrame({"sp": [1]}),
            "spectral_cols": ["sp"],
            "erds_df": pd.DataFrame({"er": [1]}),
            "erds_cols": ["er"],
            "ratios_df": fake_features.ratios_df,
            "ratios_cols": fake_features.ratios_cols,
            "asymmetry_df": fake_features.asymmetry_df,
            "asymmetry_cols": fake_features.asymmetry_cols,
            "quality_df": fake_features.quality_df,
            "quality_cols": fake_features.quality_cols,
            "aper_qc": fake_features.aper_qc,
        }

        progress = SimpleNamespace(
            subject_start=lambda *a, **k: None,
            step=lambda *a, **k: None,
            subject_done=lambda *a, **k: None,
            error=lambda *a, **k: None,
        )
        epochs = SimpleNamespace(times=np.array([0.0, 0.1]), info={"sfreq": 100.0})

        with patch("eeg_pipeline.pipelines.features.resolve_feature_categories", return_value=["power"]), patch(
            "eeg_pipeline.pipelines.features.deriv_features_path", return_value=features_dir
        ), patch("eeg_pipeline.pipelines.features.ensure_dir"), patch(
            "eeg_pipeline.pipelines.features.setup_matplotlib"
        ), patch(
            "eeg_pipeline.pipelines.features.load_epochs_for_analysis", return_value=(epochs, aligned_events)
        ), patch(
            "eeg_pipeline.pipelines.features._load_events_df", return_value=None
        ), patch(
            "eeg_pipeline.pipelines.features._load_fixed_templates", return_value=(None, None)
        ), patch(
            "eeg_pipeline.pipelines.features._precompute_tfr_if_needed", return_value=None
        ), patch(
            "eeg_pipeline.pipelines.features._precompute_complex_tfr_if_needed", return_value=None
        ), patch(
            "eeg_pipeline.pipelines.features._precompute_intermediates_if_needed", return_value=None
        ), patch(
            "eeg_pipeline.pipelines.features.extract_all_features", return_value=fake_features
        ), patch(
            "eeg_pipeline.pipelines.features._unpack_feature_results", return_value=unpacked
        ), patch(
            "eeg_pipeline.pipelines.features.align_feature_dataframes",
            return_value=(
                pd.DataFrame({"p": [1, 2]}),
                pd.DataFrame({"b": [1, 2]}),
                pd.DataFrame({"c": [1, 2]}),
                pd.DataFrame({"ap": [1, 2]}),
                pd.Series([1.0, 2.0]),
                {"extra_blocks": {}},
            ),
        ), patch(
            "eeg_pipeline.pipelines.features._build_feature_qc", return_value={"qc": True}
        ), patch(
            "eeg_pipeline.pipelines.features.save_all_features", return_value=pd.DataFrame({"power_alpha": [0.1, 0.2]})
        ), patch(
            "eeg_pipeline.pipelines.features._save_extraction_config"
        ), patch(
            "eeg_pipeline.pipelines.features._save_canonical_trial_table_artifact", create=True
        ) as save_trial_table_mock:
            pipeline.process_subject("0001", task="task", progress=progress, feature_categories=["power"])

        self.assertEqual(save_trial_table_mock.call_count, 1)

    def test_canonical_trial_table_filters_non_trialwise_columns_by_provenance(self):
        from eeg_pipeline.pipelines.features import _save_canonical_trial_table_artifact

        tmp = Path(tempfile.mkdtemp())
        deriv_root = tmp / "deriv"
        deriv_root.mkdir(parents=True, exist_ok=True)
        aligned_events = pd.DataFrame({"rating": [1.0, 2.0]})
        trialwise_df = pd.DataFrame({"power_active_alpha_global_mean": [0.1, 0.2]})
        non_trialwise_df = pd.DataFrame({"conn_active_alpha_global_wpli_mean": [0.3, 0.3]})
        non_trialwise_df.attrs["feature_granularity"] = "subject"
        non_trialwise_df.attrs["phase_estimator"] = "across_epochs"
        non_trialwise_df.attrs["broadcast_warning"] = "broadcast"

        captured: dict[str, pd.DataFrame] = {}

        def _fake_save_trial_table(wrapper, out_path, format):
            _ = out_path
            _ = format
            captured["df"] = wrapper.df.copy()

        with patch(
            "eeg_pipeline.utils.data.trial_table.save_trial_table",
            side_effect=_fake_save_trial_table,
        ):
            _save_canonical_trial_table_artifact(
                deriv_root=deriv_root,
                subject="0001",
                task="task",
                aligned_events=aligned_events,
                feature_tables=[
                    ("power", trialwise_df),
                    ("connectivity", non_trialwise_df),
                ],
                config=DotConfig({"feature_engineering": {"analysis_mode": "trial_ml_safe"}}),
                logger=Mock(),
            )

        self.assertIn("df", captured)
        self.assertIn("power_active_alpha_global_mean", captured["df"].columns)
        self.assertNotIn("conn_active_alpha_global_wpli_mean", captured["df"].columns)

    def test_load_fixed_templates(self):
        from eeg_pipeline.pipelines.features import _load_fixed_templates

        logger = Mock()
        t, names, labels = _load_fixed_templates(None, logger)
        self.assertIsNone(t)
        self.assertIsNone(names)
        self.assertIsNone(labels)

        tmp = Path(tempfile.mkdtemp())
        path = tmp / "templates.npz"
        np.savez(
            path,
            templates=np.array([[1, 2]]),
            ch_names=np.array(["Cz"]),
            labels=np.array(["A"]),
        )
        templates, ch_names, loaded_labels = _load_fixed_templates(path, logger)
        self.assertEqual(templates.shape, (1, 2))
        self.assertIsNotNone(ch_names)
        self.assertEqual(loaded_labels, ["A"])

    def test_load_fixed_templates_surfaces_invalid_files(self):
        from eeg_pipeline.pipelines.features import _load_fixed_templates

        logger = Mock()
        tmp = Path(tempfile.mkdtemp())
        path = tmp / "templates_missing_key.npz"
        np.savez(path, ch_names=np.array(["Cz"]))

        with self.assertRaisesRegex(ValueError, "missing required array 'templates'"):
            _load_fixed_templates(path, logger)

    def test_precompute_tfr_helpers(self):
        from eeg_pipeline.pipelines.features import (
            _precompute_tfr_if_needed,
            _precompute_intermediates_if_needed,
            _precompute_complex_tfr_if_needed,
        )

        epochs = SimpleNamespace(times=np.array([0.0, 0.1]), info={"sfreq": 100.0})
        logger = Mock()

        with patch("eeg_pipeline.pipelines.features.compute_tfr_morlet", return_value="TFR") as mock_tfr:
            out = _precompute_tfr_if_needed(
                epochs,
                [{"name": "a", "tmin": 0.0, "tmax": 1.0}, {"name": "b", "tmin": 1.0, "tmax": 2.0}],
                ["power"],
                DotConfig({}),
                logger,
            )
        self.assertEqual(out, "TFR")
        mock_tfr.assert_called_once()

        with patch("eeg_pipeline.pipelines.features.compute_complex_tfr", return_value="CTFR") as mock_ctfr, patch(
            "eeg_pipeline.analysis.features.preparation._get_spatial_transform_type", return_value="none"
        ):
            out2 = _precompute_complex_tfr_if_needed(
                epochs,
                [{"name": "a"}, {"name": "b"}],
                ["itpc"],
                DotConfig({}),
                logger,
            )
        self.assertEqual(out2, "CTFR")
        mock_ctfr.assert_called_once()

        fake_precomputed = SimpleNamespace(data=np.zeros((2, 2)), qc=None, metadata=None, condition_labels=None)
        with patch("eeg_pipeline.pipelines.features.precompute_data", return_value=fake_precomputed):
            out3 = _precompute_intermediates_if_needed(
                epochs,
                [{"name": "a"}, {"name": "b"}],
                ["connectivity"],
                ["alpha"],
                DotConfig({}),
                logger,
            )
        self.assertIs(out3, fake_precomputed)

    def test_feature_accumulation_and_merge_helpers(self):
        from eeg_pipeline.pipelines.features import (
            _create_feature_accumulator,
            _accumulate_features,
            _merge_dataframes,
            _get_df_cols,
            _build_extra_blocks,
            _update_from_aligned_extra,
            _build_feature_qc,
        )

        df_a = pd.DataFrame({"a": [1]})
        df_b = pd.DataFrame({"b": [2]})
        features = SimpleNamespace(
            ratios_df=df_a,
            asymmetry_df=df_b,
            quality_df=df_a,
            aper_qc={"ok": True},
        )
        unpacked = {
            "itpc_df": df_a,
            "itpc_trial_df": df_a,
            "pac_df": df_a,
            "pac_trials_df": None,
            "pac_time_df": df_a,
            "comp_df": df_a,
            "spectral_df": df_a,
            "erp_df": df_a,
            "bursts_df": df_a,
            "erds_df": df_a,
            "dconn_df": df_a,
            "source_df": df_a,
            "source_contrast_df": pd.DataFrame(),
        }

        extra = _build_extra_blocks(unpacked, features)
        self.assertIn("itpc", extra)
        self.assertIn("ratios", extra)

        _update_from_aligned_extra(unpacked, features, {"quality": df_b, "itpc": df_b})
        self.assertTrue(unpacked["itpc_df"].equals(df_b))
        self.assertTrue(features.quality_df.equals(df_b))

        acc = _create_feature_accumulator()
        aligned = {
            "pow_df_aligned": df_a,
            "baseline_df_aligned": df_b,
            "conn_df_aligned": df_a,
            "aper_df_aligned": df_b,
        }
        _accumulate_features(acc, unpacked, features, aligned)
        self.assertGreaterEqual(len(acc["power"]), 1)

        merged = _merge_dataframes([df_a, df_a, df_b])
        self.assertEqual(list(merged.columns), ["a", "b"])
        self.assertEqual(_get_df_cols(df_a), 1)
        self.assertEqual(_get_df_cols(pd.DataFrame()), 0)

        @dataclass
        class _QC:
            foo: int

        ctx = SimpleNamespace(precomputed=SimpleNamespace(qc=_QC(foo=1)))
        qc = _build_feature_qc(SimpleNamespace(aper_qc={"x": 1}), ctx)
        self.assertIn("aperiodic", qc)
        self.assertIn("precomputed_intermediates", qc)

    def test_pac_trials_alignment_round_trip(self):
        from eeg_pipeline.pipelines.features import _build_extra_blocks, _update_from_aligned_extra

        pac_trials_df = pd.DataFrame({"pac_active_theta_gamma_global_mvl": [0.1, 0.2, 0.3]})
        unpacked = {
            "itpc_df": pd.DataFrame(),
            "itpc_trial_df": pd.DataFrame(),
            "pac_df": pd.DataFrame({"pac_summary": [1.0, 1.0, 1.0]}),
            "pac_trials_df": pac_trials_df.copy(),
            "pac_time_df": pd.DataFrame(),
            "comp_df": pd.DataFrame(),
            "spectral_df": pd.DataFrame(),
            "erp_df": pd.DataFrame(),
            "bursts_df": pd.DataFrame(),
            "erds_df": pd.DataFrame(),
            "dconn_df": pd.DataFrame(),
            "source_df": pd.DataFrame(),
            "source_contrast_df": pd.DataFrame(),
            "microstates_df": pd.DataFrame(),
        }
        features = SimpleNamespace(
            ratios_df=pd.DataFrame(),
            asymmetry_df=pd.DataFrame(),
            quality_df=pd.DataFrame(),
            microstates_df=pd.DataFrame(),
        )

        extra = _build_extra_blocks(unpacked, features)
        self.assertIn("pac_trials", extra)

        aligned_extra = {
            key: value.iloc[[0, 2]].reset_index(drop=True)
            for key, value in extra.items()
        }
        _update_from_aligned_extra(unpacked, features, aligned_extra)

        self.assertEqual(len(unpacked["pac_trials_df"]), 2)
        self.assertEqual(
            list(unpacked["pac_trials_df"]["pac_active_theta_gamma_global_mvl"]),
            [0.1, 0.3],
        )

    def test_save_merged_and_extraction_config(self):
        from eeg_pipeline.pipelines.features import _save_merged_features, _save_extraction_config

        tmp = Path(tempfile.mkdtemp())
        features_dir = tmp / "derivatives" / "sub-0001" / "task-task" / "features"
        features_dir.mkdir(parents=True, exist_ok=True)

        acc = {
            "power": [pd.DataFrame({"p": [1]})],
            "baseline": [pd.DataFrame({"b": [2]})],
            "connectivity": [],
            "directedconnectivity": [],
            "sourcelocalization": [],
            "aperiodic": [],
            "erp": [],
            "itpc": [],
            "pac": [],
            "pac_time": [],
            "complexity": [],
            "bursts": [],
            "spectral": [],
            "erds": [],
            "ratios": [],
            "asymmetry": [],
            "quality": [],
        }

        fake_feature_io = types.SimpleNamespace(_get_folder_for_feature=lambda name, config=None: name)
        fake_naming = types.SimpleNamespace(generate_manifest=lambda **kwargs: {"feature_columns": kwargs["feature_columns"]})

        with patch.dict(
            sys.modules,
            {
                "eeg_pipeline.utils.data.feature_io": fake_feature_io,
                "eeg_pipeline.domain.features.naming": fake_naming,
            },
        ), patch("eeg_pipeline.pipelines.features.write_parquet") as mock_parquet:
            _save_merged_features(acc, features_dir, DotConfig({"project": {"task": "task"}}), Mock())
        self.assertTrue(mock_parquet.called)

        with patch.dict(sys.modules, {"eeg_pipeline.utils.data.feature_io": fake_feature_io}):
            _save_extraction_config(
                {"feature_categories": ["power"]},
                features_dir,
                suffix="x",
                logger=Mock(),
                feature_categories=["power"],
            )
        cfg_file = features_dir / "features_power" / "metadata" / "extraction_config_x.json"
        self.assertTrue(cfg_file.exists())

class TestFeatureGapfill(unittest.TestCase):
    def test_feature_pipeline_init_and_precompute_early_returns(self):
        from eeg_pipeline.pipelines.features import (
            FeaturePipeline,
            _precompute_tfr_if_needed,
            _precompute_complex_tfr_if_needed,
            _precompute_intermediates_if_needed,
        )

        cfg = DotConfig({})
        with patch("eeg_pipeline.pipelines.features.PipelineBase.__init__", lambda self, name, config=None: (setattr(self, "name", name), setattr(self, "config", config or cfg), setattr(self, "logger", Mock()), setattr(self, "deriv_root", Path(tempfile.mkdtemp())))):
            p = FeaturePipeline(config=cfg)
        self.assertEqual(p.name, "feature_extraction")

        epochs = SimpleNamespace(times=np.array([0.0]), info={"sfreq": 100.0})
        self.assertIsNone(_precompute_tfr_if_needed(epochs, [{"name": "a"}], ["power"], DotConfig({}), Mock()))
        self.assertIsNone(_precompute_complex_tfr_if_needed(epochs, [{"name": "a"}], ["itpc"], DotConfig({}), Mock()))
        self.assertIsNone(_precompute_intermediates_if_needed(epochs, [{"name": "a"}], ["power"], None, DotConfig({}), Mock()))

    def test_precompute_complex_spatial_transform_branch(self):
        from eeg_pipeline.pipelines.features import _precompute_complex_tfr_if_needed

        class FakeEpochs:
            def copy(self):
                return self

            def pick_types(self, **_kwargs):
                return self

        epochs = FakeEpochs()
        logger = Mock()

        with patch("eeg_pipeline.analysis.features.preparation._get_spatial_transform_type", return_value="csd"), patch(
            "eeg_pipeline.analysis.features.preparation._apply_spatial_transform", return_value=epochs
        ) as apply_spatial, patch(
            "eeg_pipeline.pipelines.features.compute_complex_tfr", return_value="CTFR"
        ) as compute_ctfr:
            out = _precompute_complex_tfr_if_needed(
                epochs,
                [{"name": "a"}, {"name": "b"}],
                ["itpc"],
                DotConfig({"feature_engineering": {"pac": {"source": "raw"}}}),
                logger,
            )

        self.assertEqual(out, "CTFR")
        apply_spatial.assert_called_once()
        compute_ctfr.assert_called_once()

    def test_precompute_complex_skips_shared_when_itpc_pac_transforms_differ(self):
        from eeg_pipeline.pipelines.features import _precompute_complex_tfr_if_needed

        epochs = SimpleNamespace()
        cfg = DotConfig({"feature_engineering": {"pac": {"source": "raw"}}})

        def _transform_for_family(_config, feature_family=None):
            if feature_family == "itpc":
                return "csd"
            if feature_family == "pac":
                return "none"
            return "none"

        with patch(
            "eeg_pipeline.analysis.features.preparation._get_spatial_transform_type",
            side_effect=_transform_for_family,
        ), patch(
            "eeg_pipeline.pipelines.features.compute_complex_tfr",
            return_value="CTFR",
        ) as compute_ctfr:
            out = _precompute_complex_tfr_if_needed(
                epochs,
                [{"name": "a"}, {"name": "b"}],
                ["itpc", "pac"],
                cfg,
                Mock(),
            )

        self.assertIsNone(out)
        compute_ctfr.assert_not_called()

    def test_precompute_complex_pac_precomputed_returns_none(self):
        from eeg_pipeline.pipelines.features import _precompute_complex_tfr_if_needed

        epochs = SimpleNamespace()
        out = _precompute_complex_tfr_if_needed(
            epochs,
            [{"name": "a"}, {"name": "b"}],
            ["pac"],
            DotConfig({"feature_engineering": {"pac": {"source": "precomputed"}}}),
            Mock(),
        )
        self.assertIsNone(out)

    def test_save_merged_features_csv_and_condition_label_branch(self):
        from eeg_pipeline.pipelines.features import FeaturePipeline, _save_merged_features

        tmp = Path(tempfile.mkdtemp())
        features_dir = tmp / "derivatives" / "sub-0001" / "task-task" / "features"
        features_dir.mkdir(parents=True, exist_ok=True)

        acc = {
            "power": [pd.DataFrame({"p": [1]})],
            "baseline": [pd.DataFrame({"b": [2]})],
            "connectivity": [],
            "directedconnectivity": [],
            "sourcelocalization": [],
            "aperiodic": [],
            "erp": [],
            "itpc": [],
            "pac": [],
            "pac_time": [],
            "complexity": [],
            "bursts": [],
            "spectral": [],
            "erds": [],
            "ratios": [],
            "asymmetry": [],
            "quality": [],
        }

        fake_feature_io = types.SimpleNamespace(_get_folder_for_feature=lambda name, config=None: name)
        fake_naming = types.SimpleNamespace(generate_manifest=lambda **kwargs: {"feature_columns": kwargs["feature_columns"]})

        with patch.dict(
            sys.modules,
            {
                "eeg_pipeline.utils.data.feature_io": fake_feature_io,
                "eeg_pipeline.domain.features.naming": fake_naming,
                "eeg_pipeline.infra.tsv": types.SimpleNamespace(write_csv=Mock()),
            },
        ), patch("eeg_pipeline.pipelines.features.write_parquet") as write_parquet, patch(
            "eeg_pipeline.utils.config.loader.get_config_value", return_value=True
        ):
            _save_merged_features(acc, features_dir, DotConfig({"project": {"task": "task"}}), Mock())
        self.assertTrue(write_parquet.called)

        p = object.__new__(FeaturePipeline)
        p.config = DotConfig({"project": {"task": "task"}, "event_columns": {"rating": ["rating"]}, "bids_root": str(tmp / "bids")})
        p.logger = Mock()
        p.deriv_root = tmp / "deriv"
        p.deriv_root.mkdir(parents=True, exist_ok=True)
        progress = SimpleNamespace(subject_start=lambda *a, **k: None, step=lambda *a, **k: None, subject_done=lambda *a, **k: None, error=lambda *a, **k: None)
        epochs = SimpleNamespace(times=np.array([0.0, 0.1]), info={"sfreq": 100.0})
        aligned = pd.DataFrame({"rating": [1.0, 2.0], "condition": ["a", "b"]})
        precomputed = SimpleNamespace(data=np.zeros((2, 2)), metadata=None, condition_labels=None)

        fake_features = SimpleNamespace(
            aper_qc=None,
            ratios_df=pd.DataFrame({"r": [1]}), ratios_cols=["r"],
            asymmetry_df=pd.DataFrame({"a": [1]}), asymmetry_cols=["a"],
            quality_df=pd.DataFrame({"q": [1]}), quality_cols=["q"],
        )
        unpacked = {
            "pow_df": pd.DataFrame({"p": [1]}), "pow_cols": ["p"],
            "baseline_df": pd.DataFrame({"b": [1]}), "baseline_cols": ["b"],
            "conn_df": pd.DataFrame({"c": [1]}), "conn_cols": ["c"],
            "aper_df": pd.DataFrame({"ap": [1]}), "aper_cols": ["ap"],
            "dconn_df": pd.DataFrame(), "dconn_cols": [],
            "source_df": pd.DataFrame(), "source_cols": [],
            "source_contrast_df": pd.DataFrame(), "source_contrast_cols": [],
            "erp_df": pd.DataFrame(), "erp_cols": [],
            "itpc_df": pd.DataFrame(), "itpc_cols": [],
            "itpc_trial_df": pd.DataFrame(), "itpc_trial_cols": [],
            "pac_df": pd.DataFrame(), "pac_trials_df": pd.DataFrame(), "pac_time_df": pd.DataFrame(),
            "comp_df": pd.DataFrame(), "comp_cols": [],
            "bursts_df": pd.DataFrame(), "bursts_cols": [],
            "spectral_df": pd.DataFrame(), "spectral_cols": [],
            "erds_df": pd.DataFrame(), "erds_cols": [],
        }

        with patch("eeg_pipeline.pipelines.features.resolve_feature_categories", return_value=["power"]), patch(
            "eeg_pipeline.pipelines.features.deriv_features_path", return_value=tmp / "f"
        ), patch("eeg_pipeline.pipelines.features.ensure_dir"), patch(
            "eeg_pipeline.pipelines.features.setup_matplotlib"
        ), patch(
            "eeg_pipeline.pipelines.features.load_epochs_for_analysis", return_value=(epochs, aligned)
        ), patch(
            "eeg_pipeline.pipelines.features._load_events_df", return_value=None
        ), patch(
            "eeg_pipeline.pipelines.features._load_fixed_templates", return_value=(None, None)
        ), patch(
            "eeg_pipeline.pipelines.features._precompute_tfr_if_needed", return_value=None
        ), patch(
            "eeg_pipeline.pipelines.features._precompute_complex_tfr_if_needed", return_value=None
        ), patch(
            "eeg_pipeline.pipelines.features._precompute_intermediates_if_needed", return_value=precomputed
        ), patch(
            "eeg_pipeline.pipelines.features.extract_all_features", return_value=fake_features
        ), patch(
            "eeg_pipeline.pipelines.features._unpack_feature_results", return_value=unpacked
        ), patch(
            "eeg_pipeline.pipelines.features.align_feature_dataframes",
            return_value=(
                pd.DataFrame({"p": [1]}),
                pd.DataFrame({"b": [1]}),
                pd.DataFrame({"c": [1]}),
                pd.DataFrame({"ap": [1]}),
                pd.Series([1.0]),
                {"extra_blocks": {}},
            ),
        ), patch(
            "eeg_pipeline.pipelines.features._build_feature_qc", return_value={}
        ), patch(
            "eeg_pipeline.pipelines.features.save_all_features", return_value=pd.DataFrame({"x": [1]})
        ), patch(
            "eeg_pipeline.pipelines.features._save_extraction_config"
        ):
            p.process_subject("0001", task="task", progress=progress, feature_categories=["power"])

        self.assertIsNotNone(precomputed.condition_labels)
