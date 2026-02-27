import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


from tests.pipelines_test_utils import DotConfig, DummyProgress, NoopBatchProgress, NoopProgress

_DummyProgress = DummyProgress
_NoopBatchProgress = NoopBatchProgress
_NoopProgress = NoopProgress


class TestPreprocessingHelpers(unittest.TestCase):
    def test_detect_conditions_from_bids(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        tmp = Path(tempfile.mkdtemp())
        ev_dir = tmp / "sub-0001" / "eeg"
        ev_dir.mkdir(parents=True, exist_ok=True)
        ev = ev_dir / "sub-0001_task-task_run-01_events.tsv"
        ev.write_text(
            "trial_type\tonset\nTrig_thermHot\t0\nVolume\t1\nTrig_thermWarm\t2\n",
            encoding="utf-8",
        )

        p = object.__new__(PreprocessingPipeline)
        p.bids_root = tmp
        p.logger = Mock()
        detected = p._detect_conditions_from_bids()
        self.assertEqual(detected, ["Trig_thermHot", "Trig_thermWarm"])

    def test_generate_mne_bids_config(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.bids_root = Path("/tmp/bids")
        p.deriv_root = Path("/tmp/deriv")
        p.logger = Mock()
        p.config = DotConfig(
            {
                "eeg": {"ch_types": "eeg", "reference": "average", "eog_channels": "EOG001,EOG002"},
                "preprocessing": {"task_is_rest": False, "l_freq": 0.1, "h_freq": 40.0, "find_breaks": True},
                "ica": {"algorithm": "picard", "n_components": 0.99},
                "epochs": {"baseline": [None, 0], "reject_method": "none", "tmin": -0.2, "tmax": 0.8},
            }
        )
        with patch.object(PreprocessingPipeline, "_detect_conditions_from_bids", return_value=["stim"]):
            cfg = p._generate_mne_bids_config("preprocessing/_07_make_epochs", subjects=["0001"])
        self.assertIn("subjects = ['0001']", cfg)
        self.assertIn("conditions = ['stim']", cfg)
        self.assertIn("baseline = (None, 0)", cfg)
        self.assertNotIn("reject =", cfg)

    def test_run_mne_bids_pipeline_success_and_failure(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.logger = Mock()
        p.config = DotConfig({})
        p.bids_root = Path("/tmp/bids")
        p.deriv_root = Path("/tmp/deriv")

        with patch.object(PreprocessingPipeline, "_generate_mne_bids_config", return_value="x=1"), patch(
            "eeg_pipeline.pipelines.preprocessing.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout="ok", stderr=""),
        ):
            p._run_mne_bids_pipeline("init", subjects=["0001"])

        with patch.object(PreprocessingPipeline, "_generate_mne_bids_config", return_value="x=1"), patch(
            "eeg_pipeline.pipelines.preprocessing.subprocess.run",
            return_value=SimpleNamespace(returncode=1, stdout="", stderr="boom"),
        ):
            with self.assertRaises(RuntimeError):
                p._run_mne_bids_pipeline("init", subjects=["0001"])

    def test_run_bad_channel_and_ica_labeling(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        tmp = Path(tempfile.mkdtemp())
        p.bids_root = tmp / "bids"
        p.deriv_root = tmp / "deriv"
        p.logger = Mock()
        p.config = DotConfig({"pyprep": {}, "icalabel": {}, "eeg": {"montage": "easycap-M1"}})

        mock_preproc = types.SimpleNamespace(
            run_bads_detection=Mock(),
            synchronize_bad_channels_across_runs=Mock(),
        )
        mock_ica = types.SimpleNamespace(run_ica_label=Mock())

        with patch.dict(
            sys.modules,
            {
                "eeg_pipeline.preprocessing.pipeline.preprocess": mock_preproc,
                "eeg_pipeline.preprocessing.pipeline.ica": mock_ica,
            },
        ):
            p._run_bad_channel_detection(["0001"], "task", n_jobs=2)
            p._run_ica_labeling(["0001"], "task")

        self.assertTrue(mock_preproc.run_bads_detection.called)
        self.assertTrue(mock_ica.run_ica_label.called)

class TestPreprocessingCompletion(unittest.TestCase):
    def test_preprocessing_init_and_ica_helpers(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        cfg = DotConfig({"bids_root": "/tmp/bids"})
        with patch("eeg_pipeline.pipelines.preprocessing.PipelineBase.__init__", lambda self, name, config=None: setattr(self, "config", config or cfg)):
            p = PreprocessingPipeline(config=cfg)
        self.assertEqual(str(p.bids_root), "/tmp/bids")
        self.assertIn("preprocessing/_06a2_find_ica_artifacts", p._get_ica_fitting_steps(use_icalabel=False))

        p.logger = Mock()
        with patch.object(PreprocessingPipeline, "_run_mne_bids_pipeline") as mock_run:
            p._run_ica_fitting(["0001"], "t", use_icalabel=True)
        mock_run.assert_called_once()

    def test_run_batch_writes_reproducibility_metadata(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.name = "preprocessing"
        p.config = DotConfig({"project": {"task": "task"}})
        p.logger = Mock()
        p.bids_root = Path(tempfile.mkdtemp()) / "bids"
        p.deriv_root = Path(tempfile.mkdtemp())

        with patch.object(PreprocessingPipeline, "_execute_steps"):
            out = p.run_batch(
                subjects=["0001"],
                task="task",
                mode="epochs",
                progress=_NoopProgress(),
            )
        self.assertEqual(out[0]["status"], "success")

        metadata_dir = p.deriv_root / "logs" / "run_metadata" / "preprocessing"
        metadata_files = sorted(metadata_dir.glob("run_*.json"))
        self.assertTrue(metadata_files)

        payload = json.loads(metadata_files[-1].read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["specifications"]["mode"], "epochs")

    def test_extract_params_and_process_subject(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.config = DotConfig({"project": {"task": "task"}})
        p.logger = Mock()

        fake_cli = types.SimpleNamespace(ProgressReporter=lambda enabled=False: _NoopProgress())
        with patch.dict(sys.modules, {"eeg_pipeline.cli.common": fake_cli}):
            task, mode, use_pyprep, use_icalabel, n_jobs, progress = p._extract_preprocessing_params(None, {})
        self.assertEqual(task, "task")
        self.assertEqual(mode, "full")
        self.assertTrue(use_pyprep)
        self.assertTrue(use_icalabel)
        self.assertEqual(n_jobs, 1)
        self.assertIsNotNone(progress)

        with patch.object(PreprocessingPipeline, "_extract_preprocessing_params", return_value=("task", "full", True, True, 1, _NoopProgress())), patch.object(
            PreprocessingPipeline, "_get_steps_for_mode", return_value=["bad-channels"]
        ), patch.object(PreprocessingPipeline, "_execute_steps") as mock_exec:
            p.process_subject("0001", task=None)
        mock_exec.assert_called_once()

    def test_execute_steps_and_epoch_related_branches(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.logger = Mock()
        p.config = DotConfig({"preprocessing": {"write_clean_events": True}})

        with patch.object(PreprocessingPipeline, "_run_bad_channel_detection") as m1, patch.object(
            PreprocessingPipeline, "_run_ica_fitting"
        ) as m2, patch.object(PreprocessingPipeline, "_run_ica_labeling") as m3, patch.object(
            PreprocessingPipeline, "_run_epoch_creation"
        ) as m4, patch.object(PreprocessingPipeline, "_collect_stats") as m5:
            p._execute_steps(["bad-channels", "ica-fit", "ica-label", "epochs", "stats"], ["0001"], "t", True, True, 1, _NoopProgress())

        self.assertTrue(m1.called and m2.called and m3.called and m4.called and m5.called)

        with patch.object(PreprocessingPipeline, "_run_ica_labeling") as m3:
            p._execute_steps(["ica-label"], ["0001"], "t", True, False, 1, _NoopProgress())
        m3.assert_not_called()

        with patch.object(PreprocessingPipeline, "_run_bad_channel_detection") as m1:
            p._execute_steps(["bad-channels"], ["0001"], "t", False, True, 1, _NoopProgress())
        m1.assert_not_called()

    def test_run_epoch_creation_and_collect_stats(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.logger = Mock()
        p.config = DotConfig({"preprocessing": {"write_clean_events": True}})
        p.bids_root = Path(tempfile.mkdtemp())
        p.deriv_root = Path(tempfile.mkdtemp())

        with patch.object(PreprocessingPipeline, "_run_mne_bids_pipeline") as run_mne, patch.object(
            PreprocessingPipeline, "_write_clean_events_tsv"
        ) as write_clean:
            p._run_epoch_creation(["0001"], "t")
        run_mne.assert_called_once()
        write_clean.assert_called_once()

        fake_stats = types.SimpleNamespace(collect_preprocessing_stats=Mock())
        with patch.dict(sys.modules, {"eeg_pipeline.preprocessing.pipeline.stats": fake_stats}):
            p._collect_stats("t")
        fake_stats.collect_preprocessing_stats.assert_called_once()

    def test_resolve_and_write_clean_events(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.logger = Mock()
        p.config = DotConfig({"epochs": {"conditions": ["a"]}, "preprocessing": {"clean_events_strict": False}})
        p.bids_root = Path(tempfile.mkdtemp())
        p.deriv_root = Path(tempfile.mkdtemp())

        self.assertEqual(p._resolve_epoch_conditions(), ["a"])

        fake_paths = types.SimpleNamespace(find_clean_epochs_path=lambda *a, **k: None)
        fake_preproc = types.SimpleNamespace(write_clean_events_tsv_for_epochs=Mock())
        with patch.dict(
            sys.modules,
            {
                "eeg_pipeline.infra.paths": fake_paths,
                "eeg_pipeline.utils.data.preprocessing": fake_preproc,
            },
        ):
            p._write_clean_events_tsv(subjects=["0001"], task="t")

        p.config = DotConfig({"preprocessing": {"clean_events_strict": True}})
        with patch.dict(
            sys.modules,
            {
                "eeg_pipeline.infra.paths": fake_paths,
                "eeg_pipeline.utils.data.preprocessing": fake_preproc,
            },
        ):
            with self.assertRaises(FileNotFoundError):
                p._write_clean_events_tsv(subjects=["0001"], task="t")

    def test_detect_conditions_edge_cases(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.bids_root = Path(tempfile.mkdtemp())
        p.logger = Mock()

        self.assertIsNone(p._detect_conditions_from_bids())

        # missing trial_type column
        ev_dir = p.bids_root / "sub-0001" / "eeg"
        ev_dir.mkdir(parents=True, exist_ok=True)
        ev = ev_dir / "x_events.tsv"
        ev.write_text("onset\n0\n", encoding="utf-8")
        self.assertIsNone(p._detect_conditions_from_bids())

        # many filtered conditions -> warning branch
        many = "\n".join([f"Cond{i}\t0" for i in range(60)])
        ev.write_text(f"trial_type\tonset\n{many}\n", encoding="utf-8")
        self.assertIsNone(p._detect_conditions_from_bids())

        # exception while reading -> debug branch
        with patch("builtins.open", side_effect=RuntimeError("bad-open")):
            self.assertIsNone(p._detect_conditions_from_bids())

    def test_generate_config_additional_branches_and_clean_events_error_paths(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.bids_root = Path("/tmp/bids")
        p.deriv_root = Path("/tmp/deriv")
        p.logger = Mock()
        p.config = DotConfig(
            {
                "eeg": {"ch_types": "eeg", "reference": "avg", "eog_channels": 1},
                "preprocessing": {"notch_freq": 60, "resample_freq": 200},
                "ica": {"reject": {"eeg": 1e-4}},
                "epochs": {
                    "conditions": ["stim"],
                    "baseline": None,
                    "reject_method": "autoreject_local",
                    "reject_tmin": 0.1,
                    "reject_tmax": 0.5,
                    "autoreject_n_interpolate": [1, 2],
                },
            }
        )
        cfg = p._generate_mne_bids_config("x", subjects=["0001"])
        self.assertIn('eog_channels = ["1"]', cfg)
        self.assertIn("notch_freq = 60", cfg)
        self.assertIn("raw_resample_sfreq = 200", cfg)
        self.assertIn("ica_reject = {'eeg': 0.0001}", cfg)
        self.assertIn('reject = "autoreject_local"', cfg)
        self.assertIn("reject_tmin = 0.1", cfg)
        self.assertIn("reject_tmax = 0.5", cfg)
        self.assertIn("autoreject_n_interpolate = [1, 2]", cfg)

        # _write_clean_events_tsv exception branches
        p.config = DotConfig({"preprocessing": {"clean_events_strict": False}})
        p.bids_root = Path(tempfile.mkdtemp())
        p.deriv_root = Path(tempfile.mkdtemp())
        epochs_path = Path(tempfile.mkdtemp()) / "epo.fif"
        epochs_path.write_text("x", encoding="utf-8")
        fake_paths = types.SimpleNamespace(find_clean_epochs_path=lambda *a, **k: epochs_path)
        fake_preproc = types.SimpleNamespace(
            write_clean_events_tsv_for_epochs=Mock(side_effect=RuntimeError("boom"))
        )
        with patch.dict(
            sys.modules,
            {
                "eeg_pipeline.infra.paths": fake_paths,
                "eeg_pipeline.utils.data.preprocessing": fake_preproc,
            },
        ):
            p._write_clean_events_tsv(subjects=["0001"], task="t")

        p.config = DotConfig({"preprocessing": {"clean_events_strict": True}})
        with patch.dict(
            sys.modules,
            {
                "eeg_pipeline.infra.paths": fake_paths,
                "eeg_pipeline.utils.data.preprocessing": fake_preproc,
            },
        ):
            with self.assertRaises(RuntimeError):
                p._write_clean_events_tsv(subjects=["0001"], task="t")

class TestPreprocessingGapfill(unittest.TestCase):
        def test_preprocessing_config_generation_and_condition_detection_branches(self):
            from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

            p = object.__new__(PreprocessingPipeline)
            p.logger = Mock()
            p.bids_root = Path(tempfile.mkdtemp())
            p.deriv_root = Path(tempfile.mkdtemp())
            p.config = DotConfig(
                {
                    "eeg": {"eog_channels": ["EOG1", "EOG2"]},
                    "epochs": {"reject": {"eeg": 0.0002}},
                }
            )
            txt = p._generate_mne_bids_config(["0001"], "t")
            self.assertIn("eog_channels = ['EOG1', 'EOG2']", txt)
            self.assertIn("reject = {'eeg': 0.0002}", txt)

            # No conditions in file -> warning path in config generation (via detect=None)
            with patch.object(PreprocessingPipeline, "_detect_conditions_from_bids", return_value=None):
                p.config = DotConfig({"epochs": {}, "eeg": {}})
                _ = p._generate_mne_bids_config(["0001"], "t")
            self.assertTrue(p.logger.warning.called)

            # _detect_conditions_from_bids: no usable conditions -> None
            ev_dir = p.bids_root / "sub-0001" / "eeg"
            ev_dir.mkdir(parents=True, exist_ok=True)
            (ev_dir / "sub-0001_task-t_run-01_events.tsv").write_text("trial_type\tonset\nVolume\t0\nPulse Artifact\t1\n", encoding="utf-8")
            self.assertIsNone(p._detect_conditions_from_bids())

            # Empty/non-usable trial_type values -> empty conditions set branch
            (ev_dir / "sub-0001_task-t_run-01_events.tsv").write_text("trial_type\tonset\nn/a\t0\nn/a\t1\n", encoding="utf-8")
            self.assertIsNone(p._detect_conditions_from_bids())

            # filtered return branch
            (ev_dir / "sub-0001_task-t_run-01_events.tsv").write_text("trial_type\tonset\nCueA\t0\nCueB\t1\n", encoding="utf-8")
            self.assertEqual(p._detect_conditions_from_bids(), ["CueA", "CueB"])
