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

        def _create_run_metadata_context(self, *, subjects, task, kwargs):
            return {
                "run_id": "test-run",
                "started_at": 0,
                "task": task,
                "subjects": list(subjects),
                "specifications": {k: v for k, v in kwargs.items() if k != "progress"},
            }

        def _write_run_metadata(
            self, run_context, *, status, error=None, outputs=None, summary=None
        ):
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


def _preprocessing_import_stubs() -> dict[str, types.ModuleType]:
    def _get_config_value(config, key, default=None):
        return config.get(key, default) if hasattr(config, "get") else default

    return {
        "eeg_pipeline.pipelines.base": _make_module(
            "eeg_pipeline.pipelines.base",
            PipelineBase=_make_pipeline_base_class(),
        ),
        "eeg_pipeline.pipelines.progress": _make_module(
            "eeg_pipeline.pipelines.progress",
            ensure_progress_reporter=lambda progress=None: progress or _NoopProgress(),
        ),
        "eeg_pipeline.utils": _make_package("eeg_pipeline.utils"),
        "eeg_pipeline.utils.config": _make_package("eeg_pipeline.utils.config"),
        "eeg_pipeline.utils.config.loader": _make_module(
            "eeg_pipeline.utils.config.loader",
            get_condition_column_candidates=lambda config: (
                config.get("event_columns.condition", []) if hasattr(config, "get") else []
            ),
        ),
        "eeg_pipeline.utils.config.roots": _make_module(
            "eeg_pipeline.utils.config.roots",
            resolve_eeg_bids_root=lambda config, task_is_rest=False: Path(
                _get_config_value(
                    config,
                    "paths.bids_rest_root" if task_is_rest else "paths.bids_root",
                    "/tmp/bids",
                )
            ),
            resolve_eeg_deriv_root=lambda config, task_is_rest=False: Path(
                _get_config_value(
                    config,
                    "paths.deriv_rest_root" if task_is_rest else "paths.deriv_root",
                    "/tmp/deriv",
                )
            ),
        ),
        "eeg_pipeline.preprocessing": _make_package("eeg_pipeline.preprocessing"),
        "eeg_pipeline.preprocessing.pipeline": _make_package("eeg_pipeline.preprocessing.pipeline"),
    }


class _PreprocessingImportMixin:
    def setUp(self):
        patcher = patch.dict(sys.modules, _preprocessing_import_stubs())
        patcher.start()
        self.addCleanup(patcher.stop)


class _TrackingProgress:
    def __init__(self):
        self.complete_calls = []
        self.subject_done_calls = []

    def start(self, *_args, **_kwargs):
        return None

    def step(self, *_args, **_kwargs):
        return None

    def subject_start(self, *_args, **_kwargs):
        return None

    def subject_done(self, subject, success=True):
        self.subject_done_calls.append((subject, success))

    def complete(self, success, duration=None, outputs=None):
        self.complete_calls.append((success, duration, outputs))

    def error(self, *_args, **_kwargs):
        return None


class TestPreprocessingHelpers(_PreprocessingImportMixin, unittest.TestCase):
    def _make_bad_channel_pipeline(self, pyprep_config):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.bids_root = Path("/tmp/bids")
        p.deriv_root = Path("/tmp/deriv")
        p.logger = Mock()
        p.config = DotConfig(
            {
                "project": {"random_state": 42},
                "eeg": {"montage": "easycap-M1"},
                "preprocessing": {"h_freq": 100, "notch_freq": None},
                "pyprep": pyprep_config,
            }
        )
        return p

    def test_bad_channel_policy_per_run_skips_subject_union_sync(self):
        calls = {"detect": 0, "sync": 0}

        def fake_run_bads_detection(**_kwargs):
            calls["detect"] += 1

        def fake_sync(**_kwargs):
            calls["sync"] += 1

        preprocess_module = _make_module(
            "eeg_pipeline.preprocessing.pipeline.preprocess",
            run_bads_detection=fake_run_bads_detection,
            synchronize_bad_channels_across_runs=fake_sync,
        )

        with patch.dict(
            sys.modules,
            {"eeg_pipeline.preprocessing.pipeline.preprocess": preprocess_module},
        ):
            pipeline = self._make_bad_channel_pipeline({"bad_channel_sync_policy": "per_run"})
            pipeline._run_bad_channel_detection(
                subjects=["0001"],
                task="task",
                n_jobs=1,
            )

        self.assertEqual(calls["detect"], 1)
        self.assertEqual(calls["sync"], 0)

    def test_bad_channel_policy_subject_union_runs_sync(self):
        calls = {"detect": 0, "sync": 0}

        def fake_run_bads_detection(**_kwargs):
            calls["detect"] += 1

        def fake_sync(**_kwargs):
            calls["sync"] += 1

        preprocess_module = _make_module(
            "eeg_pipeline.preprocessing.pipeline.preprocess",
            run_bads_detection=fake_run_bads_detection,
            synchronize_bad_channels_across_runs=fake_sync,
        )

        with patch.dict(
            sys.modules,
            {"eeg_pipeline.preprocessing.pipeline.preprocess": preprocess_module},
        ):
            pipeline = self._make_bad_channel_pipeline({"bad_channel_sync_policy": "subject_union"})
            pipeline._run_bad_channel_detection(
                subjects=["0001"],
                task="task",
                n_jobs=1,
            )

        self.assertEqual(calls["detect"], 1)
        self.assertEqual(calls["sync"], 1)

    def test_bad_channel_policy_missing_fails_fast(self):
        preprocess_module = _make_module(
            "eeg_pipeline.preprocessing.pipeline.preprocess",
            run_bads_detection=lambda **_kwargs: None,
            synchronize_bad_channels_across_runs=lambda **_kwargs: None,
        )

        with patch.dict(
            sys.modules,
            {"eeg_pipeline.preprocessing.pipeline.preprocess": preprocess_module},
        ):
            pipeline = self._make_bad_channel_pipeline({})
            with self.assertRaisesRegex(ValueError, "bad_channel_sync_policy"):
                pipeline._run_bad_channel_detection(
                    subjects=["0001"],
                    task="task",
                    n_jobs=1,
                )

    def test_preprocessing_helper_defaults_and_validation_edges(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.config = DotConfig(
            {
                "project": {"task": "task"},
                "preprocessing": {"task_is_rest": True},
            }
        )

        self.assertTrue(p._resolve_task_is_rest())
        self.assertFalse(p._resolve_task_is_rest(False))
        self.assertEqual(p._normalize_subjects(["all"]), "all")
        self.assertEqual(p._normalize_subjects(["0001"]), ["0001"])
        self.assertIn(
            "preprocessing/_06a2_find_ica_artifacts", p._get_ica_fitting_steps(use_icalabel=False)
        )
        self.assertNotIn(
            "preprocessing/_06a2_find_ica_artifacts", p._get_ica_fitting_steps(use_icalabel=True)
        )

        resolved = p._extract_preprocessing_params(
            "task",
            {
                "mode": "ica",
                "use_pyprep": False,
                "use_icalabel": False,
                "task_is_rest": False,
                "n_jobs": 4,
                "progress": _NoopProgress(),
            },
        )
        self.assertEqual(resolved[0], "task")
        self.assertEqual(resolved[1], "ica")
        self.assertFalse(resolved[2])
        self.assertFalse(resolved[3])
        self.assertFalse(resolved[4])
        self.assertEqual(resolved[5], 4)

        with self.assertRaisesRegex(ValueError, "Unknown preprocessing mode"):
            p._get_steps_for_mode("bogus")

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
                "preprocessing": {
                    "task_is_rest": False,
                    "l_freq": 0.1,
                    "h_freq": 40.0,
                    "find_breaks": True,
                },
                "ica": {"algorithm": "picard", "n_components": 0.99},
                "epochs": {
                    "baseline": [None, 0],
                    "reject_method": "none",
                    "tmin": -0.2,
                    "tmax": 0.8,
                },
            }
        )
        with patch.object(
            PreprocessingPipeline, "_detect_conditions_from_bids", return_value=["stim"]
        ):
            cfg = p._generate_mne_bids_config("preprocessing/_07_make_epochs", subjects=["0001"])
        self.assertIn("subjects = ['0001']", cfg)
        self.assertIn("conditions = ['stim']", cfg)
        self.assertIn("baseline = (None, 0)", cfg)
        self.assertNotIn("reject =", cfg)

    def test_generate_mne_bids_config_includes_requested_task(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.bids_root = Path("/tmp/bids")
        p.deriv_root = Path("/tmp/deriv")
        p.logger = Mock()
        p.config = DotConfig(
            {
                "preprocessing": {"task_is_rest": False},
                "epochs": {},
            }
        )

        with patch.object(
            PreprocessingPipeline, "_detect_conditions_from_bids", return_value=["stim"]
        ) as mock_detect:
            cfg = p._generate_mne_bids_config(
                "preprocessing/_07_make_epochs",
                subjects=["0001"],
                task="pain",
            )

        self.assertIn('task = "pain"', cfg)
        mock_detect.assert_called_once_with("pain")

    def test_generate_mne_bids_config_for_resting_state(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.bids_root = Path("/tmp/bids")
        p.deriv_root = Path("/tmp/deriv")
        p.logger = Mock()
        p.config = DotConfig(
            {
                "preprocessing": {
                    "task_is_rest": True,
                    "rest_epochs_duration": 12.0,
                    "rest_epochs_overlap": 0.0,
                },
                "epochs": {"baseline": [-0.2, 0.0], "tmin": -7.0, "tmax": 15.0},
            }
        )

        with patch.object(PreprocessingPipeline, "_detect_conditions_from_bids") as mock_detect:
            cfg = p._generate_mne_bids_config("preprocessing/_07_make_epochs", subjects=["0001"])

        mock_detect.assert_not_called()
        self.assertIn("task_is_rest = True", cfg)
        self.assertIn("conditions = None", cfg)
        self.assertIn("epochs_tmin = 0.0", cfg)
        self.assertIn("baseline = None", cfg)
        self.assertIn("rest_epochs_duration = 12.0", cfg)
        self.assertIn("rest_epochs_overlap = 0.0", cfg)
        self.assertNotIn("epochs_tmax =", cfg)

    def test_generate_mne_bids_config_omits_project_task_for_resting_state_without_override(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.bids_root = Path("/tmp/bids-rest")
        p.deriv_root = Path("/tmp/deriv-rest")
        p.logger = Mock()
        p.config = DotConfig(
            {
                "project": {"task": "pain"},
                "preprocessing": {
                    "task_is_rest": True,
                    "rest_epochs_duration": 12.0,
                    "rest_epochs_overlap": 0.0,
                },
            }
        )

        cfg = p._generate_mne_bids_config(
            "preprocessing/_07_make_epochs",
            subjects=["0001"],
            task=None,
        )

        self.assertNotIn('task = "pain"', cfg)

    def test_preprocessing_helper_failure_validation_and_config_branches(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.name = "preprocessing"
        p.logger = Mock()
        p.bids_root = Path(tempfile.mkdtemp())
        p.deriv_root = Path(tempfile.mkdtemp())
        p.config = DotConfig(
            {
                "project": {"task": "task"},
                "paths": {"deriv_root": "/tmp/deriv-task"},
                "preprocessing": {"task_is_rest": False},
            }
        )

        self.assertEqual(p._resolve_pipeline_deriv_root(), Path("/tmp/deriv-task"))

        progress = Mock()
        with (
            patch.object(
                PreprocessingPipeline,
                "_extract_preprocessing_params",
                return_value=("task", "epochs", True, True, False, 1, progress),
            ),
            patch.object(
                PreprocessingPipeline,
                "_get_steps_for_mode",
                return_value=["epochs"],
            ),
            patch.object(
                PreprocessingPipeline,
                "_execute_steps",
                side_effect=RuntimeError("boom"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                p.run_batch(subjects=["0001"], task="task", mode="epochs")

        payload = json.loads(
            (
                p.deriv_root / "logs" / "run_metadata" / "preprocessing" / "run_test-run.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["error"], "boom")

        p.config = DotConfig({"preprocessing": {"task_is_rest": True}})
        with self.assertRaisesRegex(ValueError, "requires preprocessing.rest_epochs_duration"):
            p._get_rest_epoch_parameters()

        p.config = DotConfig(
            {
                "preprocessing": {
                    "task_is_rest": True,
                    "rest_epochs_duration": 0.0,
                    "rest_epochs_overlap": 0.0,
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "must be greater than 0"):
            p._get_rest_epoch_parameters()

        p.config = DotConfig(
            {
                "preprocessing": {
                    "task_is_rest": True,
                    "rest_epochs_duration": 12.0,
                    "rest_epochs_overlap": -1.0,
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "greater than or equal to 0"):
            p._get_rest_epoch_parameters()

        p.config = DotConfig(
            {
                "project": {"task": "task"},
                "eeg": {"ch_types": "eeg"},
                "preprocessing": {"task_is_rest": False, "random_state": 7},
                "epochs": {"conditions": ["stim"]},
            }
        )
        cfg = p._generate_mne_bids_config("preprocessing/_07_make_epochs", subjects=["0001"])
        self.assertIn("random_state = 7", cfg)

    def test_write_clean_events_and_condition_preference_branches(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.logger = Mock()
        p.bids_root = Path(tempfile.mkdtemp())
        p.deriv_root = Path(tempfile.mkdtemp())
        epochs_path = p.deriv_root / "sub-0001-epo.fif"
        epochs_path.write_text("x", encoding="utf-8")

        p.config = DotConfig(
            {
                "epochs": {"conditions": ["a"]},
                "preprocessing": {"clean_events_overwrite": False, "clean_events_strict": True},
            }
        )
        fake_paths = types.SimpleNamespace(find_clean_epochs_path=lambda *a, **k: epochs_path)
        fake_preproc = types.SimpleNamespace(write_clean_events_tsv_for_epochs=Mock())
        with patch.dict(
            sys.modules,
            {
                "eeg_pipeline.infra.paths": fake_paths,
                "eeg_pipeline.utils.data.preprocessing": fake_preproc,
            },
        ):
            p._write_clean_events_tsv(subjects=["0001"], task="t")

        kwargs = fake_preproc.write_clean_events_tsv_for_epochs.call_args.kwargs
        self.assertNotIn("condition_columns", kwargs)
        self.assertFalse(kwargs["overwrite"])

        ev_dir = p.bids_root / "sub-0001" / "eeg"
        ev_dir.mkdir(parents=True, exist_ok=True)
        events_path = ev_dir / "sub-0001_task-task_run-01_events.tsv"
        events_path.write_text(
            "trial_type\tonset\nCueA\t0\nVolume\t1\nCueB\t2\n",
            encoding="utf-8",
        )

        p.config = DotConfig({"preprocessing": {"condition_preferred_prefixes": ["Cue"]}})
        self.assertEqual(p._detect_conditions_from_bids(), ["CueA", "CueB"])

        p.config = DotConfig({"preprocessing": {"condition_preferred_prefixes": "Cue"}})
        self.assertEqual(p._detect_conditions_from_bids(), ["CueA", "CueB"])

    def test_resolve_epoch_conditions_uses_requested_task(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.logger = Mock()
        p.config = DotConfig({"epochs": {}})

        with patch.object(
            PreprocessingPipeline, "_detect_conditions_from_bids", return_value=["stim"]
        ) as mock_detect:
            self.assertEqual(p._resolve_epoch_conditions(task="pain"), ["stim"])

        mock_detect.assert_called_once_with("pain")

    def test_generate_mne_bids_config_rejects_overlapping_rest_epochs(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.bids_root = Path("/tmp/bids")
        p.deriv_root = Path("/tmp/deriv")
        p.logger = Mock()
        p.config = DotConfig(
            {
                "preprocessing": {
                    "task_is_rest": True,
                    "rest_epochs_duration": 12.0,
                    "rest_epochs_overlap": 3.0,
                },
            }
        )

        with self.assertRaisesRegex(ValueError, "rest_epochs_overlap > 0"):
            p._generate_mne_bids_config("preprocessing/_07_make_epochs", subjects=["0001"])

    def test_generate_mne_bids_config_requires_conditions_for_task_data(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.bids_root = Path("/tmp/bids")
        p.deriv_root = Path("/tmp/deriv")
        p.logger = Mock()
        p.config = DotConfig({"preprocessing": {"task_is_rest": False}, "epochs": {}})

        with patch.object(PreprocessingPipeline, "_detect_conditions_from_bids", return_value=None):
            with self.assertRaisesRegex(
                ValueError, "requires epochs.conditions or detectable BIDS event conditions"
            ):
                p._generate_mne_bids_config("preprocessing/_07_make_epochs", subjects=["0001"])

    def test_run_mne_bids_pipeline_success_and_failure(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.logger = Mock()
        p.config = DotConfig({})
        p.bids_root = Path("/tmp/bids")
        p.deriv_root = Path("/tmp/deriv")

        with (
            patch.object(
                PreprocessingPipeline, "_generate_mne_bids_config", return_value="x=1"
            ) as mock_generate,
            patch(
                "eeg_pipeline.pipelines.preprocessing.subprocess.run",
                return_value=SimpleNamespace(returncode=0, stdout="ok", stderr=""),
            ),
        ):
            p._run_mne_bids_pipeline("init", subjects=["0001"], task="pain")
        self.assertEqual(mock_generate.call_args.kwargs["task"], "pain")

        with (
            patch.object(PreprocessingPipeline, "_generate_mne_bids_config", return_value="x=1"),
            patch(
                "eeg_pipeline.pipelines.preprocessing.subprocess.run",
                return_value=SimpleNamespace(returncode=1, stdout="", stderr="boom"),
            ),
        ):
            with self.assertRaises(RuntimeError):
                p._run_mne_bids_pipeline("init", subjects=["0001"], task="pain")

    def test_run_mne_bids_pipeline_failure_reports_stdout_and_stderr(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.logger = Mock()
        p.config = DotConfig({})
        p.bids_root = Path("/tmp/bids")
        p.deriv_root = Path("/tmp/deriv")

        stdout = "Traceback (most recent call last):\nValueError: actual failure"
        stderr = "UserWarning: nperseg = 2048 is greater than input length = 1"

        with (
            patch.object(PreprocessingPipeline, "_generate_mne_bids_config", return_value="x=1"),
            patch(
                "eeg_pipeline.pipelines.preprocessing.subprocess.run",
                return_value=SimpleNamespace(returncode=1, stdout=stdout, stderr=stderr),
            ),
        ):
            with self.assertRaises(RuntimeError) as exc:
                p._run_mne_bids_pipeline("init", subjects=["0001"], task="pain")

        error_message = str(exc.exception)
        self.assertIn(stdout, error_message)
        self.assertIn(stderr, error_message)

    def test_run_bad_channel_and_ica_labeling(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        tmp = Path(tempfile.mkdtemp())
        p.bids_root = tmp / "bids"
        p.deriv_root = tmp / "deriv"
        p.logger = Mock()
        p.config = DotConfig(
            {
                "pyprep": {"bad_channel_sync_policy": "per_run"},
                "icalabel": {},
                "eeg": {"montage": "easycap-M1"},
            }
        )

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

    def test_harmonize_filtered_raw_bads_uses_subject_union(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.logger = Mock()
        p.config = DotConfig({"pyprep": {"bad_channel_sync_policy": "subject_union"}})
        p.deriv_root = Path(tempfile.mkdtemp())

        eeg_dir = p.deriv_root / "preprocessed" / "eeg" / "sub-0001" / "eeg"
        eeg_dir.mkdir(parents=True)
        run_1 = eeg_dir / "sub-0001_task-pain_run-1_proc-filt_raw.fif"
        run_2 = eeg_dir / "sub-0001_task-pain_run-2_proc-filt_raw.fif"
        run_1.write_text("raw-1", encoding="utf-8")
        run_2.write_text("raw-2", encoding="utf-8")

        class FakeRaw:
            def __init__(self, bads):
                self.info = {"bads": list(bads)}
                self.ch_names = ["C3", "C4", "Cz", "ECG"]

            def get_channel_types(self, picks):
                assert picks == self.ch_names
                return ["eeg", "eeg", "eeg", "ecg"]

            def load_data(self):
                return None

            def save(self, path, overwrite, split_naming):
                self.saved_path = Path(path)
                self.saved_overwrite = overwrite
                self.saved_split_naming = split_naming
                Path(path).write_text("updated", encoding="utf-8")

        raws = {
            str(run_1): FakeRaw(["C3"]),
            str(run_2): FakeRaw(["C4"]),
        }

        def fake_read_raw_fif(path, preload, verbose):
            self.assertTrue(preload)
            self.assertFalse(verbose)
            return raws[str(path)]

        fake_mne = types.SimpleNamespace(io=types.SimpleNamespace(read_raw_fif=fake_read_raw_fif))

        with patch.dict(sys.modules, {"mne": fake_mne}):
            p._harmonize_filtered_raw_bads_for_mne_concat(["0001"], "pain")

        expected_bads = ["C3", "C4"]
        self.assertEqual(raws[str(run_1)].info["bads"], expected_bads)
        self.assertEqual(raws[str(run_2)].info["bads"], expected_bads)
        self.assertEqual(run_1.read_text(encoding="utf-8"), "updated")
        self.assertEqual(run_2.read_text(encoding="utf-8"), "updated")
        qc_path = p.deriv_root / "preprocessed" / "eeg" / "bad_channel_union_qc_task-pain.tsv"
        qc_text = qc_path.read_text(encoding="utf-8")
        self.assertIn("bad_channel_fraction", qc_text)
        self.assertIn("0.666667", qc_text)
        self.assertIn("C3,C4", qc_text)

    def test_mismatched_filtered_raw_bads_fail_with_per_run_policy(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.logger = Mock()
        p.config = DotConfig({"pyprep": {"bad_channel_sync_policy": "per_run"}})
        p.deriv_root = Path(tempfile.mkdtemp())

        eeg_dir = p.deriv_root / "preprocessed" / "eeg" / "sub-0001" / "eeg"
        eeg_dir.mkdir(parents=True)
        run_1 = eeg_dir / "sub-0001_task-pain_run-1_proc-filt_raw.fif"
        run_2 = eeg_dir / "sub-0001_task-pain_run-2_proc-filt_raw.fif"
        run_1.write_text("raw-1", encoding="utf-8")
        run_2.write_text("raw-2", encoding="utf-8")

        class FakeRaw:
            def __init__(self, bads):
                self.info = {"bads": list(bads)}
                self.ch_names = ["C3", "C4", "Cz", "ECG"]

            def get_channel_types(self, picks):
                assert picks == self.ch_names
                return ["eeg", "eeg", "eeg", "ecg"]

            def save(self, *_args, **_kwargs):
                raise AssertionError("per-run mismatch must not rewrite filtered FIFs")

        raws = {
            str(run_1): FakeRaw(["C3"]),
            str(run_2): FakeRaw(["C4"]),
        }

        def fake_read_raw_fif(path, preload, verbose):
            self.assertTrue(preload)
            self.assertFalse(verbose)
            return raws[str(path)]

        fake_mne = types.SimpleNamespace(io=types.SimpleNamespace(read_raw_fif=fake_read_raw_fif))

        with patch.dict(sys.modules, {"mne": fake_mne}):
            with self.assertRaisesRegex(ValueError, "subject_union"):
                p._harmonize_filtered_raw_bads_for_mne_concat(["0001"], "pain")

        self.assertEqual(run_1.read_text(encoding="utf-8"), "raw-1")
        self.assertEqual(run_2.read_text(encoding="utf-8"), "raw-2")

    def test_find_filtered_raw_run_files_ignores_appledouble_metadata(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.deriv_root = Path(tempfile.mkdtemp())

        eeg_dir = p.deriv_root / "preprocessed" / "eeg" / "sub-0001" / "eeg"
        eeg_dir.mkdir(parents=True)
        run_1 = eeg_dir / "sub-0001_task-pain_run-1_proc-filt_raw.fif"
        run_2 = eeg_dir / "sub-0001_task-pain_run-2_proc-filt_raw.fif"
        appledouble = eeg_dir / "._sub-0001_task-pain_run-1_proc-filt_raw.fif"
        for path in (run_1, run_2, appledouble):
            path.write_text("not a real fif", encoding="utf-8")

        self.assertEqual(
            p._find_filtered_raw_run_files("0001", "pain"),
            [run_1, run_2],
        )


class TestPreprocessingCompletion(_PreprocessingImportMixin, unittest.TestCase):
    def test_preprocessing_init_and_ica_helpers(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        cfg = DotConfig({"bids_root": "/tmp/bids"})
        with patch(
            "eeg_pipeline.pipelines.preprocessing.PipelineBase.__init__",
            lambda self, name, config=None: setattr(self, "config", config or cfg),
        ):
            p = PreprocessingPipeline(config=cfg)
        self.assertEqual(p.bids_root.as_posix(), "/tmp/bids")
        self.assertIn(
            "preprocessing/_06a2_find_ica_artifacts", p._get_ica_fitting_steps(use_icalabel=False)
        )

        p.logger = Mock()
        with (
            patch.object(PreprocessingPipeline, "_run_mne_bids_pipeline") as mock_run,
            patch.object(
                PreprocessingPipeline,
                "_harmonize_filtered_raw_bads_for_mne_concat",
            ) as mock_harmonize,
        ):
            p._run_ica_fitting(["0001"], "t", use_icalabel=True)
        self.assertEqual(mock_run.call_count, 2)
        self.assertEqual(
            mock_run.call_args_list[0].args[0],
            "init,preprocessing/_01_data_quality,preprocessing/_04_frequency_filter,"
            "preprocessing/_05_regress_artifact",
        )
        self.assertEqual(mock_run.call_args_list[1].args[0], "preprocessing/_06a1_fit_ica")
        mock_harmonize.assert_called_once_with(["0001"], "t")

    def test_preprocessing_init_uses_rest_bids_root_in_rest_mode(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        cfg = DotConfig(
            {
                "paths": {
                    "bids_root": "/tmp/bids-task",
                    "bids_rest_root": "/tmp/bids-rest",
                    "deriv_root": "/tmp/derivatives-task",
                    "deriv_rest_root": "/tmp/derivatives-rest",
                },
                "preprocessing": {"task_is_rest": True},
            }
        )
        with patch(
            "eeg_pipeline.pipelines.preprocessing.PipelineBase.__init__",
            lambda self, name, config=None: setattr(self, "config", config or cfg),
        ):
            p = PreprocessingPipeline(config=cfg)

        self.assertEqual(p.bids_root.as_posix(), "/tmp/bids-rest")
        self.assertEqual(p.deriv_root.as_posix(), "/tmp/derivatives-rest")

    def test_run_batch_updates_roots_for_runtime_rest_override(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        cfg = DotConfig(
            {
                "paths": {
                    "bids_root": "/tmp/bids-task",
                    "bids_rest_root": "/tmp/bids-rest",
                    "deriv_root": "/tmp/derivatives-task",
                    "deriv_rest_root": "/tmp/derivatives-rest",
                },
                "preprocessing": {"task_is_rest": False},
            }
        )

        def _fake_init(self, name, config=None):
            self.name = name
            self.config = config or cfg
            self.logger = Mock()

        with patch("eeg_pipeline.pipelines.preprocessing.PipelineBase.__init__", _fake_init):
            p = PreprocessingPipeline(config=cfg)

        self.assertEqual(p.bids_root.as_posix(), "/tmp/bids-task")
        self.assertEqual(p.deriv_root.as_posix(), "/tmp/derivatives-task")

        with patch.object(PreprocessingPipeline, "_execute_steps"):
            p.run_batch(
                subjects=["0001"],
                task="rest",
                mode="epochs",
                task_is_rest=True,
                progress=_NoopProgress(),
            )

        self.assertEqual(p.bids_root.as_posix(), "/tmp/bids-rest")
        self.assertEqual(p.deriv_root.as_posix(), "/tmp/derivatives-rest")

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

    def test_run_batch_preserves_primary_failure_when_metadata_write_also_fails(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.name = "preprocessing"
        p.config = DotConfig({"project": {"task": "task"}})
        p.logger = Mock()
        p.bids_root = Path(tempfile.mkdtemp()) / "bids"
        p.deriv_root = Path(tempfile.mkdtemp())

        with (
            patch.object(
                PreprocessingPipeline,
                "_execute_steps",
                side_effect=RuntimeError("boom"),
            ),
            patch.object(
                p,
                "_write_run_metadata",
                side_effect=RuntimeError("meta-fail"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom") as exc_info:
                p.run_batch(
                    subjects=["0001"],
                    task="task",
                    mode="epochs",
                    progress=_NoopProgress(),
                )

        self.assertIn("meta-fail", "".join(getattr(exc_info.exception, "__notes__", [])))

    def test_run_batch_reports_failure_completion_to_progress_sink(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.name = "preprocessing"
        p.config = DotConfig({"project": {"task": "task"}})
        p.logger = Mock()
        p.bids_root = Path(tempfile.mkdtemp()) / "bids"
        p.deriv_root = Path(tempfile.mkdtemp())

        progress = _TrackingProgress()
        with patch.object(
            PreprocessingPipeline,
            "_execute_steps",
            side_effect=RuntimeError("boom"),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                p.run_batch(
                    subjects=["0001"],
                    task="task",
                    mode="epochs",
                    progress=progress,
                )

        self.assertEqual(progress.complete_calls, [(False, None, None)])

    def test_run_batch_returns_per_subject_status_dicts(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.name = "preprocessing"
        p.config = DotConfig({"project": {"task": "task"}})
        p.logger = Mock()
        p.bids_root = Path(tempfile.mkdtemp()) / "bids"
        p.deriv_root = Path(tempfile.mkdtemp())

        with patch.object(PreprocessingPipeline, "_execute_steps"):
            out = p.run_batch(
                subjects=["0001", "0002"],
                task="task",
                mode="epochs",
                progress=_NoopProgress(),
            )

        self.assertEqual(
            out,
            [
                {"subject": "0001", "mode": "epochs", "status": "success"},
                {"subject": "0002", "mode": "epochs", "status": "success"},
            ],
        )

    def test_extract_params_and_process_subject(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.config = DotConfig({"project": {"task": "task"}})
        p.logger = Mock()

        fake_cli = types.SimpleNamespace(ProgressReporter=lambda enabled=False: _NoopProgress())
        with patch.dict(sys.modules, {"eeg_pipeline.cli.common": fake_cli}):
            task, mode, use_pyprep, use_icalabel, task_is_rest, n_jobs, progress = (
                p._extract_preprocessing_params(None, {})
            )
        self.assertEqual(task, "task")
        self.assertEqual(mode, "full")
        self.assertTrue(use_pyprep)
        self.assertTrue(use_icalabel)
        self.assertFalse(task_is_rest)
        self.assertEqual(n_jobs, 1)
        self.assertIsNotNone(progress)

        p.config = DotConfig({})
        with self.assertRaisesRegex(ValueError, "Missing required config value: project.task"):
            p._extract_preprocessing_params(None, {})

        p.config = DotConfig({"preprocessing": {"task_is_rest": True}})
        with patch.dict(sys.modules, {"eeg_pipeline.cli.common": fake_cli}):
            task, mode, use_pyprep, use_icalabel, task_is_rest, n_jobs, progress = (
                p._extract_preprocessing_params(None, {})
            )
        self.assertIsNone(task)
        self.assertEqual(mode, "full")
        self.assertTrue(use_pyprep)
        self.assertTrue(use_icalabel)
        self.assertTrue(task_is_rest)
        self.assertEqual(n_jobs, 1)
        self.assertIsNotNone(progress)

        with (
            patch.object(
                PreprocessingPipeline,
                "_extract_preprocessing_params",
                return_value=("task", "full", True, True, False, 1, _NoopProgress()),
            ),
            patch.object(
                PreprocessingPipeline, "_get_steps_for_mode", return_value=["bad-channels"]
            ),
            patch.object(PreprocessingPipeline, "_execute_steps") as mock_exec,
        ):
            p.process_subject("0001", task=None)
        mock_exec.assert_called_once()

    def test_process_subject_reports_failure_to_progress_sink(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.config = DotConfig({"project": {"task": "task"}})
        p.logger = Mock()
        p._refresh_processing_roots_if_initialized = Mock()

        progress = _TrackingProgress()
        with (
            patch.object(
                PreprocessingPipeline,
                "_extract_preprocessing_params",
                return_value=("task", "full", True, True, False, 1, progress),
            ),
            patch.object(
                PreprocessingPipeline,
                "_get_steps_for_mode",
                return_value=["bad-channels"],
            ),
            patch.object(
                PreprocessingPipeline,
                "_execute_steps",
                side_effect=RuntimeError("boom"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                p.process_subject("0001", task=None)

        self.assertEqual(progress.subject_done_calls, [("sub-0001", False)])

    def test_execute_steps_and_epoch_related_branches(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.logger = Mock()
        p.config = DotConfig({"preprocessing": {"write_clean_events": True}})

        with (
            patch.object(PreprocessingPipeline, "_run_bad_channel_detection") as m1,
            patch.object(PreprocessingPipeline, "_run_ica_fitting") as m2,
            patch.object(PreprocessingPipeline, "_run_ica_labeling") as m3,
            patch.object(PreprocessingPipeline, "_run_epoch_creation") as m4,
            patch.object(PreprocessingPipeline, "_collect_stats") as m5,
        ):
            p._execute_steps(
                ["bad-channels", "ica-fit", "ica-label", "epochs", "stats"],
                ["0001"],
                "t",
                True,
                True,
                False,
                1,
                _NoopProgress(),
            )

        self.assertTrue(m1.called and m2.called and m3.called and m4.called and m5.called)

        with patch.object(PreprocessingPipeline, "_run_ica_labeling") as m3:
            p._execute_steps(["ica-label"], ["0001"], "t", True, False, False, 1, _NoopProgress())
        m3.assert_not_called()

        with patch.object(PreprocessingPipeline, "_run_bad_channel_detection") as m1:
            p._execute_steps(
                ["bad-channels"], ["0001"], "t", False, True, False, 1, _NoopProgress()
            )
        m1.assert_not_called()

    def test_run_epoch_creation_and_collect_stats(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.logger = Mock()
        p.config = DotConfig({"preprocessing": {"write_clean_events": True}})
        p.bids_root = Path(tempfile.mkdtemp())
        p.deriv_root = Path(tempfile.mkdtemp())

        with (
            patch.object(PreprocessingPipeline, "_run_mne_bids_pipeline") as run_mne,
            patch.object(PreprocessingPipeline, "_write_clean_events_tsv") as write_clean,
        ):
            p._run_epoch_creation(["0001"], "t", task_is_rest=False)
        run_mne.assert_called_once()
        write_clean.assert_called_once()

        with (
            patch.object(PreprocessingPipeline, "_run_mne_bids_pipeline") as run_mne,
            patch.object(PreprocessingPipeline, "_write_clean_events_tsv") as write_clean,
        ):
            p._run_epoch_creation(["0001"], "t", task_is_rest=True)
        run_mne.assert_called_once()
        write_clean.assert_not_called()

        fake_stats = types.SimpleNamespace(collect_preprocessing_stats=Mock())
        with patch.dict(sys.modules, {"eeg_pipeline.preprocessing.pipeline.stats": fake_stats}):
            p._collect_stats("t")
        fake_stats.collect_preprocessing_stats.assert_called_once()

    def test_run_bad_channel_detection_preserves_explicit_zero_random_state(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.logger = Mock()
        p.config = DotConfig(
            {
                "project": {"random_state": 42},
                "pyprep": {
                    "bad_channel_sync_policy": "per_run",
                    "random_state": 0,
                },
            }
        )
        p.bids_root = Path(tempfile.mkdtemp())
        p.deriv_root = Path(tempfile.mkdtemp())

        fake_preprocess = types.SimpleNamespace(
            run_bads_detection=Mock(),
            synchronize_bad_channels_across_runs=Mock(),
        )
        with patch.dict(
            sys.modules,
            {"eeg_pipeline.preprocessing.pipeline.preprocess": fake_preprocess},
        ):
            p._run_bad_channel_detection(["0001"], "task", n_jobs=2)

        self.assertEqual(
            fake_preprocess.run_bads_detection.call_args.kwargs["random_state"],
            0,
        )

    def test_resolve_and_write_clean_events(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.logger = Mock()
        p.config = DotConfig(
            {"epochs": {"conditions": ["a"]}, "preprocessing": {"clean_events_strict": False}}
        )
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

        # read/parsing errors should surface
        with patch("builtins.open", side_effect=RuntimeError("bad-open")):
            with self.assertRaisesRegex(RuntimeError, "bad-open"):
                p._detect_conditions_from_bids()

    def test_detect_conditions_from_bids_uses_requested_task_in_session_layout(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.bids_root = Path(tempfile.mkdtemp())
        p.logger = Mock()

        session_eeg_dir = p.bids_root / "sub-0001" / "ses-01" / "eeg"
        session_eeg_dir.mkdir(parents=True, exist_ok=True)
        (session_eeg_dir / "sub-0001_ses-01_task-audio_run-01_events.tsv").write_text(
            "trial_type\tonset\nAudioOnly\t0\n",
            encoding="utf-8",
        )
        (session_eeg_dir / "sub-0001_ses-01_task-pain_run-01_events.tsv").write_text(
            "trial_type\tonset\nCueA\t0\nCueB\t1\n",
            encoding="utf-8",
        )

        self.assertEqual(p._detect_conditions_from_bids(task="pain"), ["CueA", "CueB"])

    def test_detect_conditions_from_bids_aggregates_across_matching_files(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.bids_root = Path(tempfile.mkdtemp())
        p.logger = Mock()

        subject_a_dir = p.bids_root / "sub-0001" / "eeg"
        subject_b_dir = p.bids_root / "sub-0002" / "eeg"
        subject_a_dir.mkdir(parents=True, exist_ok=True)
        subject_b_dir.mkdir(parents=True, exist_ok=True)

        (subject_a_dir / "sub-0001_task-pain_run-01_events.tsv").write_text(
            "trial_type\tonset\nVolume\t0\n",
            encoding="utf-8",
        )
        (subject_b_dir / "sub-0002_task-pain_run-01_events.tsv").write_text(
            "trial_type\tonset\nCueA\t0\nCueB\t1\n",
            encoding="utf-8",
        )

        self.assertEqual(p._detect_conditions_from_bids(task="pain"), ["CueA", "CueB"])

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

        p.config = DotConfig(
            {
                "eeg": {"ch_types": ["eeg", "eog"]},
                "epochs": {"conditions": ["stim"]},
            }
        )
        cfg = p._generate_mne_bids_config("x", subjects=["0001"])
        self.assertIn("ch_types = ['eeg', 'eog']", cfg)

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


class TestPreprocessingGapfill(_PreprocessingImportMixin, unittest.TestCase):
    def test_preprocessing_config_generation_and_condition_detection_branches(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.logger = Mock()
        p.bids_root = Path(tempfile.mkdtemp())
        p.deriv_root = Path(tempfile.mkdtemp())
        p.config = DotConfig(
            {
                "eeg": {"eog_channels": ["EOG1", "EOG2"]},
                "epochs": {"conditions": ["stim"], "reject": {"eeg": 0.0002}},
            }
        )
        txt = p._generate_mne_bids_config("t", subjects=["0001"])
        self.assertIn("eog_channels = ['EOG1', 'EOG2']", txt)
        self.assertIn("reject = {'eeg': 0.0002}", txt)

        # No conditions in file -> strict failure for task-based preprocessing
        with patch.object(PreprocessingPipeline, "_detect_conditions_from_bids", return_value=None):
            p.config = DotConfig({"epochs": {}, "eeg": {}})
            with self.assertRaisesRegex(
                ValueError, "requires epochs.conditions or detectable BIDS event conditions"
            ):
                p._generate_mne_bids_config("t", subjects=["0001"])

        # _detect_conditions_from_bids: no usable conditions -> None
        ev_dir = p.bids_root / "sub-0001" / "eeg"
        ev_dir.mkdir(parents=True, exist_ok=True)
        (ev_dir / "sub-0001_task-t_run-01_events.tsv").write_text(
            "trial_type\tonset\nVolume\t0\nPulse Artifact\t1\n", encoding="utf-8"
        )
        self.assertIsNone(p._detect_conditions_from_bids())

        # Empty/non-usable trial_type values -> empty conditions set branch
        (ev_dir / "sub-0001_task-t_run-01_events.tsv").write_text(
            "trial_type\tonset\nn/a\t0\nn/a\t1\n", encoding="utf-8"
        )
        self.assertIsNone(p._detect_conditions_from_bids())

        # filtered return branch
        (ev_dir / "sub-0001_task-t_run-01_events.tsv").write_text(
            "trial_type\tonset\nCueA\t0\nCueB\t1\n", encoding="utf-8"
        )
        self.assertEqual(p._detect_conditions_from_bids(), ["CueA", "CueB"])
