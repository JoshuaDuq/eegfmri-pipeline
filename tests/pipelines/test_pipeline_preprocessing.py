import importlib
import json
import subprocess
import sys
import tempfile
import types
import unittest

import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Derivative discovery is pure pathlib with no MNE dependency, so the stubs below hand
# back the real module rather than a fake: these tests assert on the layouts it matches.
_derivatives_module = importlib.import_module("eeg_pipeline.preprocessing.derivatives")


from tests.utils.pipelines_test_utils import DotConfig, DummyProgress, NoopBatchProgress, NoopProgress

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
        # The dataset declaration, stubbed with the same rule the real helper applies so
        # the step-selection tests below keep exercising their own configs rather than a
        # constant. See tests/pipelines/test_eeg_only_gating.py for the real thing.
        "eeg_pipeline.utils.config.acquisition": _make_module(
            "eeg_pipeline.utils.config.acquisition",
            is_eeg_fmri=lambda config: bool(
                _get_config_value(config, "preprocessing.eeg_fmri", None)
                if _get_config_value(config, "preprocessing.eeg_fmri", None) is not None
                else _get_config_value(config, "preprocessing.brainvision_analyzer.enabled", False)
            ),
        ),
        # These tests build minimal configs to exercise one branch each, so a whole-config
        # coherence check has nothing meaningful to say about them. It is covered on real
        # configs in tests/config/test_config_coherence.py.
        "eeg_pipeline.utils.config.coherence": _make_module(
            "eeg_pipeline.utils.config.coherence",
            check_config_coherence=lambda config: _NoopCoherenceReport(),
        ),
        "eeg_pipeline.preprocessing": _make_package("eeg_pipeline.preprocessing"),
        "eeg_pipeline.preprocessing.pipeline": _make_package("eeg_pipeline.preprocessing.pipeline"),
        "eeg_pipeline.preprocessing.derivatives": _derivatives_module,
    }


class _NoopCoherenceReport:
    errors = ()
    warnings = ()

    def log_warnings(self, logger):
        return None

    def raise_if_errors(self):
        return None


class _PreprocessingImportMixin:
    def setUp(self):
        patcher = patch.dict(sys.modules, _preprocessing_import_stubs())
        patcher.start()
        self.addCleanup(patcher.stop)
        # The stubs above only reach the pipeline module if it is imported *after* they
        # are installed: it binds resolve_eeg_bids_root and the rest by name at its own
        # import time. Left in sys.modules by any test file that ran earlier, it keeps
        # the real functions and every stub here is inert -- so these tests passed or
        # failed according to collection order, and a single new file in this directory
        # was enough to flip six of them. Evicting the entry makes the in-test import
        # re-execute against the stubs; patch.dict restores it on cleanup.
        sys.modules.pop("eeg_pipeline.pipelines.preprocessing", None)


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
        self.assertIn("preprocessing/_06a2_find_ica_artifacts", p._get_ica_fitting_steps())

        resolved = p._extract_preprocessing_params(
            "task",
            {
                "mode": "ica",
                "use_pyprep": False,
                "task_is_rest": False,
                "n_jobs": 4,
                "progress": _NoopProgress(),
            },
        )
        self.assertEqual(resolved[0], "task")
        self.assertEqual(resolved[1], "ica")
        self.assertFalse(resolved[2])
        self.assertFalse(resolved[3])
        self.assertEqual(resolved[4], 4)

        with self.assertRaisesRegex(ValueError, "Unknown preprocessing mode"):
            p._get_steps_for_mode("bogus")






    def test_report_review_sections_do_not_depend_on_the_analyzer_steps(self):
        """An EEG-only run must still get provenance, spectra, coverage, and continuity.

        These sections describe what preprocessing produced. Gating them on an
        Analyzer-only step made every one of them unreachable without a scanner.
        """
        from eeg_pipeline.pipelines.preprocessing import (
            STEP_EPOCHS,
            STEP_STATS,
            PreprocessingPipeline,
        )

        pipeline = object.__new__(PreprocessingPipeline)
        pipeline.logger = Mock()
        pipeline._run_epoch_creation = Mock()
        pipeline._collect_stats = Mock()
        pipeline._append_report_review_sections = Mock()

        pipeline._execute_steps(
            steps=[STEP_EPOCHS, STEP_STATS],
            subjects=["0001"],
            task="rest",
            use_pyprep=True,
            task_is_rest=True,
            n_jobs=1,
            progress=_NoopProgress(),
        )

        pipeline._append_report_review_sections.assert_called_once_with(
            subjects=["0001"], task="rest"
        )

    def test_report_review_sections_are_skipped_without_ica_or_epochs(self):
        """Bad-channel detection alone produces no report to append to."""
        from eeg_pipeline.pipelines.preprocessing import STEP_BAD_CHANNELS, PreprocessingPipeline

        pipeline = object.__new__(PreprocessingPipeline)
        pipeline.logger = Mock()
        pipeline._run_bad_channel_detection = Mock()
        pipeline._append_report_review_sections = Mock()

        pipeline._execute_steps(
            steps=[STEP_BAD_CHANNELS],
            subjects=["0001"],
            task="rest",
            use_pyprep=True,
            task_is_rest=True,
            n_jobs=1,
            progress=_NoopProgress(),
        )

        pipeline._append_report_review_sections.assert_not_called()


    def test_detect_conditions_from_bids(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        tmp = Path(tempfile.mkdtemp())
        ev_dir = tmp / "sub-0001" / "eeg"
        ev_dir.mkdir(parents=True, exist_ok=True)
        (ev_dir / "._sub-0001_task-task_run-01_events.tsv").write_bytes(
            b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X metadata"
        )
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
                "ica": {
                    "algorithm": "extended_infomax",
                    "n_components": None,
                    "l_freq": 1.0,
                    "h_freq": 100.0,
                    "reject": "autoreject_local",
                    "use_icalabel": True,
                    "use_ecg_detection": False,
                    "use_eog_detection": False,
                    "labels_to_keep": ["brain", "other"],
                    "probability_threshold": 0.8,
                    "process_raw_clean": True,
                },
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
        self.assertNotIn("\nreject =", cfg)
        self.assertIn("ica_n_components = None", cfg)
        self.assertIn("ica_reject = 'autoreject_local'", cfg)
        self.assertIn("ica_h_freq = 100.0", cfg)
        self.assertIn("ica_use_icalabel = True", cfg)
        self.assertIn("ica_use_ecg_detection = False", cfg)
        self.assertIn("ica_use_eog_detection = False", cfg)
        self.assertIn("process_raw_clean = True", cfg)
        self.assertIn("ica_icalabel_include = ('brain', 'other')", cfg)
        self.assertIn("'heart beat': 0.8", cfg)

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
                "pyprep": {"bad_channel_sync_policy": "per_run"},
            }
        )

        self.assertEqual(p._resolve_pipeline_deriv_root(), Path("/tmp/deriv-task"))

        progress = Mock()
        with (
            patch.object(
                PreprocessingPipeline,
                "_extract_preprocessing_params",
                return_value=("task", "epochs", True, False, 1, progress),
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
            p._run_mne_bids_pipeline("init", subjects=["0001"], task="pain", n_jobs=1)
        self.assertEqual(mock_generate.call_args.kwargs["task"], "pain")

        with (
            patch.object(PreprocessingPipeline, "_generate_mne_bids_config", return_value="x=1"),
            patch(
                "eeg_pipeline.pipelines.preprocessing.subprocess.run",
                return_value=SimpleNamespace(returncode=1, stdout="", stderr="boom"),
            ),
        ):
            with self.assertRaises(RuntimeError):
                p._run_mne_bids_pipeline("init", subjects=["0001"], task="pain", n_jobs=1)

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
                p._run_mne_bids_pipeline("init", subjects=["0001"], task="pain", n_jobs=1)

        error_message = str(exc.exception)
        self.assertIn(stdout, error_message)
        self.assertIn(stderr, error_message)

    def test_run_bad_channel_detection(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        tmp = Path(tempfile.mkdtemp())
        p.bids_root = tmp / "bids"
        p.deriv_root = tmp / "deriv"
        p.logger = Mock()
        p.config = DotConfig(
            {
                "pyprep": {"bad_channel_sync_policy": "per_run"},
                "eeg": {"montage": "easycap-M1"},
            }
        )

        mock_preproc = types.SimpleNamespace(
            run_bads_detection=Mock(),
            synchronize_bad_channels_across_runs=Mock(),
        )
        with patch.dict(
            sys.modules,
            {
                "eeg_pipeline.preprocessing.pipeline.preprocess": mock_preproc,
            },
        ):
            p._run_bad_channel_detection(["0001"], "task", n_jobs=2)

        self.assertTrue(mock_preproc.run_bads_detection.called)

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

        preload_calls = []

        def fake_read_raw_fif(path, preload, verbose):
            preload_calls.append(preload)
            self.assertFalse(verbose)
            return raws[str(path)]

        fake_mne = types.SimpleNamespace(io=types.SimpleNamespace(read_raw_fif=fake_read_raw_fif))

        with patch.dict(sys.modules, {"mne": fake_mne}):
            p._harmonize_filtered_raw_bads_for_mne_concat(["0001"], "pain")

        # Headers decide whether a rewrite is needed; only the runs being rewritten are
        # then read with their samples.
        self.assertEqual(preload_calls, [False, False, True, True])
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
        preload_calls = []

        def fake_read_raw_fif(path, preload, verbose):
            preload_calls.append(preload)
            self.assertFalse(verbose)
            return raws[str(path)]

        fake_mne = types.SimpleNamespace(io=types.SimpleNamespace(read_raw_fif=fake_read_raw_fif))

        with patch.dict(sys.modules, {"mne": fake_mne}):
            with self.assertRaisesRegex(ValueError, "subject_union"):
                p._harmonize_filtered_raw_bads_for_mne_concat(["0001"], "pain")

        self.assertEqual(run_1.read_text(encoding="utf-8"), "raw-1")
        self.assertEqual(run_2.read_text(encoding="utf-8"), "raw-2")
        # Deciding that the runs disagree needs headers only; a run that is never
        # rewritten must never have its samples read.
        self.assertEqual(preload_calls, [False, False])

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
    def _make_direct_entry_pipeline(self, manifest_path):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.name = "preprocessing"
        p.config = DotConfig(
            {
                "paths": {"decomb_manifest": manifest_path},
                "project": {"task": "task"},
                "preprocessing": {"notch_freq": None},
                "pyprep": {"bad_channel_sync_policy": "per_run"},
            }
        )
        p.logger = Mock()
        p.deriv_root = Path(tempfile.mkdtemp())
        return p

    def _process_subject_entry(self, pipeline):
        return pipeline.process_subject(
            "0001",
            task="task",
            mode="epochs",
            progress=_NoopProgress(),
        )

    def _run_batch_entry(self, pipeline):
        return pipeline.run_batch(
            subjects=["0001"],
            task="task",
            mode="epochs",
            progress=_NoopProgress(),
        )

    def test_configured_decomb_manifest_is_validated_by_direct_entry_points(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        entry_points = (self._process_subject_entry, self._run_batch_entry)
        for entry_point in entry_points:
            with self.subTest(entry_point=entry_point.__name__):
                loader = Mock(return_value=object())
                adapter = _make_module(
                    "eeg_pipeline.spectral_availability.decomb",
                    load_decomb_manifest=loader,
                )
                pipeline = self._make_direct_entry_pipeline("/tmp/manifest.tsv")
                with (
                    patch.dict(
                        sys.modules,
                        {"eeg_pipeline.spectral_availability.decomb": adapter},
                    ),
                    patch.object(
                        PreprocessingPipeline,
                        "_get_steps_for_run",
                        return_value=[],
                    ),
                    patch.object(
                        PreprocessingPipeline,
                        "_execute_steps",
                        return_value={},
                    ) as execute,
                ):
                    entry_point(pipeline)

                loader.assert_called_once_with("/tmp/manifest.tsv")
                execute.assert_called_once()

    def test_invalid_decomb_manifest_surfaces_before_direct_execution(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        entry_points = (self._process_subject_entry, self._run_batch_entry)
        for entry_point in entry_points:
            with self.subTest(entry_point=entry_point.__name__):
                loader = Mock(side_effect=ValueError("invalid Decomb manifest"))
                adapter = _make_module(
                    "eeg_pipeline.spectral_availability.decomb",
                    load_decomb_manifest=loader,
                )
                pipeline = self._make_direct_entry_pipeline("/tmp/manifest.tsv")
                with (
                    patch.dict(
                        sys.modules,
                        {"eeg_pipeline.spectral_availability.decomb": adapter},
                    ),
                    patch.object(
                        PreprocessingPipeline,
                        "_get_steps_for_run",
                        return_value=[],
                    ),
                    patch.object(
                        PreprocessingPipeline,
                        "_execute_steps",
                        return_value={},
                    ) as execute,
                    self.assertRaisesRegex(ValueError, "invalid Decomb manifest"),
                ):
                    entry_point(pipeline)

                execute.assert_not_called()

    def test_null_decomb_manifest_preserves_direct_entry_paths(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        entry_points = (self._process_subject_entry, self._run_batch_entry)
        for entry_point in entry_points:
            with self.subTest(entry_point=entry_point.__name__):
                loader = Mock(side_effect=AssertionError("adapter must remain inactive"))
                adapter = _make_module(
                    "eeg_pipeline.spectral_availability.decomb",
                    load_decomb_manifest=loader,
                )
                pipeline = self._make_direct_entry_pipeline(None)
                with (
                    patch.dict(
                        sys.modules,
                        {"eeg_pipeline.spectral_availability.decomb": adapter},
                    ),
                    patch.object(
                        PreprocessingPipeline,
                        "_get_steps_for_run",
                        return_value=[],
                    ),
                    patch.object(
                        PreprocessingPipeline,
                        "_execute_steps",
                        return_value={},
                    ) as execute,
                ):
                    entry_point(pipeline)

                loader.assert_not_called()
                execute.assert_called_once()

    def test_null_decomb_validation_keeps_adapter_unimported_in_fresh_process(self):
        command = """
import sys

from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline


class Config:
    def get(self, key, default=None):
        if key == "paths.decomb_manifest":
            return None
        return default


pipeline = object.__new__(PreprocessingPipeline)
pipeline.config = Config()
pipeline._validate_decomb_manifest()
assert "eeg_pipeline.spectral_availability.decomb" not in sys.modules
"""

        subprocess.run(
            [sys.executable, "-c", command],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
        )

    def test_preprocessing_init_and_ica_helpers(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        cfg = DotConfig({"bids_root": "/tmp/bids"})
        with patch(
            "eeg_pipeline.pipelines.preprocessing.PipelineBase.__init__",
            lambda self, name, config=None: setattr(self, "config", config or cfg),
        ):
            p = PreprocessingPipeline(config=cfg)
        self.assertEqual(p.bids_root.as_posix(), "/tmp/bids")
        self.assertIn("preprocessing/_06a2_find_ica_artifacts", p._get_ica_fitting_steps())

        p.logger = Mock()
        with (
            patch.object(PreprocessingPipeline, "_run_mne_bids_pipeline") as mock_run,
            patch.object(
                PreprocessingPipeline,
                "_harmonize_filtered_raw_bads_for_mne_concat",
            ) as mock_harmonize,
        ):
            p._run_ica_fitting(["0001"], "t", n_jobs=1)
        self.assertEqual(mock_run.call_count, 2)
        self.assertEqual(
            mock_run.call_args_list[0].args[0],
            "init,preprocessing/_01_data_quality,preprocessing/_04_frequency_filter,"
            "preprocessing/_05_regress_artifact",
        )
        self.assertEqual(
            mock_run.call_args_list[1].args[0],
            "preprocessing/_06a1_fit_ica,preprocessing/_06a2_find_ica_artifacts",
        )
        mock_harmonize.assert_called_once_with(["0001"], "t")

    def test_ica_fitting_runs_band_report_when_enabled(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.logger = Mock()
        p.config = DotConfig({"ica": {"band_specific_report": {"enabled": True}}})

        with (
            patch.object(PreprocessingPipeline, "_run_mne_bids_pipeline"),
            patch.object(PreprocessingPipeline, "_harmonize_filtered_raw_bads_for_mne_concat"),
            patch.object(PreprocessingPipeline, "_run_band_specific_ica_report") as report,
        ):
            p._run_ica_fitting(["0001"], "pain", n_jobs=1)

        report.assert_called_once_with(subjects=["0001"], task="pain")

    def test_ica_fitting_skips_band_report_when_disabled(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.logger = Mock()
        p.config = DotConfig({"ica": {"band_specific_report": {"enabled": False}}})

        with (
            patch.object(PreprocessingPipeline, "_run_mne_bids_pipeline"),
            patch.object(PreprocessingPipeline, "_harmonize_filtered_raw_bads_for_mne_concat"),
            patch.object(PreprocessingPipeline, "_run_band_specific_ica_report") as report,
        ):
            p._run_ica_fitting(["0001"], "pain", n_jobs=1)

        report.assert_not_called()

    def test_ica_fitting_runs_cardiac_review_only_when_enabled(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.logger = Mock()
        p.config = DotConfig(
            {
                # The review measures the ballistocardiogram against a recorded ECG, so it
                # is gated on the dataset being EEG-fMRI as well as on being enabled.
                # tests/pipelines/test_eeg_only_gating.py covers the out-of-scanner case.
                "preprocessing": {"eeg_fmri": True},
                "ica": {
                    "cardiac_review": {"enabled": True},
                    "band_specific_report": {"enabled": False},
                },
            }
        )

        with (
            patch.object(PreprocessingPipeline, "_run_mne_bids_pipeline"),
            patch.object(PreprocessingPipeline, "_harmonize_filtered_raw_bads_for_mne_concat"),
            patch.object(PreprocessingPipeline, "_run_ica_cardiac_review", create=True) as review,
            patch.object(PreprocessingPipeline, "_run_ica_ocular_review", create=True),
        ):
            p._run_ica_fitting(["0001"], "pain", n_jobs=1)

        review.assert_called_once_with(subjects=["0001"], task="pain")

        p.config = DotConfig(
            {
                "preprocessing": {"eeg_fmri": True},
                "ica": {
                    "cardiac_review": {"enabled": False},
                    "band_specific_report": {"enabled": False},
                },
            }
        )
        with (
            patch.object(PreprocessingPipeline, "_run_mne_bids_pipeline"),
            patch.object(PreprocessingPipeline, "_harmonize_filtered_raw_bads_for_mne_concat"),
            patch.object(
                PreprocessingPipeline,
                "_run_ica_cardiac_review",
                create=True,
            ) as disabled_review,
            patch.object(
                PreprocessingPipeline,
                "_run_ica_ocular_review",
                create=True,
            ) as disabled_ocular_review,
        ):
            p._run_ica_fitting(["0001"], "pain", n_jobs=1)

        disabled_review.assert_not_called()
        disabled_ocular_review.assert_not_called()

    def test_cardiac_review_uses_filtered_runs_and_standard_ica(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.config = DotConfig(
            {
                "ica": {
                    "cardiac_review": {
                        "enabled": True,
                        "ecg_channel": "ECG",
                    }
                }
            }
        )
        eeg_directory = Path("/derivatives/sub-0001/eeg")
        epochs_path = eeg_directory / "sub-0001_proc-icafit_epo.fif"
        report_path = eeg_directory / "sub-0001_report.h5"
        filtered_path = eeg_directory / "sub-0001_task-pain_run-1_proc-filt_raw.fif"
        settings = object()
        settings_class = Mock()
        settings_class.from_mapping.return_value = settings
        generate = Mock()
        cardiac_modules = {
            "eeg_pipeline.preprocessing.ica_cardiac_report": _make_module(
                "eeg_pipeline.preprocessing.ica_cardiac_report",
                generate_ica_cardiac_review=generate,
            ),
            "eeg_pipeline.preprocessing.ica_ocular_report": _make_module(
                "eeg_pipeline.preprocessing.ica_ocular_report",
                generate_ica_ocular_review=generate,
            ),
            "eeg_pipeline.preprocessing.ica_cardiac_review": _make_module(
                "eeg_pipeline.preprocessing.ica_cardiac_review",
                CardiacReviewSettings=settings_class,
            ),
        }

        with (
            patch.dict(sys.modules, cardiac_modules),
            patch.object(
                PreprocessingPipeline,
                "_resolve_bad_harmonization_subjects",
                return_value=["0001"],
            ),
            patch.object(
                PreprocessingPipeline,
                "_find_filtered_raw_run_files",
                return_value=[filtered_path],
            ),
            patch.object(
                PreprocessingPipeline,
                "_find_band_ica_report_inputs",
                return_value=[(epochs_path, report_path, "sub-0001")],
            ),
        ):
            p._run_ica_cardiac_review(subjects=["0001"], task="pain")

        generate.assert_called_once()
        arguments = generate.call_args.kwargs
        assert arguments["filtered_raw_paths"] == [filtered_path]
        assert arguments["ica_path"] == eeg_directory / "sub-0001_proc-ica_ica.fif"
        assert arguments["report_path"] == report_path
        assert arguments["output_path"] == (eeg_directory / "sub-0001_desc-icaecg_components.tsv")
        assert arguments["settings"] is settings
        settings_class.from_mapping.assert_called_once_with({"enabled": True, "ecg_channel": "ECG"})

    def test_band_report_input_discovery_preserves_session_entities(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.deriv_root = Path(tempfile.mkdtemp())
        eeg_dir = p.deriv_root / "preprocessed" / "eeg" / "sub-0001" / "ses-02" / "eeg"
        eeg_dir.mkdir(parents=True)
        epochs_path = eeg_dir / "sub-0001_ses-02_proc-icafit_epo.fif"
        report_path = eeg_dir / "sub-0001_ses-02_report.h5"
        epochs_path.write_text("epochs", encoding="utf-8")
        report_path.write_text("report", encoding="utf-8")

        assert p._find_band_ica_report_inputs("0001") == [
            (epochs_path, report_path, "sub-0001_ses-02")
        ]

    def test_provisional_component_review_uses_standard_ica_paths(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.deriv_root = Path(tempfile.mkdtemp())
        p.bids_root = Path(tempfile.mkdtemp())
        p.logger = Mock()
        p.config = DotConfig(
            {
                "ica": {
                    "band_specific_report": {
                        "comparisons": [
                            {
                                "name": "pain",
                                "column": "pain_binary_coded",
                                "group_a": {"label": "Painful", "values": [1]},
                                "group_b": {"label": "Non-painful", "values": [0]},
                            }
                        ]
                    }
                }
            }
        )
        eeg_dir = p.deriv_root / "preprocessed" / "eeg" / "sub-0001" / "eeg"
        eeg_dir.mkdir(parents=True)
        task_epochs_path = eeg_dir / "sub-0001_task-pain_epo.fif"
        task_epochs_path.write_text("epochs", encoding="utf-8")
        aligned_events_path = eeg_dir / "sub-0001_task-pain_events.tsv"
        append_report = Mock()
        settings_class = SimpleNamespace(from_mapping=Mock(return_value=object()))
        band_report_module = _make_module(
            "eeg_pipeline.preprocessing.band_ica_report",
            BandIcaReportSettings=settings_class,
            append_condition_tfr_report=append_report,
        )
        data_module = _make_module(
            "eeg_pipeline.utils.data.preprocessing",
            write_clean_events_tsv_for_epochs=Mock(return_value=aligned_events_path),
        )

        with (
            patch.object(
                PreprocessingPipeline,
                "_resolve_bad_harmonization_subjects",
                return_value=["0001"],
            ),
            patch.object(PreprocessingPipeline, "_resolve_epoch_conditions", return_value=None),
            patch.dict(
                sys.modules,
                {
                    "eeg_pipeline.preprocessing.band_ica_report": band_report_module,
                    "eeg_pipeline.utils.data": _make_package("eeg_pipeline.utils.data"),
                    "eeg_pipeline.utils.data.preprocessing": data_module,
                },
            ),
        ):
            p._append_provisional_band_ica_condition_tfrs(subjects=["0001"], task="pain")

        kwargs = append_report.call_args.kwargs
        assert kwargs["ica_fit_epochs_path"] == eeg_dir / "sub-0001_proc-icafit_epo.fif"
        assert kwargs["standard_ica_path"] == eeg_dir / "sub-0001_proc-ica_ica.fif"
        assert "output_dir" not in kwargs
        assert "output_prefix" not in kwargs

    def test_preservation_measurements_reach_the_landing_panel(self):
        """Every other headline describes removal; this is the one that describes survival.

        Without it the panel opens with "93.9% of sensor variance removed" and offers the
        reader nothing to weigh that against, which is the exact asymmetry the preservation
        section was written to close.
        """
        from eeg_pipeline.pipelines.preprocessing import _preservation_measurements

        measurements = _preservation_measurements(
            reliability=SimpleNamespace(
                correlation=0.71,
                corrected_correlation=0.83,
                n_trials=80,
                response_window_s=(0.1, 0.6),
            ),
            alpha=SimpleNamespace(
                prominence_db=6.4,
                # The prominence is a max over the band and is positive on noise,
                # so what it cleared travels with it.
                is_resolvable=lambda: True,
                resolvable_bar_db=3.35,
            ),
        )

        assert measurements == {
            # The measurement under the key named after it, and the step-up under its own.
            # These were one key holding the stepped-up value, so the record, the sidecar
            # and the cohort's reliability row all reported a corrected number as "r".
            "split_half_r": 0.71,
            "split_half_r_corrected": 0.83,
            "alpha_prominence_db": 6.4,
            # The bar that prominence had to clear, and whether it did. The prominence is
            # the largest excess over the fitted background anywhere in the band, so it is
            # positive on a spectrum carrying no rhythm at all.
            "alpha_peak_resolvable": True,
            "alpha_resolvable_bar_db": 3.35,
            # Reliability grows with test length, so the trial count and the window it was
            # measured over travel with the correlation: a cohort cannot compare two
            # participants' reliabilities without stepping both to a common length.
            "split_half_n_trials": 80,
            "split_half_window_start_s": 0.1,
            "split_half_window_end_s": 0.6,
        }

    def test_preservation_measurements_omit_what_the_paradigm_cannot_support(self):
        """Rest has no evoked response to split, so there is no reliability to report."""
        from eeg_pipeline.pipelines.preprocessing import _preservation_measurements

        measurements = _preservation_measurements(
            reliability=None,
            alpha=SimpleNamespace(
                prominence_db=6.4,
                # The prominence is a max over the band and is positive on noise,
                # so what it cleared travels with it.
                is_resolvable=lambda: True,
                resolvable_bar_db=3.35,
            ),
        )

        # The rhythm and what it had to clear; nothing about an evoked response, which
        # rest cannot supply.
        assert measurements == {
            "alpha_prominence_db": 6.4,
            "alpha_peak_resolvable": True,
            "alpha_resolvable_bar_db": 3.35,
        }
        assert _preservation_measurements(reliability=None, alpha=None) == {}

    def test_the_review_stage_records_its_headline_numbers_for_the_landing_panel(self):
        """The landing panel reads the build record, so the stage has to write to it.

        These four are measured by the review stage and by nothing else, so if it records
        nothing the panel simply has no row for them and the reader is back to hunting
        through sections for the run count.
        """
        from eeg_pipeline.pipelines.preprocessing import _review_stage_measurements

        coverage = SimpleNamespace(n_channels=63, bad_channels=("TP9", "T7"), n_runs=6)
        evidence = SimpleNamespace(
            spectra=[object()] * 6,
            marker_agreements=[
                SimpleNamespace(matched_fraction=0.93, median_lag_s=0.004, lag_iqr_s=0.002),
                # The run the landing panel reports: a share of zero with a tight lag,
                # which is a detector locking onto the magnetohydrodynamic deflection
                # rather than markers in the wrong place.
                SimpleNamespace(matched_fraction=0.0, median_lag_s=0.303, lag_iqr_s=0.019),
                SimpleNamespace(matched_fraction=None, median_lag_s=None, lag_iqr_s=None),
            ],
        )

        measurements = _review_stage_measurements(coverage=coverage, evidence=evidence)

        assert measurements["n_runs"] == 6
        assert measurements["n_channels"] == 63
        assert measurements["n_bad_channels"] == 2
        # The worst run is the one worth meeting first, and an undefined one is not a zero.
        assert measurements["worst_marker_agreement"] == 0.0

    def test_review_measurements_omit_what_this_dataset_has_no_stage_for(self):
        """An EEG-only dataset has no marker agreement, and a key absent is a row absent."""
        from eeg_pipeline.pipelines.preprocessing import _review_stage_measurements

        measurements = _review_stage_measurements(
            coverage=SimpleNamespace(n_channels=32, bad_channels=(), n_runs=1),
            evidence=SimpleNamespace(
                spectra=[object()], marker_agreements=[], cardiac_residuals=[]
            ),
        )

        assert "worst_marker_agreement" not in measurements
        assert measurements["n_bad_channels"] == 0

    def test_review_measurements_survive_a_stage_that_produced_nothing(self):
        """Coverage and run evidence are both optional; the record must still be writable."""
        from eeg_pipeline.pipelines.preprocessing import _review_stage_measurements

        assert _review_stage_measurements(coverage=None, evidence=None) == {}

    def test_provisional_preservation_reads_the_pre_rejection_task_epochs(self):
        """Preservation is the only counterweight at ICA review, so it runs there too.

        At that point every other panel measures removal — 93.9% of sensor variance on
        sub-0015 — and the epochs that would support a survival measurement have not been
        cleaned yet. Reading the pre-rejection task epochs is what lets the panel exist
        before the exclusions are approved rather than only after.
        """
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.deriv_root = Path(tempfile.mkdtemp())
        p.logger = Mock()
        p.config = DotConfig({"preprocessing": {"task_is_rest": False}})
        eeg_dir = p.deriv_root / "preprocessed" / "eeg" / "sub-0001" / "eeg"
        eeg_dir.mkdir(parents=True)
        task_epochs_path = eeg_dir / "sub-0001_task-pain_epo.fif"
        task_epochs_path.write_text("epochs", encoding="utf-8")
        report_path = eeg_dir / "sub-0001_report.h5"
        epochs = object()
        settings = object()
        read_epochs = Mock(return_value=epochs)
        add_task = Mock(return_value=(None, None))
        preservation_module = _make_module(
            "eeg_pipeline.preprocessing.report.preservation",
            add_task_preservation_review=add_task,
            add_rest_preservation_review=Mock(return_value=None),
        )

        with (
            patch.dict(
                sys.modules,
                {"eeg_pipeline.preprocessing.report.preservation": preservation_module},
            ),
            patch("mne.read_epochs", read_epochs),
        ):
            p._append_provisional_signal_preservation(
                report="report",
                report_path=report_path,
                task="pain",
                subject="0001",
                settings=settings,
            )

        assert read_epochs.call_args.args[0] == task_epochs_path
        kwargs = add_task.call_args.kwargs
        assert kwargs["epochs"] is epochs
        assert kwargs["settings"] is settings
        assert "Provisional" in kwargs["analysis_status"]

    def test_provisional_preservation_is_skipped_without_pre_rejection_epochs(self):
        """A report built from the bad-channel stage alone has no epochs to measure."""
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.deriv_root = Path(tempfile.mkdtemp())
        p.logger = Mock()
        p.config = DotConfig({"preprocessing": {"task_is_rest": False}})
        eeg_dir = p.deriv_root / "preprocessed" / "eeg" / "sub-0001" / "eeg"
        eeg_dir.mkdir(parents=True)
        settings = object()
        add_task = Mock(return_value=(None, None))
        preservation_module = _make_module(
            "eeg_pipeline.preprocessing.report.preservation",
            add_task_preservation_review=add_task,
            add_rest_preservation_review=Mock(return_value=None),
        )

        with patch.dict(
            sys.modules,
            {"eeg_pipeline.preprocessing.report.preservation": preservation_module},
        ):
            p._append_provisional_signal_preservation(
                report="report",
                report_path=eeg_dir / "sub-0001_report.h5",
                task="pain",
                subject="0001",
                settings=settings,
            )

        add_task.assert_not_called()

    def test_final_component_review_uses_standard_ica_paths(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.deriv_root = Path(tempfile.mkdtemp())
        p.logger = Mock()
        p.config = DotConfig({"ica": {"band_specific_report": {"comparisons": []}}})
        eeg_dir = p.deriv_root / "preprocessed" / "eeg" / "sub-0001" / "eeg"
        eeg_dir.mkdir(parents=True)
        paths = {
            "clean": eeg_dir / "sub-0001_task-pain_proc-clean_epo.fif",
            "pre": eeg_dir / "sub-0001_task-pain_epo.fif",
            "events": eeg_dir / "sub-0001_task-pain_proc-clean_events.tsv",
            "report": eeg_dir / "sub-0001_report.h5",
            "fit": eeg_dir / "sub-0001_proc-icafit_epo.fif",
            "ica": eeg_dir / "sub-0001_proc-ica_ica.fif",
        }
        for path in paths.values():
            path.write_text("test", encoding="utf-8")
        append_report = Mock()
        settings_class = SimpleNamespace(from_mapping=Mock(return_value=object()))
        band_report_module = _make_module(
            "eeg_pipeline.preprocessing.band_ica_report",
            BandIcaReportSettings=settings_class,
            append_condition_tfr_report=append_report,
        )

        with (
            patch.object(
                PreprocessingPipeline,
                "_resolve_bad_harmonization_subjects",
                return_value=["0001"],
            ),
            patch.dict(
                sys.modules,
                {"eeg_pipeline.preprocessing.band_ica_report": band_report_module},
            ),
        ):
            p._append_band_ica_condition_tfrs(subjects=["0001"], task="pain")

        kwargs = append_report.call_args.kwargs
        assert kwargs["ica_fit_epochs_path"] == paths["fit"]
        assert kwargs["standard_ica_path"] == paths["ica"]
        assert "output_dir" not in kwargs
        assert "output_prefix" not in kwargs

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
                "pyprep": {"bad_channel_sync_policy": "per_run"},
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

        with patch.object(PreprocessingPipeline, "_execute_steps", return_value={}):
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
        p.config = DotConfig(
            {
                "project": {"task": "task"},
                "pyprep": {"bad_channel_sync_policy": "per_run"},
            }
        )
        p.logger = Mock()
        p.bids_root = Path(tempfile.mkdtemp()) / "bids"
        p.deriv_root = Path(tempfile.mkdtemp())

        qc_outputs = {
            "scanner_harmonic_comb_png": "/tmp/comb.png",
            "scanner_harmonic_comb_tsv": "/tmp/comb.tsv",
        }
        with patch.object(
            PreprocessingPipeline,
            "_execute_steps",
            return_value=qc_outputs,
        ):
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
        self.assertEqual(payload["outputs"], qc_outputs)

    def test_run_batch_preserves_primary_failure_when_metadata_write_also_fails(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.name = "preprocessing"
        p.config = DotConfig(
            {
                "project": {"task": "task"},
                "pyprep": {"bad_channel_sync_policy": "per_run"},
            }
        )
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
        p.config = DotConfig(
            {
                "project": {"task": "task"},
                "pyprep": {"bad_channel_sync_policy": "per_run"},
            }
        )
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
        p.config = DotConfig(
            {
                "project": {"task": "task"},
                "pyprep": {"bad_channel_sync_policy": "per_run"},
            }
        )
        p.logger = Mock()
        p.bids_root = Path(tempfile.mkdtemp()) / "bids"
        p.deriv_root = Path(tempfile.mkdtemp())

        with patch.object(PreprocessingPipeline, "_execute_steps", return_value={}):
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
        p.config = DotConfig(
            {
                "project": {"task": "task"},
                "pyprep": {"bad_channel_sync_policy": "per_run"},
            }
        )
        p.logger = Mock()

        fake_cli = types.SimpleNamespace(ProgressReporter=lambda enabled=False: _NoopProgress())
        with patch.dict(sys.modules, {"eeg_pipeline.cli.common": fake_cli}):
            task, mode, use_pyprep, task_is_rest, n_jobs, progress = (
                p._extract_preprocessing_params(None, {})
            )
        self.assertEqual(task, "task")
        self.assertEqual(mode, "ica")
        self.assertTrue(use_pyprep)
        self.assertFalse(task_is_rest)
        self.assertEqual(n_jobs, 1)
        self.assertIsNotNone(progress)

        p.config = DotConfig({})
        with self.assertRaisesRegex(ValueError, "Missing required config value: project.task"):
            p._extract_preprocessing_params(None, {})

        p.config = DotConfig({"preprocessing": {"task_is_rest": True}})
        with patch.dict(sys.modules, {"eeg_pipeline.cli.common": fake_cli}):
            task, mode, use_pyprep, task_is_rest, n_jobs, progress = (
                p._extract_preprocessing_params(None, {})
            )
        self.assertIsNone(task)
        self.assertEqual(mode, "ica")
        self.assertTrue(use_pyprep)
        self.assertTrue(task_is_rest)
        self.assertEqual(n_jobs, 1)
        self.assertIsNotNone(progress)

        with (
            patch.object(
                PreprocessingPipeline,
                "_extract_preprocessing_params",
                return_value=("task", "full", True, False, 1, _NoopProgress()),
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
        p.config = DotConfig(
            {
                "project": {"task": "task"},
                "pyprep": {"bad_channel_sync_policy": "per_run"},
            }
        )
        p.logger = Mock()
        p._refresh_processing_roots_if_initialized = Mock()

        progress = _TrackingProgress()
        with (
            patch.object(
                PreprocessingPipeline,
                "_extract_preprocessing_params",
                return_value=("task", "full", True, False, 1, progress),
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
            patch.object(PreprocessingPipeline, "_run_epoch_creation") as m4,
            patch.object(PreprocessingPipeline, "_collect_stats") as m5,
            patch.object(PreprocessingPipeline, "_append_report_review_sections"),
        ):
            p._execute_steps(
                ["bad-channels", "ica-fit", "epochs", "stats"],
                ["0001"],
                "t",
                True,
                False,
                1,
                _NoopProgress(),
            )

        self.assertTrue(m1.called and m2.called and m4.called and m5.called)

        with patch.object(PreprocessingPipeline, "_run_bad_channel_detection") as m1:
            p._execute_steps(["bad-channels"], ["0001"], "t", False, False, 1, _NoopProgress())
        m1.assert_not_called()


    def test_run_epoch_creation_and_collect_stats(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.logger = Mock()
        p.config = DotConfig({"preprocessing": {"write_clean_events": True}})
        p.bids_root = Path(tempfile.mkdtemp())
        p.deriv_root = Path(tempfile.mkdtemp())

        # Appending the review sections is a separate concern with its own tests, and
        # reaching it here would depend on the report package escaping this class's
        # module stubs.
        with (
            patch.object(PreprocessingPipeline, "_run_mne_bids_pipeline") as run_mne,
            patch.object(PreprocessingPipeline, "_write_clean_events_tsv") as write_clean,
            patch.object(PreprocessingPipeline, "_append_epoch_rejection_review"),
        ):
            p._run_epoch_creation(["0001"], "t", task_is_rest=False, n_jobs=1)
        run_mne.assert_called_once()
        write_clean.assert_called_once()

        with (
            patch.object(PreprocessingPipeline, "_run_mne_bids_pipeline") as run_mne,
            patch.object(PreprocessingPipeline, "_write_clean_events_tsv") as write_clean,
            patch.object(PreprocessingPipeline, "_append_epoch_rejection_review"),
        ):
            p._run_epoch_creation(["0001"], "t", task_is_rest=True, n_jobs=1)
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


class TestPreprocessingStepSelection(_PreprocessingImportMixin, unittest.TestCase):
    def _pipeline(self, cardiac_review_enabled=True):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.logger = Mock()
        p.config = DotConfig(
            {
                "ica": {"cardiac_review": {"enabled": cardiac_review_enabled}},
                "pyprep": {"bad_channel_sync_policy": "subject_union"},
            }
        )
        return p

    def test_full_mode_runs_the_same_qc_as_ica_then_epochs(self):
        """`full` does both jobs, so it must not silently skip either job's QC."""
        p = self._pipeline()

        full = p._get_steps_for_run("full", task_is_rest=False)
        split = p._get_steps_for_run("ica", task_is_rest=False) + p._get_steps_for_run(
            "epochs", task_is_rest=False
        )

        # One QC step survives the scanner relocation, and it is attached to ICA fitting.
        self.assertIn("ica-cardiac-qc", full, "full mode dropped ica-cardiac-qc")
        self.assertIn("ica-cardiac-qc", split)

    def test_bad_channel_mode_gets_no_ica_or_epoch_qc(self):
        """QC attaches to the derivative it measures, so a mode producing neither gets none."""
        p = self._pipeline()

        steps = p._get_steps_for_run("bad-channels", task_is_rest=False)

        self.assertNotIn("ica-cardiac-qc", steps)

    def test_the_cardiac_qc_step_needs_the_cardiac_review_switch(self):
        """The marker-CTPS QC reads a beat train, and the cardiac review is what resolves
        one. This gate used to be the Analyzer declaration, which no longer exists."""
        p = self._pipeline(cardiac_review_enabled=False)

        steps = p._get_steps_for_run("full", task_is_rest=False)

        self.assertNotIn("ica-cardiac-qc", steps)




    def test_per_run_policy_fails_before_any_recording_is_opened(self):
        """The mismatch is visible in channels.tsv, so it must not wait for filtering."""
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.logger = Mock()
        p.config = DotConfig(
            {
                "preprocessing": {"brainvision_analyzer": {"enabled": False}},
                "pyprep": {"bad_channel_sync_policy": "per_run"},
            }
        )
        p.bids_root = Path(tempfile.mkdtemp())
        eeg_dir = p.bids_root / "sub-0001" / "eeg"
        eeg_dir.mkdir(parents=True)
        header = "name\ttype\tstatus\n"
        (eeg_dir / "sub-0001_task-pain_run-1_channels.tsv").write_text(
            header + "C3\teeg\tbad\nC4\teeg\tgood\n", encoding="utf-8"
        )
        (eeg_dir / "sub-0001_task-pain_run-2_channels.tsv").write_text(
            header + "C3\teeg\tgood\nC4\teeg\tbad\n", encoding="utf-8"
        )

        with self.assertRaisesRegex(ValueError, "subject_union"):
            p._get_steps_for_run("ica", False, ["0001"], "pain")

    def test_agreeing_runs_are_accepted_under_per_run(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.logger = Mock()
        p.config = DotConfig(
            {
                "preprocessing": {"brainvision_analyzer": {"enabled": False}},
                "pyprep": {"bad_channel_sync_policy": "per_run"},
            }
        )
        p.bids_root = Path(tempfile.mkdtemp())
        eeg_dir = p.bids_root / "sub-0001" / "eeg"
        eeg_dir.mkdir(parents=True)
        header = "name\ttype\tstatus\n"
        for run in (1, 2):
            (eeg_dir / f"sub-0001_task-pain_run-{run}_channels.tsv").write_text(
                # The ECG row differs, but non-EEG status is not the pipeline's concern.
                header
                + f"C3\teeg\tbad\nC4\teeg\tgood\nECG\tecg\t{'bad' if run == 1 else 'good'}\n",
                encoding="utf-8",
            )

        self.assertEqual(p._get_steps_for_run("ica", False, ["0001"], "pain"), ["ica-fit"])


class TestPreprocessingParallelism(_PreprocessingImportMixin, unittest.TestCase):
    """``n_jobs`` must reach the MNE-BIDS-Pipeline subprocess, not just PyPREP.

    MNE-BIDS-Pipeline defaults to ``n_jobs = 1``, so a config that omits the option runs
    every subject, and every autoreject threshold search inside ``_06a1_fit_ica`` and
    ``_09_ptp_reject``, on one core. The omission is invisible: the run still succeeds,
    just serially. These tests pin the value end to end so it cannot be dropped again.
    """

    def _pipeline(self, config=None):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = object.__new__(PreprocessingPipeline)
        p.logger = Mock()
        p.bids_root = Path("/tmp/bids")
        p.deriv_root = Path("/tmp/deriv")
        p.config = DotConfig(config or {})
        return p

    def test_execute_steps_forwards_n_jobs_to_the_mne_bids_steps(self):
        from eeg_pipeline.pipelines.preprocessing import (
            STEP_EPOCHS,
            STEP_ICA_FIT,
            PreprocessingPipeline,
        )

        pipeline = object.__new__(PreprocessingPipeline)
        pipeline.logger = Mock()
        pipeline._run_ica_fitting = Mock()
        pipeline._run_epoch_creation = Mock()
        pipeline._append_report_review_sections = Mock()

        pipeline._execute_steps(
            steps=[STEP_ICA_FIT, STEP_EPOCHS],
            subjects=["0001"],
            task="pain",
            use_pyprep=True,
            task_is_rest=False,
            n_jobs=6,
            progress=_NoopProgress(),
        )

        self.assertEqual(pipeline._run_ica_fitting.call_args.kwargs["n_jobs"], 6)
        self.assertEqual(pipeline._run_epoch_creation.call_args.kwargs["n_jobs"], 6)

    def test_ica_fitting_forwards_n_jobs_to_every_invocation(self):
        """ICA fitting drives MNE-BIDS more than once; one unset call is a serial step."""
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = self._pipeline({"ica": {}})

        with (
            patch.object(PreprocessingPipeline, "_run_mne_bids_pipeline") as run_mne,
            patch.object(PreprocessingPipeline, "_harmonize_filtered_raw_bads_for_mne_concat"),
        ):
            p._run_ica_fitting(["0001"], "pain", n_jobs=6)

        self.assertEqual(run_mne.call_count, 2)
        for call in run_mne.call_args_list:
            self.assertEqual(call.kwargs["n_jobs"], 6)

    def test_epoch_creation_forwards_n_jobs(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = self._pipeline({"preprocessing": {"write_clean_events": False}})

        with (
            patch.object(PreprocessingPipeline, "_run_mne_bids_pipeline") as run_mne,
            patch.object(PreprocessingPipeline, "_harmonize_filtered_raw_bads_for_mne_concat"),
            patch.object(PreprocessingPipeline, "_append_epoch_rejection_review"),
        ):
            p._run_epoch_creation(["0001"], "pain", task_is_rest=False, n_jobs=6)

        self.assertEqual(run_mne.call_args.kwargs["n_jobs"], 6)

    def test_run_mne_bids_pipeline_generates_the_config_with_the_requested_n_jobs(self):
        from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

        p = self._pipeline()

        with (
            patch.object(
                PreprocessingPipeline, "_generate_mne_bids_config", return_value="x = 1"
            ) as generate,
            patch(
                "eeg_pipeline.pipelines.preprocessing.subprocess.run",
                return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
            ),
        ):
            p._run_mne_bids_pipeline("init", subjects=["0001"], task="pain", n_jobs=6)

        self.assertEqual(generate.call_args.kwargs["n_jobs"], 6)

    def test_generated_config_binds_n_jobs_to_the_requested_value(self):
        """The emitted name must be upstream's, so assert on the executed config."""

        p = self._pipeline(
            {
                "preprocessing": {
                    "task_is_rest": True,
                    "rest_epochs_duration": 12.0,
                    "rest_epochs_overlap": 0.0,
                },
                "epochs": {"baseline": [-0.2, 0.0], "tmin": -7.0, "tmax": 15.0},
            }
        )

        source = p._generate_mne_bids_config(
            "preprocessing/_06a1_fit_ica",
            subjects=["0001"],
            task=None,
            task_is_rest=True,
            n_jobs=6,
        )

        namespace: dict = {}
        exec(compile(source, "<generated>", "exec"), namespace)
        self.assertEqual(namespace["n_jobs"], 6)

    def test_zero_n_jobs_is_rejected_before_anything_runs(self):
        """joblib reads 0 as an error and MNE-BIDS reads it as 0 workers; neither is 'all'."""

        p = self._pipeline({"project": {"task": "pain"}, "preprocessing": {"task_is_rest": False}})

        with self.assertRaisesRegex(ValueError, "n_jobs"):
            p._extract_preprocessing_params("pain", {"n_jobs": 0})

    def test_negative_n_jobs_is_accepted_as_the_all_cores_selector(self):
        p = self._pipeline({"project": {"task": "pain"}, "preprocessing": {"task_is_rest": False}})

        resolved = p._extract_preprocessing_params("pain", {"n_jobs": -1})

        self.assertEqual(resolved[4], -1)


# Moved from tests/pipelines/test_eeg_only_gating.py when preprocessing.eeg_fmri was
# deleted. That file existed to prove eeg_fmri and brainvision_analyzer were two switches
# rather than one; the distinction stopped existing with the first key. These cases are
# the part that outlived it: what they assert is now the only path, not the EEG-only one.


def test_the_ecg_coupling_metric_is_gated_on_the_recorded_lead() -> None:
    # Issue #14: a dataset with a recorded ECG lead used to have this metric switched off
    # underneath it by a scanner declaration. It runs wherever the lead is named.
    from eeg_pipeline.utils.config.loader import load_config
    from eeg_pipeline.utils.data.preprocessing import CleanEventsQCConfig

    config = load_config()
    config["preprocessing.clean_events_qc.enabled"] = True
    config["preprocessing.clean_events_qc.ecg_coupling.enabled"] = True
    config["eeg.ecg_channels"] = ["ECG"]

    assert CleanEventsQCConfig.from_config(config).ecg_coupling.enabled is True


def test_cardiac_only_qc_switches_itself_off_rather_than_raising() -> None:
    # Asking for QC and naming no metric is a config mistake worth raising on. Asking for
    # the cardiac metric alone with no ECG lead named is not: nothing is left to compute
    # and the user got nothing wrong.
    from eeg_pipeline.utils.config.loader import load_config
    from eeg_pipeline.utils.data.preprocessing import CleanEventsQCConfig

    config = load_config()
    config["preprocessing.clean_events_qc.enabled"] = True
    config["preprocessing.clean_events_qc.ecg_coupling.enabled"] = True
    config["preprocessing.clean_events_qc.peripheral_low_gamma.enabled"] = False
    config["eeg.ecg_channels"] = []

    assert CleanEventsQCConfig.from_config(config).enabled is False


def test_naming_no_metric_at_all_still_raises() -> None:
    from eeg_pipeline.utils.config.loader import load_config
    from eeg_pipeline.utils.data.preprocessing import CleanEventsQCConfig

    config = load_config()
    config["preprocessing.clean_events_qc.enabled"] = True
    config["preprocessing.clean_events_qc.ecg_coupling.enabled"] = False
    config["preprocessing.clean_events_qc.peripheral_low_gamma.enabled"] = False

    with pytest.raises(ValueError, match="at least one QC metric"):
        CleanEventsQCConfig.from_config(config)
