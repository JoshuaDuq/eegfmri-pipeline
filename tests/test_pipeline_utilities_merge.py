import json
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


class TestMergePsychopyGapfill(unittest.TestCase):
    def test_run_batch_validates_each_subject_and_qc_edges(self):
        from eeg_pipeline.pipelines.merge_psychopy import MergePsychopyPipeline

        tmp = Path(tempfile.mkdtemp())
        bids = tmp / "bids"
        fmri = tmp / "fmri"
        eeg_dir = bids / "sub-0001" / "eeg"
        fmri_dir = fmri / "sub-0001" / "func"
        eeg_dir.mkdir(parents=True, exist_ok=True)
        fmri_dir.mkdir(parents=True, exist_ok=True)

        p = object.__new__(MergePsychopyPipeline)
        p.config = DotConfig({"paths": {"bids_fmri_root": str(fmri)}, "alignment": {"cross_modal_qc_columns": ["condition", "stimulus_temp"]}})
        p.bids_root = bids
        p.source_root = tmp / "source"
        p.logger = Mock()

        with patch("eeg_pipeline.pipelines.merge_psychopy.run_merge_psychopy", return_value=2), patch.object(
            MergePsychopyPipeline, "_validate_against_fmri_events"
        ) as mval:
            out = p.run_batch(["0001", "0002"], task="t", dry_run=False)
        self.assertEqual(out, 2)
        self.assertEqual(mval.call_count, 2)

        # fallback _bold_events branch + mismatch/non-overlap branches
        (eeg_dir / "sub-0001_task-t_run-01_events.tsv").write_text("x", encoding="utf-8")
        (eeg_dir / "sub-0001_task-t_no_run_events.tsv").write_text("x", encoding="utf-8")
        (fmri_dir / "sub-0001_task-t_run-01_bold_events.tsv").write_text("x", encoding="utf-8")

        eeg_df = pd.DataFrame(
            {
                "trial_number": [1, 2, 3],
                "condition": ["a", "b", "c"],
                "stimulus_temp": [45.0, 46.0, 47.0],
            }
        )
        fmri_df = pd.DataFrame(
            {
                "trial_number": [1, 4, 5],
                "condition": ["x", "x", "y"],
                "stimulus_temp": [46.0, 50.0, 51.0],
            }
        )

        with patch("eeg_pipeline.pipelines.merge_psychopy.read_tsv", side_effect=[eeg_df, fmri_df]):
            p._validate_against_fmri_events("0001", "t")
        self.assertTrue(p.logger.warning.called)

    def test_validate_returns_for_empty_cols_missing_trial_and_empty_trials(self):
        from eeg_pipeline.pipelines.merge_psychopy import MergePsychopyPipeline

        tmp = Path(tempfile.mkdtemp())
        bids = tmp / "bids"
        fmri = tmp / "fmri"
        eeg_dir = bids / "sub-0001" / "eeg"
        fmri_dir = fmri / "sub-0001" / "func"
        eeg_dir.mkdir(parents=True, exist_ok=True)
        fmri_dir.mkdir(parents=True, exist_ok=True)

        eeg_file = eeg_dir / "sub-0001_task-t_run-01_events.tsv"
        fmri_file = fmri_dir / "sub-0001_task-t_run-01_events.tsv"
        eeg_file.write_text("x", encoding="utf-8")
        fmri_file.write_text("x", encoding="utf-8")

        p = object.__new__(MergePsychopyPipeline)
        p.config = DotConfig({"paths": {"bids_fmri_root": str(fmri)}, "alignment": {"cross_modal_qc_columns": ""}})
        p.bids_root = bids
        p.source_root = tmp / "source"
        p.logger = Mock()

        # cols empty branch
        p._validate_against_fmri_events("0001", "t")

        # missing trial_number branch
        with patch("eeg_pipeline.pipelines.merge_psychopy.read_tsv", side_effect=[pd.DataFrame({"a": [1]}), pd.DataFrame({"trial_number": [1]})]):
            p._validate_against_fmri_events("0001", "t", qc_columns=["condition"])

        # empty trials branch
        with patch("eeg_pipeline.pipelines.merge_psychopy.read_tsv", side_effect=[pd.DataFrame({"trial_number": [None]}), pd.DataFrame({"trial_number": [None]})]):
            p._validate_against_fmri_events("0001", "t", qc_columns=["condition"])

    def test_validate_additional_early_returns_and_runnum_edges(self):
        from eeg_pipeline.pipelines.merge_psychopy import MergePsychopyPipeline

        tmp = Path(tempfile.mkdtemp())
        bids = tmp / "bids"
        fmri = tmp / "fmri"
        bids.mkdir(parents=True, exist_ok=True)
        p = object.__new__(MergePsychopyPipeline)
        p.bids_root = bids
        p.source_root = tmp / "source"
        p.logger = Mock()

        # fmri_root missing -> return
        p.config = DotConfig({"paths": {"bids_fmri_root": None}})
        p._validate_against_fmri_events("0001", "t")

        # fmri_root does not exist -> return
        p.config = DotConfig({"paths": {"bids_fmri_root": str(fmri)}})
        p._validate_against_fmri_events("0001", "t")

        eeg_dir = bids / "sub-0001" / "eeg"
        fmri_dir = fmri / "sub-0001" / "func"
        eeg_dir.mkdir(parents=True, exist_ok=True)
        fmri_dir.mkdir(parents=True, exist_ok=True)
        p.config = DotConfig({"paths": {"bids_fmri_root": str(fmri)}, "alignment": {"cross_modal_qc_columns": ["condition"]}})

        # no matching glob results -> return
        p._validate_against_fmri_events("0001", "t")

        # run_num None path (regex miss) and no matching fMRI run branch
        (eeg_dir / "sub-0001_task-t_run-ab_events.tsv").write_text("x", encoding="utf-8")
        (eeg_dir / "sub-0001_task-t_run-02_events.tsv").write_text("x", encoding="utf-8")
        (fmri_dir / "sub-0001_task-t_run-01_events.tsv").write_text("x", encoding="utf-8")
        with patch("eeg_pipeline.pipelines.merge_psychopy.read_tsv", return_value=pd.DataFrame({"trial_number": [1], "condition": ["a"]})):
            p._validate_against_fmri_events("0001", "t", qc_columns=["condition"])

        # run_num int ValueError fallback via patched int
        with patch("eeg_pipeline.pipelines.merge_psychopy.int", side_effect=ValueError("bad-int")):
            p._validate_against_fmri_events("0001", "t", qc_columns=["condition"])

        # missing column branch and merged.empty branch
        (eeg_dir / "sub-0001_task-t_run-01_events.tsv").write_text("x", encoding="utf-8")
        with patch(
            "eeg_pipeline.pipelines.merge_psychopy.read_tsv",
            side_effect=[
                pd.DataFrame({"trial_number": [1], "other": ["x"]}),
                pd.DataFrame({"trial_number": [1], "other": ["y"]}),
                pd.DataFrame({"trial_number": [1], "condition": ["x"]}),
                pd.DataFrame({"trial_number": [2], "condition": ["x"]}),
            ],
        ):
            p._validate_against_fmri_events("0001", "t", qc_columns=["condition"])
            p._validate_against_fmri_events("0001", "t", qc_columns=["condition"])

class TestUtilitiesAndMergeDeep(unittest.TestCase):
        def test_eeg_raw_to_bids_and_merge_psychopy_init_paths(self):
            from eeg_pipeline.pipelines.eeg_raw_to_bids import EEGRawToBidsPipeline
            from eeg_pipeline.pipelines.merge_psychopy import MergePsychopyPipeline

            cfg = DotConfig({"bids_root": "/tmp/bids", "paths": {"source_data": "/tmp/source"}})
            with patch("eeg_pipeline.pipelines.eeg_raw_to_bids.PipelineBase.__init__", lambda self, name, config=None: setattr(self, "config", config or cfg)):
                p1 = EEGRawToBidsPipeline(config=cfg)
            self.assertEqual(str(p1.source_root), "/tmp/source")
            self.assertIsNone(p1.run_group_level([], "task"))

            with patch("eeg_pipeline.pipelines.merge_psychopy.PipelineBase.__init__", lambda self, name, config=None: setattr(self, "config", config or cfg)):
                p2 = MergePsychopyPipeline(config=cfg)
            self.assertEqual(str(p2.source_root), "/tmp/source")
            self.assertIsNone(p2.run_group_level([], "task"))

        def test_utility_pipeline_process_subject_and_wrappers(self):
            from eeg_pipeline.pipelines.utilities import UtilityPipeline

            p = object.__new__(UtilityPipeline)
            p.config = DotConfig(
                {
                    "project": {"task": "thermalactive"},
                    "eeg": {"montage": "easycap-M1"},
                    "preprocessing": {"line_freq": 60.0},
                    "alignment": {"allow_misaligned_trim": False},
                }
            )
            p.logger = Mock()
            p.bids_root = Path(tempfile.mkdtemp()) / "bids"
            p.source_root = Path(tempfile.mkdtemp()) / "source"

            progress = SimpleNamespace(subject_start=lambda *a, **k: None, subject_done=lambda *a, **k: None, step=lambda *a, **k: None)
            with patch("eeg_pipeline.pipelines.utilities.run_raw_to_bids", return_value=1) as mock_r2b, patch(
                "eeg_pipeline.pipelines.utilities.run_merge_psychopy", return_value=1
            ) as mock_merge:
                p.process_subject("0001", task="thermalactive", progress=progress)
                p.run_raw_to_bids(task="thermalactive", subjects=["0001"])
                p.run_merge_psychopy(task="thermalactive", subjects=["0001"])
            self.assertTrue(mock_r2b.called)
            self.assertTrue(mock_merge.called)

        def test_merge_psychopy_cross_modal_qc(self):
            from eeg_pipeline.pipelines.merge_psychopy import MergePsychopyPipeline

            tmp = Path(tempfile.mkdtemp())
            bids_root = tmp / "bids"
            fmri_root = tmp / "fmri_bids"
            eeg_dir = bids_root / "sub-0001" / "eeg"
            fmri_dir = fmri_root / "sub-0001" / "func"
            eeg_dir.mkdir(parents=True, exist_ok=True)
            fmri_dir.mkdir(parents=True, exist_ok=True)
            eeg_path = eeg_dir / "sub-0001_task-thermalactive_run-01_events.tsv"
            fmri_path = fmri_dir / "sub-0001_task-thermalactive_run-01_events.tsv"
            eeg_path.write_text("x", encoding="utf-8")
            fmri_path.write_text("x", encoding="utf-8")

            eeg_df = pd.DataFrame({"trial_number": [1, 2], "stimulus_temp": [44.0, 45.0]})
            fmri_df = pd.DataFrame({"trial_number": [1, 2], "stimulus_temp": [44.1, 45.0]})

            p = object.__new__(MergePsychopyPipeline)
            p.config = DotConfig(
                {
                    "paths": {"bids_fmri_root": str(fmri_root)},
                    "alignment": {"cross_modal_qc_columns": ["stimulus_temp"]},
                }
            )
            p.bids_root = bids_root
            p.source_root = tmp / "source"
            p.logger = Mock()

            with patch("eeg_pipeline.pipelines.merge_psychopy.read_tsv", side_effect=[eeg_df, fmri_df]):
                p._validate_against_fmri_events("0001", "thermalactive")

            self.assertTrue(p.logger.warning.called)

class TestUtilitiesAndMergeCompletion(unittest.TestCase):
        def test_utility_and_merge_wrapper_functions(self):
            from eeg_pipeline.pipelines.utilities import run_raw_to_bids, run_merge_psychopy, UtilityPipeline

            with patch("eeg_pipeline.pipelines.utilities._run_raw_to_bids", return_value=2) as m1, patch(
                "eeg_pipeline.pipelines.utilities._run_merge_psychopy", return_value=3
            ) as m2:
                self.assertEqual(run_raw_to_bids(Path("/a"), Path("/b"), "t"), 2)
                self.assertEqual(run_merge_psychopy(Path("/b"), Path("/a"), "t"), 3)
            m1.assert_called_once()
            m2.assert_called_once()

            cfg = DotConfig({"bids_root": "/tmp/bids", "paths": {"source_data": "/tmp/source"}})
            with patch("eeg_pipeline.pipelines.utilities.PipelineBase.__init__", lambda self, name, config=None: setattr(self, "config", config or cfg)):
                p = UtilityPipeline(config=cfg)
            self.assertEqual(str(p.bids_root), "/tmp/bids")

        def test_merge_psychopy_qc_branches(self):
            from eeg_pipeline.pipelines.merge_psychopy import MergePsychopyPipeline

            tmp = Path(tempfile.mkdtemp())
            bids = tmp / "bids"
            fmri = tmp / "fmri"
            bids.mkdir(parents=True, exist_ok=True)
            fmri.mkdir(parents=True, exist_ok=True)
            p = object.__new__(MergePsychopyPipeline)
            p.config = DotConfig({"paths": {"bids_fmri_root": str(fmri)}, "alignment": {"cross_modal_qc_columns": "col"}})
            p.bids_root = bids
            p.source_root = tmp / "src"
            p.logger = Mock()

            # no eeg/fmri subdirs -> early return
            p._validate_against_fmri_events("0001", "t")

            eeg_dir = bids / "sub-0001" / "eeg"
            fmri_dir = fmri / "sub-0001" / "func"
            eeg_dir.mkdir(parents=True, exist_ok=True)
            fmri_dir.mkdir(parents=True, exist_ok=True)
            (eeg_dir / "sub-0001_task-t_run-01_events.tsv").write_text("x")
            (fmri_dir / "sub-0001_task-t_run-01_events.tsv").write_text("x")

            # read error branch
            with patch("eeg_pipeline.pipelines.merge_psychopy.read_tsv", side_effect=RuntimeError("bad")):
                p._validate_against_fmri_events("0001", "t")
            self.assertTrue(p.logger.warning.called)

        def test_custom_run_batch_pipelines_write_reproducibility_metadata(self):
            from eeg_pipeline.pipelines.eeg_raw_to_bids import EEGRawToBidsPipeline
            from eeg_pipeline.pipelines.merge_psychopy import MergePsychopyPipeline
            from eeg_pipeline.pipelines.utilities import UtilityPipeline

            tmp = Path(tempfile.mkdtemp())
            deriv_root = tmp / "derivatives"
            bids_root = tmp / "bids"
            source_root = tmp / "source"

            cfg = DotConfig(
                {
                    "project": {"task": "thermalactive"},
                    "bids_root": str(bids_root),
                    "paths": {"source_data": str(source_root)},
                    "alignment": {"allow_misaligned_trim": False},
                }
            )

            p_raw = object.__new__(EEGRawToBidsPipeline)
            p_raw.name = "eeg_raw_to_bids"
            p_raw.config = cfg
            p_raw.logger = Mock()
            p_raw.deriv_root = deriv_root
            p_raw.bids_root = bids_root
            p_raw.source_root = source_root

            with patch("eeg_pipeline.pipelines.eeg_raw_to_bids.run_raw_to_bids", return_value=1):
                n = p_raw.run_batch(["0001"], task="thermalactive", overwrite=True)
            self.assertEqual(n, 1)

            raw_meta = sorted((deriv_root / "logs" / "run_metadata" / "eeg_raw_to_bids").glob("run_*.json"))
            self.assertTrue(raw_meta)
            raw_payload = json.loads(raw_meta[-1].read_text(encoding="utf-8"))
            self.assertEqual(raw_payload["status"], "success")
            self.assertEqual(raw_payload["specifications"]["overwrite"], True)

            p_merge = object.__new__(MergePsychopyPipeline)
            p_merge.name = "merge_psychopy"
            p_merge.config = cfg
            p_merge.logger = Mock()
            p_merge.deriv_root = deriv_root
            p_merge.bids_root = bids_root
            p_merge.source_root = source_root

            with patch("eeg_pipeline.pipelines.merge_psychopy.run_merge_psychopy", return_value=1), patch.object(
                MergePsychopyPipeline,
                "_validate_against_fmri_events",
            ):
                n = p_merge.run_batch(["0001"], task="thermalactive", dry_run=False)
            self.assertEqual(n, 1)

            merge_meta = sorted((deriv_root / "logs" / "run_metadata" / "merge_psychopy").glob("run_*.json"))
            self.assertTrue(merge_meta)
            merge_payload = json.loads(merge_meta[-1].read_text(encoding="utf-8"))
            self.assertEqual(merge_payload["status"], "success")

            p_util = object.__new__(UtilityPipeline)
            p_util.name = "utilities"
            p_util.config = cfg
            p_util.logger = Mock()
            p_util.deriv_root = deriv_root
            p_util.bids_root = bids_root
            p_util.source_root = source_root

            with patch("eeg_pipeline.pipelines.utilities.run_raw_to_bids", return_value=1), patch(
                "eeg_pipeline.pipelines.utilities.run_merge_psychopy", return_value=1
            ):
                p_util.run_batch(["0001"], task="thermalactive", progress=_NoopProgress())

            util_meta = sorted((deriv_root / "logs" / "run_metadata" / "utilities").glob("run_*.json"))
            self.assertTrue(util_meta)
            util_payload = json.loads(util_meta[-1].read_text(encoding="utf-8"))
            self.assertEqual(util_payload["status"], "success")
