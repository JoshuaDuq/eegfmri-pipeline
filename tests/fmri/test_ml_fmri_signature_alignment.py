from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests.pipelines_test_utils import DotConfig


class TestMlFmriSignatureAlignment(unittest.TestCase):
    def test_load_fmri_signature_from_subject_suffixed_file_and_trial_alignment(self):
        from eeg_pipeline.utils.data.machine_learning import _load_fmri_signature_target_for_subject

        cfg = DotConfig(
            {
                "machine_learning": {
                    "fmri_signature": {
                        "method": "lss",
                        "contrast_name": "contrast",
                        "signature_name": "NPS",
                        "metric": "dot",
                        "normalization": "none",
                        "round_decimals": 3,
                    }
                }
            }
        )

        events_df = pd.DataFrame(
            {
                "run_id": [1, 1],
                "trial_number": [1, 2],
                # EEG Trig_therm-like timing/duration (does not match fMRI plateau rows)
                "onset": [22.150, 65.084],
                "duration": [0.001, 0.001],
            }
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sig_dir = (
                root
                / "sub-0001"
                / "fmri"
                / "lss"
                / "task-task"
                / "contrast-contrast"
                / "signatures"
            )
            sig_dir.mkdir(parents=True, exist_ok=True)

            # Subject-suffixed filename (no canonical trial_signature_expression.tsv).
            sig_df = pd.DataFrame(
                {
                    "run": ["run-01", "run-01"],
                    "run_num": [1, 1],
                    "trial_index": [1, 2],
                    "signature": ["NPS", "NPS"],
                    "dot": [1.25, 2.50],
                    # fMRI plateau-like timing/duration
                    "onset": [21.532, 64.465],
                    "duration": [7.5, 7.5],
                }
            )
            sig_df.to_csv(sig_dir / "trial_signature_expression_sub_0001.tsv", sep="\t", index=False)

            trials_df = pd.DataFrame(
                {
                    "run": ["run-01", "run-01"],
                    "trial_index": [1, 2],
                    "onset": [21.532, 64.465],
                    "duration": [7.5, 7.5],
                    "events_trial_number": [1, 2],
                }
            )
            trials_df.to_csv(sig_dir.parent / "trials.tsv", sep="\t", index=False)

            y, y_label, _extra = _load_fmri_signature_target_for_subject(
                subject_raw="0001",
                task="task",
                deriv_root=root,
                config=cfg,
                events_df=events_df,
                logger=logging.getLogger(__name__),
            )

        arr = np.asarray(y, dtype=float)
        self.assertEqual(y_label, "fmri_signature.lss.contrast.NPS.dot")
        self.assertTrue(np.all(np.isfinite(arr)))
        self.assertTrue(np.allclose(arr, np.array([1.25, 2.50], dtype=float)))

    def test_prefers_subject_specific_signature_tsv_when_multiple_candidates_exist(self):
        from eeg_pipeline.utils.data.machine_learning import _load_fmri_signature_target_for_subject

        cfg = DotConfig(
            {
                "machine_learning": {
                    "fmri_signature": {
                        "method": "lss",
                        "contrast_name": "contrast",
                        "signature_name": "NPS",
                        "metric": "dot",
                        "normalization": "none",
                        "round_decimals": 3,
                    }
                }
            }
        )

        events_df = pd.DataFrame(
            {
                "run_id": [1],
                "trial_number": [1],
                "onset": [22.150],
                "duration": [0.001],
            }
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sig_dir = (
                root
                / "sub-0001"
                / "fmri"
                / "lss"
                / "task-task"
                / "contrast-contrast"
                / "signatures"
            )
            sig_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                {
                    "run": ["run-01"],
                    "run_num": [1],
                    "trial_index": [1],
                    "signature": ["NPS"],
                    "dot": [9.99],
                    "onset": [21.532],
                    "duration": [7.5],
                }
            ).to_csv(sig_dir / "trial_signature_expression_sub_9999.tsv", sep="\t", index=False)
            pd.DataFrame(
                {
                    "run": ["run-01"],
                    "run_num": [1],
                    "trial_index": [1],
                    "signature": ["NPS"],
                    "dot": [1.11],
                    "onset": [21.532],
                    "duration": [7.5],
                }
            ).to_csv(sig_dir / "trial_signature_expression_sub_0001.tsv", sep="\t", index=False)

            pd.DataFrame(
                {
                    "run": ["run-01"],
                    "trial_index": [1],
                    "events_trial_number": [1],
                }
            ).to_csv(sig_dir.parent / "trials.tsv", sep="\t", index=False)

            y, _y_label, _extra = _load_fmri_signature_target_for_subject(
                subject_raw="0001",
                task="task",
                deriv_root=root,
                config=cfg,
                events_df=events_df,
                logger=logging.getLogger(__name__),
            )

        arr = np.asarray(y, dtype=float)
        self.assertEqual(arr.shape, (1,))
        self.assertAlmostEqual(float(arr[0]), 1.11, places=6)

    def test_keeps_onset_based_alignment_when_trial_ids_unavailable(self):
        from eeg_pipeline.utils.data.machine_learning import _load_fmri_signature_target_for_subject

        cfg = DotConfig(
            {
                "machine_learning": {
                    "fmri_signature": {
                        "method": "lss",
                        "contrast_name": "contrast",
                        "signature_name": "NPS",
                        "metric": "dot",
                        "normalization": "none",
                        "round_decimals": 3,
                    }
                }
            }
        )

        events_df = pd.DataFrame(
            {
                "run_id": [1, 1],
                "onset": [10.1234, 20.5678],
                "duration": [7.5, 7.5],
            }
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sig_dir = (
                root
                / "sub-0001"
                / "fmri"
                / "lss"
                / "task-task"
                / "contrast-contrast"
                / "signatures"
            )
            sig_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                {
                    "run_num": [1, 1],
                    "onset": [10.123, 20.568],
                    "duration": [7.5, 7.5],
                    "signature": ["NPS", "NPS"],
                    "dot": [3.0, 4.0],
                }
            ).to_csv(sig_dir / "trial_signature_expression.tsv", sep="\t", index=False)

            y, _y_label, _extra = _load_fmri_signature_target_for_subject(
                subject_raw="0001",
                task="task",
                deriv_root=root,
                config=cfg,
                events_df=events_df,
                logger=logging.getLogger(__name__),
            )

        arr = np.asarray(y, dtype=float)
        self.assertTrue(np.all(np.isfinite(arr)))
        self.assertTrue(np.allclose(arr, np.array([3.0, 4.0], dtype=float)))
