import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class TestFixRestartTrialTriggersScript(unittest.TestCase):
    def test_script_relabels_extra_restart_triggers(self):
        tmp = Path(tempfile.mkdtemp())
        source_root = tmp / "source_data"
        bids_root = tmp / "bids_output" / "eeg"

        psycho_dir = source_root / "sub-0002" / "PsychoPy_Data"
        eeg_dir = bids_root / "sub-0002" / "eeg"
        psycho_dir.mkdir(parents=True, exist_ok=True)
        eeg_dir.mkdir(parents=True, exist_ok=True)

        events_path = eeg_dir / "sub-0002_task-task_run-1_events.tsv"
        behavior_path = (
            psycho_dir / "sub0001_ThermalPainEEGFMRI_run1_2026-02-09_11h11.44.706_TrialSummary.csv"
        )

        event_onsets = [
            22.150,
            65.084,
            113.135,
            159.902,
            203.203,
            250.486,
            294.994,
            342.587,
            390.405,
            432.889,
            478.806,
            599.142,
            645.162,
            688.212,
        ]
        pd.DataFrame(
            {
                "onset": event_onsets,
                "trial_type": ["Trig_therm/T 1"] * len(event_onsets),
            }
        ).to_csv(events_path, sep="\t", index=False)

        pd.DataFrame(
            {
                "stim_start_time": [
                    52.73450000025332,
                    95.66813679970800,
                    143.71921419911090,
                    190.48620080016553,
                    233.78686190024015,
                    281.06928610056640,
                    325.58110629953444,
                    373.16949559934440,
                    420.98703229986130,
                    463.47044919990000,
                    509.38815150037410,
                ]
            }
        ).to_csv(behavior_path, index=False)

        script_path = Path("scripts/fix_restart_trial_triggers.py")
        cmd = [
            sys.executable,
            str(script_path),
            "--source-root",
            str(source_root),
            "--bids-root",
            str(bids_root),
            "--subject",
            "0002",
            "--task",
            "task",
            "--run",
            "1",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(
            result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}"
        )

        repaired = pd.read_csv(events_path, sep="\t")
        bad_mask = repaired["trial_type"].astype(str).str.startswith("BAD_restart/")
        self.assertEqual(int(bad_mask.sum()), 3)
        self.assertTrue((~bad_mask).sum() >= 11)

    def test_script_ignores_appledouble_behavior_sidecars(self):
        tmp = Path(tempfile.mkdtemp())
        source_root = tmp / "source_data"
        bids_root = tmp / "bids_output" / "eeg"

        psycho_dir = source_root / "sub-0002" / "PsychoPy_Data"
        eeg_dir = bids_root / "sub-0002" / "eeg"
        psycho_dir.mkdir(parents=True, exist_ok=True)
        eeg_dir.mkdir(parents=True, exist_ok=True)

        events_path = eeg_dir / "sub-0002_task-task_run-1_events.tsv"
        behavior_path = psycho_dir / "sub0001_ThermalPainEEGFMRI_run1_TrialSummary.csv"
        sidecar_path = psycho_dir / "._sub0001_ThermalPainEEGFMRI_run1_TrialSummary.csv"

        event_onsets = [
            22.150,
            65.084,
            113.135,
            159.902,
            203.203,
            250.486,
            294.994,
            342.587,
            390.405,
            432.889,
            478.806,
            599.142,
            645.162,
            688.212,
        ]
        pd.DataFrame(
            {
                "onset": event_onsets,
                "trial_type": ["Trig_therm/T 1"] * len(event_onsets),
            }
        ).to_csv(events_path, sep="\t", index=False)
        pd.DataFrame(
            {
                "stim_start_time": [
                    52.73450000025332,
                    95.66813679970800,
                    143.71921419911090,
                    190.48620080016553,
                    233.78686190024015,
                    281.06928610056640,
                    325.58110629953444,
                    373.16949559934440,
                    420.98703229986130,
                    463.47044919990000,
                    509.38815150037410,
                ]
            }
        ).to_csv(behavior_path, index=False)
        sidecar_path.write_bytes(b"\xb0not,a,csv")

        script_path = Path("scripts/fix_restart_trial_triggers.py")
        cmd = [
            sys.executable,
            str(script_path),
            "--source-root",
            str(source_root),
            "--bids-root",
            str(bids_root),
            "--subject",
            "0002",
            "--task",
            "task",
            "--run",
            "1",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(
            result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}"
        )

        repaired = pd.read_csv(events_path, sep="\t")
        bad_mask = repaired["trial_type"].astype(str).str.startswith("BAD_restart/")
        self.assertEqual(int(bad_mask.sum()), 3)

    def test_script_regenerates_combined_events_from_repaired_run_files(self):
        tmp = Path(tempfile.mkdtemp())
        source_root = tmp / "source_data"
        bids_root = tmp / "bids_output" / "eeg"

        psycho_dir = source_root / "sub-0002" / "PsychoPy_Data"
        eeg_dir = bids_root / "sub-0002" / "eeg"
        psycho_dir.mkdir(parents=True, exist_ok=True)
        eeg_dir.mkdir(parents=True, exist_ok=True)

        run1_path = eeg_dir / "sub-0002_task-task_run-1_events.tsv"
        run2_path = eeg_dir / "sub-0002_task-task_run-2_events.tsv"
        combined_path = eeg_dir / "sub-0002_task-task_events.tsv"
        behavior_path = psycho_dir / "sub0001_ThermalPainEEGFMRI_run1_TrialSummary.csv"

        run1_onsets = [
            22.150,
            65.084,
            113.135,
            159.902,
            203.203,
            250.486,
            294.994,
            342.587,
            390.405,
            432.889,
            478.806,
            599.142,
            645.162,
            688.212,
        ]
        run2_onsets = [10.0, 20.0, 30.0]
        pd.DataFrame(
            {
                "onset": run1_onsets,
                "sample": list(range(len(run1_onsets))),
                "trial_type": ["Trig_therm/T 1"] * len(run1_onsets),
            }
        ).to_csv(run1_path, sep="\t", index=False)
        pd.DataFrame(
            {
                "onset": run2_onsets,
                "sample": list(range(len(run2_onsets))),
                "trial_type": ["Trig_therm/T 1"] * len(run2_onsets),
            }
        ).to_csv(run2_path, sep="\t", index=False)
        pd.concat(
            [
                pd.read_csv(run1_path, sep="\t"),
                pd.read_csv(run2_path, sep="\t"),
            ],
            ignore_index=True,
        ).to_csv(combined_path, sep="\t", index=False)

        pd.DataFrame(
            {
                "stim_start_time": [
                    52.73450000025332,
                    95.66813679970800,
                    143.71921419911090,
                    190.48620080016553,
                    233.78686190024015,
                    281.06928610056640,
                    325.58110629953444,
                    373.16949559934440,
                    420.98703229986130,
                    463.47044919990000,
                    509.38815150037410,
                ]
            }
        ).to_csv(behavior_path, index=False)

        script_path = Path("scripts/fix_restart_trial_triggers.py")
        cmd = [
            sys.executable,
            str(script_path),
            "--source-root",
            str(source_root),
            "--bids-root",
            str(bids_root),
            "--subject",
            "0002",
            "--task",
            "task",
            "--run",
            "1",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(
            result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}"
        )

        repaired_run1 = pd.read_csv(run1_path, sep="\t")
        combined = pd.read_csv(combined_path, sep="\t")
        run_bad_mask = repaired_run1["trial_type"].astype(str).str.startswith("BAD_restart/")
        combined_bad_mask = combined["trial_type"].astype(str).str.startswith("BAD_restart/")

        self.assertEqual(int(run_bad_mask.sum()), 3)
        self.assertEqual(int(combined_bad_mask.sum()), 3)
        self.assertEqual(combined["run_id"].tolist(), [1] * 14 + [2] * 3)
        self.assertEqual(combined["sample"].tolist(), list(range(17)))
