import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import mne
import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "crop_restart_scanner_block.py"

SFREQ = 1000.0
TR = 0.9
CHANNELS = ["Fp1", "Fp2", "Cz", "Pz"]


def _write_run(
    eeg_dir: Path,
    subject: str,
    task: str,
    run: int,
    *,
    volume_onsets: list[float],
    trigger_onsets: list[float],
    duration_s: float,
) -> Path:
    """Write a BrainVision run plus its BIDS sidecars, as the conversion leaves them.

    The ``.vmrk`` carries the raw BrainVision codes and ``events.tsv`` carries the
    sanitized names, which is how the delivered dataset is actually spelled.
    """
    eeg_dir.mkdir(parents=True, exist_ok=True)
    n_times = int(round(duration_s * SFREQ))
    rng = np.random.default_rng(0)
    data = rng.normal(scale=1e-5, size=(len(CHANNELS), n_times))
    info = mne.create_info(CHANNELS, SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    onsets = list(volume_onsets) + list(trigger_onsets)
    descriptions = ["Stimulus/S  3"] * len(volume_onsets) + ["Stimulus/S  2"] * len(trigger_onsets)
    order = np.argsort(onsets)
    raw.set_annotations(
        mne.Annotations(
            onset=[onsets[i] for i in order],
            duration=[0.001] * len(onsets),
            description=[descriptions[i] for i in order],
        ),
        verbose="ERROR",
    )

    stem = f"{subject}_task-{task}_run-{run}"
    mne.export.export_raw(eeg_dir / f"{stem}_eeg.vhdr", raw, fmt="brainvision", verbose="ERROR")

    events = pd.DataFrame(
        {
            "onset": [onsets[i] for i in order],
            "duration": [0.001] * len(onsets),
            "trial_type": [
                "Volume/V  1" if descriptions[i] == "Stimulus/S  3" else "Trig_therm/T  1"
                for i in order
            ],
            "value": [3 if descriptions[i] == "Stimulus/S  3" else 2 for i in order],
            "sample": [int(round(onsets[i] * SFREQ)) for i in order],
            "stimulus_temp": [
                np.nan if descriptions[i] == "Stimulus/S  3" else 49.3 for i in order
            ],
        }
    )
    events.to_csv(eeg_dir / f"{stem}_events.tsv", sep="\t", index=False)

    (eeg_dir / f"{stem}_eeg.json").write_text(
        json.dumps({"TaskName": task, "SamplingFrequency": SFREQ, "RecordingDuration": duration_s}),
        encoding="utf-8",
    )
    return eeg_dir / f"{stem}_eeg.vhdr"


class TestCropRestartScannerBlock(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.bids_root = self.tmp / "bids_output" / "eeg"
        self.eeg_dir = self.bids_root / "sub-9999" / "eeg"
        self.stem = "sub-9999_task-thermalactive_run-1"

    def _run_script(self, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--bids-root",
                str(self.bids_root),
                "--subject",
                "9999",
                "--task",
                "thermalactive",
                "--run",
                "1",
                *extra,
            ],
            capture_output=True,
            text=True,
        )

    def test_crops_two_block_run_to_its_first_volume_block(self):
        block_a = [round(i * TR, 3) for i in range(20)]  # 0.0 .. 17.1
        block_b = [round(30.0 + i * TR, 3) for i in range(6)]  # 30.0 .. 34.5
        _write_run(
            self.eeg_dir,
            "sub-9999",
            "thermalactive",
            1,
            volume_onsets=block_a + block_b,
            trigger_onsets=[5.0, 31.5],
            duration_s=36.0,
        )

        result = self._run_script()
        self.assertEqual(result.returncode, 0, result.stderr)

        raw = mne.io.read_raw_brainvision(
            self.eeg_dir / f"{self.stem}_eeg.vhdr", preload=False, verbose="ERROR"
        )
        # Kept through the last block-A volume plus one repetition time.
        self.assertEqual(raw.n_times, 18000)

        kept_volumes = [
            onset
            for onset, description in zip(raw.annotations.onset, raw.annotations.description)
            if description == "Stimulus/S  3"
        ]
        self.assertEqual(len(kept_volumes), len(block_a))
        self.assertAlmostEqual(max(kept_volumes), block_a[-1], places=3)

        events = pd.read_csv(self.eeg_dir / f"{self.stem}_events.tsv", sep="\t")
        self.assertEqual(int((events.trial_type == "Volume/V  1").sum()), len(block_a))
        self.assertEqual(int((events.trial_type == "Trig_therm/T  1").sum()), 1)
        self.assertIn("stimulus_temp", events.columns)

        sidecar = json.loads((self.eeg_dir / f"{self.stem}_eeg.json").read_text())
        self.assertAlmostEqual(sidecar["RecordingDuration"], 17.999, places=3)

    def test_leaves_a_single_block_run_untouched(self):
        volumes = [round(i * TR, 3) for i in range(20)]
        _write_run(
            self.eeg_dir,
            "sub-9999",
            "thermalactive",
            1,
            volume_onsets=volumes,
            trigger_onsets=[5.0],
            duration_s=20.0,
        )
        before = {
            suffix: (self.eeg_dir / f"{self.stem}_{suffix}").read_bytes()
            for suffix in ("eeg.eeg", "eeg.vmrk", "events.tsv", "eeg.json")
        }

        result = self._run_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("one acquisition block", result.stdout)
        for suffix, content in before.items():
            self.assertEqual(
                (self.eeg_dir / f"{self.stem}_{suffix}").read_bytes(),
                content,
                f"{suffix} was modified",
            )

    def test_retained_samples_are_bit_identical(self):
        block_a = [round(i * TR, 3) for i in range(20)]
        block_b = [round(30.0 + i * TR, 3) for i in range(6)]
        _write_run(
            self.eeg_dir,
            "sub-9999",
            "thermalactive",
            1,
            volume_onsets=block_a + block_b,
            trigger_onsets=[5.0, 31.5],
            duration_s=36.0,
        )
        binary_path = self.eeg_dir / f"{self.stem}_eeg.eeg"
        kept_bytes = binary_path.read_bytes()[: 18000 * len(CHANNELS) * 4]

        self.assertEqual(self._run_script().returncode, 0)

        self.assertEqual(binary_path.read_bytes(), kept_bytes)

    def test_backs_up_every_file_it_rewrites(self):
        block_a = [round(i * TR, 3) for i in range(20)]
        block_b = [round(30.0 + i * TR, 3) for i in range(6)]
        _write_run(
            self.eeg_dir,
            "sub-9999",
            "thermalactive",
            1,
            volume_onsets=block_a + block_b,
            trigger_onsets=[5.0, 31.5],
            duration_s=36.0,
        )
        original = (self.eeg_dir / f"{self.stem}_eeg.eeg").read_bytes()

        self.assertEqual(self._run_script().returncode, 0)

        for suffix in ("eeg.eeg", "eeg.vmrk", "events.tsv", "eeg.json"):
            self.assertTrue(
                (self.eeg_dir / f"{self.stem}_{suffix}.precrop.bak").exists(),
                f"no backup for {suffix}",
            )
        self.assertEqual((self.eeg_dir / f"{self.stem}_eeg.eeg.precrop.bak").read_bytes(), original)

    def test_refuses_to_overwrite_a_backup_from_an_earlier_repair(self):
        block_a = [round(i * TR, 3) for i in range(20)]
        block_b = [round(30.0 + i * TR, 3) for i in range(6)]
        _write_run(
            self.eeg_dir,
            "sub-9999",
            "thermalactive",
            1,
            volume_onsets=block_a + block_b,
            trigger_onsets=[5.0, 31.5],
            duration_s=36.0,
        )
        events_path = self.eeg_dir / f"{self.stem}_events.tsv"
        stale_backup = self.eeg_dir / f"{self.stem}_events.tsv.precrop.bak"
        stale_backup.write_text("earlier repair", encoding="utf-8")
        events_before = events_path.read_bytes()

        result = self._run_script()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("backup", result.stdout)
        self.assertEqual(stale_backup.read_text(encoding="utf-8"), "earlier repair")
        self.assertEqual(events_path.read_bytes(), events_before)


if __name__ == "__main__":
    unittest.main()
