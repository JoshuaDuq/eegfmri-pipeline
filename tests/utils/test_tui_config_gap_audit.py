from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


class TestTuiConfigGapAudit(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.script = self.repo_root / "eeg_pipeline" / "utils" / "config" / "introspect.py"

    def test_audit_script_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            json_out = Path(tmpdir) / "audit.json"
            md_out = Path(tmpdir) / "audit.md"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(self.script),
                    "--repo-root",
                    str(self.repo_root),
                    "--json-out",
                    str(json_out),
                    "--markdown-out",
                    str(md_out),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                proc.returncode,
                0,
                msg=f"audit script failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
            )
            self.assertTrue(json_out.exists(), msg="Expected JSON output to be created")
            self.assertTrue(md_out.exists(), msg="Expected markdown output to be created")

            payload = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertIn("counts", payload)
            self.assertIn("coverage", payload)
            self.assertIn("backend_keys_missing_in_tui", payload["coverage"])
            self.assertIn("set_support", payload)
            self.assertIn("enabled", payload["set_support"])
            self.assertIn("cli_parser_support", payload["set_support"])
            self.assertIn("tui_support", payload["set_support"])
            self.assertIn("precedence_ok", payload["set_support"])
            self.assertIn("direct_hydration_roots_not_loaded", payload["coverage"])

    def test_strict_mode_has_no_missing_backend_keys(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(self.script),
                "--repo-root",
                str(self.repo_root),
                "--strict",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            proc.returncode,
            0,
            msg=f"Found missing backend keys in TUI coverage:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
        )

    def test_direct_tui_hydration_roots_are_loaded_by_wizard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            json_out = Path(tmpdir) / "audit.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(self.script),
                    "--repo-root",
                    str(self.repo_root),
                    "--json-out",
                    str(json_out),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)

            payload = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(
                [],
                payload["coverage"]["direct_hydration_roots_not_loaded"],
            )

    def test_yaml_exposes_scientific_config_defaults(self) -> None:
        eeg_config_path = (
            self.repo_root
            / "eeg_pipeline"
            / "utils"
            / "config"
            / "eeg_config.yaml"
        )
        fmri_config_path = (
            self.repo_root
            / "fmri_pipeline"
            / "utils"
            / "config"
            / "fmri_config.yaml"
        )

        eeg_config = yaml.safe_load(eeg_config_path.read_text(encoding="utf-8"))
        fmri_config = yaml.safe_load(fmri_config_path.read_text(encoding="utf-8"))

        psd_config = eeg_config["feature_engineering"]["psd"]
        for key in ("fmin", "fmax", "n_fft", "n_overlap", "window", "n_jobs"):
            self.assertIn(key, psd_config)

        pyprep_config = eeg_config["pyprep"]
        self.assertEqual("per_run", pyprep_config["bad_channel_sync_policy"])

        self.assertEqual(["brain", "other"], eeg_config["ica"]["labels_to_keep"])

        cnn_config = eeg_config["machine_learning"]["models"]["cnn"]
        self.assertIn("standardization_std_floor", cnn_config)

        for section in ("fmri_contrast", "fmri_resting_state"):
            self.assertIn("auto_compcor_n", fmri_config[section])


if __name__ == "__main__":
    unittest.main()
