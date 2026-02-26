from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


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
                    "python3",
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

    def test_strict_mode_has_no_missing_backend_keys(self) -> None:
        proc = subprocess.run(
            [
                "python3",
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


if __name__ == "__main__":
    unittest.main()
