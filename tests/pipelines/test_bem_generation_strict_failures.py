from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fmri_pipeline.analysis.bem_generation import generate_bem_model_and_solution


class TestBemGenerationStrictFailures(unittest.TestCase):
    def test_bem_generation_failure_writes_qc_artifact_and_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            subjects_dir = root / "subjects"
            subject = "sub-0001"
            subject_dir = subjects_dir / subject
            subject_dir.mkdir(parents=True, exist_ok=True)
            license_path = root / "license.txt"
            license_path.write_text("license", encoding="utf-8")

            failed_run = SimpleNamespace(
                returncode=1,
                stdout="docker stdout",
                stderr="docker stderr",
            )

            with (
                patch(
                    "fmri_pipeline.analysis.bem_generation.check_docker_available",
                    return_value=True,
                ),
                patch(
                    "fmri_pipeline.analysis.bem_generation.subprocess.run",
                    return_value=failed_run,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "QC report:"):
                    generate_bem_model_and_solution(
                        subject=subject,
                        subjects_dir=subjects_dir,
                        fs_license_path=license_path,
                    )

            qc_files = list((subject_dir / "bem" / "qc").glob(f"{subject}_bem_failure_*.json"))
            self.assertEqual(len(qc_files), 1)

            payload = json.loads(qc_files[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["subject"], subject)
            self.assertEqual(payload["returncode"], 1)
            self.assertIn("docker stderr", payload["stderr_tail"])
            self.assertIn("surface_reports", payload)

    def test_bem_generation_timeout_writes_qc_artifact_and_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            subjects_dir = root / "subjects"
            subject = "sub-0002"
            subject_dir = subjects_dir / subject
            subject_dir.mkdir(parents=True, exist_ok=True)
            license_path = root / "license.txt"
            license_path.write_text("license", encoding="utf-8")

            timeout_error = subprocess.TimeoutExpired(cmd="docker run", timeout=3600)

            with (
                patch(
                    "fmri_pipeline.analysis.bem_generation.check_docker_available",
                    return_value=True,
                ),
                patch(
                    "fmri_pipeline.analysis.bem_generation.subprocess.run",
                    side_effect=timeout_error,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "QC report:"):
                    generate_bem_model_and_solution(
                        subject=subject,
                        subjects_dir=subjects_dir,
                        fs_license_path=license_path,
                    )

            qc_files = list((subject_dir / "bem" / "qc").glob(f"{subject}_bem_failure_*.json"))
            self.assertEqual(len(qc_files), 1)

            payload = json.loads(qc_files[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["subject"], subject)
            self.assertIsNone(payload["returncode"])
            self.assertIn("timed out", payload["stderr_tail"])


if __name__ == "__main__":
    unittest.main()
