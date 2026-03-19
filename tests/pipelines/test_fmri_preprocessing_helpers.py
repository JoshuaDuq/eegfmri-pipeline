from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class TestFmriPreprocessingHelpers(unittest.TestCase):
    def _import_module(self):
        fake_base = types.ModuleType("eeg_pipeline.pipelines.base")
        fake_base.PipelineBase = type("PipelineBase", (), {})

        sys.modules.pop("fmri_pipeline.pipelines.fmri_preprocessing", None)

        with patch.dict(
            sys.modules,
            {"eeg_pipeline.pipelines.base": fake_base},
        ):
            return importlib.import_module("fmri_pipeline.pipelines.fmri_preprocessing")

    def test_is_macos_metadata_path_detects_metadata_files(self) -> None:
        module = self._import_module()

        self.assertTrue(module._is_macos_metadata_path(Path("._file")))
        self.assertTrue(module._is_macos_metadata_path(Path(".DS_Store")))
        self.assertFalse(module._is_macos_metadata_path(Path("sub-01.nii.gz")))

    def test_dataset_has_macos_metadata_detects_any_metadata_file(self) -> None:
        module = self._import_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "sub-01").mkdir()
            (root / "sub-01" / ".DS_Store").write_text("", encoding="utf-8")

            self.assertTrue(module._dataset_has_macos_metadata(root))

    def test_create_sanitized_bids_view_skips_metadata_files(self) -> None:
        module = self._import_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            bids_dir = Path(tmpdir) / "bids"
            (bids_dir / "sub-01" / "func").mkdir(parents=True)
            (bids_dir / "sub-01" / "func" / "sub-01_bold.nii.gz").write_text(
                "nii", encoding="utf-8"
            )
            (bids_dir / "._sub-01").write_text("", encoding="utf-8")
            (bids_dir / ".DS_Store").write_text("", encoding="utf-8")

            sanitized_root, temp_dir, skipped_files = module._create_sanitized_bids_view(
                bids_dir,
                "/mount/bids",
            )

            self.assertEqual(skipped_files, 2)
            self.assertTrue((sanitized_root / "sub-01" / "func").is_dir())
            self.assertTrue((sanitized_root / "sub-01" / "func" / "sub-01_bold.nii.gz").is_symlink())
            self.assertEqual(
                (sanitized_root / "sub-01" / "func" / "sub-01_bold.nii.gz").readlink(),
                Path("/mount/bids/sub-01/func/sub-01_bold.nii.gz"),
            )
            temp_dir.cleanup()

    def test_resolve_bids_mount_root_uses_sanitized_view_when_metadata_exists(self) -> None:
        module = self._import_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            bids_dir = Path(tmpdir) / "bids"
            (bids_dir / "sub-01").mkdir(parents=True)
            (bids_dir / "sub-01" / ".DS_Store").write_text("", encoding="utf-8")
            logger = Mock()

            resolved_root, temp_dir = module._resolve_bids_mount_root(bids_dir, logger)

            self.assertNotEqual(resolved_root, bids_dir)
            self.assertIsNotNone(temp_dir)
            logger.warning.assert_called_once()
            temp_dir.cleanup()

    def test_resolve_bids_mount_root_returns_original_tree_when_clean(self) -> None:
        module = self._import_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            bids_dir = Path(tmpdir) / "bids"
            (bids_dir / "sub-01").mkdir(parents=True)
            logger = Mock()

            resolved_root, temp_dir = module._resolve_bids_mount_root(bids_dir, logger)

            self.assertEqual(resolved_root, bids_dir)
            self.assertIsNone(temp_dir)
            logger.warning.assert_not_called()
