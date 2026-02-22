import tempfile
import unittest
import importlib.util
import sys
from pathlib import Path
from tests import REPO_ROOT


def _load_trial_table_module():
    module_path = REPO_ROOT / "eeg_pipeline" / "utils" / "data" / "trial_table.py"
    spec = importlib.util.spec_from_file_location("test_trial_table_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestTrialTablePathResolution(unittest.TestCase):
    def test_find_trial_table_path_uses_canonical_all_for_multiple_features(self):
        module = _load_trial_table_module()

        root = Path(tempfile.mkdtemp())
        canonical_path = root / "trial_table" / "all" / "trials_all.parquet"
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        canonical_path.write_bytes(b"PAR1")

        found = module.find_trial_table_path(root, feature_files=["power", "connectivity"])
        self.assertEqual(found, canonical_path)

    def test_find_trial_table_path_raises_on_ambiguous_unfiltered_selection(self):
        module = _load_trial_table_module()

        root = Path(tempfile.mkdtemp())
        path_a = root / "trial_table" / "power" / "trials_power.parquet"
        path_b = root / "trial_table" / "aperiodic" / "trials_aperiodic.parquet"
        path_a.parent.mkdir(parents=True, exist_ok=True)
        path_b.parent.mkdir(parents=True, exist_ok=True)
        path_a.write_bytes(b"PAR1")
        path_b.write_bytes(b"PAR1")

        with self.assertRaises(ValueError):
            module.find_trial_table_path(root, feature_files=None)

    def test_find_trial_table_path_prefers_canonical_all_when_legacy_coexists(self):
        module = _load_trial_table_module()

        root = Path(tempfile.mkdtemp())
        canonical = root / "trial_table" / "all" / "trials_all.parquet"
        legacy = root / "trial_table" / "all" / "trials.parquet"
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_bytes(b"PAR1")
        legacy.write_bytes(b"PAR1")

        found = module.find_trial_table_path(root, feature_files=None)
        self.assertEqual(found, canonical)

    def test_find_trial_table_path_ignores_unsuffixed_legacy_file(self):
        module = _load_trial_table_module()

        root = Path(tempfile.mkdtemp())
        legacy = root / "trial_table" / "all" / "trials.parquet"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_bytes(b"PAR1")

        found = module.find_trial_table_path(root, feature_files=None)
        self.assertIsNone(found)

    def test_find_trial_table_path_does_not_use_joined_multi_feature_legacy_name(self):
        module = _load_trial_table_module()

        root = Path(tempfile.mkdtemp())
        legacy = (
            root
            / "trial_table"
            / "connectivity_power"
            / "trials_connectivity_power.parquet"
        )
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_bytes(b"PAR1")

        found = module.find_trial_table_path(root, feature_files=["power", "connectivity"])
        self.assertIsNone(found)
