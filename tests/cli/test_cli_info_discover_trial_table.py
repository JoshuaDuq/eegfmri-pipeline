import tempfile
import unittest
import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
from tests import REPO_ROOT


def _load_discovery_function():
    base_path = REPO_ROOT / "eeg_pipeline" / "cli" / "commands" / "base.py"
    spec = importlib.util.spec_from_file_location("test_base_module", base_path)
    assert spec is not None and spec.loader is not None

    fake_constants = types.ModuleType("eeg_pipeline.pipelines.constants")
    fake_constants.BEHAVIOR_COMPUTATIONS = []
    fake_constants.BEHAVIOR_VISUALIZE_CATEGORIES = []
    fake_constants.FEATURE_VISUALIZE_CATEGORIES = []
    fake_constants.FREQUENCY_BANDS = []

    sentinel = object()
    original_modules = {
        key: sys.modules.get(key, sentinel)
        for key in (
            "eeg_pipeline",
            "eeg_pipeline.pipelines",
            "eeg_pipeline.pipelines.constants",
            "eeg_pipeline.utils",
            "eeg_pipeline.utils.data",
            "eeg_pipeline.utils.data.trial_table",
        )
    }
    fake_trial_table = types.ModuleType("eeg_pipeline.utils.data.trial_table")
    fake_utils_pkg = types.ModuleType("eeg_pipeline.utils")
    fake_utils_pkg.__path__ = []  # type: ignore[attr-defined]
    fake_data_pkg = types.ModuleType("eeg_pipeline.utils.data")
    fake_data_pkg.__path__ = []  # type: ignore[attr-defined]

    def _discover_trial_table_candidates(stats_dir):
        stats_path = Path(stats_dir)
        candidates = []
        candidates.extend(sorted(stats_path.glob("trial_table*/*/trials_*.parquet")))
        candidates.extend(sorted(stats_path.glob("trial_table*/*/trials_*.tsv")))
        return candidates

    def _select_preferred_trial_tables(candidates):
        grouped = {}
        for path in candidates:
            key = (str(Path(path).parent), Path(path).stem)
            grouped.setdefault(key, []).append(Path(path))
        selected = []
        for key in sorted(grouped.keys()):
            options = grouped[key]
            parquet = [p for p in options if p.suffix == ".parquet"]
            selected.append(parquet[0] if parquet else sorted(options)[0])
        return selected

    fake_trial_table.discover_trial_table_candidates = _discover_trial_table_candidates
    fake_trial_table.select_preferred_trial_tables = _select_preferred_trial_tables
    fake_data_pkg.trial_table = fake_trial_table

    def _install_fakes() -> None:
        sys.modules.setdefault("eeg_pipeline", types.ModuleType("eeg_pipeline"))
        sys.modules.setdefault("eeg_pipeline.pipelines", types.ModuleType("eeg_pipeline.pipelines"))
        sys.modules["eeg_pipeline.pipelines.constants"] = fake_constants
        sys.modules["eeg_pipeline.utils"] = fake_utils_pkg
        sys.modules["eeg_pipeline.utils.data"] = fake_data_pkg
        sys.modules["eeg_pipeline.utils.data.trial_table"] = fake_trial_table

    try:
        _install_fakes()
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for key, original in original_modules.items():
            if original is sentinel:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = original

    def _wrapped_discover_trial_table_columns(*args, **kwargs):
        runtime_originals = {
            key: sys.modules.get(key, sentinel)
            for key in (
                "eeg_pipeline",
                "eeg_pipeline.pipelines",
                "eeg_pipeline.pipelines.constants",
                "eeg_pipeline.utils",
                "eeg_pipeline.utils.data",
                "eeg_pipeline.utils.data.trial_table",
            )
        }
        try:
            _install_fakes()
            return module.discover_trial_table_columns(*args, **kwargs)
        finally:
            for key, original in runtime_originals.items():
                if original is sentinel:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = original

    return _wrapped_discover_trial_table_columns


class TestDiscoverTrialTableColumns(unittest.TestCase):
    def test_discovers_columns_across_multiple_trial_tables(self):
        discover_trial_table_columns = _load_discovery_function()
        deriv_root = Path(tempfile.mkdtemp())
        stats_dir = deriv_root / "sub-0000" / "eeg" / "stats" / "trial_table"
        power_dir = stats_dir / "power"
        aperiodic_dir = stats_dir / "aperiodic"
        power_dir.mkdir(parents=True, exist_ok=True)
        aperiodic_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            {
                "stimulus_temp": [45.0, 46.0],
                "power_active_alpha_global_logratio": [0.1, 0.2],
            }
        ).to_csv(power_dir / "trials_power.tsv", sep="\t", index=False)
        pd.DataFrame(
            {
                "stimulus_temp": [45.0, 47.0],
                "aperiodic_active_global_offset_mean": [0.3, 0.4],
            }
        ).to_csv(aperiodic_dir / "trials_aperiodic.tsv", sep="\t", index=False)

        result = discover_trial_table_columns(deriv_root, subject="0000")

        self.assertEqual(result["source"], "trial_table")
        self.assertIn(
            "power_active_alpha_global_logratio",
            result["columns"],
        )
        self.assertIn(
            "aperiodic_active_global_offset_mean",
            result["columns"],
        )
        self.assertIn("45.0", result["values"].get("stimulus_temp", []))
