import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd
from tests import REPO_ROOT


def _load_dose_response_module():
    module_path = REPO_ROOT / "eeg_pipeline" / "plotting" / "behavioral" / "dose_response.py"
    spec = importlib.util.spec_from_file_location("test_dose_response_module", module_path)
    assert spec is not None and spec.loader is not None

    sentinel = object()
    module_keys = (
        "eeg_pipeline",
        "eeg_pipeline.analysis",
        "eeg_pipeline.analysis.behavior",
        "eeg_pipeline.analysis.behavior.orchestration",
        "eeg_pipeline.domain",
        "eeg_pipeline.domain.features",
        "eeg_pipeline.domain.features.naming",
        "eeg_pipeline.infra",
        "eeg_pipeline.infra.paths",
        "eeg_pipeline.infra.tsv",
        "eeg_pipeline.plotting",
        "eeg_pipeline.plotting.config",
        "eeg_pipeline.plotting.io",
        "eeg_pipeline.plotting.io.figures",
        "eeg_pipeline.utils",
        "eeg_pipeline.utils.analysis",
        "eeg_pipeline.utils.analysis.tfr",
        "eeg_pipeline.utils.config",
        "eeg_pipeline.utils.config.loader",
        "eeg_pipeline.utils.data",
        "eeg_pipeline.utils.data.alignment",
        "eeg_pipeline.utils.data.epochs",
        "eeg_pipeline.utils.data.manipulation",
    )
    original_modules = {k: sys.modules.get(k, sentinel) for k in module_keys}
    original_modules[spec.name] = sys.modules.get(spec.name, sentinel)

    fake_orchestration = types.ModuleType("eeg_pipeline.analysis.behavior.orchestration")
    fake_orchestration.CATEGORY_PREFIX_MAP = {"power": "power_", "aperiodic": "aperiodic_"}

    def _raise_multiple(_stats_dir, feature_files=None):
        raise ValueError("Multiple trial table files found in stats_dir")

    fake_orchestration._find_trial_table_path = _raise_multiple

    fake_naming = types.ModuleType("eeg_pipeline.domain.features.naming")
    fake_naming.NamingSchema = object

    fake_paths = types.ModuleType("eeg_pipeline.infra.paths")
    fake_paths.deriv_plots_path = lambda *args, **kwargs: Path(tempfile.mkdtemp())
    fake_paths.deriv_stats_path = lambda deriv_root, subject: Path(deriv_root) / f"sub-{subject}" / "eeg" / "stats"
    fake_paths.ensure_dir = lambda p: Path(p).mkdir(parents=True, exist_ok=True)

    fake_tsv = types.ModuleType("eeg_pipeline.infra.tsv")
    fake_tsv.read_table = lambda p: pd.read_csv(p, sep="\t")
    fake_tsv.write_tsv = lambda *args, **kwargs: None

    fake_plot_cfg = types.ModuleType("eeg_pipeline.plotting.config")
    fake_plot_cfg.get_plot_config = lambda _cfg: None

    fake_fig = types.ModuleType("eeg_pipeline.plotting.io.figures")
    fake_fig.save_fig = lambda *args, **kwargs: None

    fake_tfr = types.ModuleType("eeg_pipeline.utils.analysis.tfr")
    fake_tfr.get_rois = lambda _cfg: {}

    fake_loader = types.ModuleType("eeg_pipeline.utils.config.loader")
    fake_loader.get_config_value = lambda _cfg, _k, default=None: default
    fake_loader.get_frequency_band_names = lambda _cfg: ["alpha"]

    fake_align = types.ModuleType("eeg_pipeline.utils.data.alignment")
    fake_align.get_aligned_events = lambda *args, **kwargs: pd.DataFrame()

    fake_epochs = types.ModuleType("eeg_pipeline.utils.data.epochs")
    fake_epochs.load_epochs_for_analysis = lambda *args, **kwargs: None

    fake_manip = types.ModuleType("eeg_pipeline.utils.data.manipulation")
    fake_manip.find_column = lambda _df, _cands: None

    try:
        sys.modules.setdefault("eeg_pipeline", types.ModuleType("eeg_pipeline"))
        sys.modules.setdefault("eeg_pipeline.analysis", types.ModuleType("eeg_pipeline.analysis"))
        sys.modules.setdefault("eeg_pipeline.analysis.behavior", types.ModuleType("eeg_pipeline.analysis.behavior"))
        sys.modules["eeg_pipeline.analysis.behavior.orchestration"] = fake_orchestration
        sys.modules.setdefault("eeg_pipeline.domain", types.ModuleType("eeg_pipeline.domain"))
        sys.modules.setdefault("eeg_pipeline.domain.features", types.ModuleType("eeg_pipeline.domain.features"))
        sys.modules["eeg_pipeline.domain.features.naming"] = fake_naming
        sys.modules.setdefault("eeg_pipeline.infra", types.ModuleType("eeg_pipeline.infra"))
        sys.modules["eeg_pipeline.infra.paths"] = fake_paths
        sys.modules["eeg_pipeline.infra.tsv"] = fake_tsv
        sys.modules.setdefault("eeg_pipeline.plotting", types.ModuleType("eeg_pipeline.plotting"))
        sys.modules["eeg_pipeline.plotting.config"] = fake_plot_cfg
        sys.modules.setdefault("eeg_pipeline.plotting.io", types.ModuleType("eeg_pipeline.plotting.io"))
        sys.modules["eeg_pipeline.plotting.io.figures"] = fake_fig
        sys.modules.setdefault("eeg_pipeline.utils", types.ModuleType("eeg_pipeline.utils"))
        sys.modules.setdefault("eeg_pipeline.utils.analysis", types.ModuleType("eeg_pipeline.utils.analysis"))
        sys.modules["eeg_pipeline.utils.analysis.tfr"] = fake_tfr
        sys.modules.setdefault("eeg_pipeline.utils.config", types.ModuleType("eeg_pipeline.utils.config"))
        sys.modules["eeg_pipeline.utils.config.loader"] = fake_loader
        sys.modules.setdefault("eeg_pipeline.utils.data", types.ModuleType("eeg_pipeline.utils.data"))
        sys.modules["eeg_pipeline.utils.data.alignment"] = fake_align
        sys.modules["eeg_pipeline.utils.data.epochs"] = fake_epochs
        sys.modules["eeg_pipeline.utils.data.manipulation"] = fake_manip

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        for k, v in original_modules.items():
            if v is sentinel:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return module


class TestDoseResponseTrialTableLoading(unittest.TestCase):
    def test_load_trial_table_raises_when_resolver_reports_multiple(self):
        module = _load_dose_response_module()
        deriv_root = Path(tempfile.mkdtemp())

        with self.assertRaises(ValueError):
            module._load_trial_table(deriv_root, "0000", config=types.SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
