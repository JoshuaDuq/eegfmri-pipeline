import json
import unittest
from pathlib import Path


class TestPlotCatalogExcludesMachineLearning(unittest.TestCase):
    def test_plot_catalog_has_no_machine_learning_group_or_plots(self):
        catalog_path = (
            Path(__file__).resolve().parents[1]
            / "eeg_pipeline"
            / "plotting"
            / "plot_catalog.json"
        )
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))

        group_keys = {str(group.get("key")) for group in payload.get("groups", [])}
        plot_groups = {str(plot.get("group")) for plot in payload.get("plots", [])}
        plot_ids = {str(plot.get("id")) for plot in payload.get("plots", [])}

        self.assertNotIn("machine_learning", group_keys)
        self.assertNotIn("machine_learning", plot_groups)
        self.assertFalse(any(plot_id.startswith("ml_") for plot_id in plot_ids))

    def test_ml_plotting_implementation_files_removed(self):
        repo_root = Path(__file__).resolve().parents[1]
        self.assertFalse(
            (repo_root / "eeg_pipeline" / "plotting" / "orchestration" / "machine_learning.py").exists()
        )
        self.assertFalse(
            (repo_root / "eeg_pipeline" / "plotting" / "machine_learning").exists()
        )
