from __future__ import annotations

from pathlib import Path
import json
import unittest


ROOT = Path(__file__).resolve().parents[2]
PLOTTING_PARSER_PATH = (
    ROOT / "eeg_pipeline" / "cli" / "commands" / "plotting_parser.py"
)
PLOTTING_OVERRIDES_PATH = (
    ROOT / "eeg_pipeline" / "cli" / "commands" / "plotting_config_overrides.py"
)
PLOT_CATALOG_PATH = ROOT / "eeg_pipeline" / "plotting" / "plot_catalog.json"


class TestCliPlottingConnectivityOverrides(unittest.TestCase):
    def test_plotting_parser_exposes_connectivity_network_top_fraction_flag(self) -> None:
        source = PLOTTING_PARSER_PATH.read_text()

        self.assertIn("--connectivity-network-top-fraction", source)
        self.assertIn("--tfr-topomap-window-size-ms", source)
        self.assertIn("--tfr-topomap-window-count", source)
        self.assertIn("--tfr-topomap-label-x-position", source)
        self.assertIn("--tfr-topomap-title-pad", source)

    def test_plotting_config_overrides_apply_connectivity_network_top_fraction(self) -> None:
        source = PLOTTING_OVERRIDES_PATH.read_text()

        self.assertIn(
            'if _get_arg_value(args, "connectivity_network_top_fraction") is not None:',
            source,
        )
        self.assertIn(
            '_apply_config_override(config, "plotting.plots.features.connectivity.network_top_fraction", '
            "float(args.connectivity_network_top_fraction))",
            source,
        )
        self.assertIn(
            '"time_frequency_analysis.topomap.temporal.window_size_ms"',
            source,
        )
        self.assertIn(
            '"plotting.plots.tfr.topomap.label_x_position"',
            source,
        )

    def test_plot_catalog_marks_rest_compatible_connectivity_plots(self) -> None:
        payload = json.loads(PLOT_CATALOG_PATH.read_text(encoding="utf-8"))
        plots = {entry["id"]: entry for entry in payload["plots"]}

        self.assertEqual(plots["connectivity_circle"]["rest_compatibility"], "compatible")
        self.assertEqual(plots["connectivity_by_condition"]["rest_compatibility"], "task_only")
        self.assertEqual(
            plots["connectivity_circle_condition"]["rest_compatibility"], "task_only"
        )
        self.assertEqual(plots["connectivity_heatmap"]["rest_compatibility"], "compatible")
        self.assertEqual(plots["connectivity_network"]["rest_compatibility"], "compatible")

    def test_plot_catalog_requires_events_for_power_spectral_density(self) -> None:
        payload = json.loads(PLOT_CATALOG_PATH.read_text(encoding="utf-8"))
        plots = {entry["id"]: entry for entry in payload["plots"]}

        self.assertIn("events.tsv", plots["power_spectral_density"]["required_files"])


if __name__ == "__main__":
    unittest.main()
