from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
PLOTTING_PARSER_PATH = (
    ROOT / "eeg_pipeline" / "cli" / "commands" / "plotting_parser.py"
)
PLOTTING_OVERRIDES_PATH = (
    ROOT / "eeg_pipeline" / "cli" / "commands" / "plotting_config_overrides.py"
)


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


if __name__ == "__main__":
    unittest.main()
