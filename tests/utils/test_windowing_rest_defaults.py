from __future__ import annotations

import logging
import unittest

import numpy as np

from eeg_pipeline.utils.analysis.windowing import TimeWindowSpec, time_windows_from_spec


class TestWindowingRestDefaults(unittest.TestCase):
    def test_time_window_spec_defaults_to_full_epoch_analysis_window(self) -> None:
        times = np.array([0.0, 0.5, 1.0, 1.5], dtype=float)

        spec = TimeWindowSpec(
            times=times,
            config={},
            sampling_rate=2.0,
            logger=logging.getLogger("windowing-default"),
        )
        windows = time_windows_from_spec(spec, logger=logging.getLogger("windowing-default"), strict=False)

        self.assertIn("analysis", windows.masks)
        self.assertTrue(np.array_equal(windows.masks["analysis"], np.ones(times.shape, dtype=bool)))
        self.assertTrue(np.array_equal(windows.active_mask, np.ones(times.shape, dtype=bool)))
        self.assertEqual(windows.active_range, (0.0, 2.0))
        self.assertTrue(windows.valid)

    def test_named_default_window_uses_requested_name(self) -> None:
        times = np.array([-0.5, 0.0, 0.5], dtype=float)

        spec = TimeWindowSpec(
            times=times,
            config={},
            sampling_rate=2.0,
            logger=logging.getLogger("windowing-named-default"),
            name="active",
        )
        windows = time_windows_from_spec(spec, logger=logging.getLogger("windowing-named-default"), strict=False)

        self.assertIn("active", windows.masks)
        self.assertTrue(np.array_equal(windows.masks["active"], np.ones(times.shape, dtype=bool)))
        self.assertEqual(windows.active_range, (-0.5, 1.0))


if __name__ == "__main__":
    unittest.main()
