from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from eeg_pipeline.analysis.features.api import extract_precomputed_features
from eeg_pipeline.types import PrecomputedData, TimeWindows
from tests.pipelines_test_utils import DotConfig


class TestPrecomputedMicrostatesContext(unittest.TestCase):
    def test_microstates_receives_windows_train_mask_and_fixed_templates(self):
        n_epochs = 3
        n_times = 12
        epochs = SimpleNamespace()
        windows = TimeWindows(
            masks={"active": np.ones(n_times, dtype=bool)},
            ranges={"active": (0.0, 1.0)},
            times=np.linspace(0.0, 1.0, n_times, endpoint=False),
            name="active",
        )
        precomputed = PrecomputedData(
            data=np.zeros((n_epochs, 2, n_times), dtype=float),
            times=np.linspace(0.0, 1.0, n_times, endpoint=False),
            sfreq=100.0,
            ch_names=["Cz", "Pz"],
            picks=np.array([0, 1], dtype=int),
            windows=windows,
            train_mask=np.array([True, True, False], dtype=bool),
        )
        precomputed.fixed_templates = np.ones((4, 2), dtype=float)
        precomputed.fixed_template_ch_names = ["Cz", "Pz"]
        precomputed.fixed_template_labels = ["A", "B", "C", "D"]

        captured = {"ctx": None}

        def _fake_extract_microstate_features(ctx):
            captured["ctx"] = ctx
            return (
                pd.DataFrame({"microstates_active_broadband_global_coverage_a": [0.5] * n_epochs}),
                ["microstates_active_broadband_global_coverage_a"],
            )

        with patch(
            "eeg_pipeline.analysis.features.api.extract_microstate_features",
            new=_fake_extract_microstate_features,
        ):
            result = extract_precomputed_features(
                epochs=epochs,
                bands=["alpha"],
                config=DotConfig({}),
                logger=logging.getLogger("test-precomputed-microstate-context"),
                feature_groups=["microstates"],
                precomputed=precomputed,
            )

        self.assertIn("microstates", result.features)
        self.assertIsNotNone(captured["ctx"])
        self.assertIs(captured["ctx"].windows, precomputed.windows)
        self.assertEqual(captured["ctx"].name, "active")
        np.testing.assert_array_equal(captured["ctx"].train_mask, precomputed.train_mask)
        np.testing.assert_array_equal(captured["ctx"].fixed_templates, precomputed.fixed_templates)
        self.assertEqual(captured["ctx"].fixed_template_ch_names, precomputed.fixed_template_ch_names)
        self.assertEqual(captured["ctx"].fixed_template_labels, precomputed.fixed_template_labels)


if __name__ == "__main__":
    unittest.main()
