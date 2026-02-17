from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from eeg_pipeline.analysis.features.api import extract_precomputed_features, _extract_pac_features
from eeg_pipeline.types import PrecomputedData, TimeWindows
from tests.pipelines_test_utils import DotConfig


class TestPrecomputedMicrostatesContext(unittest.TestCase):
    def test_with_windows_preserves_metadata_condition_labels_and_train_mask(self):
        n_epochs = 3
        n_times = 8
        times = np.linspace(0.0, 0.7, n_times, endpoint=True)
        windows_a = TimeWindows(
            masks={"a": np.ones(n_times, dtype=bool)},
            ranges={"a": (0.0, 0.7)},
            times=times,
            name="a",
        )
        windows_b = TimeWindows(
            masks={"b": np.ones(n_times, dtype=bool)},
            ranges={"b": (0.0, 0.7)},
            times=times,
            name="b",
        )
        metadata = pd.DataFrame({"condition": ["pain", "pain", "warm"]})
        condition_labels = np.array(["pain", "pain", "warm"], dtype=object)
        train_mask = np.array([True, True, False], dtype=bool)

        precomputed = PrecomputedData(
            data=np.zeros((n_epochs, 2, n_times), dtype=float),
            times=times,
            sfreq=100.0,
            ch_names=["Cz", "Pz"],
            picks=np.array([0, 1], dtype=int),
            windows=windows_a,
            metadata=metadata,
            condition_labels=condition_labels,
            train_mask=train_mask,
        )

        updated = precomputed.with_windows(windows_b)

        self.assertIsNot(updated, precomputed)
        self.assertIs(updated.windows, windows_b)
        self.assertIs(updated.metadata, metadata)
        np.testing.assert_array_equal(updated.condition_labels, condition_labels)
        np.testing.assert_array_equal(updated.train_mask, train_mask)

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

    def test_extract_pac_features_recomputes_when_transform_mismatch(self):
        tfr_in = SimpleNamespace(times=np.array([0.0, 0.1, 0.2], dtype=float))
        tfr_recomputed = SimpleNamespace(times=np.array([0.0, 0.1, 0.2], dtype=float))
        pac_trials_df = pd.DataFrame({"pac_active_theta_gamma_global_mvl": [0.1, 0.2, 0.3]})
        ctx = SimpleNamespace(
            config=DotConfig(
                {
                    "feature_engineering": {
                        "pac": {"source": "raw"},
                        "spatial_transform_per_family": {"pac": "none"},
                    }
                }
            ),
            logger=logging.getLogger("test-pac-transform-mismatch"),
            epochs=SimpleNamespace(info={"sfreq": 100.0}),
            windows=None,
            name="active",
            spatial_modes=["global"],
            analysis_mode="group_stats",
            train_mask=None,
            tfr_complex_transform="csd",
        )

        with patch(
            "eeg_pipeline.analysis.features.api.compute_complex_tfr",
            return_value=tfr_recomputed,
        ) as mock_compute_complex, patch(
            "eeg_pipeline.analysis.features.api.get_tfr_config",
            return_value=(4.0, 40.0, 6, None, None),
        ), patch(
            "eeg_pipeline.analysis.features.api.compute_pac_comodulograms",
            return_value=(None, None, None, pac_trials_df, None),
        ):
            _pac_df, _phase_freqs, _amp_freqs, out_trials, _out_time = _extract_pac_features(
                ctx, precomputed_data=None, tfr_complex=tfr_in
            )

        mock_compute_complex.assert_called_once()
        self.assertIs(out_trials, pac_trials_df)


if __name__ == "__main__":
    unittest.main()
