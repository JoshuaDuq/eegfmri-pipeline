import logging
import unittest
from types import SimpleNamespace

import mne
import numpy as np

from tests.pipelines_test_utils import DotConfig


class _WindowStub:
    def __init__(self, mask):
        self.masks = {"active": mask}
        self.name = None

    def get_mask(self, name):
        return self.masks.get(name)


class TestMicrostateFeatures(unittest.TestCase):
    def _build_epochs(self):
        rng = np.random.default_rng(7)
        ch_names = ["Fp1", "Fp2", "C3", "C4", "P3", "P4"]
        sfreq = 100.0
        n_times = 80

        templates = np.array(
            [
                [1.0, -1.0, 0.5, -0.5, 0.0, 0.0],    # A
                [0.0, 0.0, 1.0, -1.0, 0.6, -0.6],    # B
                [1.0, 1.0, 0.0, 0.0, -1.0, -1.0],    # C
                [1.0, 0.8, 0.6, 0.6, 0.2, 0.2],      # D
            ],
            dtype=float,
        )

        seq1 = np.repeat(np.array([0, 1, 2, 3]), n_times // 4)
        seq2 = np.repeat(np.array([3, 2, 1, 0]), n_times // 4)

        data = np.zeros((2, len(ch_names), n_times), dtype=float)
        for ti, seq in enumerate([seq1, seq2]):
            for t, state in enumerate(seq):
                data[ti, :, t] = templates[state] + 0.02 * rng.standard_normal(len(ch_names))

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        epochs = mne.EpochsArray(data, info=info, tmin=0.0, verbose=False)
        epochs.set_montage("standard_1020")
        return epochs, templates, ch_names

    def test_extract_microstate_features_with_fixed_templates(self):
        from eeg_pipeline.analysis.features.microstates import extract_microstate_features

        epochs, templates, ch_names = self._build_epochs()
        mask = np.ones(epochs.get_data().shape[-1], dtype=bool)
        ctx = SimpleNamespace(
            epochs=epochs,
            windows=_WindowStub(mask),
            name="active",
            config=DotConfig(
                {
                    "feature_engineering": {
                        "microstates": {
                            "n_states": 4,
                            "min_duration_ms": 0.0,
                            "min_peak_distance_ms": 5.0,
                            "max_gfp_peaks_per_epoch": 200,
                        }
                    }
                }
            ),
            logger=logging.getLogger("microstate-test"),
            fixed_templates=templates,
            fixed_template_ch_names=ch_names,
        )

        df, cols = extract_microstate_features(ctx)

        self.assertEqual(len(df), 2)
        self.assertEqual(len(cols), len(df.columns))
        self.assertIn("microstates_active_broadband_global_coverage_a", df.columns)
        self.assertIn("microstates_active_broadband_global_duration_ms_a", df.columns)
        self.assertIn("microstates_active_broadband_global_occurrence_hz_a", df.columns)
        self.assertIn("microstates_active_broadband_global_trans_a_to_b_prob", df.columns)

        cov_cols = [
            "microstates_active_broadband_global_coverage_a",
            "microstates_active_broadband_global_coverage_b",
            "microstates_active_broadband_global_coverage_c",
            "microstates_active_broadband_global_coverage_d",
        ]
        coverage_sum = df[cov_cols].sum(axis=1).to_numpy()
        self.assertTrue(np.allclose(coverage_sum, 1.0, atol=0.1))
        self.assertGreater(df["microstates_active_broadband_global_trans_a_to_b_prob"].iloc[0], 0.9)

    def test_microstates_category_registered(self):
        from eeg_pipeline.pipelines import constants

        self.assertIn("microstates", constants.FEATURE_CATEGORIES)
