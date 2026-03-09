from __future__ import annotations

import unittest

import numpy as np

from eeg_pipeline.analysis.features.phase import (
    _compute_pac_for_channel_band_pair,
    _compute_pac_surrogates,
    _extract_frequency_ranges,
    _extract_pac_config,
    _rng_from_seed,
    _resolve_pac_surrogate_context,
)
from tests.pipelines_test_utils import DotConfig


class TestPacSurrogates(unittest.TestCase):
    def test_pac_without_normalization_is_not_rescaled_by_window_length(self):
        data = np.ones((1, 1, 2, 4), dtype=np.complex128)
        values = _compute_pac_for_channel_band_pair(
            data,
            channel_idx=0,
            phase_freqs=np.array([4.0, 6.0], dtype=float),
            amp_freqs=np.array([30.0, 40.0], dtype=float),
            phase_indices=np.array([0, 1], dtype=int),
            amp_indices=np.array([0, 1], dtype=int),
            phase_band_range=(4.0, 6.0),
            amp_band_range=(30.0, 40.0),
            normalize=False,
            epsilon=1e-12,
            n_times=4,
        )

        self.assertIsNotNone(values)
        np.testing.assert_allclose(values, np.array([1.0], dtype=float), atol=1e-12)

    def test_trial_shuffle_surrogates_use_cross_epoch_amplitudes(self):
        phase = np.array(
            [
                [1.0 + 0.0j, -1.0 + 0.0j, 1.0 + 0.0j, -1.0 + 0.0j],
                [1.0 + 0.0j, -1.0 + 0.0j, 1.0 + 0.0j, -1.0 + 0.0j],
            ],
            dtype=np.complex128,
        )
        amplitudes = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 2.0, 3.0, 4.0],
            ],
            dtype=float,
        )

        rng_shuffle = np.random.default_rng(123)
        surrogates_shuffle = _compute_pac_surrogates(
            phase,
            amplitudes,
            n_surrogates=6,
            normalize=False,
            epsilon=1e-12,
            n_times=4,
            rng=rng_shuffle,
            surrogate_method="trial_shuffle",
        )
        self.assertEqual(surrogates_shuffle.shape, (2, 6))

        rng_shift = np.random.default_rng(123)
        surrogates_shift = _compute_pac_surrogates(
            phase,
            amplitudes,
            n_surrogates=6,
            normalize=False,
            epsilon=1e-12,
            n_times=4,
            rng=rng_shift,
            surrogate_method="circular_shift",
        )
        self.assertTrue(np.allclose(surrogates_shift[0], 0.0, atol=1e-12, equal_nan=True))
        self.assertGreater(
            float(np.nanmean(np.abs(surrogates_shuffle[0]))),
            float(np.nanmean(np.abs(surrogates_shift[0]))),
        )

    def test_trial_shuffle_can_restrict_donor_epochs(self):
        phase = np.ones((3, 4), dtype=np.complex128)
        amplitudes = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],      # epoch 0
                [2.0, 2.0, 2.0, 2.0],      # epoch 1
                [100.0, 100.0, 100.0, 100.0],  # epoch 2 (excluded donor)
            ],
            dtype=float,
        )

        rng = np.random.default_rng(7)
        surrogates = _compute_pac_surrogates(
            phase,
            amplitudes,
            n_surrogates=8,
            normalize=False,
            epsilon=1e-12,
            n_times=4,
            rng=rng,
            surrogate_method="trial_shuffle",
            donor_epoch_indices=np.array([0, 1], dtype=int),
        )

        # Epoch 0 can only receive donor epoch 1 when self-donors are excluded.
        self.assertTrue(np.allclose(surrogates[0], 2.0, atol=1e-12, equal_nan=True))
        # Epoch 1 can only receive donor epoch 0.
        self.assertTrue(np.allclose(surrogates[1], 1.0, atol=1e-12, equal_nan=True))
        # Epoch 2 must never receive donor epoch 2 (value=100) when donor pool is [0,1].
        self.assertTrue(np.nanmax(surrogates[2]) < 10.0)

    def test_trial_ml_safe_without_train_mask_raises(self):
        with self.assertRaisesRegex(ValueError, "train_mask"):
            _resolve_pac_surrogate_context(
                {"surrogate_method": "trial_shuffle"},
                n_epochs=5,
                analysis_mode="trial_ml_safe",
                train_mask=None,
                logger=None,
            )

    def test_trial_ml_safe_with_train_mask_uses_training_donor_pool(self):
        method, donor_idx = _resolve_pac_surrogate_context(
            {"surrogate_method": "trial_shuffle"},
            n_epochs=5,
            analysis_mode="trial_ml_safe",
            train_mask=np.array([True, False, True, False, False], dtype=bool),
            logger=None,
        )
        self.assertEqual(method, "trial_shuffle")
        self.assertIsNotNone(donor_idx)
        np.testing.assert_array_equal(donor_idx, np.array([0, 2], dtype=int))

    def test_trial_ml_safe_with_too_few_training_trials_raises(self):
        with self.assertRaisesRegex(ValueError, "2 training trials"):
            _resolve_pac_surrogate_context(
                {"surrogate_method": "trial_shuffle"},
                n_epochs=5,
                analysis_mode="trial_ml_safe",
                train_mask=np.array([True, False, False, False, False], dtype=bool),
                logger=None,
            )

    def test_invalid_surrogate_method_raises(self):
        with self.assertRaisesRegex(ValueError, "surrogate_method"):
            _extract_pac_config(
                DotConfig(
                    {
                        "feature_engineering": {
                            "pac": {"surrogate_method": "bad_method"}
                        }
                    }
                )
            )

        with self.assertRaisesRegex(ValueError, "surrogate_method"):
            _compute_pac_surrogates(
                np.ones((2, 4), dtype=np.complex128),
                np.ones((2, 4), dtype=float),
                n_surrogates=2,
                normalize=False,
                epsilon=1e-12,
                n_times=4,
                rng=np.random.default_rng(0),
                surrogate_method="bad_method",
            )

    def test_invalid_frequency_ranges_raise(self):
        with self.assertRaisesRegex(ValueError, "phase_range"):
            _extract_frequency_ranges({"phase_range": ["theta"], "amp_range": [30.0, 80.0]})

        with self.assertRaisesRegex(ValueError, "amp_range"):
            _extract_frequency_ranges({"phase_range": [4.0, 8.0], "amp_range": [80.0, 30.0]})

    def test_seed_zero_is_deterministic(self):
        rng_a = _rng_from_seed(0)
        rng_b = _rng_from_seed(0)
        seq_a = rng_a.integers(0, 10_000, size=16)
        seq_b = rng_b.integers(0, 10_000, size=16)
        np.testing.assert_array_equal(seq_a, seq_b)


if __name__ == "__main__":
    unittest.main()
