from __future__ import annotations

import unittest

import numpy as np

from eeg_pipeline.analysis.features.phase import (
    _compute_pac_surrogates,
    _rng_from_seed,
    _resolve_pac_surrogate_context,
)


class TestPacSurrogates(unittest.TestCase):
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

    def test_trial_ml_safe_without_train_mask_falls_back_to_circular_shift(self):
        method, donor_idx = _resolve_pac_surrogate_context(
            {"surrogate_method": "trial_shuffle"},
            n_epochs=5,
            analysis_mode="trial_ml_safe",
            train_mask=None,
            logger=None,
        )
        self.assertEqual(method, "circular_shift")
        self.assertIsNone(donor_idx)

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

    def test_seed_zero_is_deterministic(self):
        rng_a = _rng_from_seed(0)
        rng_b = _rng_from_seed(0)
        seq_a = rng_a.integers(0, 10_000, size=16)
        seq_b = rng_b.integers(0, 10_000, size=16)
        np.testing.assert_array_equal(seq_a, seq_b)


if __name__ == "__main__":
    unittest.main()
