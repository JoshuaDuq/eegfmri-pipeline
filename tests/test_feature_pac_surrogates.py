from __future__ import annotations

import unittest

import numpy as np

from eeg_pipeline.analysis.features.phase import _compute_pac_surrogates


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


if __name__ == "__main__":
    unittest.main()
