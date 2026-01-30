import unittest


class TestTrialSignaturesContrastVector(unittest.TestCase):
    def test_contrast_vector_is_numpy_array(self):
        import numpy as np  # type: ignore

        from fmri_pipeline.analysis.trial_signatures import _contrast_vector_for_column

        vec = _contrast_vector_for_column(["a", "b", "c"], "b")
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.shape, (3,))
        self.assertEqual(vec.tolist(), [0.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()

