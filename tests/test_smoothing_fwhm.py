import unittest


class TestSmoothingFWHM(unittest.TestCase):
    def test_normalize_smoothing_fwhm(self):
        from fmri_pipeline.analysis.smoothing import normalize_smoothing_fwhm

        self.assertIsNone(normalize_smoothing_fwhm(None))
        self.assertIsNone(normalize_smoothing_fwhm(0))
        self.assertIsNone(normalize_smoothing_fwhm(0.0))
        self.assertIsNone(normalize_smoothing_fwhm(-1.0))
        self.assertEqual(normalize_smoothing_fwhm(5), 5.0)
        self.assertEqual(normalize_smoothing_fwhm(5.5), 5.5)

