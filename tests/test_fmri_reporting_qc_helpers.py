import unittest
from pathlib import Path

import numpy as np


class TestFmriReportingQCHelpers(unittest.TestCase):
    def test_carpet_qc_helper_empty_is_safe(self):
        from fmri_pipeline.analysis.plotting_config import FmriPlottingConfig
        from fmri_pipeline.analysis.reporting import generate_carpet_qc_images

        cfg = FmriPlottingConfig(enabled=True, formats=("png",))
        images = generate_carpet_qc_images(contrast_dir=Path("."), cfg=cfg, run_meta={})
        self.assertEqual(images, [])

    def test_tsnr_qc_helper_empty_is_safe(self):
        from fmri_pipeline.analysis.plotting_config import FmriPlottingConfig
        from fmri_pipeline.analysis.reporting import generate_tsnr_qc_images

        cfg = FmriPlottingConfig(enabled=True, formats=("png",))
        images = generate_tsnr_qc_images(contrast_dir=Path("."), cfg=cfg, run_meta={})
        self.assertEqual(images, [])

    def test_vif_from_design_orthogonal_columns_near_one(self):
        from fmri_pipeline.analysis.reporting import _vif_from_design

        n = 100
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, 3))
        vifs = _vif_from_design(X)
        self.assertEqual(vifs.shape, (3,))
        np.testing.assert_allclose(vifs, 1.0, atol=0.5)

    def test_vif_from_design_collinear_column_is_inf(self):
        from fmri_pipeline.analysis.reporting import _vif_from_design

        n = 50
        a = np.linspace(0, 1, n)
        X = np.column_stack([a, 2 * a, np.ones(n)])
        vifs = _vif_from_design(X)
        self.assertTrue(np.any(np.isposinf(vifs)))

