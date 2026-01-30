import unittest
from pathlib import Path


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

