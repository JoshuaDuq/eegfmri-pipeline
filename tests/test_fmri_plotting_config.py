import unittest


class TestFmriPlottingConfig(unittest.TestCase):
    def test_defaults_validate(self):
        from fmri_pipeline.analysis.plotting_config import FmriPlottingConfig

        cfg = FmriPlottingConfig()
        cfg.validate()  # disabled by default => no-op

    def test_enabled_requires_formats(self):
        from fmri_pipeline.analysis.plotting_config import FmriPlottingConfig

        cfg = FmriPlottingConfig(enabled=True, formats=())
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_default_threshold_is_2p3(self):
        from fmri_pipeline.analysis.plotting_config import FmriPlottingConfig

        cfg = FmriPlottingConfig(enabled=True, formats=("png",))
        self.assertAlmostEqual(cfg.z_threshold, 2.3)

    def test_build_from_args_normalizes(self):
        from fmri_pipeline.analysis.plotting_config import build_fmri_plotting_config_from_args

        cfg = build_fmri_plotting_config_from_args(
            enabled=True,
            formats=[" PNG ", "png", "Svg"],
            space="BoTh",
            plot_types=["Slices", "glass", "glass"],
        )
        self.assertEqual(cfg.space, "both")
        self.assertEqual(list(cfg.formats), ["png", "svg"])
        self.assertEqual(list(cfg.plot_types), ["slices", "glass"])
        self.assertTrue(cfg.include_carpet_qc)
        self.assertTrue(cfg.include_signatures)

    def test_fdr_requires_valid_q(self):
        from fmri_pipeline.analysis.plotting_config import build_fmri_plotting_config_from_args

        with self.assertRaises(ValueError):
            build_fmri_plotting_config_from_args(enabled=True, threshold_mode="fdr", fdr_q=0)

    def test_manual_vmax_requires_value(self):
        from fmri_pipeline.analysis.plotting_config import build_fmri_plotting_config_from_args

        with self.assertRaises(ValueError):
            build_fmri_plotting_config_from_args(enabled=True, vmax_mode="manual", vmax_manual=None)
