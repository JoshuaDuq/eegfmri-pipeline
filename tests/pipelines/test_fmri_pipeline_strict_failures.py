from __future__ import annotations

import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tests.utils.pipelines_test_utils import (
    DotConfig,
    make_mock_fitted_model,
    make_mock_run_meta,
)


class TestFmriPipelineStrictFailures(unittest.TestCase):
    def test_process_subject_is_unaffected_by_a_broken_reporting_module(self):
        from fmri_pipeline.pipelines.fmri_analysis import FmriAnalysisPipeline

        tmp = Path(tempfile.mkdtemp())
        bids_root = tmp / "bids"
        bids_root.mkdir(parents=True, exist_ok=True)

        pipeline = object.__new__(FmriAnalysisPipeline)
        pipeline.config = DotConfig({"paths": {"bids_fmri_root": str(bids_root)}})
        pipeline.deriv_root = tmp / "deriv"
        pipeline.deriv_root.mkdir(parents=True, exist_ok=True)
        pipeline.logger = Mock()

        @dataclass
        class ContrastCfg:
            name: str = "pain"
            output_type: str = "z-score"
            resample_to_freesurfer: bool = False
            fmriprep_space: str = "T1w"

        @dataclass
        class PlotCfg:
            enabled: bool = True
            space: str = "native"
            include_effect_size: bool = True
            include_standard_error: bool = True
            include_signatures: bool = False
            threshold_mode: str = "z"
            z_threshold: float = 2.3
            fdr_q: float = 0.05
            cluster_min_voxels: int = 0
            two_sided: bool = True

            def normalized(self):
                return self

        flm = make_mock_fitted_model(runs=1)
        run_meta = make_mock_run_meta(runs=1)

        fake_builder = types.SimpleNamespace(
            build_contrast_from_runs_detailed=lambda **kwargs: (
                "img",
                run_meta,
                SimpleNamespace(flm=flm, mask_img="img"),
                "cond_a",
                None,
            ),
            resample_to_freesurfer=lambda img, fs_dir: img,
            ContrastBuilderConfig=ContrastCfg,
        )
        fake_nib = types.SimpleNamespace(save=lambda img, path: None, load=lambda path: "img")
        fake_plotting = types.SimpleNamespace(FmriPlottingConfig=PlotCfg)
        # Reporting is no longer reachable from the GLM path. This used to be proved
        # with a stub module that exploded on any attribute access; the module has
        # since been deleted outright, so reaching for it would be an ImportError and
        # the guard is structural.

        with patch.dict(
            sys.modules,
            {
                "fmri_pipeline.analysis.contrast_builder": fake_builder,
                "nibabel": fake_nib,
                "fmri_pipeline.analysis.plotting_config": fake_plotting,
            },
        ):
            pipeline.process_subject(
                "0001",
                task="pain",
                contrast_cfg=ContrastCfg(),
                stats_cfg=PlotCfg(),
                dry_run=False,
            )

    def test_process_subject_raises_when_sidecar_write_fails(self):
        from fmri_pipeline.pipelines.fmri_analysis import FmriAnalysisPipeline

        tmp = Path(tempfile.mkdtemp())
        bids_root = tmp / "bids"
        bids_root.mkdir(parents=True, exist_ok=True)

        pipeline = object.__new__(FmriAnalysisPipeline)
        pipeline.config = DotConfig({"paths": {"bids_fmri_root": str(bids_root)}})
        pipeline.deriv_root = tmp / "deriv"
        pipeline.deriv_root.mkdir(parents=True, exist_ok=True)
        pipeline.logger = Mock()

        @dataclass
        class ContrastCfg:
            name: str = "pain"
            output_type: str = "z-score"
            resample_to_freesurfer: bool = False
            fmriprep_space: str = "T1w"

        flm = make_mock_fitted_model(runs=1)
        run_meta = make_mock_run_meta(runs=1)

        fake_builder = types.SimpleNamespace(
            build_contrast_from_runs_detailed=lambda **kwargs: (
                "img",
                run_meta,
                SimpleNamespace(flm=flm, mask_img="img"),
                "cond_a",
                None,
            ),
            resample_to_freesurfer=lambda img, fs_dir: img,
            ContrastBuilderConfig=ContrastCfg,
        )
        fake_nib = types.SimpleNamespace(save=lambda img, path: None, load=lambda path: "img")

        with (
            patch.dict(
                sys.modules,
                {
                    "fmri_pipeline.analysis.contrast_builder": fake_builder,
                    "nibabel": fake_nib,
                },
            ),
            patch("pathlib.Path.write_text", side_effect=RuntimeError("no-write")),
        ):
            with self.assertRaisesRegex(RuntimeError, "no-write"):
                pipeline.process_subject(
                    "0001",
                    task="pain",
                    contrast_cfg=ContrastCfg(),
                    stats_cfg=None,
                    dry_run=False,
                )
