from __future__ import annotations

import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tests.pipelines_test_utils import DotConfig


class TestFmriPipelineStrictFailures(unittest.TestCase):
    def test_process_subject_raises_when_enabled_plotting_fails(self):
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

            def normalized(self):
                return self

        fake_builder = types.SimpleNamespace(
            build_contrast_from_runs_detailed=lambda **kwargs: (
                "img",
                {"output_type": "z_score"},
                SimpleNamespace(flm=SimpleNamespace(compute_contrast=lambda *a, **k: "x")),
                "def",
                None,
            ),
            resample_to_freesurfer=lambda img, fs_dir: img,
            ContrastBuilderConfig=ContrastCfg,
        )
        fake_nib = types.SimpleNamespace(save=lambda img, path: None, load=lambda path: "img")
        fake_plotting = types.SimpleNamespace(FmriPlottingConfig=PlotCfg)
        fake_reporting = types.SimpleNamespace(
            run_fmri_plotting_and_report=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("plot-fail"))
        )

        with patch.dict(
            sys.modules,
            {
                "fmri_pipeline.analysis.contrast_builder": fake_builder,
                "nibabel": fake_nib,
                "fmri_pipeline.analysis.plotting_config": fake_plotting,
                "fmri_pipeline.analysis.reporting": fake_reporting,
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "plot-fail"):
                pipeline.process_subject(
                    "0001",
                    task="pain",
                    contrast_cfg=ContrastCfg(),
                    plotting_cfg=PlotCfg(),
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

        fake_builder = types.SimpleNamespace(
            build_contrast_from_runs_detailed=lambda **kwargs: (
                "img",
                {"output_type": "z_score"},
                SimpleNamespace(flm=SimpleNamespace(compute_contrast=lambda *a, **k: "x")),
                "def",
                None,
            ),
            resample_to_freesurfer=lambda img, fs_dir: img,
            ContrastBuilderConfig=ContrastCfg,
        )
        fake_nib = types.SimpleNamespace(save=lambda img, path: None, load=lambda path: "img")

        with patch.dict(
            sys.modules,
            {
                "fmri_pipeline.analysis.contrast_builder": fake_builder,
                "nibabel": fake_nib,
            },
        ), patch("pathlib.Path.write_text", side_effect=RuntimeError("no-write")):
            with self.assertRaisesRegex(RuntimeError, "no-write"):
                pipeline.process_subject(
                    "0001",
                    task="pain",
                    contrast_cfg=ContrastCfg(),
                    plotting_cfg=None,
                    dry_run=False,
                )
