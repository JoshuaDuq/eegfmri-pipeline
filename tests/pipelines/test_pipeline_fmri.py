import json
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


from tests.pipelines_test_utils import DotConfig, DummyProgress, NoopBatchProgress, NoopProgress

_DummyProgress = DummyProgress
_NoopBatchProgress = NoopBatchProgress
_NoopProgress = NoopProgress


class TestFmriAnalysisGapfill(unittest.TestCase):
    def test_discover_plot_assets_mni_branches(self):
        from fmri_pipeline.pipelines.fmri_analysis import FmriAnalysisPipeline

        tmp = Path(tempfile.mkdtemp())
        p = object.__new__(FmriAnalysisPipeline)
        p.deriv_root = tmp / "deriv"
        p.deriv_root.mkdir(parents=True, exist_ok=True)

        empty_anat = p.deriv_root / "preprocessed" / "fmri" / "sub-0001" / "anat"
        empty_anat.mkdir(parents=True, exist_ok=True)
        fallback_anat = p.deriv_root / "fmriprep" / "sub-0001" / "anat"
        fallback_anat.mkdir(parents=True, exist_ok=True)
        expected_anat = fallback_anat / "sub-0001_space-MNI152NLin6Asym_desc-preproc_T1w.nii.gz"
        expected_anat.write_text("x", encoding="utf-8")

        bg_img, mask_img = p._discover_plot_assets(sub_label="sub-0001", task="task", space="mni")

        self.assertEqual(bg_img, expected_anat)
        self.assertIsNone(mask_img)

    def test_resample_success_and_mni_cached_without_mutating_cfg(self):
        from fmri_pipeline.pipelines.fmri_analysis import FmriAnalysisPipeline

        tmp = Path(tempfile.mkdtemp())
        bids_root = tmp / "bids"
        bids_root.mkdir(parents=True, exist_ok=True)

        p = object.__new__(FmriAnalysisPipeline)
        p.config = DotConfig(
            {"paths": {"bids_fmri_root": str(bids_root), "freesurfer_dir": str(tmp / "fs")}}
        )
        p.deriv_root = tmp / "deriv"
        p.deriv_root.mkdir(parents=True, exist_ok=True)
        p.logger = Mock()

        fs_subj = tmp / "fs" / "sub-0001"
        fs_subj.mkdir(parents=True, exist_ok=True)

        class ContrastCfg:
            def __init__(self):
                self.name = "pain"
                self.output_type = "z-score"
                self.resample_to_freesurfer = True
                self.fmriprep_space = "T1w"

        @dataclass
        class PlotCfg:
            enabled: bool = True
            space: str = "mni"
            include_effect_size: bool = True
            include_standard_error: bool = True

            def normalized(self):
                return self

        @dataclass
        class CBuilderCfg:
            name: str = "pain"
            output_type: str = "z-score"
            resample_to_freesurfer: bool = False
            fmriprep_space: str = "T1w"

        report_calls = []
        out_mni_loads = {"count": 0}

        def fake_load(path):
            if "space-MNI152NLin2009cAsym" in str(path):
                out_mni_loads["count"] += 1
            return f"loaded:{Path(path).name}"

        def compute_contrast(_contrast_arg, output_type):
            if output_type == "effect_size":
                return "effect"
            if output_type == "effect_variance":
                return "variance"
            return "contrast"

        build_calls = {"count": 0}

        def build_contrast_from_runs_detailed(**kwargs):
            build_calls["count"] += 1
            if build_calls["count"] == 1:
                image = "native_img"
            else:
                image = "mni_img"
            return (
                image,
                {"output_type": "z_score"},
                SimpleNamespace(flm=SimpleNamespace(compute_contrast=compute_contrast)),
                "def",
                None,
            )

        fake_builder = types.SimpleNamespace(
            build_contrast_from_runs_detailed=build_contrast_from_runs_detailed,
            resample_to_freesurfer=Mock(side_effect=lambda img, fs_dir: img),
            ContrastBuilderConfig=CBuilderCfg,
        )
        fake_plot = types.SimpleNamespace(FmriPlottingConfig=PlotCfg)
        fake_report = types.SimpleNamespace(
            run_fmri_plotting_and_report=lambda **kwargs: (
                report_calls.append(kwargs) or {"ok": True}
            )
        )
        fake_nib = types.SimpleNamespace(
            save=lambda img, path: Path(path).write_text("x", encoding="utf-8"), load=fake_load
        )

        with patch.dict(
            sys.modules,
            {
                "fmri_pipeline.analysis.contrast_builder": fake_builder,
                "fmri_pipeline.analysis.plotting_config": fake_plot,
                "fmri_pipeline.analysis.reporting": fake_report,
                "nibabel": fake_nib,
            },
        ):
            cfg = ContrastCfg()
            p.process_subject(
                "0001", "task", contrast_cfg=cfg, plotting_cfg=PlotCfg(), dry_run=False
            )
            p.process_subject(
                "0001", "task", contrast_cfg=cfg, plotting_cfg=PlotCfg(), dry_run=False
            )

        self.assertEqual(cfg.fmriprep_space, "T1w")
        self.assertTrue(fake_builder.resample_to_freesurfer.called)
        self.assertGreaterEqual(out_mni_loads["count"], 1)
        # Reporting no longer runs inside the GLM path: the analysis writes stat
        # maps and a manifest, and `fmri-analysis report` renders from those. This
        # is what stops subject-level QC being recomputed once per contrast.
        self.assertEqual(report_calls, [])

    def test_mni_save_and_contrast_compute_exceptions_surface(self):
        from fmri_pipeline.pipelines.fmri_analysis import FmriAnalysisPipeline

        tmp = Path(tempfile.mkdtemp())
        bids_root = tmp / "bids"
        bids_root.mkdir(parents=True, exist_ok=True)

        p = object.__new__(FmriAnalysisPipeline)
        p.config = DotConfig({"paths": {"bids_fmri_root": str(bids_root)}})
        p.deriv_root = tmp / "deriv"
        p.deriv_root.mkdir(parents=True, exist_ok=True)
        p.logger = Mock()

        @dataclass
        class Cfg:
            name: str = "pain"
            output_type: str = "z-score"
            resample_to_freesurfer: bool = False
            fmriprep_space: str = "T1w"

        @dataclass
        class PlotCfg:
            enabled: bool = True
            space: str = "both"
            include_effect_size: bool = True
            include_standard_error: bool = True

            def normalized(self):
                return self

        class BoomFLM:
            def compute_contrast(self, *args, **kwargs):
                raise RuntimeError("boom")

        def _build(**kwargs):
            return (
                "img",
                {"output_type": "z_score"},
                SimpleNamespace(flm=BoomFLM()),
                "def",
                None,
            )

        def _save(img, path):
            if "space-MNI152NLin2009cAsym" in str(path):
                raise RuntimeError("save-fail")
            return None

        fake_builder = types.SimpleNamespace(
            build_contrast_from_runs_detailed=_build,
            resample_to_freesurfer=lambda i, d: i,
            ContrastBuilderConfig=Cfg,
        )
        fake_plot = types.SimpleNamespace(FmriPlottingConfig=PlotCfg)
        fake_report = types.SimpleNamespace(
            run_fmri_plotting_and_report=lambda **kwargs: {"ok": True}
        )
        fake_nib = types.SimpleNamespace(save=_save, load=lambda p: "img")

        with patch.dict(
            sys.modules,
            {
                "fmri_pipeline.analysis.contrast_builder": fake_builder,
                "fmri_pipeline.analysis.plotting_config": fake_plot,
                "fmri_pipeline.analysis.reporting": fake_report,
                "nibabel": fake_nib,
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "save-fail"):
                p.process_subject(
                    "0001", "t", contrast_cfg=Cfg(), plotting_cfg=PlotCfg(), dry_run=False
                )

    def test_mni_build_exception_surfaces(self):
        from fmri_pipeline.pipelines.fmri_analysis import FmriAnalysisPipeline

        tmp = Path(tempfile.mkdtemp())
        bids_root = tmp / "bids"
        bids_root.mkdir(parents=True, exist_ok=True)

        p = object.__new__(FmriAnalysisPipeline)
        p.config = DotConfig({"paths": {"bids_fmri_root": str(bids_root)}})
        p.deriv_root = tmp / "deriv"
        p.deriv_root.mkdir(parents=True, exist_ok=True)
        p.logger = Mock()

        @dataclass
        class Cfg:
            name: str = "pain"
            output_type: str = "z-score"
            resample_to_freesurfer: bool = False
            fmriprep_space: str = "T1w"

        @dataclass
        class PlotCfg:
            enabled: bool = True
            space: str = "mni"
            include_effect_size: bool = True
            include_standard_error: bool = True

            def normalized(self):
                return self

        calls = {"n": 0}

        def _build(**kwargs):
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("mni-build-fail")
            return (
                "img",
                {"output_type": "z_score"},
                SimpleNamespace(flm=SimpleNamespace(compute_contrast=lambda *a, **k: "x")),
                "def",
                None,
            )

        fake_builder = types.SimpleNamespace(
            build_contrast_from_runs_detailed=_build,
            resample_to_freesurfer=lambda i, d: i,
            ContrastBuilderConfig=Cfg,
        )
        fake_plot = types.SimpleNamespace(FmriPlottingConfig=PlotCfg)
        fake_report = types.SimpleNamespace(
            run_fmri_plotting_and_report=lambda **kwargs: {"ok": True}
        )
        fake_nib = types.SimpleNamespace(save=lambda *a, **k: None, load=lambda *a, **k: "img")

        with patch.dict(
            sys.modules,
            {
                "fmri_pipeline.analysis.contrast_builder": fake_builder,
                "fmri_pipeline.analysis.plotting_config": fake_plot,
                "fmri_pipeline.analysis.reporting": fake_report,
                "nibabel": fake_nib,
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "mni-build-fail"):
                p.process_subject(
                    "0001", "t", contrast_cfg=Cfg(), plotting_cfg=PlotCfg(), dry_run=False
                )


class TestFmriPreprocessingGapfill(unittest.TestCase):
    def test_constructor_and_missing_paths_errors(self):
        from fmri_pipeline.pipelines.fmri_preprocessing import FmriPreprocessingPipeline

        cfg = DotConfig({})
        with patch(
            "fmri_pipeline.pipelines.fmri_preprocessing.PipelineBase.__init__",
            lambda self, name, config=None: (
                setattr(self, "name", name),
                setattr(self, "config", config or cfg),
                setattr(self, "logger", Mock()),
                setattr(self, "deriv_root", Path(tempfile.mkdtemp())),
            ),
        ):
            p = FmriPreprocessingPipeline(config=cfg)
        self.assertEqual(p.name, "fmri_preprocessing")

        p = object.__new__(FmriPreprocessingPipeline)
        p.deriv_root = Path(tempfile.mkdtemp())
        p.logger = Mock()
        p.config = DotConfig(
            {
                "paths": {"bids_fmri_root": str(Path(tempfile.mkdtemp()) / "missing")},
                "fmri_preprocessing": {
                    "engine": "docker",
                    "fmriprep": {"fs_license_file": str(Path(tempfile.mkdtemp()) / "lic.txt")},
                },
            }
        )
        with self.assertRaises(FileNotFoundError):
            p.process_subject("0001", task="", dry_run=True)

    def test_process_subject_normalizes_sub_prefixed_subject_ids(self):
        from fmri_pipeline.pipelines.fmri_preprocessing import FmriPreprocessingPipeline

        tmp = Path(tempfile.mkdtemp())
        bids = tmp / "bids"
        (bids / "sub-0001").mkdir(parents=True, exist_ok=True)
        fs_license = tmp / "license.txt"
        fs_license.write_text("license", encoding="utf-8")

        p = object.__new__(FmriPreprocessingPipeline)
        p.config = DotConfig(
            {
                "paths": {"bids_fmri_root": str(bids)},
                "fmri_preprocessing": {
                    "engine": "docker",
                    "fmriprep": {
                        "fs_license_file": str(fs_license),
                        "output_spaces": ["T1w"],
                    },
                },
            }
        )
        p.deriv_root = tmp / "deriv"
        p.deriv_root.mkdir(parents=True, exist_ok=True)
        p.logger = Mock()
        p.get_subject_logger = lambda subject: Mock()

        progress = SimpleNamespace(
            subject_start=Mock(),
            subject_done=Mock(),
            step=lambda *args, **kwargs: None,
        )

        with (
            patch("fmri_pipeline.pipelines.fmri_preprocessing._require_supported_container_host"),
            patch("fmri_pipeline.pipelines.fmri_preprocessing._require_executable"),
            patch("fmri_pipeline.pipelines.fmri_preprocessing._stream_subprocess") as mock_stream,
        ):
            p.process_subject("sub-0001", task="", progress=progress, dry_run=False)

        cmd = mock_stream.call_args.args[0]
        self.assertIn("--participant-label", cmd)
        self.assertIn("0001", cmd)
        self.assertNotIn(
            "sub-0001",
            cmd[cmd.index("--participant-label") + 1 : cmd.index("--participant-label") + 2],
        )
        progress.subject_start.assert_called_once_with("sub-0001")
        progress.subject_done.assert_called_once_with("sub-0001", success=True)

        bids = Path(tempfile.mkdtemp())
        bids.mkdir(parents=True, exist_ok=True)
        p.config = DotConfig(
            {
                "paths": {"bids_fmri_root": str(bids)},
                "fmri_preprocessing": {
                    "engine": "docker",
                    "fmriprep": {"fs_license_file": str(bids / "missing_license.txt")},
                },
            }
        )
        with self.assertRaises(FileNotFoundError):
            p.process_subject("0001", task="", dry_run=True)

    def test_preprocessing_deriv_root_and_apptainer_sanitized_mount_branch(self):
        from fmri_pipeline.pipelines.fmri_preprocessing import (
            BIDS_SANITIZED_SOURCE_MOUNT,
            FmriPreprocessingPipeline,
        )

        p = object.__new__(FmriPreprocessingPipeline)
        p.config = DotConfig(
            {
                "paths": {"deriv_fmri_root": "/tmp/fmri-deriv"},
                "fmri_preprocessing": {"task_is_rest": False},
            }
        )
        with patch(
            "fmri_pipeline.pipelines.fmri_preprocessing.resolve_fmri_deriv_root",
            return_value=Path("/tmp/fmri-deriv"),
        ) as mock_resolve_deriv_root:
            self.assertEqual(p._resolve_pipeline_deriv_root(), Path("/tmp/fmri-deriv"))
        mock_resolve_deriv_root.assert_called_once_with(p.config, task_is_rest=False)

        tmp = Path(tempfile.mkdtemp())
        bids = tmp / "bids"
        bids.mkdir(parents=True, exist_ok=True)
        (bids / "sub-0001").mkdir(parents=True, exist_ok=True)
        lic = tmp / "lic.txt"
        lic.write_text("x", encoding="utf-8")

        p = object.__new__(FmriPreprocessingPipeline)
        p.config = DotConfig(
            {
                "paths": {"bids_fmri_root": str(bids)},
                "fmri_preprocessing": {
                    "engine": "apptainer",
                    "fmriprep": {"fs_license_file": str(lic)},
                },
            }
        )
        p.deriv_root = tmp / "deriv"
        p.deriv_root.mkdir(parents=True, exist_ok=True)
        p.logger = Mock()

        class _Tmp:
            def cleanup(self):
                return None

        with (
            patch(
                "fmri_pipeline.pipelines.fmri_preprocessing._resolve_bids_mount_root",
                return_value=(Path("/sanitized"), _Tmp()),
            ),
            patch("fmri_pipeline.pipelines.fmri_preprocessing._require_executable"),
        ):
            p.process_subject("0001", task="", progress=_NoopProgress(), dry_run=True)

        logged_commands = [
            call.args[1]
            for call in p.logger.info.call_args_list
            if call.args and call.args[0] == "fMRIPrep command: %s"
        ]
        self.assertTrue(any(BIDS_SANITIZED_SOURCE_MOUNT in cmd for cmd in logged_commands))

    def test_docker_and_apptainer_mount_flags(self):
        from fmri_pipeline.pipelines.fmri_preprocessing import FmriPreprocessingPipeline

        tmp = Path(tempfile.mkdtemp())
        bids = tmp / "bids"
        bids.mkdir(parents=True, exist_ok=True)
        (bids / "sub-0001").mkdir(parents=True, exist_ok=True)
        lic = tmp / "lic.txt"
        lic.write_text("x", encoding="utf-8")
        filt = tmp / "filter.json"
        filt.write_text("{}", encoding="utf-8")
        fs_dir = tmp / "fs"
        fs_dir.mkdir(parents=True, exist_ok=True)

        p = object.__new__(FmriPreprocessingPipeline)
        p.deriv_root = tmp / "deriv"
        p.deriv_root.mkdir(parents=True, exist_ok=True)
        p.logger = Mock()

        p.config = DotConfig(
            {
                "paths": {"bids_fmri_root": str(bids)},
                "fmri_preprocessing": {
                    "engine": "docker",
                    "fmriprep": {
                        "fs_license_file": str(lic),
                        "bids_filter_file": str(filt),
                        "fs_subjects_dir": str(fs_dir),
                    },
                },
            }
        )
        with patch("fmri_pipeline.pipelines.fmri_preprocessing._require_executable"):
            p.process_subject("0001", task="", dry_run=True)
        cmd_str = p.logger.info.call_args[0][1]
        self.assertIn("/bids_filter.json", cmd_str)
        self.assertIn("/fs", cmd_str)
        self.assertIn("--fs-subjects-dir", cmd_str)

        p.config = DotConfig(
            {
                "paths": {"bids_fmri_root": str(bids)},
                "fmri_preprocessing": {
                    "engine": "apptainer",
                    "fmriprep": {
                        "fs_license_file": str(lic),
                        "bids_filter_file": str(filt),
                        "fs_subjects_dir": str(fs_dir),
                    },
                },
            }
        )
        with patch("fmri_pipeline.pipelines.fmri_preprocessing._require_executable"):
            p.process_subject("0001", task="", dry_run=True)
        cmd_str = p.logger.info.call_args[0][1]
        self.assertIn("/bids_filter.json", cmd_str)
        self.assertIn("/fs", cmd_str)

    def test_apptainer_command_binds_templateflow_cache(self):
        from fmri_pipeline.pipelines.fmri_preprocessing import FmriPreprocessingPipeline

        tmp = Path(tempfile.mkdtemp())
        bids = tmp / "bids"
        bids.mkdir(parents=True, exist_ok=True)
        (bids / "sub-0001").mkdir(parents=True, exist_ok=True)
        lic = tmp / "lic.txt"
        lic.write_text("x", encoding="utf-8")
        templateflow_home = tmp / "templateflow"
        templateflow_home.mkdir(parents=True, exist_ok=True)

        p = object.__new__(FmriPreprocessingPipeline)
        p.deriv_root = tmp / "deriv"
        p.deriv_root.mkdir(parents=True, exist_ok=True)
        p.logger = Mock()
        p.config = DotConfig(
            {
                "paths": {"bids_fmri_root": str(bids)},
                "fmri_preprocessing": {
                    "engine": "apptainer",
                    "fmriprep": {"fs_license_file": str(lic)},
                },
            }
        )

        with (
            patch.dict("os.environ", {"TEMPLATEFLOW_HOME": str(templateflow_home)}),
            patch("fmri_pipeline.pipelines.fmri_preprocessing._require_executable"),
        ):
            p.process_subject("0001", task="", dry_run=True)

        cmd_str = p.logger.info.call_args[0][1]
        self.assertIn("--env", cmd_str)
        self.assertIn(f"TEMPLATEFLOW_HOME={templateflow_home.resolve()}", cmd_str)
        self.assertIn(f"{templateflow_home.resolve()}:{templateflow_home.resolve()}", cmd_str)

    def test_freesurfer_license_from_env_var(self):
        from fmri_pipeline.pipelines.fmri_preprocessing import FmriPreprocessingPipeline

        tmp = Path(tempfile.mkdtemp())
        bids = tmp / "bids"
        bids.mkdir(parents=True, exist_ok=True)
        (bids / "sub-0001").mkdir(parents=True, exist_ok=True)
        lic = tmp / "env_license.txt"
        lic.write_text("x", encoding="utf-8")

        p = object.__new__(FmriPreprocessingPipeline)
        p.deriv_root = tmp / "deriv"
        p.deriv_root.mkdir(parents=True, exist_ok=True)
        p.logger = Mock()
        p.config = DotConfig(
            {
                "paths": {"bids_fmri_root": str(bids)},
                "fmri_preprocessing": {"engine": "docker", "fmriprep": {}},
            }
        )

        with (
            patch.dict("os.environ", {"EEG_PIPELINE_FREESURFER_LICENSE": str(lic)}),
            patch("fmri_pipeline.pipelines.fmri_preprocessing._require_executable"),
        ):
            p.process_subject("0001", task="", dry_run=True)

    def test_freesurfer_license_defaults_to_home_license_txt(self):
        from fmri_pipeline.pipelines.fmri_preprocessing import FmriPreprocessingPipeline

        tmp = Path(tempfile.mkdtemp())
        bids = tmp / "bids"
        bids.mkdir(parents=True, exist_ok=True)
        (bids / "sub-0001").mkdir(parents=True, exist_ok=True)

        home_dir = Path(tempfile.mkdtemp())
        default_license = home_dir / "license.txt"
        default_license.write_text("x", encoding="utf-8")

        p = object.__new__(FmriPreprocessingPipeline)
        p.deriv_root = tmp / "deriv"
        p.deriv_root.mkdir(parents=True, exist_ok=True)
        p.logger = Mock()
        p.config = DotConfig(
            {
                "paths": {"bids_fmri_root": str(bids)},
                "fmri_preprocessing": {"engine": "docker", "fmriprep": {}},
            }
        )

        with (
            patch.dict(
                "os.environ",
                {"HOME": str(home_dir), "EEG_PIPELINE_FREESURFER_LICENSE": ""},
            ),
            patch(
                "fmri_pipeline.pipelines.fmri_preprocessing.FS_LICENSE_DEFAULT_PATH",
                str(default_license),
            ),
            patch("fmri_pipeline.pipelines.fmri_preprocessing._require_executable"),
        ):
            p.process_subject("0001", task="", dry_run=True)

        cmd_str = p.logger.info.call_args[0][1]
        self.assertIn(str(default_license.resolve()), cmd_str)

    def test_docker_mount_uses_sanitized_view_when_macos_metadata_exists(self):
        from fmri_pipeline.pipelines.fmri_preprocessing import FmriPreprocessingPipeline

        tmp = Path(tempfile.mkdtemp())
        bids = tmp / "bids"
        bids.mkdir(parents=True, exist_ok=True)
        (bids / "sub-0001" / "func").mkdir(parents=True, exist_ok=True)
        (bids / "dataset_description.json").write_text(
            '{"Name":"x","BIDSVersion":"1.7.0","DatasetType":"raw"}',
            encoding="utf-8",
        )
        (bids / "sub-0001" / "func" / "sub-0001_task-rest_run-01_bold.nii.gz").write_text(
            "x", encoding="utf-8"
        )
        (bids / "._sub-0001").write_text("x", encoding="utf-8")
        (bids / ".DS_Store").write_text("x", encoding="utf-8")

        lic = tmp / "license.txt"
        lic.write_text("x", encoding="utf-8")

        p = object.__new__(FmriPreprocessingPipeline)
        p.deriv_root = tmp / "deriv"
        p.deriv_root.mkdir(parents=True, exist_ok=True)
        p.logger = Mock()
        p.config = DotConfig(
            {
                "paths": {"bids_fmri_root": str(bids)},
                "fmri_preprocessing": {
                    "engine": "docker",
                    "fmriprep": {"fs_license_file": str(lic)},
                },
            }
        )

        class _Tmp:
            def cleanup(self):
                return None

        def _resolve_sanitized_mount(_bids_dir, logger):
            logger.warning("using sanitized BIDS mount")
            return tmp / "sanitized_bids", _Tmp()

        with (
            patch(
                "fmri_pipeline.pipelines.fmri_preprocessing._resolve_bids_mount_root",
                side_effect=_resolve_sanitized_mount,
            ),
            patch("fmri_pipeline.pipelines.fmri_preprocessing._require_executable"),
        ):
            p.process_subject("0001", task="", dry_run=True)

        cmd_str = p.logger.info.call_args[0][1]
        self.assertNotIn(f"{bids}:/data:ro", cmd_str)
        self.assertIn(":/data:ro", cmd_str)
        self.assertIn(f"{bids}:/bids_source:ro", cmd_str)
        self.assertTrue(p.logger.warning.called)


class TestBemGenerationLicensePath(unittest.TestCase):
    def test_get_fs_license_path_uses_env_var(self):
        from fmri_pipeline.analysis.bem_generation import get_fs_license_path

        tmp = Path(tempfile.mkdtemp())
        lic = tmp / "license.txt"
        lic.write_text("x", encoding="utf-8")

        with patch.dict("os.environ", {"EEG_PIPELINE_FREESURFER_LICENSE": str(lic)}):
            resolved = get_fs_license_path(DotConfig({"paths": {}}))

        self.assertEqual(resolved, lic)


class TestFmriDeep(unittest.TestCase):
    def test_fmri_analysis_process_subject_full_non_plotting(self):
        from fmri_pipeline.pipelines.fmri_analysis import FmriAnalysisPipeline

        tmp = Path(tempfile.mkdtemp())
        bids_root = tmp / "bids"
        bids_root.mkdir(parents=True, exist_ok=True)

        p = object.__new__(FmriAnalysisPipeline)
        p.config = DotConfig({"paths": {"bids_fmri_root": str(bids_root)}})
        p.deriv_root = tmp / "deriv"
        p.deriv_root.mkdir(parents=True, exist_ok=True)
        p.logger = Mock()

        @dataclass
        class Cfg:
            name: str = "contrast"
            output_type: str = "z-score"
            resample_to_freesurfer: bool = False

        contrast_cfg = Cfg()

        fake_builder = types.SimpleNamespace(
            build_contrast_from_runs_detailed=lambda **kwargs: (
                "img",
                {"output_type": "z_score"},
                SimpleNamespace(flm=SimpleNamespace(compute_contrast=lambda *a, **k: "x")),
                "def",
                None,
            ),
            resample_to_freesurfer=lambda img, fs_dir: img,
            ContrastBuilderConfig=Cfg,
        )
        fake_nib = types.SimpleNamespace(
            save=lambda img, path: Path(path).write_text("nii", encoding="utf-8"),
            load=lambda path: "img",
        )

        with patch.dict(
            sys.modules,
            {"fmri_pipeline.analysis.contrast_builder": fake_builder, "nibabel": fake_nib},
        ):
            p.process_subject(
                "0001", task="task", contrast_cfg=contrast_cfg, plotting_cfg=None, dry_run=False
            )

        out_dir = (
            p.deriv_root / "sub-0001" / "fmri" / "first_level" / "task-task" / "contrast-contrast"
        )
        sidecars = list(out_dir.glob("*.json"))
        self.assertTrue(sidecars)
        payload = json.loads(sidecars[0].read_text(encoding="utf-8"))
        self.assertEqual(payload["subject"], "sub-0001")

    def test_fmri_analysis_discover_plot_assets(self):
        from fmri_pipeline.pipelines.fmri_analysis import FmriAnalysisPipeline

        tmp = Path(tempfile.mkdtemp())
        p = object.__new__(FmriAnalysisPipeline)
        p.deriv_root = tmp

        func_dir = tmp / "preprocessed" / "fmri" / "sub-0001" / "func"
        anat_dir = tmp / "preprocessed" / "fmri" / "sub-0001" / "anat"
        func_dir.mkdir(parents=True, exist_ok=True)
        anat_dir.mkdir(parents=True, exist_ok=True)
        (func_dir / "sub-0001_task-task_run-01_space-T1w_desc-brain_mask.nii.gz").write_text("x")
        (func_dir / "sub-0001_task-task_run-01_space-T1w_boldref.nii.gz").write_text("x")
        (anat_dir / "sub-0001_desc-preproc_T1w.nii.gz").write_text("x")

        bg, mask = p._discover_plot_assets(sub_label="sub-0001", task="task", space="native")
        self.assertIsNotNone(bg)
        self.assertIsNotNone(mask)

    def test_fmri_analysis_plotting_branch_enabled(self):
        from fmri_pipeline.pipelines.fmri_analysis import FmriAnalysisPipeline

        tmp = Path(tempfile.mkdtemp())
        bids_root = tmp / "bids"
        bids_root.mkdir(parents=True, exist_ok=True)

        p = object.__new__(FmriAnalysisPipeline)
        p.config = DotConfig({"paths": {"bids_fmri_root": str(bids_root)}})
        p.deriv_root = tmp / "deriv"
        p.deriv_root.mkdir(parents=True, exist_ok=True)
        p.logger = Mock()

        @dataclass
        class Cfg:
            name: str = "contrast"
            output_type: str = "z-score"
            resample_to_freesurfer: bool = False
            fmriprep_space: str = "T1w"

        @dataclass
        class FakePlotCfg:
            enabled: bool = True
            space: str = "native"
            include_effect_size: bool = True
            include_standard_error: bool = True

            def normalized(self):
                return self

        contrast_cfg = Cfg()
        plotting_cfg = FakePlotCfg()

        fake_builder = types.SimpleNamespace(
            build_contrast_from_runs_detailed=lambda **kwargs: (
                "img",
                {"output_type": "z_score"},
                SimpleNamespace(flm=SimpleNamespace(compute_contrast=lambda *a, **k: "x")),
                "def",
                None,
            ),
            resample_to_freesurfer=lambda img, fs_dir: img,
            ContrastBuilderConfig=Cfg,
        )
        fake_nib = types.SimpleNamespace(
            save=lambda img, path: Path(path).write_text("nii", encoding="utf-8"),
            load=lambda path: "img",
        )
        fake_plot_mod = types.SimpleNamespace(FmriPlottingConfig=FakePlotCfg)
        fake_report_mod = types.SimpleNamespace(
            run_fmri_plotting_and_report=lambda **kwargs: {"ok": True}
        )

        with patch.dict(
            sys.modules,
            {
                "fmri_pipeline.analysis.contrast_builder": fake_builder,
                "nibabel": fake_nib,
                "fmri_pipeline.analysis.plotting_config": fake_plot_mod,
                "fmri_pipeline.analysis.reporting": fake_report_mod,
            },
        ):
            p.process_subject(
                "0001",
                task="task",
                contrast_cfg=contrast_cfg,
                plotting_cfg=plotting_cfg,
                dry_run=False,
            )

    def test_fmri_preprocessing_non_dry_executes_stream(self):
        from fmri_pipeline.pipelines.fmri_preprocessing import FmriPreprocessingPipeline

        tmp = Path(tempfile.mkdtemp())
        bids_root = tmp / "bids"
        bids_root.mkdir(parents=True, exist_ok=True)
        (bids_root / "sub-0001").mkdir(parents=True, exist_ok=True)
        fs_license = tmp / "license.txt"
        fs_license.write_text("dummy", encoding="utf-8")

        p = object.__new__(FmriPreprocessingPipeline)
        p.config = DotConfig(
            {
                "paths": {"bids_fmri_root": str(bids_root)},
                "fmri_preprocessing": {
                    "engine": "docker",
                    "fmriprep": {"fs_license_file": str(fs_license), "output_spaces": ["T1w"]},
                },
            }
        )
        p.deriv_root = tmp / "deriv"
        p.deriv_root.mkdir(parents=True, exist_ok=True)
        p.logger = Mock()
        p.get_subject_logger = lambda subject: Mock()

        with (
            patch("fmri_pipeline.pipelines.fmri_preprocessing._require_supported_container_host"),
            patch("fmri_pipeline.pipelines.fmri_preprocessing._require_executable"),
            patch("fmri_pipeline.pipelines.fmri_preprocessing._stream_subprocess") as mock_stream,
        ):
            p.process_subject(
                "0001",
                task="",
                progress=SimpleNamespace(
                    subject_start=lambda *a, **k: None,
                    subject_done=lambda *a, **k: None,
                    step=lambda *a, **k: None,
                ),
                dry_run=False,
            )
        mock_stream.assert_called_once()

    def test_fmri_trial_signatures_non_dry(self):
        from fmri_pipeline.pipelines.fmri_trial_signatures import FmriTrialSignaturePipeline

        tmp = Path(tempfile.mkdtemp())
        p = object.__new__(FmriTrialSignaturePipeline)
        p.config = DotConfig({})
        p.deriv_root = tmp / "deriv"
        p.deriv_root.mkdir(parents=True, exist_ok=True)
        p.logger = Mock()

        @dataclass
        class TrialSignatureExtractionConfig:
            method: str = "lss"
            task: str = "task"

        fake_mod = types.SimpleNamespace(
            TrialSignatureExtractionConfig=TrialSignatureExtractionConfig,
            run_trial_signature_extraction_for_subject=lambda **kwargs: {
                "output_dir": str(tmp / "out")
            },
        )
        with patch.dict(sys.modules, {"fmri_pipeline.analysis.trial_signatures": fake_mod}):
            p.process_subject(
                "0001",
                task="task",
                bids_fmri_root=tmp,
                trial_cfg=TrialSignatureExtractionConfig(),
                output_dir=tmp / "out",
                dry_run=False,
                progress=SimpleNamespace(
                    subject_start=lambda *a, **k: None, step=lambda *a, **k: None
                ),
            )


class TestFmriCompletion(unittest.TestCase):
    def test_fmri_trial_signatures_init_discover_and_group_level(self):
        from fmri_pipeline.pipelines.fmri_trial_signatures import FmriTrialSignaturePipeline

        cfg = DotConfig({})
        with patch(
            "fmri_pipeline.pipelines.fmri_trial_signatures.PipelineBase.__init__",
            lambda self, name, config=None: (
                setattr(self, "config", config or cfg),
                setattr(self, "deriv_root", Path(tempfile.mkdtemp())),
                setattr(self, "logger", Mock()),
            ),
        ):
            p = FmriTrialSignaturePipeline(config=cfg)
        ext = p.deriv_root.parent / "external"
        ext.mkdir(parents=True, exist_ok=True)
        sig_root, _sig_specs = p._discover_signature_root_and_specs()
        self.assertIsNone(sig_root)
        self.assertIsNone(p.run_group_level(["0001"], task="t"))

        p.config = DotConfig({"paths": {"signature_dir": "/path/does/not/exist"}})
        with self.assertRaises(FileNotFoundError):
            p._discover_signature_root_and_specs()

        class BadCfg:
            def get(self, *_a, **_k):
                raise RuntimeError("bad")

        p.config = BadCfg()
        with self.assertRaisesRegex(RuntimeError, "bad"):
            p._discover_signature_root_and_specs()

    def test_fmri_analysis_init_and_error_paths(self):
        from fmri_pipeline.pipelines.fmri_analysis import FmriAnalysisPipeline

        cfg = DotConfig({"paths": {}})
        with patch(
            "fmri_pipeline.pipelines.fmri_analysis.PipelineBase.__init__",
            lambda self, name, config=None: (
                setattr(self, "config", config or cfg),
                setattr(self, "deriv_root", Path(tempfile.mkdtemp())),
                setattr(self, "logger", Mock()),
            ),
        ):
            p = FmriAnalysisPipeline(config=cfg)

        ext = p.deriv_root.parent / "external"
        ext.mkdir(parents=True, exist_ok=True)
        sig_root, _sig_specs = p._discover_signature_root_and_specs()
        self.assertIsNone(sig_root)

        @dataclass
        class Cfg:
            name: str = "x"
            output_type: str = "z-score"
            resample_to_freesurfer: bool = False

        with self.assertRaises(ValueError):
            p.process_subject("0001", "t", contrast_cfg=Cfg(), dry_run=False)

        class BadCfg:
            def get(self, *_a, **_k):
                raise RuntimeError("bad")

        p.config = BadCfg()
        p.deriv_root = object()
        with self.assertRaisesRegex(RuntimeError, "bad"):
            p._discover_signature_root_and_specs()

    def test_fmri_analysis_resample_error_paths(self):
        from fmri_pipeline.pipelines.fmri_analysis import FmriAnalysisPipeline

        tmp = Path(tempfile.mkdtemp())
        bids_root = tmp / "bids"
        bids_root.mkdir(parents=True, exist_ok=True)

        p = object.__new__(FmriAnalysisPipeline)
        p.config = DotConfig({"paths": {"bids_fmri_root": str(bids_root)}})
        p.deriv_root = tmp / "deriv"
        p.deriv_root.mkdir(parents=True, exist_ok=True)
        p.logger = Mock()

        @dataclass
        class Cfg:
            name: str = "x"
            output_type: str = "z-score"
            resample_to_freesurfer: bool = True

        fake_builder = types.SimpleNamespace(
            build_contrast_from_runs_detailed=lambda **kwargs: (
                "img",
                {"output_type": "z_score"},
                SimpleNamespace(flm=SimpleNamespace(compute_contrast=lambda *a, **k: "x")),
                "def",
                None,
            ),
            resample_to_freesurfer=lambda img, fs_dir: img,
        )
        fake_nib = types.SimpleNamespace(save=lambda *a, **k: None, load=lambda *a, **k: "img")

        with patch.dict(
            sys.modules,
            {"fmri_pipeline.analysis.contrast_builder": fake_builder, "nibabel": fake_nib},
        ):
            with self.assertRaises(ValueError):
                p.process_subject("0001", "t", contrast_cfg=Cfg(), dry_run=False)

        fs_dir = tmp / "fs"
        fs_dir.mkdir(parents=True, exist_ok=True)
        p.config = DotConfig(
            {"paths": {"bids_fmri_root": str(bids_root), "freesurfer_dir": str(fs_dir)}}
        )
        with patch.dict(
            sys.modules,
            {"fmri_pipeline.analysis.contrast_builder": fake_builder, "nibabel": fake_nib},
        ):
            with self.assertRaises(FileNotFoundError):
                p.process_subject("0001", "t", contrast_cfg=Cfg(), dry_run=False)

    def test_fmri_analysis_mni_plotting_branch(self):
        from fmri_pipeline.pipelines.fmri_analysis import FmriAnalysisPipeline

        tmp = Path(tempfile.mkdtemp())
        bids_root = tmp / "bids"
        bids_root.mkdir(parents=True, exist_ok=True)

        p = object.__new__(FmriAnalysisPipeline)
        p.config = DotConfig({"paths": {"bids_fmri_root": str(bids_root)}})
        p.deriv_root = tmp / "deriv"
        p.deriv_root.mkdir(parents=True, exist_ok=True)
        p.logger = Mock()

        @dataclass
        class Cfg:
            name: str = "pain"
            output_type: str = "z-score"
            resample_to_freesurfer: bool = False
            fmriprep_space: str = "T1w"

        @dataclass
        class PlotCfg:
            enabled: bool = True
            space: str = "both"
            include_effect_size: bool = True
            include_standard_error: bool = True

            def normalized(self):
                return self

        calls = {"n": 0}

        def _build(**kwargs):
            calls["n"] += 1
            return (
                "img",
                {"output_type": "z_score"},
                SimpleNamespace(flm=SimpleNamespace(compute_contrast=lambda *a, **k: "x")),
                "def",
                None,
            )

        fake_builder = types.SimpleNamespace(
            build_contrast_from_runs_detailed=_build,
            resample_to_freesurfer=lambda i, d: i,
            ContrastBuilderConfig=Cfg,
        )
        fake_nib = types.SimpleNamespace(
            save=lambda img, path: Path(path).write_text("x", encoding="utf-8"),
            load=lambda path: "img",
        )
        fake_plot = types.SimpleNamespace(FmriPlottingConfig=PlotCfg)
        fake_report = types.SimpleNamespace(run_fmri_plotting_and_report=lambda **k: {"ok": True})

        with patch.dict(
            sys.modules,
            {
                "fmri_pipeline.analysis.contrast_builder": fake_builder,
                "nibabel": fake_nib,
                "fmri_pipeline.analysis.plotting_config": fake_plot,
                "fmri_pipeline.analysis.reporting": fake_report,
            },
        ):
            done = {"ok": False}
            progress = SimpleNamespace(
                subject_start=lambda *a, **k: None,
                step=lambda *a, **k: None,
                subject_done=lambda *a, **k: done.__setitem__("ok", True),
            )
            p.process_subject(
                "0001",
                "t",
                contrast_cfg=Cfg(),
                plotting_cfg=PlotCfg(),
                dry_run=False,
                progress=progress,
            )
        self.assertGreaterEqual(calls["n"], 2)
        self.assertTrue(done["ok"])

    def test_fmri_preprocessing_stream_and_apptainer_dry(self):
        from fmri_pipeline.pipelines.fmri_preprocessing import (
            _stream_subprocess,
            FmriPreprocessingPipeline,
        )

        class P:
            def __init__(self, rc=0):
                self.stdout = iter(["line1\n", "line2\n"])
                self._rc = rc

            def wait(self):
                return self._rc

        with patch(
            "fmri_pipeline.pipelines.fmri_preprocessing.subprocess.Popen", return_value=P(0)
        ):
            _stream_subprocess(["cmd"], Mock())
        with patch(
            "fmri_pipeline.pipelines.fmri_preprocessing.subprocess.Popen", return_value=P(1)
        ):
            with self.assertRaises(RuntimeError):
                _stream_subprocess(["cmd"], Mock())

        tmp = Path(tempfile.mkdtemp())
        bids = tmp / "bids"
        bids.mkdir(parents=True, exist_ok=True)
        (bids / "sub-0001").mkdir(parents=True, exist_ok=True)
        lic = tmp / "lic.txt"
        lic.write_text("x", encoding="utf-8")

        p = object.__new__(FmriPreprocessingPipeline)
        p.config = DotConfig(
            {
                "paths": {"bids_fmri_root": str(bids)},
                "fmri_preprocessing": {
                    "engine": "apptainer",
                    "fmriprep": {
                        "fs_license_file": str(lic),
                        "output_spaces": ["T1w"],
                        "ignore": ["slicetiming"],
                        "use_aroma": True,
                        "skip_bids_validation": True,
                        "clean_workdir": True,
                        "stop_on_first_crash": True,
                        "fs_no_reconall": True,
                        "mem_mb": 4096,
                        "nthreads": 2,
                        "omp_nthreads": 2,
                        "low_mem": True,
                        "longitudinal": True,
                        "cifti_output": "91k",
                        "level": "minimal",
                        "skull_strip_template": "MNI",
                        "skull_strip_fixed_seed": True,
                        "random_seed": 7,
                        "dummy_scans": 2,
                        "bold2t1w_init": "header",
                        "bold2t1w_dof": 12,
                        "slice_time_ref": 0.3,
                        "fd_spike_threshold": 0.2,
                        "dvars_spike_threshold": 1.2,
                        "me_output_echos": True,
                        "medial_surface_nan": True,
                        "no_msm": True,
                        "task_id": "task",
                        "extra_args": "--dummy-opt 1",
                    },
                },
            }
        )
        p.deriv_root = tmp / "deriv"
        p.deriv_root.mkdir(parents=True, exist_ok=True)
        p.logger = Mock()

        with patch("fmri_pipeline.pipelines.fmri_preprocessing._require_executable"):
            p.process_subject("0001", task="", progress=_NoopProgress(), dry_run=True)

    def test_fmri_preprocessing_validate_and_error_branches(self):
        from fmri_pipeline.pipelines.fmri_preprocessing import FmriPreprocessingPipeline

        p = object.__new__(FmriPreprocessingPipeline)
        with self.assertRaises(ValueError):
            p._validate_batch_inputs([], None)
        self.assertEqual(p._validate_batch_inputs(["0001"], None), "")

        tmp = Path(tempfile.mkdtemp())
        p.deriv_root = tmp / "deriv"
        p.deriv_root.mkdir(parents=True, exist_ok=True)
        p.logger = Mock()

        # missing paths.bids_fmri_root
        p.config = DotConfig({"fmri_preprocessing": {"engine": "docker", "fmriprep": {}}})
        with self.assertRaises(FileNotFoundError):
            p.process_subject("0001", task="", dry_run=True)

        # invalid engine
        p.config = DotConfig(
            {"paths": {"bids_fmri_root": str(tmp / "bids")}, "fmri_preprocessing": {"engine": "x"}}
        )
        with self.assertRaises(ValueError):
            p.process_subject("0001", task="", dry_run=True)

        bids = tmp / "bids"
        bids.mkdir(parents=True, exist_ok=True)
        license_file = tmp / "license.txt"
        license_file.write_text("x", encoding="utf-8")
        p.config = DotConfig(
            {
                "paths": {"bids_fmri_root": str(bids)},
                "fmri_preprocessing": {"engine": "docker", "fmriprep": {}},
            }
        )
        with (
            patch(
                "fmri_pipeline.pipelines.fmri_preprocessing._resolve_fs_license_path",
                return_value=license_file,
            ),
            patch("fmri_pipeline.pipelines.fmri_preprocessing._require_executable"),
        ):
            p.process_subject("0001", task="", dry_run=True)

        self.assertTrue(p.logger.info.called)
