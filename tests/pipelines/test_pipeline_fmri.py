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
    def test_resample_success_and_mni_cached_and_mutated_cfg(self):
        from fmri_pipeline.pipelines.fmri_analysis import FmriAnalysisPipeline

        tmp = Path(tempfile.mkdtemp())
        bids_root = tmp / "bids"
        bids_root.mkdir(parents=True, exist_ok=True)

        p = object.__new__(FmriAnalysisPipeline)
        p.config = DotConfig({"paths": {"bids_fmri_root": str(bids_root), "freesurfer_dir": str(tmp / "fs")}})
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

        out_mni_loads = {"count": 0}

        def fake_load(path):
            if "space-MNI152NLin2009cAsym" in str(path):
                out_mni_loads["count"] += 1
            return "img"

        fake_builder = types.SimpleNamespace(
            build_contrast_from_runs_detailed=lambda **kwargs: (
                "img",
                {"output_type": "z_score"},
                SimpleNamespace(flm=SimpleNamespace(compute_contrast=lambda *a, **k: "x")),
                "def",
                None,
            ),
            resample_to_freesurfer=Mock(side_effect=lambda img, fs_dir: img),
            ContrastBuilderConfig=CBuilderCfg,
        )
        fake_plot = types.SimpleNamespace(FmriPlottingConfig=PlotCfg)
        fake_report = types.SimpleNamespace(run_fmri_plotting_and_report=lambda **kwargs: {"ok": True})
        fake_nib = types.SimpleNamespace(save=lambda img, path: Path(path).write_text("x", encoding="utf-8"), load=fake_load)

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
            p.process_subject("0001", "task", contrast_cfg=cfg, plotting_cfg=PlotCfg(), dry_run=False)
            p.process_subject("0001", "task", contrast_cfg=cfg, plotting_cfg=PlotCfg(), dry_run=False)

        self.assertEqual(cfg.fmriprep_space, "MNI152NLin2009cAsym")
        self.assertTrue(fake_builder.resample_to_freesurfer.called)
        self.assertGreaterEqual(out_mni_loads["count"], 1)

    def test_mni_save_and_contrast_compute_exceptions_swallowed(self):
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

        fake_builder = types.SimpleNamespace(build_contrast_from_runs_detailed=_build, resample_to_freesurfer=lambda i, d: i, ContrastBuilderConfig=Cfg)
        fake_plot = types.SimpleNamespace(FmriPlottingConfig=PlotCfg)
        fake_report = types.SimpleNamespace(run_fmri_plotting_and_report=lambda **kwargs: {"ok": True})
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
            p.process_subject("0001", "t", contrast_cfg=Cfg(), plotting_cfg=PlotCfg(), dry_run=False)

    def test_mni_build_exception_logs_warning(self):
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

        fake_builder = types.SimpleNamespace(build_contrast_from_runs_detailed=_build, resample_to_freesurfer=lambda i, d: i, ContrastBuilderConfig=Cfg)
        fake_plot = types.SimpleNamespace(FmriPlottingConfig=PlotCfg)
        fake_report = types.SimpleNamespace(run_fmri_plotting_and_report=lambda **kwargs: {"ok": True})
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
            p.process_subject("0001", "t", contrast_cfg=Cfg(), plotting_cfg=PlotCfg(), dry_run=False)

        self.assertTrue(p.logger.warning.called)

class TestFmriPreprocessingGapfill(unittest.TestCase):
    def test_constructor_and_missing_paths_errors(self):
        from fmri_pipeline.pipelines.fmri_preprocessing import FmriPreprocessingPipeline

        cfg = DotConfig({})
        with patch("fmri_pipeline.pipelines.fmri_preprocessing.PipelineBase.__init__", lambda self, name, config=None: (setattr(self, "name", name), setattr(self, "config", config or cfg), setattr(self, "logger", Mock()), setattr(self, "deriv_root", Path(tempfile.mkdtemp())))):
            p = FmriPreprocessingPipeline(config=cfg)
        self.assertEqual(p.name, "fmri_preprocessing")

        p = object.__new__(FmriPreprocessingPipeline)
        p.deriv_root = Path(tempfile.mkdtemp())
        p.logger = Mock()
        p.config = DotConfig({"paths": {"bids_fmri_root": str(Path(tempfile.mkdtemp()) / "missing")}, "fmri_preprocessing": {"engine": "docker", "fmriprep": {"fs_license_file": str(Path(tempfile.mkdtemp()) / "lic.txt")}}})
        with self.assertRaises(FileNotFoundError):
            p.process_subject("0001", task="", dry_run=True)

        bids = Path(tempfile.mkdtemp())
        bids.mkdir(parents=True, exist_ok=True)
        p.config = DotConfig({"paths": {"bids_fmri_root": str(bids)}, "fmri_preprocessing": {"engine": "docker", "fmriprep": {"fs_license_file": str(bids / "missing_license.txt")}}})
        with self.assertRaises(FileNotFoundError):
            p.process_subject("0001", task="", dry_run=True)

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

        with patch.dict("os.environ", {"EEG_PIPELINE_FREESURFER_LICENSE": str(lic)}), patch(
            "fmri_pipeline.pipelines.fmri_preprocessing._require_executable"
        ):
            p.process_subject("0001", task="", dry_run=True)


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
            fake_nib = types.SimpleNamespace(save=lambda img, path: Path(path).write_text("nii", encoding="utf-8"), load=lambda path: "img")

            with patch.dict(sys.modules, {"fmri_pipeline.analysis.contrast_builder": fake_builder, "nibabel": fake_nib}):
                p.process_subject("0001", task="task", contrast_cfg=contrast_cfg, plotting_cfg=None, dry_run=False)

            out_dir = p.deriv_root / "sub-0001" / "fmri" / "first_level" / "task-task" / "contrast-contrast"
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
            fake_nib = types.SimpleNamespace(save=lambda img, path: Path(path).write_text("nii", encoding="utf-8"), load=lambda path: "img")
            fake_plot_mod = types.SimpleNamespace(FmriPlottingConfig=FakePlotCfg)
            fake_report_mod = types.SimpleNamespace(run_fmri_plotting_and_report=lambda **kwargs: {"ok": True})

            with patch.dict(
                sys.modules,
                {
                    "fmri_pipeline.analysis.contrast_builder": fake_builder,
                    "nibabel": fake_nib,
                    "fmri_pipeline.analysis.plotting_config": fake_plot_mod,
                    "fmri_pipeline.analysis.reporting": fake_report_mod,
                },
            ):
                p.process_subject("0001", task="task", contrast_cfg=contrast_cfg, plotting_cfg=plotting_cfg, dry_run=False)

        def test_fmri_analysis_plotting_exception_and_sidecar_exception(self):
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
                space: str = "native"
                include_effect_size: bool = True
                include_standard_error: bool = True

                def normalized(self):
                    return self

            fake_builder = types.SimpleNamespace(
                build_contrast_from_runs_detailed=lambda **kwargs: ("img", {"output_type": "z_score"}, SimpleNamespace(flm=SimpleNamespace(compute_contrast=lambda *a, **k: "x")), "def", None),
                resample_to_freesurfer=lambda img, fs_dir: img,
                ContrastBuilderConfig=Cfg,
            )
            fake_nib = types.SimpleNamespace(save=lambda img, path: None, load=lambda path: "img")
            fake_plot = types.SimpleNamespace(FmriPlottingConfig=PlotCfg)
            fake_report = types.SimpleNamespace(
                run_fmri_plotting_and_report=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("plot-fail"))
            )

            with patch.dict(
                sys.modules,
                {
                    "fmri_pipeline.analysis.contrast_builder": fake_builder,
                    "nibabel": fake_nib,
                    "fmri_pipeline.analysis.plotting_config": fake_plot,
                    "fmri_pipeline.analysis.reporting": fake_report,
                },
            ):
                p.process_subject("0001", task="t", contrast_cfg=Cfg(), plotting_cfg=PlotCfg(), dry_run=False)
            self.assertTrue(p.logger.warning.called)

            # sidecar write error swallowed
            with patch.dict(sys.modules, {"fmri_pipeline.analysis.contrast_builder": fake_builder, "nibabel": fake_nib}), patch(
                "pathlib.Path.write_text", side_effect=RuntimeError("no-write")
            ):
                p.process_subject("0001", task="t", contrast_cfg=Cfg(), plotting_cfg=None, dry_run=False)

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

            with patch("fmri_pipeline.pipelines.fmri_preprocessing._require_executable"), patch(
                "fmri_pipeline.pipelines.fmri_preprocessing._stream_subprocess"
            ) as mock_stream:
                p.process_subject("0001", task="", progress=SimpleNamespace(subject_start=lambda *a, **k: None, subject_done=lambda *a, **k: None, step=lambda *a, **k: None), dry_run=False)
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
                run_trial_signature_extraction_for_subject=lambda **kwargs: {"output_dir": str(tmp / "out")},
            )
            with patch.dict(sys.modules, {"fmri_pipeline.analysis.trial_signatures": fake_mod}):
                p.process_subject(
                    "0001",
                    task="task",
                    bids_fmri_root=tmp,
                    trial_cfg=TrialSignatureExtractionConfig(),
                    output_dir=tmp / "out",
                    dry_run=False,
                    progress=SimpleNamespace(subject_start=lambda *a, **k: None, step=lambda *a, **k: None),
                )

class TestFmriCompletion(unittest.TestCase):
        def test_fmri_trial_signatures_init_discover_and_group_level(self):
            from fmri_pipeline.pipelines.fmri_trial_signatures import FmriTrialSignaturePipeline

            cfg = DotConfig({})
            with patch("fmri_pipeline.pipelines.fmri_trial_signatures.PipelineBase.__init__", lambda self, name, config=None: (setattr(self, "config", config or cfg), setattr(self, "deriv_root", Path(tempfile.mkdtemp())), setattr(self, "logger", Mock()))):
                p = FmriTrialSignaturePipeline(config=cfg)
            ext = p.deriv_root.parent / "external"
            ext.mkdir(parents=True, exist_ok=True)
            self.assertEqual(p._discover_signature_root().resolve(), ext.resolve())  # type: ignore[union-attr]
            self.assertIsNone(p.run_group_level(["0001"], task="t"))

            p.config = DotConfig({"paths": {"signature_dir": "/path/does/not/exist"}})
            self.assertIsNone(p._discover_signature_root())

            class BadCfg:
                def get(self, *_a, **_k):
                    raise RuntimeError("bad")

            p.config = BadCfg()
            self.assertIsNotNone(p._discover_signature_root())
            p.deriv_root = object()
            self.assertIsNone(p._discover_signature_root())

        def test_fmri_analysis_init_and_error_paths(self):
            from fmri_pipeline.pipelines.fmri_analysis import FmriAnalysisPipeline

            cfg = DotConfig({"paths": {}})
            with patch("fmri_pipeline.pipelines.fmri_analysis.PipelineBase.__init__", lambda self, name, config=None: (setattr(self, "config", config or cfg), setattr(self, "deriv_root", Path(tempfile.mkdtemp())), setattr(self, "logger", Mock()))):
                p = FmriAnalysisPipeline(config=cfg)

            # discover signature fallback
            ext = p.deriv_root.parent / "external"
            ext.mkdir(parents=True, exist_ok=True)
            self.assertEqual(
                p._discover_signature_root().resolve(), ext.resolve()  # type: ignore[union-attr]
            )

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
            p.deriv_root = object()  # force fallback path failure branch
            self.assertIsNone(p._discover_signature_root())

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

            with patch.dict(sys.modules, {"fmri_pipeline.analysis.contrast_builder": fake_builder, "nibabel": fake_nib}):
                with self.assertRaises(ValueError):
                    p.process_subject("0001", "t", contrast_cfg=Cfg(), dry_run=False)

            fs_dir = tmp / "fs"
            fs_dir.mkdir(parents=True, exist_ok=True)
            p.config = DotConfig({"paths": {"bids_fmri_root": str(bids_root), "freesurfer_dir": str(fs_dir)}})
            with patch.dict(sys.modules, {"fmri_pipeline.analysis.contrast_builder": fake_builder, "nibabel": fake_nib}):
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

            fake_builder = types.SimpleNamespace(build_contrast_from_runs_detailed=_build, resample_to_freesurfer=lambda i, d: i, ContrastBuilderConfig=Cfg)
            fake_nib = types.SimpleNamespace(save=lambda img, path: Path(path).write_text("x", encoding="utf-8"), load=lambda path: "img")
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
                progress = SimpleNamespace(subject_start=lambda *a, **k: None, step=lambda *a, **k: None, subject_done=lambda *a, **k: done.__setitem__("ok", True))
                p.process_subject("0001", "t", contrast_cfg=Cfg(), plotting_cfg=PlotCfg(), dry_run=False, progress=progress)
            self.assertGreaterEqual(calls["n"], 2)
            self.assertTrue(done["ok"])

        def test_fmri_preprocessing_stream_and_apptainer_dry(self):
            from fmri_pipeline.pipelines.fmri_preprocessing import _stream_subprocess, FmriPreprocessingPipeline

            class P:
                def __init__(self, rc=0):
                    self.stdout = iter(["line1\n", "line2\n"])
                    self._rc = rc

                def wait(self):
                    return self._rc

            with patch("fmri_pipeline.pipelines.fmri_preprocessing.subprocess.Popen", return_value=P(0)):
                _stream_subprocess(["cmd"], Mock())
            with patch("fmri_pipeline.pipelines.fmri_preprocessing.subprocess.Popen", return_value=P(1)):
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
            with self.assertRaises(ValueError):
                p.process_subject("0001", task="", dry_run=True)

            # invalid engine
            p.config = DotConfig({"paths": {"bids_fmri_root": str(tmp / "bids")}, "fmri_preprocessing": {"engine": "x"}})
            with self.assertRaises(ValueError):
                p.process_subject("0001", task="", dry_run=True)

            bids = tmp / "bids"
            bids.mkdir(parents=True, exist_ok=True)
            p.config = DotConfig({"paths": {"bids_fmri_root": str(bids)}, "fmri_preprocessing": {"engine": "docker", "fmriprep": {}}})
            with self.assertRaises(ValueError):
                p.process_subject("0001", task="", dry_run=True)
