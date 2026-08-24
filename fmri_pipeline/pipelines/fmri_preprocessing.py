"""fMRI preprocessing pipeline (fMRIPrep-style orchestration)."""

from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from eeg_pipeline.pipelines.base import PipelineBase
from eeg_pipeline.utils.config.roots import (
    resolve_fmri_bids_root,
    resolve_fmri_deriv_root,
)

FS_LICENSE_ENV_VAR = "EEG_PIPELINE_FREESURFER_LICENSE"
FS_LICENSE_DEFAULT_PATH = "~/license.txt"
TEMPLATEFLOW_ENV_VAR = "TEMPLATEFLOW_HOME"
MACOS_METADATA_FILENAMES = {".DS_Store"}
MACOS_METADATA_PREFIX = "._"
BIDS_SANITIZED_SOURCE_MOUNT = "/bids_source"

_FMRIPREP_KEYS = {
    "aggregate_session_reports",
    "bids_filter_file",
    "bold2anat_dof",
    "bold2anat_init",
    "cifti_output",
    "clean_workdir",
    "dummy_scans",
    "dvars_spike_threshold",
    "extra_args",
    "fd_spike_threshold",
    "fs_license_file",
    "fs_no_reconall",
    "fs_subjects_dir",
    "ignore",
    "image",
    "level",
    "low_mem",
    "me_output_echos",
    "medial_surface_nan",
    "mem_mb",
    "no_msm",
    "nthreads",
    "omp_nthreads",
    "output_dir",
    "output_layout",
    "output_spaces",
    "random_seed",
    "skip_bids_validation",
    "skull_strip_fixed_seed",
    "skull_strip_template",
    "slice_time_ref",
    "stop_on_first_crash",
    "subject_anatomical_reference",
    "task_id",
    "work_dir",
}
_FMRIPREP_IGNORE_VALUES = {
    "fieldmaps",
    "slicetiming",
    "sbref",
    "t2w",
    "flair",
    "fmap-jacobian",
}


def _validate_fmriprep_config(settings: dict[str, Any]) -> None:
    """Fail before launching when settings cannot describe fMRIPrep 25.2."""
    unknown = sorted(set(settings) - _FMRIPREP_KEYS)
    if unknown:
        raise ValueError(f"Unknown fmri_preprocessing.fmriprep key(s): {unknown}")

    if not str(settings.get("image", "nipreps/fmriprep:25.2.5")).strip():
        raise ValueError("fmri_preprocessing.fmriprep.image must not be empty")

    output_spaces = settings.get("output_spaces", ["MNI152NLin2009cAsym", "T1w"])
    if not isinstance(output_spaces, list) or not output_spaces:
        raise ValueError("fmriprep.output_spaces must be a non-empty YAML list")
    if not all(isinstance(space, str) and space.strip() for space in output_spaces):
        raise ValueError("fmriprep.output_spaces entries must be non-empty strings")

    ignored = settings.get("ignore", []) or []
    if not isinstance(ignored, list):
        raise ValueError("fmriprep.ignore must be a YAML list")
    invalid_ignored = sorted(set(ignored) - _FMRIPREP_IGNORE_VALUES)
    if invalid_ignored:
        raise ValueError(
            f"Unsupported fmriprep.ignore value(s): {invalid_ignored}; "
            f"allowed: {sorted(_FMRIPREP_IGNORE_VALUES)}"
        )

    choices = {
        "level": ({"minimal", "resampling", "full"}, "full"),
        "cifti_output": ({None, "91k", "170k"}, None),
        "output_layout": ({"bids"}, "bids"),
        "subject_anatomical_reference": (
            {"first-lex", "unbiased", "sessionwise"},
            "first-lex",
        ),
        "bold2anat_init": ({"auto", "t1w", "t2w", "header"}, "t1w"),
        "bold2anat_dof": ({6, 9, 12}, 6),
    }
    for key, (allowed, default) in choices.items():
        value = settings.get(key, default)
        if value not in allowed:
            raise ValueError(
                f"fmriprep.{key} must be one of {sorted(map(str, allowed))}, " f"got {value!r}"
            )

    nonnegative = ("nthreads", "mem_mb", "random_seed")
    for key in nonnegative:
        value = int(settings.get(key, 0) or 0)
        if value < 0:
            raise ValueError(f"fmriprep.{key} must be >= 0")

    omp_nthreads = int(settings.get("omp_nthreads", 1))
    if omp_nthreads < 1:
        raise ValueError("fmriprep.omp_nthreads must be >= 1")
    if settings.get("skull_strip_fixed_seed", True) and omp_nthreads != 1:
        raise ValueError(
            "fmriprep.skull_strip_fixed_seed requires omp_nthreads=1 for "
            "run-to-run reproducibility"
        )

    dummy_scans = settings.get("dummy_scans")
    if dummy_scans is not None and int(dummy_scans) < 0:
        raise ValueError("fmriprep.dummy_scans must be null (auto) or >= 0")

    aggregate_reports = int(settings.get("aggregate_session_reports", 4))
    if aggregate_reports < 1:
        raise ValueError("fmriprep.aggregate_session_reports must be >= 1")

    slice_time_ref = float(settings.get("slice_time_ref", 0.5))
    if not 0.0 <= slice_time_ref <= 1.0:
        raise ValueError("fmriprep.slice_time_ref must be in [0, 1]")
    for key, default in (("fd_spike_threshold", 0.5), ("dvars_spike_threshold", 1.5)):
        if float(settings.get(key, default)) <= 0:
            raise ValueError(f"fmriprep.{key} must be > 0")


def _require_supported_container_host(workflow_name: str) -> None:
    if platform.system() == "Windows":
        raise RuntimeError(
            f"{workflow_name} is not supported on native Windows. "
            "Use WSL2 or run this workflow from macOS/Linux because it launches containerized neuroimaging tooling directly."
        )


def _require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable not found on PATH: {name}")


def _resolve_path(value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    trimmed = value.strip()
    if trimmed == "":
        return None
    return Path(trimmed).expanduser().resolve()


def _resolve_fs_license_path(config: Any, fmriprep_cfg: dict[str, Any]) -> Path:
    """Resolve FreeSurfer license path from config, environment, or default."""
    configured = _resolve_path(fmriprep_cfg.get("fs_license_file")) or _resolve_path(
        config.get("paths.freesurfer_license")
    )
    if configured is not None:
        return configured
    env_path = _resolve_path(os.getenv(FS_LICENSE_ENV_VAR))
    if env_path is not None:
        return env_path
    return Path(FS_LICENSE_DEFAULT_PATH).expanduser().resolve()


def _resolve_templateflow_home() -> Optional[Path]:
    templateflow_home = _resolve_path(os.getenv(TEMPLATEFLOW_ENV_VAR))
    if templateflow_home is None:
        return None
    if not templateflow_home.exists():
        raise FileNotFoundError(f"{TEMPLATEFLOW_ENV_VAR} does not exist: {templateflow_home}")
    return templateflow_home


def _is_macos_metadata_path(path: Path) -> bool:
    name = path.name
    return name.startswith(MACOS_METADATA_PREFIX) or name in MACOS_METADATA_FILENAMES


def _dataset_has_macos_metadata(dataset_root: Path) -> bool:
    return any(_is_macos_metadata_path(path) for path in dataset_root.rglob("*"))


def _create_sanitized_bids_view(
    bids_dir: Path,
    source_mount_path: str,
) -> tuple[Path, tempfile.TemporaryDirectory, int]:
    temp_dir = tempfile.TemporaryDirectory(prefix="eeg_pipeline_bids_")
    sanitized_root = Path(temp_dir.name) / "bids"
    sanitized_root.mkdir(parents=True, exist_ok=True)

    skipped_files = 0
    for source_path in bids_dir.rglob("*"):
        relative_path = source_path.relative_to(bids_dir)
        if _is_macos_metadata_path(source_path):
            skipped_files += 1
            continue

        target_path = sanitized_root / relative_path
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        container_source_path = Path(source_mount_path) / relative_path
        os.symlink(str(container_source_path), str(target_path))

    return sanitized_root, temp_dir, skipped_files


def _resolve_bids_mount_root(
    bids_dir: Path,
    logger: Any,
) -> tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    if not _dataset_has_macos_metadata(bids_dir):
        return bids_dir, None

    sanitized_root, temp_dir, skipped_files = _create_sanitized_bids_view(
        bids_dir, BIDS_SANITIZED_SOURCE_MOUNT
    )
    logger.warning(
        "Detected %d macOS metadata files (._*, .DS_Store); using sanitized BIDS mount: %s",
        skipped_files,
        sanitized_root,
    )
    return sanitized_root, temp_dir


def _stream_subprocess(
    cmd: List[str],
    logger: Any,
    *,
    env: Optional[dict] = None,
) -> None:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        logger.info(line.rstrip("\n"))
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed with exit code {rc}: {shlex.join(cmd)}")


class FmriPreprocessingPipeline(PipelineBase):
    """Run fMRIPrep in a container for each subject."""

    def __init__(self, config: Optional[Any] = None):
        super().__init__(name="fmri_preprocessing", config=config)

    def _resolve_task_is_rest(self) -> bool:
        return bool(self.config.get("fmri_preprocessing.task_is_rest", False))

    def _resolve_pipeline_deriv_root(self) -> Path:
        return resolve_fmri_deriv_root(
            self.config,
            task_is_rest=self._resolve_task_is_rest(),
        )

    def _validate_batch_inputs(self, subjects: List[str], task: Optional[str]) -> str:
        if not subjects:
            raise ValueError("No subjects specified")
        return task or ""

    def process_subject(
        self,
        subject: str,
        task: str,  # unused (fMRIPrep processes all tasks unless filtered)
        *,
        progress: Any = None,
        dry_run: bool = False,
        **_kwargs: Any,
    ) -> None:
        subject_id = str(subject)
        if subject_id.startswith("sub-"):
            subject_id = subject_id[4:]
        subj_label = f"sub-{subject_id}"
        if progress is not None and hasattr(progress, "subject_start"):
            progress.subject_start(subj_label)
        success = False
        bids_mount_tmp: Optional[tempfile.TemporaryDirectory] = None
        try:
            try:
                fmri_root = resolve_fmri_bids_root(
                    self.config,
                    task_is_rest=self._resolve_task_is_rest(),
                )
            except ValueError as exc:
                raise FileNotFoundError("fMRI BIDS root is not configured.") from exc

            engine = self.config.get("fmri_preprocessing.engine", "docker")
            if engine not in {"docker", "apptainer"}:
                raise ValueError("fmri_preprocessing.engine must be 'docker' or 'apptainer'")

            fmriprep_cfg = self.config.get("fmri_preprocessing.fmriprep", {}) or {}
            if not isinstance(fmriprep_cfg, dict):
                raise TypeError("fmri_preprocessing.fmriprep must be a YAML mapping")
            _validate_fmriprep_config(fmriprep_cfg)
            image = fmriprep_cfg.get("image", "nipreps/fmriprep:25.2.5")

            bids_dir = Path(str(fmri_root)).expanduser().resolve()
            if not bids_dir.exists():
                raise FileNotFoundError(f"fMRI BIDS root does not exist: {bids_dir}")
            bids_mount_root, bids_mount_tmp = _resolve_bids_mount_root(bids_dir, self.logger)
            needs_sanitized_source_mount = bids_mount_tmp is not None

            output_dir = _resolve_path(fmriprep_cfg.get("output_dir"))
            if output_dir is None:
                # BIDS layout writes derivatives and the subject report directly here.
                output_dir = self.deriv_root / "preprocessed" / "fmri"
            work_dir = _resolve_path(fmriprep_cfg.get("work_dir")) or (
                self.deriv_root / "work" / "fmriprep"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            work_dir.mkdir(parents=True, exist_ok=True)

            fs_license = _resolve_fs_license_path(self.config, fmriprep_cfg)
            if not fs_license.exists():
                raise FileNotFoundError(f"FreeSurfer license file not found: {fs_license}")

            fs_subjects_dir = _resolve_path(fmriprep_cfg.get("fs_subjects_dir"))

            output_spaces = fmriprep_cfg.get("output_spaces", ["MNI152NLin2009cAsym", "T1w"])
            ignore = fmriprep_cfg.get("ignore", []) or []
            bids_filter_file = _resolve_path(fmriprep_cfg.get("bids_filter_file"))

            skip_bids_validation = bool(fmriprep_cfg.get("skip_bids_validation", False))
            clean_workdir = bool(fmriprep_cfg.get("clean_workdir", True))
            stop_on_first_crash = bool(fmriprep_cfg.get("stop_on_first_crash", False))
            fs_no_reconall = bool(fmriprep_cfg.get("fs_no_reconall", False))

            mem_mb = int(fmriprep_cfg.get("mem_mb", 0) or 0)

            # Additional fMRIPrep options
            nthreads = int(fmriprep_cfg.get("nthreads", 0) or 0)
            omp_nthreads = int(fmriprep_cfg.get("omp_nthreads", 1))
            low_mem = bool(fmriprep_cfg.get("low_mem", False))
            subject_anatomical_reference = fmriprep_cfg.get(
                "subject_anatomical_reference", "first-lex"
            )
            cifti_output = fmriprep_cfg.get("cifti_output")
            level = fmriprep_cfg.get("level", "full")
            output_layout = fmriprep_cfg.get("output_layout", "bids")
            aggregate_session_reports = int(fmriprep_cfg.get("aggregate_session_reports", 4))
            skull_strip_template = fmriprep_cfg.get("skull_strip_template", "OASIS30ANTs")
            skull_strip_fixed_seed = bool(fmriprep_cfg.get("skull_strip_fixed_seed", True))
            random_seed = int(fmriprep_cfg.get("random_seed", 42))
            dummy_scans = fmriprep_cfg.get("dummy_scans")
            bold2anat_init = fmriprep_cfg.get("bold2anat_init", "t1w")
            bold2anat_dof = int(fmriprep_cfg.get("bold2anat_dof", 6))
            slice_time_ref = float(fmriprep_cfg.get("slice_time_ref", 0.5))
            fd_spike_threshold = float(fmriprep_cfg.get("fd_spike_threshold", 0.5))
            dvars_spike_threshold = float(fmriprep_cfg.get("dvars_spike_threshold", 1.5))
            me_output_echos = bool(fmriprep_cfg.get("me_output_echos", False))
            medial_surface_nan = bool(fmriprep_cfg.get("medial_surface_nan", False))
            no_msm = bool(fmriprep_cfg.get("no_msm", False))
            task_id = fmriprep_cfg.get("task_id")

            extra_args = str(fmriprep_cfg.get("extra_args", "") or "").strip()
            extra_tokens: List[str] = shlex.split(extra_args) if extra_args else []

            if progress is not None and hasattr(progress, "step"):
                progress.step("Prepare fMRIPrep command")

            participant_args = [
                "/data",
                "/out",
                "participant",
                "--participant-label",
                subject_id,
                "--work-dir",
                "/work",
                "--fs-license-file",
                "/license.txt",
            ]

            if output_spaces:
                participant_args += ["--output-spaces", *list(output_spaces)]
            if ignore:
                participant_args += ["--ignore", *list(ignore)]
            if bids_filter_file is not None:
                participant_args += ["--bids-filter-file", "/bids_filter.json"]
            if skip_bids_validation:
                participant_args += ["--skip-bids-validation"]
            if clean_workdir:
                participant_args += ["--clean-workdir"]
            if stop_on_first_crash:
                participant_args += ["--stop-on-first-crash"]
            if fs_no_reconall:
                participant_args += ["--fs-no-reconall"]

            if mem_mb > 0:
                participant_args += ["--mem-mb", str(mem_mb)]

            # Additional fMRIPrep options
            if nthreads > 0:
                participant_args += ["--nthreads", str(nthreads)]
            participant_args += ["--omp-nthreads", str(omp_nthreads)]
            if low_mem:
                participant_args += ["--low-mem"]
            participant_args += [
                "--subject-anatomical-reference",
                str(subject_anatomical_reference),
            ]
            if cifti_output:
                participant_args += ["--cifti-output", str(cifti_output)]
            participant_args += ["--level", str(level)]
            participant_args += ["--output-layout", str(output_layout)]
            participant_args += [
                "--aggregate-session-reports",
                str(aggregate_session_reports),
            ]
            participant_args += ["--skull-strip-template", str(skull_strip_template)]
            if skull_strip_fixed_seed:
                participant_args += ["--skull-strip-fixed-seed"]
            participant_args += ["--random-seed", str(random_seed)]
            if dummy_scans is not None:
                participant_args += ["--dummy-scans", str(int(dummy_scans))]
            participant_args += ["--bold2anat-init", str(bold2anat_init)]
            participant_args += ["--bold2anat-dof", str(bold2anat_dof)]
            participant_args += ["--slice-time-ref", str(slice_time_ref)]
            participant_args += [
                "--fd-spike-threshold",
                str(fd_spike_threshold),
            ]
            participant_args += [
                "--dvars-spike-threshold",
                str(dvars_spike_threshold),
            ]
            if me_output_echos:
                participant_args += ["--me-output-echos"]
            if medial_surface_nan:
                participant_args += ["--medial-surface-nan"]
            if no_msm:
                participant_args += ["--no-msm"]
            if task_id:
                participant_args += ["--task-id", task_id]

            if extra_tokens:
                managed_options = {token for token in participant_args if token.startswith("--")}
                conflicts = sorted(
                    {
                        token.partition("=")[0]
                        for token in extra_tokens
                        if token.partition("=")[0] in managed_options
                    }
                )
                if conflicts:
                    raise ValueError(
                        "fmriprep.extra_args duplicates managed option(s): "
                        f"{conflicts}. Set the corresponding YAML key instead."
                    )
                participant_args += extra_tokens

            if engine == "docker":
                executable_name = "docker"
                user_args: List[str] = []
                if hasattr(os, "getuid") and hasattr(os, "getgid"):
                    user_args = ["--user", f"{os.getuid()}:{os.getgid()}"]
                cmd: List[str] = [
                    "docker",
                    "run",
                    "--rm",
                    *user_args,
                    "-v",
                    f"{bids_mount_root}:/data:ro",
                    "-v",
                    f"{output_dir}:/out",
                    "-v",
                    f"{work_dir}:/work",
                    "-v",
                    f"{fs_license}:/license.txt:ro",
                ]
                if needs_sanitized_source_mount:
                    cmd += ["-v", f"{bids_dir}:{BIDS_SANITIZED_SOURCE_MOUNT}:ro"]
                if bids_filter_file is not None:
                    cmd += ["-v", f"{bids_filter_file}:/bids_filter.json:ro"]
                if fs_subjects_dir is not None:
                    cmd += ["-v", f"{fs_subjects_dir}:/fs"]
                    participant_args += ["--fs-subjects-dir", "/fs"]
                cmd += [image]
                cmd += participant_args
            else:
                executable_name = "apptainer"
                templateflow_home = _resolve_templateflow_home()
                cmd = [
                    "apptainer",
                    "run",
                    "--cleanenv",
                    "-B",
                    f"{bids_mount_root}:/data",
                    "-B",
                    f"{output_dir}:/out",
                    "-B",
                    f"{work_dir}:/work",
                    "-B",
                    f"{fs_license}:/license.txt",
                ]
                if templateflow_home is not None:
                    cmd += [
                        "-B",
                        f"{templateflow_home}:{templateflow_home}",
                        "--env",
                        f"{TEMPLATEFLOW_ENV_VAR}={templateflow_home}",
                    ]
                if needs_sanitized_source_mount:
                    cmd += ["-B", f"{bids_dir}:{BIDS_SANITIZED_SOURCE_MOUNT}"]
                if bids_filter_file is not None:
                    cmd += ["-B", f"{bids_filter_file}:/bids_filter.json"]
                if fs_subjects_dir is not None:
                    cmd += ["-B", f"{fs_subjects_dir}:/fs"]
                    participant_args += ["--fs-subjects-dir", "/fs"]
                cmd += [image]
                cmd += participant_args

            cmd_str = shlex.join(cmd)
            self.logger.info("fMRIPrep command: %s", cmd_str)

            if dry_run:
                success = True
                return

            _require_supported_container_host("fMRI preprocessing")
            _require_executable(executable_name)

            if progress is not None and hasattr(progress, "step"):
                progress.step("Run fMRIPrep")

            subject_logger = self.get_subject_logger(subject_id)
            _stream_subprocess(cmd, subject_logger)
            reports = sorted(output_dir.glob(f"{subj_label}*.html"))
            if not reports:
                raise FileNotFoundError(
                    f"fMRIPrep completed without a visual report for {subj_label} "
                    f"under {output_dir}"
                )
            success = True
        finally:
            if bids_mount_tmp is not None:
                bids_mount_tmp.cleanup()
            if progress is not None and hasattr(progress, "subject_done"):
                progress.subject_done(subj_label, success=success)
