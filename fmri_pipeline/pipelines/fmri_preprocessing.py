"""fMRI preprocessing pipeline (fMRIPrep-style orchestration)."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import os
from pathlib import Path
from typing import Any, List, Optional

from eeg_pipeline.pipelines.base import PipelineBase


def _require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required executable not found on PATH: {name}"
        )


def _resolve_path(value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    trimmed = value.strip()
    if trimmed == "":
        return None
    return Path(trimmed).expanduser().resolve()


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
        subj_label = f"sub-{subject}"
        if progress is not None and hasattr(progress, "subject_start"):
            progress.subject_start(subj_label)
        success = False
        try:
            fmri_root = self.config.get("paths.bids_fmri_root")
            if not fmri_root:
                raise ValueError("Missing required config value: paths.bids_fmri_root")

            engine = self.config.get("fmri_preprocessing.engine", "docker")
            if engine not in {"docker", "apptainer"}:
                raise ValueError(
                    "fmri_preprocessing.engine must be 'docker' or 'apptainer'"
                )

            fmriprep_cfg = self.config.get("fmri_preprocessing.fmriprep", {}) or {}
            image = fmriprep_cfg.get("image", "nipreps/fmriprep:23.2.1")

            bids_dir = Path(str(fmri_root)).expanduser().resolve()
            if not bids_dir.exists():
                raise FileNotFoundError(f"fMRI BIDS root does not exist: {bids_dir}")

            output_dir = _resolve_path(fmriprep_cfg.get("output_dir"))
            if output_dir is None:
                output_dir = self.deriv_root / "preprocessed" / subj_label / "fmri"
            work_dir = _resolve_path(fmriprep_cfg.get("work_dir")) or (
                self.deriv_root / "work" / "fmriprep"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            work_dir.mkdir(parents=True, exist_ok=True)

            fs_license = (
                _resolve_path(fmriprep_cfg.get("fs_license_file"))
                or _resolve_path(self.config.get("paths.freesurfer_license"))
            )
            if fs_license is None:
                raise ValueError(
                    "Missing FreeSurfer license file. Set fmri_preprocessing.fmriprep.fs_license_file "
                    "or paths.freesurfer_license (default: eeg_pipeline/licenses/license_freesurfer.txt)."
                )
            if not fs_license.exists():
                raise FileNotFoundError(
                    f"FreeSurfer license file not found: {fs_license}"
                )

            fs_subjects_dir = _resolve_path(fmriprep_cfg.get("fs_subjects_dir"))

            output_spaces = fmriprep_cfg.get(
                "output_spaces", ["MNI152NLin2009cAsym", "T1w"]
            )
            ignore = fmriprep_cfg.get("ignore", []) or []
            bids_filter_file = _resolve_path(fmriprep_cfg.get("bids_filter_file"))

            use_aroma = bool(fmriprep_cfg.get("use_aroma", False))
            skip_bids_validation = bool(
                fmriprep_cfg.get("skip_bids_validation", False)
            )
            clean_workdir = bool(fmriprep_cfg.get("clean_workdir", True))
            stop_on_first_crash = bool(fmriprep_cfg.get("stop_on_first_crash", False))
            fs_no_reconall = bool(fmriprep_cfg.get("fs_no_reconall", False))

            mem_mb = int(fmriprep_cfg.get("mem_mb", 0) or 0)

            # Additional fMRIPrep options
            nthreads = int(fmriprep_cfg.get("nthreads", 0) or 0)
            omp_nthreads = int(fmriprep_cfg.get("omp_nthreads", 0) or 0)
            low_mem = bool(fmriprep_cfg.get("low_mem", False))
            longitudinal = bool(fmriprep_cfg.get("longitudinal", False))
            cifti_output = fmriprep_cfg.get("cifti_output")
            level = fmriprep_cfg.get("level")
            skull_strip_template = fmriprep_cfg.get("skull_strip_template")
            skull_strip_fixed_seed = bool(fmriprep_cfg.get("skull_strip_fixed_seed", False))
            random_seed = int(fmriprep_cfg.get("random_seed", 0) or 0)
            dummy_scans = int(fmriprep_cfg.get("dummy_scans", 0) or 0)
            bold2t1w_init = fmriprep_cfg.get("bold2t1w_init")
            bold2t1w_dof = int(fmriprep_cfg.get("bold2t1w_dof", 0) or 0)
            slice_time_ref = fmriprep_cfg.get("slice_time_ref")
            fd_spike_threshold = fmriprep_cfg.get("fd_spike_threshold")
            dvars_spike_threshold = fmriprep_cfg.get("dvars_spike_threshold")
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
                subject,
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
            if use_aroma:
                participant_args += ["--use-aroma"]
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
            if omp_nthreads > 0:
                participant_args += ["--omp-nthreads", str(omp_nthreads)]
            if low_mem:
                participant_args += ["--low-mem"]
            if longitudinal:
                participant_args += ["--longitudinal"]
            if cifti_output:
                participant_args += ["--cifti-output", str(cifti_output)]
            if level and level != "full":
                participant_args += ["--level", level]
            if skull_strip_template and skull_strip_template != "OASIS30ANTs":
                participant_args += ["--skull-strip-template", skull_strip_template]
            if skull_strip_fixed_seed:
                participant_args += ["--skull-strip-fixed-seed"]
            if random_seed > 0:
                participant_args += ["--random-seed", str(random_seed)]
            if dummy_scans > 0:
                participant_args += ["--dummy-scans", str(dummy_scans)]
            if bold2t1w_init and bold2t1w_init != "register":
                participant_args += ["--bold2t1w-init", bold2t1w_init]
            if bold2t1w_dof > 0 and bold2t1w_dof != 6:
                participant_args += ["--bold2t1w-dof", str(bold2t1w_dof)]
            if slice_time_ref is not None and slice_time_ref != 0.5:
                participant_args += ["--slice-time-ref", str(slice_time_ref)]
            if fd_spike_threshold is not None and fd_spike_threshold != 0.5:
                participant_args += ["--fd-spike-threshold", str(fd_spike_threshold)]
            if dvars_spike_threshold is not None and dvars_spike_threshold != 1.5:
                participant_args += ["--dvars-spike-threshold", str(dvars_spike_threshold)]
            if me_output_echos:
                participant_args += ["--me-output-echos"]
            if medial_surface_nan:
                participant_args += ["--medial-surface-nan"]
            if no_msm:
                participant_args += ["--no-msm"]
            if task_id:
                participant_args += ["--task-id", task_id]

            if extra_tokens:
                participant_args += extra_tokens

            if engine == "docker":
                _require_executable("docker")
                user_args: List[str] = []
                if hasattr(os, "getuid") and hasattr(os, "getgid"):
                    user_args = ["--user", f"{os.getuid()}:{os.getgid()}"]
                cmd: List[str] = [
                    "docker",
                    "run",
                    "--rm",
                    *user_args,
                    "-v",
                    f"{bids_dir}:/data:ro",
                    "-v",
                    f"{output_dir}:/out",
                    "-v",
                    f"{work_dir}:/work",
                    "-v",
                    f"{fs_license}:/license.txt:ro",
                ]
                if bids_filter_file is not None:
                    cmd += ["-v", f"{bids_filter_file}:/bids_filter.json:ro"]
                if fs_subjects_dir is not None:
                    cmd += ["-v", f"{fs_subjects_dir}:/fs"]
                    participant_args += ["--fs-subjects-dir", "/fs"]
                cmd += [image]
                cmd += participant_args
            else:
                _require_executable("apptainer")
                cmd = [
                    "apptainer",
                    "run",
                    "--cleanenv",
                    "-B",
                    f"{bids_dir}:/data",
                    "-B",
                    f"{output_dir}:/out",
                    "-B",
                    f"{work_dir}:/work",
                    "-B",
                    f"{fs_license}:/license.txt",
                ]
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

            if progress is not None and hasattr(progress, "step"):
                progress.step("Run fMRIPrep")

            subject_logger = self.get_subject_logger(
                subject, filename="fmri_preprocessing.log"
            )
            _stream_subprocess(cmd, subject_logger)
            success = True
        finally:
            if progress is not None and hasattr(progress, "subject_done"):
                progress.subject_done(subj_label, success=success)
