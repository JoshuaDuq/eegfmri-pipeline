"""fMRI CLI command (fMRIPrep-style preprocessing)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Optional

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_output_format_args,
    add_path_args,
    add_task_arg,
    create_progress_reporter,
)


def setup_fmri(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "fmri",
        help="fMRI preprocessing (containerized fMRIPrep-style)",
        description="Run fMRIPrep-style preprocessing for BIDS fMRI datasets (Docker/Apptainer)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["preprocess"],
        help="Operation to run (currently: preprocess)",
    )
    add_common_subject_args(parser)
    add_task_arg(parser)  # accepted for TUI consistency; not required by fMRIPrep
    add_output_format_args(parser)
    add_path_args(parser)

    grp = parser.add_argument_group("fMRIPrep options")
    grp.add_argument(
        "--engine",
        choices=["docker", "apptainer"],
        default=None,
        help="Container engine to run (default from config: fmri_preprocessing.engine)",
    )
    grp.add_argument(
        "--fmriprep-image",
        type=str,
        default=None,
        help="Docker image tag or apptainer URI/path (default from config)",
    )
    grp.add_argument(
        "--fmriprep-output-dir",
        type=str,
        default=None,
        help="Output directory (BIDS derivatives root); fMRIPrep writes into <output_dir>/fmriprep",
    )
    grp.add_argument(
        "--fmriprep-work-dir",
        type=str,
        default=None,
        help="Work directory (scratch)",
    )
    grp.add_argument(
        "--fs-license-file",
        type=str,
        default=None,
        help="FreeSurfer license.txt file path (falls back to paths.freesurfer_license or EEG_PIPELINE_FREESURFER_LICENSE)",
    )
    grp.add_argument(
        "--fs-subjects-dir",
        type=str,
        default=None,
        help="FreeSurfer SUBJECTS_DIR (optional)",
    )
    grp.add_argument(
        "--output-spaces",
        nargs="+",
        default=None,
        help="Output spaces (e.g., T1w MNI152NLin2009cAsym)",
    )
    grp.add_argument(
        "--ignore",
        nargs="+",
        default=None,
        help="Ignore some modalities (e.g., fieldmaps slicetiming)",
    )
    grp.add_argument(
        "--bids-filter-file",
        type=str,
        default=None,
        help="Optional BIDS filter file (JSON)",
    )
    grp.add_argument(
        "--use-aroma",
        dest="use_aroma",
        action="store_true",
        default=None,
        help="Enable ICA-AROMA",
    )
    grp.add_argument(
        "--no-use-aroma",
        dest="use_aroma",
        action="store_false",
        help="Disable ICA-AROMA",
    )
    grp.add_argument(
        "--skip-bids-validation",
        dest="skip_bids_validation",
        action="store_true",
        default=None,
        help="Skip bids-validator",
    )
    grp.add_argument(
        "--no-skip-bids-validation",
        dest="skip_bids_validation",
        action="store_false",
        help="Do not skip bids-validator",
    )
    grp.add_argument(
        "--clean-workdir",
        dest="clean_workdir",
        action="store_true",
        default=None,
        help="Clean workdir after a successful run",
    )
    grp.add_argument(
        "--no-clean-workdir",
        dest="clean_workdir",
        action="store_false",
        help="Do not clean workdir after a successful run",
    )
    grp.add_argument(
        "--stop-on-first-crash",
        dest="stop_on_first_crash",
        action="store_true",
        default=None,
        help="Stop on first crash",
    )
    grp.add_argument(
        "--no-stop-on-first-crash",
        dest="stop_on_first_crash",
        action="store_false",
        help="Do not stop on first crash",
    )
    grp.add_argument(
        "--fs-no-reconall",
        dest="fs_no_reconall",
        action="store_true",
        default=None,
        help="Disable FreeSurfer recon-all",
    )
    grp.add_argument(
        "--fs-reconall",
        dest="fs_no_reconall",
        action="store_false",
        help="Enable FreeSurfer recon-all",
    )
    grp.add_argument(
        "--mem-mb",
        type=int,
        default=None,
        help="Memory limit in MB",
    )
    grp.add_argument(
        "--fmriprep-extra-args",
        type=str,
        default=None,
        help="Raw additional fMRIPrep CLI arguments (parsed with shlex)",
    )

    # Additional fMRIPrep options
    grp2 = parser.add_argument_group("Additional fMRIPrep options")
    grp2.add_argument(
        "--nthreads",
        type=int,
        default=None,
        help="Max threads across all processes (0=auto)",
    )
    grp2.add_argument(
        "--omp-nthreads",
        type=int,
        default=None,
        help="Max threads per process (0=auto)",
    )
    grp2.add_argument(
        "--low-mem",
        action="store_true",
        default=None,
        help="Reduce memory usage",
    )
    grp2.add_argument(
        "--longitudinal",
        action="store_true",
        default=None,
        help="Create unbiased structural template",
    )
    grp2.add_argument(
        "--cifti-output",
        choices=["91k", "170k"],
        default=None,
        help="Output CIFTI dense timeseries (91k or 170k grayordinates)",
    )
    grp2.add_argument(
        "--level",
        choices=["minimal", "resampling", "full"],
        default=None,
        help="Processing level (default: full)",
    )
    grp2.add_argument(
        "--skull-strip-template",
        type=str,
        default=None,
        help="Template for skull-stripping (default: OASIS30ANTs)",
    )
    grp2.add_argument(
        "--skull-strip-fixed-seed",
        action="store_true",
        default=None,
        help="Fixed seed for skull-stripping reproducibility",
    )
    grp2.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for run-to-run replicability",
    )
    grp2.add_argument(
        "--dummy-scans",
        type=int,
        default=None,
        help="Number of non-steady state volumes (0=auto)",
    )
    grp2.add_argument(
        "--bold2t1w-init",
        choices=["register", "header"],
        default=None,
        help="BOLD to T1w initialization method",
    )
    grp2.add_argument(
        "--bold2t1w-dof",
        type=int,
        default=None,
        help="Degrees of freedom for BOLD→T1w (default: 6)",
    )
    grp2.add_argument(
        "--slice-time-ref",
        type=float,
        default=None,
        help="Slice timing reference (0=start, 0.5=middle, 1=end)",
    )
    grp2.add_argument(
        "--fd-spike-threshold",
        type=float,
        default=None,
        help="Framewise displacement threshold in mm (default: 0.5)",
    )
    grp2.add_argument(
        "--dvars-spike-threshold",
        type=float,
        default=None,
        help="Standardized DVARS threshold (default: 1.5)",
    )
    grp2.add_argument(
        "--me-output-echos",
        action="store_true",
        default=None,
        help="Output each echo separately (multi-echo)",
    )
    grp2.add_argument(
        "--medial-surface-nan",
        action="store_true",
        default=None,
        help="Fill medial surface with NaN",
    )
    grp2.add_argument(
        "--no-msm",
        action="store_true",
        default=None,
        help="Disable MSM-Sulc alignment to fsLR",
    )
    grp2.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Process only specific task ID",
    )

    return parser


def _subjects_from_bids_root(
    bids_root: Path, requested: Optional[List[str]]
) -> List[str]:
    if not bids_root.exists():
        raise FileNotFoundError(f"fMRI BIDS root does not exist: {bids_root}")

    discovered = sorted(
        p.name.replace("sub-", "")
        for p in bids_root.glob("sub-*")
        if p.is_dir()
    )
    if requested is None:
        return discovered

    missing = sorted(set(requested) - set(discovered))
    if missing:
        raise ValueError(
            f"Requested subject(s) not found in BIDS fMRI root: {missing}"
        )
    return requested


def _parse_group_arg(group: str) -> Optional[List[str]]:
    group = (group or "").strip()
    if group.lower() in {"all", "*", "@all"}:
        return None
    normalized = group.replace(";", ",").replace(" ", ",")
    values = [s.strip() for s in normalized.split(",") if s.strip()]
    return values or None


def _resolve_subjects(args: argparse.Namespace, bids_root: Path, config: Any) -> List[str]:
    requested: Optional[List[str]] = None

    if getattr(args, "group", None):
        requested = _parse_group_arg(args.group)
    elif getattr(args, "all_subjects", False):
        requested = None
    elif getattr(args, "subject", None):
        requested = list(dict.fromkeys(args.subject))
    else:
        cfg_subjects = getattr(config, "subjects", None) or []
        requested = list(cfg_subjects) if cfg_subjects else None

    return _subjects_from_bids_root(bids_root, requested)


def _update_fmri_config_from_args(args: argparse.Namespace, config: Any) -> None:
    fmri_cfg = config.setdefault("fmri_preprocessing", {})
    fmriprep_cfg = fmri_cfg.setdefault("fmriprep", {})

    if args.engine:
        fmri_cfg["engine"] = args.engine
    if args.fmriprep_image:
        fmriprep_cfg["image"] = args.fmriprep_image
    if args.fmriprep_output_dir:
        fmriprep_cfg["output_dir"] = args.fmriprep_output_dir
    if args.fmriprep_work_dir:
        fmriprep_cfg["work_dir"] = args.fmriprep_work_dir
    if args.fs_license_file:
        fmriprep_cfg["fs_license_file"] = args.fs_license_file
    if args.fs_subjects_dir:
        fmriprep_cfg["fs_subjects_dir"] = args.fs_subjects_dir
    if args.output_spaces is not None:
        fmriprep_cfg["output_spaces"] = list(args.output_spaces)
    if args.ignore is not None:
        fmriprep_cfg["ignore"] = list(args.ignore)
    if args.bids_filter_file:
        fmriprep_cfg["bids_filter_file"] = args.bids_filter_file

    if args.use_aroma is not None:
        fmriprep_cfg["use_aroma"] = bool(args.use_aroma)
    if args.skip_bids_validation is not None:
        fmriprep_cfg["skip_bids_validation"] = bool(args.skip_bids_validation)
    if args.clean_workdir is not None:
        fmriprep_cfg["clean_workdir"] = bool(args.clean_workdir)
    if args.stop_on_first_crash is not None:
        fmriprep_cfg["stop_on_first_crash"] = bool(args.stop_on_first_crash)
    if args.fs_no_reconall is not None:
        fmriprep_cfg["fs_no_reconall"] = bool(args.fs_no_reconall)

    if args.mem_mb is not None:
        fmriprep_cfg["mem_mb"] = int(args.mem_mb)
    if args.fmriprep_extra_args is not None:
        fmriprep_cfg["extra_args"] = args.fmriprep_extra_args

    # Additional fMRIPrep options
    if getattr(args, "nthreads", None) is not None:
        fmriprep_cfg["nthreads"] = int(args.nthreads)
    if getattr(args, "omp_nthreads", None) is not None:
        fmriprep_cfg["omp_nthreads"] = int(args.omp_nthreads)
    if getattr(args, "low_mem", None):
        fmriprep_cfg["low_mem"] = True
    if getattr(args, "longitudinal", None):
        fmriprep_cfg["longitudinal"] = True
    if getattr(args, "cifti_output", None):
        fmriprep_cfg["cifti_output"] = args.cifti_output
    if getattr(args, "level", None):
        fmriprep_cfg["level"] = args.level
    if getattr(args, "skull_strip_template", None):
        fmriprep_cfg["skull_strip_template"] = args.skull_strip_template
    if getattr(args, "skull_strip_fixed_seed", None):
        fmriprep_cfg["skull_strip_fixed_seed"] = True
    if getattr(args, "random_seed", None) is not None:
        fmriprep_cfg["random_seed"] = int(args.random_seed)
    if getattr(args, "dummy_scans", None) is not None:
        fmriprep_cfg["dummy_scans"] = int(args.dummy_scans)
    if getattr(args, "bold2t1w_init", None):
        fmriprep_cfg["bold2t1w_init"] = args.bold2t1w_init
    if getattr(args, "bold2t1w_dof", None) is not None:
        fmriprep_cfg["bold2t1w_dof"] = int(args.bold2t1w_dof)
    if getattr(args, "slice_time_ref", None) is not None:
        fmriprep_cfg["slice_time_ref"] = float(args.slice_time_ref)
    if getattr(args, "fd_spike_threshold", None) is not None:
        fmriprep_cfg["fd_spike_threshold"] = float(args.fd_spike_threshold)
    if getattr(args, "dvars_spike_threshold", None) is not None:
        fmriprep_cfg["dvars_spike_threshold"] = float(args.dvars_spike_threshold)
    if getattr(args, "me_output_echos", None):
        fmriprep_cfg["me_output_echos"] = True
    if getattr(args, "medial_surface_nan", None):
        fmriprep_cfg["medial_surface_nan"] = True
    if getattr(args, "no_msm", None):
        fmriprep_cfg["no_msm"] = True
    if getattr(args, "task_id", None):
        fmriprep_cfg["task_id"] = args.task_id


def run_fmri(args: argparse.Namespace, _subjects: List[str], config: Any) -> None:
    from fmri_pipeline.pipelines.fmri_preprocessing import FmriPreprocessingPipeline

    progress = create_progress_reporter(args)

    if args.bids_fmri_root:
        config.setdefault("paths", {})["bids_fmri_root"] = args.bids_fmri_root
    if args.deriv_root:
        config.setdefault("paths", {})["deriv_root"] = args.deriv_root

    _update_fmri_config_from_args(args, config)

    bids_fmri_root = config.get("paths.bids_fmri_root")
    if not bids_fmri_root:
        raise ValueError("Missing required config value: paths.bids_fmri_root")

    subjects = _resolve_subjects(args, Path(bids_fmri_root), config)

    pipeline = FmriPreprocessingPipeline(config=config)
    pipeline.run_batch(
        subjects=subjects,
        task=None,
        progress=progress,
        dry_run=bool(getattr(args, "dry_run", False)),
    )
