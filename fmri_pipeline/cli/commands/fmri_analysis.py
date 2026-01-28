"""fMRI analysis CLI command: first-level GLM + contrasts."""

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
    resolve_task,
)


def setup_fmri_analysis(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "fmri-analysis",
        help="fMRI analysis: first-level GLM + contrasts",
        description="Compute subject-level (first-level) fMRI contrasts between conditions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["first-level"],
        help="Operation to run (currently: first-level)",
    )

    add_common_subject_args(parser)
    add_task_arg(parser)
    add_output_format_args(parser)
    add_path_args(parser)

    in_group = parser.add_argument_group("Input selection")
    in_group.add_argument(
        "--input-source",
        choices=["fmriprep", "bids_raw"],
        default=None,
        help="Use preprocessed BOLD from fMRIPrep derivatives or raw BIDS BOLD (default from config if set)",
    )
    in_group.add_argument(
        "--fmriprep-space",
        type=str,
        default=None,
        help="fMRIPrep space for preprocessed BOLD (e.g., T1w, MNI152NLin2009cAsym)",
    )
    in_group.add_argument(
        "--require-fmriprep",
        dest="require_fmriprep",
        action="store_true",
        default=None,
        help="Fail if preprocessed BOLD is missing when input-source=fmriprep",
    )
    in_group.add_argument(
        "--no-require-fmriprep",
        dest="require_fmriprep",
        action="store_false",
        help="Allow falling back to raw BOLD when fMRIPrep outputs are missing",
    )
    in_group.add_argument(
        "--runs",
        nargs="+",
        type=int,
        default=None,
        help="Explicit run numbers to include (e.g., --runs 1 2 3). Omit to auto-detect.",
    )

    contrast_group = parser.add_argument_group("Contrast definition")
    contrast_group.add_argument(
        "--contrast-name",
        type=str,
        default=None,
        help="Contrast name used for output organization (default: pain_vs_nonpain)",
    )
    contrast_group.add_argument(
        "--contrast-type",
        choices=["t-test", "custom"],
        default=None,
        help="Contrast type (default from config if set)",
    )
    contrast_group.add_argument(
        "--cond-a-column",
        type=str,
        default="trial_type",
        help="Events column for condition A selection (default: trial_type)",
    )
    contrast_group.add_argument(
        "--cond-a-value",
        type=str,
        default=None,
        help="Events value for condition A selection (required unless --formula is provided)",
    )
    contrast_group.add_argument(
        "--cond-b-column",
        type=str,
        default="trial_type",
        help="Events column for condition B selection (default: trial_type)",
    )
    contrast_group.add_argument(
        "--cond-b-value",
        type=str,
        default=None,
        help="Events value for condition B selection (optional)",
    )
    contrast_group.add_argument(
        "--formula",
        type=str,
        default=None,
        help="Custom contrast formula (nilearn syntax), e.g. 'A - B' (used when contrast-type=custom)",
    )

    glm_group = parser.add_argument_group("GLM settings")
    glm_group.add_argument(
        "--hrf-model",
        choices=["spm", "flobs", "fir"],
        default=None,
        help="HRF model (default: spm)",
    )
    glm_group.add_argument(
        "--drift-model",
        choices=["none", "cosine", "polynomial"],
        default=None,
        help="Drift model (default: cosine)",
    )
    glm_group.add_argument(
        "--high-pass-hz",
        type=float,
        default=None,
        help="High-pass cutoff in Hz (default: 0.008)",
    )
    glm_group.add_argument(
        "--low-pass-hz",
        type=float,
        default=None,
        help="Optional low-pass cutoff in Hz (set 0 to disable; default: disabled)",
    )

    qc_group = parser.add_argument_group("Confounds / QC")
    qc_group.add_argument(
        "--confounds-strategy",
        choices=[
            "auto",
            "none",
            "motion6",
            "motion12",
            "motion24",
            "motion24+wmcsf",
            "motion24+wmcsf+fd",
        ],
        default=None,
        help="Which confound regressors to include (default: auto)",
    )
    qc_group.add_argument(
        "--write-design-matrix",
        dest="write_design_matrix",
        action="store_true",
        default=None,
        help="Write design matrices (TSV; PNG best-effort) into <output>/qc/",
    )
    qc_group.add_argument(
        "--no-write-design-matrix",
        dest="write_design_matrix",
        action="store_false",
        help="Do not write design matrices",
    )

    out_group = parser.add_argument_group("Output settings")
    out_group.add_argument(
        "--output-type",
        choices=["z-score", "t-stat", "cope", "beta"],
        default=None,
        help="Output map type (default: z-score)",
    )
    out_group.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: <deriv_root>/sub-*/fmri/first_level/task-*/contrast-*/)",
    )
    out_group.add_argument(
        "--resample-to-freesurfer",
        dest="resample_to_freesurfer",
        action="store_true",
        default=None,
        help="Resample output to FreeSurfer subject MRI space (requires FreeSurfer SUBJECTS_DIR)",
    )
    out_group.add_argument(
        "--no-resample-to-freesurfer",
        dest="resample_to_freesurfer",
        action="store_false",
        help="Do not resample output to FreeSurfer space",
    )
    out_group.add_argument(
        "--freesurfer-dir",
        type=str,
        default=None,
        help="FreeSurfer SUBJECTS_DIR (overrides paths.freesurfer_dir)",
    )

    return parser


def _subjects_from_bids_root(bids_root: Path, requested: Optional[List[str]]) -> List[str]:
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
    elif getattr(args, "subjects", None):
        requested = list(dict.fromkeys(args.subjects))
    elif getattr(args, "subject", None):
        requested = list(dict.fromkeys(args.subject))
    else:
        cfg_subjects = getattr(config, "subjects", None) or []
        requested = list(cfg_subjects) if cfg_subjects else None

    return _subjects_from_bids_root(bids_root, requested)


def _map_task_to_fmri(task: str) -> str:
    """Return task name for fMRI file matching (pass-through, no mapping)."""
    task = (task or "").strip()
    return task if task else "thermalactive"


def run_fmri_analysis(args: argparse.Namespace, _subjects: List[str], config: Any) -> None:
    from fmri_pipeline.analysis.contrast_builder import ContrastBuilderConfig
    from fmri_pipeline.pipelines.fmri_analysis import FmriAnalysisPipeline

    progress = create_progress_reporter(args)

    if args.bids_fmri_root:
        config.setdefault("paths", {})["bids_fmri_root"] = args.bids_fmri_root
    if args.deriv_root:
        config.setdefault("paths", {})["deriv_root"] = args.deriv_root
    if args.freesurfer_dir:
        config.setdefault("paths", {})["freesurfer_dir"] = args.freesurfer_dir

    bids_fmri_root = config.get("paths.bids_fmri_root")
    if not bids_fmri_root:
        raise ValueError("Missing required config value: paths.bids_fmri_root")

    base_task = resolve_task(args.task, config)
    fmri_task = _map_task_to_fmri(base_task)

    subjects = _resolve_subjects(args, Path(bids_fmri_root), config)

    # Defaults are intentionally conservative and match contrast_builder defaults.
    input_source = str(args.input_source or "fmriprep").strip().lower()
    if input_source not in {"fmriprep", "bids_raw"}:
        input_source = "fmriprep"

    drift_model = args.drift_model
    if drift_model == "none":
        drift_model = None

    low_pass_hz = args.low_pass_hz
    if low_pass_hz is not None and low_pass_hz <= 0:
        low_pass_hz = None

    contrast_name = str(args.contrast_name or "pain_vs_nonpain").strip() or "contrast"
    contrast_type = str(args.contrast_type or ("custom" if args.formula else "t-test")).strip()

    if contrast_type == "custom" and not (args.formula and str(args.formula).strip()):
        raise ValueError("contrast-type=custom requires --formula")

    if contrast_type != "custom" and not (args.cond_a_value and str(args.cond_a_value).strip()):
        raise ValueError("Missing required --cond-a-value (or use --contrast-type custom --formula ...)")

    confounds_strategy = str(args.confounds_strategy or "auto").strip().lower()
    if not confounds_strategy:
        confounds_strategy = "auto"
    write_design_matrix = bool(args.write_design_matrix) if args.write_design_matrix is not None else False

    cfg = ContrastBuilderConfig(
        enabled=True,
        input_source=input_source,
        fmriprep_space=str(args.fmriprep_space).strip() if args.fmriprep_space else "T1w",
        require_fmriprep=bool(args.require_fmriprep) if args.require_fmriprep is not None else True,
        contrast_type=contrast_type,
        condition1=None,
        condition2=None,
        condition_a_column=str(args.cond_a_column or "trial_type").strip(),
        condition_a_value=str(args.cond_a_value).strip() if args.cond_a_value else None,
        condition_b_column=str(args.cond_b_column or "trial_type").strip(),
        condition_b_value=str(args.cond_b_value).strip() if args.cond_b_value else None,
        formula=str(args.formula).strip() if args.formula else None,
        name=contrast_name,
        runs=list(args.runs) if args.runs else None,
        hrf_model=str(args.hrf_model or "spm").strip(),
        drift_model=str(drift_model).strip() if drift_model else None,
        high_pass_hz=float(args.high_pass_hz) if args.high_pass_hz is not None else 0.008,
        low_pass_hz=float(low_pass_hz) if low_pass_hz is not None else None,
        cluster_correction=False,
        cluster_p_threshold=0.001,
        output_type=str(args.output_type or "z-score").strip(),
        resample_to_freesurfer=bool(args.resample_to_freesurfer) if args.resample_to_freesurfer is not None else False,
        confounds_strategy=confounds_strategy,
        write_design_matrix=write_design_matrix,
    )

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    fs_dir = None
    if cfg.resample_to_freesurfer:
        fs_cfg = config.get("paths.freesurfer_dir")
        fs_dir = Path(str(fs_cfg)).expanduser().resolve() if fs_cfg else None

    pipeline = FmriAnalysisPipeline(config=config)
    pipeline.run_batch(
        subjects=subjects,
        task=fmri_task,
        progress=progress,
        dry_run=bool(getattr(args, "dry_run", False)),
        contrast_cfg=cfg,
        output_dir=out_dir,
        freesurfer_subjects_dir=fs_dir,
    )
