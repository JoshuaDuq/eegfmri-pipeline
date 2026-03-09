"""fMRI analysis CLI command: first-level, second-level, and trial-wise analyses."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_output_format_args,
    add_path_args,
    add_task_arg,
    create_progress_reporter,
    resolve_task,
)
from eeg_pipeline.utils.config.overrides import apply_set_overrides
from fmri_pipeline.cli.commands.subject_selection import resolve_subjects
from fmri_pipeline.utils.config import apply_fmri_config_defaults


def setup_fmri_analysis(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "fmri-analysis",
        help="fMRI analysis: first-level, second-level, and trial-wise models",
        description=(
            "Run subject-level first-level contrasts, explicit second-level "
            "group inference, or trial-wise beta/signature extraction."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["first-level", "second-level", "beta-series", "lss"],
        help="Operation to run (first-level | second-level | beta-series | lss)",
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
        help="Contrast name used for output organization (default: contrast)",
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
        default=None,
        help="Events column for condition A selection (default from config if set, else trial_type)",
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
        default=None,
        help="Events column for condition B selection (default from config if set, else trial_type)",
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
    contrast_group.add_argument(
        "--condition-scope-trial-types",
        nargs="+",
        default=None,
        metavar="TT",
        help=(
            "Optional: restrict which events.tsv rows are eligible for condition A/B selection "
            "(matched against --condition-scope-column). Use 'all' to disable scoping."
        ),
    )
    contrast_group.add_argument(
        "--condition-scope-column",
        type=str,
        default=None,
        help="Events column used for --condition-scope-trial-types. Required when condition scoping is enabled.",
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
    glm_group.add_argument(
        "--smoothing-fwhm",
        type=float,
        default=None,
        help="Spatial smoothing kernel FWHM in mm (default from config if set; set 0 to disable)",
    )
    glm_group.add_argument(
        "--events-to-model",
        type=str,
        default=None,
        help=(
            "Optional comma-separated allow-list of events.tsv values to include in the GLM. "
            "Example: condition_a,condition_b,rating"
        ),
    )
    glm_group.add_argument(
        "--events-to-model-column",
        type=str,
        default=None,
        help="Events column used for values passed via --events-to-model. Required when events scoping is enabled.",
    )
    glm_group.add_argument(
        "--stim-phases-to-model",
        type=str,
        default=None,
        help=(
            "Optional comma-separated allow-list of event phase values to include when events.tsv has the configured phase column. "
            "If unset, no phase scoping is applied. "
            "Use 'all' to disable phase scoping."
        ),
    )
    glm_group.add_argument(
        "--phase-column",
        type=str,
        default=None,
        help=(
            "Events column used for phase scoping values passed via --stim-phases-to-model "
            "and required when phase scoping is enabled."
        ),
    )
    glm_group.add_argument(
        "--phase-scope-column",
        type=str,
        default=None,
        help=(
            "Events column used to scope phase filtering to a subset of rows. "
            "Required when --phase-scope-value is set."
        ),
    )
    glm_group.add_argument(
        "--phase-scope-value",
        type=str,
        default=None,
        help=(
            "Optional value in --phase-scope-column that limits where phase filtering is applied. "
            "If unset, phase filtering applies to all rows."
        ),
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
        help="Which confound regressors to include (default: auto; motion + WM/CSF + FD + outliers + aCompCor when available)",
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

    plot_group = parser.add_argument_group("Plotting / Report")
    plot_group.add_argument(
        "--plots",
        dest="plots",
        action="store_true",
        default=None,
        help="Generate per-subject figures under <contrast>/plots/",
    )
    plot_group.add_argument(
        "--no-plots",
        dest="plots",
        action="store_false",
        help="Do not generate per-subject figures",
    )
    plot_group.add_argument(
        "--plot-html-report",
        dest="plot_html_report",
        action="store_true",
        default=None,
        help="Write <contrast>/report.html embedding generated figures",
    )
    plot_group.add_argument(
        "--no-plot-html-report",
        dest="plot_html_report",
        action="store_false",
        help="Do not write HTML report",
    )
    plot_group.add_argument(
        "--plot-formats",
        nargs="+",
        choices=["png", "svg"],
        default=None,
        metavar="FMT",
        help="Figure formats to write (default: png)",
    )
    plot_group.add_argument(
        "--plot-space",
        choices=["native", "mni", "both"],
        default=None,
        help="Which space(s) to plot in (default: both)",
    )
    plot_group.add_argument(
        "--plot-z-threshold",
        type=float,
        default=None,
        help="Z threshold for thresholded overlays (default: 2.3)",
    )
    plot_group.add_argument(
        "--plot-threshold-mode",
        choices=["z", "fdr", "none"],
        default=None,
        help="Thresholding mode for overlays/cluster table (default: z)",
    )
    plot_group.add_argument(
        "--plot-fdr-q",
        type=float,
        default=None,
        help="FDR q-value for threshold-mode=fdr (default: 0.05)",
    )
    plot_group.add_argument(
        "--plot-cluster-min-voxels",
        type=int,
        default=None,
        help="Minimum cluster size (voxels) for displaying clusters (default: 0 = disabled)",
    )
    plot_group.add_argument(
        "--plot-vmax-mode",
        choices=["per-space-robust", "shared-robust", "manual"],
        default=None,
        help="Color scaling mode (default: per-space-robust)",
    )
    plot_group.add_argument(
        "--plot-vmax",
        type=float,
        default=None,
        help="Manual vmax (required if plot-vmax-mode=manual)",
    )
    plot_group.add_argument(
        "--plot-include-unthresholded",
        dest="plot_include_unthresholded",
        action="store_true",
        default=None,
        help="Also generate unthresholded panels (default: enabled)",
    )
    plot_group.add_argument(
        "--no-plot-include-unthresholded",
        dest="plot_include_unthresholded",
        action="store_false",
        help="Disable unthresholded panels",
    )
    plot_group.add_argument(
        "--plot-types",
        nargs="+",
        choices=["slices", "glass", "hist", "clusters"],
        default=None,
        metavar="PLOT",
        help="Which plot types to generate (default: slices glass hist clusters)",
    )
    plot_group.add_argument(
        "--plot-no-effect-size",
        dest="plot_effect_size",
        action="store_false",
        default=None,
        help="Do not generate effect-size (beta/cope) panels",
    )
    plot_group.add_argument(
        "--plot-effect-size",
        dest="plot_effect_size",
        action="store_true",
        help="Generate effect-size (beta/cope) panels",
    )
    plot_group.add_argument(
        "--plot-no-standard-error",
        dest="plot_standard_error",
        action="store_false",
        default=None,
        help="Do not generate standard-error panels (from variance)",
    )
    plot_group.add_argument(
        "--plot-standard-error",
        dest="plot_standard_error",
        action="store_true",
        help="Generate standard-error panels (from variance)",
    )
    plot_group.add_argument(
        "--plot-no-motion-qc",
        dest="plot_motion_qc",
        action="store_false",
        default=None,
        help="Disable motion QC panels (FD/DVARS)",
    )
    plot_group.add_argument(
        "--plot-motion-qc",
        dest="plot_motion_qc",
        action="store_true",
        help="Enable motion QC panels (FD/DVARS)",
    )
    plot_group.add_argument(
        "--plot-no-carpet-qc",
        dest="plot_carpet_qc",
        action="store_false",
        default=None,
        help="Disable carpet plot QC panels",
    )
    plot_group.add_argument(
        "--plot-carpet-qc",
        dest="plot_carpet_qc",
        action="store_true",
        help="Enable carpet plot QC panels",
    )
    plot_group.add_argument(
        "--plot-no-tsnr-qc",
        dest="plot_tsnr_qc",
        action="store_false",
        default=None,
        help="Disable tSNR QC summary",
    )
    plot_group.add_argument(
        "--plot-tsnr-qc",
        dest="plot_tsnr_qc",
        action="store_true",
        help="Enable tSNR QC summary",
    )
    plot_group.add_argument(
        "--plot-no-design-qc",
        dest="plot_design_qc",
        action="store_false",
        default=None,
        help="Disable design-matrix sanity summaries",
    )
    plot_group.add_argument(
        "--plot-design-qc",
        dest="plot_design_qc",
        action="store_true",
        help="Enable design-matrix sanity summaries",
    )
    plot_group.add_argument(
        "--plot-no-embed-images",
        dest="plot_embed_images",
        action="store_false",
        default=None,
        help="Do not embed images in HTML (use relative file paths)",
    )
    plot_group.add_argument(
        "--plot-embed-images",
        dest="plot_embed_images",
        action="store_true",
        help="Embed images in HTML",
    )
    plot_group.add_argument(
        "--plot-no-signatures",
        dest="plot_signatures",
        action="store_false",
        default=None,
        help="Disable multivariate signature readouts",
    )
    plot_group.add_argument(
        "--plot-signatures",
        dest="plot_signatures",
        action="store_true",
        help="Enable multivariate signature readouts",
    )
    plot_group.add_argument(
        "--signature-dir",
        type=str,
        default=None,
        dest="signature_dir",
        help="Root directory for signature weight maps (paths.signature_dir)",
    )
    plot_group.add_argument(
        "--signature-maps",
        nargs="+",
        default=None,
        dest="signature_maps",
        metavar="NAME:REL_PATH",
        help=(
            "Signature weight maps as NAME:RELATIVE_PATH pairs (relative to --signature-dir). "
            "Example: --signature-maps SIG_A:maps/sig_a.nii.gz SIG_B:maps/sig_b.nii.gz"
        ),
    )

    group_group = parser.add_argument_group("Second-level group analysis")
    group_group.add_argument(
        "--group-model",
        choices=["one-sample", "two-sample", "paired", "repeated-measures"],
        default=None,
        help=(
            "Second-level design. Uses existing first-level effect-size maps in "
            "MNI152NLin2009cAsym space."
        ),
    )
    group_group.add_argument(
        "--group-input-root",
        type=str,
        default=None,
        help=(
            "Root containing first-level subject outputs. Default: deriv_root. "
            "Use when first-level outputs were written to a custom --output-dir."
        ),
    )
    group_group.add_argument(
        "--group-contrast-names",
        nargs="+",
        default=None,
        metavar="CONTRAST",
        help=(
            "First-level contrast names to include. one-sample/two-sample require "
            "1, paired requires 2 ordered as A B, repeated-measures requires >=2."
        ),
    )
    group_group.add_argument(
        "--group-condition-labels",
        nargs="+",
        default=None,
        metavar="LABEL",
        help=(
            "Optional labels corresponding to --group-contrast-names. Defaults "
            "to the first-level contrast names."
        ),
    )
    group_group.add_argument(
        "--group-covariates-file",
        type=str,
        default=None,
        help="Subject-level TSV/CSV for group assignments and numeric covariates.",
    )
    group_group.add_argument(
        "--group-subject-column",
        type=str,
        default=None,
        help="Subject ID column in --group-covariates-file (default: subject).",
    )
    group_group.add_argument(
        "--group-covariate-columns",
        nargs="+",
        default=None,
        metavar="COL",
        help=(
            "Numeric nuisance covariates to include. Supported for one-sample, "
            "two-sample, and paired models."
        ),
    )
    group_group.add_argument(
        "--group-column",
        type=str,
        default=None,
        help="Grouping column for two-sample models.",
    )
    group_group.add_argument(
        "--group-a-value",
        type=str,
        default=None,
        help="Reference group label for two-sample models.",
    )
    group_group.add_argument(
        "--group-b-value",
        type=str,
        default=None,
        help="Comparison group label for two-sample models.",
    )
    group_group.add_argument(
        "--group-permutation-inference",
        dest="group_permutation_inference",
        action="store_true",
        default=None,
        help="Run second-level max-T permutation inference in addition to parametric maps.",
    )
    group_group.add_argument(
        "--no-group-permutation-inference",
        dest="group_permutation_inference",
        action="store_false",
        help="Disable second-level permutation inference.",
    )
    group_group.add_argument(
        "--group-n-permutations",
        type=int,
        default=None,
        help="Number of permutations for second-level inference (default: 5000).",
    )
    group_group.add_argument(
        "--group-two-sided",
        dest="group_two_sided",
        action="store_true",
        default=None,
        help="Use two-sided second-level permutation inference (default).",
    )
    group_group.add_argument(
        "--group-one-sided",
        dest="group_two_sided",
        action="store_false",
        help="Use one-sided second-level permutation inference.",
    )

    trial_group = parser.add_argument_group("Trial-wise betas / signatures (beta-series, lss)")
    trial_group.add_argument(
        "--include-other-events",
        dest="include_other_events",
        action="store_true",
        default=None,
        help="Include non-contrast task events as nuisance regressors (default: enabled)",
    )
    trial_group.add_argument(
        "--no-include-other-events",
        dest="include_other_events",
        action="store_false",
        help="Do not include non-contrast task events as regressors",
    )
    trial_group.add_argument(
        "--max-trials-per-run",
        type=int,
        default=None,
        help="Cap number of selected trials per run (default: no cap)",
    )
    trial_group.add_argument(
        "--fixed-effects-weighting",
        choices=["variance", "mean"],
        default=None,
        help=(
            "How to combine condition maps across inputs "
            "(beta-series: across runs; LSS: descriptive trial summaries) (default: variance)"
        ),
    )
    trial_group.add_argument(
        "--write-trial-betas",
        dest="write_trial_betas",
        action="store_true",
        default=None,
        help="Write per-trial beta maps under <output>/trial_betas/ (default: off)",
    )
    trial_group.add_argument(
        "--no-write-trial-betas",
        dest="write_trial_betas",
        action="store_false",
        help="Do not write per-trial beta maps",
    )
    trial_group.add_argument(
        "--write-trial-variances",
        dest="write_trial_variances",
        action="store_true",
        default=None,
        help="Write per-trial beta variance maps under <output>/trial_betas/ (default: off)",
    )
    trial_group.add_argument(
        "--no-write-trial-variances",
        dest="write_trial_variances",
        action="store_false",
        help="Do not write per-trial beta variance maps",
    )
    trial_group.add_argument(
        "--write-condition-betas",
        dest="write_condition_betas",
        action="store_true",
        default=None,
        help="Write condition-averaged beta maps under <output>/condition_betas/ (default: on)",
    )
    trial_group.add_argument(
        "--no-write-condition-betas",
        dest="write_condition_betas",
        action="store_false",
        help="Do not write condition-averaged beta maps",
    )
    trial_group.add_argument(
        "--signatures",
        nargs="+",
        default=None,
        metavar="SIG",
        help="Which signatures to compute (default: all available)",
    )
    trial_group.add_argument(
        "--lss-other-regressors",
        choices=["per-condition", "all"],
        default=None,
        help="LSS: model other trials per-condition or pooled (default: per-condition)",
    )
    trial_group.add_argument(
        "--signature-group-column",
        type=str,
        default=None,
        help=(
            "Optional: events column to compute signature summaries for each selected value "
            "(e.g., temperature). When set together with --signature-group-values, the pipeline "
            "will compute condition summary maps per value and write signatures/group_signature_expression.tsv."
        ),
    )
    trial_group.add_argument(
        "--signature-group-values",
        nargs="+",
        default=None,
        metavar="VAL",
        help=(
            "Optional: value(s) within --signature-group-column to summarize (e.g., 44.3 45.3 46.3). "
            "Use multiple values to compute signatures for each value."
        ),
    )
    trial_group.add_argument(
        "--signature-group-scope",
        choices=["across-runs", "per-run"],
        default=None,
        help="Signature grouping scope for --signature-group-column/values (default: across-runs).",
    )
    trial_group.add_argument(
        "--signature-scope-trial-types",
        nargs="+",
        default=None,
        metavar="TT",
        help=(
            "Optional: restrict which events.tsv trial_type rows are eligible for trial selection. "
            "This can prevent mixing phases when selecting by per-trial columns (e.g., binary_outcome_coded). "
            "Use 'all' to disable scoping."
        ),
    )
    trial_group.add_argument(
        "--signature-scope-stim-phases",
        nargs="+",
        default=None,
        metavar="PHASE",
        help=(
            "Optional: restrict which phase values are eligible for trial selection (in the configured phase column). "
            "Use 'all' to disable scoping."
        ),
    )
    trial_group.add_argument(
        "--signature-scope-trial-type-column",
        type=str,
        default=None,
        help=(
            "Events column used for --signature-scope-trial-types. Required when signature trial scoping is enabled."
        ),
    )
    trial_group.add_argument(
        "--signature-scope-phase-column",
        type=str,
        default=None,
        help=(
            "Events column used for --signature-scope-stim-phases. Required when signature phase scoping is enabled."
        ),
    )

    return parser


def _map_task_to_fmri(task: str) -> str:
    """Return task name for fMRI file matching (pass-through, no mapping)."""
    task = (task or "").strip()
    return task if task else "task"




def run_fmri_analysis(args: argparse.Namespace, _subjects: List[str], config: Any) -> None:
    progress = create_progress_reporter(args)
    apply_fmri_config_defaults(config)

    if args.bids_fmri_root:
        config.setdefault("paths", {})["bids_fmri_root"] = args.bids_fmri_root
    if args.deriv_root:
        config.setdefault("paths", {})["deriv_root"] = args.deriv_root
    if args.freesurfer_dir:
        config.setdefault("paths", {})["freesurfer_dir"] = args.freesurfer_dir
    if getattr(args, "signature_dir", None):
        config.setdefault("paths", {})["signature_dir"] = args.signature_dir
    if getattr(args, "signature_maps", None):
        parsed_specs = []
        for token in args.signature_maps:
            if ":" in token:
                name, _, rel_path = token.partition(":")
                name = name.strip()
                rel_path = rel_path.strip()
                if name and rel_path:
                    parsed_specs.append({"name": name, "path": rel_path})
        if parsed_specs:
            config.setdefault("paths", {})["signature_maps"] = parsed_specs
    apply_set_overrides(config, getattr(args, "set_overrides", None))

    bids_fmri_root = config.get("paths.bids_fmri_root")
    if not bids_fmri_root:
        raise ValueError("Missing required config value: paths.bids_fmri_root")

    base_task = resolve_task(args.task, config)
    fmri_task = _map_task_to_fmri(base_task)

    subjects = resolve_subjects(args, Path(bids_fmri_root), config)

    mode = str(getattr(args, "mode", "first-level") or "first-level").strip().lower()

    def _cfg_value(*path: str) -> Any:
        current: Any = {}
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    def _coalesce(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    def _has_value(value: Any) -> bool:
        return value is not None and str(value).strip() != ""

    def _normalize_string_list(value: Any) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            items = [part.strip() for part in value.split(",") if part.strip()]
            return items or None
        if isinstance(value, (list, tuple, set)):
            items = [str(part).strip() for part in value if str(part).strip()]
            return items or None
        item = str(value).strip()
        return [item] if item else None

    def _normalize_runs(value: Any) -> list[int] | None:
        if value is None:
            return None
        items = _normalize_string_list(value)
        if not items:
            return None
        return [int(item) for item in items]

    if mode == "second-level":
        from fmri_pipeline.analysis.second_level import (
            SecondLevelConfig,
            SecondLevelPermutationConfig,
            load_second_level_config_section,
        )
        from fmri_pipeline.pipelines.fmri_second_level import (
            FmriSecondLevelPipeline,
        )

        second_level_section = load_second_level_config_section(config)

        def _group_cfg_value(*path: str) -> Any:
            current: Any = second_level_section
            for key in path:
                if not isinstance(current, dict) or key not in current:
                    return None
                current = current[key]
            return current

        def _normalize_space_separated_list(value: Any) -> tuple[str, ...] | None:
            if value is None:
                return None
            if isinstance(value, str):
                items = [
                    part.strip()
                    for part in value.replace(",", " ").split()
                    if part.strip()
                ]
                return tuple(items) if items else None
            items = [str(part).strip() for part in value if str(part).strip()]
            return tuple(items) if items else None

        second_level_cfg = SecondLevelConfig(
            model=str(
                _coalesce(
                    getattr(args, "group_model", None),
                    _group_cfg_value("model"),
                    "one-sample",
                )
            ).strip(),
            contrast_names=tuple(
                _normalize_space_separated_list(
                    _coalesce(
                        getattr(args, "group_contrast_names", None),
                        _group_cfg_value("contrast_names"),
                    )
                )
                or ()
            ),
            input_root=_coalesce(
                getattr(args, "group_input_root", None),
                _group_cfg_value("input_root"),
            ),
            condition_labels=_normalize_space_separated_list(
                _coalesce(
                    getattr(args, "group_condition_labels", None),
                    _group_cfg_value("condition_labels"),
                )
            ),
            formula=str(
                _coalesce(getattr(args, "formula", None), _group_cfg_value("formula"))
                or ""
            ).strip()
            or None,
            output_name=str(
                _coalesce(
                    getattr(args, "contrast_name", None),
                    _group_cfg_value("output_name"),
                )
                or ""
            ).strip()
            or None,
            output_dir=str(
                _coalesce(getattr(args, "output_dir", None), _group_cfg_value("output_dir"))
                or ""
            ).strip()
            or None,
            covariates_file=_coalesce(
                getattr(args, "group_covariates_file", None),
                _group_cfg_value("covariates_file"),
            ),
            subject_column=str(
                _coalesce(
                    getattr(args, "group_subject_column", None),
                    _group_cfg_value("subject_column"),
                    "subject",
                )
            ).strip()
            or "subject",
            covariate_columns=_normalize_space_separated_list(
                _coalesce(
                    getattr(args, "group_covariate_columns", None),
                    _group_cfg_value("covariate_columns"),
                )
            ),
            group_column=str(
                _coalesce(
                    getattr(args, "group_column", None),
                    _group_cfg_value("group_column"),
                )
                or ""
            ).strip()
            or None,
            group_a_value=str(
                _coalesce(
                    getattr(args, "group_a_value", None),
                    _group_cfg_value("group_a_value"),
                )
                or ""
            ).strip()
            or None,
            group_b_value=str(
                _coalesce(
                    getattr(args, "group_b_value", None),
                    _group_cfg_value("group_b_value"),
                )
                or ""
            ).strip()
            or None,
            write_design_matrix=(
                bool(args.write_design_matrix)
                if args.write_design_matrix is not None
                else bool(_coalesce(_group_cfg_value("write_design_matrix"), True))
            ),
            permutation=SecondLevelPermutationConfig(
                enabled=(
                    bool(args.group_permutation_inference)
                    if args.group_permutation_inference is not None
                    else bool(_coalesce(_group_cfg_value("permutation", "enabled"), False))
                ),
                n_permutations=int(
                    _coalesce(
                        getattr(args, "group_n_permutations", None),
                        _group_cfg_value("permutation", "n_permutations"),
                        5000,
                    )
                ),
                two_sided=(
                    bool(args.group_two_sided)
                    if args.group_two_sided is not None
                    else bool(_coalesce(_group_cfg_value("permutation", "two_sided"), True))
                ),
            ),
        ).normalized()

        pipeline = FmriSecondLevelPipeline(config=config)
        pipeline.run_batch(
            subjects=subjects,
            task=fmri_task,
            progress=progress,
            dry_run=bool(getattr(args, "dry_run", False)),
            second_level_cfg=second_level_cfg,
        )
        return

    from fmri_pipeline.analysis.contrast_builder import (
        ContrastBuilderConfig,
        load_contrast_config_section,
        validate_contrast_config_section,
    )
    from fmri_pipeline.analysis.plotting_config import (
        build_fmri_plotting_config_from_args,
    )
    from fmri_pipeline.analysis.smoothing import normalize_smoothing_fwhm
    from fmri_pipeline.analysis.trial_signatures import (
        TrialSignatureExtractionConfig,
    )
    from fmri_pipeline.pipelines.fmri_analysis import FmriAnalysisPipeline
    from fmri_pipeline.pipelines.fmri_trial_signatures import (
        FmriTrialSignaturePipeline,
    )

    contrast_cfg_section = load_contrast_config_section(config)
    validate_contrast_config_section(contrast_cfg_section)

    def _cfg_value(*path: str) -> Any:
        current: Any = contrast_cfg_section
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    input_source = str(
        _coalesce(args.input_source, _cfg_value("input_source"), "fmriprep")
    ).strip().lower()
    if input_source not in {"fmriprep", "bids_raw"}:
        raise ValueError(
            f"input_source must be 'fmriprep' or 'bids_raw', got {input_source!r}."
        )

    drift_model = _coalesce(args.drift_model, _cfg_value("drift_model"))
    if drift_model == "none":
        drift_model = None

    low_pass_hz = _coalesce(args.low_pass_hz, _cfg_value("low_pass_hz"))
    if low_pass_hz is not None and low_pass_hz <= 0:
        low_pass_hz = None

    if args.smoothing_fwhm is None:
        smoothing_fwhm = normalize_smoothing_fwhm(_cfg_value("smoothing_fwhm"))
    else:
        smoothing_fwhm = normalize_smoothing_fwhm(args.smoothing_fwhm)

    formula = str(_coalesce(args.formula, _cfg_value("formula")) or "").strip() or None
    contrast_name = (
        str(_coalesce(args.contrast_name, _cfg_value("name"), "contrast")).strip()
        or "contrast"
    )
    contrast_type = str(
        _coalesce(args.contrast_type, _cfg_value("type"), "custom" if formula else "t-test")
    ).strip()
    if contrast_type not in {"t-test", "custom"}:
        raise ValueError(
            f"contrast-type must be 't-test' or 'custom', got {contrast_type!r}."
        )
    cond_a_column = str(
        _coalesce(args.cond_a_column, _cfg_value("condition_a", "column")) or ""
    ).strip()
    cond_a_value = _coalesce(args.cond_a_value, _cfg_value("condition_a", "value"))
    cond_b_column = str(
        _coalesce(args.cond_b_column, _cfg_value("condition_b", "column")) or ""
    ).strip()
    cond_b_value = _coalesce(args.cond_b_value, _cfg_value("condition_b", "value"))
    condition_scope_trial_types = _coalesce(
        getattr(args, "condition_scope_trial_types", None),
        _cfg_value("condition_scope_trial_types"),
    )
    condition_scope_column = str(
        _coalesce(
            getattr(args, "condition_scope_column", None),
            _cfg_value("condition_scope_column"),
        )
        or ""
    ).strip()

    if contrast_type == "custom" and not formula:
        raise ValueError("contrast-type=custom requires --formula")

    if contrast_type != "custom" and not cond_a_column:
        raise ValueError(
            "Missing required --cond-a-column or fmri_contrast.condition_a.column."
        )
    if contrast_type != "custom" and not _has_value(cond_a_value):
        raise ValueError(
            "Missing required --cond-a-value (or use --contrast-type custom --formula ...)"
        )
    if _has_value(cond_b_value) and not cond_b_column:
        raise ValueError(
            "Missing required --cond-b-column or fmri_contrast.condition_b.column."
        )
    if condition_scope_trial_types and not condition_scope_column:
        raise ValueError(
            "condition_scope_trial_types requires --condition-scope-column "
            "or fmri_contrast.condition_scope_column."
        )
    if mode in {"beta-series", "lss"} and not _has_value(cond_b_value):
        raise ValueError(
            "Trial-wise modes require --cond-b-value (e.g., condition_a vs condition_b)."
        )

    confounds_strategy = str(
        _coalesce(args.confounds_strategy, _cfg_value("confounds_strategy"), "auto")
    ).strip().lower()
    if not confounds_strategy:
        confounds_strategy = "auto"
    if args.write_design_matrix is not None:
        write_design_matrix = bool(args.write_design_matrix)
    else:
        write_design_matrix = bool(_coalesce(_cfg_value("write_design_matrix"), False))

    if args.fmriprep_space:
        fmriprep_space = str(args.fmriprep_space).strip()
    else:
        fmriprep_space = (
            "MNI152NLin2009cAsym"
            if mode in {"beta-series", "lss"}
            else str(_coalesce(_cfg_value("fmriprep_space"), "T1w")).strip()
        )

    if mode == "first-level":
        from fmri_pipeline.analysis.events_selection import normalize_trial_type_list

        events_to_model = normalize_trial_type_list(
            _coalesce(args.events_to_model, _cfg_value("events_to_model"))
        )
        events_to_model_column = str(
            _coalesce(
                getattr(args, "events_to_model_column", None),
                _cfg_value("events_to_model_column"),
            )
            or ""
        ).strip()
        stim_phases_to_model = normalize_trial_type_list(
            _coalesce(
                getattr(args, "stim_phases_to_model", None),
                _cfg_value("stim_phases_to_model"),
            )
        )
        phase_column = str(
            _coalesce(getattr(args, "phase_column", None), _cfg_value("phase_column")) or ""
        ).strip()
        phase_scope_column = str(
            _coalesce(
                getattr(args, "phase_scope_column", None),
                _cfg_value("phase_scope_column"),
            )
            or ""
        ).strip()
        phase_scope_value = str(
            _coalesce(getattr(args, "phase_scope_value", None), _cfg_value("phase_scope_value"), "")
        ).strip() or None
        if events_to_model and not events_to_model_column:
            raise ValueError(
                "events_to_model requires --events-to-model-column "
                "or fmri_contrast.events_to_model_column."
            )
        if stim_phases_to_model and not phase_column:
            raise ValueError(
                "stim_phases_to_model requires --phase-column "
                "or fmri_contrast.phase_column."
            )
        if phase_scope_value and not phase_scope_column:
            raise ValueError(
                "phase_scope_value requires --phase-scope-column "
                "or fmri_contrast.phase_scope_column."
            )
        cfg = ContrastBuilderConfig(
            enabled=True,
            input_source=input_source,
            fmriprep_space=fmriprep_space,
            require_fmriprep=(
                bool(args.require_fmriprep)
                if args.require_fmriprep is not None
                else bool(_coalesce(_cfg_value("require_fmriprep"), True))
            ),
            contrast_type=contrast_type,
            condition1=None,
            condition2=None,
            condition_a_column=cond_a_column,
            condition_a_value=str(cond_a_value).strip() if _has_value(cond_a_value) else None,
            condition_b_column=cond_b_column,
            condition_b_value=str(cond_b_value).strip() if _has_value(cond_b_value) else None,
            condition_scope_trial_types=_normalize_string_list(condition_scope_trial_types),
            condition_scope_column=condition_scope_column,
            formula=formula,
            name=contrast_name,
            runs=list(args.runs) if args.runs else _normalize_runs(_cfg_value("runs")),
            hrf_model=str(_coalesce(args.hrf_model, _cfg_value("hrf_model"), "spm")).strip(),
            drift_model=str(drift_model).strip() if drift_model else None,
            high_pass_hz=float(
                _coalesce(args.high_pass_hz, _cfg_value("high_pass_hz"), 0.008)
            ),
            low_pass_hz=float(low_pass_hz) if low_pass_hz is not None else None,
            output_type=str(
                _coalesce(args.output_type, _cfg_value("output_type"), "z-score")
            ).strip(),
            resample_to_freesurfer=(
                bool(args.resample_to_freesurfer)
                if args.resample_to_freesurfer is not None
                else bool(_coalesce(_cfg_value("resample_to_freesurfer"), True))
            ),
            confounds_strategy=confounds_strategy,
            write_design_matrix=write_design_matrix,
            smoothing_fwhm=smoothing_fwhm,
            events_to_model=events_to_model,
            events_to_model_column=events_to_model_column,
            stim_phases_to_model=stim_phases_to_model,
            phase_column=phase_column,
            phase_scope_column=phase_scope_column,
            phase_scope_value=phase_scope_value,
        )
    else:
        include_other_events = (
            True if args.include_other_events is None else bool(args.include_other_events)
        )
        fixed_weighting = str(args.fixed_effects_weighting or "variance").strip().lower()
        lss_other = str(args.lss_other_regressors or "per-condition").strip().lower()
        lss_other = "per_condition" if lss_other == "per-condition" else lss_other

        trial_cfg = TrialSignatureExtractionConfig(
            input_source=input_source,
            fmriprep_space=fmriprep_space,
            require_fmriprep=(
                bool(args.require_fmriprep)
                if args.require_fmriprep is not None
                else bool(_coalesce(_cfg_value("require_fmriprep"), True))
            ),
            runs=list(args.runs) if args.runs else _normalize_runs(_cfg_value("runs")),
            task=fmri_task,
            name=contrast_name,
            condition_a_column=cond_a_column,
            condition_a_value=str(cond_a_value).strip(),
            condition_b_column=cond_b_column,
            condition_b_value=str(cond_b_value).strip(),
            hrf_model=str(_coalesce(args.hrf_model, _cfg_value("hrf_model"), "spm")).strip(),
            drift_model=str(drift_model).strip() if drift_model else None,
            high_pass_hz=float(
                _coalesce(args.high_pass_hz, _cfg_value("high_pass_hz"), 0.008)
            ),
            low_pass_hz=float(low_pass_hz) if low_pass_hz is not None else None,
            smoothing_fwhm=smoothing_fwhm,
            confounds_strategy=confounds_strategy,
            method=mode,
            include_other_events=include_other_events,
            lss_other_regressors=lss_other,
            condition_scope_trial_type_column=str(
                getattr(args, "signature_scope_trial_type_column", "") or ""
            ).strip(),
            condition_scope_phase_column=str(
                getattr(args, "signature_scope_phase_column", "") or ""
            ).strip(),
            condition_scope_trial_types=tuple(
                _normalize_string_list(getattr(args, "signature_scope_trial_types", None))
                or ()
            )
            or None,
            condition_scope_stim_phases=tuple(
                _normalize_string_list(getattr(args, "signature_scope_stim_phases", None))
                or ()
            )
            or None,
            max_trials_per_run=int(args.max_trials_per_run)
            if args.max_trials_per_run
            else None,
            fixed_effects_weighting=fixed_weighting,
            signatures=tuple(args.signatures) if args.signatures else None,
            signature_group_column=str(
                getattr(args, "signature_group_column", "") or ""
            ).strip()
            or None,
            signature_group_values=tuple(getattr(args, "signature_group_values", None) or ())
            or None,
            signature_group_scope=(
                str(getattr(args, "signature_group_scope", "") or "")
                .strip()
                .lower()
                .replace("-", "_")
                if getattr(args, "signature_group_scope", None)
                else "across_runs"
            ),
            write_trial_betas=bool(args.write_trial_betas)
            if args.write_trial_betas is not None
            else False,
            write_trial_variances=bool(args.write_trial_variances)
            if args.write_trial_variances is not None
            else False,
            write_condition_betas=bool(args.write_condition_betas)
            if args.write_condition_betas is not None
            else True,
        )

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

    if mode == "first-level":
        vmax_mode = str(args.plot_vmax_mode).strip().lower() if args.plot_vmax_mode else None
        if vmax_mode == "per-space-robust":
            vmax_mode = "per_space_robust"
        elif vmax_mode == "shared-robust":
            vmax_mode = "shared_robust"

        plotting_cfg = build_fmri_plotting_config_from_args(
            enabled=bool(args.plots) if args.plots is not None else False,
            html_report=bool(args.plot_html_report) if args.plot_html_report is not None else False,
            formats=tuple(args.plot_formats) if args.plot_formats else None,
            space=str(args.plot_space).strip().lower() if args.plot_space else None,
            threshold_mode=str(args.plot_threshold_mode).strip().lower() if args.plot_threshold_mode else None,
            z_threshold=float(args.plot_z_threshold) if args.plot_z_threshold is not None else None,
            fdr_q=float(args.plot_fdr_q) if args.plot_fdr_q is not None else None,
            cluster_min_voxels=int(args.plot_cluster_min_voxels) if args.plot_cluster_min_voxels is not None else None,
            vmax_mode=vmax_mode,
            vmax_manual=float(args.plot_vmax) if args.plot_vmax is not None else None,
            include_unthresholded=bool(args.plot_include_unthresholded)
            if args.plot_include_unthresholded is not None
            else None,
            plot_types=tuple(args.plot_types) if args.plot_types else None,
            include_effect_size=bool(args.plot_effect_size) if args.plot_effect_size is not None else None,
            include_standard_error=bool(args.plot_standard_error) if args.plot_standard_error is not None else None,
            include_motion_qc=bool(args.plot_motion_qc) if args.plot_motion_qc is not None else None,
            include_carpet_qc=bool(args.plot_carpet_qc) if args.plot_carpet_qc is not None else None,
            include_tsnr_qc=bool(args.plot_tsnr_qc) if args.plot_tsnr_qc is not None else None,
            include_design_qc=bool(args.plot_design_qc) if args.plot_design_qc is not None else None,
            embed_images=bool(args.plot_embed_images) if args.plot_embed_images is not None else None,
            include_signatures=bool(args.plot_signatures) if args.plot_signatures is not None else None,
        )

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
            plotting_cfg=plotting_cfg,
            output_dir=out_dir,
            freesurfer_subjects_dir=fs_dir,
        )
        return

    pipeline = FmriTrialSignaturePipeline(config=config)
    pipeline.run_batch(
        subjects=subjects,
        task=fmri_task,
        progress=progress,
        dry_run=bool(getattr(args, "dry_run", False)),
        bids_fmri_root=Path(bids_fmri_root).expanduser().resolve(),
        trial_cfg=trial_cfg,
        output_dir=out_dir,
    )
