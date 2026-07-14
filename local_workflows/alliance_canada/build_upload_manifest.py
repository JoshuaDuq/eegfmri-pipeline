#!/usr/bin/env python3
"""Build narrow upload manifests for the local Alliance workflow.

The setup scripts feed these newline-delimited relative paths to
``rsync --files-from``. Required inputs are validated here so missing data
surfaces before any remote transfer starts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_METADATA_FILES = (
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
    "README",
    "CHANGES",
)
PRIMARY_MODEL_COMPARISON = (
    "feature_benchmark",
    "primary",
    "NPS",
    "alpha_beta_gamma",
    "model_comparison",
    "model_comparison.tsv",
)
STUDY1_EXTERNAL_FILES = (
    Path("NPS") / "weights_NSF_grouppred_cvpcr.nii.gz",
    Path("SIIPS1") / "nonnoc_v11_4_137subjmap_weighted_mean.nii.gz",
    Path("signature_manifest.yaml"),
    Path("tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    _add_fmriprep_parser(subparsers)
    _add_study1_parser(subparsers)
    _add_study2_parser(subparsers)
    args = parser.parse_args()

    try:
        if args.mode == "fmriprep":
            build_fmriprep_manifest(args)
        elif args.mode == "study1":
            build_study1_manifests(args)
        elif args.mode == "study2":
            build_study2_manifests(args)
        else:
            raise ValueError(f"Unsupported manifest mode: {args.mode}")
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def _add_fmriprep_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("fmriprep")
    parser.add_argument("--subjects-file", required=True, type=Path)
    parser.add_argument("--local-fmri-root", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True, type=Path)


def _add_study2_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("study2")
    parser.add_argument("--subjects-file", required=True, type=Path)
    parser.add_argument("--local-fmri-root", required=True, type=Path)
    parser.add_argument("--local-eeg-root", required=True, type=Path)
    parser.add_argument("--local-deriv-root", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--study1-root-name", required=True)
    parser.add_argument("--study2-root-name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-missing-exact-paths", action="store_true")


def _add_study1_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("study1")
    parser.add_argument("--subjects-file", required=True, type=Path)
    parser.add_argument("--local-fmri-root", required=True, type=Path)
    parser.add_argument("--local-eeg-root", required=True, type=Path)
    parser.add_argument("--local-deriv-root", required=True, type=Path)
    parser.add_argument("--local-external-root", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)


def build_fmriprep_manifest(args: argparse.Namespace) -> None:
    fmri_root = _require_dir(args.local_fmri_root, "local fMRI BIDS root")
    subjects = _read_subjects(args.subjects_file)
    task = _require_text(args.task, "task")

    files = _root_metadata(fmri_root) + _task_root_metadata(fmri_root, task)
    for subject in subjects:
        files.extend(_required_fmri_anatomy(fmri_root, subject))
        files.extend(_required_task_files(fmri_root, subject, "func", task))
        files.extend(_optional_subject_dir_files(fmri_root, subject, "fmap"))

    _write_manifest(args.output, files)


def build_study1_manifests(args: argparse.Namespace) -> None:
    fmri_root = _require_dir(args.local_fmri_root, "local fMRI BIDS root")
    eeg_root = _require_dir(args.local_eeg_root, "local EEG BIDS root")
    deriv_root = _require_dir(args.local_deriv_root, "local derivatives root")
    external_root = _require_dir(args.local_external_root, "local external data root")
    subjects = _read_subjects(args.subjects_file)
    task = _require_text(args.task, "task")

    fmri_files = _root_metadata(fmri_root) + _task_root_metadata(fmri_root, task)
    eeg_files = _root_metadata(eeg_root) + _task_root_metadata(eeg_root, task)
    eeg_derivative_files = _root_metadata(deriv_root)
    fmri_derivative_files: list[Path] = []

    for subject in subjects:
        fmri_files.extend(_required_fmri_anatomy(fmri_root, subject))
        fmri_files.extend(_required_task_files(fmri_root, subject, "func", task))
        fmri_files.extend(_optional_subject_dir_files(fmri_root, subject, "fmap"))
        eeg_files.extend(_subject_root_metadata(eeg_root, subject))
        eeg_files.extend(_required_eeg_task_files(eeg_root, subject, task))
        eeg_derivative_files.extend(_study1_eeg_derivative_files(deriv_root, subject, task))
        fmri_derivative_files.extend(
            _study1_fmriprep_derivative_files(deriv_root, fmri_root, subject, task)
        )

    for relative_path in STUDY1_EXTERNAL_FILES:
        _require_file(external_root, relative_path, "Study 1 signature asset")

    output_dir = args.output_dir
    _write_manifest(output_dir / "fmri_bids_files.txt", fmri_files)
    _write_manifest(output_dir / "eeg_bids_files.txt", eeg_files)
    _write_manifest(output_dir / "eeg_derivative_files.txt", eeg_derivative_files)
    _write_manifest(output_dir / "fmri_derivative_files.txt", fmri_derivative_files)
    _write_manifest(output_dir / "external_files.txt", list(STUDY1_EXTERNAL_FILES))


def build_study2_manifests(args: argparse.Namespace) -> None:
    fmri_root = _require_dir(args.local_fmri_root, "local fMRI BIDS root")
    eeg_root = _require_dir(args.local_eeg_root, "local EEG BIDS root")
    deriv_root = _require_dir(args.local_deriv_root, "local derivatives root")
    subjects = _read_subjects(args.subjects_file)
    task = _require_text(args.task, "task")
    study1_root_name = _require_text(args.study1_root_name, "Study 1 root name")
    study2_root_name = _require_text(args.study2_root_name, "Study 2 root name")

    fmri_files = _root_metadata(fmri_root) + _task_root_metadata(fmri_root, task)
    eeg_files = _root_metadata(eeg_root) + _task_root_metadata(eeg_root, task)
    deriv_files: list[Path] = _root_metadata(deriv_root)

    for subject in subjects:
        fmri_files.extend(_required_fmri_anatomy(fmri_root, subject))
        eeg_files.extend(_subject_root_metadata(eeg_root, subject))
        eeg_files.extend(_required_eeg_task_files(eeg_root, subject, task))
        deriv_files.extend(_required_clean_eeg_derivatives(deriv_root, subject, task))
        deriv_files.extend(
            _required_study1_subject_features(
                deriv_root,
                study1_root_name,
                subject,
                allow_missing=args.allow_missing_exact_paths,
            )
        )

    deriv_files.extend(
        _required_study1_group_artifacts(
            deriv_root,
            study1_root_name,
            allow_missing=args.allow_missing_exact_paths,
        )
    )
    deriv_files.extend(
        _required_study2_group_artifacts(
            deriv_root,
            study2_root_name,
            allow_missing=args.allow_missing_exact_paths,
        )
    )

    output_dir = args.output_dir
    _write_manifest(output_dir / "fmri_bids_files.txt", fmri_files)
    _write_manifest(output_dir / "eeg_bids_files.txt", eeg_files)
    _write_manifest(output_dir / "derivative_files.txt", deriv_files)


def _require_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Missing required {label}: {resolved}")
    return resolved


def _require_text(value: object, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must be non-empty.")
    return text


def _read_subjects(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing subjects file: {path}")

    subjects: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        subjects.append(_normalize_subject(text))

    if not subjects:
        raise ValueError(f"Subjects file contains no subjects: {path}")
    return subjects


def _normalize_subject(subject: str) -> str:
    text = subject.strip()
    if not text:
        raise ValueError("Subject identifiers must be non-empty.")
    return text if text.startswith("sub-") else f"sub-{text}"


def _root_metadata(root: Path) -> list[Path]:
    return [Path(name) for name in ROOT_METADATA_FILES if (root / name).is_file()]


def _task_root_metadata(root: Path, task: str) -> list[Path]:
    return _visible_relative_files(root, root.glob(f"task-{task}_*.json"))


def _subject_root_metadata(root: Path, subject: str) -> list[Path]:
    subject_root = root / subject
    if not subject_root.is_dir():
        return []
    return _visible_relative_files(root, subject_root.glob(f"{subject}_scans.tsv"))


def _required_fmri_anatomy(root: Path, subject: str) -> list[Path]:
    rel = Path(subject) / "anat" / f"{subject}_T1w.nii.gz"
    _require_file(root, rel, f"T1w anatomy for {subject}")
    sidecar = rel.with_suffix("").with_suffix(".json")
    if (root / sidecar).is_file():
        return [rel, sidecar]
    return [rel]



def _required_task_files(root: Path, subject: str, datatype: str, task: str) -> list[Path]:
    subject_datatype_root = root / subject / datatype
    if not subject_datatype_root.is_dir():
        raise FileNotFoundError(f"Missing required {datatype} directory for {subject}.")

    matches = _visible_relative_files(
        root,
        subject_datatype_root.glob(f"{subject}_task-{task}*"),
    )
    if not matches:
        raise FileNotFoundError(
            f"Missing required task-{task} {datatype} files for {subject}: "
            f"{subject_datatype_root}"
        )
    return matches


def _optional_subject_dir_files(root: Path, subject: str, dirname: str) -> list[Path]:
    directory = root / subject / dirname
    if not directory.is_dir():
        return []
    return _visible_relative_files(root, directory.rglob("*"))


def _required_eeg_task_files(root: Path, subject: str, task: str) -> list[Path]:
    task_files = _required_task_files(root, subject, "eeg", task)
    eeg_root = root / subject / "eeg"
    digitization_files = _visible_relative_files(
        root,
        list(eeg_root.glob(f"{subject}*electrodes.tsv"))
        + list(eeg_root.glob(f"{subject}*coordsystem.json")),
    )
    eeg_recordings = [
        path
        for path in task_files
        if path.name.endswith(("_eeg.vhdr", "_eeg.set", "_eeg.edf", "_eeg.bdf", "_eeg.fif"))
    ]
    if not eeg_recordings:
        raise FileNotFoundError(f"Missing required BIDS EEG recording for {subject}, task-{task}.")
    if not digitization_files:
        raise FileNotFoundError(f"Missing required EEG digitization files for {subject}.")
    return task_files + digitization_files


def _required_clean_eeg_derivatives(root: Path, subject: str, task: str) -> list[Path]:
    subject_patterns = (
        Path(subject) / "eeg",
        Path("preprocessed") / "eeg" / subject,
        Path("preprocessed") / "eeg" / subject / "eeg",
    )
    epochs = _matches_by_bases(root, subject_patterns, f"{subject}_task-{task}*clean*epo.fif")
    events = _matches_by_bases(root, subject_patterns, f"{subject}_task-{task}*clean*events.tsv")
    if not epochs:
        raise FileNotFoundError(f"Missing required clean EEG epochs for {subject}, task-{task}.")
    if not events:
        raise FileNotFoundError(f"Missing required clean EEG events for {subject}, task-{task}.")
    return epochs + events


def _study1_eeg_derivative_files(root: Path, subject: str, task: str) -> list[Path]:
    _required_clean_eeg_derivatives(root, subject, task)
    subject_root = root / "preprocessed" / "eeg" / subject
    return _visible_relative_files(root, subject_root.rglob("*"))


def _study1_fmriprep_derivative_files(
    root: Path,
    fmri_bids_root: Path,
    subject: str,
    task: str,
) -> list[Path]:
    fmriprep_root = root / "preprocessed" / "fmri" / "fmriprep"
    subject_root = fmriprep_root / subject
    func_root = subject_root / "func"
    if not func_root.is_dir():
        raise FileNotFoundError(f"Missing required fMRIPrep subject directory: {subject_root}")

    raw_bold_files = sorted(
        (fmri_bids_root / subject / "func").glob(f"{subject}_task-{task}*_bold.nii.gz")
    )
    if not raw_bold_files:
        raise FileNotFoundError(f"Missing required task-{task} BOLD files for {subject}.")

    for raw_bold_file in raw_bold_files:
        prefix = raw_bold_file.name.removesuffix("_bold.nii.gz")
        _require_file(
            func_root,
            Path(f"{prefix}_desc-confounds_timeseries.tsv"),
            f"fMRIPrep confounds for {subject}",
        )
        _require_glob(
            func_root,
            f"{prefix}_space-MNI152NLin2009cAsym*_desc-preproc_bold.nii.gz",
            f"fMRIPrep preprocessed BOLD for {subject}",
        )
        _require_glob(
            func_root,
            f"{prefix}_space-MNI152NLin2009cAsym*_desc-brain_mask.nii.gz",
            f"fMRIPrep brain mask for {subject}",
        )

    return _visible_relative_files(fmriprep_root, subject_root.rglob("*"))


def _required_study1_subject_features(
    root: Path,
    study1_root_name: str,
    subject: str,
    *,
    allow_missing: bool,
) -> list[Path]:
    feature_root = (
        Path("group")
        / "multimodal"
        / study1_root_name
        / "features_trial_ml_safe"
        / subject
        / "eeg"
        / "features"
        / "power"
    )
    required = (
        feature_root / "features_power.parquet",
        feature_root / "metadata" / "extraction_config.json",
    )
    for rel in required:
        _require_file(
            root,
            rel,
            f"Study 1 power feature artifact for {subject}",
            allow_missing=allow_missing,
        )
    return list(required)


def _required_study1_group_artifacts(
    root: Path,
    study1_root_name: str,
    *,
    allow_missing: bool,
) -> list[Path]:
    study_root = Path("group") / "multimodal" / study1_root_name
    required = (
        study_root / "reports" / "study1_report.tsv",
        study_root / "targets" / "primary_targets.parquet",
        study_root / Path(*PRIMARY_MODEL_COMPARISON),
    )
    for rel in required:
        _require_file(root, rel, "Study 1 group artifact", allow_missing=allow_missing)
    return list(required)


def _required_study2_group_artifacts(
    root: Path,
    study2_root_name: str,
    *,
    allow_missing: bool,
) -> list[Path]:
    rel = (
        Path("group")
        / "multimodal"
        / study2_root_name
        / "source_stage"
        / "source_stage_input.tsv"
    )
    _require_file(root, rel, "Study 2 source-stage input", allow_missing=allow_missing)
    return [rel]


def _matches_by_bases(root: Path, bases: tuple[Path, ...], pattern: str) -> list[Path]:
    matches: list[Path] = []
    for base in bases:
        directory = root / base
        if not directory.is_dir():
            continue
        matches.extend(_visible_relative_files(root, directory.rglob(pattern)))
    return matches


def _require_file(root: Path, rel: Path, label: str, *, allow_missing: bool = False) -> None:
    path = root / rel
    if path.is_file():
        return
    message = f"Missing required {label}: {path}"
    if allow_missing:
        print(message, file=sys.stderr)
        return
    raise FileNotFoundError(message)


def _require_glob(root: Path, pattern: str, label: str) -> None:
    if any(root.glob(pattern)):
        return
    raise FileNotFoundError(f"Missing required {label}: {root / pattern}")


def _visible_relative_files(root: Path, paths: object) -> list[Path]:
    relative: list[Path] = []
    for path in paths:
        candidate = Path(path)
        if candidate.is_file() and not candidate.name.startswith("._"):
            relative.append(candidate.relative_to(root))
    return sorted(set(relative), key=lambda item: item.as_posix())


def _write_manifest(path: Path, files: list[Path]) -> None:
    ordered = sorted({file.as_posix() for file in files})
    if not ordered:
        raise ValueError(f"Refusing to write empty upload manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ordered) + "\n", encoding="utf-8")
    print(f"Wrote {len(ordered)} entries to {path}")


if __name__ == "__main__":
    raise SystemExit(main())
