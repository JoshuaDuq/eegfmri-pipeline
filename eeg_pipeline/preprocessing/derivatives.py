"""Discovery of MNE-BIDS-Pipeline derivative files.

Every stage that reads preprocessing derivatives has to answer the same two questions:
which subjects exist, and which files belong to one subject and task. Answering them with
a local ``glob`` per call site is how layouts diverge — a pattern that hardcodes
``sub-X/eeg`` skips every session-organized dataset, and one that requires a ``run-``
entity skips every single-run dataset, in both cases by finding nothing rather than by
failing. These helpers search recursively and match on entities, so a layout either works
everywhere or fails everywhere.
"""

from __future__ import annotations

from pathlib import Path

FILTERED_RAW_SUFFIX = "_proc-filt_raw.fif"
CLEAN_RAW_SUFFIX = "_proc-clean_raw.fif"
ICA_SUFFIX = "_proc-ica_ica.fif"


def _is_visible(path: Path) -> bool:
    """Reject macOS resource forks, which glob happily returns as real files."""
    return path.is_file() and not path.name.startswith("._")


def resolve_subjects(pipeline_root: Path, subjects: list[str]) -> list[str]:
    """Resolve an explicit subject list, or discover every subject under the root."""
    if subjects == ["all"]:
        resolved = sorted(
            path.name.removeprefix("sub-") for path in pipeline_root.glob("sub-*") if path.is_dir()
        )
    else:
        resolved = [subject.removeprefix("sub-") for subject in subjects]
    if not resolved:
        raise FileNotFoundError(f"No subject derivatives found under {pipeline_root}.")
    return resolved


def subject_root(pipeline_root: Path, subject: str) -> Path:
    """Return the directory holding one subject's derivatives, sessions included."""
    return pipeline_root / f"sub-{subject}"


def _task_selector(task: str | None) -> str:
    return f"_task-{task}_" if task is not None else "_task-"


def find_filtered_raw_runs(
    pipeline_root: Path,
    *,
    subject: str,
    task: str | None,
) -> list[Path]:
    """Find every filtered continuous run for one subject and task.

    Matches with or without a ``run-`` entity and at any session depth. Split recordings
    are rejected rather than partially processed, because a caller that rewrites or
    concatenates them would silently use only the first part.
    """
    root = subject_root(pipeline_root, subject)
    if not root.is_dir():
        # Absence is not an error here: callers that need runs raise their own message,
        # and callers that merely harmonize what exists must be able to skip a subject.
        return []
    prefix = f"sub-{subject}_"
    selector = _task_selector(task)
    paths = sorted(
        path
        for path in root.rglob(f"*{FILTERED_RAW_SUFFIX}")
        if _is_visible(path) and path.name.startswith(prefix) and selector in path.name
    )
    split_paths = [path for path in paths if "_split-" in path.name]
    if split_paths:
        raise RuntimeError(f"Split filtered raw files are not supported: {split_paths}")
    return paths


def find_ica_solutions(pipeline_root: Path, *, subject: str) -> list[Path]:
    """Find every ICA solution for one subject, one per session where sessions exist."""
    root = subject_root(pipeline_root, subject)
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob(f"*{ICA_SUFFIX}")
        if _is_visible(path) and path.name.startswith(f"sub-{subject}_")
    )


def entity_prefix(path: Path, suffix: str) -> str:
    """Return the subject/session entity prefix shared by one decomposition's files."""
    if not path.name.endswith(suffix):
        raise ValueError(f"Expected a path ending in {suffix!r}, got {path}.")
    return path.name.removesuffix(suffix)


def runs_for_prefix(run_paths: list[Path], prefix: str) -> list[Path]:
    """Select the runs belonging to the decomposition identified by ``prefix``.

    The prefix carries the subject and session entities; ``_task-`` anchors the match so
    a prefix cannot also select a longer sibling prefix.
    """
    return [path for path in run_paths if path.name.startswith(f"{prefix}_task-")]


def clean_raw_for_filtered(filtered_path: Path) -> Path:
    """Return the ICA-cleaned counterpart of a filtered continuous run."""
    prefix = entity_prefix(filtered_path, FILTERED_RAW_SUFFIX)
    clean_path = filtered_path.with_name(prefix + CLEAN_RAW_SUFFIX)
    if not clean_path.is_file():
        raise FileNotFoundError(f"Missing ICA-cleaned raw file: {clean_path}")
    return clean_path


__all__ = [
    "CLEAN_RAW_SUFFIX",
    "FILTERED_RAW_SUFFIX",
    "ICA_SUFFIX",
    "clean_raw_for_filtered",
    "entity_prefix",
    "find_filtered_raw_runs",
    "find_ica_solutions",
    "resolve_subjects",
    "runs_for_prefix",
    "subject_root",
]
