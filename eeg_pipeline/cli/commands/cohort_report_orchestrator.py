"""Execution orchestrator for the cohort preprocessing QC report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from eeg_pipeline.preprocessing.report.cohort.aggregate import BandGates
from eeg_pipeline.preprocessing.report.cohort.collect import collect_cohort
from eeg_pipeline.preprocessing.report.cohort.report import build_cohort_report

#: Where subject reports live under a derivatives root, and where the cohort output goes.
EEG_SUBDIR = ("preprocessed", "eeg")
GROUP_DIRNAME = "group"


def run_cohort_report(args: argparse.Namespace, subjects: list[str], config: Any) -> None:
    """Aggregate the per-subject QC sidecars into one cohort document."""
    deriv_eeg_root = _deriv_eeg_root(args, config)
    selected = _selected_subjects(args, subjects)

    cohort = collect_cohort(
        deriv_eeg_root,
        task=getattr(args, "task", None),
        subjects=selected,
    )
    gates = _resolved_gates(args, config)
    output_dir = (
        Path(args.output_dir)
        if getattr(args, "output_dir", None)
        else deriv_eeg_root / GROUP_DIRNAME
    )

    paths = build_cohort_report(
        cohort,
        output_dir=output_dir,
        task=_resolved_task(args, cohort),
        gates=gates,
        title=getattr(args, "title", None),
    )

    print(
        f"Aggregated {cohort.n_participants} participant(s) "
        f"over {sum(p.n_runs for p in cohort.participants)} run(s)."
    )
    for entry in cohort.not_aggregated:
        print(f"  not aggregated: sub-{entry.subject} — {entry.reason}")
    print(f"Wrote cohort report: {paths.html}")
    print(f"Wrote cohort log: {paths.log}")
    for path in paths.audit:
        print(f"Wrote audit table: {path}")


def _resolved_gates(args: argparse.Namespace, config: Any) -> BandGates:
    """The band gates this run draws under: the config's, unless a flag overrode one.

    The config is the source, so the gates a cohort was drawn under travel with the study
    rather than with whoever typed the command, and the report's own log records them.
    The flags remain for a one-off run against a different threshold and are ``None``
    unless given, which is what distinguishes "not passed" from "passed the default".
    """
    from eeg_pipeline.preprocessing.report.settings import ReportSettings

    settings = ReportSettings.from_config(config)
    median = getattr(args, "min_subjects_for_median", None)
    outer = getattr(args, "min_subjects_for_outer_band", None)
    return BandGates(
        min_subjects_for_median=(
            settings.min_subjects_for_median if median is None else int(median)
        ),
        min_subjects_for_outer_band=(
            settings.min_subjects_for_outer_band if outer is None else int(outer)
        ),
    )


def _resolved_task(args: argparse.Namespace, cohort) -> str | None:
    """The task the outputs are named after.

    An explicit ``--task`` wins. Otherwise the cohort is named after its task only when it
    turned out to have exactly one: naming a two-task document after one of them would
    produce a filename that misdescribes its contents, and the report states the tasks it
    covers either way.
    """
    requested = getattr(args, "task", None)
    if requested:
        return str(requested)
    return cohort.tasks[0] if len(cohort.tasks) == 1 else None


def _selected_subjects(
    args: argparse.Namespace, subjects: Sequence[str]
) -> list[str] | None:
    """The participants to aggregate, or ``None`` for every one that has a sidecar.

    ``--subjects`` takes precedence over the global subject selection, because this command
    declares ``requires_subjects=False`` and the default really is the whole cohort: a
    subject list left over from another command should not silently narrow it.
    """
    requested = getattr(args, "subjects", None)
    if requested:
        return [str(subject).removeprefix("sub-") for subject in requested]
    if subjects:
        return [str(subject).removeprefix("sub-") for subject in subjects]
    return None


def _deriv_eeg_root(args: argparse.Namespace, config: Any) -> Path:
    """Where the subject reports live.

    An explicit ``--deriv-root`` may name either the derivatives root or the EEG directory
    inside it, because both are things a user reasonably has in hand, and guessing wrong
    produces a "no participant could be aggregated" error that names the wrong cause.
    """
    override = getattr(args, "deriv_root", None)
    root = Path(override) if override else _configured_deriv_root(config)
    candidate = root.joinpath(*EEG_SUBDIR)
    if candidate.is_dir():
        return candidate
    if root.is_dir():
        return root
    raise NotADirectoryError(
        f"No derivatives directory at {candidate} or {root}. Pass --deriv-root, or run "
        f"the preprocessing report stage first so there are subject reports to aggregate."
    )


def _configured_deriv_root(config: Any) -> Path:
    from eeg_pipeline.infra.paths import resolve_deriv_root

    return Path(resolve_deriv_root(config=config))
